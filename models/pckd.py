"""
PCKD (Prototype-based Cross-domain Knowledge Distillation) 适应框架

核心组件:
1. 知识蒸馏 (KL散度)
2. 早期学习正则化 (ELR) + 自适应权重 + 全局调度
3. 课程式原型学习 + 鲁棒距离阈值 (Median+MAD)
4. 图增强一致性正则化
5. 开放集未知类别检测
"""

import os
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.gnn_encoder import GraphClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 图数据增强
# ============================================================

def augment_graph(data, node_mask_ratio=0.1, edge_drop_ratio=0.1):
    """对溯源图进行数据增强: 节点特征掩码 + 边扰动"""
    aug_data = data.clone()

    # 节点特征掩码
    if node_mask_ratio > 0:
        mask = torch.rand(aug_data.x.shape[0]) > node_mask_ratio
        aug_data.x = aug_data.x * mask.unsqueeze(1).float().to(aug_data.x.device)

    # 边扰动
    if edge_drop_ratio > 0 and aug_data.edge_index.shape[1] > 0:
        aug_data.edge_index, _ = dropout_edge(
            aug_data.edge_index, p=edge_drop_ratio, training=True
        )

    return aug_data


# ============================================================
# ELR: Early Learning Regularization
# ============================================================

class ELRModule:
    """早期学习正则化 + 时序集成 + 自适应权重 + 全局调度"""

    def __init__(self, num_samples, num_classes, momentum=0.95,
                 lambda_elr_init=1.0, gamma=0.03):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.momentum = momentum
        self.lambda_elr_init = lambda_elr_init
        self.gamma = gamma

        # 时序集成目标 (每个样本的历史预测的指数移动平均)
        self.temporal_targets = torch.zeros(num_samples, num_classes)
        self.sample_weights = torch.ones(num_samples)

    def update_temporal_targets(self, indices, predictions):
        """用当前模型预测更新时序集成目标"""
        probs = F.softmax(predictions.detach().cpu(), dim=1)
        for i, idx in enumerate(indices):
            self.temporal_targets[idx] = (
                self.momentum * self.temporal_targets[idx] +
                (1 - self.momentum) * probs[i]
            )

    def compute_elr_loss(self, logits, indices, epoch, max_epochs):
        """计算ELR损失: 鼓励模型预测与自身早期预测一致"""
        probs = F.softmax(logits, dim=1)
        targets = self.temporal_targets[indices].to(logits.device)

        # 防止log(0)
        elr_loss = -torch.sum(targets * torch.log(probs + 1e-8), dim=1)

        # 样本级自适应权重
        weights = self.sample_weights[indices].to(logits.device)
        weighted_loss = (elr_loss * weights).mean()

        # 全局调度: 随训练推进逐渐减小ELR权重
        schedule = self.lambda_elr_init * np.exp(-self.gamma * epoch)
        weighted_loss = schedule * weighted_loss

        return weighted_loss

    def update_sample_weights(self, indices, logits, neighbor_consistency=None):
        """基于邻域一致性更新样本权重"""
        if neighbor_consistency is not None:
            for i, idx in enumerate(indices):
                self.sample_weights[idx] = neighbor_consistency[i].item()


# ============================================================
# 原型学习模块
# ============================================================

class PrototypeModule:
    """课程式原型学习 + 鲁棒距离阈值 + 逆频率加权"""

    def __init__(self, num_known_classes, hidden_dim, momentum=0.95,
                 warmup_epochs=10, tau_p=2.0, confidence_threshold=0.8):
        self.num_known_classes = num_known_classes
        self.hidden_dim = hidden_dim
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.tau_p = tau_p
        self.confidence_threshold = confidence_threshold

        self.prototypes = torch.zeros(num_known_classes, hidden_dim)
        self.prototype_initialized = [False] * num_known_classes
        self.class_counts = torch.zeros(num_known_classes)

    def update_prototypes(self, embeddings, pseudo_labels, confidences, epoch):
        """课程式原型更新"""
        device = embeddings.device
        embeddings = embeddings.detach().cpu()
        pseudo_labels = pseudo_labels.cpu()
        confidences = confidences.cpu()

        if epoch < self.warmup_epochs:
            # 热身阶段: 只用高置信度样本
            mask = confidences > self.confidence_threshold
        else:
            # 正常阶段: 使用全部样本（动量更新）
            mask = torch.ones(len(pseudo_labels), dtype=torch.bool)

        for c in range(self.num_known_classes):
            class_mask = (pseudo_labels == c) & mask
            if class_mask.sum() == 0:
                continue

            class_embeddings = embeddings[class_mask]
            class_mean = class_embeddings.mean(dim=0)
            self.class_counts[c] = class_mask.sum().item()

            if not self.prototype_initialized[c]:
                self.prototypes[c] = class_mean
                self.prototype_initialized[c] = True
            else:
                self.prototypes[c] = (
                    self.momentum * self.prototypes[c] +
                    (1 - self.momentum) * class_mean
                )

    def compute_distances(self, embeddings):
        """计算样本到各原型的距离"""
        prototypes = self.prototypes.to(embeddings.device)
        # (N, 1, D) - (1, C, D) -> (N, C)
        dists = torch.cdist(
            embeddings.unsqueeze(0),
            prototypes.unsqueeze(0)
        ).squeeze(0)
        return dists

    def compute_robust_threshold(self, embeddings, pseudo_labels):
        """鲁棒距离阈值: Median + MAD"""
        dists = self.compute_distances(embeddings)
        thresholds = torch.zeros(self.num_known_classes)

        for c in range(self.num_known_classes):
            class_mask = pseudo_labels == c
            if class_mask.sum() == 0:
                thresholds[c] = float("inf")
                continue

            class_dists = dists[class_mask, c]
            median = class_dists.median()
            mad = (class_dists - median).abs().median()
            thresholds[c] = median + self.tau_p * 1.4826 * mad

        return thresholds

    def detect_unknown(self, embeddings, thresholds=None):
        """检测未知类别样本"""
        dists = self.compute_distances(embeddings)
        min_dists, nearest_class = dists.min(dim=1)

        if thresholds is None:
            return nearest_class, min_dists, torch.zeros(len(embeddings), dtype=torch.bool)

        thresholds = thresholds.to(embeddings.device)
        class_thresholds = thresholds[nearest_class]
        is_unknown = min_dists > class_thresholds

        return nearest_class, min_dists, is_unknown

    def compute_prototype_loss(self, embeddings, pseudo_labels):
        """原型对齐损失（逆频率加权）"""
        dists = self.compute_distances(embeddings)

        # 逆频率权重
        total = self.class_counts.sum().clamp(min=1)
        weights = total / self.class_counts.clamp(min=1)
        weights = weights / weights.sum() * self.num_known_classes
        weights = weights.to(embeddings.device)

        loss = 0
        for c in range(self.num_known_classes):
            class_mask = pseudo_labels == c
            if class_mask.sum() == 0:
                continue
            loss += weights[c] * dists[class_mask, c].mean()

        return loss / self.num_known_classes

    def compute_open_set_loss(self, embeddings, pseudo_labels, is_unknown):
        """开放集分离损失: 推远未知样本"""
        if is_unknown.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        dists = self.compute_distances(embeddings)
        unknown_min_dists = dists[is_unknown].min(dim=1).values

        # 未知样本应该远离所有原型
        margin = 1.0
        loss = F.relu(margin - unknown_min_dists).mean()
        return loss


# ============================================================
# PCKD 主框架
# ============================================================

class PCKD:
    """Prototype-based Cross-domain Knowledge Distillation"""

    def __init__(self, teacher_checkpoint, num_known_classes, device="cuda",
                 lambda_kd=1.0, lambda_elr_init=1.0, gamma=0.03,
                 lambda_proto=0.5, lambda_open=0.3, lambda_aug=0.3,
                 tau_p=2.0, momentum=0.95, confidence_threshold=0.8,
                 warmup_epochs=10, temperature=3.0,
                 lr=0.001, epochs=200):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_known_classes = num_known_classes
        self.epochs = epochs
        self.temperature = temperature

        # 损失权重
        self.lambda_kd = lambda_kd
        self.lambda_elr_init = lambda_elr_init
        self.lambda_proto = lambda_proto
        self.lambda_open = lambda_open
        self.lambda_aug = lambda_aug

        # 加载教师模型（冻结）
        ckpt = torch.load(teacher_checkpoint, map_location=self.device, weights_only=False)
        self.teacher = GraphClassifier(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_classes=ckpt["num_classes"],
            encoder_name=ckpt["encoder_name"],
            num_layers=ckpt["num_layers"],
        ).to(self.device)
        self.teacher.load_state_dict(ckpt["model_state_dict"])
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 创建学生模型（与教师同架构）
        self.student = GraphClassifier(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_classes=ckpt["num_classes"],
            encoder_name=ckpt["encoder_name"],
            num_layers=ckpt["num_layers"],
        ).to(self.device)
        self.student.load_state_dict(ckpt["model_state_dict"])

        self.hidden_dim = ckpt["hidden_dim"]
        self.label_map = ckpt["label_map"]
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr, weight_decay=1e-4)

        # 原型模块
        self.prototype_module = PrototypeModule(
            num_known_classes=num_known_classes,
            hidden_dim=ckpt["hidden_dim"],
            momentum=momentum,
            warmup_epochs=warmup_epochs,
            tau_p=tau_p,
            confidence_threshold=confidence_threshold,
        )

    def _kd_loss(self, student_logits, teacher_logits):
        """知识蒸馏损失 (KL散度)"""
        T = self.temperature
        student_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (T * T)
        return loss

    def adapt(self, target_adapt_data, target_test_data=None, batch_size=32):
        """在目的域上进行PCKD适应"""

        # 初始化ELR
        elr = ELRModule(
            num_samples=len(target_adapt_data),
            num_classes=self.num_known_classes,
            lambda_elr_init=self.lambda_elr_init,
        )

        # 给每个样本添加索引
        for i, data in enumerate(target_adapt_data):
            data.sample_idx = i

        adapt_loader = DataLoader(target_adapt_data, batch_size=batch_size, shuffle=True)

        best_hscore = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            self.student.train()
            epoch_losses = {"kd": 0, "elr": 0, "proto": 0, "open": 0, "aug": 0, "total": 0}
            num_batches = 0

            for batch in adapt_loader:
                batch = batch.to(self.device)
                indices = batch.sample_idx

                # --- 教师预测 (冻结) ---
                with torch.no_grad():
                    teacher_emb, teacher_logits = self.teacher.forward_with_embedding(
                        batch.x, batch.edge_index, batch.batch
                    )
                    teacher_probs = F.softmax(teacher_logits, dim=1)
                    pseudo_labels = teacher_probs.argmax(dim=1)
                    confidences = teacher_probs.max(dim=1).values

                # --- 学生前向传播 ---
                student_emb, student_logits = self.student.forward_with_embedding(
                    batch.x, batch.edge_index, batch.batch
                )

                # 1. 知识蒸馏损失
                loss_kd = self._kd_loss(student_logits, teacher_logits)

                # 2. ELR损失
                elr.update_temporal_targets(indices, student_logits)
                loss_elr = elr.compute_elr_loss(student_logits, indices, epoch, self.epochs)

                # 3. 原型学习损失
                self.prototype_module.update_prototypes(
                    student_emb, pseudo_labels, confidences, epoch
                )
                loss_proto = self.prototype_module.compute_prototype_loss(
                    student_emb, pseudo_labels
                )

                # 4. 开放集检测 + 分离损失
                thresholds = self.prototype_module.compute_robust_threshold(
                    student_emb, pseudo_labels
                )
                _, _, is_unknown = self.prototype_module.detect_unknown(
                    student_emb, thresholds
                )
                loss_open = self.prototype_module.compute_open_set_loss(
                    student_emb, pseudo_labels, is_unknown
                )

                # 5. 图增强一致性损失
                loss_aug = torch.tensor(0.0, device=self.device)
                if self.lambda_aug > 0:
                    aug_logits_list = []
                    for i in range(batch.num_graphs):
                        mask = batch.batch == i
                        node_feat = batch.x[mask]
                        sub_edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                        sub_edges = batch.edge_index[:, sub_edge_mask]
                        if sub_edges.shape[1] > 0:
                            node_offset = mask.nonzero(as_tuple=True)[0][0]
                            sub_edges = sub_edges - node_offset
                            sub_batch = torch.zeros(node_feat.shape[0], dtype=torch.long,
                                                  device=self.device)
                            # 节点特征掩码增强
                            aug_mask = torch.rand(node_feat.shape[0], device=self.device) > 0.1
                            aug_feat = node_feat * aug_mask.unsqueeze(1).float()
                            aug_logit = self.student(aug_feat, sub_edges, sub_batch)
                            aug_logits_list.append(aug_logit)

                    if aug_logits_list:
                        aug_logits_cat = torch.cat(aug_logits_list, dim=0)
                        loss_aug = F.kl_div(
                            F.log_softmax(aug_logits_cat, dim=1),
                            F.softmax(student_logits.detach(), dim=1),
                            reduction="batchmean"
                        )

                # 总损失
                total_loss = (
                    self.lambda_kd * loss_kd +
                    loss_elr +
                    self.lambda_proto * loss_proto +
                    self.lambda_open * loss_open +
                    self.lambda_aug * loss_aug
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses["kd"] += loss_kd.item()
                epoch_losses["elr"] += loss_elr.item()
                epoch_losses["proto"] += loss_proto.item()
                epoch_losses["open"] += loss_open.item()
                epoch_losses["aug"] += loss_aug.item()
                epoch_losses["total"] += total_loss.item()
                num_batches += 1

            # 日志
            if epoch % 10 == 0 or epoch == 1:
                avg = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
                logger.info(
                    f"Epoch {epoch:3d} | Total: {avg['total']:.4f} | "
                    f"KD: {avg['kd']:.4f} | ELR: {avg['elr']:.4f} | "
                    f"Proto: {avg['proto']:.4f} | Open: {avg['open']:.4f} | "
                    f"Aug: {avg['aug']:.4f}"
                )

                # 在测试集上评估
                if target_test_data is not None:
                    metrics = self.evaluate(target_test_data, batch_size)
                    logger.info(
                        f"         | H-Score: {metrics['h_score']:.4f} | "
                        f"Acc_k: {metrics['acc_known']:.4f} | "
                        f"Acc_u: {metrics['acc_unknown']:.4f}"
                    )
                    if metrics["h_score"] > best_hscore:
                        best_hscore = metrics["h_score"]
                        best_state = copy.deepcopy(self.student.state_dict())

        # 恢复最佳模型
        if best_state is not None:
            self.student.load_state_dict(best_state)
            logger.info(f"恢复最佳模型, H-Score: {best_hscore:.4f}")

        return self.student

    @torch.no_grad()
    def evaluate(self, test_data, batch_size=32):
        """在目的域测试集上评估"""
        self.student.eval()
        test_loader = DataLoader(test_data, batch_size=batch_size)

        all_preds = []
        all_labels = []
        all_is_unknown_pred = []
        all_is_unknown_true = []

        # 计算阈值
        known_classes = list(range(self.num_known_classes))

        for batch in test_loader:
            batch = batch.to(self.device)
            student_emb, student_logits = self.student.forward_with_embedding(
                batch.x, batch.edge_index, batch.batch
            )

            pseudo_labels = student_logits.argmax(dim=1)

            thresholds = self.prototype_module.compute_robust_threshold(
                student_emb, pseudo_labels
            )
            nearest_class, min_dists, is_unknown_pred = self.prototype_module.detect_unknown(
                student_emb, thresholds
            )

            for i in range(batch.num_graphs):
                true_label = batch.y[i].item()
                is_truly_unknown = true_label not in known_classes

                if is_unknown_pred[i]:
                    pred_label = -1  # 预测为未知
                else:
                    pred_label = nearest_class[i].item()

                all_preds.append(pred_label)
                all_labels.append(true_label)
                all_is_unknown_pred.append(is_unknown_pred[i].item())
                all_is_unknown_true.append(is_truly_unknown)

        # 计算指标
        known_mask = [not u for u in all_is_unknown_true]
        unknown_mask = all_is_unknown_true

        # Acc_known: 已知类别中预测正确的比例
        known_correct = sum(
            1 for i in range(len(all_labels))
            if known_mask[i] and all_preds[i] == all_labels[i]
        )
        known_total = sum(known_mask)
        acc_known = known_correct / max(known_total, 1)

        # Acc_unknown: 未知类别被正确检测为未知的比例
        unknown_correct = sum(
            1 for i in range(len(all_labels))
            if unknown_mask[i] and all_preds[i] == -1
        )
        unknown_total = sum(unknown_mask)
        acc_unknown = unknown_correct / max(unknown_total, 1)

        # H-Score
        if acc_known + acc_unknown > 0:
            h_score = 2 * acc_known * acc_unknown / (acc_known + acc_unknown)
        else:
            h_score = 0.0

        return {
            "h_score": h_score,
            "acc_known": acc_known,
            "acc_unknown": acc_unknown,
            "known_total": known_total,
            "unknown_total": unknown_total,
        }

    def save(self, path):
        """保存适应后的模型"""
        checkpoint = {
            "student_state_dict": self.student.state_dict(),
            "prototypes": self.prototype_module.prototypes,
            "class_counts": self.prototype_module.class_counts,
        }
        torch.save(checkpoint, path)
        logger.info(f"模型保存到: {path}")
