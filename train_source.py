"""
源域训练脚本

在源域有标签数据上训练GNN分类器，生成预训练模型（教师模型）
"""

import os
import argparse
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from models.gnn_encoder import GraphClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        pred = logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        pred = logits.argmax(dim=1)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return acc, f1


def train_source_model(task_dir, output_dir, encoder_name="gin", hidden_dim=128,
                       num_layers=3, lr=0.001, epochs=200, batch_size=32,
                       patience=20, device="cuda"):
    """在源域上训练分类模型"""

    # 加载数据
    source_train = torch.load(os.path.join(task_dir, "source_train.pt"), weights_only=False)
    source_val = torch.load(os.path.join(task_dir, "source_val.pt"), weights_only=False)

    with open(os.path.join(task_dir, "config.json"), "r") as f:
        task_config = json.load(f)

    known_classes = task_config["known_classes"]
    num_classes = len(known_classes)
    input_dim = source_train[0].x.shape[1]

    logger.info(f"任务: {task_config['task_name']} - {task_config['name']}")
    logger.info(f"已知类别数: {num_classes}, 输入特征维度: {input_dim}")
    logger.info(f"源域训练集: {len(source_train)}, 验证集: {len(source_val)}")
    logger.info(f"GNN架构: {encoder_name}, 隐藏维度: {hidden_dim}, 层数: {num_layers}")

    # 重映射标签为连续整数 [0, num_classes)
    label_map = {old: new for new, old in enumerate(known_classes)}
    for g in source_train:
        g.y = torch.tensor([label_map[g.y.item()]], dtype=torch.long)
    for g in source_val:
        g.y = torch.tensor([label_map[g.y.item()]], dtype=torch.long)

    train_loader = DataLoader(source_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(source_val, batch_size=batch_size)

    # 创建模型
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GraphClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        encoder_name=encoder_name,
        num_layers=num_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_val_f1 = 0
    patience_counter = 0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, val_loader, device, num_classes)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "encoder_name": encoder_name,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "num_layers": num_layers,
                "known_classes": known_classes,
                "label_map": label_map,
                "best_val_f1": best_val_f1,
                "epoch": epoch,
            }
            save_path = os.path.join(output_dir, "source_model.pt")
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停于 Epoch {epoch}, 最佳 Val F1: {best_val_f1:.4f}")
                break

    logger.info(f"源域训练完成, 最佳 Val F1: {best_val_f1:.4f}")
    logger.info(f"模型保存到: {os.path.join(output_dir, 'source_model.pt')}")

    return model


def main():
    parser = argparse.ArgumentParser(description="源域GNN模型训练")
    parser.add_argument("--task_dir", required=True, help="任务数据目录 (如 experiments/T1)")
    parser.add_argument("--output_dir", default=None, help="模型输出目录")
    parser.add_argument("--encoder", default="gin", choices=["gin", "sage", "gat", "gcn"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.task_dir, "checkpoints")

    train_source_model(
        args.task_dir, args.output_dir,
        encoder_name=args.encoder, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience, device=args.device,
    )


if __name__ == "__main__":
    main()
