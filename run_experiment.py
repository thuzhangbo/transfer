"""
实验运行主脚本

完整流程: 源域训练 → PCKD适应 → 评估
支持所有12组实验任务(T1-T12)
"""

import os
import json
import argparse
import logging
from datetime import datetime

import torch

from train_source import train_source_model
from models.pckd import PCKD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_single_task(task_dir, encoder_name="gin", hidden_dim=128, num_layers=3,
                    source_epochs=200, adapt_epochs=200, batch_size=32,
                    seed=42, device="cuda"):
    """运行单个任务的完整实验流程"""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    with open(os.path.join(task_dir, "config.json"), "r") as f:
        task_config = json.load(f)

    task_name = task_config["task_name"]
    known_classes = task_config["known_classes"]
    num_known_classes = len(known_classes)

    logger.info(f"\n{'#'*60}")
    logger.info(f"运行任务: {task_name} - {task_config['name']}")
    logger.info(f"已知类: {known_classes}, 未知类: {task_config['unknown_classes']}")
    logger.info(f"GNN: {encoder_name}, hidden={hidden_dim}, layers={num_layers}")
    logger.info(f"{'#'*60}")

    ckpt_dir = os.path.join(task_dir, "checkpoints", f"{encoder_name}_seed{seed}")

    # ====== Phase 1: 源域训练 ======
    logger.info("\n===== Phase 1: 源域训练 =====")
    source_model_path = os.path.join(ckpt_dir, "source_model.pt")

    if os.path.exists(source_model_path):
        logger.info(f"发现已有源域模型: {source_model_path}, 跳过训练")
    else:
        train_source_model(
            task_dir, ckpt_dir,
            encoder_name=encoder_name, hidden_dim=hidden_dim,
            num_layers=num_layers, epochs=source_epochs,
            batch_size=batch_size, device=device,
        )

    # ====== Phase 2: PCKD适应 ======
    logger.info("\n===== Phase 2: PCKD适应 =====")

    target_adapt = torch.load(
        os.path.join(task_dir, "target_adapt.pt"), weights_only=False
    )
    target_test = torch.load(
        os.path.join(task_dir, "target_test.pt"), weights_only=False
    )

    # 重映射目的域标签（已知类别映射为连续整数，未知类别保留原标签用于评估）
    ckpt = torch.load(source_model_path, map_location="cpu", weights_only=False)
    label_map = ckpt["label_map"]

    for g in target_test:
        old_label = g.y.item()
        if old_label in label_map:
            g.y = torch.tensor([label_map[old_label]], dtype=torch.long)
        else:
            g.y = torch.tensor([old_label + 100], dtype=torch.long)  # 未知类标记

    pckd = PCKD(
        teacher_checkpoint=source_model_path,
        num_known_classes=num_known_classes,
        device=device,
        epochs=adapt_epochs,
    )

    adapted_model = pckd.adapt(
        target_adapt, target_test,
        batch_size=batch_size,
    )

    # ====== Phase 3: 最终评估 ======
    logger.info("\n===== Phase 3: 最终评估 =====")
    final_metrics = pckd.evaluate(target_test, batch_size)

    logger.info(f"\n{'='*60}")
    logger.info(f"最终结果 [{task_name}] ({encoder_name}, seed={seed})")
    logger.info(f"  H-Score:     {final_metrics['h_score']:.4f}")
    logger.info(f"  Acc_known:   {final_metrics['acc_known']:.4f}")
    logger.info(f"  Acc_unknown: {final_metrics['acc_unknown']:.4f}")
    logger.info(f"  已知样本数:  {final_metrics['known_total']}")
    logger.info(f"  未知样本数:  {final_metrics['unknown_total']}")
    logger.info(f"{'='*60}")

    # 保存结果
    result = {
        "task": task_name,
        "encoder": encoder_name,
        "seed": seed,
        "h_score": final_metrics["h_score"],
        "acc_known": final_metrics["acc_known"],
        "acc_unknown": final_metrics["acc_unknown"],
        "known_total": final_metrics["known_total"],
        "unknown_total": final_metrics["unknown_total"],
        "timestamp": datetime.now().isoformat(),
    }

    result_path = os.path.join(ckpt_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    pckd.save(os.path.join(ckpt_dir, "adapted_model.pt"))

    return result


def run_experiment(exp_dir, tasks, encoders, seeds, hidden_dim=128, num_layers=3,
                   source_epochs=200, adapt_epochs=200, batch_size=32, device="cuda"):
    """运行完整实验矩阵: 任务 × 编码器 × 种子"""

    all_results = []

    for task_name in tasks:
        task_dir = os.path.join(exp_dir, task_name)
        if not os.path.exists(task_dir):
            logger.warning(f"任务目录不存在: {task_dir}, 跳过")
            continue

        for encoder in encoders:
            for seed in seeds:
                try:
                    result = run_single_task(
                        task_dir, encoder_name=encoder,
                        hidden_dim=hidden_dim, num_layers=num_layers,
                        source_epochs=source_epochs, adapt_epochs=adapt_epochs,
                        batch_size=batch_size, seed=seed, device=device,
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"任务 {task_name}/{encoder}/seed{seed} 失败: {e}")
                    import traceback
                    traceback.print_exc()

    # 汇总结果
    if all_results:
        logger.info(f"\n{'#'*60}")
        logger.info("实验结果汇总")
        logger.info(f"{'#'*60}")
        logger.info(f"{'任务':<6} {'编码器':<8} {'H-Score':>10} {'Acc_k':>10} {'Acc_u':>10}")
        logger.info("-" * 50)
        for r in all_results:
            logger.info(
                f"{r['task']:<6} {r['encoder']:<8} "
                f"{r['h_score']:>10.4f} {r['acc_known']:>10.4f} {r['acc_unknown']:>10.4f}"
            )

        summary_path = os.path.join(exp_dir, "all_results.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n结果保存到: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="PCKD实验运行主脚本")
    parser.add_argument("--exp_dir", default="./experiments", help="实验目录")
    parser.add_argument("--tasks", nargs="+", default=["T1"],
                        help="要运行的任务 (T1-T12)")
    parser.add_argument("--encoders", nargs="+", default=["gin"],
                        choices=["gin", "sage", "gat", "gcn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="随机种子列表 (默认3次重复: 42 123 456)")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--source_epochs", type=int, default=200)
    parser.add_argument("--adapt_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_experiment(
        args.exp_dir, args.tasks, args.encoders, args.seeds,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        source_epochs=args.source_epochs, adapt_epochs=args.adapt_epochs,
        batch_size=args.batch_size, device=args.device,
    )


if __name__ == "__main__":
    main()
