"""
数据集组织脚本：将处理好的溯源图按实验任务(T1-T12)划分为源域/目的域

使用方法:
    python organize_dataset.py --data_dir ./processed_graphs --output_dir ./experiments

依赖: 先用 build_provenance_graph.py batch 模式处理完所有场景
"""

import os
import json
import random
import argparse
import logging
from collections import defaultdict, Counter

import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 攻击类型标签定义 (与 build_provenance_graph.py 一致)
# ============================================================

LABEL_NAMES = {
    0: "Normal",
    1: "RCE",
    2: "Non-RCE",
}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAMES.items()}

# ============================================================
# 场景 → 应用 / 运行时 映射
# ============================================================

SCENARIO_META = {
    # D_solr (Java, 4 CVE)
    "cve-2019-17558":   {"app": "Solr",      "runtime": "java"},
    "cve-2019-0193":    {"app": "Solr",      "runtime": "java"},
    "cve-2017-12629-rce": {"app": "Solr",    "runtime": "java"},
    "cve-2017-12629-xxe": {"app": "Solr",    "runtime": "java"},
    # D_ofbiz (Java, 8 CVE)
    "cve-2020-9496":    {"app": "OFBiz",     "runtime": "java"},
    "cve-2023-49070":   {"app": "OFBiz",     "runtime": "java"},
    "cve-2023-51467":   {"app": "OFBiz",     "runtime": "java"},
    "cve-2024-38856":   {"app": "OFBiz",     "runtime": "java"},
    "cve-2024-45507":   {"app": "OFBiz",     "runtime": "java"},
    "cve-2024-45195":   {"app": "OFBiz",     "runtime": "java"},
    "cve-2023-50968":   {"app": "OFBiz",     "runtime": "java"},
    "cve-2024-23946":   {"app": "OFBiz",     "runtime": "java"},
    # D_geo (Java, 3 CVE)
    "cve-2024-36401":   {"app": "GeoServer", "runtime": "java"},
    "cve-2023-25157":   {"app": "GeoServer", "runtime": "java"},
    "cve-2021-40822":   {"app": "GeoServer", "runtime": "java"},
    # D_tomcat (Java, 3 CVE)
    "cve-2025-24813":   {"app": "Tomcat",    "runtime": "java"},
    "cve-2020-9484":    {"app": "Tomcat",    "runtime": "java"},
    "cve-2020-1938":    {"app": "Tomcat",    "runtime": "java"},
    # D_metabase (Java, 2 CVE)
    "cve-2023-38646":   {"app": "Metabase",  "runtime": "java"},
    "cve-2021-41277":   {"app": "Metabase",  "runtime": "java"},
    # D_joomla (PHP, 3 CVE)
    "cve-2015-8562":    {"app": "Joomla",    "runtime": "php"},
    "cve-2017-8917":    {"app": "Joomla",    "runtime": "php"},
    "cve-2023-23752":   {"app": "Joomla",    "runtime": "php"},
    # D_pgadmin (Python, 3 CVE)
    "cve-2022-4223":    {"app": "pgAdmin",   "runtime": "python"},
    "cve-2023-5002":    {"app": "pgAdmin",   "runtime": "python"},
    "cve-2024-9014":    {"app": "pgAdmin",   "runtime": "python"},
    # 补充场景
    "cve-2021-26120":   {"app": "CMSMS",        "runtime": "php"},
    "cve-2019-9053":    {"app": "CMSMS",        "runtime": "php"},
    "cve-2021-43008":   {"app": "Adminer",      "runtime": "php"},
    "cve-2021-21311":   {"app": "Adminer",      "runtime": "php"},
    "cve-2018-1000533": {"app": "Gitlist",      "runtime": "php"},
    "cve-2021-34429":   {"app": "Jetty",        "runtime": "java"},
    "cve-2019-14234":   {"app": "Django",       "runtime": "python"},
    "cve-2022-34265":   {"app": "Django",       "runtime": "python"},
    "python-demo":      {"app": "PythonDemo",   "runtime": "python"},
    "owasp-juice-shop": {"app": "JuiceShop",    "runtime": "nodejs"},
    "cve-2019-10758":   {"app": "MongoExpress", "runtime": "nodejs"},
    "cve-2018-17246":   {"app": "Kibana",       "runtime": "nodejs"},
    "sandworm":         {"app": "Sandworm",     "runtime": "linux"},
    "10-step-chained":  {"app": "Chained",      "runtime": "mixed"},
}


# ============================================================
# 7 个主实验应用域
# ============================================================

MAIN_DOMAINS = {
    "D_solr":     {"app": "Solr",      "runtime": "java"},
    "D_ofbiz":    {"app": "OFBiz",     "runtime": "java"},
    "D_geo":      {"app": "GeoServer", "runtime": "java"},
    "D_tomcat":   {"app": "Tomcat",    "runtime": "java"},
    "D_metabase": {"app": "Metabase",  "runtime": "java"},
    "D_joomla":   {"app": "Joomla",    "runtime": "php"},
    "D_pgadmin":  {"app": "pgAdmin",   "runtime": "python"},
}


# ============================================================
# 12 组实验任务定义
#
# 标签体系: Normal=0, RCE=1, Non-RCE=2
# 默认开放集设定: 源域训练 Normal vs RCE (二分类)
#   已知类 = [0, 1] (Normal, RCE)
#   未知类 = [2]     (Non-RCE)
# ============================================================

TASKS = {
    # === A. 跨运行时域偏移 (T1-T6) ===
    "T1": {
        "name": "D_solr(Java) → D_joomla(PHP)",
        "shift_type": "cross_runtime",
        "source_filter": {"app": "Solr"},
        "target_filter": {"app": "Joomla"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "跨运行时基准任务 Java→PHP",
    },
    "T2": {
        "name": "D_ofbiz(Java) → D_joomla(PHP)",
        "shift_type": "cross_runtime",
        "source_filter": {"app": "OFBiz"},
        "target_filter": {"app": "Joomla"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "换源域(OFBiz)，验证源域质量的影响",
    },
    "T3": {
        "name": "D_solr(Java) → D_pgadmin(Python)",
        "shift_type": "cross_runtime",
        "source_filter": {"app": "Solr"},
        "target_filter": {"app": "pgAdmin"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "跨运行时 Java→Python",
    },
    "T4": {
        "name": "D_ofbiz(Java) → D_pgadmin(Python)",
        "shift_type": "cross_runtime",
        "source_filter": {"app": "OFBiz"},
        "target_filter": {"app": "pgAdmin"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "换源域(OFBiz)，验证源域质量的影响",
    },
    "T5": {
        "name": "D_joomla(PHP) → D_solr(Java)",
        "shift_type": "cross_runtime",
        "source_filter": {"app": "Joomla"},
        "target_filter": {"app": "Solr"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "反向迁移 PHP→Java，验证方向对称性",
    },
    "T6": {
        "name": "D_pgadmin(Python) → D_solr(Java)",
        "shift_type": "cross_runtime",
        "source_filter": {"app": "pgAdmin"},
        "target_filter": {"app": "Solr"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "反向迁移 Python→Java，验证方向对称性",
    },

    # === B. 同运行时跨应用域偏移 (T7-T9) ===
    "T7": {
        "name": "D_solr(Java) → D_ofbiz(Java)",
        "shift_type": "cross_app",
        "source_filter": {"app": "Solr"},
        "target_filter": {"app": "OFBiz"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "同运行时跨应用: 搜索引擎→ERP系统",
    },
    "T8": {
        "name": "D_ofbiz(Java) → D_geo(Java)",
        "shift_type": "cross_app",
        "source_filter": {"app": "OFBiz"},
        "target_filter": {"app": "GeoServer"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "同运行时跨应用: ERP系统→GIS服务器",
    },
    "T9": {
        "name": "D_solr(Java) → D_tomcat(Java)",
        "shift_type": "cross_app",
        "source_filter": {"app": "Solr"},
        "target_filter": {"app": "Tomcat"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "同运行时跨应用: 搜索引擎→中间件服务器",
    },

    # === C. 闭集对照 (T10) ===
    "T10": {
        "name": "D_solr(Java) → D_joomla(PHP) [闭集]",
        "shift_type": "cross_runtime",
        "source_filter": {"app": "Solr"},
        "target_filter": {"app": "Joomla"},
        "known_classes": [0, 1, 2],
        "unknown_classes": [],
        "description": "T1的闭集版本(U=0)，量化开放集的额外挑战",
    },

    # === D. 跨操作系统域偏移 (T11-T12, DARPA TC) ===
    "T11": {
        "name": "CADETS(FreeBSD) → THEIA(Linux)",
        "shift_type": "cross_os",
        "source_filter": {"dataset": "darpa_cadets"},
        "target_filter": {"dataset": "darpa_theia"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "跨OS极端域偏移（需DARPA TC数据）",
    },
    "T12": {
        "name": "THEIA(Linux) → CADETS(FreeBSD)",
        "shift_type": "cross_os",
        "source_filter": {"dataset": "darpa_theia"},
        "target_filter": {"dataset": "darpa_cadets"},
        "known_classes": [0, 1],
        "unknown_classes": [2],
        "description": "T11反向（需DARPA TC数据）",
    },
}


# ============================================================
# 数据加载与过滤
# ============================================================

def load_all_graphs(data_dir):
    """加载所有处理好的.pt文件"""
    all_graphs = []
    pt_files = [f for f in os.listdir(data_dir) if f.endswith(".pt")]

    for pt_file in sorted(pt_files):
        filepath = os.path.join(data_dir, pt_file)
        try:
            graphs = torch.load(filepath, weights_only=False)
            if isinstance(graphs, list):
                all_graphs.extend(graphs)
            elif isinstance(graphs, Data):
                all_graphs.append(graphs)
        except Exception as e:
            logger.warning(f"加载 {pt_file} 失败: {e}")

    logger.info(f"共加载 {len(all_graphs)} 张图")
    return all_graphs


def filter_graphs(graphs, filter_spec):
    """根据过滤条件筛选图"""
    result = []
    for g in graphs:
        match = True
        for key, val in filter_spec.items():
            if key == "dataset":
                g_val = getattr(g, "dataset", None)
                if g_val != val:
                    match = False
            elif key == "runtime":
                g_val = getattr(g, "runtime", None)
                if g_val != val:
                    match = False
            elif key == "app":
                g_val = getattr(g, "app", None)
                if isinstance(val, list):
                    if g_val not in val:
                        match = False
                else:
                    if g_val != val:
                        match = False
        if match:
            result.append(g)
    return result


def filter_by_classes(graphs, class_list):
    """筛选标签在指定类别列表中的图"""
    return [g for g in graphs if g.y.item() in class_list]


# ============================================================
# 数据集划分
# ============================================================

def split_train_val_test(graphs, train_ratio=0.6, val_ratio=0.2, seed=42):
    """按比例划分训练/验证/测试集"""
    random.seed(seed)
    indices = list(range(len(graphs)))
    random.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    test = [graphs[i] for i in test_idx]

    return train, val, test


def organize_task(all_graphs, task_config, task_name):
    """为单个实验任务组织源域和目的域数据"""
    logger.info(f"\n{'='*60}")
    logger.info(f"组织任务 {task_name}: {task_config['name']}")
    logger.info(f"  域偏移类型: {task_config['shift_type']}")
    logger.info(f"  已知类: {[LABEL_NAMES[c] for c in task_config['known_classes']]}")
    logger.info(f"  未知类: {[LABEL_NAMES.get(c, f'Unk_{c}') for c in task_config['unknown_classes']]}")
    logger.info(f"{'='*60}")

    known = task_config["known_classes"]
    unknown = task_config["unknown_classes"]
    all_classes = known + unknown

    source_all = filter_graphs(all_graphs, task_config["source_filter"])
    target_all = filter_graphs(all_graphs, task_config["target_filter"])

    logger.info(f"  源域候选图: {len(source_all)}")
    logger.info(f"  目的域候选图: {len(target_all)}")

    if not source_all or not target_all:
        logger.warning(f"  数据不足，跳过任务 {task_name}")
        return None

    # 源域: 只包含已知类别（用于训练教师模型）
    source_graphs = filter_by_classes(source_all, known)

    # 目的域: 包含已知类别 + 未知类别
    target_graphs = filter_by_classes(target_all, all_classes)

    logger.info(f"  源域图数(已知类): {len(source_graphs)}")
    logger.info(f"  目的域图数(已知+未知): {len(target_graphs)}")

    if len(source_graphs) < 10 or len(target_graphs) < 10:
        logger.warning(f"  数据量不足(最少需10张)，跳过任务 {task_name}")
        return None

    # 源域: 划分为训练/验证集 (80%/20%)
    source_train, source_val, _ = split_train_val_test(source_graphs, 0.8, 0.2, seed=42)

    # 目的域: 划分为适应集(60%无标签)和测试集(40%有标签用于评估)
    target_adapt, _, target_test = split_train_val_test(target_graphs, 0.6, 0.0, seed=42)

    def count_labels(graphs):
        counter = Counter(g.y.item() for g in graphs)
        return {LABEL_NAMES.get(k, f"Unknown_{k}"): v for k, v in sorted(counter.items())}

    logger.info(f"  --- 源域 ---")
    logger.info(f"    训练集: {len(source_train)} 张, 标签分布: {count_labels(source_train)}")
    logger.info(f"    验证集: {len(source_val)} 张, 标签分布: {count_labels(source_val)}")
    logger.info(f"  --- 目的域 ---")
    logger.info(f"    适应集(无标签): {len(target_adapt)} 张, 标签分布: {count_labels(target_adapt)}")
    logger.info(f"    测试集: {len(target_test)} 张, 标签分布: {count_labels(target_test)}")

    return {
        "task_name": task_name,
        "config": task_config,
        "source_train": source_train,
        "source_val": source_val,
        "target_adapt": target_adapt,
        "target_test": target_test,
    }


# ============================================================
# 保存
# ============================================================

def save_task(task_data, output_dir):
    """保存单个任务的数据"""
    task_name = task_data["task_name"]
    task_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)

    torch.save(task_data["source_train"], os.path.join(task_dir, "source_train.pt"))
    torch.save(task_data["source_val"], os.path.join(task_dir, "source_val.pt"))
    torch.save(task_data["target_adapt"], os.path.join(task_dir, "target_adapt.pt"))
    torch.save(task_data["target_test"], os.path.join(task_dir, "target_test.pt"))

    config_info = {
        "task_name": task_name,
        "name": task_data["config"]["name"],
        "shift_type": task_data["config"]["shift_type"],
        "known_classes": task_data["config"]["known_classes"],
        "unknown_classes": task_data["config"]["unknown_classes"],
        "source_train_size": len(task_data["source_train"]),
        "source_val_size": len(task_data["source_val"]),
        "target_adapt_size": len(task_data["target_adapt"]),
        "target_test_size": len(task_data["target_test"]),
    }

    with open(os.path.join(task_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    logger.info(f"  任务 {task_name} 保存到: {task_dir}")


# ============================================================
# 主流程
# ============================================================

DEFAULT_AUTOLABEL_TASKS = [
    "T1", "T2", "T3", "T4", "T5", "T6",
    "T7", "T8", "T9", "T10",
]

def main():
    parser = argparse.ArgumentParser(description="按实验任务组织溯源图数据集")
    parser.add_argument("--data_dir", required=True, help="build_provenance_graph.py输出的目录")
    parser.add_argument("--output_dir", default="./experiments", help="实验数据输出目录")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_AUTOLABEL_TASKS,
                        help="要组织的任务列表（T11/T12需要DARPA TC数据）")
    args = parser.parse_args()

    all_graphs = load_all_graphs(args.data_dir)
    if not all_graphs:
        logger.error("未找到任何图数据，请先运行 build_provenance_graph.py")
        return

    logger.info(f"\n全局数据统计:")
    label_counter = Counter(g.y.item() for g in all_graphs)
    for label, count in sorted(label_counter.items()):
        logger.info(f"  {LABEL_NAMES.get(label, f'Unknown_{label}')}: {count}")

    app_counter = Counter(getattr(g, "app", "unknown") for g in all_graphs)
    logger.info(f"应用分布:")
    for app, count in sorted(app_counter.items()):
        logger.info(f"  {app}: {count}")

    runtime_counter = Counter(getattr(g, "runtime", "unknown") for g in all_graphs)
    logger.info(f"运行时分布:")
    for rt, count in sorted(runtime_counter.items()):
        logger.info(f"  {rt}: {count}")

    os.makedirs(args.output_dir, exist_ok=True)
    success_count = 0

    for task_name in args.tasks:
        if task_name not in TASKS:
            logger.warning(f"未知任务: {task_name}, 跳过")
            continue

        task_data = organize_task(all_graphs, TASKS[task_name], task_name)
        if task_data is not None:
            save_task(task_data, args.output_dir)
            success_count += 1

    logger.info(f"\n完成！成功组织 {success_count}/{len(args.tasks)} 个任务")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"目录结构:")
    for task_name in args.tasks:
        logger.info(f"  {task_name}/")
        logger.info(f"    source_train.pt  (源域训练集)")
        logger.info(f"    source_val.pt    (源域验证集)")
        logger.info(f"    target_adapt.pt  (目的域适应集，无标签)")
        logger.info(f"    target_test.pt   (目的域测试集)")
        logger.info(f"    config.json      (任务配置)")


if __name__ == "__main__":
    main()
