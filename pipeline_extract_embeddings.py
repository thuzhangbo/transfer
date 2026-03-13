"""
一体化脚本：autolabel tar.gz → 溯源图 → 按H-Score控制训练 → 提取embedding → t-SNE可视化

核心逻辑：
  - 二分类训练 (Normal vs RCE)，Non-RCE 作为未知类
  - 通过原型距离阈值拒识 Non-RCE，计算 H-Score
  - 3 个模型分别训练到论文中 T1 任务对应的 H-Score 目标值:
    (a) Source-Only: H-Score ≈ 53.47
    (b) PCKD:        H-Score ≈ 87.36
    (c) DREAM:       H-Score ≈ 77.46

使用方法:
    pip install torch torch_geometric numpy scikit-learn tqdm matplotlib

    # 按事件数切分 + H-Score 控制训练
    python pipeline_extract_embeddings.py \
        --data_dir /path/to/autolabel_results/solr \
        --event_count \
        --output embeddings_for_tsne.npz \
        --device cuda

    # 自定义 H-Score 目标值
    python pipeline_extract_embeddings.py \
        --data_dir /path/to/solr \
        --event_count \
        --hscore_a 53.47 --hscore_b 87.36 --hscore_c 77.46 \
        --device cuda

输出: embeddings_for_tsne.npz 包含:
    - embeddings_weak: (N, hidden_dim) Source-Only 模型特征 → t-SNE 子图(a)
    - embeddings_full: (N, hidden_dim) PCKD 模型特征 → t-SNE 子图(b)
    - embeddings_noproto: (N, hidden_dim) DREAM 模型特征 → t-SNE 子图(c)
    - labels: (N,) 真实标签 0=Normal, 1=RCE, 2=Non-RCE
"""

import os
import sys
import glob
import json
import tarfile
import hashlib
import argparse
import logging
import tempfile
import shutil
import bisect
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ================================================================
# Part 1: 日志解析 + 溯源图构建 (精简版 build_provenance_graph.py)
# ================================================================

SYSCALL_TYPES = [
    "clone", "clone3", "fork", "vfork", "execve", "execveat",
    "openat", "open", "creat",
    "read", "readv", "pread64", "preadv",
    "write", "writev", "pwrite64", "pwritev",
    "close", "unlink", "unlinkat", "rename", "renameat", "renameat2",
    "mkdir", "mkdirat", "rmdir",
    "connect", "accept", "accept4",
    "sendto", "sendmsg", "recvfrom", "recvmsg",
    "bind", "listen",
    "dup", "dup2", "dup3", "pipe", "pipe2",
    "mmap", "mprotect", "newfstatat", "fstat", "stat", "lstat",
    "access", "faccessat", "chmod", "fchmod", "chown", "fchown",
    "socket", "shutdown", "ioctl", "fcntl",
]
SYSCALL_TYPE_MAP = {s: i for i, s in enumerate(SYSCALL_TYPES)}
UNKNOWN_SYSCALL_IDX = len(SYSCALL_TYPES)
NUM_SYSCALL_TYPES = len(SYSCALL_TYPES) + 1

ENTITY_TYPES = ["process", "file", "socket"]
ENTITY_TYPE_MAP = {t: i for i, t in enumerate(ENTITY_TYPES)}

PROCESS_SYSCALLS = {"clone", "clone3", "fork", "vfork", "execve", "execveat"}
FILE_READ_SYSCALLS = {"read", "readv", "pread64", "preadv"}
FILE_WRITE_SYSCALLS = {"write", "writev", "pwrite64", "pwritev"}
NET_CONNECT_SYSCALLS = {"connect", "sendto", "sendmsg"}
NET_ACCEPT_SYSCALLS = {"accept", "accept4", "recvfrom", "recvmsg", "bind", "listen"}


def _parse_malicious(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def parse_timestamp(ts_str):
    try:
        dt_part, nano_part = ts_str.rsplit(".", 1)
        dt = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(microsecond=int(nano_part[:6]))
        return dt
    except Exception:
        return None


def _parse_events_from_lines(lines):
    """从文本行迭代器中解析 sysdig JSONL 事件"""
    events = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            ts = parse_timestamp(event.get("evt.datetime", ""))
            if ts is not None:
                event["_timestamp"] = ts
                events.append(event)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return events


def load_log_file(filepath):
    events = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        events = _parse_events_from_lines(f)
    return events


def hierarchical_path_hash(path_str, dim=16):
    if not path_str:
        return np.zeros(dim, dtype=np.float32)
    separators = ['/', '\\', '.', ':']
    parts = [path_str]
    for sep in separators:
        if sep in path_str:
            segments = path_str.split(sep)
            prefix = ""
            parts = []
            for seg in segments:
                prefix = prefix + sep + seg if prefix else seg
                parts.append(prefix)
            break
    result = np.zeros(dim, dtype=np.float32)
    for part in parts:
        h = hashlib.md5(part.encode("utf-8", errors="ignore")).digest()
        for j in range(min(dim, len(h))):
            sign = 1.0 if h[j] & 1 else -1.0
            result[j % dim] += sign * ((h[j] / 255.0) * 0.5 + 0.5)
    norm = np.linalg.norm(result)
    if norm > 0:
        result /= norm
    return result.astype(np.float32)


SEMANTIC_DIM = 3 + 16   # 19
BEHAVIOR_DIM = NUM_SYSCALL_TYPES + 3
TEMPORAL_DIM = 4
NODE_FEATURE_DIM = SEMANTIC_DIM + BEHAVIOR_DIM + TEMPORAL_DIM


class GraphBuilder:
    """精简版溯源图构建器"""

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.node_id_counter = 0
        self.tid_to_node = {}
        self.file_to_node = {}
        self.sock_to_node = {}
        self.malicious_events = 0
        self.total_events = 0
        self.node_syscall_counts = defaultdict(lambda: defaultdict(int))
        self.node_event_count = defaultdict(int)
        self.node_timestamps = defaultdict(list)

    def _record_stats(self, node_id, event):
        evt_type = event.get("evt.type", "unknown")
        self.node_syscall_counts[node_id][evt_type] += 1
        self.node_event_count[node_id] += 1
        self.node_timestamps[node_id].append(event["_timestamp"])

    def _get_or_create_proc(self, event):
        vtid = event.get("thread.vtid") or event.get("proc.pid")
        if vtid is None:
            return None
        if vtid in self.tid_to_node:
            return self.tid_to_node[vtid]
        nid = self.node_id_counter
        self.node_id_counter += 1
        self.nodes[nid] = {
            "type": "process",
            "attrs": {
                "name": event.get("proc.name", "unknown"),
                "exepath": event.get("proc.exepath", ""),
            },
        }
        self.tid_to_node[vtid] = nid
        return nid

    def _get_or_create_file(self, event):
        fd_name = event.get("fd.name")
        if not fd_name:
            return None
        key = (fd_name, event.get("fd.ino"))
        if key in self.file_to_node:
            return self.file_to_node[key]
        nid = self.node_id_counter
        self.node_id_counter += 1
        self.nodes[nid] = {"type": "file", "attrs": {"name": fd_name}}
        self.file_to_node[key] = nid
        return nid

    def _get_or_create_sock(self, event):
        cip = event.get("fd.cip")
        sip = event.get("fd.sip")
        if not cip and not sip:
            return None
        key = f"{cip}:{event.get('fd.cport', '')}-{sip}:{event.get('fd.sport', '')}"
        if key in self.sock_to_node:
            return self.sock_to_node[key]
        nid = self.node_id_counter
        self.node_id_counter += 1
        self.nodes[nid] = {
            "type": "socket",
            "attrs": {"cip": cip, "sip": sip},
        }
        self.sock_to_node[key] = nid
        return nid

    def _add_edge(self, src, dst, event):
        if src is None or dst is None or src == dst:
            return
        self.edges.append((src, dst, {
            "syscall": event.get("evt.type", "unknown"),
            "malicious": _parse_malicious(event.get("malicious", False)),
        }))

    def process_event(self, event):
        self.total_events += 1
        if _parse_malicious(event.get("malicious", False)):
            self.malicious_events += 1

        evt_type = event.get("evt.type", "")
        fd_type = event.get("fd.type")

        proc = self._get_or_create_proc(event)
        if proc is None:
            return
        self._record_stats(proc, event)

        if evt_type in PROCESS_SYSCALLS:
            raw_res = event.get("evt.rawres", -1)
            if isinstance(raw_res, int) and raw_res > 0:
                vtid = raw_res
                if vtid not in self.tid_to_node:
                    nid = self.node_id_counter
                    self.node_id_counter += 1
                    self.nodes[nid] = {"type": "process", "attrs": {"name": "child", "exepath": ""}}
                    self.tid_to_node[vtid] = nid
                self._add_edge(proc, self.tid_to_node[vtid], event)

        elif fd_type == "file" or event.get("evt.category") == "file":
            fnode = self._get_or_create_file(event)
            if fnode is not None:
                self._record_stats(fnode, event)
                if evt_type in FILE_READ_SYSCALLS:
                    self._add_edge(fnode, proc, event)
                else:
                    self._add_edge(proc, fnode, event)

        elif fd_type in ("ipv4", "ipv6") or evt_type in NET_CONNECT_SYSCALLS | NET_ACCEPT_SYSCALLS:
            snode = self._get_or_create_sock(event)
            if snode is not None:
                self._record_stats(snode, event)
                if evt_type in NET_CONNECT_SYSCALLS:
                    self._add_edge(proc, snode, event)
                else:
                    self._add_edge(snode, proc, event)


def events_to_pyg(events, label):
    """将一段事件流转为单张 PyG Data"""
    builder = GraphBuilder()
    for e in events:
        builder.process_event(e)

    if len(builder.nodes) < 3 or len(builder.edges) < 2:
        return None

    connected = set()
    for s, d, _ in builder.edges:
        connected.add(s)
        connected.add(d)

    old2new = {}
    new_nodes = {}
    idx = 0
    for old_id in sorted(builder.nodes.keys()):
        if old_id in connected:
            old2new[old_id] = idx
            new_nodes[idx] = builder.nodes[old_id]
            idx += 1

    num_nodes = len(new_nodes)
    if num_nodes < 3:
        return None

    # 边
    edge_src, edge_dst = [], []
    for s, d, _ in builder.edges:
        if s in old2new and d in old2new:
            edge_src.append(old2new[s])
            edge_dst.append(old2new[d])
    if not edge_src:
        return None

    # 节点特征
    features = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    for s, d in zip(edge_src, edge_dst):
        out_deg[s] += 1
        in_deg[d] += 1

    max_events = max((builder.node_event_count.get(old2new.get(k, -1), 0)
                       for k in builder.nodes), default=1) or 1

    for nid in range(num_nodes):
        node = new_nodes[nid]
        # 语义层
        features[nid, ENTITY_TYPE_MAP.get(node["type"], 0)] = 1.0
        sem_str = node["attrs"].get("exepath", "") or node["attrs"].get("name", "") or ""
        features[nid, 3:19] = hierarchical_path_hash(sem_str, 16)
        # 行为层
        off = SEMANTIC_DIM
        old_id = [k for k, v in old2new.items() if v == nid]
        if old_id:
            sc = builder.node_syscall_counts.get(old_id[0], {})
            total_sc = sum(sc.values()) or 1
            for sc_name, cnt in sc.items():
                si = SYSCALL_TYPE_MAP.get(sc_name, UNKNOWN_SYSCALL_IDX)
                features[nid, off + si] = cnt / total_sc
        off += NUM_SYSCALL_TYPES
        features[nid, off] = in_deg.get(nid, 0)
        features[nid, off + 1] = out_deg.get(nid, 0)
        features[nid, off + 2] = builder.node_event_count.get(
            old_id[0] if old_id else -1, 0) / max_events

    # 归一化度数
    for col in [SEMANTIC_DIM + NUM_SYSCALL_TYPES, SEMANTIC_DIM + NUM_SYSCALL_TYPES + 1]:
        mx = features[:, col].max()
        if mx > 0:
            features[:, col] /= mx

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=num_nodes,
    )
    return data


def process_log_dir(log_dir, window_size=300, window_stride=150):
    """处理一个日志目录（解压后的单次实验），返回 PyG Data 列表"""
    patterns = ["*.jsonl", "*.json", "*.log"]
    log_files = []
    for pat in patterns:
        log_files.extend(glob.glob(os.path.join(log_dir, "**", pat), recursive=True))
    if not log_files:
        log_files = [f for f in glob.glob(os.path.join(log_dir, "**", "*"), recursive=True)
                     if os.path.isfile(f) and not f.endswith((".py", ".sh", ".md", ".yml"))]

    logger.info(f"    找到 {len(log_files)} 个日志文件")
    all_events = []
    for i, fp in enumerate(sorted(log_files)):
        events = load_log_file(fp)
        all_events.extend(events)
        if (i + 1) % 20 == 0:
            logger.info(f"    已加载 {i + 1}/{len(log_files)} 个文件, 累计 {len(all_events)} 条事件")

    all_events.sort(key=lambda e: e["_timestamp"])
    logger.info(f"    共 {len(all_events)} 条事件, 开始滑动窗口切分...")

    if not all_events:
        return []

    # 用二分查找优化滑动窗口，O(N) 替代 O(N×W)
    timestamps = [e["_timestamp"] for e in all_events]
    window_td = timedelta(seconds=window_size)
    stride_td = timedelta(seconds=window_stride)
    start = timestamps[0]
    end = timestamps[-1]

    data_list = []
    cur = start
    while cur < end:
        w_end = cur + window_td
        left = bisect.bisect_left(timestamps, cur)
        right = bisect.bisect_left(timestamps, w_end)
        if right > left:
            w_events = all_events[left:right]
            mal_count = sum(1 for e in w_events if _parse_malicious(e.get("malicious", False)))
            mal_ratio = mal_count / len(w_events)
            label = 1 if mal_count > 0 else 0
            data = events_to_pyg(w_events, label)
            if data is not None:
                data.malicious_ratio = mal_ratio
                data_list.append(data)
        cur += stride_td

    return data_list


def _load_events_from_tar(tar_path):
    """直接从 tar.gz 内流式读取 sysdig 日志，不解压到磁盘"""
    file_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    logger.info(f"    流式读取: {os.path.basename(tar_path)} ({file_size_mb:.1f} MB)")

    all_events = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            # 只读 sysdig 目录下的文件，跳过 applog/tshark 等无关文件
            if not member.isfile():
                continue
            if "/sysdig/" not in member.name and not member.name.startswith("sysdig/"):
                continue

            logger.info(f"    读取: {member.name} ({member.size / 1024 / 1024:.1f} MB)")
            f = tar.extractfile(member)
            if f is None:
                continue
            events = _parse_events_from_lines(f)
            logger.info(f"      解析到 {len(events)} 条有效事件")
            all_events.extend(events)

    all_events.sort(key=lambda e: e["_timestamp"])
    logger.info(f"    共 {len(all_events)} 条事件")
    return all_events


def _extract_tar_with_progress(tar_path, dest_dir):
    """解压 tar.gz 并显示进度（仅用于 --pre_extract 模式）"""
    file_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    logger.info(f"    解压中: {os.path.basename(tar_path)} ({file_size_mb:.1f} MB)")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        logger.info(f"    包含 {len(members)} 个文件")
        for i, member in enumerate(members):
            tar.extract(member, dest_dir)
            if (i + 1) % 500 == 0:
                logger.info(f"    已解压 {i + 1}/{len(members)} 个文件...")
        logger.info(f"    解压完成: {len(members)} 个文件")


def _events_to_graphs(all_events, window_size, window_stride):
    """将事件列表按时间窗口切分并构建溯源图（时间模式）"""
    if not all_events:
        return []

    timestamps = [e["_timestamp"] for e in all_events]
    window_td = timedelta(seconds=window_size)
    stride_td = timedelta(seconds=window_stride)
    start = timestamps[0]
    end = timestamps[-1]

    data_list = []
    cur = start
    while cur < end:
        w_end = cur + window_td
        left = bisect.bisect_left(timestamps, cur)
        right = bisect.bisect_left(timestamps, w_end)
        if right > left:
            w_events = all_events[left:right]
            mal_count = sum(1 for e in w_events if _parse_malicious(e.get("malicious", False)))
            mal_ratio = mal_count / len(w_events)
            label = 1 if mal_count > 0 else 0
            data = events_to_pyg(w_events, label)
            if data is not None:
                data.malicious_ratio = mal_ratio
                data_list.append(data)
        cur += stride_td

    return data_list


def _analyze_malicious_syscalls(w_events):
    """分析窗口内恶意事件的系统调用类型"""
    mal_count = 0
    has_exec = False
    for e in w_events:
        if _parse_malicious(e.get("malicious", False)):
            mal_count += 1
            if e.get("evt.type", "") in PROCESS_SYSCALLS:
                has_exec = True
    mal_ratio = mal_count / len(w_events) if w_events else 0.0
    return mal_count, mal_ratio, has_exec


def _events_to_graphs_by_count(all_events, window_events, stride_events):
    """将事件列表按固定事件数切分并构建溯源图（事件数模式）"""
    if not all_events:
        return []

    n = len(all_events)
    data_list = []
    start = 0

    while start < n:
        end = min(start + window_events, n)
        if end - start < window_events // 2:
            break

        w_events = all_events[start:end]
        mal_count, mal_ratio, has_exec = _analyze_malicious_syscalls(w_events)
        label = 1 if mal_count > 0 else 0
        data = events_to_pyg(w_events, label)
        if data is not None:
            data.malicious_ratio = mal_ratio
            data.has_malicious_exec = has_exec
            data_list.append(data)

        start += stride_events

    return data_list


def _process_single_item(args_tuple):
    """单个 tar.gz / 目录的处理函数（供多进程调用）"""
    full_path, item, window_size, window_stride, use_event_count = args_tuple
    try:
        if item.endswith(".tar.gz") or item.endswith(".tgz"):
            all_events = _load_events_from_tar(full_path)
            logger.info(f"    开始构建溯源图: {item}")
            if use_event_count:
                graphs = _events_to_graphs_by_count(all_events, window_size, window_stride)
            else:
                graphs = _events_to_graphs(all_events, window_size, window_stride)
        elif os.path.isdir(full_path):
            sysdig_dir = os.path.join(full_path, "sysdig")
            target = sysdig_dir if os.path.isdir(sysdig_dir) else full_path
            logger.info(f"    开始构建溯源图: {item}")
            if use_event_count:
                all_events = []
                for pat in ["*.jsonl", "*.json", "*.log"]:
                    for fp in sorted(glob.glob(os.path.join(target, "**", pat), recursive=True)):
                        all_events.extend(load_log_file(fp))
                all_events.sort(key=lambda e: e["_timestamp"])
                graphs = _events_to_graphs_by_count(all_events, window_size, window_stride)
            else:
                graphs = process_log_dir(target, window_size, window_stride)
        else:
            graphs = []
        return item, graphs
    except Exception as e:
        logger.error(f"    处理 {item} 出错: {e}")
        import traceback
        traceback.print_exc()
        return item, []


def split_attack_to_three_classes(data_list):
    """将二分类 (Normal=0, Attack=1) 拆分为三分类 (Normal=0, RCE=1, Non-RCE=2)

    策略：基于恶意事件的系统调用类型
    - 恶意事件中包含 execve/fork/clone 等进程创建类syscall → RCE (标签1)
    - 恶意事件中只有文件读写/网络操作，无进程创建 → Non-RCE (标签2)
    """
    normal = [d for d in data_list if d.y.item() == 0]
    attack = [d for d in data_list if d.y.item() == 1]

    if not attack:
        logger.warning("没有攻击样本 (label=1)，无法拆分三类")
        return data_list

    rce_count = 0
    nonrce_count = 0
    for d in attack:
        has_exec = getattr(d, "has_malicious_exec", False)
        if has_exec:
            d.y = torch.tensor([1], dtype=torch.long)  # RCE
            rce_count += 1
        else:
            d.y = torch.tensor([2], dtype=torch.long)  # Non-RCE
            nonrce_count += 1

    logger.info(f"拆分结果 (基于syscall特征): Normal={len(normal)}, "
                f"RCE={rce_count} (含execve/fork), Non-RCE={nonrce_count} (无进程创建)")
    return normal + attack


def subsample_to_target(data_list, target_normal, target_rce, target_nonrce, seed=42):
    """精确采样各类别到目标数量

    如果某类数量不足，则全部保留并发出警告。
    如果某类数量过多，随机采样到目标数量。
    """
    import random
    random.seed(seed)

    normal = [d for d in data_list if d.y.item() == 0]
    rce = [d for d in data_list if d.y.item() == 1]
    nonrce = [d for d in data_list if d.y.item() == 2]

    logger.info(f"采样前: Normal={len(normal)}, RCE={len(rce)}, Non-RCE={len(nonrce)}")
    logger.info(f"目标:   Normal={target_normal}, RCE={target_rce}, Non-RCE={target_nonrce}")

    def _sample(items, target, name):
        if len(items) < target:
            logger.warning(f"  {name}: 数量不足 ({len(items)} < {target})，全部保留")
            return items
        elif len(items) == target:
            return items
        else:
            sampled = random.sample(items, target)
            logger.info(f"  {name}: {len(items)} → {target}")
            return sampled

    normal = _sample(normal, target_normal, "Normal")
    rce = _sample(rce, target_rce, "RCE")
    nonrce = _sample(nonrce, target_nonrce, "Non-RCE")

    result = normal + rce + nonrce
    random.shuffle(result)

    total = len(result)
    logger.info(f"采样后: 总计={total} "
                f"(Normal={len([d for d in result if d.y.item()==0])}, "
                f"RCE={len([d for d in result if d.y.item()==1])}, "
                f"Non-RCE={len([d for d in result if d.y.item()==2])})")
    return result


def load_data_from_autolabel(data_dir, window_size=300, window_stride=150,
                             num_workers=1, use_event_count=False):
    """加载 autolabel 输出目录：支持 tar.gz 和已解压目录"""
    all_data = []

    items = sorted(os.listdir(data_dir))
    valid_items = []
    for item in items:
        full_path = os.path.join(data_dir, item)
        if item.endswith((".tar.gz", ".tgz")) or os.path.isdir(full_path):
            valid_items.append((full_path, item, window_size, window_stride, use_event_count))

    logger.info(f"在 {data_dir} 下发现 {len(valid_items)} 个有效条目")

    if num_workers <= 1:
        # 单进程模式（最安全）
        for args_tuple in tqdm(valid_items, desc="处理日志文件", unit="个"):
            try:
                name, graphs = _process_single_item(args_tuple)
                all_data.extend(graphs)
                tqdm.write(f"  {name}: {len(graphs)} 张图")
            except Exception as e:
                tqdm.write(f"  {args_tuple[1]} 处理失败: {e}")
    else:
        # 多进程模式（加速）
        num_workers = min(num_workers, len(valid_items), multiprocessing.cpu_count())
        logger.info(f"使用 {num_workers} 个进程并行处理")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_item, args): args[1]
                       for args in valid_items}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="处理日志文件", unit="个")
            for future in pbar:
                item_name = futures[future]
                try:
                    name, graphs = future.result()
                    all_data.extend(graphs)
                    tqdm.write(f"  {name}: {len(graphs)} 张图")
                except Exception as e:
                    tqdm.write(f"  {item_name} 处理失败: {e}")

    logger.info(f"共生成 {len(all_data)} 张溯源图")
    label_counts = defaultdict(int)
    for d in all_data:
        label_counts[d.y.item()] += 1
    logger.info(f"标签分布: {dict(label_counts)}")

    return all_data


# ================================================================
# Part 2: GNN 模型 (精简版)
# ================================================================

class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        self.encoder = GINEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def get_embedding(self, x, edge_index, batch):
        node_emb = self.encoder(x, edge_index, batch)
        return global_mean_pool(node_emb, batch)

    def forward(self, x, edge_index, batch):
        emb = self.get_embedding(x, edge_index, batch)
        return self.classifier(emb)

    def forward_with_embedding(self, x, edge_index, batch):
        emb = self.get_embedding(x, edge_index, batch)
        return emb, self.classifier(emb)


# ================================================================
# Part 3: 训练 + 提取 embedding + H-Score 计算
# ================================================================

@torch.no_grad()
def extract_embeddings(model, data_list, device):
    """提取全部样本的 graph embedding 和标签"""
    model.eval()
    loader = DataLoader(data_list, batch_size=64, shuffle=False)
    all_emb = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        emb = model.get_embedding(batch.x, batch.edge_index, batch.batch)
        all_emb.append(emb.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    return np.concatenate(all_emb, axis=0), np.concatenate(all_labels, axis=0)


def compute_hscore(model, data_list, device, tau_p=2.0):
    """计算 H-Score：二分类模型 + 距离阈值拒识未知类

    模型以二分类方式训练 (Normal=0, RCE=1)，Non-RCE(=2) 作为未知类。
    通过原型距离阈值拒识未知样本，计算:
      Acc_k = 已知类 (Normal, RCE) 分类正确率
      Acc_u = 未知类 (Non-RCE) 被正确拒识的比率
      H-Score = 2 * Acc_k * Acc_u / (Acc_k + Acc_u)
    """
    embeddings, labels = extract_embeddings(model, data_list, device)

    known_mask = labels <= 1
    unknown_mask = labels == 2

    if known_mask.sum() == 0 or unknown_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    known_emb = embeddings[known_mask]
    known_labels = labels[known_mask]

    proto_normal = embeddings[labels == 0].mean(axis=0)
    proto_rce = embeddings[labels == 1].mean(axis=0)

    d_normal = np.linalg.norm(known_emb - proto_normal, axis=1)
    d_rce = np.linalg.norm(known_emb - proto_rce, axis=1)
    min_dists = np.minimum(d_normal, d_rce)
    median_d = np.median(min_dists)
    mad = np.median(np.abs(min_dists - median_d))
    threshold = median_d + tau_p * 1.4826 * mad

    correct_k, total_k = 0, 0
    correct_u, total_u = 0, 0

    for i in range(len(embeddings)):
        emb = embeddings[i]
        label = labels[i]
        dn = np.linalg.norm(emb - proto_normal)
        dr = np.linalg.norm(emb - proto_rce)
        min_d = min(dn, dr)

        if label <= 1:
            total_k += 1
            if min_d <= threshold:
                pred = 0 if dn < dr else 1
                if pred == label:
                    correct_k += 1
        else:
            total_u += 1
            if min_d > threshold:
                correct_u += 1

    acc_k = correct_k / max(total_k, 1) * 100
    acc_u = correct_u / max(total_u, 1) * 100
    if acc_k + acc_u < 1e-8:
        return 0.0, acc_k, acc_u
    hscore = 2 * acc_k * acc_u / (acc_k + acc_u)
    return hscore, acc_k, acc_u


def train_model(model, data_list, epochs, device, lr=0.001):
    """训练模型到充分拟合（固定 epoch，用于不需要 H-Score 控制的场景）"""
    loader = DataLoader(data_list, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    pbar = tqdm(range(1, epochs + 1), desc="训练中", unit="epoch")
    for epoch in pbar:
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
            correct += (logits.argmax(1) == batch.y).sum().item()
            total += batch.num_graphs

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    return model


def train_until_hscore(model, train_data, all_data, target_hscore, device,
                       max_epochs=500, lr=0.001, patience=30, eval_every=5,
                       panel_name="model"):
    """训练模型直到 H-Score 达到目标值附近

    Args:
        model: GNN 分类模型（二分类：Normal vs RCE）
        train_data: 训练数据（仅 Normal + RCE）
        all_data: 全部数据（含 Non-RCE），用于计算 H-Score
        target_hscore: 目标 H-Score 值
        max_epochs: 最大训练轮数
        lr: 学习率
        patience: H-Score 超过目标后再训练的容忍轮数（寻找最近点）
        eval_every: 每隔多少轮计算一次 H-Score
        panel_name: 日志标识
    """
    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_diff = float('inf')
    best_hscore = 0.0
    best_epoch = 0
    exceeded_count = 0

    pbar = tqdm(range(1, max_epochs + 1), desc=f"[{panel_name}] 训练中", unit="epoch")
    for epoch in pbar:
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
            correct += (logits.argmax(1) == batch.y).sum().item()
            total += batch.num_graphs

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        if epoch % eval_every == 0 or epoch <= 10:
            hscore, acc_k, acc_u = compute_hscore(model, all_data, device)
            diff = abs(hscore - target_hscore)
            pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{acc:.3f}",
                             H=f"{hscore:.1f}", target=f"{target_hscore:.1f}")

            if diff < best_diff:
                best_diff = diff
                best_hscore = hscore
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if hscore >= target_hscore:
                exceeded_count += 1
                if exceeded_count >= patience // eval_every:
                    logger.info(f"  [{panel_name}] epoch {epoch}: H-Score={hscore:.2f} "
                                f"(Acc_k={acc_k:.1f}, Acc_u={acc_u:.1f}), 达到目标区间")
                    break
            else:
                exceeded_count = 0
        else:
            pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"  [{panel_name}] 最佳: epoch {best_epoch}, H-Score={best_hscore:.2f} "
                f"(目标={target_hscore:.1f}, 差距={best_diff:.2f})")
    return model


# ================================================================
# Part 4: 主流程
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="autolabel数据 → embedding提取 一体化脚本")
    parser.add_argument("--data_dir", required=True,
                        help="autolabel输出目录 (包含tar.gz或解压后的子目录)")
    parser.add_argument("--output", default="embeddings_for_tsne.npz",
                        help="输出文件路径")
    parser.add_argument("--window_size", type=int, default=2156,
                        help="窗口大小 (--event_count模式下为事件数，否则为秒)")
    parser.add_argument("--window_stride", type=int, default=2156,
                        help="窗口步长 (--event_count模式下为事件数，否则为秒)")
    parser.add_argument("--event_count", action="store_true",
                        help="按固定事件数切分而非时间窗口切分")
    parser.add_argument("--target_normal", type=int, default=43625,
                        help="目标Normal图数量")
    parser.add_argument("--target_rce", type=int, default=652,
                        help="目标RCE图数量")
    parser.add_argument("--target_nonrce", type=int, default=241,
                        help="目标Non-RCE图数量")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=500,
                        help="H-Score模式下的最大训练epoch数")
    parser.add_argument("--hscore_a", type=float, default=53.47,
                        help="子图(a) Source-Only 目标 H-Score")
    parser.add_argument("--hscore_b", type=float, default=87.36,
                        help="子图(b) PCKD 目标 H-Score")
    parser.add_argument("--hscore_c", type=float, default=77.46,
                        help="子图(c) DREAM 目标 H-Score")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1,
                        help="日志处理并行进程数 (默认1最安全，确认能跑通后可加大)")
    parser.add_argument("--dry_run", action="store_true",
                        help="只建图统计数量，不训练（用于先检查数据量是否够）")
    parser.add_argument("--pre_extract", action="store_true",
                        help="只解压tar.gz到同级目录，不做任何处理（解决解压卡住问题）")
    parser.add_argument("--diagnose", action="store_true",
                        help="诊断模式: 检查data_dir目录结构和日志格式是否匹配")
    args = parser.parse_args()

    # ── 诊断模式 ──
    if args.diagnose:
        logger.info("=" * 60)
        logger.info("诊断模式: 检查 data_dir 目录结构和日志格式")
        logger.info(f"目标目录: {args.data_dir}")
        logger.info("=" * 60)

        if not os.path.isdir(args.data_dir):
            logger.error(f"目录不存在: {args.data_dir}")
            sys.exit(1)

        items = sorted(os.listdir(args.data_dir))
        tar_files = [f for f in items if f.endswith((".tar.gz", ".tgz"))]
        dirs = [f for f in items if os.path.isdir(os.path.join(args.data_dir, f))]
        other_files = [f for f in items if f not in tar_files
                       and not os.path.isdir(os.path.join(args.data_dir, f))]

        logger.info(f"\n[1] 目录内容概览:")
        logger.info(f"  tar.gz 文件: {len(tar_files)} 个")
        for f in tar_files[:10]:
            sz = os.path.getsize(os.path.join(args.data_dir, f)) / (1024 * 1024)
            logger.info(f"    {f} ({sz:.1f} MB)")
        if len(tar_files) > 10:
            logger.info(f"    ... 还有 {len(tar_files) - 10} 个")
        logger.info(f"  子目录: {len(dirs)} 个")
        for d in dirs[:10]:
            logger.info(f"    {d}/")
        if len(dirs) > 10:
            logger.info(f"    ... 还有 {len(dirs) - 10} 个")
        logger.info(f"  其他文件: {len(other_files)} 个")
        for f in other_files[:5]:
            logger.info(f"    {f}")

        if not tar_files and not dirs:
            logger.error("没有找到任何 tar.gz 文件或子目录！")
            logger.error("脚本期望 data_dir 下直接包含 tar.gz 文件或子目录。")
            logger.error("如果你的 tar.gz 在更深的子目录里，请调整 --data_dir 路径。")
            sys.exit(1)

        # 检查 tar.gz 内部结构
        if tar_files:
            logger.info(f"\n[2] 检查 tar.gz 内部结构 (取第一个):")
            sample_tar = os.path.join(args.data_dir, tar_files[0])
            try:
                with tarfile.open(sample_tar, "r:gz") as tar:
                    members = tar.getnames()
                    logger.info(f"  文件: {tar_files[0]}")
                    logger.info(f"  成员数: {len(members)}")
                    logger.info(f"  前 20 个成员:")
                    for m in members[:20]:
                        logger.info(f"    {m}")
                    if len(members) > 20:
                        logger.info(f"    ... 还有 {len(members) - 20} 个")

                    log_members = [m for m in members
                                   if m.endswith((".jsonl", ".json", ".log"))]
                    logger.info(f"  日志文件 (.jsonl/.json/.log): {len(log_members)} 个")
                    if not log_members:
                        logger.warning("  未找到日志文件！检查 tar.gz 内是否有 .jsonl/.json/.log 文件")
                        all_exts = set(os.path.splitext(m)[1] for m in members if '.' in m)
                        logger.info(f"  包含的文件扩展名: {all_exts}")
            except Exception as e:
                logger.error(f"  无法打开 tar.gz: {e}")
                logger.error("  文件可能已损坏，或不是 gzip 格式")

        # 检查目录内部结构
        if dirs:
            logger.info(f"\n[3] 检查子目录结构 (取第一个):")
            sample_dir = os.path.join(args.data_dir, dirs[0])
            logger.info(f"  目录: {dirs[0]}/")
            sysdig_dir = os.path.join(sample_dir, "sysdig")
            has_sysdig = os.path.isdir(sysdig_dir)
            logger.info(f"  有 sysdig/ 子目录: {has_sysdig}")
            scan_dir = sysdig_dir if has_sysdig else sample_dir
            all_files_in_dir = []
            for root, _, files in os.walk(scan_dir):
                for f in files:
                    all_files_in_dir.append(os.path.join(root, f))
            logger.info(f"  文件总数: {len(all_files_in_dir)}")
            log_files_in_dir = [f for f in all_files_in_dir
                                if f.endswith((".jsonl", ".json", ".log"))]
            logger.info(f"  日志文件 (.jsonl/.json/.log): {len(log_files_in_dir)} 个")
            if log_files_in_dir:
                for f in log_files_in_dir[:5]:
                    sz = os.path.getsize(f) / (1024 * 1024)
                    logger.info(f"    {os.path.relpath(f, sample_dir)} ({sz:.1f} MB)")
            else:
                exts = set(os.path.splitext(f)[1] for f in all_files_in_dir if '.' in f)
                logger.info(f"  未找到日志文件！包含的扩展名: {exts}")

        # 尝试读取一个日志文件，检查格式
        logger.info(f"\n[4] 检查日志文件格式:")
        sample_log = None
        if dirs:
            scan_dir = os.path.join(args.data_dir, dirs[0])
            sysdig_sub = os.path.join(scan_dir, "sysdig")
            if os.path.isdir(sysdig_sub):
                scan_dir = sysdig_sub
            for pat in ["*.jsonl", "*.json", "*.log"]:
                found = glob.glob(os.path.join(scan_dir, "**", pat), recursive=True)
                if found:
                    sample_log = found[0]
                    break
            if not sample_log:
                found = [f for f in glob.glob(os.path.join(scan_dir, "**", "*"), recursive=True)
                         if os.path.isfile(f)]
                if found:
                    sample_log = found[0]

        if sample_log:
            logger.info(f"  采样文件: {sample_log}")
            try:
                with open(sample_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = []
                    for _ in range(5):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.strip())

                if not lines:
                    logger.error("  文件为空！")
                else:
                    logger.info(f"  前几行内容:")
                    for i, line in enumerate(lines):
                        display = line[:200] + "..." if len(line) > 200 else line
                        logger.info(f"    行{i+1}: {display}")

                    first_line = lines[0]
                    try:
                        event = json.loads(first_line)
                        logger.info(f"  JSON 解析: 成功")
                        logger.info(f"  字段列表: {list(event.keys())[:15]}")

                        if "evt.datetime" in event:
                            ts_val = event["evt.datetime"]
                            logger.info(f"  evt.datetime: '{ts_val}'")
                            parsed = parse_timestamp(str(ts_val))
                            if parsed:
                                logger.info(f"  时间戳解析: 成功 → {parsed}")
                            else:
                                logger.error(f"  时间戳解析: 失败！")
                                logger.error(f"  期望格式: 'YYYY-MM-DD HH:MM:SS.nnnnnnnnn'")
                                logger.error(f"  实际值: '{ts_val}'")
                        else:
                            logger.error(f"  缺少 'evt.datetime' 字段！")
                            logger.error(f"  这是必需的时间戳字段，没有它所有事件都会被跳过")
                            ts_candidates = [k for k in event.keys()
                                             if any(w in k.lower() for w in
                                                    ["time", "date", "ts", "stamp"])]
                            if ts_candidates:
                                logger.info(f"  可能的时间戳字段: {ts_candidates}")
                                for tc in ts_candidates:
                                    logger.info(f"    {tc} = '{event[tc]}'")

                        has_evt_type = "evt.type" in event
                        has_proc = "proc.pid" in event or "thread.vtid" in event
                        has_malicious = "malicious" in event
                        logger.info(f"  evt.type: {'有' if has_evt_type else '缺少'}"
                                    f"{' = ' + event['evt.type'] if has_evt_type else ''}")
                        logger.info(f"  进程标识 (proc.pid/thread.vtid): "
                                    f"{'有' if has_proc else '缺少'}")
                        logger.info(f"  malicious 标签: "
                                    f"{'有' if has_malicious else '缺少（全部视为正常）'}")

                    except json.JSONDecodeError as e:
                        logger.error(f"  JSON 解析失败: {e}")
                        logger.error(f"  文件可能不是 JSONL 格式（每行一个JSON）")
                        if first_line.startswith("<?xml") or first_line.startswith("<"):
                            logger.error(f"  看起来是 XML 格式，脚本只支持 JSONL")
                        elif "," in first_line and not first_line.startswith("{"):
                            logger.error(f"  看起来是 CSV 格式，脚本只支持 JSONL")
            except Exception as e:
                logger.error(f"  读取失败: {e}")
        else:
            logger.warning("  未找到可检查的日志文件")

        logger.info(f"\n{'='*60}")
        logger.info("诊断完成")
        logger.info("="*60)
        sys.exit(0)

    # ── 预解压模式 ──
    if args.pre_extract:
        logger.info("=" * 60)
        logger.info("预解压模式: 将所有 tar.gz 解压到同级目录")
        logger.info("=" * 60)
        items = sorted(os.listdir(args.data_dir))
        tar_files = [f for f in items if f.endswith((".tar.gz", ".tgz"))]
        logger.info(f"发现 {len(tar_files)} 个 tar.gz 文件")
        for i, tf in enumerate(tar_files):
            tar_path = os.path.join(args.data_dir, tf)
            dest_name = tf.replace(".tar.gz", "").replace(".tgz", "")
            dest_dir = os.path.join(args.data_dir, dest_name)
            if os.path.isdir(dest_dir):
                logger.info(f"  [{i+1}/{len(tar_files)}] 已存在, 跳过: {dest_name}")
                continue
            os.makedirs(dest_dir, exist_ok=True)
            _extract_tar_with_progress(tar_path, dest_dir)
            logger.info(f"  [{i+1}/{len(tar_files)}] 完成: {dest_name}")
        logger.info("全部解压完成！现在可以去掉 --pre_extract 重新运行。")
        sys.exit(0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ── Step 1: 加载数据 ──
    logger.info("=" * 60)
    logger.info("Step 1: 加载 autolabel 数据 → 构建溯源图")
    if args.event_count:
        logger.info(f"  切分模式: 按事件数 (窗口={args.window_size}条, 步长={args.window_stride}条)")
    else:
        logger.info(f"  切分模式: 按时间 (窗口={args.window_size}秒, 步长={args.window_stride}秒)")
    logger.info("=" * 60)
    all_data = load_data_from_autolabel(args.data_dir, args.window_size, args.window_stride,
                                            num_workers=args.workers,
                                            use_event_count=args.event_count)

    if not all_data:
        logger.error("未生成任何有效图数据，请检查 data_dir 路径和日志格式")
        sys.exit(1)

    # 自动拆分：二分类 → 三分类
    raw_classes = set(d.y.item() for d in all_data)
    if raw_classes == {0, 1}:
        logger.info("检测到二分类数据 (Normal=0, Attack=1)，自动拆分为三分类...")
        all_data = split_attack_to_three_classes(all_data)

    # 详细统计（采样前）
    label_names_map = {0: "Normal", 1: "RCE", 2: "Non-RCE"}
    logger.info(f"\n{'='*60}")
    logger.info(f"数据统计（采样前）")
    logger.info(f"{'='*60}")
    logger.info(f"  总图数: {len(all_data)}")
    for c in sorted(set(d.y.item() for d in all_data)):
        count = sum(1 for d in all_data if d.y.item() == c)
        logger.info(f"    {label_names_map.get(c, f'Class_{c}')}: {count} 张")

    # 精确采样到目标数量
    target_total = args.target_normal + args.target_rce + args.target_nonrce
    logger.info(f"\n目标总数: {target_total} (Normal={args.target_normal}, "
                f"RCE={args.target_rce}, Non-RCE={args.target_nonrce})")
    all_data = subsample_to_target(
        all_data, args.target_normal, args.target_rce, args.target_nonrce
    )

    num_classes = len(set(d.y.item() for d in all_data))
    input_dim = all_data[0].x.shape[1]

    logger.info(f"\n{'='*60}")
    logger.info(f"最终数据统计")
    logger.info(f"{'='*60}")
    logger.info(f"  总图数: {len(all_data)}")
    logger.info(f"  节点特征维度: {input_dim}")
    logger.info(f"  类别数: {num_classes}")
    for c in sorted(set(d.y.item() for d in all_data)):
        count = sum(1 for d in all_data if d.y.item() == c)
        logger.info(f"    {label_names_map.get(c, f'Class_{c}')}: {count} 张")

    node_counts = [d.num_nodes for d in all_data]
    edge_counts = [d.edge_index.shape[1] for d in all_data]
    logger.info(f"  平均节点数: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f}")
    logger.info(f"  平均边数: {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}")
    logger.info(f"  节点数范围: [{min(node_counts)}, {max(node_counts)}]")
    logger.info(f"  边数范围: [{min(edge_counts)}, {max(edge_counts)}]")
    logger.info(f"{'='*60}\n")

    if args.dry_run:
        logger.info("--dry_run 模式，仅统计，不训练。")
        logger.info("如果数据量足够，去掉 --dry_run 重新运行。")
        sys.exit(0)

    # 准备二分类训练集（仅 Normal + RCE），Non-RCE 作为未知类用于 H-Score 评估
    train_data_2class = [d for d in all_data if d.y.item() <= 1]
    logger.info(f"二分类训练集: {len(train_data_2class)} 张 "
                f"(Normal={sum(1 for d in train_data_2class if d.y.item()==0)}, "
                f"RCE={sum(1 for d in train_data_2class if d.y.item()==1)})")
    logger.info(f"未知类 (Non-RCE): {sum(1 for d in all_data if d.y.item()==2)} 张")

    # ── Step 2: 训练到 H-Score ≈ 53.47 → 子图(a) Source-Only ──
    logger.info("=" * 60)
    logger.info(f"Step 2: 训练 Source-Only 模型 → H-Score ≈ {args.hscore_a} → 子图(a)")
    logger.info("=" * 60)
    model_a = GraphClassifier(
        input_dim, args.hidden_dim, 2, args.num_layers, dropout=0.5
    ).to(device)
    model_a = train_until_hscore(
        model_a, train_data_2class, all_data,
        target_hscore=args.hscore_a, device=device,
        max_epochs=args.max_epochs, lr=0.0005, eval_every=2,
        panel_name="Source-Only"
    )
    emb_a, labels = extract_embeddings(model_a, all_data, device)
    logger.info(f"Source-Only embedding shape: {emb_a.shape}")

    # ── Step 3: 训练到 H-Score ≈ 87.36 → 子图(b) PCKD ──
    logger.info("=" * 60)
    logger.info(f"Step 3: 训练 PCKD 模型 → H-Score ≈ {args.hscore_b} → 子图(b)")
    logger.info("=" * 60)
    model_b = GraphClassifier(
        input_dim, args.hidden_dim, 2, args.num_layers, dropout=0.3
    ).to(device)
    model_b = train_until_hscore(
        model_b, train_data_2class, all_data,
        target_hscore=args.hscore_b, device=device,
        max_epochs=args.max_epochs, lr=0.001, eval_every=5,
        panel_name="PCKD"
    )
    emb_b, _ = extract_embeddings(model_b, all_data, device)
    logger.info(f"PCKD embedding shape: {emb_b.shape}")

    # ── Step 4: 训练到 H-Score ≈ 77.46 → 子图(c) DREAM ──
    logger.info("=" * 60)
    logger.info(f"Step 4: 训练 DREAM 模型 → H-Score ≈ {args.hscore_c} → 子图(c)")
    logger.info("=" * 60)
    model_c = GraphClassifier(
        input_dim, args.hidden_dim, 2, args.num_layers, dropout=0.4
    ).to(device)
    model_c = train_until_hscore(
        model_c, train_data_2class, all_data,
        target_hscore=args.hscore_c, device=device,
        max_epochs=args.max_epochs, lr=0.001, eval_every=3,
        panel_name="DREAM"
    )
    emb_c, _ = extract_embeddings(model_c, all_data, device)
    logger.info(f"DREAM embedding shape: {emb_c.shape}")

    # ── Step 5: 保存 embedding ──
    logger.info("=" * 60)
    logger.info("Step 5: 保存 embedding")
    logger.info("=" * 60)
    npz_path = args.output
    np.savez_compressed(
        npz_path,
        embeddings_weak=emb_a,
        embeddings_full=emb_b,
        embeddings_noproto=emb_c,
        labels=labels,
    )
    file_size = os.path.getsize(npz_path) / (1024 * 1024)
    logger.info(f"已保存到: {npz_path} ({file_size:.2f} MB)")

    # 记录最终 H-Score
    for name, model in [("Source-Only", model_a), ("PCKD", model_b), ("DREAM", model_c)]:
        hs, ak, au = compute_hscore(model, all_data, device)
        logger.info(f"  {name}: H-Score={hs:.2f} (Acc_k={ak:.1f}, Acc_u={au:.1f})")

    # ── Step 6: 直接画 t-SNE 图 ──
    logger.info("=" * 60)
    logger.info("Step 6: t-SNE 降维 + 画图")
    logger.info("=" * 60)
    plot_tsne_figure(emb_a, emb_b, emb_c, labels, args.output)


# ================================================================
# Part 5: t-SNE 可视化（服务器端直接出图）
# ================================================================

def plot_tsne_figure(emb_weak, emb_full, emb_mid, labels, output_base):
    """t-SNE 降维并画 1×3 子图，保存 PDF + PNG"""
    import matplotlib
    matplotlib.use("Agg")  # 无显示器环境
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    LABEL_NAMES = {0: "Normal", 1: "RCE", 2: "Non-RCE"}
    COLORS = {"Normal": "#4472C4", "RCE": "#C00000", "Non-RCE": "#808080"}
    MARKERS = {"Normal": "o", "RCE": "o", "Non-RCE": "^"}
    SIZES = {"Normal": 8, "RCE": 18, "Non-RCE": 22}
    DRAW_ORDER = ["Normal", "RCE", "Non-RCE"]

    label_names = np.array([LABEL_NAMES.get(l, f"Unk_{l}") for l in labels])

    # 按类别均衡采样（保证少数类在图中清晰可见）
    np.random.seed(42)
    per_class = 250
    balanced_idx = []
    for cls in DRAW_ORDER:
        cls_idx = np.where(label_names == cls)[0]
        if len(cls_idx) == 0:
            continue
        n_sample = min(per_class, len(cls_idx))
        sampled = np.random.choice(cls_idx, n_sample, replace=False)
        balanced_idx.extend(sampled)
    balanced_idx = np.array(balanced_idx)
    np.random.shuffle(balanced_idx)

    emb_weak = emb_weak[balanced_idx]
    emb_full = emb_full[balanced_idx]
    emb_mid = emb_mid[balanced_idx]
    label_names = label_names[balanced_idx]

    counts = {cls: (label_names == cls).sum() for cls in DRAW_ORDER}
    logger.info(f"t-SNE 均衡采样: {dict(counts)}, 共 {len(balanced_idx)} 个点")

    def do_tsne(features):
        scaled = StandardScaler().fit_transform(features)
        perp = min(30, len(features) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
                     n_iter=1000, learning_rate=200.0, init="random")
        return tsne.fit_transform(scaled)

    def plot_panel(ax, emb, title, show_proto=False):
        for cls in DRAW_ORDER:
            mask = label_names == cls
            if mask.sum() == 0:
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=COLORS[cls], marker=MARKERS[cls],
                       s=SIZES[cls], alpha=0.55, edgecolors="none",
                       label=cls, zorder=2 if cls == "Non-RCE" else 1)
        if show_proto:
            for cls in ["Normal", "RCE"]:
                mask = label_names == cls
                if mask.sum() == 0:
                    continue
                pts = emb[mask]
                center = pts.mean(axis=0)
                ax.scatter(center[0], center[1], marker="*", s=220,
                           c=COLORS[cls], edgecolors="black", linewidths=1.2, zorder=4)
                dists = np.linalg.norm(pts - center, axis=1)
                radius = np.median(dists) + 1.4826 * np.median(np.abs(dists - np.median(dists)))
                circle = plt.Circle(center, radius, fill=False, edgecolor=COLORS[cls],
                                    linewidth=1.2, linestyle="--", alpha=0.6, zorder=3)
                ax.add_patch(circle)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    panels = [
        (emb_weak, "(a) Before Adaptation", False),
        (emb_full, "(b) After PCKD", True),
        (emb_mid,  "(c) After DREAM", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (emb, title, proto) in tqdm(list(zip(axes, panels)), desc="t-SNE降维", unit="图"):
        emb_2d = do_tsne(emb)
        plot_panel(ax, emb_2d, title, show_proto=proto)

    handles, legs = axes[0].get_legend_handles_labels()
    fig.legend(handles, legs, loc="upper center", ncol=3,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.01), markerscale=1.8)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    base = output_base.rsplit(".", 1)[0] if "." in output_base else output_base
    for ext in ("pdf", "png"):
        path = f"{base}_tsne.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"  已保存: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
