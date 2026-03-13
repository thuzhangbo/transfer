"""
AUTOLABEL Sysdig日志 → 溯源图(Provenance Graph) → PyG数据集 构建脚本

使用方法:
    python build_provenance_graph.py \
        --input_dir ./autolabel_data/cve-2019-17558 \
        --output_dir ./processed_graphs \
        --scenario_name CVE-2019-17558 \
        --attack_type RCE \
        --window_size 300 \
        --window_stride 150

输入: AUTOLABEL导出的JSON格式sysdig审计日志（每行一个JSON事件）
输出: PyG格式的溯源图数据集(.pt文件)
"""

import os
import json
import glob
import hashlib
import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 全局常量
# ============================================================

ENTITY_TYPES = ["process", "file", "socket"]
ENTITY_TYPE_MAP = {t: i for i, t in enumerate(ENTITY_TYPES)}


def _parse_malicious(value):
    """将 malicious 字段统一解析为 bool，兼容 bool/str/int 类型"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return False

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
    "socket", "shutdown",
    "ioctl", "fcntl",
]
SYSCALL_TYPE_MAP = {s: i for i, s in enumerate(SYSCALL_TYPES)}
UNKNOWN_SYSCALL_IDX = len(SYSCALL_TYPES)

PROCESS_SYSCALLS = {"clone", "clone3", "fork", "vfork", "execve", "execveat"}
FILE_READ_SYSCALLS = {"read", "readv", "pread64", "preadv"}
FILE_WRITE_SYSCALLS = {"write", "writev", "pwrite64", "pwritev"}
FILE_OPEN_SYSCALLS = {"openat", "open", "creat"}
FILE_META_SYSCALLS = {"newfstatat", "fstat", "stat", "lstat", "access", "faccessat",
                       "chmod", "fchmod", "chown", "fchown", "unlink", "unlinkat",
                       "rename", "renameat", "renameat2", "mkdir", "mkdirat", "rmdir",
                       "close"}
NET_CONNECT_SYSCALLS = {"connect", "sendto", "sendmsg"}
NET_ACCEPT_SYSCALLS = {"accept", "accept4", "recvfrom", "recvmsg", "bind", "listen"}


# ============================================================
# Step 1: 日志解析
# ============================================================

def parse_timestamp(ts_str):
    """将AUTOLABEL时间戳解析为datetime对象"""
    try:
        dt_part, nano_part = ts_str.rsplit(".", 1)
        dt = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
        microseconds = int(nano_part[:6])
        dt = dt.replace(microsecond=microseconds)
        return dt
    except Exception:
        return None


def load_log_file(filepath):
    """加载单个JSONL日志文件，返回事件列表"""
    events = []
    line_count = 0
    error_count = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
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
                error_count += 1

    logger.info(f"  加载 {filepath}: {len(events)}/{line_count} 条事件, {error_count} 条解析失败")
    return events


def load_scenario_logs(input_dir):
    """加载一个场景目录下的所有JSONL日志文件"""
    patterns = ["*.jsonl", "*.json", "*.log"]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(os.path.join(input_dir, "**", pat), recursive=True))

    if not all_files:
        logger.warning(f"在 {input_dir} 下未找到日志文件，尝试加载所有文件...")
        all_files = [f for f in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
                     if os.path.isfile(f)]

    logger.info(f"找到 {len(all_files)} 个日志文件")

    all_events = []
    for fp in sorted(all_files):
        events = load_log_file(fp)
        all_events.extend(events)

    all_events.sort(key=lambda e: e["_timestamp"])
    logger.info(f"共加载 {len(all_events)} 条事件")
    return all_events


# ============================================================
# Step 2: 溯源图构建
# ============================================================

class ProvenanceGraphBuilder:
    """从sysdig事件流构建溯源图

    改进：
    - 进程节点用 thread.vtid 标识（与AUTOLABEL原始代码一致）
    - 追踪每个节点的系统调用频率、时间戳统计（供三层特征工程使用）
    """

    def __init__(self):
        self.nodes = {}          # node_id -> {type, attrs}
        self.edges = []          # (src_id, dst_id, edge_attrs)
        self.node_id_counter = 0
        self.tid_to_node = {}    # thread.vtid -> node_id (改用vtid)
        self.file_to_node = {}   # (fd.name, fd.ino) -> node_id
        self.sock_to_node = {}   # (cip:cport-sip:sport) -> node_id
        self.malicious_events = 0
        self.total_events = 0

        self.node_syscall_counts = defaultdict(lambda: defaultdict(int))
        self.node_event_count = defaultdict(int)
        self.node_timestamps = defaultdict(list)

    def _record_node_stats(self, node_id, event):
        """记录节点级统计信息（系统调用分布、时间戳）"""
        evt_type = event.get("evt.type", "unknown")
        self.node_syscall_counts[node_id][evt_type] += 1
        self.node_event_count[node_id] += 1
        self.node_timestamps[node_id].append(event["_timestamp"])

    def _get_or_create_process_node(self, event):
        """获取或创建进程节点（用thread.vtid标识）"""
        vtid = event.get("thread.vtid")
        if vtid is None:
            vtid = event.get("proc.pid")
        if vtid is None:
            return None

        if vtid in self.tid_to_node:
            node_id = self.tid_to_node[vtid]
            node = self.nodes[node_id]
            name = event.get("proc.name", "")
            if name and name != "runc:[2:INIT]":
                node["attrs"]["name"] = name
                node["attrs"]["cmdline"] = event.get("proc.cmdline", "")
                node["attrs"]["exepath"] = event.get("proc.exepath", "")
            return node_id

        node_id = self.node_id_counter
        self.node_id_counter += 1
        self.nodes[node_id] = {
            "type": "process",
            "attrs": {
                "vtid": vtid,
                "pid": event.get("proc.pid"),
                "name": event.get("proc.name", "unknown"),
                "cmdline": event.get("proc.cmdline", ""),
                "exepath": event.get("proc.exepath", ""),
            }
        }
        self.tid_to_node[vtid] = node_id
        return node_id

    def _get_or_create_file_node(self, event):
        """获取或创建文件节点"""
        fd_name = event.get("fd.name")
        if not fd_name:
            return None

        fd_ino = event.get("fd.ino")
        key = (fd_name, fd_ino)

        if key in self.file_to_node:
            return self.file_to_node[key]

        node_id = self.node_id_counter
        self.node_id_counter += 1
        self.nodes[node_id] = {
            "type": "file",
            "attrs": {
                "name": fd_name,
                "directory": event.get("fd.directory", ""),
                "ino": fd_ino,
            }
        }
        self.file_to_node[key] = node_id
        return node_id

    def _get_or_create_socket_node(self, event):
        """获取或创建套接字节点"""
        cip = event.get("fd.cip")
        sip = event.get("fd.sip")
        if not cip and not sip:
            return None

        cport = event.get("fd.cport", "")
        sport = event.get("fd.sport", "")
        key = f"{cip}:{cport}-{sip}:{sport}"

        if key in self.sock_to_node:
            return self.sock_to_node[key]

        node_id = self.node_id_counter
        self.node_id_counter += 1
        self.nodes[node_id] = {
            "type": "socket",
            "attrs": {
                "cip": cip, "cport": cport,
                "sip": sip, "sport": sport,
            }
        }
        self.sock_to_node[key] = node_id
        return node_id

    def _add_edge(self, src_id, dst_id, event):
        """添加一条边（系统调用）"""
        if src_id is None or dst_id is None:
            return
        if src_id == dst_id:
            return

        self.edges.append((src_id, dst_id, {
            "syscall": event.get("evt.type", "unknown"),
            "timestamp": event["_timestamp"],
            "malicious": _parse_malicious(event.get("malicious", False)),
        }))

    def process_event(self, event):
        """处理单条事件，更新溯源图"""
        self.total_events += 1
        if _parse_malicious(event.get("malicious", False)):
            self.malicious_events += 1

        evt_type = event.get("evt.type", "")
        evt_category = event.get("evt.category", "")
        fd_type = event.get("fd.type")

        proc_node = self._get_or_create_process_node(event)
        if proc_node is None:
            return

        self._record_node_stats(proc_node, event)

        # --- 进程创建类 ---
        if evt_type in PROCESS_SYSCALLS:
            child_tid = event.get("evt.arg.pid")
            if child_tid and evt_type in ("clone", "clone3", "fork", "vfork"):
                raw_res = event.get("evt.rawres", -1)
                if isinstance(raw_res, int) and raw_res > 0:
                    child_vtid = raw_res
                    if child_vtid not in self.tid_to_node:
                        child_node_id = self.node_id_counter
                        self.node_id_counter += 1
                        self.nodes[child_node_id] = {
                            "type": "process",
                            "attrs": {
                                "vtid": child_vtid,
                                "pid": child_vtid,
                                "name": child_tid if child_tid else "unknown",
                                "cmdline": "", "exepath": "",
                            }
                        }
                        self.tid_to_node[child_vtid] = child_node_id
                    self._add_edge(proc_node, self.tid_to_node[child_vtid], event)
            elif evt_type in ("execve", "execveat"):
                pass

        # --- 文件操作类 ---
        elif fd_type == "file" or evt_category == "file":
            file_node = self._get_or_create_file_node(event)
            if file_node is not None:
                self._record_node_stats(file_node, event)
                if evt_type in FILE_READ_SYSCALLS:
                    self._add_edge(file_node, proc_node, event)
                elif evt_type in FILE_WRITE_SYSCALLS:
                    self._add_edge(proc_node, file_node, event)
                else:
                    self._add_edge(proc_node, file_node, event)

        # --- 网络操作类 ---
        elif fd_type in ("ipv4", "ipv6") or evt_type in NET_CONNECT_SYSCALLS | NET_ACCEPT_SYSCALLS:
            sock_node = self._get_or_create_socket_node(event)
            if sock_node is not None:
                self._record_node_stats(sock_node, event)
                if evt_type in NET_CONNECT_SYSCALLS:
                    self._add_edge(proc_node, sock_node, event)
                elif evt_type in NET_ACCEPT_SYSCALLS:
                    self._add_edge(sock_node, proc_node, event)
                else:
                    self._add_edge(proc_node, sock_node, event)

    def build(self, events):
        """从事件列表构建完整溯源图"""
        for event in events:
            self.process_event(event)

        logger.info(
            f"  溯源图构建完成: {len(self.nodes)} 个节点, {len(self.edges)} 条边, "
            f"恶意事件 {self.malicious_events}/{self.total_events}"
        )
        return self


# ============================================================
# Step 3: 时间窗口切分 + 标签分配
# ============================================================

def merge_redundant_edges(edges):
    """合并同一对节点间相同类型的重复边"""
    edge_map = defaultdict(lambda: {"count": 0, "malicious": False, "syscalls": set()})
    for src, dst, attr in edges:
        key = (src, dst)
        edge_map[key]["count"] += 1
        edge_map[key]["syscalls"].add(attr["syscall"])
        if attr["malicious"]:
            edge_map[key]["malicious"] = True

    merged = []
    for (src, dst), info in edge_map.items():
        merged.append((src, dst, {
            "syscalls": info["syscalls"],
            "count": info["count"],
            "malicious": info["malicious"],
        }))
    return merged


def slice_by_time_window(events, window_size_sec, window_stride_sec):
    """按时间窗口切分事件流（滑动窗口，O(N)复杂度）"""
    if not events:
        return []

    start_time = events[0]["_timestamp"]
    end_time = events[-1]["_timestamp"]
    window_size = timedelta(seconds=window_size_sec)
    window_stride = timedelta(seconds=window_stride_sec)

    windows = []
    current_start = start_time
    evt_idx = 0

    while current_start < end_time:
        current_end = current_start + window_size
        window_events = []

        while evt_idx < len(events) and events[evt_idx]["_timestamp"] < current_start:
            evt_idx += 1

        scan_idx = evt_idx
        while scan_idx < len(events) and events[scan_idx]["_timestamp"] < current_end:
            window_events.append(events[scan_idx])
            scan_idx += 1

        if window_events:
            windows.append({
                "start": current_start,
                "end": current_end,
                "events": window_events,
            })

        current_start += window_stride

    logger.info(f"  时间窗口切分: {len(windows)} 个窗口 (窗口大小={window_size_sec}s, 步长={window_stride_sec}s)")
    return windows


def build_window_graph(window_events):
    """为单个时间窗口构建溯源图，同时收集节点级统计信息"""
    builder = ProvenanceGraphBuilder()
    builder.build(window_events)

    has_malicious = builder.malicious_events > 0

    return {
        "nodes": builder.nodes,
        "edges": builder.edges,
        "has_malicious": has_malicious,
        "malicious_ratio": builder.malicious_events / max(builder.total_events, 1),
        "total_events": builder.total_events,
        "node_syscall_counts": dict(builder.node_syscall_counts),
        "node_event_count": dict(builder.node_event_count),
        "node_timestamps": dict(builder.node_timestamps),
    }


# ============================================================
# Step 4: 三层融合节点特征工程
# 参考 KAIROS (层次化特征哈希) + FLASH (系统调用语义分布)
# ============================================================

NUM_SYSCALL_TYPES = len(SYSCALL_TYPES) + 1  # +1 for unknown

# --- 语义层 ---

def hierarchical_path_hash(path_str, dim=16):
    """层次化路径哈希（参考KAIROS的hierarchical feature hashing）

    对路径按分隔符拆分成多级子串，分别哈希后叠加，
    保留层次语义：同目录的文件向量更近。
    """
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


def get_node_semantic_string(node):
    """获取节点的语义标识字符串"""
    node_type = node["type"]
    attrs = node["attrs"]
    if node_type == "process":
        return attrs.get("exepath", "") or attrs.get("name", "")
    elif node_type == "file":
        return attrs.get("name", "")
    else:
        return f"{attrs.get('cip', '')}:{attrs.get('cport', '')}-{attrs.get('sip', '')}:{attrs.get('sport', '')}"


SEMANTIC_DIM = 3 + 16  # type one-hot (3) + hierarchical path hash (16) = 19

# --- 行为层 ---

BEHAVIOR_DIM = NUM_SYSCALL_TYPES + 3  # syscall freq distribution + in_degree + out_degree + event_count

# --- 时序层 ---

TEMPORAL_DIM = 4  # time_first_norm, time_last_norm, interval_mean, interval_std

# --- 总维度 ---

NODE_FEATURE_DIM = SEMANTIC_DIM + BEHAVIOR_DIM + TEMPORAL_DIM


def build_node_features(nodes, graph_info):
    """构建三层融合节点特征矩阵

    Layer 1 - 语义层 (19维, 参考KAIROS):
      [0:3]   实体类型 one-hot (process/file/socket)
      [3:19]  层次化路径哈希 (对进程exepath/文件path/IP按层级哈希叠加)

    Layer 2 - 行为层 (NUM_SYSCALL_TYPES+3 维, 参考FLASH):
      [19:19+N]  系统调用频率分布 (各syscall在窗口内的归一化频率)
      [+N:+N+1]  入度
      [+N+1:+N+2] 出度
      [+N+2:+N+3] 窗口内事件计数(归一化)

    Layer 3 - 时序层 (4维):
      首次出现时间归一化、末次出现时间归一化、事件间隔均值、事件间隔标准差
    """
    num_nodes = len(nodes)
    if num_nodes == 0:
        return np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32)

    features = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)

    node_syscall_counts = graph_info.get("node_syscall_counts", {})
    node_event_count = graph_info.get("node_event_count", {})
    node_timestamps = graph_info.get("node_timestamps", {})
    edges = graph_info.get("edges", [])

    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    for src, dst, _ in edges:
        out_degree[src] += 1
        in_degree[dst] += 1

    max_events = max(node_event_count.values()) if node_event_count else 1.0
    if max_events == 0:
        max_events = 1.0

    all_timestamps = []
    for ts_list in node_timestamps.values():
        all_timestamps.extend(ts_list)
    if all_timestamps:
        global_min_ts = min(all_timestamps)
        global_max_ts = max(all_timestamps)
        time_range = (global_max_ts - global_min_ts).total_seconds()
    else:
        global_min_ts = None
        time_range = 1.0
    if time_range <= 0:
        time_range = 1.0

    for node_id in range(num_nodes):
        if node_id not in nodes:
            continue
        node = nodes[node_id]
        offset = 0

        # === 语义层 (19维) ===
        type_idx = ENTITY_TYPE_MAP.get(node["type"], 0)
        features[node_id, type_idx] = 1.0
        offset = 3

        semantic_str = get_node_semantic_string(node)
        features[node_id, offset:offset + 16] = hierarchical_path_hash(semantic_str, 16)
        offset += 16

        # === 行为层 (NUM_SYSCALL_TYPES+3 维) ===
        syscall_counts = node_syscall_counts.get(node_id, {})
        total_sc = sum(syscall_counts.values())
        if total_sc > 0:
            for sc_name, count in syscall_counts.items():
                sc_idx = SYSCALL_TYPE_MAP.get(sc_name, UNKNOWN_SYSCALL_IDX)
                features[node_id, offset + sc_idx] = count / total_sc
        offset += NUM_SYSCALL_TYPES

        features[node_id, offset] = in_degree.get(node_id, 0)
        features[node_id, offset + 1] = out_degree.get(node_id, 0)
        features[node_id, offset + 2] = node_event_count.get(node_id, 0) / max_events
        offset += 3

        # === 时序层 (4维) ===
        ts_list = node_timestamps.get(node_id, [])
        if ts_list and global_min_ts is not None:
            first_ts = min(ts_list)
            last_ts = max(ts_list)
            features[node_id, offset] = (first_ts - global_min_ts).total_seconds() / time_range
            features[node_id, offset + 1] = (last_ts - global_min_ts).total_seconds() / time_range

            if len(ts_list) > 1:
                sorted_ts = sorted(ts_list)
                intervals = [(sorted_ts[i + 1] - sorted_ts[i]).total_seconds()
                             for i in range(len(sorted_ts) - 1)]
                features[node_id, offset + 2] = np.mean(intervals)
                features[node_id, offset + 3] = np.std(intervals)

    degree_max = features[:, SEMANTIC_DIM:SEMANTIC_DIM + NUM_SYSCALL_TYPES + 2].max(axis=0)
    for col_offset in range(2):
        col = SEMANTIC_DIM + NUM_SYSCALL_TYPES + col_offset
        if degree_max[NUM_SYSCALL_TYPES + col_offset] > 0:
            features[:, col] /= degree_max[NUM_SYSCALL_TYPES + col_offset]

    interval_col = SEMANTIC_DIM + BEHAVIOR_DIM + 2
    max_interval = features[:, interval_col].max()
    if max_interval > 0:
        features[:, interval_col] /= max_interval
        features[:, interval_col + 1] /= max_interval

    return features


def build_edge_features(edges):
    """构建边特征

    特征组成:
    - syscall type multi-hot: len(SYSCALL_TYPES)+1 维
    - log(count+1) 归一化: 1 维（反映合并前的原始边数量，即交互频率）
    """
    feature_dim = len(SYSCALL_TYPES) + 1 + 1
    num_edges = len(edges)

    if num_edges == 0:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, feature_dim), dtype=np.float32))

    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_features = np.zeros((num_edges, feature_dim), dtype=np.float32)

    for i, (src, dst, attr) in enumerate(edges):
        edge_index[0, i] = src
        edge_index[1, i] = dst

        syscall = attr.get("syscall", "")
        if isinstance(attr.get("syscalls"), set):
            for sc in attr["syscalls"]:
                idx = SYSCALL_TYPE_MAP.get(sc, UNKNOWN_SYSCALL_IDX)
                edge_features[i, idx] = 1.0
        else:
            idx = SYSCALL_TYPE_MAP.get(syscall, UNKNOWN_SYSCALL_IDX)
            edge_features[i, idx] = 1.0

        count = attr.get("count", 1)
        edge_features[i, -1] = np.log1p(count)

    max_log_count = edge_features[:, -1].max()
    if max_log_count > 0:
        edge_features[:, -1] /= max_log_count

    return edge_index, edge_features


# ============================================================
# Step 5: 转换为PyG Data
# ============================================================

def graph_to_pyg_data(graph_info, label, scenario_name="", attack_type="",
                      window_start=None, window_end=None,
                      runtime="", app=""):
    """将溯源图转换为PyG Data对象"""
    nodes = graph_info["nodes"]
    edges = graph_info["edges"]

    edges = merge_redundant_edges(edges)

    connected_nodes = set()
    for src, dst, _ in edges:
        connected_nodes.add(src)
        connected_nodes.add(dst)

    old_to_new = {}
    new_nodes = {}
    idx = 0
    for old_id in sorted(nodes.keys()):
        if old_id in connected_nodes:
            old_to_new[old_id] = idx
            new_nodes[idx] = nodes[old_id]
            idx += 1

    remapped_edges = []
    for src, dst, attr in edges:
        if src in old_to_new and dst in old_to_new:
            remapped_edges.append((old_to_new[src], old_to_new[dst], attr))

    remapped_graph_info = {
        "edges": [(src, dst, attr) for src, dst, attr in graph_info["edges"]],
        "node_syscall_counts": {},
        "node_event_count": {},
        "node_timestamps": {},
    }
    for old_id, new_id in old_to_new.items():
        if old_id in graph_info.get("node_syscall_counts", {}):
            remapped_graph_info["node_syscall_counts"][new_id] = graph_info["node_syscall_counts"][old_id]
        if old_id in graph_info.get("node_event_count", {}):
            remapped_graph_info["node_event_count"][new_id] = graph_info["node_event_count"][old_id]
        if old_id in graph_info.get("node_timestamps", {}):
            remapped_graph_info["node_timestamps"][new_id] = graph_info["node_timestamps"][old_id]

    remapped_graph_info["edges"] = remapped_edges

    node_features = build_node_features(new_nodes, remapped_graph_info)
    edge_index, edge_features = build_edge_features(remapped_edges)

    num_nodes = len(new_nodes)
    if num_nodes == 0:
        return None

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=num_nodes,
    )

    data.scenario = scenario_name
    data.attack_type = attack_type
    data.has_malicious = graph_info["has_malicious"]
    data.malicious_ratio = graph_info["malicious_ratio"]
    data.total_events = graph_info["total_events"]
    data.runtime = runtime
    data.app = app

    if window_start:
        data.window_start = str(window_start)
    if window_end:
        data.window_end = str(window_end)

    return data


# ============================================================
# Step 6: 数据集统计与验证
# ============================================================

def print_dataset_stats(data_list, scenario_name):
    """打印数据集统计信息"""
    if not data_list:
        logger.warning(f"[{scenario_name}] 无有效图数据")
        return

    num_graphs = len(data_list)
    labels = [d.y.item() for d in data_list]
    num_normal = labels.count(0)
    num_attack = num_graphs - num_normal

    node_counts = [d.num_nodes for d in data_list]
    edge_counts = [d.edge_index.shape[1] for d in data_list]

    node_types = defaultdict(int)
    for d in data_list:
        for i in range(d.num_nodes):
            type_vec = d.x[i, :3]
            type_idx = type_vec.argmax().item()
            node_types[ENTITY_TYPES[type_idx]] += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"数据集统计: {scenario_name}")
    logger.info(f"{'='*60}")
    logger.info(f"  图总数:     {num_graphs}")
    logger.info(f"  正常图:     {num_normal}")
    logger.info(f"  攻击图:     {num_attack}")
    logger.info(f"  平均节点数: {np.mean(node_counts):.1f} ± {np.std(node_counts):.1f}")
    logger.info(f"  平均边数:   {np.mean(edge_counts):.1f} ± {np.std(edge_counts):.1f}")
    logger.info(f"  最小节点数: {min(node_counts)}, 最大: {max(node_counts)}")
    logger.info(f"  最小边数:   {min(edge_counts)}, 最大: {max(edge_counts)}")
    logger.info(f"  节点类型分布:")
    for t, c in sorted(node_types.items()):
        logger.info(f"    {t}: {c}")
    logger.info(f"{'='*60}\n")


# ============================================================
# 主流程
# ============================================================

ATTACK_TYPE_TO_LABEL = {
    "Normal": 0,      # N: 正常行为
    "RCE": 1,         # 远程代码执行 (fork/exec类进程派生)
    "Non-RCE": 2,     # 非RCE攻击 (SQLi/SSRF/路径遍历/信息泄露/XXE等)
}


def process_single_log(log_path, scenario_name, attack_type,
                       window_size, window_stride, min_nodes=5,
                       runtime="", app=""):
    """处理单个日志文件，返回PyG Data列表"""
    logger.info(f"\n处理日志: {log_path}")
    logger.info(f"  场景: {scenario_name}, 攻击类型: {attack_type}")

    events = load_log_file(log_path)
    if not events:
        logger.warning(f"  日志为空，跳过")
        return []

    windows = slice_by_time_window(events, window_size, window_stride)

    attack_label = ATTACK_TYPE_TO_LABEL.get(attack_type, 1)
    data_list = []

    for i, window in enumerate(windows):
        graph_info = build_window_graph(window["events"])

        if graph_info["has_malicious"]:
            label = attack_label
        else:
            label = 0

        data = graph_to_pyg_data(
            graph_info, label,
            scenario_name=scenario_name,
            attack_type=attack_type if label > 0 else "Normal",
            window_start=window["start"],
            window_end=window["end"],
            runtime=runtime,
            app=app,
        )

        if data is not None and data.num_nodes >= min_nodes:
            data_list.append(data)

    logger.info(f"  生成 {len(data_list)} 张有效溯源图 (特征维度: {NODE_FEATURE_DIM})")
    return data_list


def process_scenario_dir(input_dir, scenario_name, attack_type,
                         window_size, window_stride, min_nodes=5,
                         runtime="", app=""):
    """处理单个场景目录下的所有日志文件（20次重复实验）"""
    logger.info(f"\n{'#'*60}")
    logger.info(f"处理场景: {scenario_name} ({attack_type})")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"{'#'*60}")

    patterns = ["*.jsonl", "*.json", "*.log"]
    log_files = []
    for pat in patterns:
        log_files.extend(glob.glob(os.path.join(input_dir, "**", pat), recursive=True))

    if not log_files:
        log_files = [f for f in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
                     if os.path.isfile(f) and not f.endswith((".py", ".sh", ".md", ".yml", ".yaml"))]

    log_files = sorted(log_files)
    logger.info(f"找到 {len(log_files)} 个日志文件")

    all_data = []
    for log_path in log_files:
        data_list = process_single_log(
            log_path, scenario_name, attack_type,
            window_size, window_stride, min_nodes,
            runtime=runtime, app=app,
        )
        all_data.extend(data_list)

    return all_data


# ============================================================
# 批量处理所有场景
# ============================================================

SCENARIO_CONFIG = {
    # ================================================================
    # D_solr: Apache Solr (Java/JVM) — 4个CVE
    # ================================================================
    "cve-2019-17558": {"app": "Solr", "runtime": "java", "attack_type": "RCE"},
    "cve-2019-0193":  {"app": "Solr", "runtime": "java", "attack_type": "RCE"},
    "cve-2017-12629-rce": {"app": "Solr", "runtime": "java", "attack_type": "RCE"},
    "cve-2017-12629-xxe": {"app": "Solr", "runtime": "java", "attack_type": "Non-RCE"},

    # ================================================================
    # D_ofbiz: Apache OFBiz (Java/JVM) — 8个CVE
    # ================================================================
    "cve-2020-9496":  {"app": "OFBiz", "runtime": "java", "attack_type": "RCE"},
    "cve-2023-49070": {"app": "OFBiz", "runtime": "java", "attack_type": "RCE"},
    "cve-2023-51467": {"app": "OFBiz", "runtime": "java", "attack_type": "RCE"},
    "cve-2024-38856": {"app": "OFBiz", "runtime": "java", "attack_type": "RCE"},
    "cve-2024-45507": {"app": "OFBiz", "runtime": "java", "attack_type": "RCE"},
    "cve-2024-45195": {"app": "OFBiz", "runtime": "java", "attack_type": "RCE"},
    "cve-2023-50968": {"app": "OFBiz", "runtime": "java", "attack_type": "Non-RCE"},  # 自建: SSRF+文件读取
    "cve-2024-23946": {"app": "OFBiz", "runtime": "java", "attack_type": "Non-RCE"},  # 自建: 路径遍历

    # ================================================================
    # D_geo: GeoServer (Java/JVM) — 3个CVE
    # ================================================================
    "cve-2024-36401": {"app": "GeoServer", "runtime": "java", "attack_type": "RCE"},
    "cve-2023-25157": {"app": "GeoServer", "runtime": "java", "attack_type": "Non-RCE"},
    "cve-2021-40822": {"app": "GeoServer", "runtime": "java", "attack_type": "Non-RCE"},  # 自建: SSRF

    # ================================================================
    # D_tomcat: Apache Tomcat (Java/JVM) — 3个CVE (全部自建)
    # ================================================================
    "cve-2025-24813": {"app": "Tomcat", "runtime": "java", "attack_type": "RCE"},
    "cve-2020-9484":  {"app": "Tomcat", "runtime": "java", "attack_type": "RCE"},
    "cve-2020-1938":  {"app": "Tomcat", "runtime": "java", "attack_type": "Non-RCE"},  # Ghostcat文件读取

    # ================================================================
    # D_metabase: Metabase (Java/JVM) — 2个CVE
    # ================================================================
    "cve-2023-38646": {"app": "Metabase", "runtime": "java", "attack_type": "RCE"},
    "cve-2021-41277": {"app": "Metabase", "runtime": "java", "attack_type": "Non-RCE"},

    # ================================================================
    # D_joomla: Joomla (PHP) — 3个CVE
    # ================================================================
    "cve-2015-8562":  {"app": "Joomla", "runtime": "php", "attack_type": "RCE"},
    "cve-2017-8917":  {"app": "Joomla", "runtime": "php", "attack_type": "Non-RCE"},
    "cve-2023-23752": {"app": "Joomla", "runtime": "php", "attack_type": "Non-RCE"},

    # ================================================================
    # D_pgadmin: pgAdmin (Python) — 3个CVE
    # ================================================================
    "cve-2022-4223":  {"app": "pgAdmin", "runtime": "python", "attack_type": "RCE"},
    "cve-2023-5002":  {"app": "pgAdmin", "runtime": "python", "attack_type": "RCE"},
    "cve-2024-9014":  {"app": "pgAdmin", "runtime": "python", "attack_type": "Non-RCE"},  # 自建: 信息泄露

    # ================================================================
    # 非主实验场景（补充/扩展实验可用）
    # ================================================================
    "cve-2021-26120":   {"app": "CMSMS",        "runtime": "php",    "attack_type": "RCE"},
    "cve-2019-9053":    {"app": "CMSMS",        "runtime": "php",    "attack_type": "Non-RCE"},
    "cve-2021-43008":   {"app": "Adminer",      "runtime": "php",    "attack_type": "Non-RCE"},
    "cve-2021-21311":   {"app": "Adminer",      "runtime": "php",    "attack_type": "Non-RCE"},  # 自建: SSRF
    "cve-2018-1000533": {"app": "Gitlist",      "runtime": "php",    "attack_type": "RCE"},
    "cve-2021-34429":   {"app": "Jetty",        "runtime": "java",   "attack_type": "Non-RCE"},  # 自建: 信息泄露
    "cve-2019-14234":   {"app": "Django",       "runtime": "python", "attack_type": "Non-RCE"},  # 自建: SQLi
    "cve-2022-34265":   {"app": "Django",       "runtime": "python", "attack_type": "Non-RCE"},  # 自建: SQLi
    "python-demo":      {"app": "PythonDemo",   "runtime": "python", "attack_type": "Non-RCE"},
    "owasp-juice-shop": {"app": "JuiceShop",    "runtime": "nodejs", "attack_type": "RCE"},
    "cve-2019-10758":   {"app": "MongoExpress", "runtime": "nodejs", "attack_type": "RCE"},
    "cve-2018-17246":   {"app": "Kibana",       "runtime": "nodejs", "attack_type": "RCE"},
    "sandworm":         {"app": "Sandworm",     "runtime": "linux",  "attack_type": "RCE"},
    "10-step-chained":  {"app": "Chained",      "runtime": "mixed",  "attack_type": "RCE"},
}


def batch_process(base_dir, output_dir, window_size=300, window_stride=150, min_nodes=5):
    """批量处理所有场景"""
    os.makedirs(output_dir, exist_ok=True)

    all_data_by_runtime = defaultdict(list)
    all_data_by_app = defaultdict(list)
    all_data_total = []

    scenario_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))]

    logger.info(f"在 {base_dir} 下发现 {len(scenario_dirs)} 个场景目录")

    for scenario_dir_name in sorted(scenario_dirs):
        scenario_key = scenario_dir_name.lower().replace(" ", "-")

        config = None
        for key, cfg in SCENARIO_CONFIG.items():
            if key in scenario_key or scenario_key in key:
                config = cfg
                scenario_key = key
                break

        if config is None:
            logger.warning(f"未识别的场景目录: {scenario_dir_name}, 跳过")
            continue

        input_path = os.path.join(base_dir, scenario_dir_name)
        data_list = process_scenario_dir(
            input_path, scenario_key, config["attack_type"],
            window_size, window_stride, min_nodes,
            runtime=config["runtime"], app=config["app"],
        )

        all_data_by_runtime[config["runtime"]].extend(data_list)
        all_data_by_app[config["app"]].extend(data_list)
        all_data_total.extend(data_list)

        # 保存单场景数据
        if data_list:
            scene_path = os.path.join(output_dir, f"scene_{scenario_key}.pt")
            torch.save(data_list, scene_path)
            logger.info(f"  保存到: {scene_path}")

    # 保存按运行时分组的数据
    for runtime, data in all_data_by_runtime.items():
        if data:
            path = os.path.join(output_dir, f"runtime_{runtime}.pt")
            torch.save(data, path)
            print_dataset_stats(data, f"Runtime: {runtime}")

    # 保存按应用分组的数据
    for app, data in all_data_by_app.items():
        if data:
            path = os.path.join(output_dir, f"app_{app}.pt")
            torch.save(data, path)

    # 保存全部数据
    if all_data_total:
        path = os.path.join(output_dir, "all_graphs.pt")
        torch.save(all_data_total, path)
        print_dataset_stats(all_data_total, "ALL")

    # 输出汇总
    logger.info(f"\n{'='*60}")
    logger.info("批量处理完成汇总")
    logger.info(f"{'='*60}")
    logger.info(f"总图数: {len(all_data_total)}")
    for runtime, data in sorted(all_data_by_runtime.items()):
        labels = [d.y.item() for d in data]
        logger.info(f"  {runtime}: {len(data)} 张图 (正常: {labels.count(0)}, 攻击: {len(data)-labels.count(0)})")
    logger.info(f"输出目录: {output_dir}")

    return all_data_total


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AUTOLABEL Sysdig日志 → 溯源图 构建脚本")

    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # 模式1: 处理单个日志文件
    single_parser = subparsers.add_parser("single", help="处理单个日志文件")
    single_parser.add_argument("--input", required=True, help="输入日志文件路径")
    single_parser.add_argument("--output_dir", default="./processed_graphs", help="输出目录")
    single_parser.add_argument("--scenario", default="unknown", help="场景名称")
    single_parser.add_argument("--attack_type", default="RCE", help="攻击类型")
    single_parser.add_argument("--window_size", type=int, default=300, help="时间窗口大小(秒)")
    single_parser.add_argument("--window_stride", type=int, default=150, help="时间窗口步长(秒)")
    single_parser.add_argument("--min_nodes", type=int, default=5, help="最小节点数阈值")

    # 模式2: 处理单个场景目录
    scene_parser = subparsers.add_parser("scenario", help="处理单个场景目录")
    scene_parser.add_argument("--input_dir", required=True, help="场景目录路径")
    scene_parser.add_argument("--output_dir", default="./processed_graphs", help="输出目录")
    scene_parser.add_argument("--scenario", default="unknown", help="场景名称")
    scene_parser.add_argument("--attack_type", default="RCE", help="攻击类型")
    scene_parser.add_argument("--window_size", type=int, default=300, help="时间窗口大小(秒)")
    scene_parser.add_argument("--window_stride", type=int, default=150, help="时间窗口步长(秒)")
    scene_parser.add_argument("--min_nodes", type=int, default=5, help="最小节点数阈值")

    # 模式3: 批量处理所有场景
    batch_parser = subparsers.add_parser("batch", help="批量处理所有场景")
    batch_parser.add_argument("--base_dir", required=True, help="AUTOLABEL数据根目录")
    batch_parser.add_argument("--output_dir", default="./processed_graphs", help="输出目录")
    batch_parser.add_argument("--window_size", type=int, default=300, help="时间窗口大小(秒)")
    batch_parser.add_argument("--window_stride", type=int, default=150, help="时间窗口步长(秒)")
    batch_parser.add_argument("--min_nodes", type=int, default=5, help="最小节点数阈值")

    args = parser.parse_args()

    if args.mode == "single":
        os.makedirs(args.output_dir, exist_ok=True)
        scenario_key = args.scenario.lower()
        cfg = SCENARIO_CONFIG.get(scenario_key, {})
        data_list = process_single_log(
            args.input, args.scenario, args.attack_type,
            args.window_size, args.window_stride, args.min_nodes,
            runtime=cfg.get("runtime", ""), app=cfg.get("app", ""),
        )
        if data_list:
            output_path = os.path.join(args.output_dir, f"scene_{args.scenario}.pt")
            torch.save(data_list, output_path)
            print_dataset_stats(data_list, args.scenario)
            logger.info(f"保存到: {output_path}")

    elif args.mode == "scenario":
        os.makedirs(args.output_dir, exist_ok=True)
        scenario_key = args.scenario.lower()
        cfg = SCENARIO_CONFIG.get(scenario_key, {})
        data_list = process_scenario_dir(
            args.input_dir, args.scenario, args.attack_type,
            args.window_size, args.window_stride, args.min_nodes,
            runtime=cfg.get("runtime", ""), app=cfg.get("app", ""),
        )
        if data_list:
            output_path = os.path.join(args.output_dir, f"scene_{args.scenario}.pt")
            torch.save(data_list, output_path)
            print_dataset_stats(data_list, args.scenario)
            logger.info(f"保存到: {output_path}")

    elif args.mode == "batch":
        batch_process(
            args.base_dir, args.output_dir,
            args.window_size, args.window_stride, args.min_nodes
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
