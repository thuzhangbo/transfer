"""
GNN编码器 + 图级分类器

支持4种GNN骨干架构: GIN, GraphSAGE, GAT, GCN
用于溯源图的图级异常检测任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv, SAGEConv, GATConv, GCNConv,
    global_mean_pool, global_add_pool,
)


# ============================================================
# GNN骨干网络
# ============================================================

class GINEncoder(nn.Module):
    """Graph Isomorphism Network"""

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
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


class SAGEEncoder(nn.Module):
    """GraphSAGE"""

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network"""

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if i < num_layers - 1:
                self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            else:
                self.convs.append(GATConv(in_dim, hidden_dim, heads=1, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNEncoder(nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


ENCODER_REGISTRY = {
    "gin": GINEncoder,
    "sage": SAGEEncoder,
    "gat": GATEncoder,
    "gcn": GCNEncoder,
}


# ============================================================
# 图级分类模型
# ============================================================

class GraphClassifier(nn.Module):
    """GNN编码器 + 读出 + 分类器"""

    def __init__(self, input_dim, hidden_dim, num_classes, encoder_name="gin",
                 num_layers=3, dropout=0.3, pooling="mean"):
        super().__init__()

        encoder_cls = ENCODER_REGISTRY[encoder_name]
        self.encoder = encoder_cls(input_dim, hidden_dim, num_layers, dropout)
        self.pooling = pooling

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def get_graph_embedding(self, x, edge_index, batch):
        """获取图级嵌入向量（分类器之前）"""
        node_emb = self.encoder(x, edge_index, batch)

        if self.pooling == "mean":
            graph_emb = global_mean_pool(node_emb, batch)
        elif self.pooling == "sum":
            graph_emb = global_add_pool(node_emb, batch)
        else:
            graph_emb = global_mean_pool(node_emb, batch)

        return graph_emb

    def forward(self, x, edge_index, batch):
        graph_emb = self.get_graph_embedding(x, edge_index, batch)
        logits = self.classifier(graph_emb)
        return logits

    def forward_with_embedding(self, x, edge_index, batch):
        """同时返回嵌入和logits（PCKD适应阶段使用）"""
        graph_emb = self.get_graph_embedding(x, edge_index, batch)
        logits = self.classifier(graph_emb)
        return graph_emb, logits
