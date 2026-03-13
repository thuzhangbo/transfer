# PCKD 实验工具链

溯源图域适应实验 + t-SNE 可视化

## 快速开始：画 t-SNE 图

```bash
# 1. 安装依赖
pip install torch torch_geometric numpy scikit-learn matplotlib

# 2. 在服务器上运行（指向 autolabel 输出目录）
python pipeline_extract_embeddings.py --data_dir /path/to/solr_tar_gz --output embeddings_for_tsne.npz --epochs 300 --device cuda

# 3. 下载 .npz 到本地，画图
python plot_tsne.py --input embeddings_for_tsne.npz --output tsne.pdf
```

## 完整实验流程

```bash
# Step 1: autolabel 日志 → 溯源图
python build_provenance_graph.py batch --base_dir /path/to/autolabel_data --output_dir ./processed_graphs

# Step 2: 按任务划分数据集
python organize_dataset.py --data_dir ./processed_graphs --output_dir ./experiments

# Step 3: 训练 + 适应 + 评估
python run_experiment.py --exp_dir ./experiments --tasks T1 --encoders gin --seeds 42 123 456
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `pipeline_extract_embeddings.py` | 一体化脚本：日志→图→训练→embedding |
| `plot_tsne.py` | t-SNE 可视化（本地运行） |
| `build_provenance_graph.py` | autolabel 日志→溯源图→PyG |
| `organize_dataset.py` | 按 T1-T10 任务划分数据集 |
| `train_source.py` | 源域 GNN 训练 |
| `run_experiment.py` | 完整实验流程 |
| `models/gnn_encoder.py` | GIN/SAGE/GAT/GCN 编码器 |
| `models/pckd.py` | PCKD 适应框架 |
