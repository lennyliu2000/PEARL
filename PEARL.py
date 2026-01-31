import os
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# -------------------- 模型定义 --------------------
class FeatureEncoder(torch.nn.Module):
    """跨模态注意力特征编码器"""
    def __init__(self, in_dim=256, hidden_dim=512):
        super().__init__()
        self.mod1_proj = Linear(128, hidden_dim//2)
        self.mod2_proj = Linear(128, hidden_dim//2)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim//2,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        self.fusion = torch.nn.Sequential(
            Linear(768, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.LeakyReLU(0.1),
            Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        mod1 = self.mod1_proj(x[:, :128].unsqueeze(1))
        mod2 = self.mod2_proj(x[:, 128:].unsqueeze(1))
        attn_out1, _ = self.attn(mod1, mod2, mod2)
        attn_out2, _ = self.attn(mod2, mod1, mod1)
        f1 = mod1 + attn_out1
        f2 = mod2 + attn_out2
        gate = torch.sigmoid(f1 + f2)  # 门控机制
        combined = gate * f1 + (1 - gate) * f2
        combined = torch.cat([f1, f2, combined], dim=-1).squeeze(1)
        return self.fusion(combined)


class HeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_heads=4, out_dim=128):
        super().__init__()
        self.pep_encoder = FeatureEncoder(in_dim=256, hidden_dim=hidden_channels * num_heads)
        self.target_encoder = FeatureEncoder(in_dim=256, hidden_dim=hidden_channels * num_heads)

        metadata = (['peptide', 'target'],
                    [('peptide', 'similar_to', 'peptide'),
                     ('peptide', 'interacts_with', 'target')])

        self.convs = torch.nn.ModuleList([
            HGTConv(
                in_channels=hidden_channels * num_heads,
                out_channels=hidden_channels * num_heads,
                metadata=metadata,
                heads=num_heads,
                group="multi-head"
            ) for _ in range(2)
        ])

        self.pep_out = Linear(hidden_channels * num_heads, out_dim)
        self.target_out = Linear(hidden_channels * num_heads, out_dim)
        self.activation = torch.nn.LeakyReLU(0.2)

    def forward(self, x_dict, edge_index_dict):
        x_dict['peptide'] = self.pep_encoder(x_dict['peptide'])
        x_dict['target'] = self.target_encoder(x_dict['target'])

        pep_res = x_dict['peptide'].clone()
        target_res = x_dict['target'].clone()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict['peptide'] = self.activation(x_dict['peptide'] + pep_res)
            x_dict['target'] = self.activation(x_dict['target'] + target_res)

        pep = self.pep_out(x_dict['peptide'])
        target = self.target_out(x_dict['target'])

        # cross interaction：基于共注意力交互
        interaction = torch.matmul(pep, target.T)  # [n_pep, n_tar]
        attn_pep = torch.matmul(interaction.softmax(dim=-1), target)
        attn_tar = torch.matmul(interaction.softmax(dim=0).T, pep)

        pep = pep + attn_pep
        target = target + attn_tar

        return pep, target


class BioInteractionPredictor(torch.nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(in_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.2),
            Linear(256, 1)
        )
        # 初始化
        for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, pep_emb, target_emb, edge_index):
        src, dst = edge_index
        p = pep_emb[src]      # [batch, dim]
        t = target_emb[dst]   # [batch, dim]

        cross = p * t
        diff = torch.abs(p - t)
        diff2 = (p - t) ** 2

        # 构建 [batch, 5, dim] 的融合张量
        fusion = torch.stack([p, t, cross, diff, diff2], dim=1)

        # 计算 attention 权重：对 dim 求和，softmax 出权重 [batch, 5, 1]
        attn_weight = torch.softmax(torch.sum(fusion, dim=2), dim=1).unsqueeze(-1)

        # 加权求和后维度为 [batch, dim]
        fused = torch.sum(fusion * attn_weight, dim=1)

        return self.mlp(fused).squeeze()


class EndToEndModel(torch.nn.Module):
    def __init__(self, hidden_channels=128, num_heads=4, out_dim=128):
        super().__init__()
        self.encoder = HeteroGAT(hidden_channels, num_heads, out_dim)
        self.predictor = BioInteractionPredictor(in_dim=128)

    def forward(self, x_dict, edge_index_dict, edges=None):
        pep_emb, target_emb = self.encoder(x_dict, edge_index_dict)
        if edges is None:  # 用于生成所有节点embedding
            return pep_emb, target_emb
        return self.predictor(pep_emb, target_emb, edges), pep_emb, target_emb


# -------------------- 数据处理 --------------------
def load_features(pep_graph_file, pep_evo_file, target_disorder_file, target_evo_file, device):
    """加载所有特征并按名称对齐"""
    # 加载多肽特征
    pep_graph = torch.load(pep_graph_file, map_location=device)
    pep_evo = torch.load(pep_evo_file, map_location=device)
    pep_names = sorted(pep_graph.keys())
    pep_feat = [torch.cat([pep_graph[name], pep_evo[name]], dim=-1) for name in pep_names]
    pep_feat = torch.stack(pep_feat, dim=0)

    # 加载靶点特征
    target_disorder = torch.load(target_disorder_file, map_location=device)
    target_evo = torch.load(target_evo_file, map_location=device)
    target_names = sorted(target_disorder.keys())
    target_feat = [torch.cat([target_disorder[name], target_evo[name]], dim=-1) for name in target_names]
    target_feat = torch.stack(target_feat, dim=0)

    # 特征归一化
    pep_feat = F.normalize(pep_feat, p=2, dim=1)
    target_feat = F.normalize(target_feat, p=2, dim=1)

    return pep_feat, target_feat


def build_hetero_graph(DD, DT, peptide_features, target_features, device):
    data = HeteroData()
    pt_src, pt_dst = torch.where(torch.tensor(DT, device=device) == 1)
    data['peptide', 'interacts_with', 'target'].edge_index = torch.stack([pt_src, pt_dst], dim=0)

    pp_src, pp_dst = torch.nonzero(torch.tensor(DD, device=device), as_tuple=True)
    data['peptide', 'similar_to', 'peptide'].edge_index = torch.stack([pp_src, pp_dst], dim=0)

    data['peptide'].x = peptide_features.to(device)
    data['target'].x = target_features.to(device)
    return data.to(device)


class EdgeDataset(Dataset):
    def __init__(self, edge_index, labels):
        self.src = edge_index[0]
        self.dst = edge_index[1]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.src[idx], self.dst[idx], self.labels[idx]


# -------------------- 训练与验证 --------------------
def split_train_test_from_matrix(DT, device, test_ratio=0.2, seed=42):
    pos_pairs = np.argwhere(DT == 1)
    neg_pairs = np.argwhere(DT == 0)

    pos_train, pos_test = train_test_split(pos_pairs, test_size=test_ratio, random_state=seed)
    neg_train, neg_test = train_test_split(neg_pairs, test_size=test_ratio, random_state=seed)

    train_edges = np.vstack([pos_train, neg_train])
    test_edges = np.vstack([pos_test, neg_test])
    train_labels = np.hstack([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    test_labels = np.hstack([np.ones(len(pos_test)), np.zeros(len(neg_test))])

    # 转 tensor
    train_edges = torch.tensor(train_edges.T, dtype=torch.long).to(device)
    test_edges = torch.tensor(test_edges.T, dtype=torch.long).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.float).to(device)

    return train_edges, train_labels, test_edges, test_labels

def train(model, train_loader, optimizer, criterion, scheduler, device, clip_norm=2.0, debug_print_every=10):
    """
    训练一个 epoch（LinkNeighborLoader 风格的 batch）
    model: DDP-wrapped model (or module)
    train_loader: LinkNeighborLoader that yields batches with edge_label_index_dict & edge_label_dict
    optimizer, criterion, scheduler: optimizer / loss / lr scheduler
    device: torch.device
    clip_norm: max norm for gradient clipping
    debug_print_every: how often (batch) to print batch-level debug
    Returns: avg_loss, auc, aupr, mcc
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    epoch_logits = []
    epoch_probs = []

    edge_type = ('peptide', 'interacts_with', 'target')

    accum_steps = 2
    for batch_idx, batch in enumerate(train_loader):
        # 将batch移至device
        batch = batch.to(device)

        # 从异构图batch中获取边索引和标签
        try:
            edge_index = batch.edge_label_index_dict[edge_type]  # shape [2, B]
            labels = batch.edge_label_dict[edge_type]  # shape [B]
        except Exception as e:
            print(f"[Train] Batch {batch_idx}: missing edge_label_index/edge_label for {edge_type}: {e}")
            continue

        # 前向传播
        preds, _, _ = model(batch.x_dict, batch.edge_index_dict, edge_index)
        preds = preds.view(-1)  # ensure shape [B]

        # numerical checks: finite
        if not torch.isfinite(preds).all():
            n_inf = (~torch.isfinite(preds)).sum().item()
            print(f"[Train] Batch {batch_idx}: Non-finite logits detected: {n_inf}, skipping batch")
            continue
        if torch.isnan(labels).any():
            print(f"[Train] Batch {batch_idx}: NaN in labels, skipping batch")
            continue

        # 计算损失
        loss = criterion(preds, labels.float()) / accum_steps
        loss.backward(retain_graph=True)
        # 梯度裁剪与梯度范数调试
        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        probs = torch.sigmoid(preds.detach())
        epoch_logits.extend(preds.detach().cpu().numpy())
        epoch_probs.extend(probs.cpu().numpy())

        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    # 如果没有任何有效样本
    if len(all_labels) == 0:
        print("[Train] Warning: no valid labels/preds collected this epoch.")
        return 0.0, 0.0, 0.0, 0.0

    # epoch-level stats
    epoch_logits_np = np.array(epoch_logits, dtype=float)
    epoch_probs_np = np.array(epoch_probs, dtype=float)
    if np.isnan(epoch_probs_np).any() or np.isnan(epoch_logits_np).any():
        print("[Train] Warning: NaN detected in epoch logits/probs!")

    preds_binary = (epoch_probs_np >= 0.5).astype(int)
    print(f"[Train][Epoch] Logits: min={epoch_logits_np.min():.4f}, max={epoch_logits_np.max():.4f}, mean={epoch_logits_np.mean():.4f}")
    print(f"[Train][Epoch] Prob : min={epoch_probs_np.min():.4f}, max={epoch_probs_np.max():.4f}, mean={epoch_probs_np.mean():.4f}")
    print(f"[Train][Epoch] Pred binary count: 0 -> {(preds_binary == 0).sum()}, 1 -> {(preds_binary == 1).sum()}")
    # sample (label, prob, pred)
    sample_n = min(5, len(all_labels))
    sample_pairs = list(zip(all_labels[:sample_n], all_preds[:sample_n], (np.array(all_preds)[:sample_n] >= 0.5).astype(int).tolist()))
    print(f"[Train][Epoch] Sample (label, prob, pred): {sample_pairs}")

    # safe metrics computation (guard against NaN)
    all_preds_np = np.array(all_preds, dtype=float)
    all_labels_np = np.array(all_labels, dtype=float)
    if np.isnan(all_preds_np).any() or np.isnan(all_labels_np).any():
        print("[Train] NaN detected in predictions/labels when computing metrics -> return zeros")
        return total_loss / len(train_loader), 0.0, 0.0, 0.0

    # need at least two classes in labels for roc
    if len(np.unique(all_labels_np)) < 2:
        print("[Train] Only one label class present in epoch, cannot compute AUC/AUPR/MCC")
        return total_loss / len(train_loader), 0.0, 0.0, 0.0

    # 计算训练指标（AUC, AUPR, MCC）
    auc = roc_auc_score(all_labels_np, all_preds_np)
    aupr = average_precision_score(all_labels_np, all_preds_np)
    mcc = matthews_corrcoef(all_labels_np, (all_preds_np >= 0.5).astype(int))

    avg_loss = total_loss / len(train_loader)
    return avg_loss, auc, aupr, mcc

def evaluate(model, data, edge_index, labels, batch_size=512, debug_print_batches=2):
    """
    model: the actual model.module (not necessarily DDP wrapper) or model
    data: HeteroData
    edge_index: test_edges (2 x N tensor on CPU or device)
    labels: test_labels (N)
    """
    model.eval()
    probs = []
    ys = []

    edge_type = ('peptide', 'interacts_with', 'target')
    edge_label_index = (['peptide', 'interacts_with', 'target'], edge_index)

    loader = LinkNeighborLoader(
        data,
        edge_label_index=edge_label_index,
        edge_label=labels,
        batch_size=batch_size,
        num_neighbors=[20, 10],
        shuffle=False
    )

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            try:
                edge_idx = batch.edge_label_index_dict[edge_type]
                batch_labels = batch.edge_label_dict[edge_type]
            except Exception as e:
                print(f"[Eval] Batch {batch_idx} missing edge_label info: {e}")
                continue

            pred_logits, _, _ = model(batch.x_dict, batch.edge_index_dict, edge_idx)
            pred_logits = pred_logits.view(-1)

            # numerical checks
            if not torch.isfinite(pred_logits).all():
                print(f"[Eval] Batch {batch_idx}: Non-finite logits detected, skipping")
                continue

            probs_batch = torch.sigmoid(pred_logits).cpu().numpy()
            labels_batch = batch_labels.cpu().numpy()

            # debug prints for first few batches
            if batch_idx < debug_print_batches:
                print(f"[Eval] Batch {batch_idx}: logits min={pred_logits.min().item():.4f}, max={pred_logits.max().item():.4f}, mean={pred_logits.mean().item():.4f}")
                print(f"[Eval] Batch {batch_idx}: label dist={np.bincount(labels_batch.astype(int))}")
                print(f"[Eval] Batch {batch_idx}: sample (label, prob, pred): {list(zip(labels_batch[:5], probs_batch[:5], (probs_batch[:5] >= 0.5).astype(int)))}")

            probs.append(probs_batch)
            ys.append(labels_batch)

    if len(probs) == 0 or len(ys) == 0:
        print("[Eval] No valid predictions collected.")
        return 0.5, 0.5, 0.0

    probs = np.concatenate(probs)
    ys = np.concatenate(ys)

    # guard against NaN/inf
    if np.isnan(probs).any() or np.isnan(ys).any():
        print("[Eval] NaN in probs/ys -> returning zeros")
        return 0.0, 0.0, 0.0

    if len(np.unique(ys)) < 2:
        print("[Eval] Only one label class in evaluation set -> returning defaults")
        return 0.5, 0.5, 0.0

    auc = roc_auc_score(ys, probs)
    aupr = average_precision_score(ys, probs)

    thresholds = np.linspace(0.01, 0.99, 99)
    mcc = -1.0
    for thr in thresholds:
        m = matthews_corrcoef(ys, (probs >= thr).astype(int))
        if m > mcc:
            mcc = m

    return auc, aupr, mcc


# -------------------- 主函数 --------------------
def data_process(arg):
    df = pd.read_csv(arg, header=None)
    df = df.drop(index=0)
    df = df.drop(columns=df.columns[0], errors='ignore')
    df = df.astype(float)
    df = df.values
    return df


def cutoff(Sc, alpha, variant='unweighted'):
    if variant == 'unweighted':
        return 0 if 0 < Sc < alpha else 1
    elif variant == 'weighted':
        return 0 if Sc < alpha else Sc


def prepare_data(DD, DT, alpha, variant='weighted'):
    # 对多肽-多肽相似性进行扩散
    DD_processed = np.array([cutoff(value, alpha, variant) for value in DD.flatten()]).reshape(DD.shape).astype(np.float32)
    DT_processed = DT.copy()
    return DD_processed, DT_processed


def main():
    # 环境变量设定
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用 InfiniBand (无特殊硬件需求可以加)
    os.environ['NCCL_P2P_DISABLE'] = '1'  # 有时能规避 NCCL Error 2
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 或 eth1，视机器网络接口而定

    parser = argparse.ArgumentParser()
    parser.add_argument("--DT", type=str, required=True, help="多肽-靶点相互作用矩阵")
    parser.add_argument("--DD", type=str, required=True, help="多肽-多肽相似性矩阵")
    parser.add_argument("--PPG", type=str, required=True, help="多肽图特征文件")
    parser.add_argument("--PPE", type=str, required=True, help="多肽进化特征文件")
    parser.add_argument("--TGD", type=str, required=True, help="靶点无序特征文件")
    parser.add_argument("--TGE", type=str, required=True, help="靶点进化特征文件")
    args = parser.parse_args()

    # 步骤 1：初始化分布式环境
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 步骤 2：数据加载与预处理
    DT = data_process(args.DT)
    DD = data_process(args.DD)
    DD_proc, DT_proc = prepare_data(DD, DT, alpha=0.4)

    # 步骤 3：加载特征 & 构图
    pep_feat, target_feat = load_features(args.PPG, args.PPE, args.TGD, args.TGE, device)
    data = build_hetero_graph(DD_proc, DT_proc, pep_feat, target_feat, device)
    data.DD_proc = DD_proc

    print("All edges in data['peptide', 'interacts_with', 'target']:", data['peptide', 'interacts_with', 'target'].edge_index.shape)

    # 步骤 4：训练数据划分
    train_edges, train_labels, test_edges, test_labels = split_train_test_from_matrix(DT_proc, device)

    print("Train edges:", train_edges.shape)
    print("Test edges:", test_edges.shape)

    # 步骤 5：构建训练集 train_loader
    edge_type = ('peptide', 'interacts_with', 'target')
    edge_label_index = (edge_type, train_edges)

    train_loader = LinkNeighborLoader(
        data,
        edge_label_index=edge_label_index,
        edge_label=train_labels,
        batch_size=256,
        num_neighbors=[20, 10],
        shuffle=True
    )
    print("Input node count:", torch.unique(train_edges[0]).shape)

    # 步骤 6：模型初始化
    model = EndToEndModel().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 步骤 7：优化器初始化
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7, verbose=True)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 步骤 8：正式训练
    e = 1
    patience = 10
    no_improve = 0
    best_val_metric = 0.5

    for epoch in range(1, 61):
        train_loss, train_auc, train_aupr, train_mcc = train(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        if rank == 0:
            val_auc, val_aupr, val_mcc = evaluate(model.module, data, test_edges, test_labels)
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | AUPR: {train_aupr:.4f} | MCC: {train_mcc:.4f}")
            print(f"               → Valid AUC: {val_auc:.4f} | AUPR: {val_aupr:.4f} | MCC: {val_mcc:.4f}")
            if val_auc > best_val_metric + 1e-4:
                best_val_metric = val_auc
                no_improve = 0
                torch.save(model.module.state_dict(), f"model/best_model_{epoch}.pt")
                e = epoch
                print(f"当前已于Epoch {epoch}保存最优模型")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"No improvement for {patience} epochs, early stopping at epoch {epoch}")
                break

            scheduler.step(val_auc)

    dist.destroy_process_group()
