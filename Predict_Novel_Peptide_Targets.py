import argparse
import torch
import torch.distributed as dist
import numpy as np
import pickle
import os
from torch_geometric.loader import LinkNeighborLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from PEARL import EndToEndModel, build_hetero_graph, data_process, prepare_data

# ==== 可修改的参数 ====
parser = argparse.ArgumentParser()
parser.add_argument("--DD", type=str, required=True, help="训练预测模型所用的多肽相似度矩阵文件路径")
parser.add_argument("--DT", type=str, required=True, help="训练预测模型所用的多肽-靶点相互作用矩阵文件路径")
parser.add_argument("--NP", type=str, required=True, help="新多肽与其余老多肽之间的相似度矩阵文件路径")
args = parser.parse_args()
top_k = 15
target_map_path = "data/UMPPI/protein_mapping.pkl"  # 靶点ID到名称的映射表路径

folder_path = "model"
file_list = os.listdir(folder_path)
file = file_list[0]
model_path = os.path.join(folder_path, file)

# ==== 环境变量设定 ====
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用 InfiniBand (无特殊硬件需求可以加)
os.environ['NCCL_P2P_DISABLE'] = '1'  # 有时能规避 NCCL Error 2
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 或 eth1，视机器网络接口而定

# ==== Step 1: 初始化分布式环境 ====
print("[INFO] 初始化分布式环境")

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

print("分布式环境初始化完毕！")

# ==== Step 2: 加载并处理多肽和靶点特征 ====
print("[INFO] 加载并处理多肽和靶点特征")

pep_all = torch.load("features/UMPPI/all_graph_features.pt", map_location=device)
evo_all = torch.load("features/UMPPI/all_evolutionary_features.pt", map_location=device)
peptide_seqs = list(pep_all.keys())

novel_peptide_feature = torch.load("TemporaryFile/novel_peptide_feature.pt", map_location=device)

# 特征对齐 + 加入新多肽
pep_tensor_list = [pep_all[seq] for seq in peptide_seqs]
evo_tensor_list = [evo_all[seq] for seq in peptide_seqs]
pep_feat_tensor = torch.stack([torch.cat([a, b], dim=0) for a, b in zip(pep_tensor_list, evo_tensor_list)], dim=0)
pep_feat_tensor = torch.nn.functional.normalize(pep_feat_tensor, dim=1)
pep_feat_tensor = torch.cat([pep_feat_tensor, novel_peptide_feature], dim=0).to(device)  # 加入新多肽

target_InD = torch.load("features/UMPPI/all_ProInD_features.pt", map_location=device)
target_Evo = torch.load("features/UMPPI/all_ProEvo_features.pt", map_location=device)
target_seqs = list(target_InD.keys())
target_feat_tensor = torch.stack([
    torch.cat([target_InD[seq], target_Evo[seq]], dim=0) for seq in target_seqs
], dim=0)
target_feat_tensor = torch.nn.functional.normalize(target_feat_tensor, dim=1)

print("多肽和靶点的特征处理完毕！")

# ==== Step 3: 加载训练预测模型所用的相似度矩阵、相互作用矩阵和新多肽与其余老多肽之间的相似度矩阵，并对这些矩阵进行适当处理以用于预测 ====
print("[INFO] 加载训练预测模型所用的相似度矩阵、相互作用矩阵和新多肽与其余老多肽之间的相似度矩阵，并对这些矩阵进行适当处理以用于预测")

DD = data_process(args.DD)
DT = data_process(args.DT)
NP = data_process(args.NP)

# 多肽相似度作用矩阵处理
DT = np.vstack([DT, [-1] * DT.shape[1]])  # 加一行全 -1 代表新多肽，shape: [N+1, M]

# 多肽相似度矩阵处理
new_row = NP
new_column = NP.reshape(-1, 1)

new_matrix = np.hstack([DD, new_column[:-1]])  # 先水平拼接新列到右侧，注意去掉最后一个重复的1
new_matrix = np.vstack([new_matrix, new_row])  # 再垂直拼接新行
DD = new_matrix

DD, DT = prepare_data(DD, DT, alpha=0.4)

print("预测所用多肽相似度矩阵和多肽-靶点相互作用矩阵构建完毕！")

# ==== Step 4: 构建异构图 ====
print("[INFO] 构建异构图")

data = build_hetero_graph(DD, DT, pep_feat_tensor, target_feat_tensor, device)
data.DD = DD

print("异构图构建完毕！")

# ==== Step 5: 加载模型并预测 ====
print("[INFO] 加载模型并进行预测")

model = EndToEndModel().to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
model.module.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()

num_pep, num_tar = DT.shape
novel_pep_id = num_pep - 1

def predict_novel_interactions(model, data, device, lambda_vote=0.05, sim_threshold=0.4, topk_neighbors=5, batch_size=256):
    """
    预测新多肽与所有靶点的交互概率，只返回 shape=(num_tar,) 的概率向量。
    """
    model.eval()
    edge_type = ('peptide', 'interacts_with', 'target')

    # 本rank要处理的节点
    tar_ids_all = torch.arange(num_tar, dtype=torch.long)
    tar_ids_rank = tar_ids_all[rank::world_size]  # 分片靶点

    # 覆盖新多肽（单个节点）和一批靶点节点的子图
    pep_cover_pairs = torch.stack(
        [torch.tensor([novel_pep_id], dtype=torch.long), torch.zeros(1, dtype=torch.long)], dim=0
    )  # 新多肽与一个固定靶点构成虚拟边

    tar_cover_pairs = torch.stack(
        [torch.zeros_like(tar_ids_rank), tar_ids_rank], dim=0
    )

    # 用 LinkNeighborLoader 做“节点覆盖式”的子图采样，批量前向计算 encoder
    loader_pep = LinkNeighborLoader(
        data,
        edge_label_index=(edge_type, pep_cover_pairs),
        batch_size=1,
        num_neighbors=[10, 10],
        shuffle=False
    )

    loader_tar = LinkNeighborLoader(
        data,
        edge_label_index=(edge_type, tar_cover_pairs),
        batch_size=batch_size,
        num_neighbors=[10, 10],
        shuffle=False
    )

    pep_emb_chunk = None
    tar_chunks = []

    with torch.no_grad():
        # 新多肽 embedding
        for batch in loader_pep:
            batch = batch.to(device)
            pep_emb_b, _ = model.module.encoder(batch.x_dict, batch.edge_index_dict)
            pep_gids = batch['peptide'].n_id.detach().cpu().numpy()
            for gid, emb in zip(pep_gids, pep_emb_b.detach().cpu().numpy()):
                if gid == novel_pep_id:
                    pep_emb_chunk = (gid, emb)
                    break

        # 靶点 embedding
        for batch in loader_tar:
            batch = batch.to(device)
            _, tar_emb_b = model.module.encoder(batch.x_dict, batch.edge_index_dict)
            tar_gids = batch['target'].n_id.detach().cpu().numpy()
            tar_chunks.append((tar_gids, tar_emb_b.detach().cpu().numpy()))

    # all_gather 到 rank0
    if world_size > 1 and dist.is_initialized():
        pep_list = [None for _ in range(world_size)]
        tar_list = [None for _ in range(world_size)]
        dist.all_gather_object(pep_list, pep_emb_chunk)
        dist.all_gather_object(tar_list, tar_chunks)
    else:
        pep_list = [pep_emb_chunk]
        tar_list = [tar_chunks]

    if rank != 0:
        return None

    # rank0 拼装 embedding
    pep_emb = None
    for item in pep_list:
        if item is not None:
            pep_emb = item[1]
            break
    assert pep_emb is not None, "未能获取新多肽 embedding"
    pep_emb = torch.tensor(pep_emb, dtype=torch.float32, device=device).unsqueeze(0)  # [1, dim]

    tar_dim = pep_emb.size(1)
    tar_emb_full = np.zeros((num_tar, tar_dim), dtype=np.float32)
    tar_filled = np.zeros((num_tar,), dtype=np.bool_)
    for parts in tar_list:
        if parts is None:
            continue
        for gids, emb in parts:
            tar_emb_full[gids] = emb
            tar_filled[gids] = True
    if not tar_filled.all():
        tar_emb_full[~tar_filled] = tar_emb_full[tar_filled].mean(axis=0, keepdims=True)

    tar_emb_t = torch.tensor(tar_emb_full, dtype=torch.float32, device=device)

    # 预测概率
    full_probs = np.zeros((num_tar,), dtype=np.float32)
    with torch.no_grad():
        tar_range = torch.arange(num_tar, device=device)
        src_idx = torch.zeros_like(tar_range)  # 新多肽索引=0（batch内），predictor不关心全局id
        flat_edges = torch.stack([src_idx, tar_range], dim=0)
        scores = model.module.predictor(pep_emb, tar_emb_t, flat_edges)
        probs = torch.sigmoid(scores).detach().cpu().numpy()
        full_probs[:] = probs

    # 对新多肽这一行进行投票增强
    # 选出Top-k相似邻居并做阈值过滤
    sim_row = NP.astype(np.float32).copy()
    sim_row[-1] = 0
    sim_row[sim_row < sim_threshold] = 0

    # 选出有意义的邻居
    cand_idx = np.flatnonzero(sim_row > 0)
    if cand_idx.size == 0:
        enhanced_row = full_probs  # 没有可用邻居，直接返回新多肽自身预测
    else:
        # 取 Top-K 相似邻居
        if cand_idx.size > topk_neighbors:
            topk_idx = np.argpartition(sim_row[cand_idx], -topk_neighbors)[-topk_neighbors:]
            neigh_idx = cand_idx[topk_idx]
        else:
            neigh_idx = cand_idx

        # 归一化权重
        w = sim_row[neigh_idx]
        w = w / (w.sum() + 1e-8)  # (K,)

        # 预测这些邻居多肽对全部靶点的概率矩阵 (K, num_tar) —— 仅按需计算，避免 OOM
        device = next(model.module.parameters()).device
        edge_type = ('peptide', 'interacts_with', 'target')

        K = len(neigh_idx)
        neighbor_probs = np.zeros((K, num_tar), dtype=np.float32)

        # 为了不OOM，按“靶点块”做分块推理
        tar_block = 512  # 可根据显存灵活调大/调小
        for t0 in range(0, num_tar, tar_block):
            t1 = min(num_tar, t0 + tar_block)
            tar_ids = torch.arange(t0, t1, device=device, dtype=torch.long)  # (B_t,)
            pep_ids = torch.tensor(neigh_idx, device=device, dtype=torch.long)  # (K,)

            # 构造 (K * B_t) 条边
            src = pep_ids.repeat_interleave(tar_ids.numel())      # (K * B_t,)
            dst = tar_ids.repeat(pep_ids.numel())                 # (K * B_t,)
            eidx = torch.stack([src, dst], dim=0)                 # (2, K * B_t)

            edge_label_index = (edge_type, eidx)

            # 用 LinkNeighborLoader 进行子图采样 + 批推理
            loader = LinkNeighborLoader(
                data,
                edge_label_index=edge_label_index,
                batch_size=batch_size,
                num_neighbors=[10, 10],
                shuffle=False
            )

            logits_blk = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    pred, _, _ = model.module(
                        batch.x_dict,
                        batch.edge_index_dict,
                        batch.edge_label_index_dict[edge_type]
                    )
                    logits_blk.append(pred.detach().view(-1).cpu())

            # (K * B_t,) -> (K, B_t)
            probs_blk = torch.sigmoid(torch.cat(logits_blk, dim=0)).view(K, -1).numpy()
            neighbor_probs[:, t0:t1] = probs_blk  # 写入对应列块

        # 投票融合：w @ neighbor_probs -> (num_tar,)
        vote = (w @ neighbor_probs).astype(np.float32)

        # 融合成最终的增强行
        enhanced_row = (1 - lambda_vote) * full_probs + lambda_vote * vote

    return enhanced_row


inter_scores = predict_novel_interactions(model=model, data=data, device=device)

print("预测完毕！")

# ==== Step 6: 输出 Top-K 结果 ====
def invert_dict(dictionary):
    return {val: key for key, val in dictionary.items()}

def find_key_by_value(dictionary, value):
    inverted_dict = invert_dict(dictionary)
    return inverted_dict.get(value)


if rank == 0:
    print("\n[预测结果]")
    top_indices = inter_scores.argsort()[-top_k:][::-1]
    np.savetxt("results/predictions_of_novel_peptide.csv", inter_scores, fmt="%.6f")

    # 映射靶点ID为名称
    with open(target_map_path, 'rb') as f:
        target_map = pickle.load(f)

    for idx in top_indices:
        seq = find_key_by_value(target_map, idx)
        print(f"{idx} | {seq[:20]}... | Score: {inter_scores[idx]:.6f}")

dist.destroy_process_group()
