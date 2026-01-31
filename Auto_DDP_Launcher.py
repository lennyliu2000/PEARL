import subprocess
import torch.cuda
from Select_Available_Gpus import setup_gpus

# 设定训练所需的最低空闲显存（单位：GB）
min_free_mem_gb = 8

# 获取当前空闲 GPU
setup_gpus(min_free_mem_gb)

nproc = torch.cuda.device_count()
print(f"使用 {nproc} 块GPU进行训练")

# 构造 torchrun 命令
cmd = [
    "torchrun",
    "--standalone",
    "--nnodes=1",
    f"--nproc_per_node={nproc}",
    "start.py",  # 你的主脚本（import 多卡逻辑）
    "--DT", "data/UMPPI/pti_adj_matrix.csv",
    "--DD", "data/UMPPI/sim_adj_matrix.csv",
    "--PPG", "features/UMPPI/all_graph_features.pt",
    "--PPE", "features/UMPPI/all_evolutionary_features.pt",
    "--TGD", "features/UMPPI/all_ProInD_features.pt",
    "--TGE", "features/UMPPI/all_ProEvo_features.pt"
]

# 用 subprocess 执行
subprocess.run(" ".join(cmd), shell=True)
