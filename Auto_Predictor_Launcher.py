import subprocess
import torch.cuda
from Select_Available_Gpus import setup_gpus

# 设定预测所需的最低空闲显存（单位：GB）
min_free_mem_gb = 8

# 获取当前空闲 GPU
setup_gpus(min_free_mem_gb)

nproc = torch.cuda.device_count()
print(f"使用 {nproc} 块GPU进行预测")

# 构造 torchrun 命令
cmd = [
    "torchrun",
    "--standalone",
    "--nnodes=1",
    f"--nproc_per_node={nproc}",
    "Predict_Novel_Peptide_Targets.py",  # 你的主脚本（import 多卡逻辑）
    "--DD", "data/UMPPI/sim_adj_matrix.csv",
    "--DT", "data/UMPPI/pti_adj_matrix.csv",
    "--NP", "TemporaryFile/np_sim_matrix.csv"
]

# 用 subprocess 执行
subprocess.run(" ".join(cmd), shell=True)
