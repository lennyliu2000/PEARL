import pynvml
import os
import sys

def select_available_gpus(min_free_mem_gb=8):
    """
    筛选出剩余显存大于 min_free_mem_gb 的显卡列表
    返回符合要求的显卡索引列表
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    available_gpus = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem_gb = mem_info.free / 1024 / 1024 / 1024

        print(f"[GPU {i}] Free memory: {free_mem_gb:.2f} GB")

        if free_mem_gb >= min_free_mem_gb:
            available_gpus.append(i)

    pynvml.nvmlShutdown()
    return available_gpus

def setup_gpus(min_free_mem_gb=8):
    available_gpus = select_available_gpus(min_free_mem_gb)

    if not available_gpus:
        print("没有找到空闲的满足剩余显存要求的GPU，终止训练!")
        sys.exit(1)

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    visible_devices = ",".join(str(i) for i in available_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    print(f"使用以下 GPU 进行训练: {visible_devices}")

