import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class IntrinsicDisorderDataset(Dataset):
    """改进后的数据集类，可正确处理矩阵文件"""
    def __init__(self, disorder_dir):
        """
        参数:
        disorder_dir (str): 无序性评分矩阵目录
        """
        self.disorder_dir = disorder_dir

        # 获取有效文件列表
        valid_files = []
        valid_lengths = []

        for f in sorted(os.listdir(self.disorder_dir)):
            if f.endswith('.txt'):
                file_path = os.path.join(self.disorder_dir, f)
                try:
                    data = np.loadtxt(file_path, skiprows=1)
                    if data.size == 0:
                        continue
                    seq_len = data.shape[0] if data.ndim > 1 else 1
                    valid_files.append(f)
                    valid_lengths.append(seq_len)
                except Exception as e:
                    print(f"忽略无效文件 {f}: {str(e)}")
        print(f"成功加载 {len(valid_files)} 个文件")
        self.file_list, self.original_lengths = valid_files, valid_lengths

        # 按原始长度降序排列文件索引
        self.sorted_indices = np.argsort(self.original_lengths)[::-1]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 加载原始数据（不填充）
        file_name = self.file_list[idx]
        data = np.loadtxt(os.path.join(self.disorder_dir, file_name), skiprows=1)
        protein_id = os.path.splitext(file_name)[0][:-7]  # 从文件名提取ID

        return {
            'data': torch.FloatTensor(data),
            'protein_id': protein_id,
            'original_length': data.shape[0]
        }

def collate_fn(batch, quantile=0.75):
    # 动态填充到批次内最大长度
    original_lengths = torch.tensor([item["original_length"] for item in batch])
    crit_length = int(np.quantile(original_lengths, quantile))  # 标准长度，根据75%分位数得到

    collated_data = []
    protein_ids = []
    for item in batch:
        data = item["data"]
        if data.shape[0] > crit_length:
            data = data[:crit_length]  # 长序列：截断尾部（优先保留N端）
            collated_data.append(data)
            protein_ids.append(item["protein_id"])
        else:
            pad_len = crit_length - data.shape[0]
            collated_data.append(F.pad(data, (0, 0, 0, pad_len)))  # 短序列：填充至标准长度
            protein_ids.append(item["protein_id"])

    return {
        "data": torch.stack(collated_data),
        "protein_id": protein_ids,
        "original_length": original_lengths,
        "collated_length": crit_length
    }

class DisorderFeatureExtractor(nn.Module):
    """优化后的特征提取网络"""
    def __init__(self, input_dim=3, lstm_dim=128):
        super().__init__()

        # 卷积模块（启用 ceil_mode）
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),  # 序列长度减半
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True)  # 序列长度再减半
        )

        # 双向LSTM,输入维度应为128（卷积层输出通道数）
        self.lstm = nn.LSTM(
            input_size=128,  # 确保与卷积输出匹配
            hidden_size=lstm_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=2
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(2 * lstm_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        # 最终投影层
        self.projection = nn.Linear(2 * lstm_dim, 128)

    def forward(self, x, collated_len):
        # 输入形状: (batch_size, padded_seq_len, 3)
        x = x.permute(0, 2, 1)  # (batch_size, 3, collated_seq_len)

        # 卷积处理
        x = self.conv_block(x)  # 输出形状: (batch, 128, pool_seq_len)

        # 调整数据维度: (batch_size, pool_seq_len, 128)
        x = x.permute(0, 2, 1)

        # 基于collated_len统一计算池化后的有效长度
        pooled_len = int(np.ceil(collated_len / 2))
        pooled_len = int(np.ceil(pooled_len / 2))
        pooled_len = max(pooled_len, 1)

        pooled_lengths = torch.full((x.size(0),), pooled_len, dtype=torch.long, device=x.device)

        # 使用pack_padded_sequence处理变长序列
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            pooled_lengths.cpu().numpy(),  # 必须使用调整后的长度
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM处理
        lstm_out, _ = self.lstm(packed)

        # 解包
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # 注意力加权
        attention_weights = self.attention(unpacked)
        context = torch.sum(attention_weights * unpacked, dim=1)

        # 特征投影
        return self.projection(context)

def invert_dict(dictionary):
    return {val: key for key, val in dictionary.items()}

def find_key_by_value(dictionary, value):
    inverted_dict = invert_dict(dictionary)
    return inverted_dict.get(value)

def extract_combined_features():
    # 配置参数
    config = {
        'disorder_dir': './data/UMPPI/IntrinsicDisorder',
        'output_path': './features/all_ProInD_features.pt',
        'batch_size': 64,
    }

    # 初始化数据加载器
    dataset = IntrinsicDisorderDataset(config['disorder_dir'])
    print(f"数据集大小: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisorderFeatureExtractor().to(device)
    model.eval()

    # 特征收集字典
    feature_dict = {}

    # 蛋白质id映射对照表
    with open('data/UMPPI/protein_mapping.pkl', 'rb') as f:
        mapping_dict = pickle.load(f)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['data'].to(device)
            collated_len = batch['collated_length']
            ids = batch['protein_id']
            prot_seqs = []
            for ID in ids:
                prot_seq = find_key_by_value(mapping_dict, int(ID[5:]))  # 反向蛋白质id映射对照表，用于寻找序列
                prot_seqs.append(prot_seq)
                print(f"Processing features for {ID}")

            # 提取特征
            features = model(inputs, collated_len)

            # 收集特征到字典
            for prot_seq, feat in zip(prot_seqs, features):
                feature_dict[prot_seq] = feat

    # 保存整体特征文件
    torch.save(feature_dict, config['output_path'])
    print(f"特征提取完成，共处理{len(feature_dict)}个蛋白质靶点")
    print(f"合并特征文件已保存至：{config['output_path']}")


if __name__ == '__main__':
    extract_combined_features()
