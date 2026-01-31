import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
import os
from Bio.Blast.Applications import NcbipsiblastCommandline


class SequenceProcessingModule:
    """
    序列处理模块：将肽序列转换为分子图和进化谱（PSSM）。
    """

    def __init__(self):
        pass

    def sequence_to_molecular_graph(self, peptide_sequence):
        """
        将肽序列转换为分子图。

        参数:
        peptide_sequence (str): 肽序列，例如 "ACDEFGHIKLMNPQRSTVWY"

        返回:
        mol_graph (dict): 分子图，包含节点和边的信息
        """
        # 使用RDKit将肽序列转换为分子结构
        mol = Chem.MolFromSequence(peptide_sequence)

        # 提取分子图中的节点和边
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]

        # 构建分子图
        mol_graph = {
            'nodes': atoms,
            'edges': bonds
        }

        return mol_graph

    def sequence_to_pssm(self, peptide_name, peptide_sequence, blast_db_path):
        """
        使用本地 PSI-BLAST 将肽序列转换为位置特异性评分矩阵（PSSM）。

        参数:
        peptide_sequence (str): 多肽序列
        blast_db_path (str): 本地 BLAST 数据库路径

        返回:
        pssm (np.array): PSSM 矩阵，形状为 (序列长度, 20)
        """
        # 将多肽序列保存到临时文件中
        fasta_file = f"TemporaryFile/{peptide_name}.fasta"
        pssm_file = f"TemporaryFile/{peptide_name}.pssm"
        with open(fasta_file, "w") as f:
            f.write(f">{peptide_name}\n{peptide_sequence}\n")

        # 使用本地 PSI-BLAST 生成 PSSM
        psi_blast_cline = NcbipsiblastCommandline(
            db=blast_db_path,  # BLAST 数据库位置
            query=fasta_file,  # 输入的多肽序列文件
            num_iterations=3,  # PSI-BLAST 迭代次数
            out_ascii_pssm=pssm_file  # PSSM 矩阵输出的位置
        )

        # 运行 PSI-BLAST,生成 PSSM
        psi_blast_cline()

        # 判断是否生成 PSSM 文件，如果未生成就增大evalue值重新生成
        if not os.path.exists(pssm_file):
            psi_blast_cline = NcbipsiblastCommandline(
                db=blast_db_path,
                query=fasta_file,
                num_iterations=3,
                out_ascii_pssm=pssm_file,
                evalue=1000
            )
            psi_blast_cline()

        # 若最终仍未生成PSSM文件，返回全0矩阵
        if not os.path.exists(pssm_file):
            seq_length = len(peptide_sequence)
            zero_matrix = np.zeros((seq_length, 20), dtype=int)
            os.remove(fasta_file)  # 删除临时FASTA文件
            return zero_matrix

        pssm_matrix = []

        # 打开PSSM文件并逐行读取
        with open(pssm_file, 'r') as file:
            for line in file:
                # 跳过空行和注释行
                if line.strip() == '' or line.startswith('Last') or line.startswith('Standard') or line.startswith(
                        'PSI'):
                    continue

                # 分割行并提取PSSM值
                parts = line.split()
                if len(parts) >= 22:  # 确保行包含足够的列
                    try:
                        # 检查第一个部分是否是行号（如 '1 F'）
                        if parts[0].isdigit():
                            # 提取从第3列到第22列的PSSM值（对应氨基酸A到V）
                            pssm_values = list(map(int, parts[2:22]))
                        else:
                            # 提取从第2列到第21列的PSSM值（对应氨基酸A到V）
                            pssm_values = list(map(int, parts[1:21]))
                        pssm_matrix.append(pssm_values)
                    except ValueError:
                        # 如果转换失败，跳过该行（可能是标题行或其他非数据行）
                        continue

        # 删除临时文件
        os.remove(fasta_file)
        os.remove(pssm_file)

        return np.array(pssm_matrix, dtype=int)


class FeatureExtractionModule(nn.Module):
    """
    特征提取模块：从分子图和PSSM中提取特征。
    """

    def __init__(self, gnn_layers=4, embedding_dim=128, cnn_filters=32, lstm_hidden_dim=64):
        super(FeatureExtractionModule, self).__init__()

        # GNN层
        self.gnn_layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(gnn_layers)
        ])

        # CNN层
        self.cnn = nn.Conv2d(1, cnn_filters, kernel_size=(3, 20), padding=(1, 0))

        # BiLSTM层
        self.bilstm = nn.LSTM(cnn_filters, lstm_hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, mol_graph, pssm):
        """
        前向传播：从分子图和PSSM中提取特征。

        参数:
        mol_graph (dict): 分子图，包含节点和边的信息
        pssm (np.array): PSSM矩阵，形状为 (序列长度, 20)

        返回:
        graph_features (torch.Tensor): 从分子图中提取的特征
        evolutionary_features (torch.Tensor): 从PSSM中提取的特征
        """
        # 从分子图中提取特征
        graph_features = self.extract_graph_features(mol_graph)

        # 从PSSM中提取特征
        evolutionary_features = self.extract_evolutionary_features(pssm)

        return graph_features, evolutionary_features

    def extract_graph_features(self, mol_graph):
        """
        从分子图中提取特征。

        参数:
        mol_graph (dict): 分子图，包含节点和边的信息

        返回:
        graph_features (torch.Tensor): 从分子图中提取的特征
        """
        # 初始化节点嵌入
        node_embeddings = torch.randn(len(mol_graph['nodes']), self.gnn_layers[0].in_features)

        # GNN层的前向传播
        for gnn_layer in self.gnn_layers:
            node_embeddings = F.relu(gnn_layer(node_embeddings))

        # 取节点嵌入的平均值作为图的特征
        graph_features = torch.mean(node_embeddings, dim=0)

        return graph_features

    def extract_evolutionary_features(self, pssm):
        """
        从PSSM中提取特征。

        参数:
        pssm (np.array): PSSM矩阵，形状为 (序列长度, 20)

        返回:
        evolutionary_features (torch.Tensor): 从PSSM中提取的特征
        """
        # 将PSSM转换为PyTorch张量
        pssm_tensor = torch.tensor(pssm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, 序列长度, 20)

        # CNN层的前向传播
        cnn_output = F.relu(self.cnn(pssm_tensor))  # 形状: (1, cnn_filters, 序列长度, 1)
        cnn_output = cnn_output.squeeze(0).permute(1, 0, 2)  # 形状: (序列长度, cnn_filters, 1)
        cnn_output = cnn_output.squeeze(-1)  # 形状: (序列长度, cnn_filters)

        # BiLSTM层的前向传播
        lstm_output, _ = self.bilstm(cnn_output.unsqueeze(0))  # 形状: (1, 序列长度, lstm_hidden_dim * 2)

        # 取LSTM输出的最后一个时间步的特征
        evolutionary_features = lstm_output[:, -1, :]  # 形状: (1, lstm_hidden_dim * 2)

        return evolutionary_features.squeeze(0)  # 形状: (lstm_hidden_dim * 2)


def save_all_features(all_graph_features, all_evolutionary_features, output_dir="features"):
    """
    将所有多肽的分子图特征和进化特征分别保存到一个文件中。

    参数:
    peptide_names (list): 多肽名称列表
    all_graph_features (dict): 所有多肽的分子图特征
    all_evolutionary_features (dict): 所有多肽的进化特征
    output_dir (str): 输出目录
    """
    # 保存所有分子图特征
    graph_features_file = os.path.join(output_dir, "all_graph_features.pt")
    torch.save(all_graph_features, graph_features_file)

    # 保存所有进化特征
    evolutionary_features_file = os.path.join(output_dir, "all_evolutionary_features.pt")
    torch.save(all_evolutionary_features, evolutionary_features_file)

    print(f"Saved all graph features to {graph_features_file}")
    print(f"Saved all evolutionary features to {evolutionary_features_file}")


def replace_nonstandard_aa(sequence):
    replacement_map = {'B': 'D', 'J': 'L', 'O': 'K', 'U': 'C', 'X': 'A', 'Z': 'E'}
    return ''.join([replacement_map.get(aa, aa) for aa in sequence])


def process_peptides_from_file(file_path, blast_db_path, output_dir="features"):
    """
    从文件中读取多肽序列，生成分子图和 PSSM 矩阵，提取特征并保存到文件中。

    参数:
    file_path (str): 多肽序列文件路径
    blast_db_path (str): 本地 BLAST 数据库路径
    output_dir (str): 输出目录
    """

    # 读取多肽序列文件
    df = pd.read_excel(file_path)
    peptide_sequences = df.iloc[:, 0].unique()

    # 初始化序列处理模块和特征提取模块
    seq_processor = SequenceProcessingModule()
    feature_extractor = FeatureExtractionModule()

    # 存储所有多肽的特征
    all_graph_features = {}
    all_evolutionary_features = {}

    count = 0  # 计数器

    # 批量处理每个多肽序列
    for peptide in peptide_sequences:
        peptide_name = peptide
        peptide_sequence = peptide

        # 对多肽序列进行处理，替换掉非标准氨基酸，以确保特征提取顺利进行
        peptide_sequence = replace_nonstandard_aa(peptide_sequence)

        # 生成分子图
        mol_graph = seq_processor.sequence_to_molecular_graph(peptide_sequence)

        # 生成 PSSM 矩阵
        pssm = seq_processor.sequence_to_pssm(peptide_name, peptide_sequence, blast_db_path)

        # 提取特征
        graph_features, evolutionary_features = feature_extractor(mol_graph, pssm)

        # 将特征存储到字典中
        all_graph_features[peptide_name] = graph_features
        all_evolutionary_features[peptide_name] = evolutionary_features

        count += 1
        print(f"Processed features for peptide{count}")

    # 保存所有特征
    save_all_features(all_graph_features, all_evolutionary_features, output_dir)


if __name__ == "__main__":
    # 多肽序列文件路径
    file_path = "data/UMPPI/total_pti.xlsx"

    # 本地 BLAST 数据库路径
    blast_db_path = "D:/blastdb/pp"

    # 处理多肽序列并保存特征
    process_peptides_from_file(file_path, blast_db_path, output_dir="features")
