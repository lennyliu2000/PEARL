import argparse
import torch
import os

from Peptide_Processing import SequenceProcessingModule, FeatureExtractionModule

# ==== 可修改的参数 ====
parser = argparse.ArgumentParser()
parser.add_argument("--Sequence", type=str, required=True, help="新多肽的氨基酸序列(Novel Peptide Sequence)")
parser.add_argument("--Name", type=str, required=True, help="新多肽的名称(Novel Peptide Name)")

args = parser.parse_args()
input_sequence = args.Sequence
peptide_name = args.Name
blast_db_path = "D:/blastdb/pp"  # 本地 BLAST 数据库路径
output_dir = "TemporaryFile"

# ==== 特征提取 ====
print(f"[INFO] 正在处理多肽序列: {input_sequence}")

seq_processor = SequenceProcessingModule()
feature_extractor = FeatureExtractionModule()

mol_graph = seq_processor.sequence_to_molecular_graph(input_sequence)
pssm = seq_processor.sequence_to_pssm(peptide_name, input_sequence, blast_db_path)

graph_feat, evo_feat = feature_extractor(mol_graph, pssm)
graph_feat = torch.nn.functional.normalize(graph_feat, dim=0)
evo_feat = torch.nn.functional.normalize(evo_feat, dim=0)
novel_peptide_feature = torch.cat([graph_feat, evo_feat], dim=0).unsqueeze(0)
novel_peptide_feature_file = os.path.join(output_dir, "novel_peptide_feature.pt")
torch.save(novel_peptide_feature, novel_peptide_feature_file)

print("新多肽的特征处理完毕！")
