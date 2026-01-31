import os
import pickle
import pandas as pd

def replace_nonstandard_aa(sequence):
    replacement_map = {'B': 'D', 'J': 'L', 'O': 'K', 'U': 'C', 'X': 'A', 'Z': 'E'}
    return ''.join([replacement_map.get(aa, aa) for aa in sequence])

file_path = "data/UMPPI/total_pti.xlsx"
df = pd.read_excel(file_path)
protein_sequences = df.iloc[:, 1].unique()

# 创建targets文件夹（如果不存在）
targets_dir = 'data/UMPPI/targets/'
os.makedirs(targets_dir, exist_ok=True)

with open('data/UMPPI/protein_mapping.pkl', 'rb') as f:
    pro_ids = pickle.load(f)

for protein in protein_sequences:
    pro_id = f"prot_{pro_ids[protein]}"
    protein = replace_nonstandard_aa(protein)
    fasta_file = f'data/UMPPI/targets/{pro_id}.fasta'
    with open(fasta_file, 'w') as f:
        f.write(f">{pro_id}\n{protein}")
    print(f"Processed fasta for {pro_id}")

print("Processing completed!")