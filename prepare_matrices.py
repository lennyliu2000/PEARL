#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-shot generation of the two known-peptide x known-peptide similarity matrices required for fusion inference.

The fusion model (embedding + sequence + interaction three views) needs, when building the deployment graph,
the static **embedding view** and **sequence view** matrices between known peptides:
  - S_embedding: cosine similarity of protT5 pooled vectors   -> data/UMPPI/sim_adj_matrix_protT5.csv
  - S_sequence : Smith-Waterman local alignment (BLAST definition) -> data/UMPPI/sim_adj_matrix_blast.csv
  (the S_interaction view is computed by the script at inference time per fold/known library, no pre-generation needed)

These two matrices are fairly large (a few hundred MB each), so they are not committed with the code; instead
this script computes them once from the dataset and protT5 features shipped in the repo. After generation you can
run predict_novel_peptide_fusion.py.

Usage:
  python prepare_matrices.py                # generate both (blast uses all CPU-2 processes, ~tens of minutes for 6603 peptides)
  python prepare_matrices.py --only embedding   # generate only protT5 cosine (seconds)
  python prepare_matrices.py --workers 32       # set blast parallel process count
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

DATASET = "data/Dataset_all_balanced_new_2.tsv"
PEP_T5 = "features/peptide_protT5.pt"
EMB_OUT = "data/UMPPI/sim_adj_matrix_protT5.csv"
BLAST_OUT = "data/UMPPI/sim_adj_matrix_blast.csv"


def peptide_order(dataset_file):
    df = pd.read_csv(dataset_file, sep="\t", usecols=["pep_seq"])
    return list(pd.unique(df["pep_seq"].astype(str)))


def build_embedding_matrix(dataset_file, pep_t5_file, out_path):
    """S_embedding: cosine similarity of protT5 pooled vectors (diagonal=1)."""
    seqs = peptide_order(dataset_file)
    store = torch.load(pep_t5_file, map_location="cpu", weights_only=False)
    missing = [s for s in seqs if s not in store]
    if missing:
        raise KeyError(f"{len(missing)} peptides are missing protT5 features in {pep_t5_file}, e.g.: {missing[0][:40]}")
    mat = torch.stack([torch.as_tensor(store[s], dtype=torch.float32).view(-1) for s in seqs])
    mat = torch.nn.functional.normalize(mat, dim=1)
    sim = (mat @ mat.T).clamp(-1.0, 1.0).numpy().astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(sim, index=seqs, columns=seqs).to_csv(out_path)
    print(f"[EMB] saved {out_path}  shape={sim.shape}")


def build_sequence_matrix(dataset_file, out_path, workers):
    """S_sequence: reuse the SW alignment implementation from build_peptide_similarity_views."""
    import build_peptide_similarity_views as B
    seqs = B.load_peptides(dataset_file)
    B.build_blast_matrix(seqs, workers, out_path)


def main():
    ap = argparse.ArgumentParser(description="Generate the similarity matrices required for fusion inference")
    ap.add_argument("--dataset-file", default=DATASET)
    ap.add_argument("--pep-t5-file", default=PEP_T5)
    ap.add_argument("--emb-out", default=EMB_OUT)
    ap.add_argument("--blast-out", default=BLAST_OUT)
    ap.add_argument("--only", choices=["embedding", "sequence"], default=None,
                    help="generate only one of them (default: generate both)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    if args.only in (None, "embedding"):
        build_embedding_matrix(args.dataset_file, args.pep_t5_file, args.emb_out)
    if args.only in (None, "sequence"):
        build_sequence_matrix(args.dataset_file, args.blast_out, args.workers)


if __name__ == "__main__":
    main()
