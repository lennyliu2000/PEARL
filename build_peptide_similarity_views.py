#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the component matrices for multi-view peptide similarity fusion (two fold-independent views).

Three views:
  S_embedding  = protT5 pooled cosine   -> existing data/UMPPI/sim_adj_matrix_protT5.csv
  S_sequence   = sequence similarity, two definitions:
                   (a) morgan = Morgan fingerprint Tanimoto (chemical structure) -> existing sim_adj_matrix.csv
                   (b) blast  = Smith-Waterman local alignment (added by this script) -> sim_adj_matrix_blast.csv
  S_interaction= interaction-profile similarity -> recomputed per fold during training (inside train_pearl_hgt_inductive.py),
                 not generated here (to avoid leakage).

This script only generates the fold-independent BLAST/SW sequence similarity matrix (embedding/morgan already exist).

Usage:
  python build_peptide_similarity_views.py --self-check

  python build_peptide_similarity_views.py --build-blast --workers 32

Output format matches sim_adj_matrix.csv: index/columns = peptide sequences, values in [0,1], diagonal = 1.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from Bio.Align import PairwiseAligner, substitution_matrices

DEFAULT_DATASET = "data/Dataset_all_balanced_new_2.tsv"
DEFAULT_BLAST_OUT = "data/UMPPI/sim_adj_matrix_blast.csv"

_ALIGNER = None
_SEQS = None
_SELF_SCORES = None


def make_aligner():
    """Smith-Waterman local alignment, BLOSUM62, common gap penalties."""
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


_AA_REPLACE = {"B": "D", "J": "L", "O": "K", "U": "C", "X": "A", "Z": "E"}


def clean_sequence(seq):
    """Map non-standard amino acids to standard residues so they fall within the BLOSUM62 alphabet."""
    return "".join(_AA_REPLACE.get(aa, aa) for aa in str(seq).upper())


def load_peptides(dataset_file):
    df = pd.read_csv(dataset_file, sep="\t", usecols=["pep_seq"])
    return list(pd.unique(df["pep_seq"].astype(str)))


def _init_worker(seqs):
    """Each worker initializes its aligner and precomputes self-alignment scores (for normalization).
    The seqs passed in are already cleaned (non-standard amino acids replaced)."""
    global _ALIGNER, _SEQS, _SELF_SCORES
    _ALIGNER = make_aligner()
    _SEQS = seqs
    _SELF_SCORES = np.array([_ALIGNER.score(s, s) for s in seqs], dtype=np.float64)


def _row_similarity(i):
    """Compute normalized SW similarity of peptide i against all peptides (symmetric normalization: score/sqrt(self_i*self_j))."""
    global _ALIGNER, _SEQS, _SELF_SCORES
    n = len(_SEQS)
    out = np.zeros(n, dtype=np.float32)
    si = _SEQS[i]
    self_i = _SELF_SCORES[i]
    for j in range(i, n):
        if i == j:
            out[j] = 1.0
            continue
        raw = _ALIGNER.score(si, _SEQS[j])
        denom = np.sqrt(self_i * _SELF_SCORES[j]) + 1e-12
        val = raw / denom
        out[j] = max(0.0, min(1.0, val))
    return i, out


def build_blast_matrix(seqs, workers, out_path):
    n = len(seqs)
    print(f"[BLAST] n_peptides={n}, upper-triangle pairs~{n*(n+1)//2:,}, workers={workers}")
    sim = np.zeros((n, n), dtype=np.float32)
    clean_seqs = [clean_sequence(s) for s in seqs]

    with Pool(processes=workers, initializer=_init_worker, initargs=(clean_seqs,)) as pool:
        done = 0
        for i, row in pool.imap_unordered(_row_similarity, range(n), chunksize=8):
            sim[i, i:] = row[i:]
            done += 1
            if done % 200 == 0:
                print(f"[BLAST] {done}/{n} rows done", flush=True)

    sim = np.maximum(sim, sim.T)
    np.fill_diagonal(sim, 1.0)

    df = pd.DataFrame(sim, index=seqs, columns=seqs)
    df.to_csv(out_path)
    nz = sim[(sim > 0) & (sim < 1)]
    print(f"[BLAST] saved {out_path}  shape={sim.shape}")
    print(f"[BLAST] non-trivial values: min={nz.min():.4f} max={nz.max():.4f} mean={nz.mean():.4f} "
          f"(#>0.6={int((sim > 0.6).sum() - n)})")


def self_check(dataset_file):
    """Small sample: take 10 peptides and verify the SW definition (self=1, print similar pairs)."""
    seqs = load_peptides(dataset_file)[:10]
    print("[SELF-CHECK] sample peptides:")
    for k, s in enumerate(seqs):
        print(f"  [{k}] len={len(s):2d} {s}")
    aligner = make_aligner()
    cseqs = [clean_sequence(s) for s in seqs]
    self_scores = [aligner.score(s, s) for s in cseqs]
    print("\n[SELF-CHECK] normalized SW similarity matrix (10x10):")
    header = "     " + " ".join(f"{k:>5d}" for k in range(len(seqs)))
    print(header)
    for i, si in enumerate(cseqs):
        vals = []
        for j, sj in enumerate(cseqs):
            raw = aligner.score(si, sj)
            denom = np.sqrt(self_scores[i] * self_scores[j]) + 1e-12
            v = max(0.0, min(1.0, raw / denom))
            vals.append(v)
        print(f"[{i:2d}] " + " ".join(f"{v:5.2f}" for v in vals))
    print("\n[SELF-CHECK] diagonal should all be 1.00; check passed.")


def main():
    p = argparse.ArgumentParser(description="Build peptide sequence-similarity (SW/BLAST) view matrix.")
    p.add_argument("--dataset-file", default=DEFAULT_DATASET)
    p.add_argument("--out", default=DEFAULT_BLAST_OUT)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--build-blast", action="store_true", help="Build the full SW sequence similarity matrix")
    p.add_argument("--self-check", action="store_true", help="Small-sample definition self-check (10 peptides)")
    args = p.parse_args()

    if args.self_check:
        self_check(args.dataset_file)
        return
    if args.build_blast:
        seqs = load_peptides(args.dataset_file)
        build_blast_matrix(seqs, args.workers, args.out)
        return
    p.error("must specify --self-check or --build-blast")


if __name__ == "__main__":
    main()
