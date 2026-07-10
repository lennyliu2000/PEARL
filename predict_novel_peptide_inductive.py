#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use the latest main model (train_pearl_hgt_inductive.py / InductivePEARLHGT) to infer,
for a single brand-new peptide (not in the training set), its interaction probability
with all known protein targets, and output the ranking.

Difference from recommend_eval.py: recommend_eval can only rank peptides already in the
split graph; this script attaches an external novel peptide as an extra node into the
"deployment graph", supporting true cold-peptide extrapolation.

Inference pipeline (fully aligned with the graph-building / scoring / voting conventions at training time):
  1. Load a fold's checkpoint (contains training hyperparameters args + weights + best threshold)
  2. Build a "deployment graph" using all known data (or that fold's training set):
       - nodes: all known peptides + novel peptide + all targets
       - binds edges: known (training) positive peptide-target associations
       - pep_sim edges: peptide-peptide similarity (including novel peptide <-> known peptide, threshold alpha, topk)
       - prot_sim edges: disabled by default (disable_prot_sim=True, consistent with the main model)
  3. Encode the whole graph -> score (novel peptide, each target) -> sigmoid
  4. lambda nearest-neighbor voting enhancement (borrowing the known training labels of similar neighbor peptides for that target)
  5. Output Top-N targets + full-set CSV

The novel peptide needs two external inputs:
  --pep-t5-file : the novel peptide's protT5 embedding, a dict {sequence: tensor}, homologous
                  with the training protT5 (per-residue [L,1024] or already pooled [1024] both work,
                  load_vector_store will auto mean-pool).
  similarity    : the novel peptide vs all known peptides similarity. By default calls compute_novel_peptide_similarity()
                  -- the algorithm of this function must be consistent with the training similarity matrix (sim_adj_matrix.csv);
                  currently a placeholder implementation is provided, see the notes inside the function. You can also use --np-sim-file to directly feed a ready-made
                  similarity row CSV (index=known peptide sequences, single column=similarity), skipping on-the-fly computation.
"""
import argparse
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train_pearl_hgt_inductive as T


def replace_nonstandard_aa(sequence):
    """Consistent with Peptide_Processing.replace_nonstandard_aa: map non-standard amino acids to standard residues."""
    replacement_map = {'B': 'D', 'J': 'L', 'O': 'K', 'U': 'C', 'X': 'A', 'Z': 'E'}
    return ''.join(replacement_map.get(aa, aa) for aa in sequence)


def _peptide_morgan_fp(seq):
    """Peptide sequence -> Morgan fingerprint (radius=2, 2048 bit). Verified consistent with the sim_adj_matrix.csv convention."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSequence(replace_nonstandard_aa(seq))
    if mol is None:
        raise ValueError(f"RDKit cannot parse peptide sequence: {seq[:60]}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def compute_novel_peptide_similarity(novel_seq, known_seqs, **kwargs):
    """
    Return a 1D np.float32 array of length = len(known_seqs), the similarity of the novel peptide to each known peptide.

    Empirically verified: the values of the training matrix data/UMPPI/sim_adj_matrix.csv =
    the Tanimoto coefficient of the peptide molecules' Morgan fingerprints (radius=2, nBits=2048)
    (three sample pairs matched bit-exactly). The peptide sequence is first converted to a molecule via
    Chem.MolFromSequence, non-standard amino acids replaced per replace_nonstandard_aa.
    """
    from rdkit.Chem import DataStructs
    novel_fp = _peptide_morgan_fp(novel_seq)
    known_fps = [_peptide_morgan_fp(s) for s in known_seqs]
    sims = DataStructs.BulkTanimotoSimilarity(novel_fp, known_fps)
    return np.asarray(sims, dtype=np.float32)


def compute_novel_peptide_similarity_prott5(novel_feat, known_seqs, pep_t5_store):
    """protT5 cosine similarity: cosine of the novel peptide pooled protT5 with each known peptide pooled protT5.
    Consistent with generate_target_similarity.py / sim_adj_matrix_protT5.csv (dot product after L2 normalization).
    novel_feat: already pooled novel peptide protT5 [1024]; pep_t5_store: {seq:[1024]}."""
    nf = F.normalize(novel_feat.view(1, -1), dim=1)
    M = F.normalize(torch.stack([pep_t5_store[s] for s in known_seqs]), dim=1)
    sims = (nf @ M.T).view(-1).numpy()
    return np.asarray(sims, dtype=np.float32)


def load_novel_similarity_row(np_sim_file, known_seqs):
    """Load a ready-made novel-peptide similarity row from CSV. index=known peptide sequences, take the first column as similarity."""
    df = pd.read_csv(np_sim_file, index_col=0)
    df.index = df.index.astype(str)
    col = df.iloc[:, 0]
    missing = [s for s in known_seqs if s not in col.index]
    if missing:
        raise KeyError(f"--np-sim-file is missing similarity for {len(missing)} known peptides (example: {missing[:2]})")
    return col.loc[list(known_seqs)].to_numpy(dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Novel-peptide inductive inference (latest PEARL HGT model).")
    p.add_argument("--sequence", required=True, help="novel peptide amino acid sequence")
    p.add_argument("--name", default="novel_peptide", help="novel peptide name (used for output)")
    p.add_argument("--ckpt", required=True, help="fold{i}_best.pt (from a loose run of train_pearl_hgt_inductive)")
    p.add_argument("--pep-t5-file", required=True,
                   help="dict file containing the novel peptide protT5 embedding (.pt/.pkl), {sequence: [L,1024] or [1024]}")
    p.add_argument("--np-sim-file", default=None,
                   help="optional: novel peptide vs known peptide similarity row CSV (index=known peptide sequences, single similarity column). If not given, computed on the fly per --sim-method")
    p.add_argument("--sim-method", choices=["tanimoto", "prott5"], default="tanimoto",
                   help="novel peptide similarity algorithm: tanimoto(Morgan fingerprint)=old model convention; prott5(cosine)=protT5 edge model convention. Must match the peptide similarity matrix used to train the checkpoint")
    p.add_argument("--fold", type=int, default=0, help="which fold's training set to use as the known library (default 0)")
    p.add_argument("--split-file", default=None, help="split .pt (defaults to the one in the checkpoint args)")
    p.add_argument("--use-all-data", action="store_true",
                   help="ignore fold, use all positive samples of the entire dataset as the known library (more complete deployment graph)")
    p.add_argument("--top-n", type=int, default=20, help="print the top N targets")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-map", default="data/UMPPI/protein_mapping.pkl",
                   help="target sequence <-> name/ID mapping (pickle)")
    p.add_argument("--output", default=None, help="output path for full-set scores CSV")
    return p.parse_args()


def main():
    ea = parse_args()
    device = torch.device(ea.device if torch.cuda.is_available() else "cpu")
    novel_seq = ea.sequence.strip()

    ckpt = torch.load(ea.ckpt, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**ckpt["args"])
    args.device = str(device)
    threshold = float(ckpt.get("threshold", 0.5))
    print(f"[INFO] checkpoint: {ea.ckpt}")
    print(f"[INFO] feature_mode={getattr(args,'resolved_feature_mode',None)} "
          f"alpha={getattr(args,'alpha',None)} lambda_vote={getattr(args,'lambda_vote',None)} "
          f"vote_topk={getattr(args,'vote_topk_neighbors',None)} "
          f"disable_prot_sim={getattr(args,'disable_prot_sim',None)} threshold={threshold:.4f}")

    pep_store, prot_store, _ = T.build_feature_stores(args)
    pep_sim_df = T.load_similarity_table(args.pep_sim_file)
    prot_sim_df = None if getattr(args, "disable_prot_sim", True) else T.load_similarity_table(args.prot_sim_file)

    df = T.load_dataset(args.dataset_file)

    if ea.use_all_data:
        known_df = df.copy()
        print(f"[INFO] known library = all data ({len(known_df)} rows)")
    else:
        split_file = ea.split_file or args.split_file
        train_idx_list, valid_idx_list, test_idx_list = T.load_split_indices(split_file)
        train_idx = set(map(int, train_idx_list[ea.fold]))
        valid_idx = set(map(int, valid_idx_list[ea.fold]))
        eff = sorted(train_idx if getattr(args, "use_valid_as_train", False) else (train_idx - valid_idx))
        known_df = T.split_dataframe(df, eff)
        print(f"[INFO] known library = fold{ea.fold} training set ({len(known_df)} rows), split={split_file}")

    known_pep_seqs = list(pd.unique(known_df["pep_seq"].astype(str)))
    if novel_seq in set(known_pep_seqs):
        print(f"[WARN] the novel peptide sequence already exists in the known library, this is not a true cold-peptide inference.")

    if str(ea.pep_t5_file).endswith(".pkl"):
        with open(ea.pep_t5_file, "rb") as f:
            raw_novel = pickle.load(f, encoding="latin1")
    else:
        raw_novel = torch.load(ea.pep_t5_file, map_location="cpu")

    def pool_per_residue(t):
        t = torch.as_tensor(t, dtype=torch.float32)
        if t.ndim > 1:
            L = len(novel_seq)
            if t.shape[0] > L:
                t = t[:L]
            return t.mean(dim=0)
        return t.view(-1)

    if "per_residue" in raw_novel or "mean" in raw_novel:
        if "per_residue" in raw_novel:
            novel_feat = pool_per_residue(raw_novel["per_residue"])
            print("[INFO] novel peptide protT5: using per_residue[:seq_len].mean(0) (training convention)")
        else:
            novel_feat = torch.as_tensor(raw_novel["mean"], dtype=torch.float32).view(-1)
            print("[INFO] novel peptide protT5: using pre-stored mean vector")
    else:
        if novel_seq not in raw_novel:
            raise KeyError(f"novel peptide sequence not found in --pep-t5-file {novel_seq[:40]}... (contains {len(raw_novel)} entries)")
        novel_feat = pool_per_residue(raw_novel[novel_seq])

    novel_feat = novel_feat.view(-1)
    exp_dim = T.store_dim(pep_store)
    if novel_feat.numel() != exp_dim:
        raise ValueError(f"novel peptide feature dim {novel_feat.numel()} != training peptide feature dim {exp_dim} (protT5 must be homologous)")
    print(f"[INFO] novel peptide protT5 pooled: dim={novel_feat.numel()} norm={novel_feat.norm():.3f} "
          f"(training peptide pooled norm approx 2.9)")
    pep_store = dict(pep_store)
    pep_store[novel_seq] = novel_feat.clone()

    if ea.np_sim_file:
        sim_row = load_novel_similarity_row(ea.np_sim_file, known_pep_seqs)
        print(f"[INFO] novel peptide similarity from {ea.np_sim_file}")
    elif ea.sim_method == "prott5":
        pep_t5_store = T.load_vector_store(args.pep_t5_file)
        sim_row = compute_novel_peptide_similarity_prott5(novel_feat, known_pep_seqs, pep_t5_store)
        print(f"[INFO] novel peptide similarity: protT5 cosine (consistent with the protT5 edge model)")
    else:
        sim_row = compute_novel_peptide_similarity(novel_seq, known_pep_seqs)
        print(f"[INFO] novel peptide similarity: Tanimoto (Morgan fingerprint)")
    sim_row = np.asarray(sim_row, dtype=np.float32)
    print(f"[INFO] novel peptide similarity: max={sim_row.max():.4f} "
          f"#>alpha({getattr(args,'alpha',0.6)})={int((sim_row>float(getattr(args,'alpha',0.6))).sum())}")

    sim_full = pep_sim_df.reindex(index=known_pep_seqs, columns=known_pep_seqs)
    sim_full.loc[novel_seq] = 0.0
    sim_full[novel_seq] = 0.0
    row_series = pd.Series(sim_row, index=known_pep_seqs)
    sim_full.loc[novel_seq, known_pep_seqs] = row_series.values
    sim_full.loc[known_pep_seqs, novel_seq] = row_series.values
    sim_full.loc[novel_seq, novel_seq] = 1.0
    pep_sim_ext = sim_full.astype(np.float32)

    all_prot_seqs = list(prot_store.keys())
    node_rows = []
    for s in known_pep_seqs + [novel_seq]:
        node_rows.append((s, all_prot_seqs[0]))
    for t in all_prot_seqs:
        node_rows.append((novel_seq, t))
    node_df = pd.DataFrame(node_rows, columns=["pep_seq", "prot_seq"])

    binds_df = known_df[known_df["label"] == 1][["pep_seq", "prot_seq"]].astype(str).copy()

    eff_pep_thr = float(getattr(args, "effective_pep_threshold", None) or getattr(args, "alpha", 0.6))
    graph, pep_local, prot_local = T.build_graph(
        node_df, binds_df, pep_store, prot_store,
        pep_sim_ext, prot_sim_df,
        eff_pep_thr, getattr(args, "prot_threshold", 0.4),
        getattr(args, "pep_topk", 0), getattr(args, "prot_topk", 0),
        device=device,
    )
    print(f"[INFO] deployment graph: {T.graph_summary('deploy', graph)}")

    model = T.InductivePEARLHGT(
        pep_input_dim=T.store_dim(pep_store), prot_input_dim=T.store_dim(prot_store),
        hidden_dim=args.hidden_dim, mid_dim=args.mid_dim, out_dim=args.out_dim,
        dropout=args.dropout, pair_scorer=args.pair_scorer,
        post_cross_attn=getattr(args, "post_cross_attn", False),
        no_gnn=getattr(args, "no_gnn", False),
        no_residual=getattr(args, "no_residual", False),
        single_layer=getattr(args, "single_layer", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    novel_pid = pep_local[novel_seq]
    prot_ids = torch.tensor([prot_local[t] for t in all_prot_seqs], dtype=torch.long, device=device)
    dst = torch.full((len(all_prot_seqs),), novel_pid, dtype=torch.long, device=device)
    with torch.no_grad():
        h = model.encode(graph)
        logits = model.predict_pairs(h, prot_ids, dst)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    lam = float(getattr(args, "lambda_vote", 0.0))
    if lam > 0:
        vote_topk = int(getattr(args, "vote_topk_neighbors", 5))
        alpha = float(getattr(args, "alpha", 0.6))
        peptide_to_idx = pep_local
        protein_to_idx = prot_local
        known_label_matrix = T.build_known_label_lookup(known_df, peptide_to_idx, protein_to_idx)
        neigh_idx, neigh_w = T.precompute_vote_neighbors(pep_sim_ext.loc[list(pep_local.keys()), list(pep_local.keys())], alpha, vote_topk)
        split_df = pd.DataFrame({
            "pep_seq": [novel_seq] * len(all_prot_seqs),
            "prot_seq": all_prot_seqs,
            "label": [0] * len(all_prot_seqs),
        })
        probs = T.apply_lambda_vote(
            probs=probs, split_df=split_df, known_label_matrix=known_label_matrix,
            peptide_to_idx=peptide_to_idx, protein_to_idx=protein_to_idx,
            neighbor_idx=neigh_idx, neighbor_weights=neigh_w, lambda_vote=lam,
        )
        print(f"[INFO] lambda voting applied (lambda={lam}, topk={vote_topk})")

    seq2id = {}
    if ea.target_map and os.path.exists(ea.target_map):
        with open(ea.target_map, "rb") as f:
            m = pickle.load(f)
        if all(isinstance(k, str) for k in list(m.keys())[:5]):
            seq2id = m
        else:
            seq2id = {v: k for k, v in m.items()}

    order = np.argsort(-probs)
    out_df = pd.DataFrame({
        "rank": np.arange(1, len(all_prot_seqs) + 1),
        "target_id": [seq2id.get(all_prot_seqs[i], "") for i in order],
        "score": probs[order],
        "above_threshold": probs[order] >= threshold,
        "target_seq": [all_prot_seqs[i] for i in order],
    })
    out_path = ea.output or f"results/novel_{ea.name}_predictions.csv"
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_pos = int((probs >= threshold).sum())
    print(f"\n[Top-{ea.top_n} predicted targets] (threshold={threshold:.4f}, {n_pos}/{len(probs)} targets above threshold)")
    print(f"{'rank':>4} {'score':>8}  {'id':>6}  target_seq")
    for r in range(min(ea.top_n, len(order))):
        i = order[r]
        seq = all_prot_seqs[i]
        tid = str(seq2id.get(seq, ""))
        flag = "*" if probs[i] >= threshold else " "
        print(f"{r+1:>4} {probs[i]:>8.4f}{flag} {tid:>6}  {seq[:50]}...")
    print(f"\n[INFO] full scores saved -> {out_path}")


if __name__ == "__main__":
    main()
