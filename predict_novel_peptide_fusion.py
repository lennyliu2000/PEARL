#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use the multi-view fusion main model (train_pearl_hgt_inductive.py, a checkpoint
trained with --pep-sim-views) to infer, for a single brand-new peptide, its
interaction probability with all known protein targets and rank them.

Difference from predict_novel_peptide_inductive.py:
  The old script only supports a single similarity (tanimoto or prott5);
  this script supports S = a*S_embedding + b*S_sequence + c*S_interaction three-view
  fusion, fully aligned with the graph-building convention used when training the
  fusion checkpoint (per-view min-max normalization + convex combination).

The three-view similarity of the novel peptide (novel peptide vs each known peptide)
is computed separately:
  S_embedding  = protT5 pooled cosine (novel peptide protT5 vs known peptide protT5)
  S_sequence   = morgan Tanimoto or blast/SW local alignment (decided by the checkpoint's seq_sim_kind)
  S_interaction= binding profile of the novel peptide against known targets vs known peptide profile, jaccard/cosine
                 -- the novel peptide is cold, has no known binding -> this view's row is all zeros (consistent with the training cold behavior, no leakage)

The fused novel-peptide similarity row/column is expanded into the pep_sim matrix; the
remaining pipeline (deployment graph building / encoding / scoring / lambda voting)
reuses the conventions of train_pearl_hgt_inductive and predict_novel_peptide_inductive.

Usage:
  python predict_novel_peptide_fusion.py \
    --sequence NNTRKSIHLGPGRAFYATGDIIG --name myPep \
    --ckpt runs/sim_fusion_clu_20260705/new_protein_clu04_fused/fold0_best.pt \
    --pep-t5-file peptide_prott5.pt --top-n 20
"""
import argparse
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train_pearl_hgt_inductive as T
import predict_novel_peptide_inductive as P


def compute_novel_sequence_similarity(novel_seq, known_seqs, seq_sim_kind, pep_t5_store=None,
                                      novel_feat=None):
    """S_sequence view: the sequence-similarity row of the novel peptide vs each known peptide.
    seq_sim_kind=morgan -> Morgan fingerprint Tanimoto (reuses P.compute_novel_peptide_similarity);
    seq_sim_kind=blast  -> Smith-Waterman local alignment (BLOSUM62), aligned with build_peptide_similarity_views."""
    if seq_sim_kind == "morgan":
        return P.compute_novel_peptide_similarity(novel_seq, known_seqs)
    if seq_sim_kind == "blast":
        from Bio.Align import PairwiseAligner, substitution_matrices
        aa_replace = {"B": "D", "J": "L", "O": "K", "U": "C", "X": "A", "Z": "E"}
        clean = lambda s: "".join(aa_replace.get(a, a) for a in str(s).upper())
        aligner = PairwiseAligner()
        aligner.mode = "local"
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1
        nv = clean(novel_seq)
        self_nv = aligner.score(nv, nv)
        out = np.zeros(len(known_seqs), dtype=np.float32)
        for i, ks in enumerate(known_seqs):
            cks = clean(ks)
            raw = aligner.score(nv, cks)
            denom = np.sqrt(self_nv * aligner.score(cks, cks)) + 1e-12
            out[i] = max(0.0, min(1.0, raw / denom))
        return out
    raise ValueError(f"unsupported seq_sim_kind: {seq_sim_kind}")


def minmax_row(vec):
    """Min-max normalize a similarity row to [0,1] (same convention as the per-view
    normalization in the training minmax_normalize_sim, but at inference time there is
    only this one novel-peptide row, so its own min/max are used)."""
    vec = np.asarray(vec, dtype=np.float32)
    vmin, vmax = float(vec.min()), float(vec.max())
    span = vmax - vmin
    if span <= 1e-12:
        return np.zeros_like(vec)
    return (vec - vmin) / span


def build_novel_interaction_row(known_df, known_pep_seqs, metric):
    """S_interaction view: the novel peptide has no binding to known targets (cold) -> profile all zeros -> similarity row all zeros.
    Directly return an all-zero row (consistent with the training cold-peptide behavior, guaranteeing no leakage)."""
    return np.zeros(len(known_pep_seqs), dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="Novel-peptide inference with multi-view similarity fusion.")
    p.add_argument("--sequence", required=True, help="novel peptide amino acid sequence")
    p.add_argument("--name", default="novel_peptide")
    p.add_argument("--ckpt", required=True, help="fusion-trained fold{i}_best.pt")
    p.add_argument("--pep-t5-file", required=True, help="dict file containing the novel peptide protT5 (.pt/.pkl)")
    p.add_argument("--fold", type=int, default=0, help="which fold's training set to use as the known library")
    p.add_argument("--split-file", default=None)
    p.add_argument("--use-all-data", action="store_true", help="use all positive samples of the entire dataset as the known library")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-map", default="data/UMPPI/protein_mapping.pkl")
    p.add_argument("--output", default=None)
    ea = p.parse_args()

    device = torch.device(ea.device if torch.cuda.is_available() else "cpu")
    novel_seq = ea.sequence.strip()

    ckpt = torch.load(ea.ckpt, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**ckpt["args"])
    args.device = str(device)
    threshold = float(ckpt.get("threshold", 0.5))
    views = getattr(args, "fusion_views", None)
    weights = getattr(args, "fusion_weights", None)
    if not views:
        raise ValueError(
            "This checkpoint is not a fusion model (no fusion_views). For single-view use predict_novel_peptide_inductive.py"
        )
    seq_kind = getattr(args, "seq_sim_kind", "morgan")
    print(f"[INFO] checkpoint={ea.ckpt}")
    print(f"[INFO] fusion_views={views} weights={weights} seq_sim_kind={seq_kind} "
          f"alpha={getattr(args,'alpha',None)} lambda_vote={getattr(args,'lambda_vote',None)} "
          f"threshold={threshold:.4f}")

    pep_store, prot_store, _ = T.build_feature_stores(args)
    prot_sim_df = None if getattr(args, "disable_prot_sim", True) else T.load_similarity_table(args.prot_sim_file)
    df = T.load_dataset(args.dataset_file)

    view_files = {
        "embedding": getattr(args, "pep_sim_emb_file", T.DEFAULT_PEP_SIM_PROTT5),
        "sequence": getattr(args, "pep_sim_seq_file", None),
    }
    static_view_dfs = {}
    for v in views:
        if v in ("embedding", "sequence"):
            static_view_dfs[v] = T.load_similarity_table(view_files[v])

    if ea.use_all_data:
        known_df = df.copy()
    else:
        split_file = ea.split_file or args.split_file
        tr, va, te = T.load_split_indices(split_file)
        train_idx = set(map(int, tr[ea.fold])); valid_idx = set(map(int, va[ea.fold]))
        eff = sorted(train_idx if getattr(args, "use_valid_as_train", False) else (train_idx - valid_idx))
        known_df = T.split_dataframe(df, eff)
    known_pep_seqs = list(pd.unique(known_df["pep_seq"].astype(str)))
    print(f"[INFO] known library {len(known_df)} rows, {len(known_pep_seqs)} peptides")
    if novel_seq in set(known_pep_seqs):
        print("[WARN] the novel peptide is already in the known library, not a true cold-peptide inference.")

    if str(ea.pep_t5_file).endswith(".pkl"):
        with open(ea.pep_t5_file, "rb") as f:
            raw_novel = pickle.load(f, encoding="latin1")
    else:
        raw_novel = torch.load(ea.pep_t5_file, map_location="cpu")

    def pool(t):
        t = torch.as_tensor(t, dtype=torch.float32)
        if t.ndim > 1:
            L = len(novel_seq)
            if t.shape[0] > L:
                t = t[:L]
            return t.mean(dim=0)
        return t.view(-1)

    if isinstance(raw_novel, dict) and ("per_residue" in raw_novel or "mean" in raw_novel):
        novel_feat = pool(raw_novel["per_residue"]) if "per_residue" in raw_novel \
            else torch.as_tensor(raw_novel["mean"], dtype=torch.float32).view(-1)
    else:
        if novel_seq not in raw_novel:
            raise KeyError(f"novel peptide sequence not found in --pep-t5-file {novel_seq[:40]}...")
        novel_feat = pool(raw_novel[novel_seq])
    novel_feat = novel_feat.view(-1)
    exp_dim = T.store_dim(pep_store)
    if novel_feat.numel() != exp_dim:
        raise ValueError(f"novel peptide feature dim {novel_feat.numel()} != training peptide dim {exp_dim}")
    pep_store = dict(pep_store)
    pep_store[novel_seq] = novel_feat.clone()
    print(f"[INFO] novel peptide protT5 dim={novel_feat.numel()} norm={novel_feat.norm():.3f}")

    pep_t5_store_for_cos = None
    fused_row = np.zeros(len(known_pep_seqs), dtype=np.float32)
    print("[INFO] per-view novel peptide similarity rows:")
    for v, w in zip(views, weights):
        if v == "embedding":
            if pep_t5_store_for_cos is None:
                pep_t5_store_for_cos = T.load_vector_store(args.pep_t5_file)
            row = P.compute_novel_peptide_similarity_prott5(novel_feat, known_pep_seqs, pep_t5_store_for_cos)
        elif v == "sequence":
            row = compute_novel_sequence_similarity(novel_seq, known_pep_seqs, seq_kind)
        elif v == "interaction":
            row = build_novel_interaction_row(known_df, known_pep_seqs, getattr(args, "interaction_sim_metric", "jaccard"))
        else:
            raise ValueError(f"unknown view {v}")
        row_n = minmax_row(row)
        fused_row += float(w) * row_n
        print(f"    {v:12s} w={w:.2f}  raw[max={float(np.max(row)):.3f}] "
              f"norm[max={float(row_n.max()):.3f}] #>alpha={int((row_n>float(getattr(args,'alpha',0.3))).sum())}")

    fuse_components = []
    for v in views:
        if v == "interaction":
            fuse_components.append(
                T.build_interaction_similarity(known_df, known_pep_seqs,
                                               getattr(args, "interaction_sim_metric", "jaccard"))
            )
        else:
            fuse_components.append(static_view_dfs[v].reindex(index=known_pep_seqs, columns=known_pep_seqs).fillna(0.0))
    known_fused = T.fuse_similarity_views(fuse_components, weights)

    sim_full = known_fused.copy()
    sim_full.loc[novel_seq] = 0.0
    sim_full[novel_seq] = 0.0
    row_series = pd.Series(fused_row, index=known_pep_seqs)
    sim_full.loc[novel_seq, known_pep_seqs] = row_series.values
    sim_full.loc[known_pep_seqs, novel_seq] = row_series.values
    sim_full.loc[novel_seq, novel_seq] = 0.0
    pep_sim_ext = sim_full.astype(np.float32)

    all_prot_seqs = list(prot_store.keys())
    node_rows = [(s, all_prot_seqs[0]) for s in known_pep_seqs + [novel_seq]]
    node_rows += [(novel_seq, t) for t in all_prot_seqs]
    node_df = pd.DataFrame(node_rows, columns=["pep_seq", "prot_seq"])
    binds_df = known_df[known_df["label"] == 1][["pep_seq", "prot_seq"]].astype(str).copy()

    eff_thr = float(getattr(args, "effective_pep_threshold", None) or getattr(args, "alpha", 0.3))
    graph, pep_local, prot_local = T.build_graph(
        node_df, binds_df, pep_store, prot_store, pep_sim_ext, prot_sim_df,
        eff_thr, getattr(args, "prot_threshold", 0.4),
        getattr(args, "pep_topk", 20), getattr(args, "prot_topk", 0), device=device,
    )
    print(f"[INFO] deployment graph: {T.graph_summary('deploy', graph)}")

    model = T.InductivePEARLHGT(
        pep_input_dim=T.store_dim(pep_store), prot_input_dim=T.store_dim(prot_store),
        hidden_dim=args.hidden_dim, mid_dim=args.mid_dim, out_dim=args.out_dim,
        dropout=args.dropout, pair_scorer=args.pair_scorer,
        post_cross_attn=getattr(args, "post_cross_attn", False),
        no_gnn=getattr(args, "no_gnn", False), no_residual=getattr(args, "no_residual", False),
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
        alpha = float(getattr(args, "alpha", 0.3))
        known_label_matrix = T.build_known_label_lookup(known_df, pep_local, prot_local)
        neigh_idx, neigh_w = T.precompute_vote_neighbors(
            pep_sim_ext.loc[list(pep_local.keys()), list(pep_local.keys())], alpha, vote_topk)
        split_df = pd.DataFrame({"pep_seq": [novel_seq]*len(all_prot_seqs),
                                 "prot_seq": all_prot_seqs, "label": [0]*len(all_prot_seqs)})
        probs = T.apply_lambda_vote(probs=probs, split_df=split_df, known_label_matrix=known_label_matrix,
                                    peptide_to_idx=pep_local, protein_to_idx=prot_local,
                                    neighbor_idx=neigh_idx, neighbor_weights=neigh_w, lambda_vote=lam)
        print(f"[INFO] lambda voting applied (lambda={lam}, topk={vote_topk})")

    seq2id = {}
    if ea.target_map and os.path.exists(ea.target_map):
        with open(ea.target_map, "rb") as f:
            m = pickle.load(f)
        seq2id = m if all(isinstance(k, str) for k in list(m.keys())[:5]) else {v: k for k, v in m.items()}

    order = np.argsort(-probs)
    out_df = pd.DataFrame({
        "rank": np.arange(1, len(all_prot_seqs)+1),
        "target_id": [seq2id.get(all_prot_seqs[i], "") for i in order],
        "score": probs[order],
        "above_threshold": probs[order] >= threshold,
        "target_seq": [all_prot_seqs[i] for i in order],
    })
    out_path = ea.output or f"results/novel_fusion_{ea.name}_predictions.csv"
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n_pos = int((probs >= threshold).sum())
    print(f"\n[Top-{ea.top_n} predicted targets] (threshold={threshold:.4f}, {n_pos}/{len(probs)} above threshold)")
    print(f"{'rank':>4} {'score':>8}  {'id':>6}  target_seq")
    for r in range(min(ea.top_n, len(order))):
        i = order[r]
        flag = "*" if probs[i] >= threshold else " "
        print(f"{r+1:>4} {probs[i]:>8.4f}{flag} {str(seq2id.get(all_prot_seqs[i],'')):>6}  {all_prot_seqs[i][:50]}...")
    print(f"\n[INFO] full scores saved -> {out_path}")


if __name__ == "__main__":
    main()
