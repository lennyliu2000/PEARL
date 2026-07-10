import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")

import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_DATASET = "data/Dataset_all_balanced_new_2.tsv"
DEFAULT_SPLIT = "data/clu_thre_0.4/new_peptide_balanced"
DEFAULT_PEP_SIM = "data/UMPPI/sim_adj_matrix.csv"
DEFAULT_PEP_SIM_PROTT5 = "data/UMPPI/sim_adj_matrix_protT5.csv"
DEFAULT_PEP_SIM_BLAST = "data/UMPPI/sim_adj_matrix_blast.csv"
DEFAULT_PROT_SIM = "data/UMPPI/target_sim_adj_matrix.csv"
DEFAULT_PEP_T5 = "features/peptide_protT5.pt"
DEFAULT_PROT_T5 = "features/target_protT5.pt"
DEFAULT_HGT_PEP_T5 = "features/peptide_protT5.pt"
DEFAULT_HGT_PROT_T5 = "features/target_protT5.pt"
DEFAULT_PEP_GRAPH = "features/UMPPI/all_graph_features.pt"
DEFAULT_PEP_EVO = "features/UMPPI/all_evolutionary_features.pt"
DEFAULT_PROT_DISORDER = "features/UMPPI/all_ProInD_features.pt"
DEFAULT_PROT_EVO = "features/UMPPI/all_ProEvo_features.pt"
DEFAULT_PEP_DENSE = "data/feature/peptide_dense_feature_dict"
DEFAULT_PROT_DENSE = "data/feature/protein_dense_feature_dict"
DEFAULT_OUTPUT = "runs/pearl_hgt_inductive"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict inductive PEARL trainer that borrows HGT-PepPI's graph protocol."
    )
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT)
    parser.add_argument("--pep-sim-file", default=DEFAULT_PEP_SIM)
    parser.add_argument("--prot-sim-file", default=DEFAULT_PROT_SIM)
    parser.add_argument("--pep-t5-file", default=DEFAULT_PEP_T5)
    parser.add_argument("--prot-t5-file", default=DEFAULT_PROT_T5)
    parser.add_argument("--pep-graph-file", default=DEFAULT_PEP_GRAPH)
    parser.add_argument("--pep-evo-file", default=DEFAULT_PEP_EVO)
    parser.add_argument("--prot-disorder-file", default=DEFAULT_PROT_DISORDER)
    parser.add_argument("--prot-evo-file", default=DEFAULT_PROT_EVO)
    parser.add_argument("--pep-dense-file", default=DEFAULT_PEP_DENSE)
    parser.add_argument("--prot-dense-file", default=DEFAULT_PROT_DENSE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--mid-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--pep-threshold", type=float, default=0.6203)
    parser.add_argument("--prot-threshold", type=float, default=0.4104)
    parser.add_argument("--pep-topk", type=int, default=20)
    parser.add_argument("--prot-topk", type=int, default=20)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional explicit peptide-similarity threshold. Overrides --pep-threshold when set.",
    )
    parser.add_argument(
        "--lambda-vote",
        type=float,
        default=0.0,
        help="Optional peptide-neighbor vote fusion weight applied at validation/test time.",
    )
    parser.add_argument(
        "--vote-topk-neighbors",
        type=int,
        default=5,
        help="Top-k peptide neighbors used by lambda vote. 0 keeps all thresholded neighbors.",
    )
    parser.add_argument("--disable-pep-sim", action="store_true")
    parser.add_argument("--disable-prot-sim", action="store_true")
    parser.add_argument(
        "--pep-sim-views",
        default=None,
        help=(
            "Comma-separated similarity views to fuse, e.g. 'embedding,sequence,interaction'. "
            "When unset, falls back to the single --pep-sim-file (zero behavior change)."
        ),
    )
    parser.add_argument(
        "--pep-sim-emb-file",
        default=DEFAULT_PEP_SIM_PROTT5,
        help="protT5 cosine peptide similarity CSV (S_embedding view).",
    )
    parser.add_argument(
        "--pep-sim-seq-file",
        default=None,
        help="Sequence-similarity CSV for S_sequence view. Defaults resolved by --seq-sim-kind.",
    )
    parser.add_argument(
        "--seq-sim-kind",
        choices=["morgan", "blast"],
        default="morgan",
        help="Which sequence-similarity source backs the S_sequence view.",
    )
    parser.add_argument(
        "--sim-fusion-weights",
        default=None,
        help=(
            "Comma-separated non-negative weights aligned with --pep-sim-views, "
            "e.g. '0.5,0.3,0.2'. Auto-normalized to sum 1. Defaults to equal weights."
        ),
    )
    parser.add_argument(
        "--interaction-sim-metric",
        choices=["cosine", "jaccard"],
        default="jaccard",
        help="Similarity metric over per-fold training binding profiles for S_interaction.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=[
            "auto",
            "pearl_dense",
            "prot_t5_only",
            "hgt_match_t5",
            "no_t5_256",
            "t5_graph_disorder_1152",
            "t5_evo_1152",
            "trimodal_1280",
        ],
        default="auto",
        help=(
            "Node feature recipe. "
            "'pearl_dense' = local ProtT5 + dense features (1027/1047); "
            "'prot_t5_only' = local ProtT5 only (1024/1024); "
            "'hgt_match_t5' = HGT-PepPI ProtT5 pkl only (1024/1024); "
            "'no_t5_256' = peptide graph+evo / receptor disorder+evo, no ProtT5 (256/256); "
            "'t5_graph_disorder_1152' = local ProtT5 + peptide graph / receptor disorder (1152/1152); "
            "'t5_evo_1152' = local ProtT5 + peptide/receptor evolutionary features (1152/1152); "
            "'trimodal_1280' = graph/evo/ProtT5 concatenation (1280/1280)."
        ),
    )
    parser.add_argument(
        "--pair-scorer",
        choices=["full", "concat", "attn_full"],
        default="full",
        help=(
            "Pair scoring features. "
            "'full' uses interaction/difference terms; "
            "'concat' uses only node embeddings; "
            "'attn_full' applies old-PEARL-style attention fusion over full pair terms."
        ),
    )
    parser.add_argument(
        "--post-cross-attn",
        action="store_true",
        help="Apply old-PEARL-style peptide-receptor co-attention after graph encoding.",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--fold", type=int, default=-1, help="-1 means all folds")
    parser.add_argument(
        "--match-hgt-conditions",
        action="store_true",
        help="Align graph protocol and feature source as closely as possible to the HGT baseline.",
    )
    parser.add_argument(
        "--hgt-parity",
        action="store_true",
        help="Strict PEARL-vs-HGT parity: HGT ProtT5 only, train positives only, no eval binds, split-only eval graphs, full thresholded similarity graph, and valid kept inside train supervision.",
    )
    parser.add_argument(
        "--strict-compare",
        action="store_true",
        help="Traditional model-comparison protocol: HGT ProtT5 only, train positives only, no eval binds, split-only eval graphs, full thresholded similarity graph, and independent validation.",
    )
    parser.add_argument(
        "--train-binds-mode",
        choices=["positive", "all"],
        default="positive",
        help="Which training pairs are inserted into the train graph as binds edges.",
    )
    parser.add_argument(
        "--eval-binds-mode",
        choices=["strict", "positive", "all"],
        default="strict",
        help="Which eval-fold pairs are inserted into val/test graphs as binds edges.",
    )
    parser.add_argument(
        "--eval-node-scope",
        choices=["backbone", "split_only"],
        default="backbone",
        help="Whether val/test graphs contain train nodes as a backbone or only nodes from that split.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["auc", "aupr", "mcc"],
        default="auc",
        help="Validation metric used for model selection.",
    )
    parser.add_argument(
        "--use-valid-as-train",
        action="store_true",
        help="Keep valid indices inside the training supervision, matching the original split semantics.",
    )
    parser.add_argument(
        "--no-dense-features",
        action="store_true",
        help="Disable PEARL dense features and use ProtT5 only.",
    )
    parser.add_argument("--no-gnn", action="store_true", help="Skip graph convolution; pure MLP baseline.")
    parser.add_argument("--no-residual", action="store_true", help="Remove residual connections in GNN.")
    parser.add_argument("--single-layer", action="store_true", help="Use only the first SAGEConv layer.")
    args = parser.parse_args()
    enabled_protocols = sum(
        int(flag)
        for flag in (
            args.match_hgt_conditions,
            args.hgt_parity,
            args.strict_compare,
        )
    )
    if enabled_protocols > 1:
        raise ValueError(
            "--match-hgt-conditions, --hgt-parity, and --strict-compare are mutually exclusive"
        )
    if args.match_hgt_conditions:
        args.train_binds_mode = "all"
        args.eval_binds_mode = "all"
        args.eval_node_scope = "split_only"
        args.selection_metric = "auc"
        args.use_valid_as_train = False
        args.pep_topk = 0
        args.prot_topk = 0
        if args.feature_mode in {"auto", "hgt_match_t5"}:
            args.pep_t5_file = DEFAULT_HGT_PEP_T5
            args.prot_t5_file = DEFAULT_HGT_PROT_T5
            args.no_dense_features = True
    if args.hgt_parity:
        args.train_binds_mode = "positive"
        args.eval_binds_mode = "strict"
        args.eval_node_scope = "split_only"
        args.selection_metric = "auc"
        args.use_valid_as_train = True
        args.pep_topk = 0
        args.prot_topk = 0
        if args.feature_mode == "auto":
            args.feature_mode = "hgt_match_t5"
        if args.feature_mode == "hgt_match_t5":
            args.pep_t5_file = DEFAULT_HGT_PEP_T5
            args.prot_t5_file = DEFAULT_HGT_PROT_T5
            args.no_dense_features = True
    if args.strict_compare:
        args.train_binds_mode = "positive"
        args.eval_binds_mode = "strict"
        args.eval_node_scope = "split_only"
        args.selection_metric = "auc"
        args.use_valid_as_train = False
        args.pep_topk = 0
        args.prot_topk = 0
        if args.feature_mode == "auto":
            args.feature_mode = "hgt_match_t5"
        if args.feature_mode == "hgt_match_t5":
            args.pep_t5_file = DEFAULT_HGT_PEP_T5
            args.prot_t5_file = DEFAULT_HGT_PROT_T5
            args.no_dense_features = True
    if args.lambda_vote < 0.0 or args.lambda_vote > 1.0:
        raise ValueError("--lambda-vote must be in [0, 1]")
    if args.vote_topk_neighbors < 0:
        raise ValueError("--vote-topk-neighbors must be >= 0")
    if args.lambda_vote > 0 and args.disable_pep_sim:
        raise ValueError("--lambda-vote requires peptide similarity; remove --disable-pep-sim")
    args.effective_pep_threshold = (
        float(args.alpha) if args.alpha is not None else float(args.pep_threshold)
    )
    args.resolved_feature_mode = resolve_feature_mode(args)

    args.fusion_views = None
    args.fusion_weights = None
    if args.pep_sim_views:
        views = [v.strip() for v in args.pep_sim_views.split(",") if v.strip()]
        valid_views = {"embedding", "sequence", "interaction"}
        bad = [v for v in views if v not in valid_views]
        if bad:
            raise ValueError(f"Unknown --pep-sim-views entries {bad}; allowed: {sorted(valid_views)}")
        if not views:
            raise ValueError("--pep-sim-views resolved to an empty list")
        if len(set(views)) != len(views):
            raise ValueError(f"--pep-sim-views has duplicates: {views}")
        if args.sim_fusion_weights:
            weights = [float(w) for w in args.sim_fusion_weights.split(",")]
            if len(weights) != len(views):
                raise ValueError(
                    f"--sim-fusion-weights count ({len(weights)}) != views count ({len(views)})"
                )
        else:
            weights = [1.0] * len(views)
        if any(w < 0 for w in weights):
            raise ValueError("--sim-fusion-weights must be non-negative")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("--sim-fusion-weights must sum to a positive value")
        weights = [w / total for w in weights]
        args.fusion_views = views
        args.fusion_weights = weights
        if args.disable_pep_sim:
            raise ValueError("--pep-sim-views is incompatible with --disable-pep-sim")
        if "sequence" in views and args.pep_sim_seq_file is None:
            args.pep_sim_seq_file = (
                DEFAULT_PEP_SIM_BLAST if args.seq_sim_kind == "blast" else DEFAULT_PEP_SIM
            )
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_file):
    df = pd.read_csv(
        dataset_file,
        sep="\t",
        usecols=["prot_seq", "pep_seq", "label"],
    )
    df["prot_seq"] = df["prot_seq"].astype(str)
    df["pep_seq"] = df["pep_seq"].astype(str)
    df["label"] = df["label"].astype(np.int64)
    return df


def load_split_indices(split_file):
    train_idx_list, valid_idx_list, test_idx_list = torch.load(
        split_file,
        map_location="cpu",
        encoding="latin1",
    )
    return train_idx_list, valid_idx_list, test_idx_list


def load_similarity_table(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.astype(np.float32)


def minmax_normalize_sim(df):
    """Min-max normalize a symmetric similarity matrix into [0, 1], zeroing the diagonal.

    Uses off-diagonal min/max so a matrix already in [0,1] (e.g. Tanimoto) is largely
    preserved while a cosine matrix in [-1,1] is rescaled onto a comparable range.
    """
    values = df.to_numpy(dtype=np.float32, copy=True)
    np.fill_diagonal(values, np.nan)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        out = np.zeros_like(values)
        np.fill_diagonal(out, 0.0)
        return pd.DataFrame(out, index=df.index, columns=df.columns)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = vmax - vmin
    if span <= 1e-12:
        normed = np.zeros_like(values)
    else:
        normed = (values - vmin) / span
    normed = np.nan_to_num(normed, nan=0.0)
    np.fill_diagonal(normed, 0.0)
    return pd.DataFrame(normed.astype(np.float32), index=df.index, columns=df.columns)


def build_interaction_similarity(train_df, peptides, metric, eval_peptides=None):
    """Peptide-peptide similarity from *training-only* binding profiles (leakage-safe).

    Each peptide is represented by the set/vector of protein targets it binds among
    training positives. Similarity is Jaccard (set overlap) or cosine over those
    binary profiles. Peptides absent from the training positives (e.g. cold test
    peptides) get an all-zero profile -> all-zero similarity row, which is the
    correct non-leaking behavior.

    peptides: full ordered index (all nodes) for the returned square matrix.
    eval_peptides: optional dict {"valid": [...], "test": [...]} (or a flat iterable)
        for the [LEAK-CHECK] print. In a cold-peptide split the *test* count must be
        0; valid peptides may legitimately overlap training when valid is carved from
        train, so they are reported separately rather than flagged.
    Returns a DataFrame indexed/columned by `peptides` with values in [0, 1].
    """
    peptides = list(map(str, peptides))
    pep_to_row = {seq: i for i, seq in enumerate(peptides)}

    pos_df = train_df[train_df["label"] == 1]
    proteins = list(pd.unique(pos_df["prot_seq"].astype(str)))
    prot_to_col = {seq: j for j, seq in enumerate(proteins)}

    n, m = len(peptides), len(proteins)
    profile = np.zeros((n, m), dtype=np.float32)
    for pep, prot in zip(pos_df["pep_seq"].astype(str), pos_df["prot_seq"].astype(str)):
        r = pep_to_row.get(pep)
        if r is not None:
            profile[r, prot_to_col[prot]] = 1.0

    if m == 0:
        sim = np.zeros((n, n), dtype=np.float32)
    elif metric == "cosine":
        norms = np.linalg.norm(profile, axis=1, keepdims=True)
        safe = np.where(norms > 0, norms, 1.0)
        unit = profile / safe
        sim = (unit @ unit.T).astype(np.float32)
    elif metric == "jaccard":
        inter = (profile @ profile.T).astype(np.float32)
        counts = profile.sum(axis=1)
        union = counts[:, None] + counts[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported interaction-sim metric: {metric}")

    np.fill_diagonal(sim, 0.0)
    sim = np.nan_to_num(sim, nan=0.0)

    if eval_peptides is not None:
        if isinstance(eval_peptides, dict):
            groups = eval_peptides
        else:
            groups = {"eval": eval_peptides}
        for group_name, seqs in groups.items():
            subset = [str(s) for s in seqs if str(s) in pep_to_row]
            if not subset:
                continue
            rows = np.array([pep_to_row[s] for s in subset], dtype=np.int64)
            nz_profile = int((profile[rows].sum(axis=1) > 0).sum())
            nz_simrow = int((np.abs(sim[rows]).sum(axis=1) > 0).sum())
            note = ""
            if group_name == "test" and nz_profile > 0:
                note = "  <-- WARNING: cold-peptide test leakage!"
            elif group_name == "valid":
                note = "  (valid may overlap train by split design)"
            print(
                f"[LEAK-CHECK] {group_name}: n={len(subset)} "
                f"nonzero_train_profile={nz_profile} nonzero_sim_row={nz_simrow}{note}"
            )

    return pd.DataFrame(sim, index=peptides, columns=peptides)


def fuse_similarity_views(components, weights):
    """Convex combination of aligned, min-max-normalized similarity DataFrames.

    components: list of DataFrames sharing the same index/columns.
    weights: list of non-negative weights summing to 1 (already normalized upstream).
    """
    if not components:
        raise ValueError("fuse_similarity_views needs at least one component")
    base_index = components[0].index
    base_cols = components[0].columns
    fused = np.zeros((len(base_index), len(base_cols)), dtype=np.float32)
    for weight, comp in zip(weights, components):
        comp = comp.reindex(index=base_index, columns=base_cols).fillna(0.0)
        normed = minmax_normalize_sim(comp)
        fused += weight * normed.to_numpy(dtype=np.float32)
    np.fill_diagonal(fused, 0.0)
    return pd.DataFrame(fused, index=base_index, columns=base_cols)


def load_vector_store(path):
    if str(path).endswith(".pkl"):
        with open(path, "rb") as f:
            raw_store = pickle.load(f, encoding="latin1")
        store = {}
        for seq, vec in raw_store.items():
            tensor = torch.as_tensor(vec, dtype=torch.float32)
            if tensor.ndim > 1:
                tensor = tensor.mean(dim=0)
            else:
                tensor = tensor.view(-1)
            store[str(seq)] = tensor.detach().clone().view(-1)
        return store

    raw_store = torch.load(path, map_location="cpu")
    return {
        str(seq): torch.as_tensor(vec, dtype=torch.float32).detach().clone().view(-1)
        for seq, vec in raw_store.items()
    }


def load_dense_store(path):
    with open(path, "rb") as f:
        raw_store = pickle.load(f, encoding="latin1")

    dense_store = {}
    for seq, value in raw_store.items():
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=0)
        else:
            tensor = tensor.view(-1)
        dense_store[str(seq)] = tensor.detach().clone().view(-1)
    return dense_store


def combine_feature_stores(base_store, dense_store=None):
    if dense_store is None:
        return base_store

    merged = {}
    for seq, vector in base_store.items():
        if seq not in dense_store:
            raise KeyError(f"Missing dense feature for sequence: {seq[:80]}")
        merged[seq] = torch.cat([vector, dense_store[seq].view(-1)], dim=0)
    return merged


def concatenate_feature_stores(store_specs):
    base_name, base_store = store_specs[0]
    merged = {}
    for seq, vector in base_store.items():
        parts = [vector.view(-1)]
        for name, store in store_specs[1:]:
            if seq not in store:
                raise KeyError(f"Missing {name} feature for sequence: {seq[:80]}")
            parts.append(store[seq].view(-1))
        merged[seq] = torch.cat(parts, dim=0)
    return merged


def resolve_feature_mode(args):
    if args.feature_mode != "auto":
        return args.feature_mode
    if args.match_hgt_conditions:
        return "hgt_match_t5"
    return "prot_t5_only" if args.no_dense_features else "pearl_dense"


def store_dim(store):
    return next(iter(store.values())).numel()


def build_feature_stores(args):
    mode = args.resolved_feature_mode
    if mode == "pearl_dense":
        pep_t5_store = load_vector_store(args.pep_t5_file)
        prot_t5_store = load_vector_store(args.prot_t5_file)
        pep_dense_store = load_dense_store(args.pep_dense_file)
        prot_dense_store = load_dense_store(args.prot_dense_file)
        pep_store = combine_feature_stores(pep_t5_store, pep_dense_store)
        prot_store = combine_feature_stores(prot_t5_store, prot_dense_store)
        feature_blocks = {
            "peptide": [("prot_t5", store_dim(pep_t5_store)), ("dense", store_dim(pep_dense_store))],
            "receptor": [("prot_t5", store_dim(prot_t5_store)), ("dense", store_dim(prot_dense_store))],
        }
        return pep_store, prot_store, feature_blocks

    if mode in {"prot_t5_only", "hgt_match_t5"}:
        pep_store = load_vector_store(args.pep_t5_file)
        prot_store = load_vector_store(args.prot_t5_file)
        feature_blocks = {
            "peptide": [("prot_t5", store_dim(pep_store))],
            "receptor": [("prot_t5", store_dim(prot_store))],
        }
        return pep_store, prot_store, feature_blocks

    if mode == "no_t5_256":
        pep_graph_store = load_vector_store(args.pep_graph_file)
        pep_evo_store = load_vector_store(args.pep_evo_file)
        prot_disorder_store = load_vector_store(args.prot_disorder_file)
        prot_evo_store = load_vector_store(args.prot_evo_file)
        pep_store = concatenate_feature_stores(
            [
                ("pep_graph", pep_graph_store),
                ("pep_evo", pep_evo_store),
            ]
        )
        prot_store = concatenate_feature_stores(
            [
                ("prot_disorder", prot_disorder_store),
                ("prot_evo", prot_evo_store),
            ]
        )
        feature_blocks = {
            "peptide": [
                ("graph", store_dim(pep_graph_store)),
                ("evo", store_dim(pep_evo_store)),
            ],
            "receptor": [
                ("disorder", store_dim(prot_disorder_store)),
                ("evo", store_dim(prot_evo_store)),
            ],
        }
        return pep_store, prot_store, feature_blocks

    if mode == "t5_graph_disorder_1152":
        pep_graph_store = load_vector_store(args.pep_graph_file)
        pep_t5_store = load_vector_store(args.pep_t5_file)
        prot_disorder_store = load_vector_store(args.prot_disorder_file)
        prot_t5_store = load_vector_store(args.prot_t5_file)
        pep_store = concatenate_feature_stores(
            [
                ("pep_graph", pep_graph_store),
                ("pep_t5", pep_t5_store),
            ]
        )
        prot_store = concatenate_feature_stores(
            [
                ("prot_disorder", prot_disorder_store),
                ("prot_t5", prot_t5_store),
            ]
        )
        feature_blocks = {
            "peptide": [
                ("graph", store_dim(pep_graph_store)),
                ("prot_t5", store_dim(pep_t5_store)),
            ],
            "receptor": [
                ("disorder", store_dim(prot_disorder_store)),
                ("prot_t5", store_dim(prot_t5_store)),
            ],
        }
        return pep_store, prot_store, feature_blocks

    if mode == "t5_evo_1152":
        pep_evo_store = load_vector_store(args.pep_evo_file)
        pep_t5_store = load_vector_store(args.pep_t5_file)
        prot_evo_store = load_vector_store(args.prot_evo_file)
        prot_t5_store = load_vector_store(args.prot_t5_file)
        pep_store = concatenate_feature_stores(
            [
                ("pep_evo", pep_evo_store),
                ("pep_t5", pep_t5_store),
            ]
        )
        prot_store = concatenate_feature_stores(
            [
                ("prot_evo", prot_evo_store),
                ("prot_t5", prot_t5_store),
            ]
        )
        feature_blocks = {
            "peptide": [
                ("evo", store_dim(pep_evo_store)),
                ("prot_t5", store_dim(pep_t5_store)),
            ],
            "receptor": [
                ("evo", store_dim(prot_evo_store)),
                ("prot_t5", store_dim(prot_t5_store)),
            ],
        }
        return pep_store, prot_store, feature_blocks

    if mode == "trimodal_1280":
        pep_graph_store = load_vector_store(args.pep_graph_file)
        pep_evo_store = load_vector_store(args.pep_evo_file)
        pep_t5_store = load_vector_store(args.pep_t5_file)
        prot_disorder_store = load_vector_store(args.prot_disorder_file)
        prot_evo_store = load_vector_store(args.prot_evo_file)
        prot_t5_store = load_vector_store(args.prot_t5_file)
        pep_store = concatenate_feature_stores(
            [
                ("pep_graph", pep_graph_store),
                ("pep_evo", pep_evo_store),
                ("pep_t5", pep_t5_store),
            ]
        )
        prot_store = concatenate_feature_stores(
            [
                ("prot_disorder", prot_disorder_store),
                ("prot_evo", prot_evo_store),
                ("prot_t5", prot_t5_store),
            ]
        )
        feature_blocks = {
            "peptide": [
                ("graph", store_dim(pep_graph_store)),
                ("evo", store_dim(pep_evo_store)),
                ("prot_t5", store_dim(pep_t5_store)),
            ],
            "receptor": [
                ("disorder", store_dim(prot_disorder_store)),
                ("evo", store_dim(prot_evo_store)),
                ("prot_t5", store_dim(prot_t5_store)),
            ],
        }
        return pep_store, prot_store, feature_blocks

    raise ValueError(f"Unsupported feature mode: {mode}")


def ensure_coverage(df, pep_store, prot_store, pep_sim_df, prot_sim_df):
    peptide_set = set(df["pep_seq"].astype(str))
    protein_set = set(df["prot_seq"].astype(str))

    missing_pep_feat = sorted(seq for seq in peptide_set if seq not in pep_store)
    missing_prot_feat = sorted(seq for seq in protein_set if seq not in prot_store)
    missing_pep_sim = sorted(seq for seq in peptide_set if seq not in pep_sim_df.index)
    missing_prot_sim = sorted(seq for seq in protein_set if seq not in prot_sim_df.index)

    if missing_pep_feat:
        raise KeyError(f"Missing peptide features for {len(missing_pep_feat)} sequences")
    if missing_prot_feat:
        raise KeyError(f"Missing protein features for {len(missing_prot_feat)} sequences")
    if missing_pep_sim:
        raise KeyError(f"Missing peptide similarity rows for {len(missing_pep_sim)} sequences")
    if missing_prot_sim:
        raise KeyError(f"Missing protein similarity rows for {len(missing_prot_sim)} sequences")


def zscore_features(features):
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (features - mean) / std


def split_dataframe(df, row_indices):
    split_df = df.iloc[list(row_indices)][["pep_seq", "prot_seq", "label"]].copy()
    split_df.reset_index(drop=True, inplace=True)
    return split_df


def select_binds_df(split_df, mode):
    if mode == "positive":
        return split_df[split_df["label"] == 1].reset_index(drop=True)
    if mode == "all":
        return split_df.reset_index(drop=True)
    raise ValueError(f"Unsupported binds mode: {mode}")


def compose_eval_binds_df(train_df, eval_df, eval_mode):
    return compose_eval_binds_df_with_scope(train_df, eval_df, eval_mode, "backbone")


def build_eval_node_df(train_df, eval_df, node_scope):
    if node_scope == "backbone":
        return pd.concat([train_df, eval_df], ignore_index=True)
    if node_scope == "split_only":
        return eval_df.reset_index(drop=True)
    raise ValueError(f"Unsupported eval node scope: {node_scope}")


def compose_eval_binds_df_with_scope(train_df, eval_df, eval_mode, node_scope):
    train_pos_df = select_binds_df(train_df, "positive")
    if node_scope == "backbone":
        if eval_mode == "strict":
            return train_pos_df
        if eval_mode == "positive":
            eval_pos_df = select_binds_df(eval_df, "positive")
            return pd.concat([train_pos_df, eval_pos_df], ignore_index=True)
        if eval_mode == "all":
            return pd.concat([train_pos_df, eval_df.reset_index(drop=True)], ignore_index=True)
    elif node_scope == "split_only":
        if eval_mode == "strict":
            return eval_df.iloc[0:0].copy()
        if eval_mode == "positive":
            return select_binds_df(eval_df, "positive")
        if eval_mode == "all":
            return eval_df.reset_index(drop=True)
    raise ValueError(f"Unsupported eval binds mode: {eval_mode}")


def build_similarity_edges(sim_df, sequences, threshold, topk):
    if not sequences:
        return [], [], []

    sub = sim_df.loc[sequences, sequences].to_numpy(dtype=np.float32, copy=True)
    np.fill_diagonal(sub, 0.0)

    if sub.shape[0] == 1:
        return [], [], []

    topk = int(topk)
    if topk > 0 and topk < sub.shape[1]:
        kth = sub.shape[1] - topk
        candidate_idx = np.argpartition(sub, kth=kth, axis=1)[:, -topk:]
        row_idx = np.repeat(np.arange(sub.shape[0]), topk)
        col_idx = candidate_idx.reshape(-1)
        values = sub[row_idx, col_idx]
        keep = values > threshold
        src = row_idx[keep]
        dst = col_idx[keep]
        weights = values[keep]
    else:
        src, dst = np.where(sub > threshold)
        weights = sub[src, dst]

    return src.tolist(), dst.tolist(), weights.tolist()


def build_graph(
    node_df,
    binds_edge_df,
    pep_store,
    prot_store,
    pep_sim_df,
    prot_sim_df,
    pep_threshold,
    prot_threshold,
    pep_topk,
    prot_topk,
    device,
):
    peptides = pd.Index(pd.unique(node_df["pep_seq"].astype(str)))
    proteins = pd.Index(pd.unique(node_df["prot_seq"].astype(str)))

    pep_local = {seq: idx for idx, seq in enumerate(peptides)}
    prot_local = {seq: idx for idx, seq in enumerate(proteins)}

    pep_feat = torch.stack([pep_store[seq] for seq in peptides], dim=0)
    prot_feat = torch.stack([prot_store[seq] for seq in proteins], dim=0)
    pep_feat = zscore_features(pep_feat)
    prot_feat = zscore_features(prot_feat)

    if pep_sim_df is None:
        pep_src, pep_dst, pep_weight = [], [], []
    else:
        pep_src, pep_dst, pep_weight = build_similarity_edges(
            pep_sim_df, peptides.tolist(), pep_threshold, pep_topk
        )
    if prot_sim_df is None:
        prot_src, prot_dst, prot_weight = [], [], []
    else:
        prot_src, prot_dst, prot_weight = build_similarity_edges(
            prot_sim_df, proteins.tolist(), prot_threshold, prot_topk
        )

    bind_src = [prot_local[seq] for seq in binds_edge_df["prot_seq"].astype(str)]
    bind_dst = [pep_local[seq] for seq in binds_edge_df["pep_seq"].astype(str)]

    graph = dgl.heterograph(
        {
            ("receptor", "binds", "peptide"): (bind_src, bind_dst),
            ("peptide", "pep_sim", "peptide"): (pep_src, pep_dst),
            ("receptor", "prot_sim", "receptor"): (prot_src, prot_dst),
        },
        num_nodes_dict={"peptide": len(peptides), "receptor": len(proteins)},
    )
    graph.nodes["peptide"].data["pre_feat"] = pep_feat
    graph.nodes["receptor"].data["pre_feat"] = prot_feat
    if pep_weight:
        graph.edges[("peptide", "pep_sim", "peptide")].data["weight"] = torch.tensor(
            pep_weight, dtype=torch.float32
        )
    if prot_weight:
        graph.edges[("receptor", "prot_sim", "receptor")].data["weight"] = torch.tensor(
            prot_weight, dtype=torch.float32
        )
    return graph.to(device), pep_local, prot_local


def build_edge_tensors(split_df, pep_local, prot_local, device):
    src = torch.tensor(
        [prot_local[seq] for seq in split_df["prot_seq"].astype(str)],
        dtype=torch.long,
        device=device,
    )
    dst = torch.tensor(
        [pep_local[seq] for seq in split_df["pep_seq"].astype(str)],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(
        split_df["label"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    return src, dst, labels


def build_known_label_lookup(split_df, peptide_to_idx, protein_to_idx):
    known = np.full((len(peptide_to_idx), len(protein_to_idx)), -1.0, dtype=np.float32)
    pep_ids = [peptide_to_idx[seq] for seq in split_df["pep_seq"].astype(str)]
    prot_ids = [protein_to_idx[seq] for seq in split_df["prot_seq"].astype(str)]
    labels = split_df["label"].to_numpy(dtype=np.float32)
    known[np.asarray(pep_ids, dtype=np.int64), np.asarray(prot_ids, dtype=np.int64)] = labels
    return known


def precompute_vote_neighbors(pep_sim_df, alpha: float, topk: int):
    sim = pep_sim_df.to_numpy(dtype=np.float32, copy=True)
    np.fill_diagonal(sim, 0.0)
    sim[sim < alpha] = 0.0

    neighbors = []
    weights = []
    for row_idx in range(sim.shape[0]):
        row = sim[row_idx]
        cand_idx = np.flatnonzero(row > 0)
        if cand_idx.size == 0:
            neighbors.append(np.empty((0,), dtype=np.int64))
            weights.append(np.empty((0,), dtype=np.float32))
            continue
        if topk > 0 and cand_idx.size > topk:
            top_idx = np.argpartition(row[cand_idx], -topk)[-topk:]
            cand_idx = cand_idx[top_idx]
        neighbors.append(cand_idx.astype(np.int64, copy=False))
        weights.append(row[cand_idx].astype(np.float32, copy=True))
    return neighbors, weights


def apply_lambda_vote(
    probs,
    split_df,
    known_label_matrix,
    peptide_to_idx,
    protein_to_idx,
    neighbor_idx,
    neighbor_weights,
    lambda_vote,
):
    if lambda_vote <= 0:
        return probs

    fused = probs.copy()
    pep_ids = [peptide_to_idx[seq] for seq in split_df["pep_seq"].astype(str)]
    prot_ids = [protein_to_idx[seq] for seq in split_df["prot_seq"].astype(str)]
    for idx, (pep_id, prot_id) in enumerate(zip(pep_ids, prot_ids)):
        neigh = neighbor_idx[pep_id]
        if neigh.size == 0:
            continue
        labels = known_label_matrix[neigh, prot_id]
        known_mask = labels >= 0
        if not np.any(known_mask):
            continue
        local_weights = neighbor_weights[pep_id][known_mask]
        if local_weights.sum() <= 0:
            continue
        vote = float(np.dot(local_weights, labels[known_mask]) / (local_weights.sum() + 1e-8))
        fused[idx] = (1.0 - lambda_vote) * probs[idx] + lambda_vote * vote
    return fused


def find_best_threshold_by_mcc(y_true, y_pred_probs, thresholds=None):
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 500)

    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        score = matthews_corrcoef(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def calculate_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    aupr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "aupr": float(aupr),
        "mcc": float(mcc),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


from model import InductivePEARLHGT


def evaluate_split(model, graph, src, dst, labels, loss_fn, threshold=None, split_df=None, vote_context=None):
    model.eval()
    with torch.no_grad():
        h = model.encode(graph)
        logits = model.predict_pairs(h, src, dst)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        if vote_context is not None and vote_context["lambda_vote"] > 0:
            if split_df is None:
                raise ValueError("split_df is required when lambda vote is enabled")
            probs = apply_lambda_vote(
                probs=probs,
                split_df=split_df,
                known_label_matrix=vote_context["known_label_matrix"],
                peptide_to_idx=vote_context["peptide_to_idx"],
                protein_to_idx=vote_context["protein_to_idx"],
                neighbor_idx=vote_context["neighbor_idx"],
                neighbor_weights=vote_context["neighbor_weights"],
                lambda_vote=vote_context["lambda_vote"],
            )
        y_true = labels.detach().cpu().numpy()
        if threshold is None:
            threshold = find_best_threshold_by_mcc(y_true, probs)
        metrics = calculate_metrics(y_true, probs, threshold)
        metrics["loss"] = float(loss_fn(logits, labels).item())
    return metrics, float(threshold)


def graph_summary(name, graph):
    return {
        "name": name,
        "num_peptides": int(graph.num_nodes("peptide")),
        "num_receptors": int(graph.num_nodes("receptor")),
        "bind_edges": int(graph.num_edges(("receptor", "binds", "peptide"))),
        "pep_sim_edges": int(graph.num_edges(("peptide", "pep_sim", "peptide"))),
        "prot_sim_edges": int(graph.num_edges(("receptor", "prot_sim", "receptor"))),
    }


def train_fold(
    fold_id,
    args,
    train_df,
    valid_df,
    test_df,
    pep_store,
    prot_store,
    pep_sim_df,
    prot_sim_df,
    vote_assets,
    output_dir,
    pep_input_dim,
    prot_input_dim,
):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loss_fn = nn.BCEWithLogitsLoss()

    train_binds_df = select_binds_df(train_df, args.train_binds_mode)
    valid_binds_df = compose_eval_binds_df_with_scope(
        train_df, valid_df, args.eval_binds_mode, args.eval_node_scope
    )
    test_binds_df = compose_eval_binds_df_with_scope(
        train_df, test_df, args.eval_binds_mode, args.eval_node_scope
    )
    valid_node_df = build_eval_node_df(train_df, valid_df, args.eval_node_scope)
    test_node_df = build_eval_node_df(train_df, test_df, args.eval_node_scope)
    active_pep_sim_df = None if args.disable_pep_sim else pep_sim_df
    active_prot_sim_df = None if args.disable_prot_sim else prot_sim_df
    vote_context = None
    if vote_assets is not None:
        vote_context = {
            "lambda_vote": float(args.lambda_vote),
            "peptide_to_idx": vote_assets["peptide_to_idx"],
            "protein_to_idx": vote_assets["protein_to_idx"],
            "neighbor_idx": vote_assets["neighbor_idx"],
            "neighbor_weights": vote_assets["neighbor_weights"],
            "known_label_matrix": build_known_label_lookup(
                train_df,
                vote_assets["peptide_to_idx"],
                vote_assets["protein_to_idx"],
            ),
        }

    train_graph, train_pep_local, train_prot_local = build_graph(
        train_df,
        train_binds_df,
        pep_store,
        prot_store,
        active_pep_sim_df,
        active_prot_sim_df,
        args.effective_pep_threshold,
        args.prot_threshold,
        args.pep_topk,
        args.prot_topk,
        device=device,
    )
    valid_graph, valid_pep_local, valid_prot_local = build_graph(
        valid_node_df,
        valid_binds_df,
        pep_store,
        prot_store,
        active_pep_sim_df,
        active_prot_sim_df,
        args.effective_pep_threshold,
        args.prot_threshold,
        args.pep_topk,
        args.prot_topk,
        device=device,
    )
    test_graph, test_pep_local, test_prot_local = build_graph(
        test_node_df,
        test_binds_df,
        pep_store,
        prot_store,
        active_pep_sim_df,
        active_prot_sim_df,
        args.effective_pep_threshold,
        args.prot_threshold,
        args.pep_topk,
        args.prot_topk,
        device=device,
    )

    train_src, train_dst, train_labels = build_edge_tensors(train_df, train_pep_local, train_prot_local, device)
    valid_src, valid_dst, valid_labels = build_edge_tensors(valid_df, valid_pep_local, valid_prot_local, device)
    test_src, test_dst, test_labels = build_edge_tensors(test_df, test_pep_local, test_prot_local, device)

    train_stats = graph_summary(f"fold{fold_id}_train", train_graph)
    valid_stats = graph_summary(f"fold{fold_id}_valid", valid_graph)
    test_stats = graph_summary(f"fold{fold_id}_test", test_graph)
    print(json.dumps(train_stats, ensure_ascii=True))
    print(json.dumps(valid_stats, ensure_ascii=True))
    print(json.dumps(test_stats, ensure_ascii=True))

    model = InductivePEARLHGT(
        pep_input_dim=pep_input_dim,
        prot_input_dim=prot_input_dim,
        hidden_dim=args.hidden_dim,
        mid_dim=args.mid_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
        pair_scorer=args.pair_scorer,
        post_cross_attn=args.post_cross_attn,
        no_gnn=args.no_gnn,
        no_residual=args.no_residual,
        single_layer=args.single_layer,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {
        "fold": fold_id,
        "epoch": 0,
        "threshold": 0.5,
        "selection_metric": args.selection_metric,
        "selection_value": -1.0,
        "val_metrics": None,
        "test_metrics": None,
        "graph_stats": {
            "train": train_stats,
            "valid": valid_stats,
            "test": test_stats,
        },
    }
    patience_counter = 0
    ckpt_path = output_dir / f"fold{fold_id}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        h = model.encode(train_graph)
        logits = model.predict_pairs(h, train_src, train_dst)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue

        val_metrics, threshold = evaluate_split(
            model,
            valid_graph,
            valid_src,
            valid_dst,
            valid_labels,
            loss_fn,
            split_df=valid_df,
            vote_context=vote_context,
        )
        score = float(val_metrics[args.selection_metric])

        if epoch == 1 or epoch % args.log_every == 0:
            train_metrics, _ = evaluate_split(
                model,
                train_graph,
                train_src,
                train_dst,
                train_labels,
                loss_fn,
                threshold=threshold,
                split_df=train_df,
                vote_context=vote_context,
            )
            print(
                f"fold={fold_id} epoch={epoch:03d} "
                f"train_loss={loss.item():.4f} train_auc={train_metrics['auc']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_aupr={val_metrics['aupr']:.4f} "
                f"val_mcc={val_metrics['mcc']:.4f}"
            )

        if score > best["selection_value"] + 1e-6:
            test_metrics, _ = evaluate_split(
                model,
                test_graph,
                test_src,
                test_dst,
                test_labels,
                loss_fn,
                threshold=threshold,
                split_df=test_df,
                vote_context=vote_context,
            )
            best.update(
                {
                    "epoch": epoch,
                    "threshold": float(threshold),
                    "selection_value": score,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
            )
            torch.save(
                {
                    "fold": fold_id,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "threshold": threshold,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "graph_stats": best["graph_stats"],
                },
                ckpt_path,
            )
            patience_counter = 0
            print(
                f"fold={fold_id} epoch={epoch:03d} new_best "
                f"select={args.selection_metric}:{score:.4f} "
                f"test_auc={test_metrics['auc']:.4f} "
                f"test_aupr={test_metrics['aupr']:.4f} "
                f"test_mcc={test_metrics['mcc']:.4f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"fold={fold_id} early_stop epoch={epoch}")
                break

    return best


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    start = time.time()
    print(f"loading dataset from {args.dataset_file}")
    df = load_dataset(args.dataset_file)
    print(
        f"dataset rows={len(df)} pos={int((df['label'] == 1).sum())} "
        f"neg={int((df['label'] == 0).sum())}"
    )

    print(f"loading split file from {args.split_file}")
    train_idx_list, valid_idx_list, test_idx_list = load_split_indices(args.split_file)
    num_folds = len(train_idx_list)
    print(f"split folds={num_folds}")

    print("loading feature stores")
    pep_store, prot_store, feature_blocks = build_feature_stores(args)
    print(f"resolved feature mode={args.resolved_feature_mode}")
    for node_type, blocks in feature_blocks.items():
        parts = " + ".join(f"{name}:{dim}" for name, dim in blocks)
        print(f"{node_type} feature blocks={parts}")
    pep_input_dim = next(iter(pep_store.values())).numel()
    prot_input_dim = next(iter(prot_store.values())).numel()
    print(f"peptide feature dim={pep_input_dim} protein feature dim={prot_input_dim}")

    print("loading similarity tables")
    prot_sim_df = load_similarity_table(args.prot_sim_file)
    dataset_peptides = pd.Index(pd.unique(df["pep_seq"].astype(str)))
    dataset_proteins = pd.Index(pd.unique(df["prot_seq"].astype(str)))
    prot_sim_df = prot_sim_df.loc[dataset_proteins, dataset_proteins]

    fusion_static_components = None
    if args.fusion_views is None:
        pep_sim_df = load_similarity_table(args.pep_sim_file)
        ensure_coverage(df, pep_store, prot_store, pep_sim_df, prot_sim_df)
        pep_sim_df = pep_sim_df.loc[dataset_peptides, dataset_peptides]
        print(f"peptide sim shape={pep_sim_df.shape} protein sim shape={prot_sim_df.shape}")
    else:
        component_files = {
            "embedding": args.pep_sim_emb_file,
            "sequence": args.pep_sim_seq_file,
        }
        static = []
        for view in args.fusion_views:
            if view == "interaction":
                continue
            table = load_similarity_table(component_files[view])
            missing = [s for s in dataset_peptides if s not in table.index]
            if missing:
                raise KeyError(
                    f"similarity view '{view}' ({component_files[view]}) missing "
                    f"{len(missing)} dataset peptides"
                )
            static.append((view, table.loc[dataset_peptides, dataset_peptides]))
        fusion_static_components = static
        ensure_coverage(
            df, pep_store, prot_store,
            static[0][1] if static else pd.DataFrame(index=dataset_peptides, columns=dataset_peptides),
            prot_sim_df,
        )
        if "interaction" not in args.fusion_views:
            pep_sim_df = fuse_similarity_views(
                [comp for _, comp in static], args.fusion_weights
            )
            print(
                f"fused peptide sim (static) views={args.fusion_views} "
                f"weights={[round(w, 3) for w in args.fusion_weights]} shape={pep_sim_df.shape}"
            )
        else:
            pep_sim_df = None
            print(
                f"fused peptide sim is per-fold (interaction view) views={args.fusion_views} "
                f"weights={[round(w, 3) for w in args.fusion_weights]}"
            )
    print("feature/sim coverage check passed")

    def build_vote_assets(active_pep_sim_df):
        if args.lambda_vote <= 0:
            return None
        neighbor_idx, neighbor_weights = precompute_vote_neighbors(
            pep_sim_df=active_pep_sim_df,
            alpha=args.effective_pep_threshold,
            topk=args.vote_topk_neighbors,
        )
        assets = {
            "peptide_to_idx": {seq: idx for idx, seq in enumerate(active_pep_sim_df.index.astype(str).tolist())},
            "protein_to_idx": {seq: idx for idx, seq in enumerate(prot_sim_df.index.astype(str).tolist())},
            "neighbor_idx": neighbor_idx,
            "neighbor_weights": neighbor_weights,
        }
        nonempty = int(sum(item.size > 0 for item in assets["neighbor_idx"]))
        print(
            f"lambda vote enabled lambda={args.lambda_vote:.2f} "
            f"alpha={args.effective_pep_threshold:.4f} topk={args.vote_topk_neighbors} "
            f"nonempty_peptides={nonempty}"
        )
        return assets

    vote_assets = build_vote_assets(pep_sim_df) if pep_sim_df is not None else None

    def fuse_for_fold(train_df, eval_peptides):
        """Assemble the per-fold fused peptide similarity when an interaction view is used."""
        components = []
        static_map = dict(fusion_static_components or [])
        for view in args.fusion_views:
            if view == "interaction":
                components.append(
                    build_interaction_similarity(
                        train_df,
                        dataset_peptides,
                        args.interaction_sim_metric,
                        eval_peptides=eval_peptides,
                    )
                )
            else:
                components.append(static_map[view])
        return fuse_similarity_views(components, args.fusion_weights)

    if args.fold == -1:
        folds = range(num_folds)
    else:
        if args.fold < 0 or args.fold >= num_folds:
            raise ValueError(f"--fold must be -1 or one of 0..{num_folds - 1}")
        folds = [args.fold]

    results = []
    for fold_id in folds:
        train_idx = set(map(int, train_idx_list[fold_id]))
        valid_idx = set(map(int, valid_idx_list[fold_id]))
        test_idx = list(map(int, test_idx_list[fold_id]))
        effective_train = sorted(train_idx if args.use_valid_as_train else (train_idx - valid_idx))
        effective_valid = sorted(valid_idx)

        if not effective_train:
            raise RuntimeError(f"fold {fold_id} has empty effective training set")

        train_df = split_dataframe(df, effective_train)
        valid_df = split_dataframe(df, effective_valid)
        test_df = split_dataframe(df, test_idx)

        print(
            f"fold={fold_id} train_rows={len(train_df)} valid_rows={len(valid_df)} "
            f"test_rows={len(test_df)} use_valid_as_train={args.use_valid_as_train}"
        )

        if pep_sim_df is None:
            eval_peptides = {
                "valid": pd.unique(valid_df["pep_seq"].astype(str)),
                "test": pd.unique(test_df["pep_seq"].astype(str)),
            }
            fold_pep_sim_df = fuse_for_fold(train_df, eval_peptides)
            print(
                f"fold={fold_id} fused peptide sim (per-fold) shape={fold_pep_sim_df.shape}"
            )
            fold_vote_assets = build_vote_assets(fold_pep_sim_df)
        else:
            fold_pep_sim_df = pep_sim_df
            fold_vote_assets = vote_assets

        best = train_fold(
            fold_id,
            args,
            train_df,
            valid_df,
            test_df,
            pep_store,
            prot_store,
            fold_pep_sim_df,
            prot_sim_df,
            fold_vote_assets,
            output_dir,
            pep_input_dim,
            prot_input_dim,
        )
        results.append(best)

    summary = {
        "args": vars(args),
        "elapsed_seconds": time.time() - start,
        "folds": results,
    }
    aucs = [item["test_metrics"]["auc"] for item in results if item["test_metrics"]]
    auprs = [item["test_metrics"]["aupr"] for item in results if item["test_metrics"]]
    mccs = [item["test_metrics"]["mcc"] for item in results if item["test_metrics"]]
    if aucs:
        summary["test_mean"] = {
            "auc": float(np.mean(aucs)),
            "aupr": float(np.mean(auprs)),
            "mcc": float(np.mean(mccs)),
            "auc_std": float(np.std(aucs)),
            "aupr_std": float(np.std(auprs)),
            "mcc_std": float(np.std(mccs)),
        }
        print(
            "test_mean "
            f"auc={summary['test_mean']['auc']:.4f} "
            f"aupr={summary['test_mean']['aupr']:.4f} "
            f"mcc={summary['test_mean']['mcc']:.4f}"
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"summary saved to {summary_path}")


if __name__ == "__main__":
    main()
