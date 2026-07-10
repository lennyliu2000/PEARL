"""PEARL model definition.

Two-layer heterogeneous GraphSAGE encoder + pairwise scoring head. Given a
heterogeneous graph (peptide / receptor node types, binds / pep_sim / prot_sim
edge types), it encodes node embeddings and then scores (receptor, peptide) pairs.

Extracted from the original train script; the class interface and weight format
are kept unchanged (existing checkpoints load directly).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


class InductivePEARLHGT(nn.Module):
    def __init__(
        self,
        pep_input_dim,
        prot_input_dim,
        hidden_dim,
        mid_dim,
        out_dim,
        dropout,
        pair_scorer,
        post_cross_attn=False,
        no_gnn=False,
        no_residual=False,
        single_layer=False,
    ):
        super().__init__()
        self.peptide_proj = nn.Linear(pep_input_dim, hidden_dim)
        self.receptor_proj = nn.Linear(prot_input_dim, hidden_dim)
        self.pair_scorer = pair_scorer
        self.post_cross_attn = post_cross_attn
        self.no_gnn = no_gnn
        self.no_residual = no_residual
        self.single_layer = single_layer

        self.pep_bn1 = nn.BatchNorm1d(hidden_dim)
        self.prot_bn1 = nn.BatchNorm1d(hidden_dim)
        self.pep_bn2 = nn.BatchNorm1d(mid_dim)
        self.prot_bn2 = nn.BatchNorm1d(mid_dim)
        self.pep_bn3 = nn.BatchNorm1d(out_dim)
        self.prot_bn3 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

        if not no_gnn:
            self.conv1 = dglnn.HeteroGraphConv(
                {
                    "pep_sim": dglnn.SAGEConv(hidden_dim, mid_dim, aggregator_type="mean"),
                    "prot_sim": dglnn.SAGEConv(hidden_dim, mid_dim, aggregator_type="mean"),
                    "binds": dglnn.SAGEConv(hidden_dim, mid_dim, aggregator_type="mean"),
                },
                aggregate="mean",
            )
            if not single_layer:
                self.conv2 = dglnn.HeteroGraphConv(
                    {
                        "pep_sim": dglnn.SAGEConv(mid_dim, out_dim, aggregator_type="gcn"),
                        "prot_sim": dglnn.SAGEConv(mid_dim, out_dim, aggregator_type="gcn"),
                        "binds": dglnn.SAGEConv(mid_dim, out_dim, aggregator_type="gcn"),
                    },
                    aggregate="mean",
                )

        if not no_gnn:
            self.pep_res_proj1 = nn.Linear(hidden_dim, mid_dim)
            self.prot_res_proj1 = nn.Linear(hidden_dim, mid_dim)
            if not single_layer:
                self.pep_res_proj2 = nn.Linear(mid_dim, out_dim)
                self.prot_res_proj2 = nn.Linear(mid_dim, out_dim)

        if no_gnn:
            embed_dim = hidden_dim
        elif single_layer:
            embed_dim = mid_dim
        else:
            embed_dim = out_dim

        self.out_proj = nn.Linear(embed_dim, out_dim) if embed_dim != out_dim else None
        self.out_bn = nn.BatchNorm1d(out_dim) if embed_dim != out_dim else None

        if pair_scorer == "full":
            pred_input_dim = out_dim * 5
        elif pair_scorer == "attn_full":
            pred_input_dim = out_dim
        else:
            pred_input_dim = out_dim * 2
        self.edge_predictor = nn.Sequential(
            nn.Linear(pred_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, graph):
        graph = graph.to(next(self.parameters()).device)
        peptide_feat = self.dropout(F.relu(self.peptide_proj(graph.nodes["peptide"].data["pre_feat"])))
        receptor_feat = self.dropout(F.relu(self.receptor_proj(graph.nodes["receptor"].data["pre_feat"])))

        peptide_feat = self.pep_bn1(peptide_feat)
        receptor_feat = self.prot_bn1(receptor_feat)
        h = {"peptide": peptide_feat, "receptor": receptor_feat}

        if self.no_gnn:
            if self.out_proj is not None:
                h = {
                    "peptide": self.out_bn(self.dropout(F.relu(self.out_proj(h["peptide"])))),
                    "receptor": self.out_bn(self.dropout(F.relu(self.out_proj(h["receptor"])))),
                }
            return h

        h1 = self.conv1(graph, h)
        if self.no_residual:
            h1 = {
                "peptide": self.pep_bn2(self.dropout(F.relu(h1["peptide"]))),
                "receptor": self.prot_bn2(self.dropout(F.relu(h1["receptor"]))),
            }
        else:
            h1 = {
                "peptide": self.pep_bn2(
                    self.dropout(F.relu(h1["peptide"] + self.pep_res_proj1(h["peptide"])))
                ),
                "receptor": self.prot_bn2(
                    self.dropout(F.relu(h1["receptor"] + self.prot_res_proj1(h["receptor"])))
                ),
            }

        if self.single_layer:
            if self.out_proj is not None:
                h2 = {
                    "peptide": self.out_bn(self.dropout(F.relu(self.out_proj(h1["peptide"])))),
                    "receptor": self.out_bn(self.dropout(F.relu(self.out_proj(h1["receptor"])))),
                }
            else:
                h2 = h1
        else:
            h2 = self.conv2(graph, h1)
            if self.no_residual:
                h2 = {
                    "peptide": self.pep_bn3(self.dropout(F.relu(h2["peptide"]))),
                    "receptor": self.prot_bn3(self.dropout(F.relu(h2["receptor"]))),
                }
            else:
                h2 = {
                    "peptide": self.pep_bn3(
                        self.dropout(F.relu(h2["peptide"] + self.pep_res_proj2(h1["peptide"])))
                    ),
                    "receptor": self.prot_bn3(
                        self.dropout(F.relu(h2["receptor"] + self.prot_res_proj2(h1["receptor"])))
                    ),
                }

        if self.post_cross_attn:
            interaction = torch.matmul(h2["peptide"], h2["receptor"].T)
            attn_peptide = torch.matmul(interaction.softmax(dim=-1), h2["receptor"])
            attn_receptor = torch.matmul(interaction.softmax(dim=0).T, h2["peptide"])
            h2 = {
                "peptide": h2["peptide"] + attn_peptide,
                "receptor": h2["receptor"] + attn_receptor,
            }
        return h2

    def predict_pairs(self, h, src_ids, dst_ids):
        src_h = h["receptor"][src_ids]
        dst_h = h["peptide"][dst_ids]
        pair_terms = [
            src_h,
            dst_h,
            src_h * dst_h,
            torch.abs(src_h - dst_h),
            (src_h - dst_h) ** 2,
        ]
        if self.pair_scorer == "full":
            feats = torch.cat(pair_terms, dim=1)
        elif self.pair_scorer == "attn_full":
            fusion = torch.stack(pair_terms, dim=1)
            attn_weight = torch.softmax(torch.sum(fusion, dim=2), dim=1).unsqueeze(-1)
            feats = torch.sum(fusion * attn_weight, dim=1)
        else:
            feats = torch.cat([src_h, dst_h], dim=1)
        return self.edge_predictor(feats).squeeze(-1)
