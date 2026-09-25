"""Typed evidence encoders and soft-token projection.

    text and linearized records   PubMedBERT
    knowledge-graph subgraphs     R-GCN
    proteins                      ESM-2
    protein-ligand complexes      EquiformerV2 with Uni-Mol ligand features

The pooled structural embedding is invariant to global rigid motions. Each item
embedding is projected into the LLM's input space as a soft token, alongside
the linearized evidence text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import Evidence, Relation, SourceType

RELATION_VOCAB: Dict[str, int] = {r.value: i for i, r in enumerate(Relation)}


@dataclass
class EncoderConfig:
    text_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    esm_model: str = "facebook/esm2_t33_650M_UR50D"
    text_dim: int = 768
    esm_dim: int = 1280
    graph_dim: int = 256
    equi_dim: int = 128
    unimol_dim: int = 512
    struct_dim: int = 256
    llm_dim: int = 4096
    rgcn_layers: int = 2
    num_bases: int = 8


# --------------------------------------------------------------------------- #
# Text and curated records
# --------------------------------------------------------------------------- #

class TextEncoder(nn.Module):
    """[CLS] embedding of the rendered item."""

    def __init__(self, backbone: nn.Module, tokenizer) -> None:
        super().__init__()
        self.backbone, self.tok = backbone, tokenizer

    def forward(self, items: Sequence[Evidence]) -> torch.Tensor:
        enc = self.tok([e.render() for e in items], padding=True, truncation=True,
                       max_length=256, return_tensors="pt").to(
            next(self.backbone.parameters()).device)
        return self.backbone(**enc).last_hidden_state[:, 0]


# --------------------------------------------------------------------------- #
# Knowledge-graph subgraphs
# --------------------------------------------------------------------------- #

class RGCNLayer(nn.Module):
    """h_i' = ReLU(W_0 h_i + sum_r sum_{j in N_r(i)} |N_r(i)|^-1 W_r h_j), basis-decomposed W_r."""

    def __init__(self, dim: int, num_rel: int, num_bases: int) -> None:
        super().__init__()
        self.bases = nn.Parameter(torch.randn(num_bases, dim, dim) * dim ** -0.5)
        self.coef = nn.Parameter(torch.randn(num_rel, num_bases) / num_bases)
        self.self_loop = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        W = torch.einsum("rb,bio->rio", self.coef, self.bases)            # (R, d, d)
        src, dst = edge_index
        msg = torch.bmm(h[src].unsqueeze(1), W[edge_type]).squeeze(1)
        deg = torch.zeros(h.size(0), W.size(0), device=h.device, dtype=h.dtype)
        deg.index_put_((dst, edge_type), torch.ones_like(dst, dtype=h.dtype), accumulate=True)
        norm = deg[dst, edge_type].clamp_min(1.0).reciprocal()
        agg = torch.zeros_like(h).index_add_(0, dst, msg * norm.unsqueeze(-1))
        return F.relu(self.self_loop(h) + agg)


def build_subgraph(items: Sequence[Evidence]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Nodes, edge index (with reverse edges) and edge types of the retrieved subgraph."""
    nodes: Dict[str, int] = {}
    names: List[str] = []

    def node(x) -> int:
        key = x.cid or x.name.lower()
        if key not in nodes:
            nodes[key] = len(names)
            names.append(x.name)
        return nodes[key]

    src = [node(e.triple.head) for e in items]
    dst = [node(e.triple.tail) for e in items]
    rel = [RELATION_VOCAB[e.triple.relation.value] for e in items]
    edge_index = torch.tensor([src + dst, dst + src])
    edge_type = torch.tensor(rel + rel)
    return names, edge_index, edge_type


class GraphEncoder(nn.Module):
    """R-GCN over the subgraph; each KG item (edge) is represented by its endpoints."""

    def __init__(self, cfg: EncoderConfig, node_text: TextEncoder) -> None:
        super().__init__()
        self.node_text = node_text
        self.inp = nn.Linear(cfg.text_dim, cfg.graph_dim)
        self.layers = nn.ModuleList(RGCNLayer(cfg.graph_dim, len(RELATION_VOCAB), cfg.num_bases)
                                    for _ in range(cfg.rgcn_layers))

    def _node_features(self, names: Sequence[str]) -> torch.Tensor:
        return self.node_text([Evidence(eid=n, source_type=SourceType.KG, text=n)
                               for n in names])

    def forward(self, items: Sequence[Evidence]) -> torch.Tensor:
        names, edge_index, edge_type = build_subgraph(items)
        h = self.inp(self._node_features(names))
        edge_index, edge_type = edge_index.to(h.device), edge_type.to(h.device)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        n = len(items)
        return 0.5 * (h[edge_index[0, :n]] + h[edge_index[1, :n]])


# --------------------------------------------------------------------------- #
# Proteins and complexes
# --------------------------------------------------------------------------- #

class ProteinEncoder(nn.Module):
    """Mean-pooled ESM-2 residue embeddings."""

    def __init__(self, esm: nn.Module, tokenizer, max_len: int = 1022) -> None:
        super().__init__()
        self.esm, self.tok, self.max_len = esm, tokenizer, max_len

    def forward(self, seqs: Sequence[str]) -> torch.Tensor:
        enc = self.tok([s[: self.max_len] for s in seqs], padding=True,
                       return_tensors="pt").to(next(self.esm.parameters()).device)
        h = self.esm(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


#: Evidence -> (atom types (N,), centred coordinates (N, 3), Uni-Mol ligand inputs).
ComplexFeaturizer = Callable[[Evidence], Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]]


def struct_confidence_features(e: Evidence) -> List[float]:
    """[is_predicted, pLDDT / 100, 1 - PAE / 30]; experimental structures give [0, 1, 1]."""
    c = e.struct_confidence or {}
    if not c:
        return [0.0, 1.0, 1.0]
    return [1.0, c.get("plddt", 70.0) / 100.0, 1.0 - min(c.get("pae", 10.0), 30.0) / 30.0]


class ComplexEncoder(nn.Module):
    """EquiformerV2 over the complex, Uni-Mol ligand features and ESM-2 protein features.

    The equivariant network returns per-atom invariant (l = 0) features; mean
    pooling over atoms makes the complex embedding invariant to rotations and
    translations. pLDDT / PAE of predicted complexes are appended as features.
    """

    def __init__(self, cfg: EncoderConfig, equiformer: nn.Module, unimol: nn.Module,
                 protein: ProteinEncoder, featurize: ComplexFeaturizer) -> None:
        super().__init__()
        self.equiformer, self.unimol, self.protein = equiformer, unimol, protein
        self.featurize = featurize
        self.head = nn.Sequential(
            nn.Linear(cfg.equi_dim + cfg.unimol_dim + cfg.esm_dim + 3, cfg.struct_dim),
            nn.GELU(), nn.Linear(cfg.struct_dim, cfg.struct_dim))

    def forward(self, items: Sequence[Evidence]) -> torch.Tensor:
        device = self.head[0].weight.device
        rows = []
        for e in items:
            atom_types, pos, ligand = self.featurize(e)
            node_inv = self.equiformer(atom_types.to(device), pos.to(device))   # (N, equi_dim)
            lig = self.unimol(**ligand)                                        # (unimol_dim,)
            rows.append(torch.cat([node_inv.mean(0), lig.reshape(-1)]))
        prot = self.protein([e.protein_seq or "X" for e in items]).to(device)
        conf = torch.tensor([struct_confidence_features(e) for e in items], device=device)
        return self.head(torch.cat([torch.stack(rows), prot, conf], -1))


# --------------------------------------------------------------------------- #
# Soft tokens
# --------------------------------------------------------------------------- #

class SoftTokenProjector(nn.Module):
    """Modality-specific projection into the LLM input space plus a modality embedding."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        dims = {SourceType.TEXT: cfg.text_dim, SourceType.DB: cfg.text_dim,
                SourceType.KG: cfg.graph_dim, SourceType.STRUCT: cfg.struct_dim}
        self.proj = nn.ModuleDict({
            s.value: nn.Sequential(nn.Linear(d, cfg.llm_dim), nn.GELU(),
                                   nn.Linear(cfg.llm_dim, cfg.llm_dim))
            for s, d in dims.items()})
        self.modality = nn.Embedding(len(SourceType), cfg.llm_dim)
        self.modality_idx = {s: i for i, s in enumerate(SourceType)}

    def forward(self, src: SourceType, h: torch.Tensor) -> torch.Tensor:
        return self.proj[src.value](h) + self.modality.weight[self.modality_idx[src]]


class EvidenceEncoder(nn.Module):
    """Route each evidence item to its typed encoder and return one soft token per item."""

    def __init__(self, cfg: EncoderConfig, text: TextEncoder, graph: GraphEncoder,
                 complex_: ComplexEncoder) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({
            SourceType.TEXT.value: text, SourceType.DB.value: text,
            SourceType.KG.value: graph, SourceType.STRUCT.value: complex_})
        self.projector = SoftTokenProjector(cfg)

    def forward(self, evidence: Sequence[Evidence]) -> torch.Tensor:
        """(m, llm_dim) soft tokens, in the order of `evidence`."""
        device = self.projector.modality.weight.device
        out = torch.zeros(len(evidence), self.cfg.llm_dim, device=device)
        for src in (SourceType.TEXT, SourceType.DB, SourceType.KG, SourceType.STRUCT):
            idx = [i for i, e in enumerate(evidence) if e.source_type is src]
            if not idx:
                continue
            h = self.encoders[src.value]([evidence[i] for i in idx])
            out[idx] = self.projector(src, h.to(device))
        return out
