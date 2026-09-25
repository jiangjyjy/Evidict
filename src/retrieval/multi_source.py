"""Multi-source evidence retrieval.

    text    BM25 + SPLADE + ColBERTv2 (MedCPT) fused by weighted RRF
    db      curated records e_i = (s_i, r_i, o_i, a_i, rho_i)
    kg      query-focused subgraph around the linked claim entities
    struct  PDB / PDBbind / BioLiP complexes, AlphaFold3 when none exists

Each source's candidates are merged by dev-tuned weighted reciprocal rank
fusion and the top-k (k = 8) items per source are kept. No neural reranker is
applied: general-domain rerankers demote structured evidence.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.data.schema import Claim, Evidence, SourceType, Triple

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


class Scorer(Protocol):
    name: str

    def score(self, query: str, docs: Sequence[Evidence]) -> List[float]: ...


# --------------------------------------------------------------------------- #
# Sparse / dense / late-interaction scorers
# --------------------------------------------------------------------------- #

class BM25Scorer:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def score(self, query: str, docs: Sequence[Evidence]) -> List[float]:
        toks = [tokenize(d.render()) for d in docs]
        n = max(1, len(toks))
        avgdl = sum(map(len, toks)) / n or 1.0
        df = Counter(t for doc in toks for t in set(doc))
        out = []
        for doc in toks:
            tf = Counter(doc)
            s = 0.0
            for q in tokenize(query):
                if q not in tf:
                    continue
                idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
                s += idf * tf[q] * (self.k1 + 1) / (
                    tf[q] + self.k1 * (1 - self.b + self.b * len(doc) / avgdl))
            out.append(s)
        return out


class SpladeScorer:
    """SPLADE: log(1 + ReLU(MLM logits)) max-pooled into a sparse vocabulary vector."""

    name = "splade"

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        self.model, self.tok, self.device = model, tokenizer, device

    @torch.no_grad()
    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=256,
                       return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits
        w = torch.log1p(F.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
        return w.max(dim=1).values

    def score(self, query: str, docs: Sequence[Evidence]) -> List[float]:
        q = self._encode([query])
        d = self._encode([e.render() for e in docs])
        return (d @ q[0]).tolist()


class ColBERTScorer:
    """ColBERTv2-style late interaction (MaxSim) over MedCPT token embeddings.

    `proj` is the checkpoint's linear compression layer to the late-interaction width.
    """

    name = "colbert"

    def __init__(self, query_encoder, doc_encoder, tokenizer, proj: torch.nn.Module,
                 device: str = "cpu") -> None:
        self.qe, self.de, self.tok, self.device = query_encoder, doc_encoder, tokenizer, device
        self.proj = proj

    @torch.no_grad()
    def _tokens(self, enc_model, texts: Sequence[str]) -> List[torch.Tensor]:
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=256,
                       return_tensors="pt").to(self.device)
        h = F.normalize(self.proj(enc_model(**enc).last_hidden_state), dim=-1)
        return [h[i, enc["attention_mask"][i].bool()] for i in range(h.size(0))]

    def score(self, query: str, docs: Sequence[Evidence]) -> List[float]:
        q = self._tokens(self.qe, [query])[0]
        return [float((q @ d.T).max(dim=1).values.sum())
                for d in self._tokens(self.de, [e.render() for e in docs])]


# --------------------------------------------------------------------------- #
# Weighted reciprocal rank fusion
# --------------------------------------------------------------------------- #

def weighted_rrf(rankings: Dict[str, Sequence[str]], weights: Dict[str, float],
                 k: int = 60) -> List[Tuple[str, float]]:
    """score(d) = sum_r w_r / (k + rank_r(d))."""
    fused: Dict[str, float] = defaultdict(float)
    for name, ranked in rankings.items():
        w = weights.get(name, 1.0)
        for rank, eid in enumerate(ranked, 1):
            fused[eid] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


# --------------------------------------------------------------------------- #
# Structured sources
# --------------------------------------------------------------------------- #

class CuratedRecordRetriever:
    """Exact / ontology-id match of claim tuples against curated records.

    A record carries subject s, relation r, object o, assay/organism a and
    provenance rho; matching on linked ids avoids surface-form drift.
    """

    def __init__(self, records: Iterable[Evidence]) -> None:
        self.by_key: Dict[Tuple[str, str], List[Evidence]] = defaultdict(list)
        for e in records:
            if e.triple is not None:
                self.by_key[self._pair(e.triple)].append(e)

    @staticmethod
    def _pair(t: Triple) -> Tuple[str, str]:
        return (t.head.cid or t.head.name.lower(), t.tail.cid or t.tail.name.lower())

    def retrieve(self, claim: Claim) -> List[Evidence]:
        out: List[Evidence] = []
        for t in claim.triples:
            hits = self.by_key.get(self._pair(t), [])
            # Same relation first, then other relations between the same endpoints,
            # so that opposite-direction records reach the contradiction score.
            out += sorted(hits, key=lambda e: (e.triple.relation != t.relation,
                                               -e.provenance.confidence))
        return out


class SubgraphRetriever:
    """Query-focused k-hop subgraph around the claim's linked entities."""

    def __init__(self, edges: Iterable[Evidence], hops: int = 2) -> None:
        self.hops = hops
        self.adj: Dict[str, List[Evidence]] = defaultdict(list)
        for e in edges:
            if e.triple is None:
                continue
            h = e.triple.head.cid or e.triple.head.name.lower()
            t = e.triple.tail.cid or e.triple.tail.name.lower()
            self.adj[h].append(e)
            self.adj[t].append(e)

    def retrieve(self, claim: Claim) -> List[Evidence]:
        seeds = {x.cid or x.name.lower() for t in claim.triples for x in (t.head, t.tail)}
        frontier, seen_nodes, edges = set(seeds), set(seeds), {}
        for _ in range(self.hops):
            nxt = set()
            for node in frontier:
                for e in self.adj.get(node, []):
                    edges[e.eid] = e
                    for x in (e.triple.head, e.triple.tail):
                        key = x.cid or x.name.lower()
                        if key not in seen_nodes:
                            nxt.add(key)
            seen_nodes |= nxt
            frontier = nxt
        # Rank edges by the number of claim entities they touch.
        def focus(e: Evidence) -> int:
            return sum((x.cid or x.name.lower()) in seeds
                       for x in (e.triple.head, e.triple.tail))
        return sorted(edges.values(), key=lambda e: -focus(e))


class StructureRetriever:
    """Experimental complexes (PDB / PDBbind / BioLiP), AlphaFold3 as fallback.

    Predicted complexes carry pLDDT / PAE in `struct_confidence`, which the
    structural encoder consumes so that low-confidence models count for less.
    """

    def __init__(self, experimental: Dict[Tuple[str, str], List[Evidence]],
                 predictor: Optional[Callable[[str, str], Optional[Evidence]]] = None) -> None:
        self.experimental = experimental
        self.predictor = predictor      # wraps an AlphaFold3 run

    def retrieve(self, claim: Claim) -> List[Evidence]:
        out: List[Evidence] = []
        for t in claim.triples:
            key = (t.head.cid or "", t.tail.cid or "")
            hits = self.experimental.get(key, [])
            if not hits and self.predictor is not None:
                pred = self.predictor(*key)
                hits = [pred] if pred is not None else []
            out += hits
        return out


# --------------------------------------------------------------------------- #
# Hybrid multi-source retriever
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalConfig:
    top_k_per_source: int = 8
    rrf_k: int = 60
    # Dev-tuned fusion weights; see `tune_weights`.
    weights: Dict[str, float] = field(default_factory=lambda: {
        "bm25": 1.0, "splade": 1.0, "colbert": 1.0})


class MultiSourceRetriever:

    def __init__(self, text_corpus: Sequence[Evidence], text_scorers: Sequence[Scorer],
                 db: CuratedRecordRetriever, kg: SubgraphRetriever,
                 struct: StructureRetriever, config: Optional[RetrievalConfig] = None) -> None:
        self.text_corpus = list(text_corpus)
        self.text_scorers = list(text_scorers)
        self.db, self.kg, self.struct = db, kg, struct
        self.cfg = config or RetrievalConfig()

    def _text(self, claim: Claim) -> List[Evidence]:
        rankings = {}
        for s in self.text_scorers:
            scores = s.score(claim.text, self.text_corpus)
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            rankings[s.name] = [self.text_corpus[i].eid for i in order[:200]]
        fused = weighted_rrf(rankings, self.cfg.weights, self.cfg.rrf_k)
        by_id = {e.eid: e for e in self.text_corpus}
        return [by_id[eid] for eid, _ in fused]

    def retrieve(self, claim: Claim) -> List[Evidence]:
        k = self.cfg.top_k_per_source
        pools = {
            SourceType.TEXT: self._text(claim),
            SourceType.DB: self.db.retrieve(claim),
            SourceType.KG: self.kg.retrieve(claim),
            SourceType.STRUCT: self.struct.retrieve(claim),
        }
        out: List[Evidence] = []
        for src, items in pools.items():
            seen = set()
            for rank, e in enumerate(items):
                if e.eid in seen:
                    continue
                seen.add(e.eid)
                e.score = 1.0 / (self.cfg.rrf_k + rank + 1)
                out.append(e)
                if len(seen) >= k:
                    break
        return out

    def tune_weights(self, dev: Sequence[Claim],
                     grid: Sequence[float] = (0.5, 1.0, 1.5, 2.0)) -> Dict[str, float]:
        """Grid search of the RRF weights on the selection half of dev (Recall@8)."""
        best, best_w = -1.0, dict(self.cfg.weights)
        names = [s.name for s in self.text_scorers]
        from itertools import product
        for combo in product(grid, repeat=len(names)):
            self.cfg.weights = dict(zip(names, combo))
            r = sum(recall_at_k([e.eid for e in self._text(c)], c.gold_eids,
                                self.cfg.top_k_per_source) for c in dev) / max(1, len(dev))
            if r > best:
                best, best_w = r, dict(self.cfg.weights)
        self.cfg.weights = best_w
        return best_w


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))
