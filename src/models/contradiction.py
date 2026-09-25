"""Reliability-weighted, direction-preserving contradiction score C_phi.

    pi_i(y) = p_phi(stance_i = y | e_i, c)                       item stance
    r_ij    in {ENTAIL, CONTRADICT, NEUTRAL}                      pair relation
    w_i     = sigma(a_i^T v),   w_ij = sigma(a_ij^T v)            reliabilities

    C_phi(c, E, y) = log( sum_i w_i pi_i(y) / sum_i w_i ) - kappa(c, E) * 1[y != INS]
    kappa(c, E)    = - sum_{i<j} w_ij log(1 - p_phi(r_ij = CONTRADICT))

Stances are read against the normalized relation, so an item reporting
activation is a REFUTED stance toward an inhibition claim. Items whose stance
is ambiguous are escalated to an LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import (LABEL_ORDER, NUM_LABELS, RELATION_INVERSE, Claim, Evidence,
                             Label, SourceType, Triple)

#: Output order of the entailment head.
NLI_ORDER = ("entail", "neutral", "contradict")
ENTAIL, NEUTRAL, CONTRADICT = range(3)

CONTEXT_KEYS = ("organism", "assay", "disease_context")
SOURCE_TYPES = list(SourceType)
EVIDENCE_LEVELS = ("curated", "experimental", "predicted", "inferred", "text")


@dataclass
class ContradictionConfig:
    nli_model: str = "microsoft/deberta-v3-large"
    ambiguity_margin: float = 0.15    # top-2 stance gap below which the LLM judge is queried
    max_pairs: int = 256
    eps: float = 1e-6


# --------------------------------------------------------------------------- #
# Reliability features a_i, a_ij and weights w_i, w_ij
# --------------------------------------------------------------------------- #

def _source_features(e: Evidence) -> List[float]:
    """Source type, curation confidence, recency and evidence level."""
    p = e.provenance
    src = [float(e.source_type == s) for s in SOURCE_TYPES]
    recency = 0.0 if p.year is None else max(0.0, min(1.0, (p.year - 2000) / 26.0))
    level = [float(p.evidence_level == l) for l in EVIDENCE_LEVELS]
    return src + [p.confidence, recency] + level


def _context(e: Evidence) -> Dict[str, object]:
    return dict(e.triple.qualifiers) if e.triple is not None else {}


def _agreement(a: Dict[str, object], b: Dict[str, object]) -> List[float]:
    """Agreement of organism, assay and disease context."""
    return [float(k in a and k in b and a[k] == b[k]) for k in CONTEXT_KEYS]


def item_features(e: Evidence, claim: Claim) -> List[float]:
    """a_i: source features of e_i and agreement of its context with the claim's."""
    claim_q = claim.triples[0].qualifiers if claim.triples else {}
    return _source_features(e) + _agreement(_context(e), claim_q)


def pair_features(a: Evidence, b: Evidence) -> List[float]:
    """a_ij: averaged source features and agreement of the two items' contexts."""
    fa, fb = _source_features(a), _source_features(b)
    return [(x + y) / 2 for x, y in zip(fa, fb)] + _agreement(_context(a), _context(b))


N_FEATS = len(SOURCE_TYPES) + 2 + len(EVIDENCE_LEVELS) + len(CONTEXT_KEYS)


class Reliability(nn.Module):
    """w = sigma(a^T v), with v shared between items and pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.v = nn.Linear(N_FEATS, 1)

    def items(self, evidence: Sequence[Evidence], claim: Claim) -> torch.Tensor:
        a = torch.tensor([item_features(e, claim) for e in evidence])
        return torch.sigmoid(self.v(a.to(self.v.weight))).squeeze(-1)

    def pairs(self, evidence: Sequence[Evidence], pairs: Sequence[Tuple[int, int]]
              ) -> torch.Tensor:
        a = torch.tensor([pair_features(evidence[i], evidence[j]) for i, j in pairs])
        return torch.sigmoid(self.v(a.to(self.v.weight))).squeeze(-1)


# --------------------------------------------------------------------------- #
# Direction-preserving stance and pair relation
# --------------------------------------------------------------------------- #

def _endpoints(t: Triple) -> Tuple[str, str]:
    return (t.head.cid or t.head.name.lower(), t.tail.cid or t.tail.name.lower())


def relational_stance(claim_triple: Triple, e: Evidence) -> Optional[Label]:
    """Stance of a structured item toward the claim's normalized relation."""
    if e.triple is None or _endpoints(e.triple) != _endpoints(claim_triple):
        return None
    if e.triple.relation == claim_triple.relation:
        return Label.SUPPORTED
    if RELATION_INVERSE.get(claim_triple.relation) == e.triple.relation:
        return Label.REFUTED
    return None


def relational_pair(a: Evidence, b: Evidence) -> Optional[int]:
    """Pair relation of two structured items over the same endpoints."""
    if a.triple is None or b.triple is None or _endpoints(a.triple) != _endpoints(b.triple):
        return None
    if a.triple.relation == b.triple.relation:
        return ENTAIL
    if RELATION_INVERSE.get(a.triple.relation) == b.triple.relation:
        return CONTRADICT
    return None


def relation_hypothesis(t: Triple) -> str:
    """NLI hypothesis stating the claim's directional relation."""
    return f"{t.head.name} {t.relation.value.replace('_', ' ')} {t.tail.name}."


def nli_to_stance(p: torch.Tensor) -> torch.Tensor:
    """(entail, neutral, contradict) -> (SUPPORTED, REFUTED, INSUFFICIENT)."""
    return p[..., [ENTAIL, CONTRADICT, NEUTRAL]]


class DebertaNLI(nn.Module):
    """DeBERTaV3 entailment model: (premise, hypothesis) -> p(entail, neutral, contradict)."""

    def __init__(self, name: str) -> None:
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)

    def forward(self, premises: Sequence[str], hypotheses: Sequence[str]) -> torch.Tensor:
        enc = self.tok(list(premises), list(hypotheses), padding=True, truncation=True,
                       max_length=512, return_tensors="pt").to(self.model.device)
        return F.softmax(self.model(**enc).logits, -1)


#: (claim, evidence) -> distribution over (SUPPORTED, REFUTED, INSUFFICIENT).
LLMJudge = Callable[[str, str], Sequence[float]]

JUDGE_PROMPT = (
    "Claim: {claim}\nEvidence: {evidence}\n"
    "Take the direction of the relation into account (e.g. activation vs. inhibition). "
    "Does the evidence support the claim, refute it, or neither? "
    "Answer with one word: Supported, Refuted or Insufficient."
)


class ContradictionScorer(nn.Module):

    def __init__(self, nli: DebertaNLI, judge: Optional[LLMJudge] = None,
                 config: Optional[ContradictionConfig] = None) -> None:
        super().__init__()
        self.nli, self.judge = nli, judge
        self.cfg = config or ContradictionConfig()
        self.reliability = Reliability()

    def _hypothesis(self, claim: Claim) -> str:
        return relation_hypothesis(claim.triples[0]) if claim.triples else claim.text

    def _pairs(self, evidence: Sequence[Evidence]) -> List[Tuple[int, int]]:
        m = len(evidence)
        return [(i, j) for i in range(m) for j in range(i + 1, m)][: self.cfg.max_pairs]

    # -- stances pi_i(y) ---------------------------------------------------- #

    def stances(self, claim: Claim, evidence: Sequence[Evidence]) -> torch.Tensor:
        """(m, 3) stance distributions over (SUPPORTED, REFUTED, INSUFFICIENT)."""
        pi = nli_to_stance(self.nli([e.render() for e in evidence],
                                    [self._hypothesis(claim)] * len(evidence))).clone()
        for i, e in enumerate(evidence):
            rel = relational_stance(claim.triples[0], e) if claim.triples else None
            if rel is not None:
                pi[i] = F.one_hot(torch.tensor(rel.index), NUM_LABELS).to(pi)
                continue
            top2 = pi[i].topk(2).values
            if self.judge is not None and float(top2[0] - top2[1]) < self.cfg.ambiguity_margin:
                pi[i] = torch.tensor(list(self.judge(claim.text, e.render()))).to(pi)
        return pi

    # -- pair relations r_ij -------------------------------------------------- #

    def pair_relations(self, evidence: Sequence[Evidence], pairs: Sequence[Tuple[int, int]]
                       ) -> torch.Tensor:
        """(P, 3) distributions over (entail, neutral, contradict)."""
        p = self.nli([evidence[i].render() for i, _ in pairs],
                     [evidence[j].render() for _, j in pairs]).clone()
        for k, (i, j) in enumerate(pairs):
            rel = relational_pair(evidence[i], evidence[j])
            if rel is not None:
                p[k] = F.one_hot(torch.tensor(rel), 3).to(p)
        return p

    def conflict_mass(self, evidence: Sequence[Evidence]) -> torch.Tensor:
        """kappa(c, E) >= 0."""
        pairs = self._pairs(evidence)
        if not pairs:
            return torch.tensor(0.0)
        p_contra = self.pair_relations(evidence, pairs)[:, CONTRADICT]
        w_ij = self.reliability.pairs(evidence, pairs).to(p_contra)
        return -(w_ij * torch.log1p(-p_contra.clamp(max=1 - self.cfg.eps))).sum()

    # -- C_phi ------------------------------------------------------------------ #

    def forward(self, claim: Claim, evidence: Optional[Sequence[Evidence]] = None
                ) -> Dict[str, torch.Tensor]:
        evidence = list(evidence if evidence is not None else claim.evidence)
        if not evidence:
            return {"score": torch.full((NUM_LABELS,), 1.0 / NUM_LABELS).log(),
                    "kappa": torch.tensor(0.0)}
        pi = self.stances(claim, evidence)                                   # (m, 3)
        w = self.reliability.items(evidence, claim).to(pi)                   # (m,)
        vote = torch.log((w[:, None] * pi).sum(0) / w.sum() + self.cfg.eps)
        kappa = self.conflict_mass(evidence).to(pi)
        polar = torch.tensor([float(y is not Label.INSUFFICIENT) for y in LABEL_ORDER]).to(pi)
        return {"score": vote - kappa * polar, "kappa": kappa, "stances": pi, "w": w}

    # -- L_contr ------------------------------------------------------------------ #

    def loss(self, claim: Claim, evidence: Sequence[Evidence], stance_gold: torch.Tensor,
             pair_gold: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-entropy of item stances (label order) and pair relations (NLI order)."""
        evidence = list(evidence)[: len(stance_gold)]
        if not evidence:
            return torch.zeros((), device=self.reliability.v.weight.device)
        p = self.nli([e.render() for e in evidence], [self._hypothesis(claim)] * len(evidence))
        loss = F.nll_loss(torch.log(nli_to_stance(p) + self.cfg.eps), stance_gold.to(p.device))
        if pair_gold is not None:
            pairs = self._pairs(evidence)
            q = self.nli([evidence[i].render() for i, _ in pairs],
                         [evidence[j].render() for _, j in pairs])
            loss = loss + F.nll_loss(torch.log(q + self.cfg.eps), pair_gold.to(q.device))
        return loss


def stance_targets(claim: Claim, max_items: int = 32) -> torch.Tensor:
    """Stance labels of the claim's evidence items.

    Relational stances are used where they apply; otherwise a gold evidence item
    takes the claim's label and any other item is labelled INSUFFICIENT.
    """
    gold = set(claim.gold_eids)
    out = []
    for e in claim.evidence[:max_items]:
        rel = relational_stance(claim.triples[0], e) if claim.triples else None
        if rel is None:
            rel = claim.label if e.eid in gold else Label.INSUFFICIENT
        out.append(rel.index)
    return torch.tensor(out)


def pair_targets(claim: Claim, max_items: int = 32, max_pairs: int = 256) -> torch.Tensor:
    """Pair-relation labels; pairs without a relational label are NEUTRAL."""
    ev = claim.evidence[:max_items]
    pairs = [(i, j) for i in range(len(ev)) for j in range(i + 1, len(ev))][:max_pairs]
    return torch.tensor([r if (r := relational_pair(ev[i], ev[j])) is not None else NEUTRAL
                         for i, j in pairs])
