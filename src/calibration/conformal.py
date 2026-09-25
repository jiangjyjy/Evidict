"""Group-conditional split-conformal selective verification.

    nu(c, E, y) = 1 - p~(y | c, E)
    q_g         = ceil((n_g + 1)(1 - alpha))-th smallest calibration score in group g
                  (+inf if that index exceeds n_g)
    C(c, E)     = { y : nu(c, E, y) <= q_{g(c, E)} }

Commit to the label when |C| = 1, abstain otherwise. Groups are the 4 claim
types x 3 source profiles; the profile is computed from retrieved evidence and
the contradiction detector only, never from the gold label.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

import torch

from src.data.schema import LABEL_ORDER, Claim, Evidence, Label


class SourceProfile(str, Enum):
    SINGLE_DOMINANT = "single-modality-dominant"
    CONCORDANT = "concordant-multi-source"
    FLAGGED_CONFLICT = "flagged-conflict"


@dataclass
class ProfileConfig:
    dominance: float = 0.75      # share of items from one modality to count as dominant
    kappa_flag: float = 0.5      # conflict mass above which the claim is flagged


def source_profile(evidence: Sequence[Evidence], kappa: float,
                   cfg: ProfileConfig = ProfileConfig()) -> SourceProfile:
    """Test-time-visible profile: depends on (E, kappa) only, not on y."""
    if kappa > cfg.kappa_flag:
        return SourceProfile.FLAGGED_CONFLICT
    counts = Counter(e.source_type for e in evidence)
    if not counts or max(counts.values()) / sum(counts.values()) >= cfg.dominance:
        return SourceProfile.SINGLE_DOMINANT
    return SourceProfile.CONCORDANT


def group_of(claim: Claim, evidence: Sequence[Evidence], kappa: float) -> str:
    return f"{claim.claim_type.short}|{source_profile(evidence, kappa).value}"


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return float("inf")
    return sorted(scores)[k - 1]


class MondrianConformal:

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.q: Dict[str, float] = {}

    def fit(self, probs: torch.Tensor, labels: torch.Tensor, groups: Sequence[str]
            ) -> "MondrianConformal":
        """probs: (n, 3) fused p~ on the calibration half; labels: (n,)."""
        nu = 1 - probs.gather(1, labels[:, None]).squeeze(1)
        by_g: Dict[str, List[float]] = defaultdict(list)
        for g, s in zip(groups, nu.tolist()):
            by_g[g].append(s)
        self.q = {g: conformal_quantile(v, self.alpha) for g, v in by_g.items()}
        return self

    def prediction_sets(self, probs: torch.Tensor, groups: Sequence[str]) -> List[List[int]]:
        out = []
        for p, g in zip(probs, groups):
            q = self.q.get(g, float("inf"))       # unseen group: keep every label
            out.append([y for y in range(len(LABEL_ORDER)) if 1 - float(p[y]) <= q])
        return out

    def decide(self, probs: torch.Tensor, groups: Sequence[str]) -> List[Optional[Label]]:
        """A label when the set is a singleton, None (= ABSTAIN) otherwise."""
        return [LABEL_ORDER[s[0]] if len(s) == 1 else None
                for s in self.prediction_sets(probs, groups)]


def coverage_and_risk(sets: Sequence[Sequence[int]], labels: Sequence[int]) -> Dict[str, float]:
    cov = sum(y in s for s, y in zip(sets, labels)) / max(1, len(labels))
    committed = [(s, y) for s, y in zip(sets, labels) if len(s) == 1]
    wrong = sum(s[0] != y for s, y in committed)
    return {
        "coverage": cov,
        "risk": 1 - cov,                                  # upper-bounds wrong commitments
        "commit_rate": len(committed) / max(1, len(labels)),
        "wrong_commit_rate": wrong / max(1, len(labels)),
    }


# --------------------------------------------------------------------------- #
# Diagnostics only (no guarantee)
# --------------------------------------------------------------------------- #

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, steps: int = 200) -> float:
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp())


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return float(total)


def brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
    onehot = torch.nn.functional.one_hot(labels, probs.size(1)).float()
    return float(((probs - onehot) ** 2).sum(1).mean())
