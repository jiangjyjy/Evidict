"""Metrics: macro-F1, modality-aware evidence F1, paired bootstrap."""

from __future__ import annotations

import random
from typing import Callable, Dict, Sequence

from src.data.schema import Claim


def macro_f1(pred: Sequence[int], gold: Sequence[int], n_classes: int = 3) -> float:
    f1s = []
    for k in range(n_classes):
        tp = sum(p == k and g == k for p, g in zip(pred, gold))
        fp = sum(p == k and g != k for p, g in zip(pred, gold))
        fn = sum(p != k and g == k for p, g in zip(pred, gold))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return 100.0 * sum(f1s) / n_classes


def evidence_f1(claims: Sequence[Claim], verdicts) -> float:
    """Item-level F1 against gold pointers, credited only when the modality also matches."""
    scores = []
    for c, v in zip(claims, verdicts):
        gold = {(e.eid, e.source_type) for e in c.evidence if e.eid in set(c.gold_eids)}
        cited = {s for step in v.trace for s in step.cited_eids}
        pred = {(e.eid, e.source_type) for e in v.evidence if e.eid in cited}
        if not gold and not pred:
            continue
        tp = len(gold & pred)
        scores.append(0.0 if tp == 0 else 2 * tp / (len(gold) + len(pred)))
    return 100.0 * sum(scores) / max(1, len(scores))


def paired_bootstrap(pred_a: Sequence[int], pred_b: Sequence[int], gold: Sequence[int],
                     metric: Callable = macro_f1, n: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """Two-sided paired bootstrap over test claims."""
    rng = random.Random(seed)
    idx = range(len(gold))
    obs = metric(pred_a, gold) - metric(pred_b, gold)
    below = 0
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        d = metric([pred_a[i] for i in s], [gold[i] for i in s]) - \
            metric([pred_b[i] for i in s], [gold[i] for i in s])
        below += d <= 0
    frac = below / n
    return {"delta": obs, "p_value": min(1.0, 2 * min(frac, 1 - frac))}
