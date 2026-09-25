"""Robustness to the five counterfactual perturbation families.

For each operator, a system is scored on the perturbed test claims; robustness
is macro-F1 on the perturbed set relative to the unperturbed set.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence

from src.data.counterfactual import OPERATORS, OntologyNeighbours
from src.data.schema import Claim
from src.eval.metrics import macro_f1


def perturbation_suite(test: Sequence[Claim], onto: OntologyNeighbours, seed: int = 42
                       ) -> Dict[str, List[Claim]]:
    rng = random.Random(seed)
    suite: Dict[str, List[Claim]] = {}
    for name, op in OPERATORS.items():
        suite[name] = [n for c in test if c.label.value == "Supported"
                       for n in [op(c, onto, rng)] if n is not None]
    return suite


def robustness(predict: Callable[[Sequence[Claim]], List[int]], test: Sequence[Claim],
               suite: Dict[str, List[Claim]]) -> Dict[str, float]:
    base = macro_f1(predict(test), [c.label.index for c in test])
    out = {"clean": base}
    for name, claims in suite.items():
        if claims:
            acc = sum(p == c.label.index for p, c in zip(predict(claims), claims)) / len(claims)
            out[name] = 100.0 * acc
    return out
