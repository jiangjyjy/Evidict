"""Robustness to the five counterfactual perturbation families.

    python -m scripts.evaluate_robustness --config configs/default.yaml --checkpoint outputs/default
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import yaml

from src.data.counterfactual import OntologyNeighbours
from src.data.schema import read_jsonl
from src.eval.robustness import perturbation_suite, robustness
from src.train import build, predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, tok = build(cfg)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, "trainable.pt")), strict=False)
    model.eval()
    llm = model.energy.backbone

    test = read_jsonl(cfg["data"]["test"])
    with open(cfg["data"]["ontology"]) as f:
        onto = OntologyNeighbours.from_json(json.load(f))
    suite = perturbation_suite(test, onto, seed=cfg["seed"])
    with torch.no_grad():
        scores = robustness(lambda cs: predict(model, cs, llm, tok), test, suite)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
