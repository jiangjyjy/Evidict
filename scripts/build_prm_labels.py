"""Step-level labels for the process reward model.

    python -m scripts.build_prm_labels --config configs/default.yaml

For each training claim and candidate verdict, sample reasoning traces with the
backbone, then label every step with
    g_l   alignment of the step's normalized triple to a retrieved record
    v_mc  fraction of Monte-Carlo rollouts from s_<=l that reach the gold verdict
"""

from __future__ import annotations

import argparse
import json

import yaml

from src.data.schema import LABEL_ORDER, read_jsonl
from src.models.prm import PRMConfig, make_rollout, sample_traces, step_targets
from scripts.prepare_evidence import build_normalizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--corpus", default="data/corpus")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pcfg = PRMConfig(**cfg["prm"])

    normalizer = build_normalizer(cfg, args.corpus)
    llm, tok = normalizer.llm, normalizer.tok

    records = []
    for claim in read_jsonl(cfg["data"]["train"]):
        rollout = make_rollout(llm, tok, claim, claim.evidence, pcfg)
        for y in LABEL_ORDER:
            for trace in sample_traces(llm, tok, claim, claim.evidence, y, pcfg):
                g, v = step_targets(trace, claim.evidence, normalizer.extract, claim.label,
                                    rollout, pcfg.n_rollouts)
                records.append({"cid": claim.cid, "verdict": y.value,
                                "steps": [s.to_dict() for s in trace],
                                "g": g.tolist(), "v_mc": v.tolist()})

    with open(cfg["data"]["prm_labels"], "w") as f:
        json.dump(records, f)


if __name__ == "__main__":
    main()
