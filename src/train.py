"""Training, conformal calibration and evaluation.

    python -m src.train --config configs/default.yaml               # all five seeds
    python -m src.train --config configs/default.yaml --seed 42

Per seed, the dev set is randomly halved into a selection half (model
selection, fusion and retrieval weights) and a calibration half used only by
the conformal layer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.calibration.conformal import coverage_and_risk
from src.data.counterfactual import OntologyNeighbours, counterfactuals
from src.data.schema import LABEL_ORDER, Claim, read_jsonl
from src.encoders.typed_encoders import (ComplexEncoder, EncoderConfig, EvidenceEncoder,
                                         GraphEncoder, ProteinEncoder, TextEncoder)
from src.eval.metrics import evidence_f1, macro_f1
from src.models.contradiction import (ContradictionConfig, ContradictionScorer, DebertaNLI,
                                     pair_targets, stance_targets)
from src.models.energy_verifier import (EnergyConfig, GenerativeEnergyVerifier, dpo_pairs,
                                        hard_negative_evidence, load_qlora_backbone,
                                        resolve_label_token_ids)
from src.models.prm import BidirectionalPRM, PRMConfig, load_step_items
from src.models.verifier import FusionConfig, RiskControlledVerifier

SEEDS = (13, 21, 42, 87, 2026)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dev(dev: Sequence[Claim], seed: int) -> Tuple[List[Claim], List[Claim]]:
    idx = list(range(len(dev)))
    random.Random(seed).shuffle(idx)
    half = len(idx) // 2
    return [dev[i] for i in idx[:half]], [dev[i] for i in idx[half:]]


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #

def collate(claims: Sequence[Claim], cf_by_parent: Dict[str, Claim],
            prm_items: Dict[str, List[tuple]], n_neg: int = 4) -> Dict[str, Any]:
    """Assemble every input of the full objective for a batch of claims."""
    claims = list(claims)
    negatives = []
    for i, c in enumerate(claims):
        sets = [hard_negative_evidence(c, c.evidence)]                 # retrieved hard negative
        others = [claims[j].evidence for j in range(len(claims)) if j != i]
        sets += others[: n_neg - 1]                                    # in-batch negatives
        while len(sets) < n_neg:
            sets.append([])
        negatives.append(sets)

    pairs = [(c, cf_by_parent[c.cid]) for c in claims if c.cid in cf_by_parent]
    return {
        "claims": claims,
        "evidence": [c.evidence for c in claims],
        "negative_evidence": negatives,
        "counterfactual_pairs": pairs,
        "dpo": [d for c, n in pairs for d in dpo_pairs(c, n)],
        "stance_gold": [stance_targets(c) for c in claims],
        "pair_gold": [pair_targets(c) for c in claims],
        "prm_items": [it for c in claims for it in prm_items.get(c.cid, [])],
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build(cfg: Dict[str, Any]) -> Tuple[RiskControlledVerifier, Any]:
    from transformers import AutoModel, AutoTokenizer

    m = cfg["model"]
    backbone, tok = load_qlora_backbone(m["backbone"], m["lora_r"], m["lora_alpha"],
                                        m["lora_dropout"])
    energy = GenerativeEnergyVerifier(backbone, resolve_label_token_ids(tok),
                                      EnergyConfig(**cfg["energy"]))

    ecfg = EncoderConfig(llm_dim=backbone.config.hidden_size)
    text = TextEncoder(AutoModel.from_pretrained(ecfg.text_model),
                       AutoTokenizer.from_pretrained(ecfg.text_model))
    protein = ProteinEncoder(AutoModel.from_pretrained(ecfg.esm_model),
                             AutoTokenizer.from_pretrained(ecfg.esm_model))
    from src.encoders.structure import build_equiformer, build_unimol, featurize_complex
    complex_ = ComplexEncoder(ecfg, build_equiformer(ecfg.equi_dim), build_unimol(ecfg.unimol_dim),
                              protein, featurize_complex)
    evidence_encoder = EvidenceEncoder(ecfg, text, GraphEncoder(ecfg, text), complex_)

    contra = ContradictionScorer(DebertaNLI(m["nli"]), config=ContradictionConfig(m["nli"]))
    pcfg = PRMConfig(**cfg["prm"])
    prm = BidirectionalPRM(AutoModel.from_pretrained(pcfg.encoder_name),
                           AutoTokenizer.from_pretrained(pcfg.encoder_name), pcfg)
    model = RiskControlledVerifier(energy, contra, prm, evidence_encoder, tok,
                                   FusionConfig(**cfg["fusion"]))
    return model, tok


def predict(model: RiskControlledVerifier, claims: Sequence[Claim], llm, tok) -> List[int]:
    evidence = [c.evidence for c in claims]
    out = model.fused_scores(claims, evidence, model.sample_all_traces(claims, evidence, llm, tok))
    return out["probs"].argmax(-1).tolist()


# --------------------------------------------------------------------------- #
# One seed
# --------------------------------------------------------------------------- #

def run(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    set_seed(seed)
    d = cfg["data"]
    train, dev, test = (read_jsonl(d[k]) for k in ("train", "dev", "test"))
    select, calib = split_dev(dev, seed)
    model, tok = build(cfg)
    llm = model.energy.backbone

    with open(d["ontology"]) as f:
        onto = OntologyNeighbours.from_json(json.load(f))
    cf_by_parent = {n.parent_cid: n for n in counterfactuals(train, onto, seed)}
    with open(d["prm_labels"]) as f:
        prm_items = load_step_items(json.load(f), train)

    loader = DataLoader(train, batch_size=cfg["optim"]["batch_size"], shuffle=True,
                        collate_fn=lambda b: collate(b, cf_by_parent, prm_items))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"]["weight_decay"])

    best_f1, best_state = -1.0, None
    for _ in range(cfg["optim"]["epochs"]):
        model.train()
        for batch in loader:
            loss = model.training_loss(batch)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["optim"]["max_grad_norm"])
            opt.step()
            opt.zero_grad()
        model.eval()
        with torch.no_grad():
            f1 = macro_f1(predict(model, select, llm, tok), [c.label.index for c in select])
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters()
                          if v.requires_grad}
    model.load_state_dict(best_state, strict=False)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    torch.save(best_state, os.path.join(cfg["output_dir"], "trainable.pt"))

    model.calibrate(calib, [c.evidence for c in calib], llm, tok, alpha=cfg["conformal"]["alpha"])
    verdicts = model.verify(test, [c.evidence for c in test], llm, tok)

    gold = [c.label.index for c in test]
    pred = [int(np.argmax(v.probs)) for v in verdicts]
    sets = [[LABEL_ORDER.index(y) for y in v.prediction_set] for v in verdicts]
    result = {"seed": seed, "macro_f1": macro_f1(pred, gold),
              "evidence_f1": evidence_f1(test, verdicts), **coverage_and_risk(sets, gold)}
    conflict = [i for i, c in enumerate(test) if c.is_conflict]
    if conflict:
        result["conflict"] = {
            "macro_f1": macro_f1([pred[i] for i in conflict], [gold[i] for i in conflict]),
            **coverage_and_risk([sets[i] for i in conflict], [gold[i] for i in conflict])}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    results = [run(cfg, s) for s in seeds]
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
