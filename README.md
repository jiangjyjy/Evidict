# Evidict: Risk-Controlled Multi-Source Biomedical Claim Verification

Code for the paper under double-blind review.

## Overview

Given a claim, the verifier returns a label in {Supported, Refuted, Insufficient}
or **Abstain**, together with the selected evidence and an auditable reasoning trace.

```
claim ──► normalization & entity linking
      ──► multi-source retrieval (text / curated records / KG / structure)
      ──► typed encoders ─► soft tokens
      ──► three label-conditional scores
             • generative energy          -E_θ(c,E,y)
             • contradiction score         C_φ(c,E,y)
             • bidirectional PRM           P_ψ(c,E,y)
      ──► fused score  s_y = -E_θ + λ_c C_φ + λ_p P_ψ
      ──► group-conditional split conformal ─► commit or abstain
```

## Code structure

```
configs/default.yaml                 hyper-parameters
scripts/
  prepare_evidence.py                claim normalization + multi-source retrieval
  build_prm_labels.py                step-level grounding / Monte-Carlo value labels
  evaluate_robustness.py             five counterfactual perturbation families
src/
  data/schema.py                     typed claims, tuples (h, r, t, q) and evidence
  data/counterfactual.py             counterfactual operators and qualifier corruption
  normalize/claim_normalizer.py      LLM tuple extraction + SapBERT entity linking
  retrieval/multi_source.py          BM25 / SPLADE / ColBERTv2, weighted RRF, records, KG, structures
  encoders/typed_encoders.py         PubMedBERT, R-GCN, ESM-2, complex encoder, soft tokens
  encoders/structure.py              EquiformerV2 and Uni-Mol backbones
  models/energy_verifier.py          generative energy, L_cls, L_nce, L_rank, L_dpo
  models/contradiction.py            reliability-weighted, direction-preserving C_phi
  models/prm.py                      bidirectional process reward model
  models/verifier.py                 score fusion, full objective, commit-or-abstain
  calibration/conformal.py           source profiles, group-conditional split conformal
  eval/metrics.py                    macro-F1, evidence F1, paired bootstrap
  eval/robustness.py                 robustness evaluation
  train.py                           training, calibration and evaluation
```

## Usage

```bash
pip install -r requirements.txt

python -m scripts.prepare_evidence  --config configs/default.yaml
python -m scripts.build_prm_labels  --config configs/default.yaml
python -m src.train                 --config configs/default.yaml     # seeds 13, 21, 42, 87, 2026
python -m scripts.evaluate_robustness --config configs/default.yaml --checkpoint outputs/default
```

Claims are stored as JSONL following `src/data/schema.py` (one `Claim` per
line, with its evidence items and gold evidence pointers).

## Notes

- The backbone is adapted with QLoRA; the reference policy for DPO is the same
  backbone with adapters disabled.
- Conformal groups are claim type × source profile. The profile is computed
  from the retrieved evidence and the contradiction detector only, never from
  the gold label.
- Data and checkpoints will be released upon acceptance.

## License

Released for review purposes only.
