"""Full verifier: score fusion, training objective and commit-or-abstain inference.

    s_y(c, E) = -E_theta(c, E, y) + lambda_c C_phi(c, E, y) + lambda_p P_psi(c, E, y)
    p~(y | c, E) = softmax_y s_y

    L = L_cls + l_nce L_nce + l_rank L_rank + l_dpo L_dpo + l_contr L_contr + l_prm L_prm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.calibration.conformal import MondrianConformal, group_of
from src.data.schema import LABEL_ORDER, Claim, Evidence, Label, ReasoningStep
from src.encoders.typed_encoders import EvidenceEncoder
from src.models.contradiction import ContradictionScorer
from src.models.energy_verifier import GenerativeEnergyVerifier, encode
from src.models.prm import BidirectionalPRM, sample_traces

Traces = Dict[Label, List[List[ReasoningStep]]]


@dataclass
class FusionConfig:
    lambda_c: float = 1.0
    lambda_p: float = 1.0
    lambda_nce: float = 0.5
    lambda_rank: float = 0.3
    lambda_dpo: float = 0.5
    lambda_contr: float = 0.3
    lambda_prm: float = 0.5


@dataclass
class Verdict:
    label: Optional[Label]                   # None means ABSTAIN
    prediction_set: List[Label]
    probs: List[float]
    evidence: List[Evidence]
    trace: List[ReasoningStep] = field(default_factory=list)
    group: str = ""


class RiskControlledVerifier(nn.Module):

    def __init__(self, energy: GenerativeEnergyVerifier, contradiction: ContradictionScorer,
                 prm: BidirectionalPRM, evidence_encoder: EvidenceEncoder, tokenizer: Any,
                 config: Optional[FusionConfig] = None) -> None:
        super().__init__()
        self.energy, self.contra, self.prm = energy, contradiction, prm
        self.evidence_encoder = evidence_encoder
        self.tok = tokenizer
        self.cfg = config or FusionConfig()
        self.conformal: Optional[MondrianConformal] = None

    @property
    def device(self) -> torch.device:
        return next(self.energy.parameters()).device

    # -- energy with soft tokens ---------------------------------------------- #

    def _soft_tokens(self, evidence_sets: Sequence[Sequence[Evidence]]):
        toks = [self.evidence_encoder(E) for E in evidence_sets]
        m = max(1, max(t.size(0) for t in toks))
        dim = self.evidence_encoder.cfg.llm_dim
        pad = torch.zeros(len(toks), m, dim, device=self.device)
        mask = torch.zeros(len(toks), m, dtype=torch.long, device=self.device)
        for b, t in enumerate(toks):
            pad[b, : t.size(0)] = t.to(self.device)
            mask[b, : t.size(0)] = 1
        return pad, mask

    def energies(self, claims: Sequence[Claim], evidence_sets: Sequence[Sequence[Evidence]],
                 reference: bool = False) -> torch.Tensor:
        batch = encode(self.tok, claims, self.energy.cfg, evidence_sets)
        soft, soft_mask = self._soft_tokens(evidence_sets)
        fn = self.energy.reference_energies if reference else self.energy.energies
        return fn(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device),
                  soft, soft_mask)

    # -- fused score s_y ------------------------------------------------------ #

    def fused_scores(self, claims: Sequence[Claim], evidence_sets: Sequence[Sequence[Evidence]],
                     traces: Sequence[Traces]) -> Dict[str, Any]:
        E = self.energies(claims, evidence_sets)
        s = -E
        kappas, best_traces, rows = [], [], []
        for b, (c, ev) in enumerate(zip(claims, evidence_sets)):
            co = self.contra(c, ev)
            p, best = self.prm(traces[b], ev)
            rows.append(self.cfg.lambda_c * co["score"].to(self.device)
                        + self.cfg.lambda_p * p.to(self.device))
            kappas.append(float(co["kappa"]))
            best_traces.append(best)
        s = s + torch.stack(rows)
        return {"energy": E, "score": s, "probs": F.softmax(s, -1), "kappa": kappas,
                "best_traces": best_traces}

    # -- objective --------------------------------------------------------------- #

    def training_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Full objective on a batch produced by `src.train.collate`."""
        cfg, ev = self.cfg, self.energy
        claims, evidence = batch["claims"], batch["evidence"]
        y = torch.tensor([c.label.index for c in claims], device=self.device)
        E = self.energies(claims, evidence)
        parts = {"cls": ev.loss_cls(E, y)}

        # InfoNCE: each claim against K negative evidence sets.
        neg_sets = batch["negative_evidence"]                     # B lists of K sets
        K = len(neg_sets[0])
        E_neg = self.energies([c for c in claims for _ in range(K)],
                              [s for sets in neg_sets for s in sets]).view(len(claims), K, -1)
        parts["nce"] = cfg.lambda_nce * ev.loss_nce(E, E_neg, y)

        # Margin ranking against counterfactuals under the same evidence.
        pairs = batch["counterfactual_pairs"]                     # [(c, c-)]
        if pairs:
            E_true = self.energies([c for c, _ in pairs], [c.evidence for c, _ in pairs])
            E_cf = self.energies([n for _, n in pairs], [c.evidence for c, _ in pairs])
            parts["rank"] = cfg.lambda_rank * ev.loss_rank(E_true, E_cf)

        # DPO over contexts {(c, E), (c-, E)}.
        dpo = batch["dpo"]
        if dpo:
            xs, es = [d["claim"] for d in dpo], [d["evidence"] for d in dpo]
            y_w = torch.tensor([d["y_w"] for d in dpo], device=self.device)
            y_l = torch.tensor([d["y_l"] for d in dpo], device=self.device)
            parts["dpo"] = cfg.lambda_dpo * ev.loss_dpo(
                self.energies(xs, es), self.energies(xs, es, reference=True), y_w, y_l)

        # Stance and pair-relation cross-entropy.
        parts["contr"] = cfg.lambda_contr * torch.stack([
            self.contra.loss(c, e, sg, pg) for c, e, sg, pg in zip(
                claims, evidence, batch["stance_gold"], batch["pair_gold"])]).mean()

        # Step-level grounding cross-entropy and value regression.
        steps = batch["prm_items"]                                # [(steps, E, y, g, v_mc)]
        if steps:
            parts["prm"] = cfg.lambda_prm * torch.stack(
                [self.prm.loss(*item) for item in steps]).mean()

        parts["total"] = sum(parts.values())
        return parts

    # -- calibration and inference ------------------------------------------------ #

    def sample_all_traces(self, claims, evidence_sets, llm, llm_tok) -> List[Traces]:
        """N traces per candidate verdict for every claim."""
        return [{y: sample_traces(llm, llm_tok, c, e, y, self.prm.cfg) for y in LABEL_ORDER}
                for c, e in zip(claims, evidence_sets)]

    @torch.no_grad()
    def calibrate(self, claims: Sequence[Claim], evidence_sets: Sequence[Sequence[Evidence]],
                  llm: Any, llm_tok: Any, alpha: float = 0.10) -> MondrianConformal:
        """Fit per-group thresholds on the calibration half of dev."""
        out = self.fused_scores(claims, evidence_sets,
                                self.sample_all_traces(claims, evidence_sets, llm, llm_tok))
        groups = [group_of(c, e, k) for c, e, k in zip(claims, evidence_sets, out["kappa"])]
        labels = torch.tensor([c.label.index for c in claims])
        self.conformal = MondrianConformal(alpha).fit(out["probs"].cpu(), labels, groups)
        return self.conformal

    @torch.no_grad()
    def verify(self, claims: Sequence[Claim], evidence_sets: Sequence[Sequence[Evidence]],
               llm: Any, llm_tok: Any) -> List[Verdict]:
        """Commit to the label when the prediction set is a singleton, abstain otherwise."""
        if self.conformal is None:
            raise RuntimeError("calibrate() must be called before verify()")
        out = self.fused_scores(claims, evidence_sets,
                                self.sample_all_traces(claims, evidence_sets, llm, llm_tok))
        groups = [group_of(c, e, k) for c, e, k in zip(claims, evidence_sets, out["kappa"])]
        sets = self.conformal.prediction_sets(out["probs"].cpu(), groups)
        verdicts = []
        for b, E in enumerate(evidence_sets):
            label = LABEL_ORDER[sets[b][0]] if len(sets[b]) == 1 else None
            trace: List[ReasoningStep] = []
            if label is not None and label in out["best_traces"][b]:
                trace = self.prm.audited_trace(out["best_traces"][b][label], E)
            verdicts.append(Verdict(label=label, prediction_set=[LABEL_ORDER[i] for i in sets[b]],
                                    probs=out["probs"][b].tolist(), evidence=list(E),
                                    trace=trace, group=groups[b]))
        return verdicts
