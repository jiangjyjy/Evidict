"""Bidirectional process reward model P_psi.

For each candidate verdict y the backbone samples N reasoning traces
tau_y = (s_1, ..., s_L) ending in y. Each step is scored by

    P_psi(s_l) = p_back(g_l = 1 | s_l, E) * v_fwd(s_<=l ; y)
                 (evidence grounding)       (value-to-go toward y)

    P_psi(c, E, y) = max_{tau_y} (1/L) sum_l log P_psi(s_l)

Training labels
    g_l    alignment of the step's normalized triple to a retrieved record
    v_mc   Monte-Carlo fraction of rollouts from s_<=l that reach the gold verdict

L_prm = step-level cross-entropy on g_l + value_weight * regression loss on v_fwd.
The trace returned to the user keeps only steps with p_back > 0.5.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import (LABEL_ORDER, NUM_LABELS, Claim, ClaimType, Evidence, Label,
                             ReasoningStep, Triple)


@dataclass
class PRMConfig:
    encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    hidden: int = 768
    n_traces: int = 8           # N traces per candidate verdict
    n_rollouts: int = 8         # rollouts per step for the value labels
    max_steps: int = 6
    value_weight: float = 0.5
    keep_threshold: float = 0.5
    temperature: float = 0.7
    eps: float = 1e-6


TRACE_PROMPT = (
    "Evidence:\n{evidence}\n\nClaim: {claim}\n\n"
    "Reason step by step. Each step states one relation and cites the evidence it "
    "relies on in brackets, e.g. [3]. Conclude with 'Verdict: {verdict}'.\n"
)

ROLLOUT_PROMPT = (
    "Evidence:\n{evidence}\n\nClaim: {claim}\n\n"
    "Reason step by step. Each step states one relation and cites the evidence it "
    "relies on in brackets, e.g. [3]. Conclude with 'Verdict: Supported', "
    "'Verdict: Refuted' or 'Verdict: Insufficient'.\n{prefix}"
)


def _render_evidence(evidence: Sequence[Evidence]) -> str:
    return "\n".join(f"[{i + 1}] {e.render()}" for i, e in enumerate(evidence))


def _render_steps(steps: Sequence[ReasoningStep]) -> str:
    return "".join(f"Step {i + 1}: {s.text}\n" for i, s in enumerate(steps))


def parse_steps(text: str, evidence: Sequence[Evidence], max_steps: int
                ) -> List[ReasoningStep]:
    body = text.split("Verdict:")[0]
    chunks = [c.strip() for c in re.split(r"Step \d+:", body) if c.strip()]
    steps = []
    for c in chunks[:max_steps]:
        cited = [evidence[int(k) - 1].eid for k in re.findall(r"\[(\d+)\]", c)
                 if 0 < int(k) <= len(evidence)]
        steps.append(ReasoningStep(text=c, cited_eids=cited))
    return steps


def parse_verdict(text: str) -> Optional[Label]:
    m = re.search(r"Verdict:\s*(Supported|Refuted|Insufficient)", text)
    return Label(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Trace sampling and Monte-Carlo rollouts
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _generate(llm: Any, tok: Any, prompt: str, n: int, max_new_tokens: int,
              temperature: float) -> List[str]:
    enc = tok(prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(**enc, do_sample=True, temperature=temperature,
                       num_return_sequences=n, max_new_tokens=max_new_tokens)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def sample_traces(llm: Any, tok: Any, claim: Claim, evidence: Sequence[Evidence],
                  verdict: Label, cfg: PRMConfig) -> List[List[ReasoningStep]]:
    """N chain-of-thought traces ending in `verdict`; single-relation claims give one step."""
    max_steps = 1 if claim.claim_type is ClaimType.DT else cfg.max_steps
    prompt = TRACE_PROMPT.format(evidence=_render_evidence(evidence), claim=claim.text,
                                 verdict=verdict.value) + "Step 1:"
    texts = _generate(llm, tok, prompt, cfg.n_traces, 96 * max_steps, cfg.temperature)
    traces = [parse_steps("Step 1:" + t, evidence, max_steps) for t in texts]
    return [t for t in traces if t]


def make_rollout(llm: Any, tok: Any, claim: Claim, evidence: Sequence[Evidence],
                 cfg: PRMConfig) -> Callable[[Sequence[ReasoningStep], int], List[Label]]:
    """Continue a partial trace n times without fixing the verdict; return the final verdicts."""
    def rollout(prefix: Sequence[ReasoningStep], n: int) -> List[Label]:
        prompt = ROLLOUT_PROMPT.format(evidence=_render_evidence(evidence), claim=claim.text,
                                       prefix=_render_steps(prefix))
        texts = _generate(llm, tok, prompt, n, 96 * cfg.max_steps, cfg.temperature)
        return [v for v in map(parse_verdict, texts) if v is not None]
    return rollout


# --------------------------------------------------------------------------- #
# Step labels
# --------------------------------------------------------------------------- #

def grounding_label(step_triples: Sequence[Triple], evidence: Sequence[Evidence]) -> int:
    """g_l = 1 iff a normalized triple of the step aligns with a retrieved record."""
    keys = {e.triple.key() for e in evidence if e.triple is not None}
    return int(any(t.key() in keys for t in step_triples))


def value_label(rollout: Callable[[Sequence[ReasoningStep], int], List[Label]],
                prefix: Sequence[ReasoningStep], gold: Label, n: int) -> float:
    """v_mc(s_<=l): fraction of rollouts from the prefix that reach the gold verdict."""
    finals = rollout(prefix, n)
    return sum(f is gold for f in finals) / max(1, len(finals))


def step_targets(trace: Sequence[ReasoningStep], evidence: Sequence[Evidence],
                 normalize_step: Callable[[str], Sequence[Triple]], gold: Label,
                 rollout: Callable[[Sequence[ReasoningStep], int], List[Label]],
                 n_rollouts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(g, v_mc) for every step; `normalize_step` maps step text to typed triples
    (e.g. `ClaimNormalizer.extract`)."""
    g = [grounding_label(normalize_step(s.text), evidence) for s in trace]
    v = [value_label(rollout, trace[: l + 1], gold, n_rollouts) for l in range(len(trace))]
    return torch.tensor(g, dtype=torch.float), torch.tensor(v, dtype=torch.float)


def load_step_items(records: Sequence[Dict[str, Any]], claims: Sequence[Claim]
                    ) -> Dict[str, List[tuple]]:
    """Group precomputed step labels by claim: cid -> [(steps, E, y, g, v_mc)]."""
    by_cid = {c.cid: c for c in claims}
    out: Dict[str, List[tuple]] = {}
    for r in records:
        c = by_cid.get(r["cid"])
        if c is None:
            continue
        steps = [ReasoningStep.from_dict(s) for s in r["steps"]]
        out.setdefault(c.cid, []).append((
            steps, c.evidence, Label(r["verdict"]),
            torch.tensor(r["g"], dtype=torch.float), torch.tensor(r["v_mc"], dtype=torch.float)))
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class BidirectionalPRM(nn.Module):
    """Shared encoder with a backward (grounding) head and a forward (value) head.

    backward:  [s_l ; cited evidence]        -> p_back(g_l = 1 | s_l, E)
    forward:   [s_1 .. s_l] and verdict y    -> v_fwd(s_<=l ; y)
    """

    def __init__(self, encoder: nn.Module, tokenizer: Any, cfg: Optional[PRMConfig] = None
                 ) -> None:
        super().__init__()
        self.enc, self.tok = encoder, tokenizer
        self.cfg = cfg or PRMConfig()
        h = self.cfg.hidden
        self.back_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.verdict_emb = nn.Embedding(NUM_LABELS, h)
        self.fwd_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))

    def _cls(self, a: Sequence[str], b: Optional[Sequence[str]] = None) -> torch.Tensor:
        enc = self.tok(list(a), list(b) if b is not None else None, padding=True,
                       truncation=True, max_length=512, return_tensors="pt"
                       ).to(next(self.enc.parameters()).device)
        return self.enc(**enc).last_hidden_state[:, 0]

    def backward_logits(self, steps: Sequence[ReasoningStep],
                        evidence: Sequence[Evidence]) -> torch.Tensor:
        by_id = {e.eid: e for e in evidence}
        cited = [" ".join(by_id[x].render() for x in s.cited_eids if x in by_id)
                 or "(no evidence cited)" for s in steps]
        return self.back_head(self._cls([s.text for s in steps], cited)).squeeze(-1)

    def forward_value(self, steps: Sequence[ReasoningStep], verdict: Label) -> torch.Tensor:
        prefixes = [_render_steps(steps[: l + 1]) for l in range(len(steps))]
        h = self._cls(prefixes)
        y = self.verdict_emb.weight[verdict.index].expand_as(h)
        return torch.sigmoid(self.fwd_head(torch.cat([h, y], -1))).squeeze(-1)

    def step_scores(self, steps: Sequence[ReasoningStep], evidence: Sequence[Evidence],
                    verdict: Label) -> Dict[str, torch.Tensor]:
        p_back = torch.sigmoid(self.backward_logits(steps, evidence))
        v_fwd = self.forward_value(steps, verdict)
        return {"p_back": p_back, "v_fwd": v_fwd, "P": p_back * v_fwd}

    def trace_score(self, steps: Sequence[ReasoningStep], evidence: Sequence[Evidence],
                    verdict: Label) -> torch.Tensor:
        """(1/L) sum_l log P_psi(s_l)."""
        P = self.step_scores(steps, evidence, verdict)["P"]
        return torch.log(P + self.cfg.eps).mean()

    def forward(self, traces_by_label: Dict[Label, List[List[ReasoningStep]]],
                evidence: Sequence[Evidence]
                ) -> Tuple[torch.Tensor, Dict[Label, List[ReasoningStep]]]:
        """P_psi(c, E, y) for every y (max over traces) and the arg-max trace per label."""
        device = self.verdict_emb.weight.device
        scores, best = [], {}
        for y in LABEL_ORDER:
            traces = traces_by_label.get(y, [])
            if not traces:
                scores.append(torch.tensor(math.log(self.cfg.eps), device=device))
                continue
            s = torch.stack([self.trace_score(t, evidence, y) for t in traces])
            k = int(s.argmax())
            scores.append(s[k])
            best[y] = traces[k]
        return torch.stack(scores), best

    def audited_trace(self, steps: Sequence[ReasoningStep], evidence: Sequence[Evidence]
                      ) -> List[ReasoningStep]:
        """Keep only the steps whose grounding p_back exceeds the threshold."""
        p = torch.sigmoid(self.backward_logits(steps, evidence))
        return [s for s, q in zip(steps, p.tolist()) if q > self.cfg.keep_threshold]

    def loss(self, steps: Sequence[ReasoningStep], evidence: Sequence[Evidence],
             verdict: Label, g: torch.Tensor, v_mc: torch.Tensor) -> torch.Tensor:
        logits = self.backward_logits(steps, evidence)
        v_fwd = self.forward_value(steps, verdict)
        l_back = F.binary_cross_entropy_with_logits(logits, g.to(logits.device))
        l_fwd = F.mse_loss(v_fwd, v_mc.to(v_fwd.device))
        return l_back + self.cfg.value_weight * l_fwd
