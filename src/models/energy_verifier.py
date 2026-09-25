"""Generative energy E_theta and the verdict-level DPO objective.

    E_theta(c, E, y) = -log p_LM(<y> | c, E)
    p_theta(y | c, E) ∝ exp(-E_theta(c, E, y))       (renormalized over the three labels)

Losses
    L_cls   = -log p_theta(y* | c, E)
    L_nce   = InfoNCE over in-batch and retrieved hard-negative evidence sets
    L_rank  = max{0, gamma + E(c, E, SUP) - E(c-, E, SUP)}
    L_dpo   = -log sigma(beta log pi/pi_ref (y_w | x) - beta log pi/pi_ref (y_l | x)),
              with the same context x in numerator and denominator

Item embeddings from the typed encoders enter as soft tokens prepended to the
prompt. The backbone is adapted with QLoRA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.schema import LABEL_ORDER, Claim, Evidence, Label

#: Single-token verbalizers of the three verdicts.
LABEL_WORDS: Dict[Label, str] = {
    Label.SUPPORTED: "True",
    Label.REFUTED: "False",
    Label.INSUFFICIENT: "Maybe",
}

PROMPT = (
    "You are a biomedical claim verifier. Decide whether the evidence supports "
    "the claim (True), refutes it (False), or is insufficient (Maybe).\n\n"
    "Evidence:\n{evidence}\n\nClaim: {claim}\n\nVerdict:"
)


@dataclass
class EnergyConfig:
    margin: float = 0.5          # gamma
    dpo_beta: float = 0.1        # beta
    nce_temperature: float = 1.0
    max_length: int = 1536
    max_evidence: int = 32       # top-8 per source, four sources


def resolve_label_token_ids(tokenizer: Any) -> List[int]:
    ids = []
    for lab in LABEL_ORDER:
        for variant in (LABEL_WORDS[lab], " " + LABEL_WORDS[lab]):
            t = tokenizer.encode(variant, add_special_tokens=False)
            if len(t) == 1:
                ids.append(t[0])
                break
        else:
            raise ValueError(f"verbalizer for {lab.value} is not a single token")
    return ids


def render_prompt(claim_text: str, evidence: Sequence[Evidence], max_evidence: int) -> str:
    items = list(evidence)[:max_evidence]
    body = "\n".join(f"[{i + 1}] {e.render()}" for i, e in enumerate(items)) \
        or "(no evidence retrieved)"
    return PROMPT.format(evidence=body, claim=claim_text)


def load_qlora_backbone(name: str, r: int = 64, alpha: int = 16, dropout: float = 0.05):
    """4-bit NF4 backbone with LoRA adapters on all attention and MLP projections."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    return get_peft_model(model, lora), tok


class GenerativeEnergyVerifier(nn.Module):

    def __init__(self, backbone: nn.Module, label_token_ids: Sequence[int],
                 config: Optional[EnergyConfig] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.cfg = config or EnergyConfig()
        self.register_buffer("label_ids", torch.tensor(list(label_token_ids)),
                             persistent=False)

    # -- energy ----------------------------------------------------------- #

    def _embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               soft_tokens: Optional[torch.Tensor], soft_mask: Optional[torch.Tensor]):
        emb = self.backbone.get_input_embeddings()(input_ids)
        if soft_tokens is None:
            return emb, attention_mask
        soft = soft_tokens.to(device=emb.device, dtype=emb.dtype)        # (B, m, H)
        if soft_mask is None:
            soft_mask = torch.ones(soft.shape[:2], dtype=attention_mask.dtype)
        soft_mask = soft_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
        return torch.cat([soft, emb], 1), torch.cat([soft_mask, attention_mask], 1)

    def energies(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 soft_tokens: Optional[torch.Tensor] = None,
                 soft_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """E_theta(c, E, y) = -log p_LM(<y> | c, E) for every y, shape (B, 3)."""
        emb, mask = self._embed(input_ids, attention_mask, soft_tokens, soft_mask)
        logits = self.backbone(inputs_embeds=emb, attention_mask=mask).logits
        last = mask.long().cumsum(1).argmax(1)                       # last non-pad position
        lp = F.log_softmax(logits[torch.arange(len(last)), last].float(), -1)
        return -lp[:, self.label_ids]

    @torch.no_grad()
    def reference_energies(self, input_ids, attention_mask, soft_tokens=None, soft_mask=None):
        """Energies under pi_ref: the same backbone with the LoRA adapters disabled."""
        disable = getattr(self.backbone, "disable_adapter", None)
        if disable is None:
            return self.energies(input_ids, attention_mask, soft_tokens, soft_mask)
        with disable():
            return self.energies(input_ids, attention_mask, soft_tokens, soft_mask)

    @staticmethod
    def log_probs(energy: torch.Tensor) -> torch.Tensor:
        """log p_theta(y | c, E): renormalization of exp(-E) over the three labels."""
        return F.log_softmax(-energy, -1)

    # -- losses ----------------------------------------------------------- #

    def loss_cls(self, energy: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(self.log_probs(energy), labels)

    def loss_nce(self, energy_pos: torch.Tensor, energy_negs: torch.Tensor,
                 labels: torch.Tensor) -> torch.Tensor:
        """InfoNCE: the claim's own evidence set against K negative sets.

        energy_pos:  (B, 3)       E(c, E, y)
        energy_negs: (B, K, 3)    E(c, E'_k, y) for in-batch / hard-negative sets E'_k
        """
        e_pos = energy_pos.gather(1, labels[:, None])                          # (B, 1)
        idx = labels[:, None, None].expand(-1, energy_negs.size(1), 1)
        e_neg = energy_negs.gather(2, idx).squeeze(-1)                          # (B, K)
        logits = -torch.cat([e_pos, e_neg], 1) / self.cfg.nce_temperature
        return F.cross_entropy(logits, torch.zeros_like(labels))

    def loss_rank(self, energy_true: torch.Tensor, energy_cf: torch.Tensor) -> torch.Tensor:
        """A true claim must receive lower SUPPORTED-energy than its counterfactual c-."""
        s = Label.SUPPORTED.index
        return F.relu(self.cfg.margin + energy_true[:, s] - energy_cf[:, s]).mean()

    def loss_dpo(self, energy: torch.Tensor, ref_energy: torch.Tensor,
                 y_w: torch.Tensor, y_l: torch.Tensor) -> torch.Tensor:
        """DPO over verdict tokens; chosen and rejected share the same context x.

        With log pi(y | x) = -E(x, y), the implicit reward beta log pi / pi_ref is
        a negative energy over the same verdict tokens that define E_theta.
        """
        lp, ref = -energy, -ref_energy
        r_w = lp.gather(1, y_w[:, None]) - ref.gather(1, y_w[:, None])
        r_l = lp.gather(1, y_l[:, None]) - ref.gather(1, y_l[:, None])
        return -F.logsigmoid(self.cfg.dpo_beta * (r_w - r_l)).mean()


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def encode(tokenizer: Any, claims: Sequence[Claim], cfg: EnergyConfig,
           evidence: Optional[Sequence[Sequence[Evidence]]] = None) -> Dict[str, torch.Tensor]:
    texts = [render_prompt(c.text, evidence[i] if evidence is not None else c.evidence,
                           cfg.max_evidence) for i, c in enumerate(claims)]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_length,
                    return_tensors="pt")
    enc["labels"] = torch.tensor([c.label.index for c in claims])
    return dict(enc)


def hard_negative_evidence(claim: Claim, pool: Sequence[Evidence], k: int = 8
                           ) -> List[Evidence]:
    """Highly ranked retrieved items that are not gold evidence for the claim."""
    gold = set(claim.gold_eids)
    return [e for e in pool if e.eid not in gold][:k]


def dpo_pairs(claim: Claim, counterfactual: Claim) -> List[Dict[str, Any]]:
    """Contexts x in {(c, E), (c-, E)}: chosen = the context's own verdict,
    rejected = the other claim's verdict."""
    E = claim.evidence
    return [
        {"claim": claim, "evidence": E, "y_w": claim.label.index,
         "y_l": counterfactual.label.index},
        {"claim": counterfactual, "evidence": E, "y_w": counterfactual.label.index,
         "y_l": claim.label.index},
    ]
