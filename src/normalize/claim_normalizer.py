"""Claim normalization and entity linking.

A free-text claim c is mapped to typed tuples z(c) = {(h, r, t, q)} by an
instruction-tuned biomedical LLM, and every entity mention is linked to an
ontology identifier with SapBERT nearest-neighbour search:

    drug      -> DrugBank / ChEMBL
    protein   -> UniProt
    disease   -> UMLS
    pathway   -> Reactome / KEGG / GO
    structure -> PDB
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.data.schema import Claim, Entity, EntityKind, Relation, Triple

QUALIFIER_KEYS = ("organism", "cell_type", "assay", "mutation", "residue", "condition")

NORMALIZE_PROMPT = """Extract every biomedical relation asserted by the claim.
Return a JSON list; each item has keys "head", "head_type", "relation",
"tail", "tail_type" and "qualifiers" (a dict with any of: {qualifiers}).
Relations must be one of: {relations}. Keep the direction of the relation
(e.g. "inhibits" is not "associated_with").

Claim: {claim}
JSON:"""

NAMESPACE_BY_KIND: Dict[EntityKind, Tuple[str, ...]] = {
    EntityKind.DRUG: ("DB", "CHEMBL"),
    EntityKind.LIGAND: ("CHEMBL", "PDBLIG"),
    EntityKind.PROTEIN: ("UNIPROT",),
    EntityKind.GENE: ("UNIPROT", "HGNC"),
    EntityKind.DISEASE: ("UMLS",),
    EntityKind.PHENOTYPE: ("UMLS", "HP"),
    EntityKind.PATHWAY: ("REACT", "KEGG", "GO"),
}


@dataclass
class NormalizerConfig:
    llm_name: str = "BioMistral/BioMistral-7B"
    linker_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    max_new_tokens: int = 384
    link_threshold: float = 0.75   # cosine similarity below which a mention stays unlinked


class OntologyIndex:
    """Dense SapBERT index over ontology synonyms, one matrix per namespace."""

    def __init__(self, encoder: Any, tokenizer: Any, device: str = "cpu") -> None:
        self.encoder, self.tokenizer, self.device = encoder, tokenizer, device
        self.names: Dict[str, List[Tuple[str, str]]] = {}   # ns -> [(cid, synonym)]
        self.matrix: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def embed(self, strings: Sequence[str], batch_size: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(strings), batch_size):
            enc = self.tokenizer(list(strings[i:i + batch_size]), padding=True,
                                 truncation=True, max_length=25,
                                 return_tensors="pt").to(self.device)
            cls = self.encoder(**enc).last_hidden_state[:, 0]    # SapBERT uses [CLS]
            out.append(F.normalize(cls, dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 768)

    def add_namespace(self, ns: str, entries: Sequence[Tuple[str, str]]) -> None:
        self.names[ns] = list(entries)
        self.matrix[ns] = self.embed([syn for _, syn in entries])

    def link(self, mention: str, kind: EntityKind, threshold: float
             ) -> Optional[str]:
        q = self.embed([mention])
        best: Tuple[float, Optional[str]] = (threshold, None)
        for ns in NAMESPACE_BY_KIND.get(kind, ()):
            if ns not in self.matrix or len(self.matrix[ns]) == 0:
                continue
            sims = self.matrix[ns] @ q[0]
            j = int(sims.argmax())
            if float(sims[j]) >= best[0]:
                best = (float(sims[j]), f"{ns}:{self.names[ns][j][0]}")
        return best[1]


class ClaimNormalizer:
    """LLM tuple extraction followed by SapBERT entity linking."""

    def __init__(self, llm: Any, llm_tokenizer: Any, index: OntologyIndex,
                 config: Optional[NormalizerConfig] = None) -> None:
        self.llm, self.tok, self.index = llm, llm_tokenizer, index
        self.cfg = config or NormalizerConfig()

    def _generate(self, claim_text: str) -> str:
        prompt = NORMALIZE_PROMPT.format(
            qualifiers=", ".join(QUALIFIER_KEYS),
            relations=", ".join(r.value for r in Relation),
            claim=claim_text)
        enc = self.tok(prompt, return_tensors="pt").to(self.llm.device)
        with torch.no_grad():
            out = self.llm.generate(**enc, max_new_tokens=self.cfg.max_new_tokens,
                                    do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)

    @staticmethod
    def _parse(raw: str) -> List[Dict[str, Any]]:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        return [x for x in items if isinstance(x, dict)]

    def _entity(self, name: str, kind: str) -> Entity:
        try:
            k = EntityKind(kind)
        except ValueError:
            k = EntityKind.OTHER
        cid = self.index.link(name, k, self.cfg.link_threshold)
        return Entity(name=name, kind=k, cid=cid)

    def extract(self, text: str) -> List[Triple]:
        """Typed, linked tuples asserted by a piece of text."""
        triples: List[Triple] = []
        for item in self._parse(self._generate(text)):
            try:
                rel = Relation(str(item.get("relation", "")).lower())
            except ValueError:
                continue
            quals = {k: v for k, v in (item.get("qualifiers") or {}).items()
                     if k in QUALIFIER_KEYS}
            triples.append(Triple(
                head=self._entity(item.get("head", "?"), item.get("head_type", "other")),
                relation=rel,
                tail=self._entity(item.get("tail", "?"), item.get("tail_type", "other")),
                qualifiers=quals))
        return triples

    def normalize(self, claim: Claim) -> Claim:
        claim.triples = self.extract(claim.text)
        return claim
