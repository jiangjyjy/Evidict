"""Normalize claims and retrieve multi-source evidence.

    python -m scripts.prepare_evidence --config configs/default.yaml --corpus data/corpus

Reads the raw claims, normalizes them into typed tuples, retrieves the top-8
items per source and writes train / dev / test JSONL files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import yaml

from src.data.schema import Evidence, read_jsonl, write_jsonl
from src.normalize.claim_normalizer import ClaimNormalizer, OntologyIndex
from src.retrieval.multi_source import (BM25Scorer, ColBERTScorer, CuratedRecordRetriever,
                                        MultiSourceRetriever, RetrievalConfig, SpladeScorer,
                                        StructureRetriever, SubgraphRetriever)


def load_items(path: str):
    with open(path) as f:
        return [Evidence.from_dict(json.loads(line)) for line in f if line.strip()]


def build_retriever(corpus: str, cfg: dict) -> MultiSourceRetriever:
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    text = load_items(f"{corpus}/text.jsonl")
    splade = SpladeScorer(AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil"),
                          AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil"))
    q_enc = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    d_enc = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    import torch
    proj = torch.nn.Linear(q_enc.config.hidden_size, 128, bias=False)
    proj.load_state_dict(torch.load(f"{corpus}/colbert_proj.pt"))
    colbert = ColBERTScorer(q_enc, d_enc, AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder"),
                            proj)

    structures = defaultdict(list)
    for e in load_items(f"{corpus}/structures.jsonl"):
        structures[(e.triple.head.cid or "", e.triple.tail.cid or "")].append(e)

    return MultiSourceRetriever(
        text_corpus=text,
        text_scorers=[BM25Scorer(), splade, colbert],
        db=CuratedRecordRetriever(load_items(f"{corpus}/records.jsonl")),
        kg=SubgraphRetriever(load_items(f"{corpus}/kg_edges.jsonl")),
        struct=StructureRetriever(structures),
        config=RetrievalConfig(top_k_per_source=cfg["retrieval"]["top_k_per_source"],
                               rrf_k=cfg["retrieval"]["rrf_k"]))


def build_normalizer(cfg: dict, corpus: str) -> ClaimNormalizer:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    index = OntologyIndex(AutoModel.from_pretrained(cfg["model"]["linker"]),
                          AutoTokenizer.from_pretrained(cfg["model"]["linker"]))
    with open(f"{corpus}/ontology_synonyms.json") as f:
        for ns, entries in json.load(f).items():
            index.add_namespace(ns, [tuple(x) for x in entries])
    llm = AutoModelForCausalLM.from_pretrained(cfg["model"]["backbone"])
    return ClaimNormalizer(llm, AutoTokenizer.from_pretrained(cfg["model"]["backbone"]), index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--corpus", default="data/corpus")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    claims = read_jsonl(cfg["data"]["raw"])
    normalizer = build_normalizer(cfg, args.corpus)
    retriever = build_retriever(args.corpus, cfg)

    claims = [normalizer.normalize(c) for c in claims]
    select_half = [c for c in claims if c.split == "dev"][::2]
    retriever.tune_weights(select_half)
    for c in claims:
        c.evidence = retriever.retrieve(c)

    for split in ("train", "dev", "test"):
        write_jsonl([c for c in claims if c.split == split], cfg["data"][split])


if __name__ == "__main__":
    main()
