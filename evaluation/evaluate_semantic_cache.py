#!/usr/bin/env python3
"""Evaluate semantic match thresholds on a small, labeled pairwise dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_cache.evaluation import threshold_metrics


DEFAULT_THRESHOLDS = "0.60,0.65,0.70,0.75,0.80,0.85,0.90"
DEFAULT_DATASET = Path(__file__).with_name("semantic_pairs.jsonl")


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            pair = json.loads(line)
            if (
                not isinstance(pair.get("query_a"), str)
                or not isinstance(pair.get("query_b"), str)
                or not isinstance(pair.get("same_intent"), bool)
            ):
                raise ValueError(f"Invalid evaluation pair on line {line_number}")
            pairs.append(pair)
    if not pairs:
        raise ValueError("The evaluation dataset is empty")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--model",
        default=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    queries = [text for pair in pairs for text in (pair["query_a"], pair["query_b"])]
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    similarities = np.sum(embeddings[0::2] * embeddings[1::2], axis=1)
    labels = [pair["same_intent"] for pair in pairs]
    thresholds = [float(value) for value in args.thresholds.split(",")]

    print(f"model={args.model} pairs={len(pairs)}")
    print("threshold  precision  recall  f1  accuracy  fp  fn")
    for threshold in thresholds:
        metrics = threshold_metrics(labels, similarities.tolist(), threshold)
        print(
            f"{threshold:9.2f}  "
            f"{metrics['precision']:9.3f}  "
            f"{metrics['recall']:6.3f}  "
            f"{metrics['f1']:4.3f}  "
            f"{metrics['accuracy']:8.3f}  "
            f"{metrics['false_positives']:2d}  "
            f"{metrics['false_negatives']:2d}"
        )


if __name__ == "__main__":
    main()
