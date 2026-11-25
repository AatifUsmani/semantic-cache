from __future__ import annotations

import math
from collections.abc import Sequence


def threshold_metrics(
    labels: Sequence[bool],
    similarities: Sequence[float],
    threshold: float,
) -> dict[str, int | float]:
    if not labels or len(labels) != len(similarities):
        raise ValueError("labels and similarities must be non-empty and have equal length")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1")
    if any(not isinstance(label, bool) for label in labels):
        raise ValueError("labels must contain booleans")
    if any(not math.isfinite(score) or not -1.0 <= score <= 1.0 for score in similarities):
        raise ValueError("similarities must be finite cosine scores")

    true_positives = false_positives = true_negatives = false_negatives = 0
    for label, similarity in zip(labels, similarities, strict=True):
        prediction = similarity >= threshold
        if label and prediction:
            true_positives += 1
        elif prediction:
            false_positives += 1
        elif label:
            false_negatives += 1
        else:
            true_negatives += 1

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positives + true_negatives) / len(labels)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }
