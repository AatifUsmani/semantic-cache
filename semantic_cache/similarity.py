from __future__ import annotations

import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return cosine similarity for two finite, non-zero vectors."""
    first = np.asarray(vec1, dtype=np.float32).reshape(-1)
    second = np.asarray(vec2, dtype=np.float32).reshape(-1)

    if first.shape != second.shape:
        raise ValueError("Vectors must have the same dimensions")
    if first.size == 0:
        raise ValueError("Vectors must not be empty")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Vectors must contain only finite values")

    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0

    similarity = float(np.dot(first, second) / denominator)
    return float(np.clip(similarity, -1.0, 1.0))
