import numpy as np
import pytest

from semantic_cache.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():
    assert cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_zero_vector_has_similarity_zero():
    assert cosine_similarity(np.zeros(2), np.array([1.0, 0.0])) == 0.0


@pytest.mark.parametrize(
    "first, second",
    [
        (np.array([]), np.array([])),
        (np.array([1.0]), np.array([1.0, 2.0])),
        (np.array([np.nan]), np.array([1.0])),
    ],
)
def test_invalid_vectors_are_rejected(first, second):
    with pytest.raises(ValueError):
        cosine_similarity(first, second)
