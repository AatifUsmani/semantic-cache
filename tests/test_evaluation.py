import pytest

from semantic_cache.evaluation import threshold_metrics


def test_threshold_metrics_report_false_positive_and_false_negative_counts():
    metrics = threshold_metrics(
        labels=[True, True, False, False],
        similarities=[0.9, 0.7, 0.8, 0.2],
        threshold=0.75,
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "labels, similarities, threshold",
    [
        ([], [], 0.75),
        ([True], [], 0.75),
        ([True], [float("nan")], 0.75),
        ([True], [1.1], 0.75),
        ([True], [0.8], 1.1),
        ([1], [0.8], 0.75),
    ],
)
def test_threshold_metrics_reject_invalid_input(labels, similarities, threshold):
    with pytest.raises(ValueError):
        threshold_metrics(labels, similarities, threshold)
