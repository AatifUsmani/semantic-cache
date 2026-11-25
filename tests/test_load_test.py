from load_test import LoadTestResults


def test_load_test_percentiles_work_with_one_successful_request():
    results = LoadTestResults()
    results.add_result(0.012, "cache", True)

    summary = results.get_summary()

    assert summary["p95_latency_ms"] == "12.00"
    assert summary["p99_latency_ms"] == "12.00"


def test_load_test_reports_empty_successful_sample():
    results = LoadTestResults()

    assert results.get_summary() == {
        "error": "No successful requests",
        "total_requests": 0,
        "errors": 0,
    }
