import pytest

from scripts.validate_tqs_repeatability import _metric_stats, _select_cases, _summarize


def _checkpoint():
    return {
        "output_values": [[730.0], [291.0], [1001.0], [500.0]],
        "generated_valid_configurations": [[{"id": index}] for index in range(4)],
        "feature_vectors": [[float(index)] for index in range(4)],
    }


def test_select_cases_uses_seed_and_global_extremes():
    selected = _select_cases(_checkpoint())

    assert [case["label"] for case in selected] == [
        "seed",
        "minimum_steel",
        "maximum_steel",
    ]
    assert [case["source_index"] for case in selected] == [0, 1, 2]
    assert [case["reference_steel_kgf"] for case in selected] == [
        730.0,
        291.0,
        1001.0,
    ]


def test_metric_stats_distinguishes_exact_and_tolerated_repeatability():
    exact = _metric_stats([10.0, 10.0, 10.0], reference=10.0, tolerance=0.1)
    tolerated = _metric_stats(
        [10.0, 10.05, 9.98], reference=10.0, tolerance=0.1
    )

    assert exact["exact_repeatability"] is True
    assert exact["within_repeatability_tolerance"] is True
    assert tolerated["exact_repeatability"] is False
    assert tolerated["within_repeatability_tolerance"] is True
    assert tolerated["within_reference_tolerance"] is True


def test_summarize_requires_validity_repeatability_and_checkpoint_consistency():
    cases = _select_cases(_checkpoint())
    records = []
    for case in cases:
        for repetition in range(1, 4):
            records.append(
                {
                    "label": case["label"],
                    "repetition": repetition,
                    "steel_kgf": case["reference_steel_kgf"],
                    "concrete_m3": 12.345,
                    "is_valid": True,
                    "critical_errors": [],
                }
            )

    summaries, passed = _summarize(
        records=records,
        cases=cases,
        repeats=3,
        steel_tolerance=1.0,
        concrete_tolerance=0.001,
    )
    assert passed is True
    assert all(summary["status"] == "passed" for summary in summaries)

    records[-1]["steel_kgf"] += 2.0
    _, passed = _summarize(
        records=records,
        cases=cases,
        repeats=3,
        steel_tolerance=1.0,
        concrete_tolerance=0.001,
    )
    assert passed is False


def test_select_cases_rejects_non_distinct_extremes():
    checkpoint = _checkpoint()
    checkpoint["output_values"] = [[1.0], [2.0], [3.0], [4.0]]

    with pytest.raises(RuntimeError, match="three distinct"):
        _select_cases(checkpoint)
