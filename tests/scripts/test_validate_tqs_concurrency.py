from scripts.validate_tqs_concurrency import _compare


def _execution(case_id, *, steel, concrete, valid=True, success=True):
    return {
        "case_id": case_id,
        "slot": f"Slot_{case_id:02d}",
        "success": success,
        "is_valid": valid,
        "steel_kgf": steel,
        "concrete_m3": concrete,
        "report_sha256": "a" * 64,
    }


def test_concurrency_comparison_accepts_equivalent_tqs_results():
    cases = [{"case_id": 1, "source_index": 7, "origin": "test"}]
    baseline = [_execution(1, steel=500.0, concrete=10.0)]
    concurrent = [_execution(1, steel=500.2, concrete=10.0005)]

    comparisons, passed = _compare(
        cases,
        baseline,
        concurrent,
        steel_tolerance=0.5,
        concrete_tolerance=0.001,
    )

    assert passed is True
    assert comparisons[0]["passed"] is True


def test_concurrency_comparison_rejects_cross_mode_divergence():
    cases = [{"case_id": 1, "source_index": 7, "origin": "test"}]
    baseline = [_execution(1, steel=500.0, concrete=10.0)]
    concurrent = [_execution(1, steel=502.0, concrete=10.0)]

    comparisons, passed = _compare(
        cases,
        baseline,
        concurrent,
        steel_tolerance=0.5,
        concrete_tolerance=0.001,
    )

    assert passed is False
    assert comparisons[0]["passed"] is False
