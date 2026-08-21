import csv
from pathlib import Path

import pytest

from scripts.record_final_tqs_verification import (
    _parse_tqs_number,
    _write_cost_breakdown_csv,
    build_verification_payload,
    update_cost_breakdown_with_verification,
)


def _cost_breakdown():
    return {
        "seed": {
            "cost": 1_009_700.0,
            "steel": 288.0,
            "concrete": 10.0,
            "form_area": 20.0,
            "cost_steel_rs": 3_456.0,
            "cost_concrete_rs": 4_500.0,
            "cost_form_rs": 1_400.0,
        },
        "optimal": {
            "cost": 10_128.394833984374,
            "steel": 253.16290283203125,
            "concrete": 11.1832,
            "form_area": 29.4,
            "cost_steel_rs": 3_037.954833984375,
            "cost_concrete_rs": 5_032.44,
            "cost_form_rs": 2_058.0,
        },
        "reduction_pct": {"cost": 98.9},
    }


def test_parse_tqs_number_supports_brazilian_and_decimal_formats():
    assert _parse_tqs_number("1.234,56") == pytest.approx(1234.56)
    assert _parse_tqs_number("11.18") == pytest.approx(11.18)


def test_builds_verified_cost_from_tqs_steel_and_geometric_quantities(tmp_path):
    report = tmp_path / "RESDES.HTM"
    solution = tmp_path / "solucao_otima.csv"
    report.write_text("report", encoding="utf-8")
    solution.write_text("solution", encoding="utf-8")

    payload = build_verification_payload(
        report_path=report,
        archived_report_path=report,
        report_sha256="abc",
        solution_csv=solution,
        cost_breakdown=_cost_breakdown(),
        tqs_steel_kgf=238.0,
        tqs_concrete_m3=11.18,
        tqs_column_formwork_m2=29.4,
        critical_errors=[],
    )

    assert payload["status"] == "passed"
    assert payload["steel_verification"]["absolute_error_kgf"] == pytest.approx(
        15.16290283203125
    )
    assert payload["steel_verification"][
        "absolute_relative_error_pct"
    ] == pytest.approx(6.37096757648)
    assert payload["verified_cost"][
        "tqs_verified_material_cost_brl"
    ] == pytest.approx(9946.44)
    assert payload["verified_cost"][
        "absolute_relative_error_pct"
    ] == pytest.approx(1.82934631873)


def test_penalized_seed_is_not_reported_as_physical_cost_reduction(tmp_path):
    report = tmp_path / "RESDES.HTM"
    solution = tmp_path / "solucao_otima.csv"
    report.write_text("report", encoding="utf-8")
    solution.write_text("solution", encoding="utf-8")
    original = _cost_breakdown()
    payload = build_verification_payload(
        report_path=report,
        archived_report_path=report,
        report_sha256="abc",
        solution_csv=solution,
        cost_breakdown=original,
        tqs_steel_kgf=238.0,
        tqs_concrete_m3=11.18,
        tqs_column_formwork_m2=29.4,
        critical_errors=[],
    )

    updated = update_cost_breakdown_with_verification(original, payload)

    assert updated["seed"]["penalty"] > 0
    assert updated["reduction_pct"]["cost"] is None
    assert updated["objective_reduction_pct_including_penalty"] == 98.9
    assert updated["comparison_validity"][
        "verified_cost_reduction_vs_seed_pct"
    ] is None

    updated_again = update_cost_breakdown_with_verification(updated, payload)
    assert updated_again["objective_reduction_pct_including_penalty"] == 98.9

    csv_path = tmp_path / "cost_breakdown.csv"
    _write_cost_breakdown_csv(csv_path, updated_again)
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "optimal_tqs_verified" in csv_text
    assert "material_cost" in csv_text
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = {row["metric"]: row for row in csv.DictReader(stream)}
    assert float(rows["material_cost"]["optimal_tqs_verified"]) == pytest.approx(
        9946.44
    )
