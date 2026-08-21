"""Record the independent TQS verification of an optimized design.

This command is intentionally post-processing only: it never launches TQS or
changes a TQS building. It reads an existing ``RESDES.HTM``, checks critical
column-design errors, archives the report, and persists the surrogate-versus-
TQS quantities and verified material cost used by the thesis figures/tables.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from config import paths
from results.resultsext import (
    extract_material_breakdown,
    extract_material_summary,
)
from tqs_interface.tqs_errors import TQSErrorReader
from visualization.nn_diagnostics import OptimizationDiagnosticsPlotter


def _parse_tqs_number(value: str | float | int | None) -> float:
    if value is None:
        raise ValueError("TQS material value is missing.")
    if isinstance(value, (float, int)):
        return float(value)
    normalized = value.strip().replace(" ", "")
    if not normalized or normalized == "-":
        raise ValueError("TQS material value is empty.")
    if "," in normalized and "." in normalized:
        normalized = normalized.replace(".", "").replace(",", ".")
    else:
        normalized = normalized.replace(",", ".")
    return float(normalized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _material_cost(metrics: dict[str, Any]) -> float:
    return float(
        metrics["cost_steel_rs"]
        + metrics["cost_concrete_rs"]
        + metrics["cost_form_rs"]
    )


def _unit_price(component_cost: float, quantity: float, label: str) -> float:
    if quantity <= 0:
        raise ValueError(f"Cannot infer {label} unit price from {quantity}.")
    return float(component_cost / quantity)


def build_verification_payload(
    *,
    report_path: Path,
    archived_report_path: Path,
    report_sha256: str,
    solution_csv: Path,
    cost_breakdown: dict[str, Any],
    tqs_steel_kgf: float,
    tqs_concrete_m3: float,
    tqs_column_formwork_m2: float | None,
    critical_errors: list[dict[str, Any]],
    experiment_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable final-verification record from validated inputs."""
    optimal = cost_breakdown["optimal"]
    predicted_steel = float(optimal["steel"])
    geometric_concrete = float(optimal["concrete"])
    column_formwork = float(optimal["form_area"])

    steel_price = _unit_price(
        float(optimal["cost_steel_rs"]), predicted_steel, "steel"
    )
    concrete_price = _unit_price(
        float(optimal["cost_concrete_rs"]), geometric_concrete, "concrete"
    )
    formwork_price = _unit_price(
        float(optimal["cost_form_rs"]), column_formwork, "formwork"
    )

    predicted_material_cost = _material_cost(optimal)
    verified_steel_cost = tqs_steel_kgf * steel_price
    verified_concrete_cost = geometric_concrete * concrete_price
    verified_formwork_cost = column_formwork * formwork_price
    verified_material_cost = (
        verified_steel_cost + verified_concrete_cost + verified_formwork_cost
    )

    steel_signed_error = predicted_steel - tqs_steel_kgf
    steel_abs_error = abs(steel_signed_error)
    steel_relative_error = (
        steel_abs_error / tqs_steel_kgf * 100.0 if tqs_steel_kgf else None
    )
    cost_signed_error = predicted_material_cost - verified_material_cost
    cost_abs_error = abs(cost_signed_error)
    cost_relative_error = (
        cost_abs_error / verified_material_cost * 100.0
        if verified_material_cost
        else None
    )

    steel_test_reference = None
    if experiment_summary:
        steel_test_reference = (
            experiment_summary.get("final_metrics", {}).get("steel")
        )

    return {
        "verification_format_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not critical_errors else "failed",
        "scope": "column reinforcement and optimized-design material cost",
        "building_name": report_path.parent.parent.name,
        "solution_csv": {
            "path": str(solution_csv.resolve()),
            "sha256": _sha256(solution_csv) if solution_csv.exists() else None,
        },
        "tqs_report": {
            "source_path": str(report_path.resolve()),
            "archived_path": str(archived_report_path.resolve()),
            "sha256": report_sha256,
        },
        "structural_validity": {
            "checked_scopes": ["PILAR"],
            "critical_error_class": 2,
            "critical_error_count": len(critical_errors),
            "critical_errors": critical_errors,
            "is_valid": not critical_errors,
        },
        "steel_verification": {
            "surrogate_kgf": predicted_steel,
            "tqs_kgf": tqs_steel_kgf,
            "prediction_minus_tqs_kgf": steel_signed_error,
            "absolute_error_kgf": steel_abs_error,
            "absolute_relative_error_pct": steel_relative_error,
            "prediction_is_conservative": steel_signed_error >= 0.0,
            "test_set_reference": steel_test_reference,
        },
        "quantity_verification": {
            "concrete": {
                "geometric_objective_m3": geometric_concrete,
                "tqs_report_m3": tqs_concrete_m3,
                "note": (
                    "The objective retains the full-precision geometric value; "
                    "the TQS report is rounded to two decimals."
                ),
            },
            "column_formwork": {
                "geometric_objective_m2": column_formwork,
                "tqs_report_m2": tqs_column_formwork_m2,
            },
        },
        "verified_cost": {
            "currency": "BRL",
            "unit_prices": {
                "steel_brl_per_kgf_numeric": steel_price,
                "concrete_brl_per_m3": concrete_price,
                "column_formwork_brl_per_m2": formwork_price,
            },
            "surrogate_material_cost_brl": predicted_material_cost,
            "tqs_verified_material_cost_brl": verified_material_cost,
            "components_brl": {
                "steel_tqs": verified_steel_cost,
                "concrete_geometric": verified_concrete_cost,
                "column_formwork_geometric": verified_formwork_cost,
            },
            "surrogate_minus_verified_brl": cost_signed_error,
            "absolute_error_brl": cost_abs_error,
            "absolute_relative_error_pct": cost_relative_error,
            "penalty_brl": 0.0 if not critical_errors else None,
        },
    }


def update_cost_breakdown_with_verification(
    cost_breakdown: dict[str, Any], verification: dict[str, Any]
) -> dict[str, Any]:
    """Augment cost reporting without treating a penalty as construction cost."""
    for key in ("seed", "optimal"):
        metrics = cost_breakdown[key]
        material_cost = _material_cost(metrics)
        objective_cost = float(metrics["cost"])
        metrics["material_cost"] = material_cost
        metrics["penalty"] = max(0.0, objective_cost - material_cost)
        metrics["objective_cost"] = objective_cost

    old_objective_reduction = cost_breakdown.get(
        "objective_reduction_pct_including_penalty"
    )
    if old_objective_reduction is None:
        old_objective_reduction = cost_breakdown.get("reduction_pct", {}).get(
            "cost"
        )
    seed_has_penalty = cost_breakdown["seed"]["penalty"] > 1e-6
    optimal_has_penalty = cost_breakdown["optimal"]["penalty"] > 1e-6

    cost_breakdown["objective_reduction_pct_including_penalty"] = (
        old_objective_reduction
    )
    if "reduction_pct" in cost_breakdown:
        cost_breakdown["reduction_pct"]["cost"] = (
            None
            if seed_has_penalty or optimal_has_penalty
            else old_objective_reduction
        )

    verified = verification["verified_cost"]
    cost_breakdown["verified_optimal"] = {
        "status": verification["status"],
        "steel_tqs_kgf": verification["steel_verification"]["tqs_kgf"],
        "concrete_geometric_m3": verification["quantity_verification"][
            "concrete"
        ]["geometric_objective_m3"],
        "column_formwork_m2": verification["quantity_verification"][
            "column_formwork"
        ]["geometric_objective_m2"],
        "material_cost_brl": verified["tqs_verified_material_cost_brl"],
        "cost_steel_brl": verified["components_brl"]["steel_tqs"],
        "cost_concrete_brl": verified["components_brl"][
            "concrete_geometric"
        ],
        "cost_column_formwork_brl": verified["components_brl"][
            "column_formwork_geometric"
        ],
        "penalty_brl": verified["penalty_brl"],
    }
    cost_breakdown["comparison_validity"] = {
        "seed_is_feasible": not seed_has_penalty,
        "optimal_is_tqs_verified": verification["status"] == "passed",
        "verified_cost_reduction_vs_seed_pct": None,
        "note": (
            "No physical cost reduction is reported because the seed received "
            "a structural-invalidity penalty and is not a feasible baseline."
            if seed_has_penalty
            else "The seed is unpenalized; a verified comparison may be reported."
        ),
    }
    if not seed_has_penalty and verification["status"] == "passed":
        seed_material_cost = cost_breakdown["seed"]["material_cost"]
        verified_cost = verified["tqs_verified_material_cost_brl"]
        cost_breakdown["comparison_validity"][
            "verified_cost_reduction_vs_seed_pct"
        ] = (seed_material_cost - verified_cost) / seed_material_cost * 100.0

    return cost_breakdown


def _write_verification_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        ("structural_valid", payload["structural_validity"]["is_valid"], "-"),
        ("critical_error_count", payload["structural_validity"]["critical_error_count"], "count"),
        ("steel_surrogate", payload["steel_verification"]["surrogate_kgf"], "kgf"),
        ("steel_tqs", payload["steel_verification"]["tqs_kgf"], "kgf"),
        ("steel_absolute_error", payload["steel_verification"]["absolute_error_kgf"], "kgf"),
        ("steel_absolute_relative_error", payload["steel_verification"]["absolute_relative_error_pct"], "%"),
        ("concrete_geometric", payload["quantity_verification"]["concrete"]["geometric_objective_m3"], "m3"),
        ("concrete_tqs_report", payload["quantity_verification"]["concrete"]["tqs_report_m3"], "m3"),
        ("column_formwork_geometric", payload["quantity_verification"]["column_formwork"]["geometric_objective_m2"], "m2"),
        ("column_formwork_tqs_report", payload["quantity_verification"]["column_formwork"]["tqs_report_m2"], "m2"),
        ("surrogate_material_cost", payload["verified_cost"]["surrogate_material_cost_brl"], "BRL"),
        ("tqs_verified_material_cost", payload["verified_cost"]["tqs_verified_material_cost_brl"], "BRL"),
        ("cost_absolute_relative_error", payload["verified_cost"]["absolute_relative_error_pct"], "%"),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value", "unit"])
        writer.writerows(rows)


def _write_cost_breakdown_csv(
    path: Path, cost_breakdown: dict[str, Any]
) -> None:
    """Write a thesis-safe cost table with penalties explicitly separated."""
    seed = cost_breakdown["seed"]
    optimal = cost_breakdown["optimal"]
    verified = cost_breakdown["verified_optimal"]
    comparison_valid = cost_breakdown["comparison_validity"][
        "seed_is_feasible"
    ]
    rows = [
        (
            "objective_cost",
            "Objective function value (BRL)",
            seed["objective_cost"],
            optimal["objective_cost"],
            None,
            cost_breakdown["objective_reduction_pct_including_penalty"],
            False,
        ),
        (
            "penalty",
            "Structural-invalidity penalty (BRL)",
            seed["penalty"],
            optimal["penalty"],
            verified["penalty_brl"],
            None,
            False,
        ),
        (
            "material_cost",
            "Physical material cost (BRL)",
            seed["material_cost"],
            optimal["material_cost"],
            verified["material_cost_brl"],
            cost_breakdown["comparison_validity"][
                "verified_cost_reduction_vs_seed_pct"
            ],
            comparison_valid,
        ),
        (
            "steel",
            "Column reinforcement steel weight (kgf)",
            seed["steel"],
            optimal["steel"],
            verified["steel_tqs_kgf"],
            None,
            comparison_valid,
        ),
        (
            "concrete",
            "Concrete volume (m3)",
            seed["concrete"],
            optimal["concrete"],
            verified["concrete_geometric_m3"],
            None,
            comparison_valid,
        ),
        (
            "column_formwork",
            "Column formwork area (m2)",
            seed["form_area"],
            optimal["form_area"],
            verified["column_formwork_m2"],
            None,
            comparison_valid,
        ),
        (
            "steel_cost",
            "Column reinforcement steel cost (BRL)",
            seed["cost_steel_rs"],
            optimal["cost_steel_rs"],
            verified["cost_steel_brl"],
            None,
            comparison_valid,
        ),
        (
            "concrete_cost",
            "Concrete cost (BRL)",
            seed["cost_concrete_rs"],
            optimal["cost_concrete_rs"],
            verified["cost_concrete_brl"],
            None,
            comparison_valid,
        ),
        (
            "column_formwork_cost",
            "Column formwork cost (BRL)",
            seed["cost_form_rs"],
            optimal["cost_form_rs"],
            verified["cost_column_formwork_brl"],
            None,
            comparison_valid,
        ),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "metric",
                "label",
                "seed",
                "optimal_surrogate",
                "optimal_tqs_verified",
                "reduction_pct",
                "feasible_baseline_comparison",
            ]
        )
        writer.writerows(rows)


def record_final_verification(
    *,
    report_path: Path,
    cost_breakdown_path: Path,
    solution_csv: Path,
    output_dir: Path,
    experiment_dir: Path | None = None,
) -> dict[str, Any]:
    report_path = report_path.resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"TQS report not found: {report_path}")
    if not cost_breakdown_path.exists():
        raise FileNotFoundError(
            f"Cost breakdown not found: {cost_breakdown_path}"
        )

    steel_raw, concrete_raw = extract_material_summary(report_path)
    tqs_steel = _parse_tqs_number(steel_raw)
    tqs_concrete = _parse_tqs_number(concrete_raw)
    material_rows = extract_material_breakdown(report_path)
    column_formwork_raw = material_rows.get("columns", {}).get("formwork_m2")
    column_formwork = (
        _parse_tqs_number(column_formwork_raw)
        if column_formwork_raw is not None
        else None
    )

    building_name = report_path.parent.parent.name
    errors = TQSErrorReader().get_critical_errors(
        building_name=building_name,
        strict=True,
        target_scopes=("PILAR",),
    )
    critical_errors = [
        {
            "element_number": int(error.elm_number),
            "description": str(error.error_header),
        }
        for error in errors
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_hash = _sha256(report_path)
    archive_dir = output_dir / "final_tqs_verification"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_report = archive_dir / f"RESDES_{report_hash[:12]}.HTM"
    if not archived_report.exists():
        shutil.copy2(report_path, archived_report)

    with cost_breakdown_path.open("r", encoding="utf-8") as stream:
        cost_breakdown = json.load(stream)

    experiment_summary = None
    if experiment_dir is not None:
        summary_path = experiment_dir / "metrics" / "summary.json"
        if summary_path.exists():
            experiment_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    payload = build_verification_payload(
        report_path=report_path,
        archived_report_path=archived_report,
        report_sha256=report_hash,
        solution_csv=solution_csv,
        cost_breakdown=cost_breakdown,
        tqs_steel_kgf=tqs_steel,
        tqs_concrete_m3=tqs_concrete,
        tqs_column_formwork_m2=column_formwork,
        critical_errors=critical_errors,
        experiment_summary=experiment_summary,
    )

    verification_json = output_dir / "final_tqs_verification.json"
    verification_csv = output_dir / "final_tqs_verification.csv"
    verification_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_verification_csv(verification_csv, payload)

    updated_breakdown = update_cost_breakdown_with_verification(
        cost_breakdown, payload
    )
    cost_breakdown_path.write_text(
        json.dumps(updated_breakdown, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_cost_breakdown_csv(
        cost_breakdown_path.with_suffix(".csv"), updated_breakdown
    )

    plots_dir = output_dir / "plots"
    log_path = output_dir / "optimization_log.json"
    plotter = OptimizationDiagnosticsPlotter(plots_dir, log_path)
    plotter.plot_surrogate_vs_tqs(
        surrogate_steel=payload["steel_verification"]["surrogate_kgf"],
        tqs_steel=payload["steel_verification"]["tqs_kgf"],
        surrogate_concrete=payload["quantity_verification"]["concrete"][
            "geometric_objective_m3"
        ],
        tqs_concrete=payload["quantity_verification"]["concrete"][
            "tqs_report_m3"
        ],
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record the final TQS verification of solucao_otima.csv."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--cost-breakdown",
        type=Path,
        default=paths.RESULTS_DIR / "cost_breakdown.json",
    )
    parser.add_argument(
        "--solution-csv",
        type=Path,
        default=paths.RESULTS_DIR / "solucao_otima.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=paths.RESULTS_DIR
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Optional experiment directory used to embed test-set references.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = record_final_verification(
        report_path=args.report,
        cost_breakdown_path=args.cost_breakdown,
        solution_csv=args.solution_csv,
        output_dir=args.output_dir,
        experiment_dir=args.experiment_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
