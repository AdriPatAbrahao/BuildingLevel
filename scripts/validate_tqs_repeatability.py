"""Validate repeatability of TQS steel labels without training a model.

The test reprocesses three known-valid geometries from a completed collection:
the seed, the minimum-steel case, and the maximum-steel case.  Executions are
interleaved by round so that the test also exposes state leaking from the
previous model processed in the same isolated TQS slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from TQS import TQSBuild

from config.paths import PROJECT_ROOT, TQS_OUTPUT_DIR
from config.settings import NeuralNetConfig
from geometry.length_input_processor import LengthProcessor
from tqs_interface.tqs_errors import TQSErrorReader
from tqs_interface.tqs_worker_pool import TQSWorkerPool
from utils.feature_engineer import FeatureEngineer
from utils.geometric_calculator import get_geometric_concrete_volume


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _provision_slot(source_name: str, slot_name: str, *, allow_existing: bool) -> None:
    probe = TQSBuild.Building()
    if probe.file.Open(slot_name) == 0:
        if allow_existing:
            return
        raise RuntimeError(
            f"Validation slot '{slot_name}' already exists; choose a fresh --slot-base."
        )

    source = TQSBuild.Building()
    if source.file.Open(source_name) != 0:
        raise RuntimeError(f"Could not open TQS template '{source_name}'.")
    if source.file.SaveAs(slot_name) != 0:
        raise RuntimeError(
            f"Could not provision TQS slot '{slot_name}' from '{source_name}'."
        )


def _load_source_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("collection_complete") is not True:
        raise RuntimeError("Source checkpoint is not marked as a complete collection.")

    feature_vectors = checkpoint.get("feature_vectors")
    output_values = checkpoint.get("output_values")
    configurations = checkpoint.get("generated_valid_configurations")
    if not all(isinstance(values, list) for values in (
        feature_vectors,
        output_values,
        configurations,
    )):
        raise RuntimeError("Source checkpoint is missing aligned collection arrays.")
    lengths = {len(feature_vectors), len(output_values), len(configurations)}
    if lengths == {0} or len(lengths) != 1:
        raise RuntimeError(
            "Feature, target, and geometry arrays in the checkpoint are not aligned."
        )

    expected_names = FeatureEngineer.feature_names()
    if checkpoint.get("feature_names") != expected_names:
        raise RuntimeError("Checkpoint feature names differ from the current schema.")
    if checkpoint.get("feature_schema_version") != int(
        NeuralNetConfig.FEATURE_SCHEMA_VERSION
    ):
        raise RuntimeError("Checkpoint feature schema version is not current.")

    for index, (features, output, segments) in enumerate(zip(
        feature_vectors,
        output_values,
        configurations,
    )):
        if not isinstance(features, list) or len(features) != len(expected_names):
            raise RuntimeError(f"Invalid feature vector at checkpoint index {index}.")
        if not isinstance(output, list) or len(output) != 1:
            raise RuntimeError(f"Invalid steel target at checkpoint index {index}.")
        numeric = [*features, output[0]]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
            raise RuntimeError(f"Non-finite numeric value at checkpoint index {index}.")
        if not isinstance(segments, list) or not segments:
            raise RuntimeError(f"Invalid geometry at checkpoint index {index}.")
    return checkpoint


def _select_cases(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    targets = [float(value[0]) for value in checkpoint["output_values"]]
    indexes = [0, int(np.argmin(targets)), int(np.argmax(targets))]
    labels = ["seed", "minimum_steel", "maximum_steel"]
    if len(set(indexes)) != len(indexes):
        raise RuntimeError(
            "Seed, minimum, and maximum do not identify three distinct geometries."
        )
    return [
        {
            "label": label,
            "source_index": index,
            "reference_steel_kgf": targets[index],
            "segments": checkpoint["generated_valid_configurations"][index],
            "stored_features": checkpoint["feature_vectors"][index],
        }
        for label, index in zip(labels, indexes)
    ]


def _metric_stats(
    values: Iterable[float],
    *,
    reference: float,
    tolerance: float,
) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("At least one value is required.")
    mean = statistics.fmean(numbers)
    value_range = max(numbers) - min(numbers)
    sample_std = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
    deltas = [value - float(reference) for value in numbers]
    return {
        "count": len(numbers),
        "values": numbers,
        "mean": mean,
        "minimum": min(numbers),
        "maximum": max(numbers),
        "range": value_range,
        "sample_standard_deviation": sample_std,
        "coefficient_of_variation_percent": (
            100.0 * sample_std / abs(mean) if mean != 0.0 else None
        ),
        "reference": float(reference),
        "deltas_from_reference": deltas,
        "maximum_absolute_delta_from_reference": max(abs(value) for value in deltas),
        "exact_repeatability": value_range == 0.0,
        "within_repeatability_tolerance": value_range <= float(tolerance),
        "within_reference_tolerance": max(abs(value) for value in deltas)
        <= float(tolerance),
        "tolerance": float(tolerance),
    }


def _summarize(
    *,
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    repeats: int,
    steel_tolerance: float,
    concrete_tolerance: float,
) -> tuple[list[dict[str, Any]], bool]:
    summaries: list[dict[str, Any]] = []
    for case in cases:
        case_records = sorted(
            (record for record in records if record["label"] == case["label"]),
            key=lambda record: record["repetition"],
        )
        if len(case_records) != repeats:
            summaries.append(
                {
                    "label": case["label"],
                    "source_index": case["source_index"],
                    "status": "incomplete",
                    "completed_repetitions": len(case_records),
                    "expected_repetitions": repeats,
                }
            )
            continue

        steel = _metric_stats(
            (record["steel_kgf"] for record in case_records),
            reference=case["reference_steel_kgf"],
            tolerance=steel_tolerance,
        )
        concrete_reference = float(case_records[0]["concrete_m3"])
        concrete = _metric_stats(
            (record["concrete_m3"] for record in case_records),
            reference=concrete_reference,
            tolerance=concrete_tolerance,
        )
        all_valid = all(
            record["is_valid"] and not record["critical_errors"]
            for record in case_records
        )
        passed = (
            all_valid
            and steel["within_repeatability_tolerance"]
            and steel["within_reference_tolerance"]
            and concrete["within_repeatability_tolerance"]
        )
        summaries.append(
            {
                "label": case["label"],
                "source_index": case["source_index"],
                "status": "passed" if passed else "failed",
                "all_structurally_valid": all_valid,
                "steel_kgf": steel,
                "concrete_m3": concrete,
            }
        )
    overall = len(summaries) == len(cases) and all(
        summary["status"] == "passed" for summary in summaries
    )
    return summaries, overall


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 2:
        raise ValueError("Repeatability validation requires at least two repetitions.")
    if args.steel_tolerance < 0 or args.concrete_tolerance < 0:
        raise ValueError("Tolerances cannot be negative.")

    source_path = Path(args.checkpoint).resolve()
    source = _load_source_checkpoint(source_path)
    cases = _select_cases(source)
    source_sha256 = _sha256(source_path)
    slot_name = f"{args.slot_base}_01"

    output_dir = Path(args.output_dir).resolve()
    report_dir = output_dir / "reports"
    checkpoint_path = output_dir / "checkpoint.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    expected_context = {
        "source_checkpoint": _display_path(source_path),
        "source_checkpoint_sha256": source_sha256,
        "feature_schema_version": int(NeuralNetConfig.FEATURE_SCHEMA_VERSION),
        "feature_names": FeatureEngineer.feature_names(),
        "slot": slot_name,
        "repeats": int(args.repeats),
        "steel_tolerance_kgf": float(args.steel_tolerance),
        "concrete_tolerance_m3": float(args.concrete_tolerance),
        "selected_cases": [
            {
                "label": case["label"],
                "source_index": case["source_index"],
                "reference_steel_kgf": case["reference_steel_kgf"],
                "segments_sha256": _json_sha256(case["segments"]),
            }
            for case in cases
        ],
    }

    records: list[dict[str, Any]] = []
    if args.resume:
        if not checkpoint_path.exists():
            raise RuntimeError("--resume was requested, but no validation checkpoint exists.")
        previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for key, expected in expected_context.items():
            if previous.get(key) != expected:
                raise RuntimeError(f"Resume context differs for '{key}'.")
        records = previous.get("executions", [])
    elif checkpoint_path.exists():
        raise RuntimeError(
            f"Output checkpoint already exists at '{checkpoint_path}'; use --resume "
            "or choose a fresh --output-dir."
        )

    record_keys = [(record.get("label"), record.get("repetition")) for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise RuntimeError("Validation checkpoint contains duplicate executions.")

    error_reader = TQSErrorReader()
    if not error_reader._dlls_available():
        raise RuntimeError("TQS validity DLLs are unavailable; validation aborted.")
    _provision_slot(args.template, slot_name, allow_existing=args.resume)

    processor = LengthProcessor()
    prepared: dict[str, dict[str, Any]] = {}
    for case in cases:
        columns, beams = processor.process_segments(case["segments"])
        if not columns or not beams:
            raise RuntimeError(f"Case '{case['label']}' produced incomplete geometry.")
        recomputed = FeatureEngineer(columns, beams).extract_features()
        stored = np.asarray(case["stored_features"], dtype=float)
        current = np.asarray(recomputed, dtype=float)
        if current.shape != stored.shape or not np.allclose(
            current, stored, rtol=1e-10, atol=1e-8
        ):
            maximum_delta = (
                float(np.max(np.abs(current - stored)))
                if current.shape == stored.shape
                else None
            )
            raise RuntimeError(
                f"Case '{case['label']}' no longer reproduces its stored feature "
                f"vector (maximum delta={maximum_delta})."
            )
        prepared[case["label"]] = {
            "columns": columns,
            "beams": beams,
            "feature_maximum_absolute_delta": float(
                np.max(np.abs(current - stored))
            ),
            "geometric_concrete_m3": float(
                get_geometric_concrete_volume(columns, beams)
            ),
        }

    completed = set(record_keys)
    schedule = [
        (repetition, case)
        for repetition in range(1, args.repeats + 1)
        for case in cases
    ]
    with TQSWorkerPool(
        num_workers=1,
        base_name=args.slot_base,
        timeout_sec=args.timeout,
        validity_check_dll=True,
    ) as pool:
        for repetition, case in schedule:
            key = (case["label"], repetition)
            if key in completed:
                continue
            geometry = prepared[case["label"]]
            job_id = pool.submit(geometry["columns"], geometry["beams"])
            result = pool.get_result(timeout=float(args.timeout) + 60.0)
            if result.job_id != job_id:
                raise RuntimeError(
                    f"Unexpected job id {result.job_id}; expected {job_id}."
                )
            if not result.success or result.concrete is None:
                raise RuntimeError(
                    f"TQS failed for {case['label']} repetition {repetition}: "
                    f"{result.error}"
                )

            critical_errors = error_reader.get_critical_errors(slot_name, strict=True)
            independently_valid = len(critical_errors) == 0
            if independently_valid != result.is_valid:
                raise RuntimeError(
                    f"Validity mismatch for {case['label']} repetition {repetition}: "
                    f"worker={result.is_valid}, independent={independently_valid}."
                )

            source_report = (
                Path(TQS_OUTPUT_DIR) / slot_name / "ESPACIAL" / "RESDES.HTM"
            )
            if not source_report.exists():
                raise RuntimeError(f"TQS report was not found at '{source_report}'.")
            archived_report = report_dir / (
                f"{case['label']}_repeat_{repetition:02d}_RESDES.HTM"
            )
            shutil.copy2(source_report, archived_report)

            record = {
                "execution_order": len(records) + 1,
                "label": case["label"],
                "source_index": case["source_index"],
                "repetition": repetition,
                "reference_steel_kgf": case["reference_steel_kgf"],
                "steel_kgf": float(result.steel),
                "steel_delta_from_reference_kgf": (
                    float(result.steel) - case["reference_steel_kgf"]
                ),
                "concrete_m3": float(result.concrete),
                "geometric_concrete_m3": geometry["geometric_concrete_m3"],
                "feature_maximum_absolute_delta": geometry[
                    "feature_maximum_absolute_delta"
                ],
                "columns": len(geometry["columns"]),
                "beams": len(geometry["beams"]),
                "is_valid": bool(result.is_valid),
                "critical_errors": [
                    {
                        "element": error.elm_number,
                        "description": error.error_header,
                    }
                    for error in critical_errors
                ],
                "elapsed_sec": float(result.elapsed),
                "report": _display_path(archived_report),
                "report_sha256": _sha256(archived_report),
            }
            records.append(record)
            completed.add(key)
            _write_json_atomic(
                checkpoint_path,
                {
                    "status": "running",
                    "training_executed": False,
                    **expected_context,
                    "completed_executions": len(records),
                    "target_executions": len(schedule),
                    "executions": records,
                },
            )
            print(
                f"REPEATABILITY {len(records)}/{len(schedule)} "
                f"case={case['label']} repeat={repetition} "
                f"valid={result.is_valid} steel={result.steel:.1f} kgf "
                f"concrete={result.concrete:.4f} m3",
                flush=True,
            )

    case_summaries, passed = _summarize(
        records=records,
        cases=cases,
        repeats=args.repeats,
        steel_tolerance=args.steel_tolerance,
        concrete_tolerance=args.concrete_tolerance,
    )
    summary = {
        "test": 18,
        "scope": "TQS label repeatability before full data collection",
        "status": "passed" if passed else "failed",
        "training_executed": False,
        **expected_context,
        "execution_strategy": (
            "One isolated worker; rounds interleave seed, minimum-steel, and "
            "maximum-steel geometries."
        ),
        "completed_executions": len(records),
        "target_executions": len(schedule),
        "exact_steel_repeatability_all_cases": all(
            case.get("steel_kgf", {}).get("exact_repeatability", False)
            for case in case_summaries
        ),
        "cases": case_summaries,
        "executions": records,
        "interpretation": (
            "TQS labels are repeatable and consistent with the 230-sample checkpoint "
            "within the declared tolerances."
            if passed
            else "TQS labels failed repeatability, checkpoint-consistency, or validity checks."
        ),
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    _write_json_atomic(checkpoint_path, summary)
    (output_dir / "failure.json").unlink(missing_ok=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/experiments/20260815-213341_Coleta_com_230_amostras/"
            "checkpoint.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/validation/teste18/tqs_repeatability",
    )
    parser.add_argument("--slot-base", default="ValRep816B")
    parser.add_argument("--template", default="OptimizedBuilding")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--steel-tolerance", type=float, default=1.0)
    parser.add_argument("--concrete-tolerance", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        result = run_validation(parsed_args)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(0 if result["status"] == "passed" else 1)
    except Exception as exc:
        failure = {
            "status": "failed",
            "training_executed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(
            Path(parsed_args.output_dir).resolve() / "failure.json",
            failure,
        )
        raise
