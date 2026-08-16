"""Compare sequential and two-building TQS execution on identical geometries.

This validation never writes to the production collection checkpoint. It
creates three isolated copies of a validated TQS template: one sequential
baseline slot and two simultaneous slots. The same known-valid geometries are
processed in both modes and compared before two-worker collection is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from TQS import TQSBuild

from config.paths import OUTPUTS_DIR, PROJECT_ROOT, SEED_VECTOR_CSV
from config.settings import NeuralNetConfig
from geometry.length_input_processor import LengthProcessor
from tqs_interface.tqs_errors import TQSErrorReader
from tqs_interface.tqs_worker_pool import TQSWorkerPool, WorkerResult
from utils.feature_engineer import FeatureEngineer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _ntqshtm_running() -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq NTQSHTM.EXE"],
        capture_output=True,
        text=True,
    )
    return "NTQSHTM.EXE" in result.stdout.upper()


def _terminate_idle_ntqshtm() -> bool:
    """Clean only a process created by this pilot while no jobs are active."""
    if not _ntqshtm_running():
        return False
    subprocess.run(
        ["taskkill", "/F", "/IM", "NTQSHTM.EXE", "/T"],
        capture_output=True,
        text=True,
    )
    time.sleep(0.2)
    return True


def _provision_slot(
    source_name: str,
    slot_name: str,
    *,
    allow_existing: bool,
) -> None:
    probe = TQSBuild.Building()
    if probe.file.Open(slot_name) == 0:
        if allow_existing:
            return
        raise RuntimeError(
            f"TQS slot '{slot_name}' already exists. Choose a fresh --slot-base "
            "or explicitly pass --reuse-slots."
        )

    source = TQSBuild.Building()
    if source.file.Open(source_name) != 0:
        raise RuntimeError(f"Could not open TQS template '{source_name}'.")
    if source.file.SaveAs(slot_name) != 0:
        raise RuntimeError(
            f"Could not provision TQS slot '{slot_name}' from '{source_name}'."
        )


def _load_cases(checkpoint_path: Path, sample_count: int) -> list[dict[str, Any]]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("collection_complete") is not True:
        raise RuntimeError("The source checkpoint is not a complete collection.")
    if checkpoint.get("feature_schema_version") != int(
        NeuralNetConfig.FEATURE_SCHEMA_VERSION
    ):
        raise RuntimeError("The source checkpoint feature schema is not current.")
    if checkpoint.get("feature_names") != FeatureEngineer.feature_names():
        raise RuntimeError("The source checkpoint feature names are not current.")
    if checkpoint.get("seed_sha256") != _sha256(Path(SEED_VECTOR_CSV)):
        raise RuntimeError("The source checkpoint does not match BuildingInput.csv.")

    configurations = checkpoint.get("generated_valid_configurations", [])
    outputs = checkpoint.get("output_values", [])
    valid_case_count = sample_count - 1
    if len(configurations) != len(outputs) or len(configurations) < valid_case_count:
        raise RuntimeError("The source checkpoint arrays are missing or misaligned.")
    targets = np.asarray([row[0] for row in outputs], dtype=float)
    if not np.isfinite(targets).all() or np.any(targets <= 0.0):
        raise RuntimeError("The source checkpoint contains invalid steel targets.")

    order = np.argsort(targets, kind="mergesort")
    positions = np.linspace(0, len(order) - 1, valid_case_count, dtype=int)
    indexes = [int(order[position]) for position in positions]
    if len(set(indexes)) != valid_case_count:
        raise RuntimeError("Could not select distinct concurrency cases.")

    # BuildingInput.csv itself was evaluated during the pilot but was not
    # included in generated_valid_configurations when structurally invalid.
    # Re-evaluating it exercises concurrent DLL validity classification rather
    # than testing only known-valid material reports.
    seed_segments = LengthProcessor(str(SEED_VECTOR_CSV)).read_length_from_csv()
    if not seed_segments:
        raise RuntimeError("Could not load the seed geometry for the invalid case.")
    cases = [
        {
            "case_id": 1,
            "source_index": None,
            "stored_steel_kgf": None,
            "segments": seed_segments,
            "origin": "seed_validity_probe",
        }
    ]
    cases.extend(
        {
            "case_id": case_id,
            "source_index": index,
            "stored_steel_kgf": float(targets[index]),
            "segments": configurations[index],
            "origin": "known_valid_checkpoint_case",
        }
        for case_id, index in enumerate(indexes, start=2)
    )
    return cases


def _prepare_jobs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processor = LengthProcessor()
    expected_feature_count = int(NeuralNetConfig.INPUT_SIZE)
    prepared = []
    for case in cases:
        columns, beams = processor.process_segments(case["segments"])
        if not columns:
            raise RuntimeError(f"Case {case['case_id']} produced no columns.")
        features = FeatureEngineer(columns, beams).extract_features()
        if len(features) != expected_feature_count or not all(
            math.isfinite(float(value)) for value in features
        ):
            raise RuntimeError(
                f"Case {case['case_id']} produced an invalid feature vector."
            )
        prepared.append(
            {
                **case,
                "columns": columns,
                "beams": beams,
                "feature_count": len(features),
            }
        )
    return prepared


def _record(case: dict[str, Any], result: WorkerResult) -> dict[str, Any]:
    return {
        "case_id": int(case["case_id"]),
        "source_index": (
            int(case["source_index"])
            if case["source_index"] is not None
            else None
        ),
        "origin": case["origin"],
        "stored_steel_kgf": (
            float(case["stored_steel_kgf"])
            if case["stored_steel_kgf"] is not None
            else None
        ),
        "job_id": int(result.job_id),
        "slot": result.slot_name,
        "success": bool(result.success),
        "is_valid": bool(result.is_valid),
        "steel_kgf": float(result.steel) if result.steel is not None else None,
        "concrete_m3": (
            float(result.concrete) if result.concrete is not None else None
        ),
        "elapsed_sec": float(result.elapsed),
        "report_sha256": result.report_sha256,
        "error": result.error,
    }


def _run_sequential(
    cases: list[dict[str, Any]],
    *,
    slot_base: str,
    timeout: int,
) -> tuple[list[dict[str, Any]], float]:
    records = []
    started = time.perf_counter()
    with TQSWorkerPool(
        num_workers=1,
        base_name=slot_base,
        timeout_sec=timeout,
        validity_check_dll=True,
    ) as pool:
        for case in cases:
            job_id = pool.submit(case["columns"], case["beams"])
            result = pool.get_result(timeout=float(timeout) + 60.0)
            if result.job_id != job_id:
                raise RuntimeError(
                    f"Sequential result job {result.job_id} does not match {job_id}."
                )
            records.append(_record(case, result))
    return records, time.perf_counter() - started


def _run_concurrent(
    cases: list[dict[str, Any]],
    *,
    slot_base: str,
    timeout: int,
) -> tuple[list[dict[str, Any]], float]:
    # Reverse the order so each case is evaluated in a different copied slot
    # than its position would naturally suggest in the baseline schedule.
    schedule = list(reversed(cases))
    started = time.perf_counter()
    with TQSWorkerPool(
        num_workers=2,
        base_name=slot_base,
        timeout_sec=timeout,
        validity_check_dll=True,
        allow_simultaneous_tqs=True,
    ) as pool:
        results = pool.map(
            [(case["columns"], case["beams"]) for case in schedule],
            timeout_per_job=float(timeout) + 60.0,
        )
    records = [_record(case, result) for case, result in zip(schedule, results)]
    return records, time.perf_counter() - started


def _compare(
    cases: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    concurrent: list[dict[str, Any]],
    *,
    steel_tolerance: float,
    concrete_tolerance: float,
) -> tuple[list[dict[str, Any]], bool]:
    baseline_by_case = {record["case_id"]: record for record in baseline}
    concurrent_by_case = {record["case_id"]: record for record in concurrent}
    comparisons = []
    for case in cases:
        case_id = case["case_id"]
        sequential = baseline_by_case[case_id]
        simultaneous = concurrent_by_case[case_id]
        usable = (
            sequential["success"]
            and simultaneous["success"]
            and sequential["steel_kgf"] is not None
            and simultaneous["steel_kgf"] is not None
            and sequential["concrete_m3"] is not None
            and simultaneous["concrete_m3"] is not None
        )
        steel_delta = (
            abs(simultaneous["steel_kgf"] - sequential["steel_kgf"])
            if usable
            else None
        )
        concrete_delta = (
            abs(simultaneous["concrete_m3"] - sequential["concrete_m3"])
            if usable
            else None
        )
        passed = bool(
            usable
            and sequential["is_valid"] == simultaneous["is_valid"]
            and steel_delta <= steel_tolerance
            and concrete_delta <= concrete_tolerance
            and sequential["report_sha256"]
            and simultaneous["report_sha256"]
        )
        comparisons.append(
            {
                "case_id": case_id,
                "source_index": case["source_index"],
                "origin": case["origin"],
                "sequential_slot": sequential["slot"],
                "simultaneous_slot": simultaneous["slot"],
                "validity_matches": (
                    sequential["is_valid"] == simultaneous["is_valid"]
                ),
                "steel_absolute_delta_kgf": steel_delta,
                "concrete_absolute_delta_m3": concrete_delta,
                "steel_tolerance_kgf": steel_tolerance,
                "concrete_tolerance_m3": concrete_tolerance,
                "passed": passed,
            }
        )
    return comparisons, all(item["passed"] for item in comparisons)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_simultaneous_tqs:
        raise RuntimeError(
            "Explicit acknowledgement is required: pass "
            "--confirm-simultaneous-tqs."
        )
    if not 4 <= args.samples <= 10:
        raise ValueError("Concurrency pilot samples must be between 4 and 10.")
    if args.timeout <= 0 or args.minimum_speedup <= 0.0:
        raise ValueError("Timeout and minimum speedup must be positive.")
    if args.steel_tolerance < 0.0 or args.concrete_tolerance < 0.0:
        raise ValueError("Comparison tolerances cannot be negative.")

    active_locks = list(Path(OUTPUTS_DIR).glob(".*_collection.lock"))
    if active_locks:
        raise RuntimeError(
            "A collection lock exists; concurrency validation must not run "
            f"beside production collection: {active_locks}"
        )
    if _ntqshtm_running():
        raise RuntimeError(
            "NTQSHTM.EXE is already running. Close or finish the current TQS "
            "processing before starting the isolated pilot."
        )

    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    failure_path = output_dir / "failure.json"
    if summary_path.exists() or failure_path.exists():
        raise RuntimeError(
            f"A previous pilot result exists in '{output_dir}'. Choose a fresh "
            "--output-dir so evidence is not overwritten."
        )

    error_reader = TQSErrorReader()
    if not error_reader._dlls_available():
        raise RuntimeError("TQS validity DLLs are unavailable; pilot aborted.")

    checkpoint_path = Path(args.checkpoint).resolve()
    cases = _prepare_jobs(_load_cases(checkpoint_path, args.samples))
    baseline_base = f"{args.slot_base}Baseline"
    baseline_slot = f"{baseline_base}_01"
    concurrent_slots = [f"{args.slot_base}_01", f"{args.slot_base}_02"]
    for slot in [baseline_slot, *concurrent_slots]:
        _provision_slot(
            args.template,
            slot,
            allow_existing=args.reuse_slots,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline, baseline_wall = _run_sequential(
        cases,
        slot_base=baseline_base,
        timeout=args.timeout,
    )
    cleaned_between_phases = _terminate_idle_ntqshtm()
    concurrent, concurrent_wall = _run_concurrent(
        cases,
        slot_base=args.slot_base,
        timeout=args.timeout,
    )
    cleaned_after_pilot = _terminate_idle_ntqshtm()

    comparisons, consistent = _compare(
        cases,
        baseline,
        concurrent,
        steel_tolerance=args.steel_tolerance,
        concrete_tolerance=args.concrete_tolerance,
    )
    used_slots = sorted({record["slot"] for record in concurrent})
    both_slots_used = used_slots == sorted(concurrent_slots)
    invalid_cases_exercised = sum(
        1 for record in baseline if record["success"] and not record["is_valid"]
    )
    speedup = baseline_wall / concurrent_wall if concurrent_wall > 0.0 else None
    worthwhile = speedup is not None and speedup >= args.minimum_speedup
    approved = (
        consistent
        and both_slots_used
        and worthwhile
        and invalid_cases_exercised >= 1
    )
    if approved:
        recommendation = "approve_two_workers_for_extended_collection_pilot"
    elif consistent:
        recommendation = "keep_one_worker_because_speedup_is_insufficient"
    else:
        recommendation = "reject_simultaneous_tqs_execution"

    summary = {
        "status": "passed" if approved else "failed",
        "training_executed": False,
        "production_checkpoint_modified": False,
        "recommendation": recommendation,
        "consistency_passed": consistent,
        "both_concurrent_slots_used": both_slots_used,
        "invalid_cases_exercised": invalid_cases_exercised,
        "used_concurrent_slots": used_slots,
        "sample_count": len(cases),
        "baseline_worker_count": 1,
        "concurrent_worker_count": 2,
        "baseline_wall_sec": baseline_wall,
        "concurrent_wall_sec": concurrent_wall,
        "speedup": speedup,
        "minimum_required_speedup": args.minimum_speedup,
        "baseline_throughput_samples_per_hour": (
            3600.0 * len(cases) / baseline_wall
        ),
        "concurrent_throughput_samples_per_hour": (
            3600.0 * len(cases) / concurrent_wall
        ),
        "timeout_recovery_validated": False,
        "global_timeout_kill_risk": (
            "A timeout still terminates all NTQSHTM.EXE instances; any pilot "
            "timeout rejects simultaneous production use."
        ),
        "idle_process_cleaned_between_phases": cleaned_between_phases,
        "idle_process_cleaned_after_pilot": cleaned_after_pilot,
        "source_checkpoint": _display_path(checkpoint_path),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "template": args.template,
        "baseline_slot": baseline_slot,
        "concurrent_slots": concurrent_slots,
        "comparisons": comparisons,
        "baseline_executions": baseline,
        "concurrent_executions": concurrent,
    }
    _write_json_atomic(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(
            Path(PROJECT_ROOT)
            / "outputs"
            / "experiments"
            / "20260815-213341_Coleta_com_230_amostras"
            / "checkpoint.json"
        ),
    )
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--template", default="TrainBuild815_01")
    parser.add_argument("--slot-base", default="Concur816")
    parser.add_argument("--steel-tolerance", type=float, default=0.5)
    parser.add_argument("--concrete-tolerance", type=float, default=0.001)
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(PROJECT_ROOT)
            / "outputs"
            / "validation"
            / "tqs_concurrency_pilot"
        ),
    )
    parser.add_argument("--reuse-slots", action="store_true")
    parser.add_argument("--confirm-simultaneous-tqs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        result = run_validation(parsed_args)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output = Path(parsed_args.output_dir).resolve() / "failure.json"
        if not output.exists():
            _write_json_atomic(output, failure)
        raise
