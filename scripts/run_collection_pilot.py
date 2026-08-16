"""Run a small, isolated TQS data-collection pilot without training a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString
from TQS import TQSBuild

from config.constants import DEFAULT_BEAM_WIDTH_CM
from config.paths import PROJECT_ROOT, SEED_VECTOR_CSV, TQS_OUTPUT_DIR
from config.settings import NeuralNetConfig
from config.vector_config import VectorConfig
from geometry.length_input_processor import LengthProcessor
from optimization.design_space import DesignSpace
from tqs_interface.tqs_errors import TQSErrorReader
from tqs_interface.tqs_worker_pool import TQSWorkerPool
from utils.feature_engineer import FeatureEngineer
from utils.geometric_calculator import get_geometric_concrete_volume


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _provision_slot(source_name: str, slot_name: str) -> None:
    probe = TQSBuild.Building()
    if probe.file.Open(slot_name) == 0:
        return

    source = TQSBuild.Building()
    if source.file.Open(source_name) != 0:
        raise RuntimeError(f"Could not open TQS template '{source_name}'.")
    if source.file.SaveAs(slot_name) != 0:
        raise RuntimeError(
            f"Could not provision TQS slot '{slot_name}' from '{source_name}'."
        )


def _candidate_vectors(
    design_space: DesignSpace,
    sample_count: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    lower = design_space.lower_bounds.astype(float)
    upper = design_space.upper_bounds.astype(float)
    midpoint = np.floor(((lower + upper) / 2.0) / 5.0) * 5.0
    midpoint = np.clip(midpoint, lower, upper)

    candidates: list[dict[str, Any]] = [
        {"origin": "minimum", "vector": lower.tolist()},
        {"origin": "midpoint", "vector": midpoint.tolist()},
        {"origin": "maximum", "vector": upper.tolist()},
    ]
    seen = {tuple(item["vector"]) for item in candidates}
    rng = random.Random(random_seed)

    attempts = 0
    while len(candidates) < sample_count:
        attempts += 1
        if attempts > sample_count * 100:
            raise RuntimeError("Could not generate enough unique pilot vectors.")
        vector = []
        for low, high in zip(lower, upper):
            steps = int(math.floor((high - low) / 5.0))
            vector.append(float(low + 5.0 * rng.randint(0, steps)))
        key = tuple(vector)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"origin": "random", "vector": vector})

    return candidates[:sample_count]


def _minimum_clear_span(column_polygons) -> float | None:
    gaps: list[float] = []
    for wall in VectorConfig.WALL_SEGMENTS:
        wall_line = LineString([wall["start"], wall["end"]])
        vertical = abs(wall["end"][0] - wall["start"][0]) < 0.001
        intersections = []
        for column in column_polygons:
            if column.distance(wall_line) >= DEFAULT_BEAM_WIDTH_CM:
                continue
            min_x, min_y, max_x, max_y = column.bounds
            intersections.append(
                (
                    column.centroid.y if vertical else column.centroid.x,
                    min_y if vertical else min_x,
                    max_y if vertical else max_x,
                )
            )
        intersections.sort()
        gaps.extend(
            intersections[index + 1][1] - intersections[index][2]
            for index in range(len(intersections) - 1)
        )
    return min(gaps) if gaps else None


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if not 5 <= args.samples <= 10:
        raise ValueError("Pilot sample count must be between 5 and 10.")

    output_dir = Path(args.output_dir).resolve()
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    seed_path = Path(args.seed_csv).resolve()
    feature_schema_version = int(NeuralNetConfig.FEATURE_SCHEMA_VERSION)
    feature_names = FeatureEngineer.feature_names()
    if len(feature_names) != int(NeuralNetConfig.INPUT_SIZE):
        raise RuntimeError(
            "Feature name count differs from NeuralNetConfig.INPUT_SIZE."
        )
    design_space = DesignSpace(seed_path)
    processor = LengthProcessor(str(seed_path))
    candidates = _candidate_vectors(design_space, args.samples, args.random_seed)

    slot_name = f"{args.slot_base}_01"
    error_reader = TQSErrorReader()
    if not error_reader._dlls_available():
        raise RuntimeError("TQS validity DLLs are unavailable; pilot aborted.")
    _provision_slot(args.template, slot_name)

    checkpoint_path = output_dir / "checkpoint.json"
    records: list[dict[str, Any]] = []
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("seed_sha256") != _sha256(seed_path):
            raise RuntimeError("Checkpoint seed differs from the current seed CSV.")
        if checkpoint.get("feature_schema_version") != feature_schema_version:
            raise RuntimeError(
                "Checkpoint feature schema differs from the current schema."
            )
        if checkpoint.get("feature_names") != feature_names:
            raise RuntimeError(
                "Checkpoint feature names differ from the current feature vector."
            )
        if checkpoint.get("target_samples") != args.samples:
            raise RuntimeError(
                "Checkpoint target sample count differs from the requested pilot."
            )
        records = checkpoint.get("samples", [])

    with TQSWorkerPool(
        num_workers=1,
        base_name=args.slot_base,
        timeout_sec=args.timeout,
        validity_check_dll=True,
    ) as pool:
        for sample_index, candidate in enumerate(candidates, start=1):
            if sample_index <= len(records):
                continue

            vector = np.asarray(candidate["vector"], dtype=float)
            segments = design_space.segments_from_vector(vector)
            columns, beams = processor.process_segments(segments)
            if not columns:
                raise RuntimeError(f"Sample {sample_index} produced no columns.")

            features = FeatureEngineer(columns, beams).extract_features()
            if len(features) != len(feature_names):
                raise RuntimeError(
                    f"Sample {sample_index} feature count differs from schema v"
                    f"{feature_schema_version}."
                )
            geometric_concrete = get_geometric_concrete_volume(columns, beams)
            job_id = pool.submit(columns, beams)
            result = pool.get_result(timeout=float(args.timeout) + 60.0)
            if result.job_id != job_id:
                raise RuntimeError(
                    f"Unexpected job id {result.job_id}; expected {job_id}."
                )
            if not result.success:
                raise RuntimeError(
                    f"TQS sample {sample_index} failed closed: {result.error}"
                )

            critical_errors = error_reader.get_critical_errors(
                slot_name,
                strict=True,
            )
            independently_valid = len(critical_errors) == 0
            if independently_valid != result.is_valid:
                raise RuntimeError(
                    f"Validity mismatch in sample {sample_index}: "
                    f"worker={result.is_valid}, independent={independently_valid}."
                )

            source_report = (
                Path(TQS_OUTPUT_DIR) / slot_name / "ESPACIAL" / "RESDES.HTM"
            )
            archived_report = sample_dir / f"sample_{sample_index:02d}_RESDES.HTM"
            shutil.copy2(source_report, archived_report)

            record = {
                "sample_id": sample_index,
                "origin": candidate["origin"],
                "requested_vector": candidate["vector"],
                "columns": len(columns),
                "beams": len(beams),
                "minimum_clear_span_cm": _minimum_clear_span(columns),
                "feature_count": len(features),
                "features": [float(value) for value in features],
                "steel_kgf": result.steel,
                "tqs_concrete_m3": result.concrete,
                "geometric_concrete_m3": geometric_concrete,
                "concrete_delta_m3": result.concrete - geometric_concrete,
                "is_valid": result.is_valid,
                "critical_errors": [
                    {
                        "element": error.elm_number,
                        "description": error.error_header,
                    }
                    for error in critical_errors
                ],
                "elapsed_sec": result.elapsed,
                "report": str(archived_report.relative_to(PROJECT_ROOT)),
                "report_sha256": _sha256(archived_report),
            }
            records.append(record)
            _write_json_atomic(
                checkpoint_path,
                {
                    "status": "running",
                    "target_samples": args.samples,
                    "completed_samples": len(records),
                    "seed_csv": str(seed_path),
                    "seed_sha256": _sha256(seed_path),
                    "feature_schema_version": feature_schema_version,
                    "feature_names": feature_names,
                    "slot": slot_name,
                    "dll_required": True,
                    "samples": records,
                },
            )
            print(
                f"PILOT_SAMPLE {sample_index}/{args.samples} "
                f"valid={result.is_valid} steel={result.steel:.1f} "
                f"concrete={result.concrete:.4f}",
                flush=True,
            )

    valid_records = [record for record in records if record["is_valid"]]
    invalid_records = [record for record in records if not record["is_valid"]]
    dataset = {
        "feature_schema_version": feature_schema_version,
        "feature_names": feature_names,
        "classifier_features": [record["features"] for record in records],
        "classifier_labels": [1 if record["is_valid"] else 0 for record in records],
        "regression_features": [record["features"] for record in valid_records],
        "steel_targets_kgf": [record["steel_kgf"] for record in valid_records],
    }
    _write_json_atomic(output_dir / "pilot_dataset.json", dataset)

    summary = {
        "status": "complete",
        "training_executed": False,
        "target_samples": args.samples,
        "completed_samples": len(records),
        "valid_samples": len(valid_records),
        "invalid_samples": len(invalid_records),
        "feature_count": len(records[0]["features"]) if records else 0,
        "feature_schema_version": feature_schema_version,
        "feature_names": feature_names,
        "slot": slot_name,
        "worker_count": 1,
        "dll_required": True,
        "seed_csv": str(seed_path),
        "seed_sha256": _sha256(seed_path),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "dataset": str((output_dir / "pilot_dataset.json").relative_to(PROJECT_ROOT)),
        "samples": records,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    _write_json_atomic(checkpoint_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--slot-base", default="ValPilot814")
    parser.add_argument("--template", default="OptimizedBuilding")
    parser.add_argument("--seed-csv", default=str(SEED_VECTOR_CSV))
    parser.add_argument(
        "--output-dir",
        default=str(Path(PROJECT_ROOT) / "outputs" / "validation" / "teste10"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        result = run_pilot(parsed_args)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(Path(parsed_args.output_dir).resolve() / "failure.json", failure)
        raise
