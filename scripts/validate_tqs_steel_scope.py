"""Compare column-only and full beam-detailing TQS material totals."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from TQS import TQSBuild, TQSExec

from config.paths import PROJECT_ROOT, TQS_OUTPUT_DIR
from geometry.length_input_processor import LengthProcessor
from results.resultsext import extract_material_summary
from tqs_interface.tqs_errors import TQSErrorReader
from tqs_interface.tqs_exec import _cleanup_report_files
from tqs_interface.tqs_manager import TQSModelManager
from tqs_interface.tqs_worker_pool import _run_model_with_timeout


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


def _provision_new_slot(source_name: str, slot_name: str) -> None:
    probe = TQSBuild.Building()
    if probe.file.Open(slot_name) == 0:
        raise RuntimeError(
            f"Validation slot '{slot_name}' already exists; choose a fresh name."
        )

    source = TQSBuild.Building()
    if source.file.Open(source_name) != 0:
        raise RuntimeError(f"Could not open TQS template '{source_name}'.")
    if source.file.SaveAs(slot_name) != 0:
        raise RuntimeError(
            f"Could not provision TQS slot '{slot_name}' from '{source_name}'."
        )


def _run_global_mode(building_name: str, *, full_beam_detailing: bool) -> None:
    running = subprocess.getoutput('tasklist /FI "IMAGENAME eq NTQSHTM.EXE"')
    if "NTQSHTM.EXE" in running:
        subprocess.run(
            ["taskkill", "/F", "/IM", "NTQSHTM.EXE", "/T"],
            capture_output=True,
            check=False,
        )
        time.sleep(0.1)

    _cleanup_report_files(building_name)
    job = TQSExec.Job()
    job.EnterTask(
        TQSExec.TaskFolder(building_name, TQSExec.TaskFolder.FOLDER_FRAMES)
    )
    job.EnterTask(
        TQSExec.TaskGlobalProc(
            gridSlabsTrnsf=0,
            frameBeamsTrnsf=1 if full_beam_detailing else 0,
            slabs=0,
            beams=3 if full_beam_detailing else 1,
            columns=2,
        )
    )
    job.EnterTask(TQSExec.TaskStructuralReport())
    job.Execute()


def _evaluate_mode(
    *,
    slot_name: str,
    template: str,
    columns: list,
    beams: list,
    output_dir: Path,
    full_beam_detailing: bool,
    timeout: int,
) -> dict[str, Any]:
    _provision_new_slot(template, slot_name)
    manager = TQSModelManager(building_name=slot_name)
    try:
        if not manager.create_building_model_and_elements(columns, beams):
            raise RuntimeError(f"Could not build validation model '{slot_name}'.")

        _run_model_with_timeout(
            lambda name: _run_global_mode(
                name,
                full_beam_detailing=full_beam_detailing,
            ),
            slot_name,
            timeout,
        )
    finally:
        if manager.model is not None:
            manager.model.file.Close()

    report = Path(TQS_OUTPUT_DIR) / slot_name / "ESPACIAL" / "RESDES.HTM"
    if not report.exists():
        raise RuntimeError(f"TQS report was not produced for '{slot_name}'.")
    steel_raw, concrete_raw = extract_material_summary(report)
    if steel_raw is None or concrete_raw is None:
        raise RuntimeError(f"Could not parse the report for '{slot_name}'.")

    archived = output_dir / f"{slot_name}_RESDES.HTM"
    shutil.copy2(report, archived)
    errors = TQSErrorReader().get_critical_errors(slot_name, strict=True)
    return {
        "slot": slot_name,
        "beams_mode": 3 if full_beam_detailing else 1,
        "columns_mode": 2,
        "frame_beams_transfer": 1 if full_beam_detailing else 0,
        "frame_columns_transfer": 1,
        "steel_kgf": float(str(steel_raw).replace(",", ".")),
        "concrete_m3": float(str(concrete_raw).replace(",", ".")),
        "critical_error_count": len(errors),
        "report": str(archived.relative_to(PROJECT_ROOT)),
        "report_sha256": _sha256(archived),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/experiments/20260815-165953_Coleta_com_10_amostras/checkpoint.json",
    )
    parser.add_argument("--output-dir", default="outputs/validation/teste13/steel_scope")
    parser.add_argument("--template", default="OptimizedBuilding")
    parser.add_argument("--column-slot", default="ValColSteel815")
    parser.add_argument("--full-slot", default="ValFullSteel815")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    configurations = checkpoint.get("generated_valid_configurations", [])
    if not configurations:
        raise RuntimeError("Checkpoint contains no known-valid configuration.")

    segments = configurations[0]
    processor = LengthProcessor()
    columns, beams = processor.process_segments(segments)
    if not columns or not beams:
        raise RuntimeError("Validation geometry must contain columns and beams.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    column_only = _evaluate_mode(
        slot_name=args.column_slot,
        template=args.template,
        columns=columns,
        beams=beams,
        output_dir=output_dir,
        full_beam_detailing=False,
        timeout=args.timeout,
    )
    full = _evaluate_mode(
        slot_name=args.full_slot,
        template=args.template,
        columns=columns,
        beams=beams,
        output_dir=output_dir,
        full_beam_detailing=True,
        timeout=args.timeout,
    )

    steel_difference = full["steel_kgf"] - column_only["steel_kgf"]
    concrete_difference = full["concrete_m3"] - column_only["concrete_m3"]
    passed = (
        column_only["steel_kgf"] > 0
        and steel_difference > 0
        and abs(concrete_difference) <= 0.01
        and column_only["critical_error_count"] == 0
        and full["critical_error_count"] == 0
    )
    summary = {
        "test": 13,
        "scope": "TQS steel target: columns only versus columns plus detailed beams",
        "status": "passed" if passed else "failed",
        "training_executed": False,
        "source_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "source_configuration_index": 0,
        "geometry": {"columns": len(columns), "beams": len(beams)},
        "column_only": column_only,
        "full_beam_detailing": full,
        "comparison": {
            "steel_difference_kgf": steel_difference,
            "concrete_difference_m3": concrete_difference,
            "full_to_column_only_steel_ratio": (
                full["steel_kgf"] / column_only["steel_kgf"]
            ),
            "beam_detailing_share_of_full_total": (
                steel_difference / full["steel_kgf"]
            ),
        },
        "interpretation": (
            "Current TaskGlobalProc excludes beam reinforcement from the material total."
            if passed
            else "The A/B result did not prove that beam reinforcement is excluded."
        ),
        "checkpoint_compatibility": {
            "test12_checkpoint_compatible_with_column_only_target": False,
            "reason": (
                "The test 12 checkpoint was collected before TaskGlobalProc changed "
                "from beams=3 to beams=1 and frameBeamsTrnsf=0."
            ),
            "decision": (
                "Do not resume or train from the test 12 checkpoint. Restart collection "
                "only after the feature schema is finalized."
            ),
        },
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
