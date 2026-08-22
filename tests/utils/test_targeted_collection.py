import json

import pytest

import scripts.run_targeted_collection as runner

from utils.targeted_collection import (
    initial_checkpoint,
    load_or_initialize_checkpoint,
    pending_candidates,
    record_attempt,
    summarize_checkpoint,
    validate_dedicated_building_name,
    write_json_atomic,
)


def _candidates():
    return [
        {"candidate_id": "dev-a", "sequence": 1, "role": "development"},
        {"candidate_id": "dev-b", "sequence": 2, "role": "development"},
        {"candidate_id": "test-a", "sequence": 3, "role": "protected_evaluation"},
    ]


def _checkpoint():
    return initial_checkpoint(
        run_id="run-1",
        candidate_plan_sha256="plan-hash",
        manifest_sha256="manifest-hash",
        building_slot="Targeted_01",
    )


def test_protected_building_names_are_rejected():
    with pytest.raises(ValueError, match="protected"):
        validate_dedicated_building_name(
            "OptimizedBuilding",
            forbidden_names={"OptimizedBuilding", "OptimizedBuilding_01"},
        )
    assert validate_dedicated_building_name(
        "TargetedPilot", forbidden_names={"OptimizedBuilding"}
    ) == "TargetedPilot_01"


def test_pending_candidates_keep_protected_cases_locked():
    checkpoint = _checkpoint()
    pending = pending_candidates(
        _candidates(), checkpoint, include_protected=False, retry_failed=False
    )
    assert [row["candidate_id"] for row in pending] == ["dev-a", "dev-b"]


def test_completed_and_failed_candidates_are_not_resubmitted_by_default():
    checkpoint = _checkpoint()
    record_attempt(
        checkpoint,
        candidate_id="dev-a",
        status="tqs_completed",
        result={"structural_validity": {"is_valid": True}},
    )
    record_attempt(
        checkpoint,
        candidate_id="dev-b",
        status="infrastructure_failed",
        error="timeout",
    )
    assert pending_candidates(
        _candidates(), checkpoint, include_protected=False, retry_failed=False
    ) == []
    retry = pending_candidates(
        _candidates(), checkpoint, include_protected=False, retry_failed=True
    )
    assert [row["candidate_id"] for row in retry] == ["dev-b"]


def test_checkpoint_round_trip_and_contract_validation(tmp_path):
    path = tmp_path / "checkpoint.json"
    original = _checkpoint()
    write_json_atomic(path, original)
    loaded = load_or_initialize_checkpoint(
        path,
        run_id="run-1",
        candidate_plan_sha256="plan-hash",
        manifest_sha256="manifest-hash",
        building_slot="Targeted_01",
    )
    assert loaded == original
    assert not (tmp_path / "checkpoint.json.tmp").exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate_plan_sha256"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate_plan_sha256"):
        load_or_initialize_checkpoint(
            path,
            run_id="run-1",
            candidate_plan_sha256="plan-hash",
            manifest_sha256="manifest-hash",
            building_slot="Targeted_01",
        )


def test_summary_separates_development_and_protected_results():
    checkpoint = _checkpoint()
    for candidate_id, valid in (("dev-a", True), ("test-a", False)):
        record_attempt(
            checkpoint,
            candidate_id=candidate_id,
            status="tqs_completed",
            result={"structural_validity": {"is_valid": valid}},
        )
    summary = summarize_checkpoint(checkpoint, _candidates())
    assert summary["development"] == {"processed": 1, "valid": 1, "invalid": 0}
    assert summary["protected_evaluation"] == {"processed": 1, "valid": 0, "invalid": 1}


def test_report_helper_cleanup_terminates_only_new_pids(monkeypatch):
    monkeypatch.setattr(runner, "_process_ids_for_image", lambda _name: {10, 20, 30})
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **_kwargs: calls.append(command))

    runner._terminate_new_report_helpers({10, 30})

    assert calls == [["taskkill", "/F", "/PID", "20"]]
