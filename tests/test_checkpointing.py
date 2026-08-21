import json
from types import SimpleNamespace

import pytest

from config.settings import ParallelConfig, RunConfig
from main import BuildingOptimizer


def _optimizer_stub(tmp_path):
    seed = tmp_path / "seed.csv"
    seed.write_text("seed-content", encoding="utf-8")
    optimizer = BuildingOptimizer.__new__(BuildingOptimizer)
    optimizer.length_processor = SimpleNamespace(csv_path=seed)
    optimizer.exp_manager = SimpleNamespace(run_dir=tmp_path)
    optimizer.current_iteration = 0
    optimizer.num_target_samples = 25
    optimizer._clf_features = [[1.0, 2.0], [3.0, 4.0]]
    optimizer._clf_labels = [0, 1]
    optimizer.generated_valid_configurations = [[{"length": 20.0}]]
    optimizer._seen_segments_hash = {"hash-a", "hash-b"}
    return optimizer, seed


def test_checkpoint_round_trip_preserves_collection_state(tmp_path):
    optimizer, _ = _optimizer_stub(tmp_path)

    optimizer._save_checkpoint(
        [[3.0, 4.0]],
        [[1200.0]],
        1,
        current_iteration=7,
        seed_processed=True,
        collection_complete=False,
    )

    checkpoint = optimizer._load_checkpoint()
    restored = optimizer._restore_collection_state(checkpoint)

    assert checkpoint["checkpoint_version"] == 3
    assert checkpoint["feature_schema_version"] == 11
    assert len(checkpoint["feature_names"]) == 23
    assert checkpoint["python_random_state"]
    assert restored[:4] == ([[3.0, 4.0]], [[1200.0]], 1, 7)
    assert restored[4] is True
    assert optimizer._clf_labels == [0, 1]
    assert optimizer._seen_segments_hash == {"hash-a", "hash-b"}
    assert not (tmp_path / "checkpoint.json.tmp").exists()


def test_checkpoint_resume_rejects_changed_seed(tmp_path):
    optimizer, seed = _optimizer_stub(tmp_path)
    optimizer._save_checkpoint([], [], 0)
    seed.write_text("changed-seed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="seed differs"):
        optimizer._load_checkpoint()


def test_checkpoint_resume_rejects_changed_feature_schema(tmp_path):
    optimizer, _ = _optimizer_stub(tmp_path)
    optimizer._save_checkpoint([], [], 0)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["feature_schema_version"] = 1
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(RuntimeError, match="feature schema differs"):
        optimizer._load_checkpoint()


def test_checkpoint_resume_rejects_changed_feature_names(tmp_path):
    optimizer, _ = _optimizer_stub(tmp_path)
    optimizer._save_checkpoint([], [], 0)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["feature_names"][0] = "obsolete_feature"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(RuntimeError, match="feature names differ"):
        optimizer._load_checkpoint()


def test_sequential_collection_does_not_reprocess_seed_on_resume(tmp_path, monkeypatch):
    """2026-08-17 regression: unlike the parallel path, the sequential
    ``_collect_training_data`` used to re-score the seed unconditionally on
    every resume and never restored ``_clf_features``/``_clf_labels``/
    ``generated_valid_configurations``/``_seen_segments_hash`` from the
    checkpoint. The first checkpoint save after such a resume would
    overwrite the on-disk classifier history with just the duplicated seed,
    destroying every other sample already collected."""
    monkeypatch.setattr(ParallelConfig, "ENABLED", False)
    monkeypatch.setattr(RunConfig, "CHECKPOINTS_ENABLED", True)
    monkeypatch.setattr(RunConfig, "RESUME_FROM_CHECKPOINT", True)
    monkeypatch.setattr(RunConfig, "MAX_ITERATION_FACTOR", 5)

    optimizer, _ = _optimizer_stub(tmp_path)
    optimizer.analysis_mode = "TQS"
    optimizer.use_geometric_estimate = False
    optimizer.use_vector_input = True
    optimizer._heartbeat_ts = 0.0
    optimizer.segment_plotter = SimpleNamespace(plot_segments=lambda *_a, **_k: None)
    # Target already met by the restored checkpoint, so the generation loop
    # itself never has to run either - isolates the seed/restore behaviour.
    optimizer.num_target_samples = 1

    optimizer._save_checkpoint(
        [[3.0, 4.0]], [[1200.0]], 1,
        current_iteration=7, seed_processed=True, collection_complete=False,
    )

    calls = {"analysis": 0}

    def _fake_get_analysis_results(_segments):
        calls["analysis"] += 1
        return (999.0, 20.0, ["should-not-be-scored-again"], [], True)

    optimizer._get_analysis_results = _fake_get_analysis_results
    optimizer._extract_feature_vector = lambda *_a, **_k: [9.0, 9.0]

    feature_vectors, output_values = optimizer._collect_training_data([{"id": "seed"}])

    assert calls["analysis"] == 0
    assert feature_vectors == [[3.0, 4.0]]
    assert output_values == [[1200.0]]
    assert optimizer._clf_labels == [0, 1]
    assert optimizer._seen_segments_hash == {"hash-a", "hash-b"}
    assert optimizer.generated_valid_configurations == [[{"length": 20.0}]]


def test_sequential_collection_only_counts_genuinely_valid_samples(tmp_path, monkeypatch):
    """2026-08-17 regression: the sequential loop used to increment
    ``processed_valid_configs_count`` for every technically-successful TQS
    run, including invalid designs. That stopped collection as soon as
    enough *attempts* (not valid samples) succeeded, and wrote an inflated
    ``valid_count`` to the checkpoint that no longer matched
    ``len(feature_vectors)``."""
    monkeypatch.setattr(ParallelConfig, "ENABLED", False)
    monkeypatch.setattr(RunConfig, "CHECKPOINTS_ENABLED", True)
    monkeypatch.setattr(RunConfig, "RESUME_FROM_CHECKPOINT", True)
    monkeypatch.setattr(RunConfig, "MAX_ITERATION_FACTOR", 10)

    optimizer, _ = _optimizer_stub(tmp_path)
    optimizer._clf_features = []
    optimizer._clf_labels = []
    optimizer.generated_valid_configurations = []
    optimizer._seen_segments_hash = set()
    optimizer.analysis_mode = "TQS"
    optimizer.use_geometric_estimate = False
    optimizer.use_vector_input = True
    optimizer._heartbeat_ts = 0.0
    optimizer.segment_plotter = SimpleNamespace(plot_segments=lambda *_a, **_k: None)
    optimizer.num_target_samples = 1  # only one TRULY valid sample needed

    # Seed already scored - isolates the while-loop body being tested.
    optimizer._save_checkpoint(
        [], [], 0, current_iteration=0, seed_processed=True, collection_complete=False,
    )

    # Two invalid designs (technically successful TQS runs) followed by one
    # valid design. The buggy code stopped after the first "successful"
    # (but invalid) run because it counted it toward the target.
    results = iter([
        (100.0, 5.0, ["invalid_1"], [], False),
        (200.0, 6.0, ["invalid_2"], [], False),
        (300.0, 7.0, ["valid_1"], [], True),
    ])
    segment_calls = iter([[{"id": "invalid_1"}], [{"id": "invalid_2"}], [{"id": "valid_1"}]])

    optimizer._generate_segment_variation = lambda *_a, **_k: next(segment_calls)
    optimizer._get_analysis_results = lambda _segments: next(results)
    optimizer._extract_feature_vector = lambda columns, _beams: [float(ord(columns[0][0]))]

    feature_vectors, output_values = optimizer._collect_training_data([{"id": "seed"}])

    assert feature_vectors == [[float(ord("v"))]]
    assert output_values == [[300.0]]
    assert optimizer._clf_labels == [0, 0, 1]
