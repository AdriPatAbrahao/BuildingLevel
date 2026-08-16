import json
from types import SimpleNamespace

import pytest

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
    assert checkpoint["feature_schema_version"] == 10
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
