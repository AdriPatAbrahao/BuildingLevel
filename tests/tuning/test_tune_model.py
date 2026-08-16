import json

import numpy as np
import pytest

from config.settings import NeuralNetConfig
from tuning.tune_model import (
    candidate_grid,
    development_indices,
    load_checkpoint,
    paired_mae_comparison,
    regression_metrics,
)
from utils.feature_engineer import FeatureEngineer


def _checkpoint(path, samples=230, complete=True):
    X = np.zeros((samples, NeuralNetConfig.INPUT_SIZE), dtype=float)
    y = np.linspace(300.0, 900.0, samples).reshape(-1, 1)
    payload = {
        "feature_schema_version": NeuralNetConfig.FEATURE_SCHEMA_VERSION,
        "feature_names": FeatureEngineer.feature_names(),
        "feature_vectors": X.tolist(),
        "output_values": y.tolist(),
        "collection_complete": complete,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_candidate_grid_contains_current_small_and_tree_models():
    candidates = candidate_grid()
    names = {candidate.name for candidate in candidates}
    assert len(candidates) == 10
    assert "mlp_64_32_d0.1_huber" in names
    assert "mlp_128_128_64_d0.2_mse_current" in names
    assert any(candidate.family == "extra_trees" for candidate in candidates)


def test_checkpoint_loader_is_steel_only_and_schema_strict(tmp_path):
    path = tmp_path / "checkpoint.json"
    _checkpoint(path)
    X, y, _ = load_checkpoint(path)
    assert X.shape == (230, NeuralNetConfig.INPUT_SIZE)
    assert y.shape == (230,)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["feature_names"][0] = "obsolete"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="names/order"):
        load_checkpoint(path)


def test_pilot_uses_all_samples_as_development():
    y = np.linspace(300.0, 900.0, 230)
    development, protected, mode = development_indices(
        y, {"collection_complete": True}
    )
    assert mode == "pilot_only"
    assert np.array_equal(development, np.arange(230))
    assert protected.size == 0


def test_full_collection_protects_new_only_final_test():
    y = np.linspace(300.0, 900.0, 2500)
    development, protected, mode = development_indices(
        y, {"collection_complete": True}
    )
    assert mode == "full_development_only"
    assert len(development) == 2125
    assert len(protected) == 375
    assert np.all(protected >= 230)
    assert not set(development).intersection(set(protected))


def test_incomplete_full_collection_cannot_be_used_for_selection():
    with pytest.raises(ValueError, match="must be complete"):
        development_indices(
            np.linspace(300.0, 900.0, 500),
            {"collection_complete": False},
        )


def test_regression_metrics_measure_structural_underprediction():
    metrics = regression_metrics(
        np.array([100.0, 200.0, 300.0, 400.0]),
        np.array([90.0, 220.0, 270.0, 450.0]),
        low_steel_threshold=200.0,
    )
    assert metrics["mae_kgf"] == pytest.approx(27.5)
    assert metrics["underprediction_rate"] == pytest.approx(0.5)
    assert metrics["mean_underprediction_kgf"] == pytest.approx(10.0)
    assert metrics["low_steel_quartile_mae_kgf"] == pytest.approx(15.0)


def test_paired_mae_comparison_uses_matching_folds():
    rows = [
        {"fold": 1, "candidate": "current", "mae_kgf": 25.0},
        {"fold": 1, "candidate": "challenger", "mae_kgf": 20.0},
        {"fold": 2, "candidate": "current", "mae_kgf": 22.0},
        {"fold": 2, "candidate": "challenger", "mae_kgf": 20.0},
        {"fold": 3, "candidate": "current", "mae_kgf": 21.0},
        {"fold": 3, "candidate": "challenger", "mae_kgf": 22.0},
    ]

    result = paired_mae_comparison(
        rows,
        reference_candidate="current",
        challenger_candidate="challenger",
        bootstrap_runs=200,
        seed=7,
    )

    assert result["mean_mae_improvement_kgf"] == pytest.approx(2.0)
    assert result["challenger_better_fold_fraction"] == pytest.approx(2 / 3)
    assert result["paired_folds"] == 3
