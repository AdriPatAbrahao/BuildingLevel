from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from scripts.build_final_thesis_figures import load_final_plot_inputs
from utils.plot_data_store import PlotDataStore


def _write_complete_source(
    experiment_dir: Path,
    operational_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.arange(60, dtype=float).reshape(20, 3)
    X_test = np.arange(18, dtype=float).reshape(6, 3)
    y_train = np.linspace(200.0, 390.0, 20)
    y_test = np.linspace(220.0, 320.0, 6)
    y_pred = y_test + np.linspace(-5.0, 5.0, 6)
    np.savez_compressed(
        experiment_dir / "arrays.npz",
        regression_X_train_raw=X_train,
        regression_y_train_raw=y_train[:, None],
        regression_X_test_raw=X_test,
        regression_y_test_raw=y_test[:, None],
    )
    scaler = StandardScaler().fit(X_train)
    joblib.dump({"scaler_X": scaler}, experiment_dir / "feature_pipeline.pkl")
    store = PlotDataStore(experiment_dir)
    store.save_regression_test(y_test, y_pred)
    store.save_classifier_test(
        [0, 1, 1, 0, 1, 1],
        [0.8, 0.1, 0.2, 0.7, 0.3, 0.4],
        [0, 1, 1, 0, 1, 1],
        invalid_threshold=0.6,
    )
    store.update(
        "retraining_classifier_comparison",
        {
            "classifier_candidate_invalid_probability": np.asarray(
                [0.8, 0.1, 0.2, 0.7, 0.3, 0.4]
            ),
            "classifier_baseline_threshold": np.asarray([0.6]),
        },
    )
    (operational_dir / "metrics").mkdir(parents=True)
    (operational_dir / "metrics" / "validity_threshold.json").write_text(
        '{"threshold": 0.6}', encoding="utf-8"
    )
    return X_train, X_test


def test_load_final_plot_inputs_uses_unique_raw_training_rows(tmp_path: Path) -> None:
    operational_dir = tmp_path / "operational"
    X_train, X_test = _write_complete_source(tmp_path, operational_dir)

    data = load_final_plot_inputs(tmp_path, operational_dir)

    assert data["coverage_X_train_scaled"].shape == X_train.shape
    assert data["coverage_X_test_scaled"].shape == X_test.shape
    assert np.allclose(data["coverage_X_train_scaled"].mean(axis=0), 0.0)
    assert data["classifier_invalid_threshold"].item() == pytest.approx(0.6)


def test_load_final_plot_inputs_rejects_disagreeing_test_targets(tmp_path: Path) -> None:
    operational_dir = tmp_path / "operational"
    _write_complete_source(tmp_path, operational_dir)
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["regression_y_test_raw"] = arrays["regression_y_test_raw"] + 1.0
    np.savez_compressed(tmp_path / "arrays.npz", **arrays)

    with pytest.raises(RuntimeError, match="targets disagree"):
        load_final_plot_inputs(tmp_path, operational_dir)
