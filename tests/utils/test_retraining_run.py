import numpy as np

from utils.retraining_run import (
    metric_deltas,
    paired_bootstrap_error_deltas,
    publish_new_directory,
    regression_metrics,
    regression_metrics_by_observed_band,
)


def test_regression_metrics_use_predicted_minus_observed_bias():
    metrics = regression_metrics([100.0, 200.0], [110.0, 180.0])
    assert metrics["mean_absolute_error_kgf"] == 15.0
    assert metrics["bias_predicted_minus_observed_kgf"] == -5.0
    assert metrics["mape_pct"] == 10.0


def test_regression_bands_are_lower_inclusive_and_upper_exclusive():
    bands = regression_metrics_by_observed_band(
        np.array([249.0, 250.0, 300.0]),
        np.array([250.0, 251.0, 302.0]),
    )
    assert bands["[0, 250)"]["n_samples"] == 1
    assert bands["[250, 300)"]["n_samples"] == 1
    assert bands["[300, 350)"]["n_samples"] == 1


def test_paired_bootstrap_detects_uniformly_better_candidate():
    actual = np.arange(100.0, 120.0)
    baseline = actual + 10.0
    candidate = actual + 2.0
    result = paired_bootstrap_error_deltas(
        actual, baseline, candidate, n_resamples=200, seed=7
    )
    assert result["mae"]["ci_95_kgf"][1] < 0.0
    assert result["rmse"]["probability_candidate_lower"] == 1.0


def test_metric_deltas_follow_candidate_minus_baseline_convention():
    result = metric_deltas(
        {"r2_score": 0.90, "mean_absolute_error_kgf": 10.0},
        {"r2_score": 0.92, "mean_absolute_error_kgf": 8.0},
    )
    assert np.isclose(result["r2_score"], 0.02)
    assert result["mean_absolute_error_kgf"] == -2.0


def test_publish_new_directory_moves_complete_staging_tree(tmp_path):
    staging = tmp_path / ".staging"
    destination = tmp_path / "published"
    staging.mkdir()
    (staging / "completion.json").write_text("{}", encoding="utf-8")
    publish_new_directory(staging, destination)
    assert not staging.exists()
    assert (destination / "completion.json").is_file()
