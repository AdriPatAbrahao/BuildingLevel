import pytest

from scripts.plot_mlp_seed_stability import bias_variance_rows, seed_metric_rows


def test_seed_metric_rows_preserve_seed_and_engineering_metrics():
    summary = {
        "per_seed": [
            {
                "seed": 42,
                "best_epoch": 10,
                "best_validation_loss": 0.01,
                "global_test": {
                    "r2_score": 0.98,
                    "mean_absolute_error_kgf": 16.0,
                    "rmse_kgf": 20.0,
                    "bias_predicted_minus_observed_kgf": 1.0,
                },
                "protected_low_steel_posthoc": {
                    "mean_absolute_error_kgf": 24.0,
                    "bias_predicted_minus_observed_kgf": 12.0,
                },
                "final_verified_solution": {"predicted_steel_kgf": 222.0},
            }
        ]
    }

    assert seed_metric_rows(summary) == [
        [42, 10, 0.01, 0.98, 16.0, 20.0, 1.0, 24.0, 12.0, 222.0]
    ]


def test_bias_variance_rows_report_percentages():
    record = {
        "expected_seed_mse_kgf2": 100.0,
        "mean_squared_bias_kgf2": 80.0,
        "mean_model_variance_kgf2": 20.0,
        "mean_pointwise_sample_std_kgf": 4.0,
        "maximum_pointwise_sample_std_kgf": 8.0,
    }
    rows = bias_variance_rows(
        {"global_test": record, "protected_low_steel_posthoc": record}
    )

    assert len(rows) == 2
    assert rows[0][4] == pytest.approx(80.0)
    assert rows[0][5] == pytest.approx(20.0)
    assert rows[1][6:] == [4.0, 8.0]
