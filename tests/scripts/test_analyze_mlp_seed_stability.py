import argparse

import numpy as np
import pytest

from scripts.analyze_mlp_seed_stability import (
    parse_seeds,
    summarize_prediction_matrix,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", [42]),
        ("42,44", [42, 44]),
        ("42:44", [42, 43, 44]),
    ],
)
def test_parse_seeds(text, expected):
    assert parse_seeds(text) == expected


@pytest.mark.parametrize("text", ["", "44:42", "42,42", "-1", "42:44:46"])
def test_parse_seeds_rejects_invalid_specifications(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seeds(text)


def test_bias_variance_decomposition_identity():
    observed = np.array([10.0, 20.0])
    predictions = np.array([[9.0, 22.0], [11.0, 18.0]])

    summary, mean_prediction, sample_std = summarize_prediction_matrix(
        observed, predictions
    )

    np.testing.assert_allclose(mean_prediction, observed)
    np.testing.assert_allclose(sample_std, [np.sqrt(2.0), np.sqrt(8.0)])
    assert summary["mean_squared_bias_kgf2"] == pytest.approx(0.0)
    assert summary["mean_model_variance_kgf2"] == pytest.approx(2.5)
    assert summary["expected_seed_mse_kgf2"] == pytest.approx(2.5)
    assert summary["decomposition_identity_error_kgf2"] == pytest.approx(0.0)


def test_single_model_has_zero_seed_variance():
    summary, _, sample_std = summarize_prediction_matrix(
        np.array([10.0, 20.0]), np.array([[11.0, 19.0]])
    )

    np.testing.assert_array_equal(sample_std, [0.0, 0.0])
    assert summary["mean_model_variance_kgf2"] == 0.0


def test_prediction_matrix_shape_is_validated():
    with pytest.raises(ValueError, match="shape"):
        summarize_prediction_matrix(np.array([1.0, 2.0]), np.array([[1.0]]))
