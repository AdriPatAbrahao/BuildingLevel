import numpy as np
import pytest

from scripts.audit_classifier_errors import classifier_outcomes, threshold_metrics


def test_classifier_outcomes_use_infeasibility_as_positive_class() -> None:
    actual = np.asarray([0, 0, 1, 1])
    predicted = np.asarray([0, 1, 0, 1])

    assert classifier_outcomes(actual, predicted).tolist() == [
        "true_infeasible",
        "infeasible_as_feasible",
        "feasible_as_infeasible",
        "true_feasible",
    ]


def test_threshold_metrics_report_both_class_specific_errors() -> None:
    actual = np.asarray([0, 0, 1, 1, 1])
    probability = np.asarray([0.8, 0.4, 0.7, 0.2, 0.1])

    result = threshold_metrics(actual, probability, 0.6)

    assert result["true_infeasible"] == 1
    assert result["infeasible_as_feasible"] == 1
    assert result["feasible_as_infeasible"] == 1
    assert result["true_feasible"] == 2
    assert result["infeasible_miss_rate"] == pytest.approx(0.5)
    assert result["feasible_rejection_rate"] == pytest.approx(1 / 3)
