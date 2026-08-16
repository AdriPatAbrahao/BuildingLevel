import json

import numpy as np
import pytest

from utils.classifier_evaluation import (
    classifier_metrics,
    evaluate_invalidity_rule,
    invalid_probability_index,
    invalid_roc_payload,
    validity_labels_from_invalid_probability,
    youden_threshold,
)


def test_invalidity_rule_uses_declared_threshold_and_inclusive_boundary():
    labels = validity_labels_from_invalid_probability(
        [0.10, 0.60, 0.61],
        threshold=0.60,
    )
    assert labels.tolist() == [1, 0, 0]


def test_classifier_metrics_report_critical_invalid_as_valid_error():
    metrics = classifier_metrics(
        y_true=[0, 0, 0, 1, 1],
        y_pred=[0, 1, 0, 1, 0],
    )

    assert metrics["positive_class"] == "invalid"
    assert metrics["confusion_matrix"] == [[2, 1], [1, 1]]
    assert metrics["invalid_predicted_valid_count"] == 1
    assert metrics["invalid_predicted_valid_rate"] == pytest.approx(1 / 3)
    assert metrics["recall_invalid"] == pytest.approx(2 / 3)


def test_invalid_roc_payload_converts_validity_zero_to_positive_event():
    payload = invalid_roc_payload(
        y_true_validity=[0, 0, 1, 1],
        invalid_probability=[0.9, 0.8, 0.2, 0.1],
        split="test",
    )

    assert payload["positive_class"] == "invalid"
    assert payload["source_validity_label"] == 0
    assert payload["split"] == "test"
    assert payload["auc"] == pytest.approx(1.0)
    assert payload["thresholds"][0] is None
    json.dumps(payload, allow_nan=False)


def test_youden_threshold_ignores_infinite_threshold():
    payload = {
        "fpr": [0.0, 0.0, 1.0],
        "tpr": [0.0, 1.0, 1.0],
        "thresholds": [float("inf"), 0.7, 0.1],
    }
    assert youden_threshold(payload) == pytest.approx(0.7)


def test_evaluate_invalidity_rule_keeps_metrics_and_roc_consistent():
    metrics, roc_payload = evaluate_invalidity_rule(
        [0, 0, 1, 1],
        [0.9, 0.4, 0.3, 0.1],
        threshold=0.5,
        split="test",
    )

    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["roc_auc_invalid"] == roc_payload["auc"]
    assert metrics["threshold_used"] == pytest.approx(0.5)


def test_invalid_probability_index_requires_class_zero():
    assert invalid_probability_index(np.array([0, 1])) == 0
    with pytest.raises(ValueError, match="invalid class 0"):
        invalid_probability_index([1, 2])
