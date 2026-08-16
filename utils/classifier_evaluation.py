"""Shared evaluation contract for the structural-validity classifier.

Validity labels use ``0 = invalid`` and ``1 = valid``.  Safety reporting treats
the invalid class as the positive event, because accepting an invalid design as
valid is the critical error for the optimizer.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


INVALID_LABEL = 0
VALID_LABEL = 1


def invalid_probability_index(classes: Sequence[Any]) -> int:
    """Return the probability-column index for the invalid label (zero)."""
    normalized = list(classes)
    if INVALID_LABEL not in normalized:
        raise ValueError("Classifier does not expose the required invalid class 0.")
    return int(normalized.index(INVALID_LABEL))


def validity_labels_from_invalid_probability(
    invalid_probability: Iterable[float],
    threshold: float,
) -> np.ndarray:
    """Apply the same invalidity rule used by the optimization objective."""
    probability = np.asarray(invalid_probability, dtype=float).reshape(-1)
    threshold = float(threshold)
    if probability.size == 0:
        raise ValueError("Invalid-probability array cannot be empty.")
    if not np.isfinite(probability).all():
        raise ValueError("Invalid-probability array contains non-finite values.")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Invalidity threshold must be finite and within [0, 1].")
    return np.where(probability >= threshold, INVALID_LABEL, VALID_LABEL).astype(int)


def classifier_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> dict[str, Any]:
    """Calculate thesis metrics with invalid structures as the positive class."""
    actual = np.asarray(y_true, dtype=int).reshape(-1)
    predicted = np.asarray(y_pred, dtype=int).reshape(-1)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Classifier labels must be non-empty and aligned.")
    allowed = {INVALID_LABEL, VALID_LABEL}
    if not set(np.unique(actual)).issubset(allowed):
        raise ValueError("True labels must follow 0=invalid and 1=valid.")
    if not set(np.unique(predicted)).issubset(allowed):
        raise ValueError("Predicted labels must follow 0=invalid and 1=valid.")

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[INVALID_LABEL, VALID_LABEL],
        zero_division=0,
    )
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[INVALID_LABEL, VALID_LABEL],
    )
    invalid_count = int(matrix[0].sum())
    invalid_as_valid_count = int(matrix[0, 1])
    return {
        "label_semantics": {"0": "invalid", "1": "valid"},
        "positive_class": "invalid",
        "positive_label": INVALID_LABEL,
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_invalid": float(precision[0]),
        "recall_invalid": float(recall[0]),
        "f1_invalid": float(f1[0]),
        "precision_by_class": {
            "0": float(precision[0]),
            "1": float(precision[1]),
        },
        "recall_by_class": {
            "0": float(recall[0]),
            "1": float(recall[1]),
        },
        "f1_by_class": {
            "0": float(f1[0]),
            "1": float(f1[1]),
        },
        "confusion_matrix": matrix.tolist(),
        "invalid_predicted_valid_count": invalid_as_valid_count,
        "invalid_predicted_valid_rate": (
            float(invalid_as_valid_count / invalid_count)
            if invalid_count
            else None
        ),
    }


def invalid_roc_payload(
    y_true_validity: Iterable[int],
    invalid_probability: Iterable[float],
    *,
    split: str,
) -> dict[str, Any]:
    """Build a ROC payload for the event ``structure is invalid``."""
    validity = np.asarray(y_true_validity, dtype=int).reshape(-1)
    probability = np.asarray(invalid_probability, dtype=float).reshape(-1)
    if validity.size == 0 or validity.shape != probability.shape:
        raise ValueError("ROC labels and probabilities must be non-empty and aligned.")
    invalid_event = (validity == INVALID_LABEL).astype(int)
    if np.unique(invalid_event).size != 2:
        raise ValueError("ROC AUC requires both invalid and valid cases.")
    fpr, tpr, thresholds = roc_curve(invalid_event, probability, pos_label=1)
    return {
        "fpr": list(map(float, fpr)),
        "tpr": list(map(float, tpr)),
        "thresholds": [
            float(value) if np.isfinite(value) else None
            for value in thresholds
        ],
        "auc": float(roc_auc_score(invalid_event, probability)),
        "positive_class": "invalid",
        "positive_label_in_roc": 1,
        "source_validity_label": INVALID_LABEL,
        "split": str(split),
        "n_samples": int(validity.size),
    }


def youden_threshold(roc_payload: dict[str, Any], *, fallback: float = 0.5) -> float:
    """Select a finite Youden threshold from a validation ROC payload."""
    fpr = np.asarray(roc_payload["fpr"], dtype=float)
    tpr = np.asarray(roc_payload["tpr"], dtype=float)
    thresholds = np.asarray(roc_payload["thresholds"], dtype=float)
    if not (fpr.shape == tpr.shape == thresholds.shape) or fpr.size == 0:
        raise ValueError("ROC payload arrays are empty or misaligned.")
    finite = np.isfinite(thresholds) & (thresholds >= 0.0) & (thresholds <= 1.0)
    if not finite.any():
        return float(fallback)
    candidates = np.flatnonzero(finite)
    best = candidates[int(np.argmax((tpr - fpr)[candidates]))]
    return float(thresholds[best])


def evaluate_invalidity_rule(
    y_true_validity: Iterable[int],
    invalid_probability: Iterable[float],
    *,
    threshold: float,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the deployed rule and return metrics plus invalid-class ROC."""
    y_true = np.asarray(y_true_validity, dtype=int).reshape(-1)
    probability = np.asarray(invalid_probability, dtype=float).reshape(-1)
    prediction = validity_labels_from_invalid_probability(probability, threshold)
    metrics = classifier_metrics(y_true, prediction)
    roc_payload = invalid_roc_payload(y_true, probability, split=split)
    metrics.update(
        {
            "split": str(split),
            "n_samples": int(y_true.size),
            "threshold_used": float(threshold),
            "roc_auc_invalid": float(roc_payload["auc"]),
        }
    )
    return metrics, roc_payload
