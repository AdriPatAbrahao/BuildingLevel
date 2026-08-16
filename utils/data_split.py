"""Reproducible, auditable train/validation/test splits for regression."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class RegressionSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    preused_development_prefix: int
    stratification_bins: int

    def as_manifest(self, targets: np.ndarray) -> dict:
        """Return a JSON-serializable audit record for this split."""
        y = np.asarray(targets, dtype=float).reshape(-1)

        def describe(indices: np.ndarray) -> dict:
            values = y[indices]
            return {
                "count": int(len(indices)),
                "indices": indices.astype(int).tolist(),
                "target_min": float(np.min(values)),
                "target_max": float(np.max(values)),
                "target_mean": float(np.mean(values)),
                "target_std": float(np.std(values)),
            }

        return {
            "method": "rank_quantile_stratified",
            "preused_development_prefix": int(self.preused_development_prefix),
            "test_contains_preused_samples": bool(
                np.any(self.test_indices < self.preused_development_prefix)
            ),
            "stratification_bins": int(self.stratification_bins),
            "train": describe(self.train_indices),
            "validation": describe(self.validation_indices),
            "test": describe(self.test_indices),
        }


@dataclass(frozen=True)
class ClassificationSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    preused_development_prefix: int

    def as_manifest(self, labels: np.ndarray) -> dict:
        """Return an auditable split record including class counts."""
        y = np.asarray(labels, dtype=int).reshape(-1)

        def describe(indices: np.ndarray) -> dict:
            values = y[indices]
            classes, counts = np.unique(values, return_counts=True)
            return {
                "count": int(len(indices)),
                "indices": indices.astype(int).tolist(),
                "class_counts": {
                    str(int(label)): int(count)
                    for label, count in zip(classes, counts)
                },
            }

        return {
            "method": "stratified_with_protected_development_prefix",
            "preused_development_prefix": int(self.preused_development_prefix),
            "test_contains_preused_samples": bool(
                np.any(self.test_indices < self.preused_development_prefix)
            ),
            "train": describe(self.train_indices),
            "validation": describe(self.validation_indices),
            "test": describe(self.test_indices),
        }


def regression_rank_strata(
    targets: np.ndarray,
    *,
    holdout_count: int,
    max_bins: int,
) -> tuple[np.ndarray | None, int]:
    """Build balanced target-rank bins compatible with a stratified split."""
    y = np.asarray(targets, dtype=float).reshape(-1)
    n_samples = len(y)
    n_bins = min(
        int(max_bins),
        n_samples // 2,
        int(holdout_count),
        n_samples - int(holdout_count),
    )
    if n_bins < 2:
        return None, 0

    order = np.argsort(y, kind="mergesort")
    strata = np.empty(n_samples, dtype=int)
    strata[order] = (np.arange(n_samples, dtype=int) * n_bins) // n_samples
    return strata, n_bins


def regression_train_validation_test_split(
    targets: np.ndarray,
    *,
    test_ratio: float,
    validation_ratio_of_development: float,
    random_state: int,
    preused_development_prefix: int = 0,
    max_stratification_bins: int = 10,
) -> RegressionSplit:
    """Split indices while keeping previously used samples out of final test.

    ``test_ratio`` is a fraction of the complete dataset. Validation is a
    fraction of the remaining development set, matching the existing project
    convention.
    """
    y = np.asarray(targets, dtype=float).reshape(-1)
    if y.size < 3:
        raise ValueError("At least three targets are required for a three-way split.")
    if not np.isfinite(y).all():
        raise ValueError("Regression targets contain NaN or infinite values.")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")
    if not 0.0 < validation_ratio_of_development < 1.0:
        raise ValueError("validation_ratio_of_development must be between 0 and 1.")
    if not 0 <= preused_development_prefix <= len(y):
        raise ValueError("preused_development_prefix is outside the dataset.")
    if max_stratification_bins < 2:
        raise ValueError("max_stratification_bins must be at least 2.")

    all_indices = np.arange(len(y), dtype=int)
    preused_indices = all_indices[:preused_development_prefix]
    new_indices = all_indices[preused_development_prefix:]
    test_count = int(math.ceil(len(y) * float(test_ratio)))
    if len(new_indices) <= test_count:
        raise ValueError(
            "There are not enough new samples to create a final test set after "
            f"protecting the {preused_development_prefix} preused samples."
        )

    test_strata, test_bins = regression_rank_strata(
        y[new_indices],
        holdout_count=test_count,
        max_bins=max_stratification_bins,
    )
    new_development, test_indices = train_test_split(
        new_indices,
        test_size=test_count,
        random_state=random_state,
        stratify=test_strata,
    )
    development_indices = np.sort(
        np.concatenate([preused_indices, np.asarray(new_development, dtype=int)])
    )

    validation_count = int(
        math.ceil(len(development_indices) * validation_ratio_of_development)
    )
    validation_strata, validation_bins = regression_rank_strata(
        y[development_indices],
        holdout_count=validation_count,
        max_bins=max_stratification_bins,
    )
    train_indices, validation_indices = train_test_split(
        development_indices,
        test_size=validation_count,
        random_state=random_state,
        stratify=validation_strata,
    )

    result = RegressionSplit(
        train_indices=np.sort(np.asarray(train_indices, dtype=int)),
        validation_indices=np.sort(np.asarray(validation_indices, dtype=int)),
        test_indices=np.sort(np.asarray(test_indices, dtype=int)),
        preused_development_prefix=int(preused_development_prefix),
        stratification_bins=min(test_bins, validation_bins),
    )
    combined = np.concatenate(
        [result.train_indices, result.validation_indices, result.test_indices]
    )
    if len(np.unique(combined)) != len(y) or set(combined.tolist()) != set(all_indices.tolist()):
        raise RuntimeError("Split indices are overlapping or incomplete.")
    if np.any(result.test_indices < preused_development_prefix):
        raise RuntimeError("A preused development sample leaked into the final test set.")
    return result


def classification_train_validation_test_split(
    labels: np.ndarray,
    *,
    test_ratio: float,
    validation_ratio_of_development: float,
    random_state: int,
    preused_development_prefix: int = 0,
) -> ClassificationSplit:
    """Create a stratified split while protecting previously inspected rows.

    Previously used rows remain eligible for model development, but are never
    placed in the final test set. The final test is drawn exclusively from new
    observations collected after the protected prefix.
    """
    y = np.asarray(labels, dtype=int).reshape(-1)
    if y.size < 3:
        raise ValueError("At least three labels are required for a three-way split.")
    if len(np.unique(y)) < 2:
        raise ValueError("At least two classes are required for classification.")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")
    if not 0.0 < validation_ratio_of_development < 1.0:
        raise ValueError("validation_ratio_of_development must be between 0 and 1.")
    if not 0 <= preused_development_prefix <= len(y):
        raise ValueError("preused_development_prefix is outside the dataset.")

    all_indices = np.arange(len(y), dtype=int)
    preused_indices = all_indices[:preused_development_prefix]
    new_indices = all_indices[preused_development_prefix:]
    test_count = int(math.ceil(len(y) * float(test_ratio)))
    if len(new_indices) <= test_count:
        raise ValueError(
            "There are not enough new classifier samples to create a final "
            f"test set after protecting {preused_development_prefix} rows."
        )

    try:
        new_development, test_indices = train_test_split(
            new_indices,
            test_size=test_count,
            random_state=random_state,
            stratify=y[new_indices],
        )
    except ValueError as exc:
        raise ValueError(
            "The new classifier samples do not contain enough observations "
            "of every class for a stratified final test set."
        ) from exc

    development_indices = np.sort(
        np.concatenate([preused_indices, np.asarray(new_development, dtype=int)])
    )
    validation_count = int(
        math.ceil(len(development_indices) * validation_ratio_of_development)
    )
    try:
        train_indices, validation_indices = train_test_split(
            development_indices,
            test_size=validation_count,
            random_state=random_state,
            stratify=y[development_indices],
        )
    except ValueError as exc:
        raise ValueError(
            "The classifier development data does not contain enough "
            "observations of every class for stratified validation."
        ) from exc

    result = ClassificationSplit(
        train_indices=np.sort(np.asarray(train_indices, dtype=int)),
        validation_indices=np.sort(np.asarray(validation_indices, dtype=int)),
        test_indices=np.sort(np.asarray(test_indices, dtype=int)),
        preused_development_prefix=int(preused_development_prefix),
    )
    combined = np.concatenate(
        [result.train_indices, result.validation_indices, result.test_indices]
    )
    if len(np.unique(combined)) != len(y) or set(combined.tolist()) != set(all_indices.tolist()):
        raise RuntimeError("Classification split indices are overlapping or incomplete.")
    if np.any(result.test_indices < preused_development_prefix):
        raise RuntimeError("A preused classifier sample leaked into the final test set.")
    return result
