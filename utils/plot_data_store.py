"""Versioned, model-independent source data for post-training figures."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLOT_DATA_FORMAT_VERSION = 1
ARCHIVE_NAME = "figure_data.npz"
MANIFEST_NAME = "figure_data_manifest.json"


class PlotDataStore:
    """Persist the numerical inputs required to redraw evaluation figures."""

    def __init__(self, experiment_dir: Path | str):
        self.experiment_dir = Path(experiment_dir)
        self.metrics_dir = self.experiment_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.metrics_dir / ARCHIVE_NAME
        self.manifest_path = self.metrics_dir / MANIFEST_NAME

    def _existing_arrays(self) -> dict[str, np.ndarray]:
        if not self.archive_path.exists():
            return {}
        with np.load(self.archive_path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}

    def _existing_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "plot_data_format_version": PLOT_DATA_FORMAT_VERSION,
                "archive": f"metrics/{ARCHIVE_NAME}",
                "sections": {},
            }
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        version = manifest.get("plot_data_format_version")
        if version != PLOT_DATA_FORMAT_VERSION:
            raise RuntimeError(
                "Unsupported figure-data format "
                f"({version} != {PLOT_DATA_FORMAT_VERSION})."
            )
        return manifest

    @staticmethod
    def _normalise_array(name: str, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError(f"Figure-data array '{name}' cannot use object dtype.")
        return array

    def update(
        self,
        section: str,
        arrays: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically merge one logical section into the compressed archive."""
        if not section.strip():
            raise ValueError("Figure-data section name cannot be empty.")
        normalised = {
            name: self._normalise_array(name, value)
            for name, value in arrays.items()
        }
        merged = self._existing_arrays()
        merged.update(normalised)

        archive_tmp = self.archive_path.with_suffix(".npz.tmp")
        with open(archive_tmp, "wb") as stream:
            np.savez_compressed(stream, **merged)
        os.replace(archive_tmp, self.archive_path)

        manifest = self._existing_manifest()
        timestamp = datetime.now(timezone.utc).isoformat()
        manifest.setdefault("created_utc", timestamp)
        manifest["updated_utc"] = timestamp
        manifest["sections"][section] = {
            "arrays": {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
                for name, array in normalised.items()
            },
            "metadata": dict(metadata or {}),
        }
        manifest_tmp = self.manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(manifest_tmp, self.manifest_path)
        return self.archive_path

    def load(self) -> dict[str, np.ndarray]:
        """Load and validate all saved arrays without requiring a trained model."""
        self._existing_manifest()
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Figure-data archive not found: {self.archive_path}")
        return self._existing_arrays()

    def save_regression_test(
        self,
        y_true_steel: Sequence[float],
        y_pred_steel: Sequence[float],
        *,
        sample_indices: Sequence[int] | None = None,
    ) -> Path:
        """Save held-out steel predictions and a human-readable residual table."""
        actual = np.asarray(y_true_steel, dtype=float).reshape(-1)
        predicted = np.asarray(y_pred_steel, dtype=float).reshape(-1)
        if actual.shape != predicted.shape or actual.size == 0:
            raise ValueError("Regression actual/predicted arrays must be non-empty and aligned.")
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError("Regression figure data contain NaN or infinite values.")
        indices = (
            np.arange(actual.size, dtype=np.int64)
            if sample_indices is None
            else np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        )
        if indices.shape != actual.shape:
            raise ValueError("Regression sample indices are not aligned with predictions.")

        residual = actual - predicted
        abs_error = np.abs(residual)
        relative_error = np.divide(
            residual,
            actual,
            out=np.full(actual.shape, np.nan, dtype=float),
            where=actual != 0,
        )
        path = self.update(
            "regression_test",
            {
                "regression_test_indices": indices,
                "regression_y_true_steel_kgf": actual,
                "regression_y_pred_steel_kgf": predicted,
                "regression_residual_steel_kgf": residual,
                "regression_abs_error_steel_kgf": abs_error,
                "regression_relative_error": relative_error,
            },
            metadata={
                "split": "test",
                "target": "column reinforcement steel weight",
                "unit": "kgf",
                "residual_definition": "observed_minus_predicted",
            },
        )
        self._write_csv_atomic(
            self.metrics_dir / "regression_test_predictions.csv",
            [
                "sample_index",
                "observed_steel_kgf",
                "predicted_steel_kgf",
                "residual_kgf",
                "absolute_error_kgf",
                "relative_error",
            ],
            zip(indices, actual, predicted, residual, abs_error, relative_error),
        )
        return path

    def save_classifier_test(
        self,
        y_true_validity: Sequence[int],
        invalid_probability: Sequence[float],
        y_pred_validity: Sequence[int],
        *,
        invalid_threshold: float,
        X_test: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> Path:
        """Save held-out classifier decisions, scores and optional test features."""
        actual = np.asarray(y_true_validity, dtype=np.int64).reshape(-1)
        probability = np.asarray(invalid_probability, dtype=float).reshape(-1)
        predicted = np.asarray(y_pred_validity, dtype=np.int64).reshape(-1)
        if not (actual.shape == probability.shape == predicted.shape) or actual.size == 0:
            raise ValueError("Classifier labels, scores and predictions must be aligned.")
        if not np.isfinite(probability).all():
            raise ValueError("Classifier probabilities contain NaN or infinite values.")
        if not 0.0 <= float(invalid_threshold) <= 1.0:
            raise ValueError("Classifier threshold must be between zero and one.")

        arrays: dict[str, Any] = {
            "classifier_y_true_validity": actual,
            "classifier_invalid_probability": probability,
            "classifier_y_pred_validity": predicted,
            "classifier_invalid_threshold": np.asarray([invalid_threshold], dtype=float),
        }
        metadata: dict[str, Any] = {
            "split": "test",
            "validity_labels": {"0": "infeasible", "1": "feasible"},
            "probability": "P(infeasible)",
        }
        if X_test is not None:
            features = np.asarray(X_test, dtype=float)
            if features.ndim != 2 or len(features) != len(actual):
                raise ValueError("Classifier test features are not aligned with labels.")
            arrays["classifier_X_test"] = features
            metadata["feature_names"] = list(feature_names or [])

        path = self.update("classifier_test", arrays, metadata=metadata)
        rows = zip(
            np.arange(actual.size, dtype=np.int64),
            actual,
            predicted,
            probability,
            np.full(actual.size, float(invalid_threshold)),
        )
        self._write_csv_atomic(
            self.metrics_dir / "classifier_test_predictions.csv",
            [
                "heldout_row",
                "actual_validity_label",
                "predicted_validity_label",
                "invalid_probability",
                "invalid_threshold",
            ],
            rows,
        )
        return path

    @staticmethod
    def _write_csv_atomic(
        path: Path,
        header: Sequence[str],
        rows: Any,
    ) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(tmp_path, path)
