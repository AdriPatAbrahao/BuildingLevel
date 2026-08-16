"""Preliminary, out-of-sample feature-importance screening from a checkpoint.

This script does not train or save the final neural network. It fits temporary
tree ensembles inside validation folds to quantify predictive sensitivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split

from config.settings import NeuralNetConfig
from utils.feature_engineer import FeatureEngineer


SEED = 42


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _model(random_state: int, *, trees: int = 300) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )


def feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    named_groups = {
        "column_area": {
            "columns_total_area_cm2",
            "columns_std_area_cm2",
            "columns_min_area_cm2",
            "columns_max_area_cm2",
        },
        "beam_clear_spans": {
            "beams_std_clear_span_x_cm",
            "beams_std_clear_span_y_cm",
            "beams_max_clear_span_x_cm",
            "beams_max_clear_span_y_cm",
            "beams_span_entropy_x",
            "beams_span_entropy_y",
        },
        "column_inertia": {
            "inertia_sum_Ix",
            "inertia_sum_Iy",
            "inertia_ratio_Iy_over_Ix",
        },
        "spatial_distribution": {
            "column_area_spread_x_norm",
            "column_area_spread_y_norm",
            "columns_stiffness_spread_x_response_norm",
            "columns_stiffness_spread_y_response_norm",
        },
        "section_shape": {
            "columns_mean_shape_factor",
            "columns_p95_shape_factor",
        },
        "directional_radius_balance": {
            "columns_mean_radius_gyration_directional_balance",
        },
        "section_orientation": {
            "columns_mean_log_aspect_ratio",
            "columns_std_log_aspect_ratio",
            "columns_max_abs_log_aspect_ratio",
        },
    }
    known = set().union(*named_groups.values())
    actual = set(feature_names)
    if known != actual:
        raise ValueError(
            "Feature groups do not match the checkpoint schema: "
            f"missing={sorted(actual-known)}, obsolete={sorted(known-actual)}"
        )
    return {
        group: [feature_names.index(name) for name in feature_names if name in members]
        for group, members in named_groups.items()
    }


def load_checkpoint(checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    names = checkpoint.get("feature_names")
    expected_names = FeatureEngineer.feature_names()
    if checkpoint.get("feature_schema_version") != NeuralNetConfig.FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature schema is not current.")
    if names != expected_names:
        raise ValueError("Checkpoint feature names are not current.")
    X = np.asarray(checkpoint.get("feature_vectors", []), dtype=float)
    y = np.asarray(checkpoint.get("output_values", []), dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[1] != len(names) or len(X) != len(y):
        raise ValueError(f"Invalid checkpoint arrays: X={X.shape}, y={y.shape}.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Checkpoint contains non-finite values.")
    return X, y, names, checkpoint


def repeated_cv_permutation(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    splits: int,
    repeats: int,
    permutation_repeats: int,
) -> tuple[list[dict], list[dict]]:
    cv = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=SEED)
    fold_metrics: list[dict] = []
    values = {name: [] for name in names}
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        model = _model(SEED + fold)
        model.fit(X[train_idx], y[train_idx])
        prediction = model.predict(X[test_idx])
        fold_metrics.append(
            {
                "fold": fold,
                "mae_kgf": float(mean_absolute_error(y[test_idx], prediction)),
                "rmse_kgf": float(mean_squared_error(y[test_idx], prediction) ** 0.5),
                "r2": float(r2_score(y[test_idx], prediction)),
                "test_samples": int(len(test_idx)),
            }
        )
        pfi = permutation_importance(
            model,
            X[test_idx],
            y[test_idx],
            scoring="neg_mean_absolute_error",
            n_repeats=permutation_repeats,
            random_state=SEED + 1000 + fold,
            n_jobs=1,
        )
        for index, name in enumerate(names):
            values[name].extend(float(value) for value in pfi.importances[index])

    rows = []
    for name in names:
        sample = np.asarray(values[name], dtype=float)
        rows.append(
            {
                "feature": name,
                "delta_mae_mean_kgf": float(sample.mean()),
                "delta_mae_median_kgf": float(np.median(sample)),
                "delta_mae_p05_kgf": float(np.quantile(sample, 0.05)),
                "delta_mae_p95_kgf": float(np.quantile(sample, 0.95)),
                "positive_fraction": float(np.mean(sample > 0.0)),
            }
        )
    rows.sort(key=lambda row: row["delta_mae_mean_kgf"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return fold_metrics, rows


def grouped_ablation(
    X: np.ndarray,
    y: np.ndarray,
    groups: dict[str, list[int]],
    *,
    splits: int,
) -> list[dict]:
    cv = KFold(n_splits=splits, shuffle=True, random_state=SEED)
    paired = {group: [] for group in groups}
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        baseline = _model(SEED + 2000 + fold, trees=250)
        baseline.fit(X[train_idx], y[train_idx])
        baseline_mae = mean_absolute_error(y[test_idx], baseline.predict(X[test_idx]))
        for group, removed in groups.items():
            keep = [index for index in range(X.shape[1]) if index not in removed]
            reduced = _model(SEED + 3000 + fold, trees=250)
            reduced.fit(X[train_idx][:, keep], y[train_idx])
            reduced_mae = mean_absolute_error(
                y[test_idx], reduced.predict(X[test_idx][:, keep])
            )
            paired[group].append(float(reduced_mae - baseline_mae))

    rows = []
    for group, deltas in paired.items():
        sample = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "group": group,
                "feature_count": len(groups[group]),
                "delta_mae_mean_kgf": float(sample.mean()),
                "delta_mae_median_kgf": float(np.median(sample)),
                "delta_mae_min_kgf": float(sample.min()),
                "delta_mae_max_kgf": float(sample.max()),
                "positive_fold_fraction": float(np.mean(sample > 0.0)),
            }
        )
    rows.sort(key=lambda row: row["delta_mae_mean_kgf"], reverse=True)
    return rows


def bootstrap_stability(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    bootstrap_runs: int,
) -> list[dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in names}
    for run in range(bootstrap_runs):
        sampled = rng.integers(0, len(X_train), size=len(X_train))
        model = _model(SEED + 4000 + run, trees=150)
        model.fit(X_train[sampled], y_train[sampled])
        pfi = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=SEED + 5000 + run,
            n_jobs=1,
        )
        for index, name in enumerate(names):
            values[name].append(float(pfi.importances_mean[index]))
    rows = []
    for name in names:
        sample = np.asarray(values[name], dtype=float)
        rows.append(
            {
                "feature": name,
                "delta_mae_mean_kgf": float(sample.mean()),
                "delta_mae_p05_kgf": float(np.quantile(sample, 0.05)),
                "delta_mae_p95_kgf": float(np.quantile(sample, 0.95)),
                "positive_fraction": float(np.mean(sample > 0.0)),
            }
        )
    rows.sort(key=lambda row: row["delta_mae_mean_kgf"], reverse=True)
    return rows


def cross_validated_shap(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    splits: int,
) -> tuple[str, list[dict]]:
    try:
        import shap
    except Exception as exc:
        return f"skipped: {exc}", []
    values = np.zeros(X.shape[1], dtype=float)
    explained = 0
    cv = KFold(n_splits=splits, shuffle=True, random_state=SEED)
    try:
        for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
            model = _model(SEED + 6000 + fold, trees=250)
            model.fit(X[train_idx], y[train_idx])
            shap_values = shap.TreeExplainer(model).shap_values(X[test_idx])
            array = np.asarray(shap_values, dtype=float)
            values += np.sum(np.abs(array), axis=0)
            explained += len(test_idx)
    except Exception as exc:
        return f"skipped: {exc}", []
    rows = [
        {"feature": name, "mean_abs_shap_kgf": float(values[index] / explained)}
        for index, name in enumerate(names)
    ]
    rows.sort(key=lambda row: row["mean_abs_shap_kgf"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return "complete", rows


def run(args: argparse.Namespace) -> dict:
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    X, y, names, checkpoint = load_checkpoint(checkpoint_path)
    minimum = args.minimum_samples or (10 * len(names))
    if len(X) < minimum:
        raise ValueError(
            f"Need at least {minimum} valid samples for preliminary screening; got {len(X)}."
        )
    groups = feature_groups(names)
    fold_metrics, pfi = repeated_cv_permutation(
        X,
        y,
        names,
        splits=args.cv_splits,
        repeats=args.cv_repeats,
        permutation_repeats=args.permutation_repeats,
    )
    ablation = grouped_ablation(X, y, groups, splits=args.cv_splits)
    bootstrap = bootstrap_stability(
        X, y, names, bootstrap_runs=args.bootstrap_runs
    )
    shap_status, shap_rows = cross_validated_shap(
        X, y, names, splits=args.cv_splits
    )

    metrics_frame = pd.DataFrame(fold_metrics)
    summary = {
        "status": "complete",
        "analysis_scope": "preliminary regression feature screening; not final DNN training",
        "checkpoint": str(checkpoint_path),
        "feature_schema_version": checkpoint["feature_schema_version"],
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "model": "ExtraTreesRegressor (temporary, not persisted)",
        "cross_validation": {
            "splits": args.cv_splits,
            "repeats": args.cv_repeats,
            "folds": len(fold_metrics),
            "mae_mean_kgf": float(metrics_frame["mae_kgf"].mean()),
            "mae_std_kgf": float(metrics_frame["mae_kgf"].std(ddof=1)),
            "rmse_mean_kgf": float(metrics_frame["rmse_kgf"].mean()),
            "r2_mean": float(metrics_frame["r2"].mean()),
            "r2_min": float(metrics_frame["r2"].min()),
            "r2_max": float(metrics_frame["r2"].max()),
        },
        "permutation_importance": pfi,
        "grouped_ablation": ablation,
        "bootstrap_stability": bootstrap,
        "shap_status": shap_status,
        "cross_validated_shap": shap_rows,
        "interpretation_constraints": [
            "Importance is predictive, not causal.",
            "Correlated features can share or exchange importance.",
            "No feature should be removed from this preliminary ranking alone.",
            "Classifier importance is deferred because only 23 invalid samples exist.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    pd.DataFrame(pfi).to_csv(output_dir / "permutation_importance.csv", index=False)
    pd.DataFrame(ablation).to_csv(output_dir / "group_ablation.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(output_dir / "bootstrap_stability.csv", index=False)
    if shap_rows:
        pd.DataFrame(shap_rows).to_csv(output_dir / "shap_importance.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-runs", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["cross_validation"], indent=2))
