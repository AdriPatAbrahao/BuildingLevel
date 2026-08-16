"""Compare steel surrogate candidates without touching the final test set.

The script consumes a collection ``checkpoint.json``. For the 230-sample
pilot, all observations are development data. Once the full collection exists,
the protected split policy used by ``main.py`` removes final-test indices before
any candidate is fitted or scored. No production model is persisted here.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.settings import DataSplitConfig, NeuralNetConfig, RunConfig
from models.dnnmodel import SimpleNN
from utils.data_split import (
    regression_rank_strata,
    regression_train_validation_test_split,
)
from utils.feature_engineer import FeatureEngineer


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    parameters: dict[str, Any]


def candidate_grid() -> list[Candidate]:
    """Return the deliberately small, auditable candidate set."""
    candidates = []
    for hidden, dropout in (
        ([32, 16], 0.1),
        ([64, 32], 0.1),
        ([128, 128, 64], 0.2),
    ):
        for loss in ("mse", "huber"):
            suffix = "_current" if hidden == [128, 128, 64] and loss == "mse" else ""
            candidates.append(
                Candidate(
                    name=f"mlp_{'_'.join(map(str, hidden))}_d{dropout}_{loss}{suffix}",
                    family="mlp",
                    parameters={
                        "hidden_layers": hidden,
                        "dropout_rate": dropout,
                        "loss_type": loss,
                    },
                )
            )
    for min_leaf in (1, 2):
        for max_features in (0.8, 1.0):
            candidates.append(
                Candidate(
                    name=f"extra_trees_leaf{min_leaf}_features{max_features}",
                    family="extra_trees",
                    parameters={
                        "n_estimators": 400,
                        "min_samples_leaf": min_leaf,
                        "max_features": max_features,
                    },
                )
            )
    return candidates


def load_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and strictly validate the steel-only collection checkpoint."""
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    if checkpoint.get("feature_schema_version") != NeuralNetConfig.FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature schema is not current.")
    if checkpoint.get("feature_names") != FeatureEngineer.feature_names():
        raise ValueError("Checkpoint feature names/order are not current.")

    X = np.asarray(checkpoint.get("feature_vectors", []), dtype=np.float32)
    y_raw = np.asarray(checkpoint.get("output_values", []), dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != NeuralNetConfig.INPUT_SIZE:
        raise ValueError(f"Invalid checkpoint feature matrix: {X.shape}.")
    if y_raw.ndim != 2 or y_raw.shape[1] != NeuralNetConfig.OUTPUT_SIZE:
        raise ValueError(f"Expected one steel target column; got {y_raw.shape}.")
    y = y_raw[:, 0]
    if len(X) != len(y) or len(X) < 10:
        raise ValueError(f"Checkpoint arrays are too small or misaligned: X={X.shape}, y={y.shape}.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Checkpoint contains NaN or infinite values.")
    return X, y, checkpoint


def development_indices(
    y: np.ndarray,
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return development and untouched-test indices for pilot or full data."""
    prefix = int(DataSplitConfig.PREUSED_DEVELOPMENT_PREFIX_SAMPLES)
    if len(y) <= prefix:
        return np.arange(len(y), dtype=int), np.array([], dtype=int), "pilot_only"
    if not checkpoint.get("collection_complete", False):
        raise ValueError(
            "A collection larger than the pilot must be complete before model selection."
        )
    split = regression_train_validation_test_split(
        y,
        test_ratio=NeuralNetConfig.TEST_SPLIT_RATIO,
        validation_ratio_of_development=NeuralNetConfig.VALIDATION_SPLIT_RATIO,
        random_state=RunConfig.SEED,
        preused_development_prefix=prefix,
        max_stratification_bins=DataSplitConfig.REGRESSION_STRATIFICATION_BINS,
    )
    development = np.sort(
        np.concatenate([split.train_indices, split.validation_indices])
    )
    return development, split.test_indices, "full_development_only"


def _safe_batch_size(num_samples: int, configured: int) -> int:
    candidate = min(int(configured), int(num_samples))
    while candidate >= 2:
        if num_samples % candidate != 1:
            return candidate
        candidate -= 1
    if num_samples >= 2:
        return int(num_samples)
    raise ValueError("At least two MLP training samples are required.")


def _fit_mlp_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_internal_val: np.ndarray,
    y_internal_val: np.ndarray,
    X_evaluate: np.ndarray,
    candidate: Candidate,
    *,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train_scaled = scaler_X.transform(X_train).astype(np.float32)
    X_val_scaled = scaler_X.transform(X_internal_val).astype(np.float32)
    X_eval_scaled = scaler_X.transform(X_evaluate).astype(np.float32)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = scaler_y.transform(y_internal_val.reshape(-1, 1)).astype(np.float32)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = SimpleNN(
        input_size=X_train.shape[1],
        output_size=1,
        hidden_layers=candidate.parameters["hidden_layers"],
        dropout_rate=candidate.parameters["dropout_rate"],
    ).to(device)
    criterion: nn.Module = (
        nn.HuberLoss()
        if candidate.parameters["loss_type"] == "huber"
        else nn.MSELoss()
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NeuralNetConfig.LEARNING_RATE,
        weight_decay=NeuralNetConfig.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=NeuralNetConfig.LR_SCHEDULER_PATIENCE,
        factor=NeuralNetConfig.LR_SCHEDULER_FACTOR,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_scaled),
            torch.from_numpy(y_train_scaled),
        ),
        batch_size=_safe_batch_size(len(X_train_scaled), batch_size),
        shuffle=True,
        generator=generator,
    )
    X_val_tensor = torch.from_numpy(X_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(y_val_scaled).to(device)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(max_epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(X_val_tensor), y_val_tensor).item())
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a finite best state.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(
            torch.from_numpy(X_eval_scaled).to(device)
        ).cpu().numpy()
    predictions = scaler_y.inverse_transform(predictions_scaled)[:, 0]
    return predictions, {
        "best_epoch": int(best_epoch),
        "best_internal_validation_loss": float(best_loss),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }


def _fit_tree_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_evaluate: np.ndarray,
    candidate: Candidate,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = ExtraTreesRegressor(
        **candidate.parameters,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_evaluate), {
        "best_epoch": None,
        "best_internal_validation_loss": None,
        "trainable_parameters": None,
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    low_steel_threshold: float,
) -> dict[str, float]:
    """Metrics aligned with structural optimization risk."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    underprediction = np.maximum(residual, 0.0)
    low_mask = y_true <= low_steel_threshold
    return {
        "mae_kgf": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kgf": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "underprediction_rate": float(np.mean(residual > 0.0)),
        "mean_underprediction_kgf": float(np.mean(underprediction)),
        "p90_underprediction_kgf": float(np.percentile(underprediction, 90)),
        "low_steel_quartile_mae_kgf": float(
            mean_absolute_error(y_true[low_mask], y_pred[low_mask])
        ) if np.any(low_mask) else float("nan"),
    }


def evaluate_candidates(
    X: np.ndarray,
    y: np.ndarray,
    candidates: list[Candidate],
    *,
    cv_splits: int,
    cv_repeats: int,
    internal_validation_ratio: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Repeated outer CV with an inner early-stopping split in every fold."""
    holdout_count = int(math.ceil(len(y) / cv_splits))
    strata, bins = regression_rank_strata(
        y,
        holdout_count=holdout_count,
        max_bins=DataSplitConfig.REGRESSION_STRATIFICATION_BINS,
    )
    if strata is None:
        raise ValueError("Not enough development samples for stratified CV.")
    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=RunConfig.SEED,
    )
    low_steel_threshold = float(np.quantile(y, 0.25))
    fold_rows: list[dict[str, Any]] = []

    for fold_number, (outer_train, outer_evaluate) in enumerate(
        cv.split(X, strata), start=1
    ):
        internal_count = int(math.ceil(len(outer_train) * internal_validation_ratio))
        internal_strata, _ = regression_rank_strata(
            y[outer_train], holdout_count=internal_count, max_bins=bins
        )
        fit_local, internal_local = train_test_split(
            np.arange(len(outer_train), dtype=int),
            test_size=internal_count,
            random_state=RunConfig.SEED + fold_number,
            stratify=internal_strata,
        )
        fit_indices = outer_train[fit_local]
        internal_indices = outer_train[internal_local]

        for candidate_number, candidate in enumerate(candidates, start=1):
            seed = RunConfig.SEED + fold_number * 100 + candidate_number
            started = time.perf_counter()
            if candidate.family == "mlp":
                predictions, fit_info = _fit_mlp_predict(
                    X[fit_indices], y[fit_indices],
                    X[internal_indices], y[internal_indices],
                    X[outer_evaluate], candidate,
                    seed=seed, device=device, max_epochs=max_epochs,
                    patience=patience, batch_size=batch_size,
                )
            elif candidate.family == "extra_trees":
                predictions, fit_info = _fit_tree_predict(
                    X[fit_indices], y[fit_indices], X[outer_evaluate], candidate,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown candidate family: {candidate.family}")
            fold_rows.append({
                "fold": fold_number,
                "candidate": candidate.name,
                "family": candidate.family,
                "fit_samples": int(len(fit_indices)),
                "internal_validation_samples": int(len(internal_indices)),
                "evaluation_samples": int(len(outer_evaluate)),
                "elapsed_sec": float(time.perf_counter() - started),
                **fit_info,
                **regression_metrics(
                    y[outer_evaluate], predictions,
                    low_steel_threshold=low_steel_threshold,
                ),
            })
        print(
            f"Fold {fold_number}/{cv_splits * cv_repeats} complete "
            f"({len(candidates)} candidates).", flush=True,
        )

    metric_names = [
        "mae_kgf", "rmse_kgf", "r2", "underprediction_rate",
        "mean_underprediction_kgf", "p90_underprediction_kgf",
        "low_steel_quartile_mae_kgf", "elapsed_sec",
    ]
    summaries = []
    for candidate in candidates:
        rows = [row for row in fold_rows if row["candidate"] == candidate.name]
        summary: dict[str, Any] = {
            "candidate": candidate.name,
            "family": candidate.family,
            "parameters": candidate.parameters,
            "folds": len(rows),
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        epochs = [row["best_epoch"] for row in rows if row["best_epoch"] is not None]
        summary["best_epoch_median"] = float(np.median(epochs)) if epochs else None
        summary["trainable_parameters"] = rows[0]["trainable_parameters"]
        summaries.append(summary)
    summaries.sort(key=lambda item: item["mae_kgf_mean"])
    for rank, summary in enumerate(summaries, start=1):
        summary["mae_rank"] = rank
    return summaries, fold_rows


def paired_mae_comparison(
    fold_rows: list[dict[str, Any]],
    *,
    reference_candidate: str,
    challenger_candidate: str,
    bootstrap_runs: int = 10_000,
    seed: int = RunConfig.SEED,
) -> dict[str, Any]:
    """Approximate paired CV improvement of challenger over reference."""
    reference = {
        int(row["fold"]): float(row["mae_kgf"])
        for row in fold_rows
        if row["candidate"] == reference_candidate
    }
    challenger = {
        int(row["fold"]): float(row["mae_kgf"])
        for row in fold_rows
        if row["candidate"] == challenger_candidate
    }
    common_folds = sorted(set(reference).intersection(challenger))
    if len(common_folds) < 2:
        raise ValueError("Paired comparison requires at least two common folds.")
    # Positive delta means the challenger has a lower MAE than the reference.
    improvements = np.asarray(
        [reference[fold] - challenger[fold] for fold in common_folds],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_runs, dtype=float)
    for run in range(bootstrap_runs):
        sample = rng.integers(0, len(improvements), size=len(improvements))
        bootstrap_means[run] = float(np.mean(improvements[sample]))
    return {
        "reference": reference_candidate,
        "challenger": challenger_candidate,
        "paired_folds": len(common_folds),
        "mean_mae_improvement_kgf": float(np.mean(improvements)),
        "std_mae_improvement_kgf": float(np.std(improvements, ddof=1)),
        "challenger_better_fold_fraction": float(np.mean(improvements > 0.0)),
        "bootstrap_mean_improvement_ci_95_kgf": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "note": (
            "Approximate interval: repeated-CV folds are paired but not fully "
            "independent. Confirm model choice after the full collection."
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/validation/teste17/model_tuning"),
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--internal-validation-ratio", type=float, default=0.20)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=NeuralNetConfig.BATCH_SIZE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cv_splits < 2 or args.cv_repeats < 1:
        raise ValueError("CV requires at least 2 splits and 1 repeat.")
    if not 0.0 < args.internal_validation_ratio < 0.5:
        raise ValueError("Internal validation ratio must be between 0 and 0.5.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(1)

    X_all, y_all, checkpoint = load_checkpoint(args.checkpoint)
    development, protected_test, selection_mode = development_indices(y_all, checkpoint)
    candidates = candidate_grid()
    started = time.perf_counter()
    summaries, folds = evaluate_candidates(
        X_all[development], y_all[development], candidates,
        cv_splits=args.cv_splits, cv_repeats=args.cv_repeats,
        internal_validation_ratio=args.internal_validation_ratio,
        max_epochs=args.max_epochs, patience=args.patience,
        batch_size=args.batch_size, device=device,
    )

    dataset_hash = hashlib.sha256(X_all.tobytes() + y_all.tobytes()).hexdigest()
    current_name = "mlp_128_128_64_d0.2_mse_current"
    current_summary = next(
        item for item in summaries if item["candidate"] == current_name
    )
    paired_winner_vs_current = paired_mae_comparison(
        folds,
        reference_candidate=current_name,
        challenger_candidate=summaries[0]["candidate"],
    )
    best_mlp = next(item for item in summaries if item["family"] == "mlp")
    paired_best_mlp_vs_current = paired_mae_comparison(
        folds,
        reference_candidate=current_name,
        challenger_candidate=best_mlp["candidate"],
    )
    result = {
        "status": "complete",
        "purpose": "development-only model/hyperparameter screening",
        "production_model_saved": False,
        "selection_mode": selection_mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_hash": dataset_hash,
        "all_samples": int(len(X_all)),
        "development_samples_used": int(len(development)),
        "protected_final_test_samples": int(len(protected_test)),
        "protected_final_test_indices_disclosed_to_models": False,
        "feature_schema_version": NeuralNetConfig.FEATURE_SCHEMA_VERSION,
        "feature_names": FeatureEngineer.feature_names(),
        "target": {"name": "column_steel_weight", "unit": "kgf"},
        "cv": {
            "splits": args.cv_splits,
            "repeats": args.cv_repeats,
            "folds": args.cv_splits * args.cv_repeats,
            "stratification": "target rank bins",
            "internal_validation_ratio": args.internal_validation_ratio,
        },
        "training": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": NeuralNetConfig.LEARNING_RATE,
            "weight_decay": NeuralNetConfig.WEIGHT_DECAY,
            "device": str(device),
        },
        "elapsed_sec": float(time.perf_counter() - started),
        "winner_by_mean_cv_mae": summaries[0]["candidate"],
        "current_configuration": current_summary,
        "paired_comparisons": {
            "winner_vs_current": paired_winner_vs_current,
            "best_mlp_vs_current": paired_best_mlp_vs_current,
        },
        "recommendation_scope": (
            "Pilot evidence only; rerun after the 2500-sample collection before "
            "choosing the production model."
            if selection_mode == "pilot_only"
            else "Development-only evidence; final test remains untouched."
        ),
        "candidates": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "summary.json", result)
    _write_json(
        args.output_dir / "candidates.json",
        [asdict(candidate) for candidate in candidates],
    )
    _write_csv(args.output_dir / "fold_metrics.csv", folds)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
