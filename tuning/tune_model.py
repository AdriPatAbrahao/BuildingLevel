"""
Lightweight hyperparameter tuning for the surrogate model.

Grid parameters:
- hidden_layers: list of layer configurations
- dropout_rate: list of floats
- loss: ['mse', 'huber']
- target_weights: list of (w_steel, w_concrete)

Input dataset options:
- NPZ file with arrays 'X' (n_samples, n_features) and 'y' (n_samples, 2)
- CSV with numeric feature columns and output columns (auto-detected among common names)

Results:
- Prints best configuration and validation metrics
- Optionally saves best model .pth and scalers under outputs/experiments/<timestamp>_tuning_*
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from models.dnnmodel import SimpleNN
from utils.feature_pipeline import FeaturePipeline
from utils.experiment_manager import ExperimentManager
from config import paths


def load_dataset(input_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    input_path = Path(input_path)
    if input_path.suffix.lower() == '.npz':
        data = np.load(input_path)
        X, y = data['X'], data['y']
        assert y.shape[1] == 2, "y must have shape (n_samples, 2) for [steel, concrete]"
        return X.astype(np.float32), y.astype(np.float32)

    # CSV: try auto-separator, Python engine for flexibility
    df = pd.read_csv(input_path, sep=None, engine='python')
    # Heuristics for target columns
    steel_candidates = [
        'steel', 'aco', 'steel_kg', 'peso_aco', 'peso_acao', 'acao_kgf', 'aco_kgf'
    ]
    conc_candidates = [
        'concrete', 'concreto', 'concrete_m3', 'volume_concreto'
    ]
    cols = {c.lower(): c for c in df.columns}

    steel_col = next((cols[c] for c in cols if c in steel_candidates), None)
    conc_col = next((cols[c] for c in cols if c in conc_candidates), None)
    if steel_col is None or conc_col is None:
        raise ValueError("CSV must contain target columns for steel and concrete (e.g., 'steel' & 'concrete').")

    y = df[[steel_col, conc_col]].to_numpy(dtype=np.float32)
    feature_cols = [c for c in df.columns if c not in (steel_col, conc_col)]
    X = df[feature_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("Feature matrix X is invalid or mismatched with y.")
    return X, y


def build_model(input_size: int, output_size: int, hidden_layers: List[int], dropout: float) -> SimpleNN:
    return SimpleNN(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers, dropout_rate=dropout)


def _normalize_weights(weights, out_dim: int) -> torch.Tensor:
    # Accept scalar, list/tuple length 1 or 2; broadcast to out_dim
    if isinstance(weights, (int, float)):
        w = [float(weights)] * out_dim
    else:
        w_list = list(weights)
        if len(w_list) == out_dim:
            w = w_list
        elif len(w_list) == 1:
            w = w_list * out_dim
        else:
            # Default to equal weights
            w = [1.0] * out_dim
    return torch.tensor(w, dtype=torch.float32)


def make_criterion(loss_name: str, weights, out_dim: int):
    w = _normalize_weights(weights, out_dim)
    if loss_name == 'mse':
        def loss_fn(pred, target):
            # per-sample weighted MSE over 2 outputs
            se = (pred - target) ** 2
            return (se * w).mean()
        return loss_fn
    elif loss_name == 'huber':
        huber = nn.SmoothL1Loss(reduction='none')
        def loss_fn(pred, target):
            l = huber(pred, target)
            return (l * w).mean()
        return loss_fn
    else:
        raise ValueError(f"Unknown loss '{loss_name}'")


def train_once(X: np.ndarray, y: np.ndarray,
               hidden_layers: List[int], dropout: float,
               loss_name: str, weights,
               lr: float = 1e-3, batch_size: int = 32, epochs: int = 300,
               val_ratio: float = 0.2, patience: int = 30, seed: int = 42) -> Tuple[float, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=seed)

    # Scale
    pipeline = FeaturePipeline()
    X_tr, y_tr = pipeline.fit_transform(X_train.tolist(), y_train.tolist())
    X_va = pipeline.transform_features(X_val.tolist())
    y_va = pipeline.scaler_y.transform(y_val.astype(np.float32))

    # Datasets
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    # Model
    out_dim = y_tr.shape[1]
    model = build_model(X_tr.shape[1], out_dim, hidden_layers, dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_tr_t, y_tr_t = X_tr_t.to(device), y_tr_t.to(device)
    X_va_t, y_va_t = X_va_t.to(device), y_va_t.to(device)

    # Opt / Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = make_criterion(loss_name, weights, out_dim)

    # Training loop with early stopping on val loss
    best_val = float('inf')
    best_state = None
    patience_ctr = 0

    n = X_tr_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_va_t)
            val_loss = criterion(pred_val, y_va_t).item()

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        best_state = {
            'model': model.state_dict(),
            'input_size': int(X_tr.shape[1]),
            'output_size': int(y_tr.shape[1]),
            'hidden_layers': hidden_layers,
            'dropout_rate': float(dropout),
            'scaler_X': pipeline.scaler_X,
            'scaler_y': pipeline.scaler_y,
        }
        patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    # Also compute weighted MAE in original scale on validation set
    with torch.no_grad():
        pred_val_scaled = model(X_va_t)
    pipeline.is_fitted = True  # ensure inverse_transform works
    y_val_pred = pipeline.inverse_transform_outputs(pred_val_scaled.cpu().numpy())
    y_val_true = y_val
    mae = np.abs(y_val_pred - y_val_true).mean(axis=0)  # per-target
    mae = np.atleast_1d(mae)
    # Normalize weights to vector length out_dim
    if isinstance(weights, (list, tuple, np.ndarray)):
        w_vec = np.array(weights, dtype=float)
        if w_vec.size == 1:
            w_vec = np.repeat(w_vec[0], mae.shape[0])
        elif w_vec.size != mae.shape[0]:
            w_vec = np.ones_like(mae)
    else:
        w_vec = np.repeat(float(weights), mae.shape[0])
    weighted_mae = float(np.sum(w_vec[:mae.shape[0]] * mae))

    metrics = {
        'val_loss': float(best_val),
        'mae_steel': float(mae[0]),
        'weighted_mae': float(weighted_mae),
    }
    if mae.shape[0] > 1:
        metrics['mae_concrete'] = float(mae[1])
    return weighted_mae, {'state': best_state, 'metrics': metrics}


def main():
    ap = argparse.ArgumentParser(description="Lightweight grid search for surrogate model")
    ap.add_argument('--data', type=str, required=True,
                    help="Path to dataset (.npz with X,y or .csv with features + steel/concrete columns)")
    ap.add_argument('--targets', type=str, choices=['steel', 'both'], default='both',
                    help="Select prediction targets: 'steel' (1 output) or 'both' (2 outputs)")
    ap.add_argument('--save', action='store_true', help="Save best model and scalers under outputs/experiments")
    args = ap.parse_args()

    X, y = load_dataset(Path(args.data))
    # Reduce targets if steel-only requested
    if args.targets == 'steel':
        if y.ndim == 2 and y.shape[1] >= 1:
            y = y[:, [0]]  # only steel
        else:
            raise ValueError("Dataset must have at least 1 target column for steel.")

    # Define grid
    hidden_grid = [
        [64, 64],
        [128, 64],
        [128, 128, 64],
    ]
    dropout_grid = [0.0, 0.1, 0.2, 0.3]
    loss_grid = ['mse', 'huber']
    if y.shape[1] == 1:
        weight_grid = [1.0, 2.0]
    else:
        weight_grid = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]

    best = None
    best_payload = None
    tried = 0

    start = time.time()
    for hl in hidden_grid:
        for dr in dropout_grid:
            for loss in loss_grid:
                for w in weight_grid:
                    tried += 1
                    score, payload = train_once(
                        X, y,
                        hidden_layers=hl, dropout=dr,
                        loss_name=loss, weights=w,
                        lr=1e-3, batch_size=32, epochs=300, val_ratio=0.2,
                        patience=30, seed=42
                    )
                    metrics = payload['metrics']
                    if y.shape[1] == 1:
                        print(f"Tried #{tried:02d} | HL={hl} DR={dr} LOSS={loss} W={w} | wMAE={metrics['weighted_mae']:.4f} | MAE(steel)={metrics['mae_steel']:.4f}")
                    else:
                        print(f"Tried #{tried:02d} | HL={hl} DR={dr} LOSS={loss} W={w} | wMAE={metrics['weighted_mae']:.4f} | MAE(steel)={metrics['mae_steel']:.4f} MAE(conc)={metrics['mae_concrete']:.4f}")

                    if best is None or score < best:
                        best = score
                        best_payload = {
                            'hidden_layers': hl,
                            'dropout': dr,
                            'loss': loss,
                            'weights': w,
                            'metrics': metrics,
                        }
    elapsed = time.time() - start

    print("\n=== Best configuration ===")
    print(json.dumps(best_payload, indent=2))
    print(f"Tried {tried} combos in {elapsed:.2f}s")

    if args.save and best_payload is not None:
        # Save under a dedicated experiment dir
        run_name = (
            f"tuning_best_HL{best_payload['hidden_layers']}_DR{best_payload['dropout']}_"
            f"{best_payload['loss']}_W{best_payload['weights']}"
        )
        exp = ExperimentManager(paths.EXPERIMENTS_DIR, run_name=run_name)

        # Save scalers and torch state
        state = best_payload.get('state') or {}
        state_obj = {
            'hidden_layers': best_payload['hidden_layers'],
            'dropout': best_payload['dropout'],
            'loss': best_payload['loss'],
            'weights': best_payload['weights'],
            'metrics': best_payload['metrics'],
        }
        # Save metadata
        (exp.run_dir / 'tuning_best.json').write_text(json.dumps(state_obj, indent=2), encoding='utf-8')
        if state:
            # Save model checkpoint in nn_manager-compatible format
            checkpoint = {
                'model_state_dict': state['model'],
                'input_size': state['input_size'],
                'output_size': state['output_size'],
                'hidden_layers': state['hidden_layers'],
                'dropout_rate': state['dropout_rate'],
            }
            torch.save(checkpoint, exp.get_model_path())
            # Save scalers
            import joblib
            joblib.dump({'scaler_X': state['scaler_X'], 'scaler_y': state['scaler_y']}, exp.get_pipeline_path())
        print(f"Saved best artifacts to {exp.run_dir}")


if __name__ == '__main__':
    main()
