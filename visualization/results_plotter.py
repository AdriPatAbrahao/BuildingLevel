import warnings
import matplotlib
from sklearn.metrics import r2_score
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Callable, Sequence

class ResultsPlotter:
    def __init__(self, output_dir: Path):
        """Initialize plotter with output directory"""
        self.output_dir = output_dir
        # Garante que o diretório exista (o ExperimentManager já faz isso, mas é uma boa prática)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"ResultsPlotter configurado para salvar gráficos em: {self.output_dir.resolve()}")
        
    def plot_comparison(self, predictions, actual_values, material_type='steel'):
        """Plot predicted vs actual values comparison."""
        try:
            plt.close('all')  # Close any existing figures
            
            # Create new figure
            plt.figure(figsize=(10, 6))
            
            # Extract values for the specific material
            idx = 0 if material_type == 'steel' else 1
            pred_values = [p[idx] for p in predictions]
            true_values = [a[idx] for a in actual_values]

            if not true_values or not pred_values:
                print(f"Warning: No data to plot for {material_type}.")
                return
            
            # Plot predicted vs actual
            plt.scatter(true_values, pred_values, c='blue', alpha=0.5, label='Test Points')
            print(f"Plotting {material_type}:")
            print(f"  Actual - Min: {min(true_values):.2f}, Max: {max(true_values):.2f}, Count: {len(true_values)}")
            print(f"  Predicted - Min: {min(pred_values):.2f}, Max: {max(pred_values):.2f}, Count: {len(pred_values)}")
            # Add perfect prediction line
            min_val = min(min(true_values), min(pred_values))
            max_val = max(max(true_values), max(pred_values))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            # Calculate R² score (Coefficient of Determination)
            r2 = r2_score(true_values, pred_values)
            
            # Add labels and title
            plt.xlabel(f'TQS Calculated {material_type.title()} {"(kgf)" if material_type=="steel" else "(m³)"}')
            plt.ylabel(f'DNN Predicted {material_type.title()} {"(kgf)" if material_type=="steel" else "(m³)"}')
            plt.title(f'{material_type.title()} Prediction Comparison\nR² = {r2:.3f}')
            
            plt.grid(True)
            plt.legend()
            
            # Save and close
            filepath = self.output_dir / f'{material_type}_comparison.png'
            plt.savefig(str(filepath))
            plt.close()
        except Exception as e:
            print(f"Error plotting comparison: {str(e)}")
        
    def plot_distribution(self, outputs: List[List[float]]):
        """Plot distribution of available outputs (steel-only or steel+concrete)."""
        if not outputs:
            print("Warning: No outputs provided for distribution plot.")
            return

        steel_values = [out[0] for out in outputs if len(out) >= 1]
        has_concrete = all(len(out) >= 2 for out in outputs)
        concrete_values = [out[1] for out in outputs] if has_concrete else []

        if has_concrete:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

        # Steel distribution
        ax1.hist(steel_values, bins=30)
        ax1.set_title('Steel Distribution')
        ax1.set_xlabel('Steel (kgf)')
        ax1.set_ylabel('Frequency')

        # Concrete distribution (if present)
        if has_concrete:
            ax2.hist(concrete_values, bins=30)
            ax2.set_title('Concrete Distribution')
            ax2.set_xlabel('Concrete (m³)')
            ax2.set_ylabel('Frequency')

        plt.tight_layout()
        filepath = self.output_dir / 'material_distribution.png'
        plt.savefig(filepath)
        plt.close()

        # Print statistics
        print("\nSample Statistics:")
        print(f"Steel (kgf):")
        if steel_values:
            print(f"  Min: {min(steel_values):.2f}")
            print(f"  Max: {max(steel_values):.2f}")
            print(f"  Range: {max(steel_values) - min(steel_values):.2f}")
            print(f"  Std Dev: {np.std(steel_values):.2f}")
        if has_concrete and concrete_values:
            print(f"\nConcrete (m³):")
            print(f"  Min: {min(concrete_values):.2f}")
            print(f"  Max: {max(concrete_values):.2f}")
            print(f"  Range: {max(concrete_values) - min(concrete_values):.2f}")
            print(f"  Std Dev: {np.std(concrete_values):.2f}")

    def plot_residuals(self, predictions, actual_values, material_type='steel'):
        """Plot residuals (prediction errors) to diagnose model bias."""
        try:
            plt.close('all')
            plt.figure(figsize=(10, 6))

            idx = 0 if material_type == 'steel' else 1
            pred_values = np.array([p[idx] for p in predictions])
            true_values = np.array([a[idx] for a in actual_values])
            
            residuals = true_values - pred_values

            plt.scatter(true_values, residuals, c='green', alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--') # Zero error line
            
            plt.xlabel(f'Actual {material_type.title()} {"(kgf)" if material_type=="steel" else "(m³)"}')
            plt.ylabel('Residuals (Actual - Predicted)')
            plt.title(f'Residual Plot for {material_type.title()}')
            plt.grid(True)

            filepath = self.output_dir / f'{material_type}_residuals.png'
            plt.savefig(str(filepath))
            plt.close()
        except Exception as e:
            print(f"Error plotting residuals: {str(e)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Permutation Feature Importance  (sklearn.inspection)
    # ──────────────────────────────────────────────────────────────────────────

    def plot_permutation_importance_sklearn(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Sequence[str],
        *,
        task: str = "regression",          # "regression" | "classification"
        n_repeats: int = 15,
        top_n: int = 25,
        random_state: int = 42,
        output_file: str = "permutation_importance_sklearn.png",
    ) -> Optional[Path]:
        """
        Compute and plot Permutation Feature Importance via
        ``sklearn.inspection.permutation_importance``.

        Parameters
        ----------
        predict_fn : callable
            ``X (n, f) → predictions (n,)``  —  model-agnostic prediction
            function operating on the *already-scaled* validation array.
        X_val : np.ndarray
            Scaled validation features  (n_samples, n_features).
        y_val : np.ndarray
            True targets in **real scale** for regression (kgf), or 0/1 labels
            for classification.
        feature_names : sequence of str
            Names aligned with columns of X_val.
        task : {"regression", "classification"}
            Selects the scoring metric:
            regression      → negative MSE (higher Δ = more important)
            classification  → accuracy
        n_repeats : int
            Number of permutation repeats per feature.
        top_n : int
            How many features to show in the chart (sorted by mean importance).
        output_file : str
            Name of the saved PNG file inside ``self.output_dir``.

        Returns
        -------
        Path of the saved figure, or None on failure.
        """
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import mean_squared_error, accuracy_score
        except ImportError:
            print("[ResultsPlotter] sklearn not available — PFI skipped.")
            return None

        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val, dtype=float)

        # ── Wrap predict_fn in a minimal sklearn-compatible estimator ────────
        class _Wrapper:
            def fit(self_, X, y):           # noqa: N805  required by sklearn API
                return self_

            def predict(self_, X):          # noqa: N805
                return predict_fn(X)

            def score(self_, X, y):         # noqa: N805
                preds = self_.predict(X)
                if task == "classification":
                    return accuracy_score(y.astype(int), (preds >= 0.5).astype(int))
                return float(-mean_squared_error(y, preds))

        scoring = "neg_mean_squared_error" if task == "regression" else "accuracy"

        print(
            f"[ResultsPlotter] Computing sklearn PFI "
            f"({n_repeats} repeats, {len(feature_names)} features, task={task})…"
        )
        result = permutation_importance(
            _Wrapper(),
            X_val,
            y_val,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )

        imp_mean = result.importances_mean
        imp_std  = result.importances_std

        # Sort descending, take top_n
        order   = np.argsort(imp_mean)[::-1][:top_n]
        imp_m   = imp_mean[order]
        imp_s   = imp_std[order]
        names   = [
            feature_names[i] if i < len(feature_names) else f"f{i}"
            for i in order
        ]

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.38)))
        y_pos  = np.arange(len(names))
        colors = ["#2563EB" if v >= 0 else "#94A3B8" for v in imp_m]

        ax.barh(
            y_pos, imp_m[::-1], xerr=imp_s[::-1],
            color=colors[::-1], edgecolor="white",
            error_kw=dict(ecolor="#94A3B8", capsize=3, linewidth=1),
        )
        ax.axvline(0, color="#EF4444", lw=1.2, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=9)

        metric_label = "Δ neg-MSE" if task == "regression" else "Δ Accuracy"
        ax.set_xlabel(
            f"Variation in {metric_label} after permutation  (higher → more important)"
        )
        ax.set_title(
            f"Permutation Feature Importance — sklearn  "
            f"({'Regressão Aço' if task == 'regression' else 'Classificador Validade'})\n"
            f"Top {len(names)} features · {n_repeats} repeats",
            fontsize=11,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        out = self.output_dir / output_file
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"[ResultsPlotter] Saved: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # SHAP Summary Plot  (DeepExplainer → GradientExplainer fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def plot_shap_summary(
        self,
        model,                              # PyTorch nn.Module (eval mode)
        X_background: np.ndarray,           # scaled array used as SHAP background
        X_explain: np.ndarray,              # scaled array of instances to explain
        feature_names: Sequence[str],
        *,
        feature_pipeline=None,              # optional; used only for title
        output_file: str = "shap_summary.png",
        max_background: int = 200,
        max_explain: int = 500,
        output_index: int = 0,              # which model output to explain
    ) -> Optional[Path]:
        """
        Generate a SHAP beeswarm + bar summary plot for the DNN.

        Attempts ``shap.DeepExplainer`` first; falls back to
        ``shap.GradientExplainer`` if it fails.

        Parameters
        ----------
        model : nn.Module
            Trained PyTorch model in eval mode.  Must accept float32 tensors.
        X_background : np.ndarray
            Scaled background dataset (reference distribution).
            Up to *max_background* rows are sampled.
        X_explain : np.ndarray
            Scaled instances to explain.
            Up to *max_explain* rows are sampled.
        feature_names : sequence of str
            Names aligned with columns of X_explain.
        feature_pipeline : optional
            Unused in computation; present for future inverse-transform hooks.
        output_file : str
            Name of the saved PNG file.
        max_background : int
            Maximum background rows passed to the explainer.
        max_explain : int
            Maximum rows to explain (SHAP is O(n·f) in compute).
        output_index : int
            Which model output neuron to explain (0 = steel).

        Returns
        -------
        Path of the saved figure, or None on failure.
        """
        try:
            import shap
            import torch
        except ImportError as exc:
            print(f"[ResultsPlotter] SHAP skipped — missing library: {exc}")
            return None

        # ── Subsample ────────────────────────────────────────────────────────
        rng = np.random.default_rng(42)
        bg_idx  = rng.choice(len(X_background), min(max_background, len(X_background)), replace=False)
        exp_idx = rng.choice(len(X_explain),    min(max_explain,    len(X_explain)),    replace=False)

        bg_np  = X_background[bg_idx].astype(np.float32)
        exp_np = X_explain[exp_idx].astype(np.float32)

        bg_t   = torch.tensor(bg_np,  dtype=torch.float32)
        exp_t  = torch.tensor(exp_np, dtype=torch.float32)

        # Move model to CPU for SHAP (DeepExplainer requires CPU tensors)
        model_cpu = model.cpu()
        model_cpu.eval()

        # ── Try DeepExplainer, fall back to GradientExplainer ─────────────
        shap_values = None
        explainer_name = ""

        for attempt, ExplainerCls, name in [
            (0, shap.DeepExplainer,    "DeepExplainer"),
            (1, shap.GradientExplainer, "GradientExplainer"),
        ]:
            try:
                print(f"[ResultsPlotter] Trying SHAP {name}…")
                explainer   = ExplainerCls(model_cpu, bg_t)
                raw_values  = explainer.shap_values(exp_t)

                # shap_values can be a list (one per output) or an ndarray
                if isinstance(raw_values, list):
                    shap_values = np.array(raw_values[output_index])
                elif isinstance(raw_values, np.ndarray):
                    if raw_values.ndim == 3:
                        shap_values = raw_values[:, :, output_index]
                    else:
                        shap_values = raw_values
                else:
                    shap_values = np.array(raw_values)

                explainer_name = name
                print(
                    f"[ResultsPlotter] {name} succeeded "
                    f"({shap_values.shape[0]} instances, "
                    f"{shap_values.shape[1]} features)."
                )
                break
            except Exception as exc:
                print(f"[ResultsPlotter] {name} failed: {exc}")
                shap_values = None

        if shap_values is None:
            print("[ResultsPlotter] Both SHAP explainers failed — skipping.")
            return None

        fn_arr = list(feature_names)

        # ── Figure: beeswarm + bar (2 panels) ────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(fn_arr) * 0.30)))

        # Panel 1 — beeswarm (shows direction of impact)
        plt.sca(axes[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.summary_plot(
                shap_values,
                exp_np,
                feature_names=fn_arr,
                show=False,
                plot_size=None,
                max_display=min(25, len(fn_arr)),
            )
        axes[0].set_title(
            f"SHAP Beeswarm — {explainer_name}\n"
            f"(saída: índice {output_index} = Aço kgf)",
            fontsize=10,
        )

        # Panel 2 — bar chart of mean |SHAP|
        mean_abs = np.abs(shap_values).mean(axis=0)
        order    = np.argsort(mean_abs)[-25:]          # top 25
        plt.sca(axes[1])
        axes[1].barh(
            np.arange(len(order)),
            mean_abs[order],
            color="#2563EB",
            edgecolor="white",
        )
        axes[1].set_yticks(np.arange(len(order)))
        axes[1].set_yticklabels(
            [fn_arr[i] if i < len(fn_arr) else f"f{i}" for i in order],
            fontsize=9,
        )
        axes[1].set_xlabel("mean |SHAP value|  (impacto médio na predição)")
        axes[1].set_title("Importância Global (mean |SHAP|)", fontsize=10)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        fig.suptitle(
            f"SHAP Feature Impact — {explainer_name}  "
            f"({exp_np.shape[0]} amostras de validação)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        out = self.output_dir / output_file
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"[ResultsPlotter] Saved: {out}")
        return out
