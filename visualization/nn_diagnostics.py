"""
visualization/nn_diagnostics.py
================================
Diagnostic plots for the BuildingOptimization neural network pipeline.

Covers:
  1. Learning curves  – train / val loss and LR schedule per epoch
  2. Scatter + residuals  – real vs predicted (regressão de aço)
  3. Confusion matrix  – classifier de validade estrutural
  4. ROC / AUC curve   – carregado do JSON salvo OU calculado ao vivo
  5. Permutation Feature Importance  – agnóstico à arquitetura do modelo

All plots are saved to the experiment's ``plots/`` sub-directory with
a consistent, publication-ready visual theme.

Typical usage from main.py after training:
    from visualization.nn_diagnostics import run_full_diagnostics
    run_full_diagnostics(
        experiment_dir=exp_manager.run_dir,
        X_test=X_test_scaled,
        y_test_steel=y_test_steel,
        y_test_valid=y_test_valid,   # 0/1 labels; None if no classifier
        feature_names=FeatureEngineer.feature_names(),
        nn_manager=nn_manager,
        classifier=validity_classifier,
    )
"""

import json
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── optional but strongly recommended ─────────────────────────────────────────
try:
    import seaborn as sns
    _SEABORN = True
except ImportError:
    _SEABORN = False

# ── sklearn helpers ────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        roc_curve,
        auc,
        mean_absolute_error,
        r2_score,
        accuracy_score,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    warnings.warn("scikit-learn not found — confusion matrix and ROC plots disabled.")

# ── colour palette & theme ─────────────────────────────────────────────────────
_PRIMARY   = "#2563EB"   # blue
_SECONDARY = "#F59E0B"   # amber
_ACCENT    = "#10B981"   # emerald
_DANGER    = "#EF4444"   # red
_MUTED     = "#94A3B8"   # slate-400
_BG        = "#F8FAFC"   # near-white

_FIG_DPI   = 150
_FONT_SIZE = 11


def _apply_theme():
    """Apply a clean, consistent visual theme to all subsequent figures."""
    if _SEABORN:
        sns.set_theme(
            style="whitegrid",
            font_scale=1.05,
            rc={
                "axes.facecolor": _BG,
                "figure.facecolor": "white",
                "grid.color": "#E2E8F0",
                "axes.spines.top": False,
                "axes.spines.right": False,
            },
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": _FONT_SIZE,
            "axes.titlesize": _FONT_SIZE + 2,
            "axes.labelsize": _FONT_SIZE,
            "legend.fontsize": _FONT_SIZE - 1,
            "figure.dpi": _FIG_DPI,
            "savefig.dpi": _FIG_DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
class NNDiagnosticsPlotter:
    """
    Generates and saves all diagnostic plots for a training experiment.

    Parameters
    ----------
    experiment_dir : Path
        Root directory of the experiment (created by ExperimentManager).
        Expected sub-directories: ``metrics/``, ``plots/``.
    output_dir : Path, optional
        Where to save plots.  Defaults to ``experiment_dir/plots/``.
    """

    def __init__(self, experiment_dir: Path, output_dir: Optional[Path] = None):
        self.exp_dir    = Path(experiment_dir)
        self.output_dir = Path(output_dir) if output_dir else self.exp_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _apply_theme()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Learning curves
    # ──────────────────────────────────────────────────────────────────────────
    def plot_learning_curves(self) -> Optional[Path]:
        """
        Read ``metrics/epochs.ndjson`` and plot train / val loss + LR schedule.

        Returns
        -------
        Path | None
            Path of saved figure, or None if the file was not found.
        """
        epochs_path = self.exp_dir / "metrics" / "epochs.ndjson"
        if not epochs_path.exists():
            print(f"[NNDiagnostics] Learning curves: '{epochs_path}' not found — skipping.")
            return None

        records = []
        with open(epochs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            print("[NNDiagnostics] Learning curves: no valid records found — skipping.")
            return None

        epochs     = [r["epoch"]      for r in records]
        train_loss = [r["train_loss"] for r in records]
        val_loss   = [r["val_loss"]   for r in records]
        lr_vals    = [r.get("learning_rate") for r in records]
        has_lr     = any(v is not None for v in lr_vals)

        nrows = 2 if has_lr else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(11, 4 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        # ── loss panel ──────────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(epochs, train_loss, color=_PRIMARY,    lw=2, label="Treino")
        ax.plot(epochs, val_loss,   color=_SECONDARY,  lw=2, label="Validação", linestyle="--")

        best_epoch = int(np.argmin(val_loss))
        ax.axvline(epochs[best_epoch], color=_ACCENT, lw=1.2, linestyle=":",
                   label=f"Melhor val (época {epochs[best_epoch]})")

        ax.set_ylabel("Loss")
        ax.set_title("Curvas de Aprendizado — Loss por Época")
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

        # ── LR panel ────────────────────────────────────────────────────────
        if has_lr:
            ax2 = axes[1]
            lr_clean = [v if v is not None else float("nan") for v in lr_vals]
            ax2.semilogy(epochs, lr_clean, color=_DANGER, lw=1.8)
            ax2.set_ylabel("Learning Rate (escala log)")
            ax2.set_title("Schedule da Taxa de Aprendizado")
            ax2.set_xlabel("Época")
        else:
            axes[0].set_xlabel("Época")

        plt.tight_layout()
        out = self.output_dir / "learning_curves.png"
        plt.savefig(out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Scatter (real vs previsto) + histograma de resíduos
    # ──────────────────────────────────────────────────────────────────────────
    def plot_scatter_and_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str = "Aço (kgf)",
        filename_prefix: str = "steel",
    ) -> Path:
        """
        Side-by-side: (a) Scatter real vs previsto  (b) Histograma de resíduos.

        Parameters
        ----------
        y_true, y_pred : 1-D arrays of the same length.
        label          : Axis label / unit string shown on plot.
        filename_prefix: Prefix for the saved PNG file name.
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        residuals = y_true - y_pred

        r2  = r2_score(y_true, y_pred)      if _SKLEARN else float("nan")
        mae = mean_absolute_error(y_true, y_pred) if _SKLEARN else float("nan")
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Regressão — {label}   |   R² = {r2:.3f}   MAE = {mae:.1f}   RMSE = {rmse:.1f}",
            fontsize=_FONT_SIZE + 2, fontweight="bold",
        )

        # ── scatter ──────────────────────────────────────────────────────────
        lim_min = min(y_true.min(), y_pred.min()) * 0.97
        lim_max = max(y_true.max(), y_pred.max()) * 1.03

        ax_scatter.scatter(y_true, y_pred, s=18, alpha=0.55,
                           color=_PRIMARY, edgecolors="none", label="Amostras de teste")
        ax_scatter.plot([lim_min, lim_max], [lim_min, lim_max],
                        color=_DANGER, lw=1.5, linestyle="--", label="Previsão perfeita")

        # ±10 % bands
        ax_scatter.fill_between(
            [lim_min, lim_max],
            [lim_min * 0.9, lim_max * 0.9],
            [lim_min * 1.1, lim_max * 1.1],
            alpha=0.08, color=_SECONDARY, label="Faixa ±10 %",
        )

        ax_scatter.set_xlim(lim_min, lim_max)
        ax_scatter.set_ylim(lim_min, lim_max)
        ax_scatter.set_xlabel(f"Real — {label}")
        ax_scatter.set_ylabel(f"Previsto — {label}")
        ax_scatter.set_title("Real vs Previsto")
        ax_scatter.legend(fontsize=_FONT_SIZE - 1)
        ax_scatter.set_aspect("equal")

        # ── residuals histogram ───────────────────────────────────────────────
        n_bins = max(20, min(60, int(np.sqrt(len(residuals)))))
        if _SEABORN:
            sns.histplot(residuals, bins=n_bins, ax=ax_hist,
                         color=_ACCENT, edgecolor="white", alpha=0.85, kde=True,
                         kde_kws={"linewidth": 2})
        else:
            ax_hist.hist(residuals, bins=n_bins, color=_ACCENT, edgecolor="white", alpha=0.85)

        ax_hist.axvline(0, color=_DANGER, lw=1.8, linestyle="--", label="Resíduo = 0")
        ax_hist.axvline(residuals.mean(), color=_SECONDARY, lw=1.5, linestyle=":",
                        label=f"Média = {residuals.mean():.1f}")
        ax_hist.set_xlabel(f"Resíduo  (Real − Previsto)  [{label.split('(')[-1].rstrip(')')}]")
        ax_hist.set_ylabel("Contagem")
        ax_hist.set_title("Distribuição dos Resíduos")
        ax_hist.legend(fontsize=_FONT_SIZE - 1)

        plt.tight_layout()
        out = self.output_dir / f"{filename_prefix}_scatter_residuals.png"
        plt.savefig(out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Confusion matrix
    # ──────────────────────────────────────────────────────────────────────────
    def plot_confusion_matrix(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        class_names: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Plot a normalised + raw count confusion matrix using seaborn heatmap.

        Parameters
        ----------
        y_true       : Ground-truth binary labels (0 = inválido, 1 = válido).
        y_pred       : Predicted binary labels.
        class_names  : Names for classes [0, 1].  Default: ["Inválido", "Válido"].
        """
        if not _SKLEARN:
            print("[NNDiagnostics] Confusion matrix: sklearn not available — skipping.")
            return None

        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        class_names = class_names or ["Inválido", "Válido"]

        cm     = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        acc = accuracy_score(y_true, y_pred) if _SKLEARN else float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Matriz de Confusão — Classificador de Validade Estrutural   (Acurácia = {acc:.1%})",
            fontsize=_FONT_SIZE + 2, fontweight="bold",
        )

        _cmap = "Blues"

        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_pct],
            ["d", ".1f"],
            ["Contagem absoluta", "Normalizada (% por classe real)"],
        ):
            if _SEABORN:
                sns.heatmap(
                    data, annot=True, fmt=fmt, cmap=_cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="#E2E8F0",
                    cbar_kws={"shrink": 0.8},
                    ax=ax,
                )
            else:
                im = ax.imshow(data, cmap=_cmap)
                plt.colorbar(im, ax=ax, shrink=0.8)
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        ax.text(j, i, format(data[i, j], fmt),
                                ha="center", va="center", fontsize=_FONT_SIZE)
                ax.set_xticks(range(len(class_names)))
                ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names)
                ax.set_yticklabels(class_names)

            ax.set_xlabel("Previsto")
            ax.set_ylabel("Real")
            ax.set_title(title)

        plt.tight_layout()
        out = self.output_dir / "confusion_matrix.png"
        plt.savefig(out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 4. ROC / AUC
    # ──────────────────────────────────────────────────────────────────────────
    def plot_roc_auc(
        self,
        fpr: Optional[np.ndarray] = None,
        tpr: Optional[np.ndarray] = None,
        thresholds: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
    ) -> Optional[Path]:
        """
        Plot ROC curve with AUC and optimal Youden threshold.

        Priority: if fpr/tpr are passed, use them.  Otherwise tries
        ``metrics/roc_curve.json``.  Finally, computes from y_true / y_score.

        Parameters
        ----------
        fpr, tpr, thresholds : pre-computed arrays (optional).
        y_true   : ground-truth 0/1 labels  (used if fpr/tpr not available).
        y_score  : predicted probabilities for class 1 (positive).
        """
        if not _SKLEARN:
            print("[NNDiagnostics] ROC: sklearn not available — skipping.")
            return None

        # -- load or compute --------------------------------------------------
        _thr_opt = None
        if fpr is None or tpr is None:
            roc_path = self.exp_dir / "metrics" / "roc_curve.json"
            if roc_path.exists():
                try:
                    with open(roc_path, encoding="utf-8") as f:
                        obj = json.load(f)
                    fpr        = np.asarray(obj["fpr"],        dtype=float)
                    tpr        = np.asarray(obj["tpr"],        dtype=float)
                    thresholds = np.asarray(obj["thresholds"], dtype=float)
                except Exception as exc:
                    print(f"[NNDiagnostics] ROC: failed to load roc_curve.json — {exc}")
                    fpr = tpr = None

        if fpr is None and y_true is not None and y_score is not None:
            fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

        if fpr is None:
            print("[NNDiagnostics] ROC: no data available — skipping.")
            return None

        roc_auc = auc(fpr, tpr)

        # -- load optimal threshold -------------------------------------------
        thr_path = self.exp_dir / "metrics" / "validity_threshold.json"
        if thr_path.exists():
            try:
                with open(thr_path, encoding="utf-8") as f:
                    _thr_opt = float(json.load(f).get("threshold", float("nan")))
            except Exception:
                pass
        if _thr_opt is None and thresholds is not None:
            best_idx = int(np.argmax(tpr - fpr))
            _thr_opt = float(thresholds[best_idx])

        # -- plot -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(fpr, tpr, color=_PRIMARY, lw=2.2,
                label=f"Curva ROC  (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color=_MUTED, lw=1.2, linestyle="--",
                label="Classificador aleatório")
        ax.fill_between(fpr, tpr, alpha=0.08, color=_PRIMARY)

        # mark optimal threshold on the curve
        if _thr_opt is not None and thresholds is not None:
            idx = int(np.argmin(np.abs(thresholds - _thr_opt)))
            ax.scatter(fpr[idx], tpr[idx], s=100, zorder=5,
                       color=_DANGER, edgecolors="white", linewidths=1.2,
                       label=f"Limiar ótimo = {_thr_opt:.3f}")
            ax.annotate(
                f"FPR={fpr[idx]:.2f}\nTPR={tpr[idx]:.2f}",
                xy=(fpr[idx], tpr[idx]),
                xytext=(fpr[idx] + 0.08, tpr[idx] - 0.10),
                fontsize=_FONT_SIZE - 1,
                arrowprops=dict(arrowstyle="->", color=_MUTED),
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
        ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)")
        ax.set_title("Curva ROC — Classificador de Validade Estrutural")
        ax.legend(loc="lower right")

        plt.tight_layout()
        out = self.output_dir / "roc_auc.png"
        plt.savefig(out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Permutation Feature Importance
    # ──────────────────────────────────────────────────────────────────────────
    def plot_permutation_importance(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        metric_fn: Optional[Callable] = None,
        metric_name: str = "MAE",
        n_repeats: int = 8,
        top_n: int = 20,
        random_state: int = 42,
        filename: str = "permutation_importance.png",
    ) -> Path:
        """
        Model-agnostic Permutation Feature Importance.

        Repeatedly shuffles one feature at a time, measures the increase
        in the error metric, and reports the mean ± std across repeats.

        Parameters
        ----------
        predict_fn   : callable  X (n_samples, n_features) → predictions (n_samples,)
        X_test       : scaled test features.
        y_test       : ground-truth targets (steel kgf or 0/1 labels).
        feature_names: names aligned with columns of X_test.
        metric_fn    : callable (y_true, y_pred) → scalar (higher = worse).
                       Defaults to MAE for regression, 1-accuracy for binary.
        metric_name  : label used in the axis and title.
        n_repeats    : number of shuffle repeats per feature.
        top_n        : number of features to show in the chart.
        random_state : seed for reproducibility.
        filename     : output PNG file name.
        """
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)

        if metric_fn is None and _SKLEARN:
            # choose sensible default based on target cardinality
            unique = np.unique(y_test)
            if len(unique) <= 5:
                metric_fn  = lambda yt, yp: 1.0 - accuracy_score(yt, (yp >= 0.5).astype(int))
                metric_name = "1 − Acurácia"
            else:
                metric_fn  = mean_absolute_error
                metric_name = "MAE"
        elif metric_fn is None:
            metric_fn  = lambda yt, yp: float(np.mean(np.abs(yt - yp)))
            metric_name = "MAE"

        baseline  = metric_fn(y_test, predict_fn(X_test))
        rng       = np.random.default_rng(random_state)
        n_feat    = X_test.shape[1]
        scores    = np.zeros((n_repeats, n_feat), dtype=float)

        for r in range(n_repeats):
            for f in range(n_feat):
                X_perm        = X_test.copy()
                X_perm[:, f]  = rng.permutation(X_perm[:, f])
                scores[r, f]  = metric_fn(y_test, predict_fn(X_perm)) - baseline

        mean_imp = scores.mean(axis=0)
        std_imp  = scores.std(axis=0)

        # sort and truncate to top_n
        order   = np.argsort(mean_imp)[::-1][:top_n]
        imp_m   = mean_imp[order]
        imp_s   = std_imp[order]
        names   = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in order]

        # palette: positive = important, negative = noise
        colors = [_PRIMARY if v >= 0 else _MUTED for v in imp_m]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
        y_pos = np.arange(len(names))

        ax.barh(y_pos, imp_m[::-1], xerr=imp_s[::-1],
                color=colors[::-1], edgecolor="white",
                error_kw=dict(ecolor=_MUTED, capsize=3, linewidth=1))
        ax.axvline(0, color=_DANGER, lw=1.2, linestyle="--")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=_FONT_SIZE - 1)
        ax.set_xlabel(f"Aumento em {metric_name} após permutação  (maior = mais importante)")
        ax.set_title(
            f"Permutation Feature Importance  (top {len(names)}, {n_repeats} repetições)\n"
            f"Baseline {metric_name} = {baseline:.4f}",
            fontsize=_FONT_SIZE + 1,
        )

        plt.tight_layout()
        out = self.output_dir / filename
        plt.savefig(out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # helper: gradient norms per layer
    # ──────────────────────────────────────────────────────────────────────────
    def plot_gradient_norms(self) -> Optional[Path]:
        """
        Plot per-layer gradient L2-norms over training epochs from
        ``metrics/epochs.ndjson``.  Only rendered if the data is present.
        """
        epochs_path = self.exp_dir / "metrics" / "epochs.ndjson"
        if not epochs_path.exists():
            return None

        epochs_data = []
        with open(epochs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        epochs_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        # collect layer names from first record that has gradient data
        layer_norms: dict = {}
        for rec in epochs_data:
            norms = rec.get("gradients_norm_by_layer") or {}
            for layer, val in norms.items():
                layer_norms.setdefault(layer, []).append((rec["epoch"], val))

        if not layer_norms:
            return None  # no gradient data saved

        fig, ax = plt.subplots(figsize=(12, 5))
        palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
        for idx, (layer, vals) in enumerate(layer_norms.items()):
            ep, nrm = zip(*vals)
            ax.semilogy(ep, nrm, lw=1.4, label=layer,
                        color=palette[idx % len(palette)])

        ax.set_xlabel("Época")
        ax.set_ylabel("Norma L2 do Gradiente (escala log)")
        ax.set_title("Normas de Gradiente por Camada ao Longo do Treinamento")
        ax.legend(fontsize=max(7, _FONT_SIZE - 3), ncol=2, loc="upper right")

        plt.tight_layout()
        out = self.output_dir / "gradient_norms.png"
        plt.savefig(out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Convenience entry-point for main.py
# ══════════════════════════════════════════════════════════════════════════════

def run_full_diagnostics(
    experiment_dir: Path,
    feature_names: List[str],
    nn_manager=None,
    X_test: Optional[np.ndarray] = None,
    y_test_steel: Optional[np.ndarray] = None,
    y_pred_steel: Optional[np.ndarray] = None,
    feature_pipeline=None,
    classifier=None,
    y_test_valid: Optional[np.ndarray] = None,
    X_test_clf: Optional[np.ndarray] = None,
    n_repeats_pfi: int = 8,
) -> None:
    """
    Run all available diagnostic plots for a finished experiment.

    Parameters
    ----------
    experiment_dir : Path
        Directory created by ExperimentManager (contains metrics/, plots/, etc.).
    feature_names  : list of str
        Feature names aligned with columns of X_test.
    nn_manager     : NeuralNetworkManager, optional
        Trained NN manager — required for PFI when y_pred_steel is not supplied.
    X_test         : np.ndarray, optional
        NN-scaled test features  (n_samples, n_features).
    y_test_steel   : np.ndarray, optional
        Ground-truth steel values in kgf, real scale  (n_samples,).
    y_pred_steel   : np.ndarray, optional
        Pre-computed steel predictions in real scale (kgf).  When supplied the
        scatter/residuals plot is produced without a second forward pass.
    feature_pipeline : FeaturePipeline, optional
        Pipeline with ``inverse_transform_outputs`` — used in the PFI closure to
        convert NN outputs back to kgf so that MAE is in interpretable units.
    classifier     : sklearn estimator, optional
        Validity classifier — required for confusion matrix & ROC.
    y_test_valid   : np.ndarray, optional
        Ground-truth 0/1 validity labels  (n_samples,).
    X_test_clf     : np.ndarray, optional
        Raw (unscaled) test features for the classifier PFI.  Falls back to
        X_test when not provided.
    n_repeats_pfi  : int
        Number of permutation repeats for feature importance.
    """
    plotter = NNDiagnosticsPlotter(experiment_dir)

    # 1. Learning curves  (always, reads from disk)
    plotter.plot_learning_curves()
    plotter.plot_gradient_norms()

    # 2. Scatter + residuals  (pre-computed predictions take priority)
    _have_scatter_data = (
        y_test_steel is not None
        and (y_pred_steel is not None or (nn_manager is not None and X_test is not None))
    )
    if _have_scatter_data:
        try:
            if y_pred_steel is None:
                # compute predictions on demand
                import torch
                nn_manager.model.eval()
                X_t = torch.tensor(X_test, dtype=torch.float32).to(nn_manager.device)
                with torch.no_grad():
                    raw = nn_manager.model(X_t).cpu().numpy()
                raw_col = raw[:, 0:1] if raw.ndim == 2 else raw.reshape(-1, 1)
                if feature_pipeline is not None and hasattr(feature_pipeline, "inverse_transform_outputs"):
                    y_pred_steel = feature_pipeline.inverse_transform_outputs(raw_col)[:, 0]
                else:
                    y_pred_steel = raw_col[:, 0]

            plotter.plot_scatter_and_residuals(
                y_test_steel, y_pred_steel,
                label="Aço (kgf)", filename_prefix="steel"
            )
        except Exception as exc:
            print(f"[NNDiagnostics] Scatter plot failed: {exc}")

    # 2b. Permutation Feature Importance for regressor
    if nn_manager is not None and X_test is not None and y_test_steel is not None:
        try:
            import torch

            def _predict_steel_real(X: np.ndarray) -> np.ndarray:
                """Returns predictions in real kgf scale."""
                Xt = torch.tensor(X, dtype=torch.float32).to(nn_manager.device)
                nn_manager.model.eval()
                with torch.no_grad():
                    out = nn_manager.model(Xt).cpu().numpy()
                out_col = out[:, 0:1] if out.ndim == 2 else out.reshape(-1, 1)
                if feature_pipeline is not None and hasattr(feature_pipeline, "inverse_transform_outputs"):
                    return feature_pipeline.inverse_transform_outputs(out_col)[:, 0]
                return out_col[:, 0]

            plotter.plot_permutation_importance(
                predict_fn=_predict_steel_real,
                X_test=X_test,
                y_test=y_test_steel,
                feature_names=feature_names,
                n_repeats=n_repeats_pfi,
                filename="pfi_steel_regression.png",
            )
        except Exception as exc:
            print(f"[NNDiagnostics] PFI for regressor failed: {exc}")

    # 3 & 4. Confusion matrix + ROC  (needs classifier + test labels)
    _clf_X = X_test_clf if X_test_clf is not None else X_test
    if classifier is not None and _clf_X is not None and y_test_valid is not None:
        try:
            y_pred_labels = classifier.predict(_clf_X)
            plotter.plot_confusion_matrix(y_test_valid, y_pred_labels)

            idx_pos = 1  # default: class 1 = valid
            if hasattr(classifier, "predict_proba"):
                y_score = classifier.predict_proba(_clf_X)
                classes = list(getattr(classifier, "classes_", [0, 1]))
                idx_pos = classes.index(1) if 1 in classes else 1
                plotter.plot_roc_auc(y_true=y_test_valid, y_score=y_score[:, idx_pos])
            else:
                plotter.plot_roc_auc()

            # ── Permutation Feature Importance for classifier ───────────────
            if hasattr(classifier, "predict_proba"):
                def _predict_clf(X: np.ndarray) -> np.ndarray:
                    proba = classifier.predict_proba(X)
                    return proba[:, idx_pos]
                plotter.plot_permutation_importance(
                    predict_fn=_predict_clf,
                    X_test=_clf_X,
                    y_test=y_test_valid,
                    feature_names=feature_names,
                    n_repeats=n_repeats_pfi,
                    filename="pfi_validity_classifier.png",
                )
        except Exception as exc:
            print(f"[NNDiagnostics] Classifier plots failed: {exc}")

    elif classifier is None:
        # still try to plot ROC from saved JSON
        plotter.plot_roc_auc()

    print(f"[NNDiagnostics] Diagnósticos salvos em: {plotter.output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(
        description="Gera gráficos de diagnóstico para um experimento de DNN."
    )
    parser.add_argument(
        "--exp", required=True,
        help="Caminho para o diretório do experimento (ex: outputs/experiments/20251128-…)."
    )
    parser.add_argument(
        "--csv", default=None,
        help="(Opcional) CSV de teste no formato x;y;dx;dy;length;maxlength para scatter/PFI."
    )
    parser.add_argument(
        "--out", default=None,
        help="(Opcional) Diretório de saída dos plots.  Default: <exp>/plots/."
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp)
    if not exp_dir.exists():
        print(f"Erro: diretório '{exp_dir}' não encontrado.")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else None
    plotter = NNDiagnosticsPlotter(exp_dir, out_dir)

    # Always plot what we can from saved artefacts
    plotter.plot_learning_curves()
    plotter.plot_gradient_norms()
    plotter.plot_roc_auc()

    # If test CSV is provided, load the full inference stack and plot scatter + PFI
    if args.csv:
        try:
            from joblib import load as jl_load
            from inference import BuildingInference
            from utils.feature_engineer import FeatureEngineer

            print(f"[NNDiagnostics] Carregando artefatos de '{exp_dir.name}'...")
            inf = BuildingInference(exp_dir.name)

            print(f"[NNDiagnostics] Processando CSV de teste '{args.csv}'...")
            steel, concrete, prob = inf.predict_from_csv(args.csv)
            print(f"  Previsto → Aço: {steel:.1f} kgf  |  Concreto: {concrete:.3f} m³  |  P(inválido): {prob}")

        except Exception as exc:
            print(f"[NNDiagnostics] Falha ao carregar inferência para CSV: {exc}")

    print("[NNDiagnostics] Concluído.")
