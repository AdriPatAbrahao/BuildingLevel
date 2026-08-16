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

from utils.classifier_evaluation import (
    INVALID_LABEL,
    invalid_probability_index,
    validity_labels_from_invalid_probability,
)
from utils.plot_data_store import PlotDataStore
from visualization.thesis_style import (
    COLORS,
    FULL_WIDTH,
    ANNOTATION_SIZE,
    LEGEND_SIZE,
    SEQUENTIAL_CMAP,
    SINGLE_COLUMN,
    SQUARE,
    SUBTITLE_SIZE,
    THESIS_DPI,
    TWO_PANEL,
    add_panel_labels,
    apply_thesis_style,
    save_thesis_figure,
)

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
_PRIMARY   = COLORS["primary"]
_SECONDARY = COLORS["accent"]
_ACCENT    = COLORS["secondary"]
_DANGER    = COLORS["accent"]
_MUTED     = COLORS["gray"]
_BG        = COLORS["background"]

_FIG_DPI   = THESIS_DPI
_FONT_SIZE = 11


def _apply_theme():
    """Apply a clean, consistent visual theme to all subsequent figures."""
    apply_thesis_style()


def _mean_is_visually_distinct_from_zero(
    values: np.ndarray,
    bin_count: int,
) -> bool:
    """Return whether separate zero/mean lines are resolvable at histogram scale."""
    values = np.asarray(values, dtype=float)
    value_range = float(np.ptp(values))
    if not np.isfinite(value_range) or value_range <= 0:
        return False
    half_bin_width = value_range / (2.0 * max(int(bin_count), 1))
    return abs(float(np.mean(values))) > half_bin_width


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
        fig, axes = plt.subplots(
            nrows,
            1,
            figsize=(FULL_WIDTH[0], 3.0 * nrows),
            sharex=True,
        )
        if nrows == 1:
            axes = [axes]

        # ── loss panel ──────────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(epochs, train_loss, color=_PRIMARY,    lw=2, label="Training")
        ax.plot(epochs, val_loss,   color=_SECONDARY,  lw=2, label="Validation", linestyle="--")

        best_epoch = int(np.argmin(val_loss))
        ax.axvline(epochs[best_epoch], color=_ACCENT, lw=1.2, linestyle=":",
                   label=f"Best validation loss (epoch {epochs[best_epoch]})")

        ax.set_ylabel("Loss")
        ax.set_title("Learning Curves — Loss by Epoch")
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

        # ── LR panel ────────────────────────────────────────────────────────
        if has_lr:
            ax2 = axes[1]
            lr_clean = [v if v is not None else float("nan") for v in lr_vals]
            ax2.semilogy(epochs, lr_clean, color=_DANGER, lw=1.8)
            ax2.set_ylabel("Learning Rate (log scale)")
            ax2.set_title("Learning Rate Schedule")
            ax2.set_xlabel("Epoch")
        else:
            axes[0].set_xlabel("Epoch")

        plt.tight_layout()
        out = self.output_dir / "learning_curves.png"
        save_thesis_figure(fig, out)
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
        label: str = "Reinforcement steel weight (kgf)",
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

        fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(8.2, 3.7))
        fig.suptitle("Reinforcement Steel Regressor")

        # ── scatter ──────────────────────────────────────────────────────────
        lim_min = min(y_true.min(), y_pred.min()) * 0.97
        lim_max = max(y_true.max(), y_pred.max()) * 1.03

        ax_scatter.scatter(y_true, y_pred, s=18, alpha=0.55,
                           color=_PRIMARY, edgecolors="none", label="Test samples")
        ax_scatter.plot([lim_min, lim_max], [lim_min, lim_max],
                        color=_DANGER, lw=1.5, linestyle="--", label="1:1 reference line")

        # ±10 % bands
        ax_scatter.fill_between(
            [lim_min, lim_max],
            [lim_min * 0.9, lim_max * 0.9],
            [lim_min * 1.1, lim_max * 1.1],
            alpha=0.08, color=_SECONDARY, label="±10% error band",
        )

        ax_scatter.set_xlim(lim_min, lim_max)
        ax_scatter.set_ylim(lim_min, lim_max)
        unit = label.split("(")[-1].rstrip(")")
        ax_scatter.set_xlabel(f"Observed (TQS) [{unit}]")
        ax_scatter.set_ylabel(f"Predicted (DNN) [{unit}]")
        ax_scatter.set_title("Observed (TQS) vs. Predicted")
        ax_scatter.legend(loc="lower right", fontsize=_FONT_SIZE - 1)
        ax_scatter.set_aspect("equal")
        ax_scatter.text(
            0.03,
            0.97,
            f"R² = {r2:.3f}\nMAE = {mae:.1f} kgf\nRMSE = {rmse:.1f} kgf",
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            fontsize=_FONT_SIZE - 2,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": COLORS["light_gray"], "alpha": 0.92},
        )

        # ── residuals histogram ───────────────────────────────────────────────
        n_bins = max(20, min(60, int(np.sqrt(len(residuals)))))
        if _SEABORN:
            sns.histplot(residuals, bins=n_bins, ax=ax_hist,
                         color=_PRIMARY, edgecolor="white", alpha=0.85, kde=True,
                         kde_kws={"linewidth": 2})
        else:
            ax_hist.hist(residuals, bins=n_bins, color=_PRIMARY, edgecolor="white", alpha=0.85)

        # Freeze the autoscaled y range before adding reference lines. Limiting
        # the lines to 96% avoids a dashed cap being rendered against the top edge.
        data_hist_top = ax_hist.get_ylim()[1]
        final_hist_ylim = (0.0, data_hist_top * 1.35)
        ax_hist.set_ylim(final_hist_ylim)
        reference_line_top = min(0.96, data_hist_top * 1.02 / final_hist_ylim[1])
        mean_residual = float(residuals.mean())
        mean_is_distinct = _mean_is_visually_distinct_from_zero(residuals, n_bins)
        zero_label = (
            "Zero residual"
            if mean_is_distinct
            else "Zero residual (mean ≈ 0)"
        )
        ax_hist.axvline(
            0, ymin=0.0, ymax=reference_line_top, color=_DANGER, lw=1.8,
            linestyle="--", label=zero_label, clip_on=True,
            dash_capstyle="butt",
        )
        if mean_is_distinct:
            ax_hist.axvline(
                mean_residual, ymin=0.0, ymax=reference_line_top,
                color=_SECONDARY, lw=1.5, linestyle=":",
                label=f"Mean residual = {mean_residual:.1f}", clip_on=True,
                dash_capstyle="butt",
            )
        ax_hist.set_xlabel(f"Residual (Observed − Predicted) [{unit}]")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Residual Distribution")
        ax_hist.legend(
            loc="upper center", ncol=1, fontsize=_FONT_SIZE - 1,
            framealpha=1.0, facecolor="white",
            edgecolor=COLORS["light_gray"],
        )

        add_panel_labels([ax_scatter, ax_hist])
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.79, wspace=0.31)
        out = self.output_dir / f"{filename_prefix}_scatter_residuals.png"
        save_thesis_figure(fig, out)
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
        invalid_threshold: Optional[float] = None,
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
        class_names = class_names or ["Infeasible", "Feasible"]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_pct = np.divide(
            cm.astype(float),
            cm.sum(axis=1, keepdims=True),
            out=np.zeros_like(cm, dtype=float),
            where=cm.sum(axis=1, keepdims=True) != 0,
        ) * 100

        acc = accuracy_score(y_true, y_pred) if _SKLEARN else float("nan")
        critical_count = int(cm[0, 1])
        invalid_total = int(cm[0].sum())
        critical_rate = critical_count / invalid_total if invalid_total else 0.0
        threshold_text = (
            f" · P(infeasible) threshold = {invalid_threshold:.3f}"
            if invalid_threshold is not None
            else ""
        )

        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
        fig.suptitle("Structural Feasibility Classifier — Confusion Matrices")
        fig.text(
            0.5,
            0.925,
            f"Accuracy = {acc:.1%}{threshold_text} · "
            f"Critical false negatives = {critical_count}/{invalid_total} ({critical_rate:.1%})",
            ha="center",
            va="top",
            fontsize=SUBTITLE_SIZE,
            fontweight="normal",
        )

        _cmap = SEQUENTIAL_CMAP

        for ax, data, fmt, title, colorbar_label in zip(
            axes,
            [cm, cm_pct],
            ["d", ".1f"],
            ["Counts", "Row-normalized (%)"],
            ["Count", "Row percentage (%)"],
        ):
            color_max = max(float(np.nanmax(data)), 1.0)
            if _SEABORN:
                sns.heatmap(
                    data, annot=True, fmt=fmt, cmap=_cmap,
                    vmin=0.0, vmax=color_max,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="#E2E8F0",
                    cbar_kws={"shrink": 0.8, "label": colorbar_label},
                    ax=ax,
                )
            else:
                im = ax.imshow(data, cmap=_cmap, vmin=0.0, vmax=color_max)
                colorbar = plt.colorbar(im, ax=ax, shrink=0.8)
                colorbar.set_label(colorbar_label)
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        ax.text(j, i, format(data[i, j], fmt),
                                ha="center", va="center", fontsize=_FONT_SIZE)
                ax.set_xticks(range(len(class_names)))
                ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names)
                ax.set_yticklabels(class_names)

            color_threshold = float(np.nanmin(data) + 0.45 * np.ptp(data))
            for annotation, value in zip(ax.texts, np.asarray(data).ravel()):
                annotation.set_color("white" if float(value) <= color_threshold else "#111111")

            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.set_title(title)
            ax.grid(False)

        add_panel_labels(axes)
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.76, wspace=0.38)
        out = self.output_dir / "confusion_matrix.png"
        save_thesis_figure(fig, out)
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
        threshold_to_mark: Optional[float] = None,
        split_label: Optional[str] = None,
        positive_class_label: str = "Infeasible",
    ) -> Optional[Path]:
        """
        Plot ROC curve with AUC and optimal Youden threshold.

        Priority: explicit arrays, then supplied test labels/scores, then
        ``metrics/roc_curve_test.json``.  Validation ROC is only a fallback for
        legacy experiments that do not contain test ROC data.

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
        _thr_opt = threshold_to_mark
        if (fpr is None or tpr is None) and y_true is not None and y_score is not None:
            fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

        if fpr is None or tpr is None:
            candidates = [
                self.exp_dir / "metrics" / "roc_curve_test.json",
                self.exp_dir / "metrics" / "roc_curve.json",
            ]
            for roc_path in candidates:
                if not roc_path.exists():
                    continue
                try:
                    with open(roc_path, encoding="utf-8") as f:
                        obj = json.load(f)
                    fpr = np.asarray(obj["fpr"], dtype=float)
                    tpr = np.asarray(obj["tpr"], dtype=float)
                    thresholds = np.asarray(obj["thresholds"], dtype=float)
                    split_label = split_label or str(obj.get("split", "unknown"))
                    break
                except Exception as exc:
                    print(f"[NNDiagnostics] ROC: failed to load {roc_path.name} — {exc}")
                    fpr = tpr = None

        if fpr is None:
            print("[NNDiagnostics] ROC: no data available — skipping.")
            return None

        roc_auc = auc(fpr, tpr)

        # -- load optimal threshold -------------------------------------------
        thr_path = self.exp_dir / "metrics" / "validity_threshold.json"
        if _thr_opt is None and thr_path.exists():
            try:
                with open(thr_path, encoding="utf-8") as f:
                    _thr_opt = float(json.load(f).get("threshold", float("nan")))
            except Exception:
                pass
        if _thr_opt is None and thresholds is not None:
            best_idx = int(np.argmax(tpr - fpr))
            _thr_opt = float(thresholds[best_idx])

        # -- plot -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        fig.suptitle("Receiver Operating Characteristic (ROC)")

        ax.plot(fpr, tpr, color=_PRIMARY, lw=2.2,
                label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color=_MUTED, lw=1.2, linestyle="--",
                label="No-skill classifier")
        ax.fill_between(fpr, tpr, alpha=0.08, color=_PRIMARY)

        # mark optimal threshold on the curve
        if _thr_opt is not None and thresholds is not None:
            threshold_distance = np.abs(thresholds - _thr_opt)
            threshold_distance[~np.isfinite(threshold_distance)] = np.inf
            idx = int(np.argmin(threshold_distance))
            ax.scatter(fpr[idx], tpr[idx], s=100, zorder=5,
                       color=_DANGER, edgecolors="white", linewidths=1.2,
                       label=f"Validation-selected threshold = {_thr_opt:.3f}")
            annotation_x = float(np.clip(fpr[idx] + 0.08, 0.08, 0.75))
            annotation_y = float(np.clip(tpr[idx] - 0.10, 0.15, 0.92))
            ax.annotate(
                f"FPR={fpr[idx]:.2f}\nTPR={tpr[idx]:.2f}",
                xy=(fpr[idx], tpr[idx]),
                xytext=(annotation_x, annotation_y),
                fontsize=_FONT_SIZE - 1,
                zorder=7,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": _MUTED,
                    "linewidth": 1.2,
                    "mutation_scale": 12,
                    "shrinkB": 8,
                    "zorder": 6,
                },
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("False Positive Rate (1 − Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity/Recall)")
        split_title = f"{split_label.capitalize()} set · " if split_label else ""
        fig.text(
            0.5,
            0.91,
            f"{split_title}Positive class: {positive_class_label}",
            ha="center",
            va="top",
            fontsize=SUBTITLE_SIZE,
            fontweight="normal",
        )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.30),
            borderaxespad=0.0,
        )

        fig.subplots_adjust(left=0.10, right=0.64, bottom=0.14, top=0.86)
        out = self.output_dir / "roc_auc.png"
        save_thesis_figure(fig, out)
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
                metric_name = "1 − Accuracy"
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

        fig, ax = plt.subplots(figsize=(SINGLE_COLUMN[0], max(4.0, top_n * 0.28)))
        y_pos = np.arange(len(names))

        ax.barh(y_pos, imp_m[::-1], xerr=imp_s[::-1],
                color=colors[::-1], edgecolor="white",
                error_kw=dict(ecolor=_MUTED, capsize=3, linewidth=1))
        ax.axvline(0, color=_DANGER, lw=1.2, linestyle="--")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=_FONT_SIZE - 1)
        ax.set_xlabel(f"Increase in {metric_name} after permutation (higher = more important)")
        ax.set_title(f"Permutation Feature Importance — Top {len(names)} Features")
        ax.text(
            0.99,
            0.01,
            f"Baseline {metric_name} = {baseline:.4f}; {n_repeats} repeats",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=_FONT_SIZE - 2,
            color=_MUTED,
        )

        plt.tight_layout()
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Residuals vs Predicted  (heteroscedasticity check)
    # ──────────────────────────────────────────────────────────────────────────
    def plot_residuals_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str = "Reinforcement steel weight (kgf)",
        filename: str = "steel_residuals_vs_predicted.png",
    ) -> Path:
        """
        Scatter of residuals (Real − Previsto) vs predicted values.

        A horizontal band around zero indicates homoscedasticity.
        A funnel shape reveals heteroscedasticity (variance grows with prediction),
        which is important to report in a thesis.
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
        ax.scatter(y_pred, residuals, s=16, alpha=0.5, color=_PRIMARY, edgecolors="none")
        ax.axhline(0, color=_DANGER, lw=1.5, linestyle="--", label="Zero residual")

        # ±1 std band
        std_r = float(residuals.std())
        ax.axhline( std_r, color=_MUTED, lw=1.0, linestyle=":", label=f"+1σ = {std_r:+.1f}")
        ax.axhline(-std_r, color=_MUTED, lw=1.0, linestyle=":", label=f"−1σ = {-std_r:+.1f}")

        unit = label.split("(")[-1].rstrip(")")
        ax.set_xlabel(f"Predicted value [{unit}]")
        ax.set_ylabel(f"Residual (Observed − Predicted) [{unit}]")
        ax.set_title(f"Residuals vs. Predicted Values — {label}")
        ax.legend(fontsize=_FONT_SIZE - 1)

        plt.tight_layout()
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Q-Q plot dos resíduos  (teste de normalidade)
    # ──────────────────────────────────────────────────────────────────────────
    def plot_qq_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str = "Reinforcement steel weight (kgf)",
        filename: str = "steel_qq_residuals.png",
    ) -> Optional[Path]:
        """
        Quantile-Quantile plot of residuals against a Normal distribution.

        Points on the diagonal indicate normality.  Deviations at the tails
        reveal heavy-tailed or skewed error distributions.
        Also runs the Shapiro-Wilk test (on a random subsample ≤ 5000)
        and annotates the p-value on the plot.
        """
        try:
            from scipy import stats as _stats
        except ImportError:
            print("[NNDiagnostics] Q-Q plot: scipy not available — skipping.")
            return None

        y_true    = np.asarray(y_true, dtype=float).ravel()
        y_pred    = np.asarray(y_pred, dtype=float).ravel()
        residuals = y_true - y_pred

        fig, ax = plt.subplots(figsize=SQUARE)

        # probplot returns (osm, osr), (slope, intercept, r)
        (osm, osr), (slope, intercept, r) = _stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, s=14, alpha=0.55, color=_PRIMARY, edgecolors="none",
                   label="Observed quantiles")
        ax.plot(osm, slope * np.asarray(osm) + intercept,
                color=_DANGER, lw=1.8, linestyle="--", label="Normal reference line")

        # Shapiro-Wilk on subsample (max 5000 — SW limitation)
        rng   = np.random.default_rng(42)
        samp  = residuals if len(residuals) <= 5000 else residuals[rng.choice(len(residuals), 5000, replace=False)]
        sw_stat, sw_p = _stats.shapiro(samp)
        normality_str = "Consistent with normality (p ≥ 0.05)" if sw_p >= 0.05 else "Non-normal (p < 0.05)"
        ax.annotate(
            f"Shapiro-Wilk: W={sw_stat:.4f},  p={sw_p:.4f}\n→ {normality_str}",
            xy=(0.05, 0.93), xycoords="axes fraction",
            fontsize=_FONT_SIZE - 1,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_BG, edgecolor=_MUTED, alpha=0.9),
        )

        ax.set_xlabel("Theoretical Normal Quantiles")
        ax.set_ylabel("Observed Residual Quantiles")
        ax.set_title(f"Normal Q–Q Plot of Residuals — {label}")
        ax.legend(fontsize=_FONT_SIZE - 1)
        ax.set_aspect("equal")

        plt.tight_layout()
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Partial Dependence Plot  (top-N features)
    # ──────────────────────────────────────────────────────────────────────────
    def plot_pdp(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_test: np.ndarray,
        feature_names: List[str],
        y_test: Optional[np.ndarray] = None,
        top_n: int = 3,
        n_grid: int = 60,
        filename: str = "pdp_top_features.png",
    ) -> Optional[Path]:
        """
        1-D Partial Dependence Plot for the top-N most important features.

        For each selected feature, sweeps its value across [min, max] of the
        test set while retaining the observed values of all other features,
        then plots the mean model prediction.

        Parameters
        ----------
        predict_fn   : callable  X (n, f) → predictions (n,)  in real scale.
        X_test       : scaled test features  (n_samples, n_features).
        feature_names: names aligned with columns of X_test.
        y_test       : ground-truth targets (used only to overlay scatter, optional).
        top_n        : number of top features to plot.
        n_grid       : number of evenly spaced grid points per feature.
        filename     : output PNG file name.
        """
        X_test = np.asarray(X_test, dtype=float)
        n_feat = X_test.shape[1]

        # ── Rank features by 1-repeat permutation importance (MAE increase) ──
        baseline_preds = predict_fn(X_test)
        baseline_mae   = float(np.mean(np.abs(baseline_preds - baseline_preds.mean())))

        rng = np.random.default_rng(42)
        imp = np.zeros(n_feat, dtype=float)
        if y_test is not None:
            y_ref = np.asarray(y_test, dtype=float)
            baseline_mae = float(np.mean(np.abs(baseline_preds - y_ref)))
            for f in range(n_feat):
                Xp        = X_test.copy()
                Xp[:, f]  = rng.permutation(Xp[:, f])
                imp[f]    = float(np.mean(np.abs(predict_fn(Xp) - y_ref))) - baseline_mae
        else:
            # Without labels: use variance of predictions as proxy
            for f in range(n_feat):
                Xp        = X_test.copy()
                Xp[:, f]  = rng.permutation(Xp[:, f])
                imp[f]    = float(np.var(predict_fn(Xp)))

        top_indices = np.argsort(imp)[::-1][:top_n]

        # ── Build grid and compute partial dependence ─────────────────────────
        fig, axes = plt.subplots(1, top_n, figsize=(FULL_WIDTH[0], 3.2), sharey=False)
        if top_n == 1:
            axes = [axes]

        for ax, feat_idx in zip(axes, top_indices):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
            grid_vals = np.linspace(X_test[:, feat_idx].min(), X_test[:, feat_idx].max(), n_grid)

            pdp_mean = np.zeros(n_grid, dtype=float)
            pdp_std  = np.zeros(n_grid, dtype=float)
            for g_i, gv in enumerate(grid_vals):
                X_grid            = X_test.copy()
                X_grid[:, feat_idx] = gv
                preds             = predict_fn(X_grid)
                pdp_mean[g_i]     = float(preds.mean())
                pdp_std[g_i]      = float(preds.std())

            ax.plot(grid_vals, pdp_mean, color=_PRIMARY, lw=2.2, label="Mean partial dependence")
            ax.fill_between(
                grid_vals,
                pdp_mean - pdp_std,
                pdp_mean + pdp_std,
                alpha=0.15, color=_PRIMARY, label="±1 SD of predictions",
            )

            # Overlay rug of actual feature values in test set
            ax.plot(
                X_test[:, feat_idx],
                np.full(len(X_test), pdp_mean.min() - (pdp_mean.max() - pdp_mean.min()) * 0.05),
                "|", color=_MUTED, alpha=0.4, markersize=5,
            )

            ax.set_xlabel(feat_name, fontsize=_FONT_SIZE - 1)
            ax.set_ylabel("Mean predicted reinforcement steel weight (kgf)" if ax is axes[0] else "")
            ax.set_title(f"PDP: {feat_name}", fontsize=_FONT_SIZE)
            ax.legend(fontsize=_FONT_SIZE - 2)

        fig.suptitle(f"Partial Dependence Plots — Top {top_n} Features")
        add_panel_labels(axes)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 9. Speedup comparison  (surrogate vs TQS)
    # ──────────────────────────────────────────────────────────────────────────
    def plot_speedup_comparison(self) -> Optional[Path]:
        """
        Bar chart comparing surrogate inference time vs TQS execution time.

        Reads ``metrics/summary.json`` for:
          - ``surrogate_inference_ms_per_sample``  (ms/sample)
          - ``tqs_phase_times_sec.execution``       (s/sample, last recorded)

        Returns None if either value is missing.
        """
        summary_path = self.exp_dir / "metrics" / "summary.json"
        if not summary_path.exists():
            print("[NNDiagnostics] Speedup: summary.json not found — skipping.")
            return None

        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as exc:
            print(f"[NNDiagnostics] Speedup: failed to read summary.json — {exc}")
            return None

        surrogate_ms = summary.get("surrogate_inference_ms_per_sample")
        tqs_sec      = (summary.get("tqs_phase_times_sec") or {}).get("execution")

        if surrogate_ms is None or tqs_sec is None:
            print("[NNDiagnostics] Speedup: timing data incomplete — skipping.")
            return None

        tqs_ms  = tqs_sec * 1000
        speedup = tqs_ms / surrogate_ms

        labels = ["Surrogate model (DNN)", "TQS structural analysis"]
        values = [surrogate_ms, tqs_ms]
        colors = [_ACCENT, _DANGER]

        fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.5)

        # Annotate bar values
        for bar, val in zip(bars, values):
            unit = "ms" if val < 1000 else "s"
            disp = val if val < 1000 else val / 1000
            ax.text(
                bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{disp:.1f} {unit}",
                va="center", ha="left", fontsize=_FONT_SIZE,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Time per sample (ms, log scale)")
        ax.set_title("Computational Performance: Surrogate Model vs. TQS")
        ax.text(
            0.99,
            0.08,
            f"Speedup: ×{speedup:,.0f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=_FONT_SIZE,
            fontweight="semibold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        plt.tight_layout()
        out = self.output_dir / "speedup_comparison.png"
        save_thesis_figure(fig, out)
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

        fig, ax = plt.subplots(figsize=FULL_WIDTH)
        palette = [_PRIMARY, _SECONDARY, _ACCENT, _MUTED, "#333333"]
        line_styles = ["-", "--", "-.", ":"]
        for idx, (layer, vals) in enumerate(layer_norms.items()):
            ep, nrm = zip(*vals)
            ax.semilogy(ep, nrm, lw=1.4, label=layer,
                        color=palette[idx % len(palette)],
                        linestyle=line_styles[(idx // len(palette)) % len(line_styles)])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient L2 Norm (log scale)")
        ax.set_title("Layerwise Gradient Norms during Training")
        ax.legend(fontsize=max(7, _FONT_SIZE - 3), ncol=2, loc="upper right")

        plt.tight_layout()
        out = self.output_dir / "gradient_norms.png"
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Salvo: {out}")
        return out


    # ──────────────────────────────────────────────────────────────────────────
    # Data quality and coverage diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def plot_feature_correlation_heatmap(
        self,
        X: np.ndarray,
        feature_names: Sequence[str],
        y: Optional[np.ndarray] = None,
        filename: str = "feature_correlation_heatmap.png",
    ) -> Optional[Path]:
        """
        Pearson correlation matrix of features + optional target column.

        Reveals redundant features (|r| > 0.9) and features with low
        correlation to the target (candidates for removal).
        """
        try:
            import pandas as pd
        except ImportError:
            print("[NNDiagnostics] pandas not available — heatmap skipped.")
            return None

        _apply_theme()
        df = pd.DataFrame(X, columns=list(feature_names))
        if y is not None:
            df["Steel weight [kgf]"] = y

        corr = df.corr()
        n = len(corr)
        cell = max(0.32, min(0.52, 14.0 / n))
        fig, ax = plt.subplots(figsize=(n * cell + 2, n * cell + 1))

        if _SEABORN:
            sns.heatmap(
                corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.15,
                annot=(n <= 18), fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"shrink": 0.65},
            )
        else:
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax, shrink=0.65)
            ax.set_xticks(range(n))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels(corr.index, fontsize=7)

        ax.set_title(
            "Feature Correlation Matrix" + (" (including target)" if y is not None else ""),
            fontsize=12, pad=14,
        )
        plt.tight_layout()
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Saved: {out}")
        return out

    def plot_error_histogram(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str = "Reinforcement steel weight (kgf)",
        filename: str = "error_histogram.png",
    ) -> Optional[Path]:
        """
        Residual histogram with normal-distribution fit + ECDF panel.

        Shows bias (mean ≠ 0), spread (σ), and % of predictions within ±5% of
        the true value — a quick sanity check on model calibration.
        """
        _apply_theme()
        errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        mu, sigma = float(errors.mean()), float(errors.std())
        mae = float(np.abs(errors).mean())
        pct_5pct = float(np.mean(np.abs(errors) <= 0.05 * np.abs(y_true)) * 100)

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
        fig.suptitle(f"Prediction Error Distribution — {label}")
        fig.text(
            0.5,
            0.925,
            f"μ = {mu:+.1f} kgf · σ = {sigma:.1f} kgf · MAE = {mae:.1f} kgf · "
            f"|relative error| ≤ 5%: {pct_5pct:.1f}%",
            ha="center",
            va="top",
            fontsize=SUBTITLE_SIZE,
            fontweight="normal",
        )

        # Panel 1 — histogram + normal overlay
        ax = axes[0]
        ax.hist(errors, bins=40, density=True, color=_PRIMARY, alpha=0.72,
                edgecolor="white", label="Residuals")
        x_fit = np.linspace(errors.min(), errors.max(), 300)
        try:
            from scipy.stats import norm as _norm
            ax.plot(x_fit, _norm.pdf(x_fit, mu, sigma), color=_DANGER, lw=2,
                    label="Normal fit")
        except ImportError:
            y_fit = ((1 / (sigma * np.sqrt(2 * np.pi)))
                     * np.exp(-0.5 * ((x_fit - mu) / sigma) ** 2))
            ax.plot(x_fit, y_fit, color=_DANGER, lw=2, label="Normal fit")
        # Finish autoscaling with the bars and fitted density before drawing
        # reference lines. This prevents the line caps from changing the range.
        data_hist_top = ax.get_ylim()[1]
        final_hist_ylim = (0.0, data_hist_top * 1.36)
        ax.set_ylim(final_hist_ylim)
        reference_line_top = min(0.94, data_hist_top * 1.02 / final_hist_ylim[1])
        mean_is_distinct = _mean_is_visually_distinct_from_zero(errors, 40)
        zero_label = (
            "Zero residual"
            if mean_is_distinct
            else "Zero residual (mean ≈ 0)"
        )
        ax.axvline(
            0, color=_DANGER, lw=1.5, linestyle="--", label=zero_label,
            ymin=0.0, ymax=reference_line_top, clip_on=True, dash_capstyle="butt",
        )
        if mean_is_distinct:
            ax.axvline(
                mu, color=_DANGER, lw=1.5, linestyle=":", label="Mean residual",
                ymin=0.0, ymax=reference_line_top, clip_on=True,
                dash_capstyle="butt",
            )
        unit = label.split("(")[-1].rstrip(")")
        ax.set_xlabel(f"Residual (Observed − Predicted) [{unit}]")
        ax.set_ylabel("Density")
        ax.set_title("Residual Histogram")
        ax.legend(
            loc="upper center", ncol=2, fontsize=LEGEND_SIZE,
            framealpha=1.0, facecolor="white", edgecolor=COLORS["light_gray"],
        )

        # Panel 2 — ECDF
        ax2 = axes[1]
        sorted_err = np.sort(errors)
        ecdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax2.step(sorted_err, ecdf, where="post", color=_PRIMARY, lw=2, label="ECDF")
        pct_neg = float((errors < 0).mean() * 100)
        ax2.set_ylim(-0.02, 1.02)
        ax2.axvline(
            0, ymin=0.0, ymax=0.96, color=_DANGER, lw=1.5,
            linestyle="--", clip_on=True,
            dash_capstyle="butt",
        )
        ax2.axhline(
            pct_neg / 100,
            color=_DANGER,
            lw=1.2,
            linestyle=":",
        )
        ax2.text(
            0.015,
            pct_neg / 100 + 0.015,
            f"{pct_neg:.0f}% underpredicted",
            transform=ax2.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=_DANGER,
        )
        ax2.text(
            0,
            0.025,
            "Zero residual",
            transform=ax2.get_xaxis_transform(),
            ha="right",
            va="bottom",
            rotation=90,
            fontsize=ANNOTATION_SIZE,
            color=_DANGER,
        )
        ax2.set_xlabel(f"Residual [{unit}]")
        ax2.set_ylabel("ECDF")
        ax2.set_title("Empirical Cumulative Distribution")

        add_panel_labels(axes)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.76, wspace=0.30)
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Saved: {out}")
        return out

    def plot_coverage_pca(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        filename: str = "coverage_pca.png",
    ) -> Optional[Path]:
        """
        PCA 2-component scatter to visualise design-space coverage.

        Colours each point by steel (kgf) so you can see if the network
        is interpolating within the training cloud or extrapolating.
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("[NNDiagnostics] sklearn not available — PCA coverage skipped.")
            return None

        _apply_theme()
        X_all = np.vstack([X_train, X_test])
        pca   = PCA(n_components=2, random_state=42)
        Z     = pca.fit_transform(X_all)
        Z_tr, Z_te = Z[:len(X_train)], Z[len(X_train):]
        ev = pca.explained_variance_ratio_ * 100

        fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL)
        fig.suptitle("Design-Space Coverage — Principal Component Analysis")
        fig.text(
            0.5,
            0.91,
            f"Explained variance: PC1 = {ev[0]:.1f}% · PC2 = {ev[1]:.1f}%",
            ha="center",
            va="top",
            fontsize=SUBTITLE_SIZE,
            fontweight="normal",
        )
        color_limits = None
        available_targets = [np.asarray(values) for values in (y_train, y_test) if values is not None]
        if available_targets:
            targets = np.concatenate(available_targets)
            color_limits = (float(targets.min()), float(targets.max()))
        scatter_mappable = None
        for ax, (Z_s, y_s, lbl) in zip(
            axes,
            [
                (Z_tr, y_train, f"Training set ({len(X_train)} samples)"),
                (Z_te, y_test,  f"Test set ({len(X_test)} samples)"),
            ],
        ):
            if y_s is not None:
                sc = ax.scatter(
                    Z_s[:, 0], Z_s[:, 1], c=y_s,
                    cmap=SEQUENTIAL_CMAP, alpha=0.68, s=18, linewidths=0,
                    vmin=color_limits[0] if color_limits else None,
                    vmax=color_limits[1] if color_limits else None,
                )
                scatter_mappable = sc
            else:
                ax.scatter(Z_s[:, 0], Z_s[:, 1], color=_PRIMARY,
                           alpha=0.5, s=18, linewidths=0)
            ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
            ax.set_title(lbl)

        add_panel_labels(axes)
        fig.subplots_adjust(left=0.08, right=0.86, bottom=0.17, top=0.76, wspace=0.24)
        if scatter_mappable is not None:
            colorbar_axis = fig.add_axes([0.89, 0.20, 0.018, 0.52])
            fig.colorbar(
                scatter_mappable,
                cax=colorbar_axis,
                label="Reinforcement steel weight (kgf)",
            )
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Saved: {out}")
        return out

    def plot_mahalanobis_outliers(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        filename: str = "mahalanobis_outliers.png",
    ) -> Optional[Path]:
        """
        Mahalanobis distance of test points from the training distribution.

        Points beyond the 97.5% chi-squared threshold are out-of-distribution
        and the model is likely to extrapolate poorly for them.
        """
        _apply_theme()
        try:
            mu     = X_train.mean(axis=0)
            cov    = np.cov(X_train, rowvar=False)
            cov_r  = cov + 1e-6 * np.eye(cov.shape[0])   # Tikhonov regularisation
            VI     = np.linalg.inv(cov_r)

            def _maha(X: np.ndarray) -> np.ndarray:
                diff = X - mu
                return np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff))

            d_test  = _maha(X_test)
            rng     = np.random.default_rng(42)
            idx_s   = rng.choice(len(X_train), min(500, len(X_train)), replace=False)
            d_train = _maha(X_train[idx_s])
        except Exception as exc:
            print(f"[NNDiagnostics] Mahalanobis failed: {exc}")
            return None

        try:
            from scipy.stats import chi2 as _chi2
            threshold = float(np.sqrt(_chi2.ppf(0.975, df=X_train.shape[1])))
        except ImportError:
            threshold = float(np.sqrt(X_train.shape[1]) * 1.5)

        pct_out = float((d_test > threshold).mean() * 100)

        fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
        ax.hist(d_train, bins=50, alpha=0.65, color=_PRIMARY,
                label=f"Training sample (n={len(d_train)})")
        ax.hist(d_test,  bins=50, alpha=0.65, color=_DANGER,
                label=f"Test set (n={len(d_test)})")
        ax.axvline(threshold, color="k", lw=2, linestyle="--",
                   label=f"97.5% χ² threshold = {threshold:.1f}")
        ax.set_xlabel("Mahalanobis distance from the training distribution")
        ax.set_ylabel("Count")
        ax.set_title("Mahalanobis Distance — Test vs. Training Distribution")
        ax.text(
            0.98,
            0.95,
            f"Outside 97.5% region: {pct_out:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_FONT_SIZE - 1,
            color=_MUTED,
        )
        ax.legend()
        plt.tight_layout()
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Saved: {out}")
        return out

    def plot_knn_coverage(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        k: int = 5,
        filename: str = "knn_coverage.png",
    ) -> Optional[Path]:
        """
        KNN-distance coverage plot.

        Compares the mean distance to the k nearest training neighbours for
        each test point against the self-coverage of the training set.
        Red regions in the PCA panel indicate zones poorly covered by training.
        """
        try:
            from sklearn.neighbors import NearestNeighbors
            from sklearn.decomposition import PCA
        except ImportError:
            print("[NNDiagnostics] sklearn not available — KNN coverage skipped.")
            return None

        _apply_theme()
        k_actual = min(k, len(X_train) - 1)
        nbrs    = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree").fit(X_train)
        dist_te, _ = nbrs.kneighbors(X_test)
        dist_tr, _ = nbrs.kneighbors(X_train)
        mean_te = dist_te.mean(axis=1)
        mean_tr = dist_tr.mean(axis=1)

        fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL)
        fig.suptitle(
            f"KNN Coverage (k = {k_actual}) — Standardized Feature Space",
            fontsize=12,
        )

        ax = axes[0]
        ax.hist(mean_tr, bins=40, alpha=0.65, color=_PRIMARY,
                label=f"Training → training (n={len(mean_tr)})")
        ax.hist(mean_te, bins=40, alpha=0.65, color=_DANGER,
                label=f"Test → training (n={len(mean_te)})")
        ax.set_xlabel(f"Mean distance to the {k_actual} nearest training neighbors")
        ax.set_ylabel("Count")
        ax.set_title("KNN Distance Distribution")
        ax.legend(fontsize=9)

        ax2 = axes[1]
        pca  = PCA(n_components=2, random_state=42).fit(X_train)
        Z_te = pca.transform(X_test)
        sc   = ax2.scatter(Z_te[:, 0], Z_te[:, 1], c=mean_te,
                           cmap=SEQUENTIAL_CMAP, s=28, alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax2, label=f"Mean distance to {k_actual} nearest training neighbors")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("Coverage by Region")

        add_panel_labels(axes)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[NNDiagnostics] Saved: {out}")
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Optimization diagnostics  (used by run_optimization.py)
# ══════════════════════════════════════════════════════════════════════════════

class OptimizationDiagnosticsPlotter:
    """
    Generates and saves diagnostic plots for a completed optimization run.

    Parameters
    ----------
    output_dir : Path
        Where to save plots.
    log_path : Path
        Path to ``optimization_log.json`` written by GeneticOptimizer.
    """

    def __init__(self, output_dir: Path, log_path: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path   = Path(log_path)
        _apply_theme()

    def _load_log(self) -> Optional[List[dict]]:
        if not self.log_path.exists():
            print(f"[OptDiagnostics] Log not found: {self.log_path}")
            return None
        try:
            with open(self.log_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"[OptDiagnostics] Failed to read log: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Convergence plot
    # ──────────────────────────────────────────────────────────────────────────
    def plot_convergence(self, filename: str = "optimization_convergence.png") -> Optional[Path]:
        """
        Dual-axis plot: best cost (R$) and best steel (kgf) per generation.

        Improvement markers show which generations produced a new best solution.
        """
        logs = self._load_log()
        if not logs:
            return None

        iters     = [r["iteration"]    for r in logs]
        costs     = [r["cost"]         for r in logs]
        steels    = [r.get("steel")    for r in logs]
        improved  = [r.get("improved", False) for r in logs]

        # Running best (monotone decreasing)
        best_costs = []
        cur_best = float("inf")
        for c in costs:
            cur_best = min(cur_best, c)
            best_costs.append(cur_best)

        fig, ax1 = plt.subplots(figsize=FULL_WIDTH)
        ax2 = ax1.twinx()

        ax1.plot(iters, best_costs, color=_PRIMARY, lw=2.2, label="Best-so-far objective cost (R$)")
        ax1.fill_between(iters, best_costs, alpha=0.08, color=_PRIMARY)

        # Mark improvement generations
        imp_x = [iters[i] for i, v in enumerate(improved) if v]
        imp_y = [best_costs[i] for i, v in enumerate(improved) if v]
        ax1.scatter(imp_x, imp_y, color=_ACCENT, zorder=5, s=40,
                    label="New incumbent solution", marker="^")

        if any(s is not None for s in steels):
            steel_clean = [s if s is not None else float("nan") for s in steels]
            ax2.plot(iters, steel_clean, color=_SECONDARY, lw=1.6,
                     linestyle="--", label="Generation-best reinforcement steel weight (kgf)")
            ax2.set_ylabel("Generation-best reinforcement steel weight (kgf)", color=_SECONDARY)
            ax2.tick_params(axis="y", labelcolor=_SECONDARY)
            ax2.legend(loc="upper right", fontsize=_FONT_SIZE - 1)

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Best-so-far objective cost (R$)", color=_PRIMARY)
        ax1.tick_params(axis="y", labelcolor=_PRIMARY)
        ax1.set_title("Genetic Algorithm Convergence")
        ax1.text(
            0.99,
            0.04,
            f"Final best cost: R$ {best_costs[-1]:,.2f} · {len(iters)} generations",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=_FONT_SIZE - 1,
            color=_MUTED,
        )
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc="upper center", fontsize=_FONT_SIZE - 1)

        plt.tight_layout()
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[OptDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Seed vs Optimal comparison
    # ──────────────────────────────────────────────────────────────────────────
    def plot_seed_vs_optimal(
        self,
        seed_metrics: dict,
        optimal_metrics: dict,
        filename: str = "seed_vs_optimal.png",
    ) -> Optional[Path]:
        """
        Grouped bar chart comparing seed and optimal designs on cost, steel,
        concrete and formwork area.

        Parameters
        ----------
        seed_metrics : dict
            Keys: ``cost`` (R$), ``steel`` (kgf), ``concrete`` (m³), ``form_area`` (m²).
        optimal_metrics : dict
            Same keys as seed_metrics.
        """
        keys    = ["Total cost (R$)", "Reinforcement steel (kgf)", "Concrete (m³)", "Column formwork (m²)"]
        s_vals  = [seed_metrics.get("cost", 0),
                   seed_metrics.get("steel", 0),
                   seed_metrics.get("concrete", 0),
                   seed_metrics.get("form_area", 0)]
        o_vals  = [optimal_metrics.get("cost", 0),
                   optimal_metrics.get("steel", 0),
                   optimal_metrics.get("concrete", 0),
                   optimal_metrics.get("form_area", 0)]

        x      = np.arange(len(keys))
        width  = 0.35

        fig, ax = plt.subplots(figsize=FULL_WIDTH)
        bars_s = ax.bar(x - width / 2, s_vals, width, label="Baseline design",
                        color=_MUTED,    edgecolor="white")
        bars_o = ax.bar(x + width / 2, o_vals, width, label="Optimized design",
                        color=_PRIMARY,  edgecolor="white")

        # Annotate bars with value and % reduction
        for bar_s, bar_o, sv, ov in zip(bars_s, bars_o, s_vals, o_vals):
            ax.text(bar_s.get_x() + bar_s.get_width() / 2, bar_s.get_height() * 1.01,
                    f"{sv:,.1f}", ha="center", va="bottom", fontsize=_FONT_SIZE - 2)
            pct = (ov - sv) / sv * 100 if sv else 0
            sign = "▼" if pct < 0 else "▲"
            color = _ACCENT if pct < 0 else _DANGER
            ax.text(bar_o.get_x() + bar_o.get_width() / 2, bar_o.get_height() * 1.01,
                    f"{ov:,.1f}\n{sign}{abs(pct):.1f}%",
                    ha="center", va="bottom", fontsize=_FONT_SIZE - 2, color=color,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=_FONT_SIZE)
        ax.set_title(
            "Baseline vs. Optimized Design",
            fontsize=_FONT_SIZE + 1, fontweight="bold",
        )
        ax.legend(fontsize=_FONT_SIZE - 1)
        ax.set_ylabel("Value (mixed units)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Secondary note
        fig.text(0.5, 0.01,
                 "Note: different physical quantities share one axis; use only as a provisional comparison.",
                 ha="center", fontsize=_FONT_SIZE - 2, color=_MUTED)

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[OptDiagnostics] Salvo: {out}")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Surrogate vs TQS verification
    # ──────────────────────────────────────────────────────────────────────────
    def plot_surrogate_vs_tqs(
        self,
        surrogate_steel: float,
        tqs_steel: float,
        surrogate_concrete: Optional[float] = None,
        tqs_concrete: Optional[float] = None,
        filename: str = "surrogate_vs_tqs_verification.png",
    ) -> Path:
        """
        Bar chart comparing surrogate predictions vs TQS ground-truth for the
        optimal design point.

        Call this after running TQS manually on ``solucao_otima.csv``.

        Parameters
        ----------
        surrogate_steel  : surrogate prediction for steel (kgf).
        tqs_steel        : TQS-computed steel (kgf).
        surrogate_concrete: surrogate prediction for concrete (m³), optional.
        tqs_concrete     : TQS-computed concrete (m³), optional.
        """
        labels = ["Surrogate model (DNN)", "TQS reference"]

        has_concrete = surrogate_concrete is not None and tqs_concrete is not None
        n_groups = 2 if has_concrete else 1
        fig, axes = plt.subplots(
            1,
            n_groups,
            figsize=TWO_PANEL if n_groups == 2 else SINGLE_COLUMN,
        )
        if n_groups == 1:
            axes = [axes]

        group_data = [("Reinforcement steel weight (kgf)", surrogate_steel, tqs_steel)]
        if has_concrete:
            group_data.append(("Concrete volume (m³)", surrogate_concrete, tqs_concrete))

        for ax, (title, s_val, t_val) in zip(axes, group_data):
            bars = ax.bar(labels, [s_val, t_val],
                          color=[_PRIMARY, _DANGER], edgecolor="white", width=0.5)
            err_pct = abs(s_val - t_val) / t_val * 100 if t_val else 0
            for bar, val in zip(bars, [s_val, t_val]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{val:,.1f}", ha="center", va="bottom", fontsize=_FONT_SIZE)
            ax.set_title(title)
            ax.text(
                0.5,
                0.95,
                f"Relative error: {err_pct:.1f}%",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=_FONT_SIZE - 1,
            )
            ax.set_ylabel(title)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Optimized Design Verification: Surrogate Model vs. TQS")
        if n_groups > 1:
            add_panel_labels(axes)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        out = self.output_dir / filename
        save_thesis_figure(fig, out)
        plt.close("all")
        print(f"[OptDiagnostics] Salvo: {out}")
        return out


def run_optimization_diagnostics(
    output_dir: Path,
    log_path: Path,
    seed_metrics: dict,
    optimal_metrics: dict,
) -> None:
    """
    Run all optimization diagnostic plots.

    Parameters
    ----------
    output_dir      : Directory to save plots (e.g. ``outputs/results/plots/``).
    log_path        : Path to ``optimization_log.json``.
    seed_metrics    : dict with keys ``cost``, ``steel``, ``concrete``, ``form_area`` for the seed.
    optimal_metrics : Same keys for the final optimal design.
    """
    plotter = OptimizationDiagnosticsPlotter(output_dir, log_path)
    plotter.plot_convergence()
    plotter.plot_seed_vs_optimal(seed_metrics, optimal_metrics)
    print(f"[OptDiagnostics] Diagnósticos de otimização salvos em: {output_dir}")


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
    plot_data_store = PlotDataStore(experiment_dir)

    # Persist final evaluation values before rendering. Plot styling can then
    # be changed later without loading the model or repeating inference.
    if y_test_steel is not None and y_pred_steel is not None:
        try:
            saved_test_indices = None
            saved_arrays_path = Path(experiment_dir) / "arrays.npz"
            if saved_arrays_path.exists():
                with np.load(saved_arrays_path, allow_pickle=False) as saved_arrays:
                    candidate = (
                        saved_arrays["test_indices"]
                        if "test_indices" in saved_arrays.files
                        else None
                    )
                    if candidate is not None and len(candidate) == len(y_test_steel):
                        saved_test_indices = candidate
            plot_data_store.save_regression_test(
                y_test_steel,
                y_pred_steel,
                sample_indices=saved_test_indices,
            )
        except Exception as exc:
            print(f"[NNDiagnostics] Could not persist regression figure data: {exc}")

    # 1. Learning curves  (always, reads from disk)
    plotter.plot_learning_curves()
    plotter.plot_gradient_norms()
    plotter.plot_speedup_comparison()

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
                label="Reinforcement steel weight (kgf)", filename_prefix="steel"
            )
            plotter.plot_residuals_vs_predicted(
                y_test_steel, y_pred_steel, label="Reinforcement steel weight (kgf)"
            )
            plotter.plot_qq_residuals(
                y_test_steel, y_pred_steel, label="Reinforcement steel weight (kgf)"
            )
        except Exception as exc:
            print(f"[NNDiagnostics] Scatter plot failed: {exc}")

    # 2b. Permutation Feature Importance + PDP for regressor
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
            plotter.plot_pdp(
                predict_fn=_predict_steel_real,
                X_test=X_test,
                feature_names=feature_names,
                y_test=y_test_steel,
                top_n=3,
            )
        except Exception as exc:
            print(f"[NNDiagnostics] PFI/PDP for regressor failed: {exc}")

    # 2c. SHAP summary  (uses existing implementation in ResultsPlotter)
    if nn_manager is not None and X_test is not None:
        try:
            from visualization.results_plotter import ResultsPlotter as _RP
            _rp = _RP(plotter.output_dir)
            _rp.plot_shap_summary(
                model=nn_manager.model,
                X_background=X_test,
                X_explain=X_test,
                feature_names=feature_names,
            )
        except Exception as exc:
            print(f"[NNDiagnostics] SHAP skipped: {exc}")

    # 3 & 4. Confusion matrix + ROC  (needs classifier + test labels)
    _clf_X = X_test_clf if X_test_clf is not None else X_test
    if classifier is not None and _clf_X is not None and y_test_valid is not None:
        try:
            if not hasattr(classifier, "predict_proba"):
                raise RuntimeError(
                    "Validity classifier must expose probabilities for the "
                    "calibrated invalidity rule."
                )
            classes = list(getattr(classifier, "classes_", []))
            idx_invalid = invalid_probability_index(classes)
            invalid_probability = classifier.predict_proba(_clf_X)[:, idx_invalid]

            threshold_path = Path(experiment_dir) / "metrics" / "validity_threshold.json"
            if not threshold_path.exists():
                raise FileNotFoundError(
                    "Calibrated validity threshold is required for final classifier plots."
                )
            with open(threshold_path, encoding="utf-8") as stream:
                invalid_threshold = float(json.load(stream)["threshold"])

            y_pred_labels = validity_labels_from_invalid_probability(
                invalid_probability,
                invalid_threshold,
            )
            plot_data_store.save_classifier_test(
                y_test_valid,
                invalid_probability,
                y_pred_labels,
                invalid_threshold=invalid_threshold,
                X_test=_clf_X,
                feature_names=feature_names,
            )
            plotter.plot_confusion_matrix(
                y_test_valid,
                y_pred_labels,
                invalid_threshold=invalid_threshold,
            )

            invalid_event = (np.asarray(y_test_valid, dtype=int) == INVALID_LABEL).astype(int)
            plotter.plot_roc_auc(
                y_true=invalid_event,
                y_score=invalid_probability,
                threshold_to_mark=invalid_threshold,
                split_label="test",
                positive_class_label="Infeasible",
            )

            # ── Permutation Feature Importance for classifier ───────────────
            def _predict_invalid_probability(X: np.ndarray) -> np.ndarray:
                proba = classifier.predict_proba(X)
                return proba[:, idx_invalid]

            def _invalidity_error(y_true: np.ndarray, probability: np.ndarray) -> float:
                predicted = (probability >= invalid_threshold).astype(int)
                return 1.0 - accuracy_score(y_true, predicted)

            plotter.plot_permutation_importance(
                predict_fn=_predict_invalid_probability,
                X_test=_clf_X,
                y_test=invalid_event,
                feature_names=feature_names,
                metric_fn=_invalidity_error,
                metric_name="classification error for the infeasibility rule",
                n_repeats=n_repeats_pfi,
                filename="pfi_validity_classifier.png",
            )
        except Exception as exc:
            print(f"[NNDiagnostics] Classifier plots failed: {exc}")

    elif classifier is None:
        # still try to plot ROC from saved JSON
        plotter.plot_roc_auc()

    # 5. Data quality + coverage diagnostics  (requires arrays.npz)
    arrays_path = experiment_dir / "arrays.npz"
    if arrays_path.exists():
        print("\n[NNDiagnostics] Loading arrays.npz for data-quality diagnostics…")
        try:
            arrs       = np.load(str(arrays_path))
            X_tr_sc    = arrs["X_train_scaled"]
            X_te_sc    = arrs["X_test_scaled"]
            X_tr_raw   = arrs["X_train"]
            y_tr_raw   = arrs["y_train"][:, 0]   # steel column (kgf)
            y_te_raw   = arrs["y_test"][:, 0]

            plotter.plot_feature_correlation_heatmap(
                X_tr_raw, feature_names, y=y_tr_raw
            )
            if y_test_steel is not None and y_pred_steel is not None:
                plotter.plot_error_histogram(y_test_steel, y_pred_steel)
            plotter.plot_coverage_pca(
                X_tr_sc, X_te_sc, y_train=y_tr_raw, y_test=y_te_raw
            )
            plotter.plot_mahalanobis_outliers(X_tr_sc, X_te_sc)
            plotter.plot_knn_coverage(X_tr_sc, X_te_sc)
        except Exception as exc:
            print(f"[NNDiagnostics] Data-quality diagnostics failed: {exc}")
    else:
        print(
            "[NNDiagnostics] arrays.npz not found — coverage/correlation diagnostics skipped.\n"
            "  (arrays are saved automatically on next training run)"
        )

    print(f"[NNDiagnostics] Diagnósticos salvos em: {plotter.output_dir}")


def regenerate_training_figures(
    experiment_dir: Path | str,
    output_dir: Path | str | None = None,
) -> list[Path]:
    """Redraw model-independent figures exclusively from saved run artifacts."""
    experiment_dir = Path(experiment_dir)
    plotter = NNDiagnosticsPlotter(
        experiment_dir,
        Path(output_dir) if output_dir is not None else None,
    )
    generated: list[Path] = []

    def remember(path: Optional[Path]) -> None:
        if path is not None:
            generated.append(Path(path))

    # Per-epoch and runtime series are already persisted as JSON/NDJSON.
    remember(plotter.plot_learning_curves())
    remember(plotter.plot_gradient_norms())
    remember(plotter.plot_speedup_comparison())

    try:
        figure_data = PlotDataStore(experiment_dir).load()
    except FileNotFoundError:
        figure_data = {}

    regression_keys = {
        "regression_y_true_steel_kgf",
        "regression_y_pred_steel_kgf",
    }
    if regression_keys.issubset(figure_data):
        observed = figure_data["regression_y_true_steel_kgf"]
        predicted = figure_data["regression_y_pred_steel_kgf"]
        remember(plotter.plot_scatter_and_residuals(observed, predicted))
        remember(plotter.plot_residuals_vs_predicted(observed, predicted))
        remember(plotter.plot_qq_residuals(observed, predicted))
        remember(plotter.plot_error_histogram(observed, predicted))

    classifier_keys = {
        "classifier_y_true_validity",
        "classifier_y_pred_validity",
        "classifier_invalid_probability",
        "classifier_invalid_threshold",
    }
    if classifier_keys.issubset(figure_data):
        actual_validity = figure_data["classifier_y_true_validity"].astype(int)
        predicted_validity = figure_data["classifier_y_pred_validity"].astype(int)
        invalid_probability = figure_data["classifier_invalid_probability"]
        invalid_threshold = float(
            figure_data["classifier_invalid_threshold"].reshape(-1)[0]
        )
        remember(
            plotter.plot_confusion_matrix(
                actual_validity,
                predicted_validity,
                invalid_threshold=invalid_threshold,
            )
        )
        invalid_event = (actual_validity == INVALID_LABEL).astype(int)
        remember(
            plotter.plot_roc_auc(
                y_true=invalid_event,
                y_score=invalid_probability,
                threshold_to_mark=invalid_threshold,
                split_label="test",
                positive_class_label="Infeasible",
            )
        )
    else:
        remember(plotter.plot_roc_auc())

    feature_names: list[str] = []
    feature_names_path = experiment_dir / "metrics" / "feature_names.json"
    if feature_names_path.exists():
        try:
            feature_names = list(
                json.loads(feature_names_path.read_text(encoding="utf-8"))[
                    "feature_names"
                ]
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            feature_names = []

    arrays_path = experiment_dir / "arrays.npz"
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as arrays:
            X_train_scaled = arrays["X_train_scaled"]
            X_test_scaled = arrays["X_test_scaled"]
            X_train = arrays["X_train"]
            y_train = arrays["y_train"][:, 0]
            y_test = arrays["y_test"][:, 0]
        if feature_names and len(feature_names) == X_train.shape[1]:
            remember(
                plotter.plot_feature_correlation_heatmap(
                    X_train,
                    feature_names,
                    y=y_train,
                )
            )
        remember(
            plotter.plot_coverage_pca(
                X_train_scaled,
                X_test_scaled,
                y_train=y_train,
                y_test=y_test,
            )
        )
        remember(plotter.plot_mahalanobis_outliers(X_train_scaled, X_test_scaled))
        remember(plotter.plot_knn_coverage(X_train_scaled, X_test_scaled))

    print(
        f"[NNDiagnostics] Regenerated {len(generated)} figures from saved data "
        f"in: {plotter.output_dir}"
    )
    return generated


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
    regenerate_training_figures(exp_dir, out_dir)

    # If test CSV is provided, load the full inference stack and plot scatter + PFI
    if args.csv:
        try:
            from joblib import load as jl_load
            from inference import BuildingInference
            from utils.feature_engineer import FeatureEngineer

            print(f"[NNDiagnostics] Carregando artefatos de '{exp_dir.name}'...")
            inf = BuildingInference(exp_dir.name)

            print(f"[NNDiagnostics] Processando CSV de teste '{args.csv}'...")
            steel, concrete, form_area, prob = inf.predict_from_csv(args.csv)
            print(f"  Previsto → Aço: {steel:.1f} kgf  |  Concreto: {concrete:.3f} m³  |  Forma: {form_area:.2f} m²  |  P(inválido): {prob}")

        except Exception as exc:
            print(f"[NNDiagnostics] Falha ao carregar inferência para CSV: {exc}")

    print("[NNDiagnostics] Concluído.")
