from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import visualization.nn_diagnostics as diagnostics

from visualization.thesis_style import (
    COLORS,
    SEQUENTIAL_CMAP,
    THESIS_DPI,
    add_panel_labels,
    apply_thesis_style,
    save_thesis_figure,
)


def test_thesis_style_exports_png_and_vector_pdf(tmp_path: Path) -> None:
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2)
    add_panel_labels(axes)

    png_path = save_thesis_figure(fig, tmp_path / "figure.png")

    assert png_path.exists()
    assert png_path.with_suffix(".pdf").exists()
    assert plt.rcParams["savefig.dpi"] == THESIS_DPI
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.grid.axis"] == "y"
    assert plt.rcParams["font.sans-serif"][:2] == ["Arial", "Helvetica"]
    assert [[text.get_text() for text in axis.texts] for axis in axes] == [
        ["(a)"],
        ["(b)"],
    ]
    assert COLORS["primary"] == "#1B3A5C"
    assert SEQUENTIAL_CMAP.name == "thesis_navy"
    plt.close(fig)


def test_error_histogram_legend_is_opaque_and_reference_lines_are_clipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    real_close = diagnostics.plt.close
    monkeypatch.setattr(diagnostics.plt, "close", lambda *args, **kwargs: None)
    plotter = diagnostics.NNDiagnosticsPlotter(tmp_path, output_dir=tmp_path)
    y_true = np.linspace(100.0, 300.0, 50)
    y_pred = y_true + np.linspace(5.0, 35.0, 50)

    plotter.plot_error_histogram(y_true, y_pred)

    fig = diagnostics.plt.gcf()
    histogram_axis = fig.axes[0]
    vertical_lines = [
        line for line in histogram_axis.lines
        if len(np.unique(np.asarray(line.get_xdata(), dtype=float))) == 1
    ]
    assert len(vertical_lines) == 2
    assert all(float(np.max(line.get_ydata())) <= 0.94 for line in vertical_lines)
    legend = histogram_axis.get_legend()
    assert legend.get_frame().get_alpha() == 1.0
    assert [text.get_text() for text in legend.get_texts()] == [
        "Residuals",
        "Normal fit",
        "Zero residual",
        "Mean residual",
    ]
    assert any("overpredicted" in text.get_text() for text in fig.axes[1].texts)
    assert not any("underpredicted" in text.get_text() for text in fig.axes[1].texts)
    real_close(fig)


def test_error_histogram_combines_mean_and_zero_when_visually_coincident(
    tmp_path: Path,
    monkeypatch,
) -> None:
    real_close = diagnostics.plt.close
    monkeypatch.setattr(diagnostics.plt, "close", lambda *args, **kwargs: None)
    plotter = diagnostics.NNDiagnosticsPlotter(tmp_path, output_dir=tmp_path)
    y_true = np.linspace(100.0, 300.0, 50)
    y_pred = y_true + np.linspace(-15.0, 15.0, 50)

    plotter.plot_error_histogram(y_true, y_pred)

    fig = diagnostics.plt.gcf()
    histogram_axis = fig.axes[0]
    vertical_lines = [
        line for line in histogram_axis.lines
        if len(np.unique(np.asarray(line.get_xdata(), dtype=float))) == 1
    ]
    assert len(vertical_lines) == 1
    legend_labels = [
        text.get_text() for text in histogram_axis.get_legend().get_texts()
    ]
    assert legend_labels == [
        "Residuals",
        "Normal fit",
        "Zero residual (mean ≈ 0)",
    ]
    real_close(fig)


def test_scatter_metrics_and_legend_do_not_cover_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    real_close = diagnostics.plt.close
    monkeypatch.setattr(diagnostics.plt, "close", lambda *args, **kwargs: None)
    plotter = diagnostics.NNDiagnosticsPlotter(tmp_path, output_dir=tmp_path)
    y_true = np.linspace(1000.0, 2000.0, 50)
    y_pred = y_true + np.linspace(-40.0, 40.0, 50)

    plotter.plot_scatter_and_residuals(y_true, y_pred)

    fig = diagnostics.plt.gcf()
    scatter_axis = fig.axes[0]
    metrics_text = next(
        text for text in scatter_axis.texts if text.get_text().startswith("R²")
    )
    assert metrics_text.get_position() == (0.03, 0.97)
    assert metrics_text.get_horizontalalignment() == "left"
    assert metrics_text.get_verticalalignment() == "top"
    assert scatter_axis.get_legend() is None
    assert len(fig.legends) == 1
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "Test samples",
        "1:1 reference line",
        "±10% error band",
    ]
    assert "Predicted (MLP)" in scatter_axis.get_ylabel()
    real_close(fig)
