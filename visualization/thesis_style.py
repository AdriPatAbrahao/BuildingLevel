"""Shared publication style for figures used in the doctoral thesis."""

from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase
from typing import Iterable

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


# Semantic thesis palette. Line styles and markers remain part of the encoding
# so that categories are still distinguishable in grayscale printing.
COLORS = {
    "primary": "#1B3A5C",
    "accent": "#8C2F1B",
    "secondary": "#5C7A99",
    # Backward-compatible semantic aliases used by existing plotters.
    "blue": "#1B3A5C",
    "orange": "#8C2F1B",
    "green": "#5C7A99",
    "vermillion": "#8C2F1B",
    "gray": "#6B7280",
    "light_gray": "#D1D5DB",
    "grid": "#D9DEE7",
    "background": "#FFFFFF",
}

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "thesis_navy",
    [COLORS["primary"], "#A9BCCB"],
    N=256,
)

THESIS_DPI = 300
TITLE_SIZE = 14
SUBTITLE_SIZE = 11
PANEL_LABEL_SIZE = 12.5
PANEL_TITLE_SIZE = 12
AXIS_LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
ANNOTATION_SIZE = 9
SINGLE_COLUMN = (6.3, 4.0)
FULL_WIDTH = (7.2, 4.4)
TWO_PANEL = (7.2, 3.5)
SQUARE = (5.4, 5.0)


def apply_thesis_style() -> None:
    """Apply stable Matplotlib defaults suitable for print and PDF export."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": COLORS["background"],
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.30,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "legend.fontsize": LEGEND_SIZE,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.facecolor": "white",
            "legend.edgecolor": COLORS["light_gray"],
            "figure.titlesize": TITLE_SIZE,
            "figure.titleweight": "bold",
            "figure.facecolor": COLORS["background"],
            "figure.dpi": 150,
            "savefig.dpi": THESIS_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "savefig.facecolor": COLORS["background"],
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_labels(axes: object, labels: Iterable[str] | None = None) -> None:
    """Add panel labels without making them part of the axes titles."""
    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    else:
        axes_list = [axes]

    panel_labels = list(labels) if labels is not None else [
        f"({letter})" for letter in ascii_lowercase[: len(axes_list)]
    ]
    for axis, label in zip(axes_list, panel_labels):
        current_title = axis.get_title()
        axis.set_title("")
        if current_title:
            axis.set_title(
                current_title,
                loc="left",
                x=0.12,
                fontsize=PANEL_TITLE_SIZE,
                fontweight="bold",
                pad=10,
            )
        axis.text(
            0.0,
            1.04,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            clip_on=False,
        )


def save_thesis_figure(fig: Figure, output_path: Path | str) -> Path:
    """Save a raster PNG and a vector PDF with the same base file name."""
    requested = Path(output_path)
    png_path = requested.with_suffix(".png")
    pdf_path = requested.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_path, dpi=THESIS_DPI, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.06,
        metadata={"Creator": "BuildingOptimization thesis figure pipeline"},
    )
    return png_path
