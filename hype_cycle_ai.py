"""
hype_cycle_ai.py
----------------
Publication-quality Hype Cycle for a PhD thesis.
Stage labels appear BELOW the x-axis baseline (as in the Gartner reference).
Vertical dividers separate the five phases.

Usage:  python hype_cycle_ai.py
Output: hype_cycle_ai.png  (300 dpi)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.interpolate import CubicSpline

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Curve — cubic spline through control points
# ─────────────────────────────────────────────────────────────────────────────
CTRL_X = np.array([
    0.00, 0.05, 0.13, 0.22, 0.30, 0.36,   # innovation trigger → peak
    0.42, 0.49, 0.56, 0.60,               # sharp descent → trough
    0.65, 0.70, 0.75, 0.80,               # trough → slope of enlightenment
    0.85, 0.91, 0.96, 1.00,               # plateau
])
CTRL_Y = np.array([
    0.00, 0.03, 0.12, 0.55, 0.92, 1.00,
    0.76, 0.28, 0.12, 0.12,
    0.16, 0.24, 0.34, 0.41,
    0.45, 0.46, 0.47, 0.47,
])

cs      = CubicSpline(CTRL_X, CTRL_Y)
x_dense = np.linspace(0, 1, 2000)
y_dense = np.clip(cs(x_dense), 0, None)

def y_curve(x: float) -> float:
    return float(np.clip(cs(x), 0, None))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Technologies
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "lt2":   ("#f5f5f5", "#2c2c2c"),   # < 2 years : near-white fill, dark edge
    "2to5":  ("#a8cfe0", "#2b6a8a"),   # 2–5 years : steel blue
    "5to10": ("#2c5f7a", "#142d3a"),   # 5–10 years: deep teal
}

TECHNOLOGIES = [
    # ( name,                           x,     cat,     (tx,  ty),  ha,      hl    )
    ("Decision\nIntelligence",          0.20,  "2to5", (-20,   -1), "right", False ),
    ("Intelligent\nSimulation",         0.22,  "5to10", (-20,   1), "right", False ),
    ("Composite AI",                    0.25,  "lt2",   (-20,   0), "right", False ),
    ("Domain-Specific\nGenAI Models",   0.28,  "2to5",  (-20, -2), "right", False ),
    ("AI-Ready Data",                   0.34,  "5to10", ( 20,  10), "left",  False ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Phase dividers and stage labels
# ─────────────────────────────────────────────────────────────────────────────
DIVIDERS_X  = [0.26, 0.46, 0.64, 0.84]

_edges      = [0.00] + DIVIDERS_X + [1.00]
STAGE_MID_X = [(_edges[i] + _edges[i + 1]) / 2.0 for i in range(5)]

STAGE_LABELS = [
    "Innovation\nTrigger",
    "Peak of Inflated\nExpectations",
    "Trough of\nDisillusionment",
    "Slope of\nEnlightenment",
    "Plateau of\nProductivity",
]

# Alternating background shades for phase zones (very subtle)
PHASE_BG = ["#f9f9f9", "#f4f4f4", "#f9f9f9", "#f4f4f4", "#f9f9f9"]

LABEL_Y = -0.105   # data-unit offset below the y=0 baseline


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Style — serif typeface, clean academic palette
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif", "Georgia", "Palatino"],
    "font.size":    10,
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(5.9, 4.3), facecolor="white")
ax.set_facecolor("white")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Phase background bands (extend from footer to top of plot)
# ─────────────────────────────────────────────────────────────────────────────
for i, (x0, x1) in enumerate(zip(_edges[:-1], _edges[1:])):
    ax.axvspan(x0, x1, ymin=0, ymax=1, color=PHASE_BG[i], zorder=0, alpha=1.0)

# Slightly darker footer band below baseline
ax.axhspan(-0.28, 0, color="#efefef", zorder=0, alpha=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main curve
# ─────────────────────────────────────────────────────────────────────────────
ax.plot(x_dense, y_dense, color="#1c1c1c", linewidth=2.0,
        solid_capstyle="round", zorder=3)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Horizontal baseline (y=0) + dashed vertical phase dividers
# ─────────────────────────────────────────────────────────────────────────────
ax.axhline(y=0, color="#888888", linewidth=0.9, zorder=2)

for xd in DIVIDERS_X:
    ax.axvline(x=xd, color="#bbbbbb", linewidth=0.75,
               linestyle=(0, (5, 4)),   # long-dash
               zorder=2)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Stage labels — italic, below the baseline
# ─────────────────────────────────────────────────────────────────────────────
for label, x_s in zip(STAGE_LABELS, STAGE_MID_X):
    ax.text(
        x_s, LABEL_Y, label,
        ha="center", va="top",
        fontsize=8.5, color="#555555",
        style="italic", linespacing=1.5,
        zorder=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Technology markers + annotated labels
# ─────────────────────────────────────────────────────────────────────────────
for name, x_t, cat, (tx, ty), ha, highlight in TECHNOLOGIES:
    y_t        = y_curve(x_t)
    face, edge = COLORS[cat]

    ax.scatter(x_t, y_t, s=40, color=face, edgecolors=edge,
               linewidths=1.8, zorder=4)
    ax.annotate(
        name,
        xy=(x_t, y_t), xytext=(tx, ty), textcoords="offset points",
        ha=ha, va="center",
        fontsize=7.8, color="#1c1c1c",
        linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.9),
        zorder=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Axes
# ─────────────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.02, 1.04)
ax.set_ylim(-0.28, 1.10)

ax.set_xlabel("Time", fontsize=10, labelpad=12,
              color="#333333", style="italic")
ax.set_ylabel("Expectations", fontsize=10, labelpad=14,
              color="#333333", style="italic")

ax.set_xticks([])
ax.set_yticks([])

# Only left spine, clipped to data range
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_color("#888888")
ax.spines["left"].set_linewidth(0.9)
ax.spines["left"].set_bounds(0, 1.10)

# Thin separator line between footer and plot area (reinforces baseline)
ax.axhline(y=0, color="#888888", linewidth=0.9, zorder=2)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Legend — thin framed box, upper left
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mlines.Line2D([], [], marker="o", linestyle="none",
                  markerfacecolor="#f5f5f5", markeredgecolor="#2c2c2c",
                  markeredgewidth=1.6, markersize=6, label="< 2 years"),
    mlines.Line2D([], [], marker="o", linestyle="none",
                  markerfacecolor="#a8cfe0", markeredgecolor="#2b6a8a",
                  markeredgewidth=1.6, markersize=6, label="2–5 years"),
    mlines.Line2D([], [], marker="o", linestyle="none",
                  markerfacecolor="#2c5f7a", markeredgecolor="#142d3a",
                  markeredgewidth=1.6, markersize=6, label="5–10 years"),
]
leg = ax.legend(
    handles=legend_items,
    title="Time to Plateau",
    title_fontsize=8.5, fontsize=8.0,
    loc="upper left", bbox_to_anchor=(0.75, 0.97),
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    fancybox=False,
    handletextpad=0.8, labelspacing=0.7,
)
leg.get_title().set_color("#333333")
leg.get_title().set_fontstyle("italic")
leg.get_frame().set_linewidth(0.7)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Save
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout()
OUTPUT_FILE = "hype_cycle_ai.png"
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_FILE}")
plt.show()
