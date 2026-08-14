import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# -----------------------------
# Hype Cycle base curve
# -----------------------------
x_base = np.array([0.0, 0.8, 1.6, 2.2, 3.0, 4.2, 5.5, 7.0, 8.5, 10.0])
y_base = np.array([0.8, 1.8, 5.8, 9.5, 4.2, 2.0, 3.2, 5.0, 6.1, 6.3])

spline = make_interp_spline(x_base, y_base, k=3)

x = np.linspace(x_base.min(), x_base.max(), 500)
y = spline(x)

def y_on_curve(x_value):
    return float(spline(x_value))

# -----------------------------
# Technology points
# -----------------------------
points = [
    {
        "name": "Composite AI",
        "x": 6.6,
        "color": "white",
        "edge": "#1f4e79",
        "category": "< 2 years",
        "label_offset": (0.15, 0.35)
    },
    {
        "name": "Decision Intelligence",
        "x": 3.7,
        "color": "#08306b",
        "edge": "#08306b",
        "category": "5–10 years",
        "label_offset": (0.15, 0.35)
    },
    {
        "name": "AI-Ready Data",
        "x": 2.4,
        "color": "#08306b",
        "edge": "#08306b",
        "category": "5–10 years",
        "label_offset": (0.15, 0.45)
    },
    {
        "name": "Intelligent Simulation",
        "x": 4.8,
        "color": "#08306b",
        "edge": "#08306b",
        "category": "5–10 years",
        "label_offset": (0.15, 0.35)
    },
    {
        "name": "Domain-Specific\nGenAI Models",
        "x": 1.75,
        "color": "#5dade2",
        "edge": "#5dade2",
        "category": "2–5 years",
        "label_offset": (0.15, 0.45)
    }
]

for p in points:
    p["y"] = y_on_curve(p["x"])

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(11, 6.5))
ax = plt.gca()

# Curve
ax.plot(x, y, color="#222222", linewidth=2.8)

# Points
for p in points:
    ax.scatter(
        p["x"], p["y"],
        s=120,
        facecolor=p["color"],
        edgecolor=p["edge"],
        linewidth=2,
        zorder=5
    )

    ax.annotate(
        p["name"],
        xy=(p["x"], p["y"]),
        xytext=(p["x"] + p["label_offset"][0], p["y"] + p["label_offset"][1]),
        fontsize=10,
        ha="left",
        va="bottom"
    )

# -----------------------------
# Stage annotations
# -----------------------------
stages = [
    ("Innovation\nTrigger", 0.8, y_on_curve(0.8) - 0.8),
    ("Peak of Inflated\nExpectations", 2.15, y_on_curve(2.15) + 0.8),
    ("Trough of\nDisillusionment", 4.2, y_on_curve(4.2) - 1.0),
    ("Slope of\nEnlightenment", 6.3, y_on_curve(6.3) - 1.0),
    ("Plateau of\nProductivity", 8.8, y_on_curve(8.8) + 0.6)
]

for text, sx, sy in stages:
    ax.text(
        sx, sy,
        text,
        fontsize=10,
        ha="center",
        va="center",
        color="#333333"
    )

# -----------------------------
# Legend
# -----------------------------
legend_handles = [
    plt.Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="white", markeredgecolor="#1f4e79",
        markeredgewidth=2, markersize=9,
        label="< 2 years"
    ),
    plt.Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="#5dade2", markeredgecolor="#5dade2",
        markersize=9,
        label="2–5 years"
    ),
    plt.Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="#08306b", markeredgecolor="#08306b",
        markersize=9,
        label="5–10 years"
    )
]

ax.legend(
    handles=legend_handles,
    title="Time to plateau",
    frameon=False,
    loc="lower right",
    fontsize=10,
    title_fontsize=10
)

# -----------------------------
# Axes and styling
# -----------------------------
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Expectations", fontsize=12)

ax.set_xlim(-0.2, 10.4)
ax.set_ylim(0, 10.8)

ax.set_xticks([])
ax.set_yticks([])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")

ax.set_title(
    "AI Hype Cycle Technologies Relevant to Structural Engineering Research",
    fontsize=13,
    pad=18
)

plt.tight_layout()
plt.savefig("hype_cycle_ai.png", dpi=300, bbox_inches="tight")
plt.show()