#!/usr/bin/env python3
"""
Show how observing a company's operating points across multiple model
generations — combined with knowledge of each model's ROC curve —
reveals the company's cost structure (c_fp / c_fn ratio).

Key result: at the cost-minimising threshold, dTPR/dFPR = c_fp/c_fn.
So the tangent iso-cost line at every operating point has the SAME slope,
and that slope is the company's revealed cost ratio.

Requires: knowing (or estimating) each model's ROC curve, not just the point.
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

WS = [pe.withStroke(linewidth=4, foreground="white")]


class ROC:
    def __init__(self, m, n):
        self.m, self.n = m, n

    def generate(self, n_points=2000):
        t = np.linspace(0, 1, n_points)
        return -(-t + 1) ** self.m + 1, t ** self.n


def cost_fn(fpr, tpr, c_fp, c_fn):
    return c_fp * fpr + c_fn * (1 - tpr)


def roc_slope_at(t, m, n):
    """Analytical slope dTPR/dFPR of the parametric ROC at parameter t."""
    if t <= 0 or t >= 1:
        return np.inf
    return (n * t ** (n - 1)) / (m * (1 - t) ** (m - 1))


# ── Company's hidden cost structure ──────────────────────
C_FP, C_FN = 0.74, 0.26
k = C_FP / C_FN  # ≈ 2.85

# ── 5 model generations, improving quality ───────────────
models = [
    (0.65, 0.62, "Model v1"),
    (0.50, 0.52, "Model v2"),
    (0.40, 0.42, "Model v3"),
    (0.30, 0.34, "Model v4"),
    (0.20, 0.28, "Model v5"),
]

fig, ax = plt.subplots(figsize=(10, 9))

# ── Masked iso-cost contours (company only) ──────────────
g = np.linspace(0, 1, 300)
FG, TG = np.meshgrid(g, g)
Z_co = cost_fn(FG, TG, C_FP, C_FN)
Z_co_m = np.where(FG > TG, np.nan, Z_co)

ax.fill_between([0, 1], [0, 1], 0, color="0.95", zorder=0)

lvls = np.arange(0.05, 0.75, 0.10)
cs = ax.contour(FG, TG, Z_co_m, levels=lvls, colors="steelblue",
                alpha=0.20, linewidths=0.6)
ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f", colors="steelblue")

ax.plot([0, 1], [0, 1], "k--", alpha=0.15, lw=1)

# ── Plot each model generation ───────────────────────────
measured_slopes = []

for i, (m, n, label) in enumerate(models):
    roc = ROC(m, n)
    fpr, tpr = roc.generate()

    shade = 0.35 + 0.12 * i
    ax.plot(fpr, tpr, color=str(1 - shade), lw=2.0, alpha=0.7, zorder=2)

    # Curve label near FPR ≈ 0.55
    idx_label = np.argmin(np.abs(fpr - 0.55))
    ax.text(fpr[idx_label] + 0.02, tpr[idx_label] - 0.03, label,
            fontsize=7.5, color=str(1 - shade), alpha=0.8,
            path_effects=WS)

    # Exact optimal operating point (no noise)
    cost = cost_fn(fpr, tpr, C_FP, C_FN)
    idx_op = np.argmin(cost)
    op_fpr, op_tpr = fpr[idx_op], tpr[idx_op]

    # Measure the ROC slope at this point (what an observer would compute)
    t_op = np.linspace(0, 1, 2000)[idx_op]
    slope_here = roc_slope_at(t_op, m, n)
    measured_slopes.append(slope_here)

    # Plot operating point
    ax.plot(op_fpr, op_tpr, "o", color="red", ms=10, zorder=10,
            markeredgecolor="darkred", markeredgewidth=1)

    # Tangent iso-cost line segment at the operating point
    intercept = op_tpr - k * op_fpr
    seg = 0.10
    x_lo = max(0, op_fpr - seg)
    x_hi = min(1, op_fpr + seg)
    ax.plot([x_lo, x_hi],
            [k * x_lo + intercept, k * x_hi + intercept],
            color="steelblue", lw=2.5, alpha=0.7, zorder=6)

    # Small annotation: measured slope at this point
    ax.annotate(f"slope = {slope_here:.2f}",
                xy=(op_fpr, op_tpr),
                xytext=(op_fpr + 0.10, op_tpr + 0.01),
                fontsize=7.5, color="steelblue", alpha=0.8,
                arrowprops=dict(arrowstyle="-", color="steelblue",
                                alpha=0.4, lw=0.8),
                path_effects=WS)

# ── Text box: the inference method ───────────────────────
mean_slope = np.mean(measured_slopes)
box_txt = (
    f"Inference method:\n"
    f"1. Estimate each model's ROC curve\n"
    f"   (from public benchmarks / evals)\n"
    f"2. Observe the operating point\n"
    f"3. Compute ROC slope at that point\n"
    f"\n"
    f"Measured slopes: {', '.join(f'{s:.2f}' for s in measured_slopes)}\n"
    f"Mean = {mean_slope:.2f}\n"
    f"\n"
    f"$\\Rightarrow$  $c_{{FP}} / c_{{FN}}$ = {mean_slope:.2f}"
)
ax.text(0.42, 0.40, box_txt, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow",
                  edgecolor="steelblue", alpha=0.95),
        family="monospace", zorder=12, va="top", color="#333")

# ── Axes ─────────────────────────────────────────────────
ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.08)
ax.set_aspect("equal")

legend_els = [
    Line2D([0], [0], color="0.5", lw=2, label="ROC curves (v1–v5)"),
    Line2D([0], [0], color="red", marker="o", lw=0, ms=8,
           markeredgecolor="darkred", label="Operating points"),
    Line2D([0], [0], color="steelblue", lw=2.5,
           label="Tangent iso-cost lines (all parallel)"),
    Line2D([0], [0], color="steelblue", lw=1, ls="--", alpha=0.5,
           label="Company iso-cost contours"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=9, framealpha=0.95)

plt.tight_layout()
plt.savefig(os.path.join(_HERE, "revealed_preference.png"),
            dpi=200, bbox_inches="tight")
print("Saved revealed_preference.png")
plt.close()
