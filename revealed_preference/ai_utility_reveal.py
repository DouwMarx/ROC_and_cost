#!/usr/bin/env python3
"""
Production-quality 2D + 3D ROC cost-surface plots.
Settings tuned via the interactive explorer.
Outputs: roc_utility_2d.png and roc_utility_3d.png
"""


import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

WHITE_STROKE = [pe.withStroke(linewidth=3, foreground="white")]
WS5 = [pe.withStroke(linewidth=5, foreground="white")]


class ROC:
    def __init__(self, m, n):
        self.m, self.n = m, n

    def generate(self, n_points=500):
        t = np.linspace(0, 1, n_points)
        return -(-t + 1) ** self.m + 1, t ** self.n


def cost_fn(fpr, tpr, c_fp, c_fn):
    return c_fp * fpr + c_fn * (1 - tpr)


def draw_brace(ax, x0, y0, x1, y1, text, color="black", fontsize=10,
               text_offset=0.06, lw=2.0, side="left"):
    cap = 0.016
    if abs(x1 - x0) < 0.001:  # Vertical
        mid = (y0 + y1) / 2
        ax.plot([x0, x0], [y0, y1], color=color, lw=lw, zorder=8)
        ax.plot([x0 - cap, x0 + cap], [y0, y0], color=color, lw=lw, zorder=8)
        ax.plot([x0 - cap, x0 + cap], [y1, y1], color=color, lw=lw, zorder=8)
        ha = "right" if side == "left" else "left"
        sgn = -1 if side == "left" else 1
        ax.text(x0 + sgn * text_offset, mid, text, rotation=90,
                va="center", ha=ha, fontsize=fontsize, color=color,
                fontweight="bold", path_effects=WS5)
    else:  # Horizontal
        mid = (x0 + x1) / 2
        ax.plot([x0, x1], [y0, y0], color=color, lw=lw, zorder=8)
        ax.plot([x0, x0], [y0 - cap, y0 + cap], color=color, lw=lw, zorder=8)
        ax.plot([x1, x1], [y0 - cap, y0 + cap], color=color, lw=lw, zorder=8)
        ax.text(mid, y0 + text_offset, text, rotation=0,
                va="bottom", ha="center", fontsize=fontsize, color=color,
                fontweight="bold", path_effects=WS5)


# ==========================================================
# Parameters (tuned in interactive explorer)
# ==========================================================
M, N = 0.40, 0.48
C_FP_CO, C_FP_SO = 0.74, 0.39
C_FN_CO, C_FN_SO = 1 - C_FP_CO, 1 - C_FP_SO

ELEV, AZIM = 26, -127
NG = 22
FILL_ALPHA = 0.00       # wireframe only
WIRE_ALPHA = 0.75
WIRE_LW = 0.8
BAR_LW = 10
PROJ_ALPHA = 0.4

# ── Compute ──────────────────────────────────────────────
roc = ROC(M, N)
fpr, tpr = roc.generate()

cost_co = cost_fn(fpr, tpr, C_FP_CO, C_FN_CO)
cost_so = cost_fn(fpr, tpr, C_FP_SO, C_FN_SO)
idx_op = np.argmin(cost_co)
op_fpr, op_tpr = fpr[idx_op], tpr[idx_op]

idx_soc = np.argmin(cost_so)
soc_fpr, soc_tpr = fpr[idx_soc], tpr[idx_soc]

c_co_at_op = cost_co[idx_op]
c_so_at_op = cost_fn(op_fpr, op_tpr, C_FP_SO, C_FN_SO)
c_so_opt = cost_so[idx_soc]
harm = c_so_at_op - c_so_opt
safety = op_tpr
usefulness = 1 - op_fpr

print(f"Company op:  FPR={op_fpr:.3f}  TPR={op_tpr:.3f}")
print(f"Society op:  FPR={soc_fpr:.3f}  TPR={soc_tpr:.3f}")
print(f"Safety={safety:.2f}  Usefulness={usefulness:.2f}")
print(f"Cost @ co pt — co:{c_co_at_op:.3f}  so:{c_so_at_op:.3f}")
print(f"Harm: {harm:.3f}")

# ROC above diagonal only
valid = tpr >= fpr
fv, tv = fpr[valid], tpr[valid]
co_v = cost_fn(fv, tv, C_FP_CO, C_FN_CO)
so_v = cost_fn(fv, tv, C_FP_SO, C_FN_SO)


# ==========================================================
# Figure 1 — 3D
# ==========================================================
fig3 = plt.figure(figsize=(12, 10))
ax3 = fig3.add_axes([0.02, 0.14, 0.96, 0.80], projection="3d")

for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
    pane.fill = True
    pane.set_facecolor("white")
    pane.set_edgecolor("lightgray")

fg = np.linspace(0, 1, NG)
tg = np.linspace(0, 1, NG)
F3, T3 = np.meshgrid(fg, tg)
Z3_co = cost_fn(F3, T3, C_FP_CO, C_FN_CO)
Z3_so = cost_fn(F3, T3, C_FP_SO, C_FN_SO)
below = F3 > T3
Z3_co[below] = np.nan
Z3_so[below] = np.nan

# Fill (off when FILL_ALPHA=0)
if FILL_ALPHA > 0.01:
    ax3.plot_surface(F3, T3, Z3_co, alpha=FILL_ALPHA, color="steelblue",
                     shade=True, edgecolor="none", linewidth=0)
    ax3.plot_surface(F3, T3, Z3_so, alpha=FILL_ALPHA, color="darkorange",
                     shade=True, edgecolor="none", linewidth=0)

# Wireframe (always visible)
stride = max(1, NG // 10)
ax3.plot_wireframe(F3, T3, Z3_co, alpha=WIRE_ALPHA, color="steelblue",
                   linewidth=WIRE_LW, rstride=stride, cstride=stride)
ax3.plot_wireframe(F3, T3, Z3_so, alpha=WIRE_ALPHA, color="darkorange",
                   linewidth=WIRE_LW, rstride=stride, cstride=stride)

# ROC curves
diag = np.linspace(0, 1, 50)
ax3.plot(diag, diag, np.zeros_like(diag), "k--", lw=1, alpha=0.25)
ax3.plot(fv, tv, np.zeros_like(fv), "k-", lw=2.5, alpha=0.5)
ax3.plot(fv, tv, co_v, "k-", lw=2.5, alpha=PROJ_ALPHA, zorder=5)
ax3.plot(fv, tv, so_v, "k-", lw=2.5, alpha=PROJ_ALPHA, zorder=5)

# Operating point + harm bar
ax3.scatter([op_fpr], [op_tpr], [0], color="red", s=80, zorder=10,
            depthshade=False)
ax3.scatter([op_fpr], [op_tpr], [c_co_at_op], color="steelblue", s=100,
            zorder=10, edgecolors="k", linewidths=1, depthshade=False)
ax3.scatter([op_fpr], [op_tpr], [c_so_at_op], color="darkorange", s=100,
            zorder=10, edgecolors="k", linewidths=1, depthshade=False)

ax3.plot([op_fpr, op_fpr], [op_tpr, op_tpr], [0, c_co_at_op],
         "r:", lw=2, alpha=0.7)
ax3.plot([op_fpr, op_fpr], [op_tpr, op_tpr], [c_co_at_op, c_so_at_op],
         "r-", lw=BAR_LW, alpha=0.95, solid_capstyle="round", zorder=15)

# Crosshairs
ax3.plot([op_fpr, op_fpr], [0, op_tpr], [0, 0], "r:", lw=1.2, alpha=0.35)
ax3.plot([0, op_fpr], [op_tpr, op_tpr], [0, 0], "r:", lw=1.2, alpha=0.35)

# Safety + Usefulness
ax3.plot([op_fpr, op_fpr], [0, op_tpr], [0, 0],
         color="seagreen", lw=3.5, alpha=0.8)
ax3.text(op_fpr + 0.06, op_tpr * 0.25, 0.01,
         f"Safety ({safety:.2f})", color="seagreen",
         fontsize=9, fontweight="bold", path_effects=WHITE_STROKE)

ax3.plot([op_fpr, 1.0], [op_tpr, op_tpr], [0, 0],
         color="purple", lw=3.5, alpha=0.8)
ax3.text(0.6, op_tpr - 0.08, 0.01,
         f"Usefulness ({usefulness:.2f})", color="purple",
         fontsize=9, fontweight="bold", ha="center", path_effects=WHITE_STROKE)

# Harm label
mid_z = (c_co_at_op + c_so_at_op) / 2
ax3.plot([op_fpr, op_fpr + 0.28], [op_tpr, op_tpr - 0.02], [mid_z, mid_z],
         "r-", lw=1.2, alpha=0.5)
ax3.text(op_fpr + 0.30, op_tpr - 0.03, mid_z,
         f"Externalised harm\n\u0394 = {harm:.2f}", color="red",
         fontsize=9, fontweight="bold", ha="left",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.9))

# Cost value labels
ax3.text(op_fpr, op_tpr, c_co_at_op - 0.06,
         f"{c_co_at_op:.2f}", color="steelblue",
         fontsize=9, fontweight="bold", ha="center", path_effects=WHITE_STROKE)
ax3.text(op_fpr, op_tpr, c_so_at_op + 0.06,
         f"{c_so_at_op:.2f}", color="darkorange",
         fontsize=9, fontweight="bold", ha="center", path_effects=WHITE_STROKE)

ax3.set_xlabel("\nFPR", fontsize=11, labelpad=8)
ax3.set_ylabel("\nTPR", fontsize=11, labelpad=8)
ax3.set_zlabel("Cost", fontsize=11, labelpad=6)
ax3.set_title("Cost to Company vs Cost to Society\n"
              "at the Company's Chosen Operating Point",
              fontsize=14, fontweight="bold", pad=15)
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 0.85)
ax3.view_init(elev=ELEV, azim=AZIM)

legend_els = [
    Line2D([0], [0], color="steelblue", lw=3, label="Cost to Company"),
    Line2D([0], [0], color="darkorange", lw=3, label="Cost to Society"),
    Line2D([0], [0], color="k", lw=2, alpha=0.5, label="ROC curve"),
    Line2D([0], [0], color="red", lw=5, label="Externalised harm"),
]
ax3.legend(handles=legend_els, loc="upper left", fontsize=8, framealpha=0.9)

# Fine-print
fig3.text(0.5, 0.06,
          "Companies may value not annoying users (Usefulness) "
          "over blocking harmful answers (Safety)",
          ha="center", fontsize=8, color="gray")
fig3.text(0.5, 0.035,
          "Society may value Safety over minor annoyance",
          ha="center", fontsize=8, color="gray")
fig3.text(0.5, 0.01,
          "Simplifying assumptions: Cost(TP) = Cost(TN) = 0",
          ha="center", fontsize=7.5, color="gray", fontstyle="italic")

fig3.savefig(os.path.join(_HERE, "roc_utility_3d.png"),
             dpi=200, bbox_inches="tight")
print("Saved roc_utility_3d.png")
plt.close(fig3)


# ==========================================================
# Figure 2 — 2D
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(10, 9))

g = np.linspace(0, 1, 200)
FG, TG = np.meshgrid(g, g)
Z_co = cost_fn(FG, TG, C_FP_CO, C_FN_CO)
Z_so = cost_fn(FG, TG, C_FP_SO, C_FN_SO)
below2 = FG > TG
Z_co_m = np.where(below2, np.nan, Z_co)
Z_so_m = np.where(below2, np.nan, Z_so)

ax2.fill_between([0, 1], [0, 1], 0, color="0.95", zorder=0)

lvls_co = np.arange(0.10, 0.80, 0.15)
lvls_so = np.arange(0.15, 0.80, 0.15)
cs_co = ax2.contour(FG, TG, Z_co_m, levels=lvls_co, colors="steelblue",
                     alpha=0.45, linewidths=0.9)
ax2.clabel(cs_co, inline=True, fontsize=9, fmt="%.2f", colors="steelblue")
cs_so = ax2.contour(FG, TG, Z_so_m, levels=lvls_so, colors="darkorange",
                     alpha=0.45, linewidths=0.9)
ax2.clabel(cs_so, inline=True, fontsize=9, fmt="%.2f", colors="darkorange")

ax2.contour(FG, TG, Z_co_m, levels=[c_co_at_op], colors="steelblue",
            linewidths=2.8, linestyles="-", alpha=0.85)
ax2.contour(FG, TG, Z_so_m, levels=[c_so_opt], colors="darkorange",
            linewidths=2.8, linestyles="-", alpha=0.85)

ax2.plot(fpr, tpr, "k-", lw=3.5, label="ROC curve", zorder=5)
ax2.plot([0, 1], [0, 1], "k--", alpha=0.15, lw=1)

ax2.axvline(op_fpr, color="red", ls=":", lw=1.5, alpha=0.5, zorder=4)
ax2.axhline(op_tpr, color="red", ls=":", lw=1.5, alpha=0.5, zorder=4)

ax2.plot(op_fpr, op_tpr, "o", color="red", ms=12, zorder=10,
         markeredgecolor="darkred", markeredgewidth=1,
         label="Company operating pt")
ax2.plot(soc_fpr, soc_tpr, "s", color="darkorange", ms=10,
         zorder=10, markerfacecolor="none", markeredgewidth=2.5,
         label="Society-optimal pt")

bx = -0.04
draw_brace(ax2, bx, 0, bx, op_tpr,
           f"Safety (TPR={safety:.2f})",
           color="seagreen", text_offset=0.05, side="left", fontsize=9)
ax2.plot([bx + 0.016, op_fpr], [op_tpr, op_tpr],
         color="seagreen", ls=":", lw=1, alpha=0.4, zorder=7)

by = op_tpr + 0.06
draw_brace(ax2, op_fpr, by, 1.0, by,
           f"Usefulness (1\u2212FPR={usefulness:.2f})",
           color="purple", text_offset=0.04, fontsize=9)

ax2.text(0.50, 1.05, "Company iso-cost", color="steelblue",
         fontsize=8.5, fontstyle="italic", alpha=0.8, path_effects=WS5)
ax2.text(0.65, 0.78, "Society iso-cost", color="darkorange",
         fontsize=8.5, fontstyle="italic", alpha=0.8, rotation=20,
         path_effects=WS5)

ax2.set_xlabel("False Positive Rate (FPR)", fontsize=11)
ax2.set_ylabel("True Positive Rate (TPR)", fontsize=11)
ax2.set_xlim(-0.16, 1.06)
ax2.set_ylim(-0.04, 1.14)
ax2.set_aspect("equal")
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.legend(loc="lower right", fontsize=8, framealpha=0.95)
plt.tight_layout()

fig2.savefig(os.path.join(_HERE, "roc_utility_2d.png"),
             dpi=200, bbox_inches="tight")
print("Saved roc_utility_2d.png")
plt.close(fig2)

print("Done — v3 figures saved.")
