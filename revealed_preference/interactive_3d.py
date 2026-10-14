#!/usr/bin/env python3
"""
Interactive 2D + 3D ROC cost-surface explorer.

Left panel: 3D view with cost planes, externalised harm bar, in-plot labels.
Right panel: 2D contour view with masked iso-cost lines.
Bottom: sliders for all parameters.

Surfaces are rendered as fill + wireframe separately so you can set
fill opacity to 0 and still see the grid.
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

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


# ── Defaults ──────────────────────────────────────────────
DEFAULTS = dict(
    elev=25, azim=-120,
    m=0.40, n=0.48,
    c_fp_co=0.72, c_fp_so=0.28,
    ng=25,
    fill_alpha=0.12, wire_alpha=0.40,
    wire_lw=0.6,
    bar_lw=10,
    proj_alpha=0.5,
)

save_counter = [0]

# ── Figure layout ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
# 3D on the left, 2D on the right, sliders below
ax3 = fig.add_axes([0.02, 0.35, 0.48, 0.58], projection="3d")
ax2 = fig.add_axes([0.56, 0.35, 0.42, 0.58])


def compute(p):
    """Compute all derived quantities from parameters."""
    roc = ROC(p["m"], p["n"])
    fpr, tpr = roc.generate()
    c_fp_co, c_fn_co = p["c_fp_co"], 1 - p["c_fp_co"]
    c_fp_so, c_fn_so = p["c_fp_so"], 1 - p["c_fp_so"]

    cost_co = cost_fn(fpr, tpr, c_fp_co, c_fn_co)
    cost_so = cost_fn(fpr, tpr, c_fp_so, c_fn_so)
    idx_op = np.argmin(cost_co)
    op_fpr, op_tpr = fpr[idx_op], tpr[idx_op]
    idx_soc = np.argmin(cost_so)
    soc_fpr, soc_tpr = fpr[idx_soc], tpr[idx_soc]

    c_co_at_op = cost_co[idx_op]
    c_so_at_op = cost_fn(op_fpr, op_tpr, c_fp_so, c_fn_so)
    c_so_opt = cost_so[idx_soc]

    return dict(
        fpr=fpr, tpr=tpr, cost_co=cost_co, cost_so=cost_so,
        c_fp_co=c_fp_co, c_fn_co=c_fn_co, c_fp_so=c_fp_so, c_fn_so=c_fn_so,
        op_fpr=op_fpr, op_tpr=op_tpr, soc_fpr=soc_fpr, soc_tpr=soc_tpr,
        c_co_at_op=c_co_at_op, c_so_at_op=c_so_at_op, c_so_opt=c_so_opt,
        harm=c_so_at_op - c_so_opt,
        safety=op_tpr, usefulness=1 - op_fpr,
    )


def draw_3d(p, d):
    """Draw the 3D panel."""
    ax3.cla()
    ng = int(p["ng"])
    fg = np.linspace(0, 1, ng)
    tg = np.linspace(0, 1, ng)
    F3, T3 = np.meshgrid(fg, tg)
    Z3_co = cost_fn(F3, T3, d["c_fp_co"], d["c_fn_co"])
    Z3_so = cost_fn(F3, T3, d["c_fp_so"], d["c_fn_so"])

    # Mask below diagonal
    below = F3 > T3
    Z3_co[below] = np.nan
    Z3_so[below] = np.nan

    # White panes
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = True
        pane.set_facecolor("white")
        pane.set_edgecolor("lightgray")

    # ── Surfaces: FILL and WIREFRAME drawn separately ────
    # Fill only (no edge) — controlled by fill_alpha
    if p["fill_alpha"] > 0.01:
        ax3.plot_surface(F3, T3, Z3_co, alpha=p["fill_alpha"],
                         color="steelblue", shade=True,
                         edgecolor="none", linewidth=0)
        ax3.plot_surface(F3, T3, Z3_so, alpha=p["fill_alpha"],
                         color="darkorange", shade=True,
                         edgecolor="none", linewidth=0)

    # Wireframe only — controlled by wire_alpha (always visible)
    stride = max(1, ng // 10)
    ax3.plot_wireframe(F3, T3, Z3_co, alpha=p["wire_alpha"],
                       color="steelblue", linewidth=p["wire_lw"],
                       rstride=stride, cstride=stride)
    ax3.plot_wireframe(F3, T3, Z3_so, alpha=p["wire_alpha"],
                       color="darkorange", linewidth=p["wire_lw"],
                       rstride=stride, cstride=stride)

    # ── ROC curves ───────────────────────────────────────
    fpr, tpr = d["fpr"], d["tpr"]
    valid = tpr >= fpr
    fv, tv = fpr[valid], tpr[valid]
    co_v = cost_fn(fv, tv, d["c_fp_co"], d["c_fn_co"])
    so_v = cost_fn(fv, tv, d["c_fp_so"], d["c_fn_so"])

    diag = np.linspace(0, 1, 50)
    ax3.plot(diag, diag, np.zeros_like(diag), "k--", lw=1, alpha=0.25)
    ax3.plot(fv, tv, np.zeros_like(fv), "k-", lw=2.5, alpha=0.5)

    # Projected ROC curves — BLACK, separate alpha
    ax3.plot(fv, tv, co_v, "k-", lw=2.5, alpha=p["proj_alpha"], zorder=5)
    ax3.plot(fv, tv, so_v, "k-", lw=2.5, alpha=p["proj_alpha"], zorder=5)

    # ── Operating point + harm bar ───────────────────────
    op_f, op_t = d["op_fpr"], d["op_tpr"]
    c_co, c_so = d["c_co_at_op"], d["c_so_at_op"]

    ax3.scatter([op_f], [op_t], [0], color="red", s=80, zorder=10,
                depthshade=False)
    ax3.scatter([op_f], [op_t], [c_co], color="steelblue", s=100,
                zorder=10, edgecolors="k", linewidths=1, depthshade=False)
    ax3.scatter([op_f], [op_t], [c_so], color="darkorange", s=100,
                zorder=10, edgecolors="k", linewidths=1, depthshade=False)

    ax3.plot([op_f, op_f], [op_t, op_t], [0, c_co], "r:", lw=2, alpha=0.7)
    ax3.plot([op_f, op_f], [op_t, op_t], [c_co, c_so], "r-",
             lw=p["bar_lw"], alpha=0.95, solid_capstyle="round", zorder=15)

    # Crosshairs
    ax3.plot([op_f, op_f], [0, op_t], [0, 0], "r:", lw=1.2, alpha=0.35)
    ax3.plot([0, op_f], [op_t, op_t], [0, 0], "r:", lw=1.2, alpha=0.35)

    # Safety + Usefulness
    ax3.plot([op_f, op_f], [0, op_t], [0, 0],
             color="seagreen", lw=3.5, alpha=0.8)
    ax3.text(op_f + 0.06, op_t * 0.25, 0.01,
             f"Safety ({d['safety']:.2f})", color="seagreen",
             fontsize=9, fontweight="bold", path_effects=WHITE_STROKE)

    ax3.plot([op_f, 1.0], [op_t, op_t], [0, 0],
             color="purple", lw=3.5, alpha=0.8)
    ax3.text(0.6, op_t - 0.08, 0.01,
             f"Usefulness ({d['usefulness']:.2f})", color="purple",
             fontsize=9, fontweight="bold", ha="center",
             path_effects=WHITE_STROKE)

    # ── In-plot labels (agent B style: boxed + connectors) ─
    mid_z = (c_co + c_so) / 2
    # Harm label
    ax3.plot([op_f, op_f + 0.28], [op_t, op_t - 0.02], [mid_z, mid_z],
             "r-", lw=1.2, alpha=0.5)
    ax3.text(op_f + 0.30, op_t - 0.03, mid_z,
             f"Externalised harm\n\u0394 = {d['harm']:.2f}", color="red",
             fontsize=9, fontweight="bold", ha="left",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.9))

    # Company cost
    ax3.text(op_f, op_t, c_co - 0.06,
             f"{c_co:.2f}", color="steelblue",
             fontsize=9, fontweight="bold", ha="center",
             path_effects=WHITE_STROKE)
    # Society cost
    ax3.text(op_f, op_t, c_so + 0.06,
             f"{c_so:.2f}", color="darkorange",
             fontsize=9, fontweight="bold", ha="center",
             path_effects=WHITE_STROKE)

    # ── Axes ─────────────────────────────────────────────
    ax3.set_xlabel("\nFPR", fontsize=11, labelpad=8)
    ax3.set_ylabel("\nTPR", fontsize=11, labelpad=8)
    ax3.set_zlabel("Cost", fontsize=11, labelpad=6)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 0.85)
    ax3.view_init(elev=p["elev"], azim=p["azim"])

    # Legend
    legend_els = [
        Line2D([0], [0], color="steelblue", lw=3, label="Cost to Company"),
        Line2D([0], [0], color="darkorange", lw=3, label="Cost to Society"),
        Line2D([0], [0], color="k", lw=2, alpha=0.5, label="ROC curve"),
        Line2D([0], [0], color="red", lw=5, label="Externalised harm"),
    ]
    ax3.legend(handles=legend_els, loc="upper left", fontsize=7.5,
               framealpha=0.9)


def draw_2d(p, d):
    """Draw the 2D panel."""
    ax2.cla()

    fpr, tpr = d["fpr"], d["tpr"]
    c_fp_co, c_fn_co = d["c_fp_co"], d["c_fn_co"]
    c_fp_so, c_fn_so = d["c_fp_so"], d["c_fn_so"]

    g = np.linspace(0, 1, 200)
    FG, TG = np.meshgrid(g, g)
    Z_co = cost_fn(FG, TG, c_fp_co, c_fn_co)
    Z_so = cost_fn(FG, TG, c_fp_so, c_fn_so)
    below = FG > TG
    Z_co_m = np.where(below, np.nan, Z_co)
    Z_so_m = np.where(below, np.nan, Z_so)

    # Gray below-diagonal
    ax2.fill_between([0, 1], [0, 1], 0, color="0.95", zorder=0)

    # Contours
    lvls_co = np.arange(0.10, 0.80, 0.15)
    lvls_so = np.arange(0.15, 0.80, 0.15)
    cs_co = ax2.contour(FG, TG, Z_co_m, levels=lvls_co, colors="steelblue",
                         alpha=0.45, linewidths=0.9)
    ax2.clabel(cs_co, inline=True, fontsize=9, fmt="%.2f", colors="steelblue")
    cs_so = ax2.contour(FG, TG, Z_so_m, levels=lvls_so, colors="darkorange",
                         alpha=0.45, linewidths=0.9)
    ax2.clabel(cs_so, inline=True, fontsize=9, fmt="%.2f", colors="darkorange")

    # Tangent iso-cost lines
    ax2.contour(FG, TG, Z_co_m, levels=[d["c_co_at_op"]], colors="steelblue",
                linewidths=2.8, linestyles="-", alpha=0.85)
    ax2.contour(FG, TG, Z_so_m, levels=[d["c_so_opt"]], colors="darkorange",
                linewidths=2.8, linestyles="-", alpha=0.85)

    # ROC + diagonal
    ax2.plot(fpr, tpr, "k-", lw=3.5, label="ROC curve", zorder=5)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.15, lw=1)

    # Crosshairs
    op_f, op_t = d["op_fpr"], d["op_tpr"]
    ax2.axvline(op_f, color="red", ls=":", lw=1.5, alpha=0.5, zorder=4)
    ax2.axhline(op_t, color="red", ls=":", lw=1.5, alpha=0.5, zorder=4)

    # Operating points
    ax2.plot(op_f, op_t, "o", color="red", ms=12, zorder=10,
             markeredgecolor="darkred", markeredgewidth=1,
             label="Company operating pt")
    ax2.plot(d["soc_fpr"], d["soc_tpr"], "s", color="darkorange", ms=10,
             zorder=10, markerfacecolor="none", markeredgewidth=2.5,
             label="Society-optimal pt")

    # Measuring sticks
    bx = -0.04
    draw_brace(ax2, bx, 0, bx, op_t,
               f"Safety (TPR={d['safety']:.2f})",
               color="seagreen", text_offset=0.05, side="left", fontsize=9)
    ax2.plot([bx + 0.016, op_f], [op_t, op_t],
             color="seagreen", ls=":", lw=1, alpha=0.4, zorder=7)

    by = op_t + 0.06
    draw_brace(ax2, op_f, by, 1.0, by,
               f"Usefulness (1\u2212FPR={d['usefulness']:.2f})",
               color="purple", text_offset=0.04, fontsize=9)

    # Labels
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


def draw_all(params):
    d = compute(params)
    # Clear figure-level texts (fine-print) before redrawing
    for t in list(fig.texts):
        t.remove()
    draw_3d(params, d)
    draw_2d(params, d)
    # Fine-print
    fig.text(0.25, 0.31,
             "Companies may value not annoying users (Usefulness) "
             "over blocking harmful answers (Safety)",
             ha="center", fontsize=7, color="gray")
    fig.text(0.25, 0.295,
             "Society may value Safety over minor annoyance",
             ha="center", fontsize=7, color="gray")
    fig.text(0.25, 0.28,
             "Simplifying assumptions: Cost(TP) = Cost(TN) = 0",
             ha="center", fontsize=6.5, color="gray", fontstyle="italic")


def get_params():
    return {name: s.val for name, s in sliders.items()}


def on_update(val):
    draw_all(get_params())
    fig.canvas.draw_idle()


def on_save(event):
    """Export a CLEAN figure (no sliders) using current settings."""
    save_counter[0] += 1
    p = get_params()
    d = compute(p)

    # Create a fresh figure with just the two plots
    export_fig = plt.figure(figsize=(20, 10))
    export_ax3 = export_fig.add_axes([0.02, 0.14, 0.48, 0.78], projection="3d")
    export_ax2 = export_fig.add_axes([0.56, 0.14, 0.42, 0.78])

    # Temporarily swap global axes, draw, then restore
    global ax3, ax2
    orig_ax3, orig_ax2 = ax3, ax2
    ax3, ax2 = export_ax3, export_ax2
    draw_3d(p, d)
    draw_2d(p, d)
    ax3, ax2 = orig_ax3, orig_ax2

    # Fine-print on export figure
    export_fig.text(0.25, 0.06,
                    "Companies may value not annoying users (Usefulness) "
                    "over blocking harmful answers (Safety)",
                    ha="center", fontsize=7.5, color="gray")
    export_fig.text(0.25, 0.04,
                    "Society may value Safety over minor annoyance",
                    ha="center", fontsize=7.5, color="gray")
    export_fig.text(0.25, 0.02,
                    "Simplifying assumptions: Cost(TP) = Cost(TN) = 0",
                    ha="center", fontsize=7, color="gray", fontstyle="italic")

    # Print current slider values as subtitle
    export_fig.text(0.75, 0.06,
                    f"m={p['m']:.2f}  n={p['n']:.2f}  "
                    f"Co c_FP={p['c_fp_co']:.2f}  So c_FP={p['c_fp_so']:.2f}  "
                    f"elev={p['elev']:.0f}  azim={p['azim']:.0f}",
                    ha="center", fontsize=7, color="gray", family="monospace")

    fname = os.path.join(_HERE, f"interactive_export_{save_counter[0]}.png")
    export_fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(export_fig)
    print(f"Saved (clean, no sliders): {fname}")


# ── Sliders ───────────────────────────────────────────────
slider_defs = [
    # name          label               min   max   default             step  color
    ("elev",       "Elevation",          0,   90,   DEFAULTS["elev"],      1, "#888"),
    ("azim",       "Azimuth",         -180,  180,   DEFAULTS["azim"],      1, "#888"),
    ("m",          "ROC m",           0.05,  1.0,   DEFAULTS["m"],      0.05, "#888"),
    ("n",          "ROC n",           0.05,  1.0,   DEFAULTS["n"],      0.05, "#888"),
    ("c_fp_co",    "Company c_FP",    0.50, 0.95,   DEFAULTS["c_fp_co"], 0.01, "steelblue"),
    ("c_fp_so",    "Society c_FP",    0.05, 0.50,   DEFAULTS["c_fp_so"], 0.01, "darkorange"),
    ("ng",         "Grid density",      8,   50,    DEFAULTS["ng"],        1, "#888"),
    ("fill_alpha", "Surface fill α",  0.00, 0.50,   DEFAULTS["fill_alpha"], 0.01, "#888"),
    ("wire_alpha", "Wireframe α",     0.05, 0.90,   DEFAULTS["wire_alpha"], 0.01, "#888"),
    ("wire_lw",    "Wire linewidth",  0.2,   2.0,   DEFAULTS["wire_lw"], 0.1, "#888"),
    ("bar_lw",     "Harm bar width",     4,   20,   DEFAULTS["bar_lw"],    1, "red"),
    ("proj_alpha", "Projected ROC α", 0.00, 1.00,   DEFAULTS["proj_alpha"], 0.05, "k"),
]

sliders = {}
n_sl = len(slider_defs)
sl_h = 0.016
sl_gap = 0.020
sl_bot = 0.27 - n_sl * sl_gap

for i, (name, label, vmin, vmax, vinit, vstep, col) in enumerate(slider_defs):
    y = sl_bot + (n_sl - 1 - i) * sl_gap
    ax_s = fig.add_axes([0.16, y, 0.50, sl_h])
    s = Slider(ax_s, label, vmin, vmax, valinit=vinit, valstep=vstep, color=col)
    s.on_changed(on_update)
    sliders[name] = s

# Save button
ax_btn = fig.add_axes([0.70, sl_bot, 0.08, 0.025])
btn = Button(ax_btn, "Save PNG", color="#eee", hovercolor="#ddd")
btn.on_clicked(on_save)

# ── Initial draw ──────────────────────────────────────────
draw_all(DEFAULTS)
plt.show()
