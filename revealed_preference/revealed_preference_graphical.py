#!/usr/bin/env python3
"""
Graphical solution: show how each constraint slices the feasible
region for c_fp, producing the final interval.

Also: report AUC for each model generation, and interpret accuracy.
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

WS = [pe.withStroke(linewidth=3, foreground="white")]


def roc_from_gaussians(mu_sep, n_points=2000):
    thresholds = np.linspace(-4, mu_sep + 4, n_points)
    fpr = 1 - stats.norm.cdf(thresholds, loc=0, scale=1)
    tpr = 1 - stats.norm.cdf(thresholds, loc=mu_sep, scale=1)
    order = np.argsort(fpr)
    return fpr[order], tpr[order]


def cost_fn(fpr, tpr, c_fp, c_fn):
    return c_fp * fpr + c_fn * (1 - tpr)


def optimal_op(fpr, tpr, c_fp, c_fn):
    cost = c_fp * fpr + c_fn * (1 - tpr)
    idx = np.argmin(cost)
    return fpr[idx], tpr[idx]


def cfp_to_k(c):
    return c / max(1 - c, 1e-9)


# ── Setup ─────────────────────────────────────────────────
true_k = 3.0
true_cfp = true_k / (1 + true_k)
true_cfn = 1 - true_cfp

n_gen = 8
mu_seps = np.linspace(0.5, 3.5, n_gen)

# Compute ROCs, AUCs, operating points
rocs, ops, aucs = [], [], []
for mu in mu_seps:
    fpr, tpr = roc_from_gaussians(mu)
    auc = np.trapz(tpr, fpr)
    f_op, t_op = optimal_op(fpr, tpr, true_cfp, true_cfn)
    rocs.append((fpr, tpr))
    ops.append((f_op, t_op))
    aucs.append(auc)

print("Model generations:")
print(f"{'Gen':>4} {'μ_sep':>6} {'AUC':>6} {'FPR':>8} {'TPR':>8}")
for i, (mu, auc, (f, t)) in enumerate(zip(mu_seps, aucs, ops)):
    print(f"  {i+1:2d}   {mu:5.2f}  {auc:5.3f}  {f:7.4f}  {t:7.4f}")

print(f"\nAUC range: {aucs[0]:.3f} to {aucs[-1]:.3f}")
print(f"True k = {true_k}, true c_fp = {true_cfp:.4f}")


# ── Collect all constraints ───────────────────────────────
constraints = []  # (type, label, lo_or_hi, bound_cfp)

for i, (fi, ti) in enumerate(ops):
    # vs (0,0): c_fp < tpr/(tpr+fpr)
    if fi + ti > 1e-12:
        b = ti / (ti + fi)
        constraints.append(("upper", f"Gen {i+1} vs (0,0)", "hi", b))
    # vs (1,1): c_fp > (1-tpr)/((1-tpr)+(1-fpr))
    denom = (1 - ti) + (1 - fi)
    if denom > 1e-12:
        b = (1 - ti) / denom
        constraints.append(("lower", f"Gen {i+1} vs (1,1)", "lo", b))

for i in range(n_gen):
    for j in range(i + 1, n_gen):
        fi, ti = ops[i]
        fj, tj = ops[j]
        A = (fj - fi) + (tj - ti)
        B = tj - ti
        if abs(A) < 1e-12:
            continue
        bound = B / A
        if A > 0:
            constraints.append(("upper", f"Gen {i+1}→{j+1} pair", "hi", bound))
        else:
            constraints.append(("lower", f"Gen {i+1}→{j+1} pair", "lo", bound))

# Compute final interval
c_lo = max([c[3] for c in constraints if c[2] == "lo"], default=0.001)
c_hi = min([c[3] for c in constraints if c[2] == "hi"], default=0.999)
k_lo, k_hi = cfp_to_k(c_lo), cfp_to_k(c_hi)

print(f"\nFinal interval: c_fp ∈ [{c_lo:.4f}, {c_hi:.4f}]")
print(f"              → k ∈ [{k_lo:.3f}, {k_hi:.3f}]")
print(f"True k = {true_k:.1f}, interval width = {np.log10(k_hi/k_lo):.2f} decades")


# ── Figure 1: Graphical constraint solution ───────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                         gridspec_kw={"height_ratios": [1, 1]})

# --- Panel (0,0): The ROC curves + operating points ---
ax = axes[0, 0]
for i, ((fpr, tpr), (fo, to), auc) in enumerate(zip(rocs, ops, aucs)):
    shade = 0.3 + 0.07 * i
    ax.plot(fpr, tpr, color=str(1 - shade), lw=1.5, alpha=0.7)
    ax.plot(fo, to, "o", color="red", ms=8, markeredgecolor="darkred",
            markeredgewidth=0.8, zorder=10)
    # AUC label near each curve
    idx_lbl = np.argmin(np.abs(fpr - 0.5))
    ax.text(fpr[idx_lbl] + 0.02, tpr[idx_lbl] - 0.025,
            f"AUC={auc:.2f}", fontsize=6.5, color=str(1 - shade),
            alpha=0.85, path_effects=WS)

ax.plot([0, 1], [0, 1], "k--", alpha=0.15, lw=1)
ax.set_xlabel("FPR", fontsize=11)
ax.set_ylabel("TPR", fontsize=11)
ax.set_title(f"{n_gen} model generations (AUC {aucs[0]:.2f} → {aucs[-1]:.2f})",
             fontsize=12, fontweight="bold")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.set_aspect("equal")

# --- Panel (0,1): All constraints as horizontal bars on c_fp axis ---
ax = axes[0, 1]

# Sort constraints by bound value
lower_cons = sorted([c for c in constraints if c[2] == "lo"],
                    key=lambda c: c[3])
upper_cons = sorted([c for c in constraints if c[2] == "hi"],
                    key=lambda c: -c[3])

all_sorted = lower_cons + upper_cons
y_pos = 0
colors_lo, colors_hi = "seagreen", "steelblue"

for c in lower_cons:
    _, label, _, bnd = c
    ax.barh(y_pos, bnd - 0, left=0, height=0.7, color=colors_lo,
            alpha=0.3, edgecolor=colors_lo, linewidth=0.5)
    ax.plot([bnd], [y_pos], ">", color=colors_lo, ms=8)
    ax.text(0.01, y_pos, label, fontsize=6, va="center", color=colors_lo,
            fontweight="bold")
    y_pos += 1

y_pos += 0.5  # gap
sep_y = y_pos - 0.25

for c in upper_cons[:10]:  # show top 10 only (there can be many)
    _, label, _, bnd = c
    ax.barh(y_pos, 1.0 - bnd, left=bnd, height=0.7, color=colors_hi,
            alpha=0.3, edgecolor=colors_hi, linewidth=0.5)
    ax.plot([bnd], [y_pos], "<", color=colors_hi, ms=8)
    ax.text(0.99, y_pos, label, fontsize=6, va="center", ha="right",
            color=colors_hi, fontweight="bold")
    y_pos += 1

if len(upper_cons) > 10:
    ax.text(0.99, y_pos, f"... +{len(upper_cons)-10} more upper bounds",
            fontsize=6, va="center", ha="right", color="gray")

# True value + feasible region
ax.axvline(true_cfp, color="red", lw=2, ls="--", zorder=10, label=f"True c_fp = {true_cfp:.3f}")
ax.axvspan(c_lo, c_hi, alpha=0.15, color="gold", zorder=0,
           label=f"Feasible: [{c_lo:.3f}, {c_hi:.3f}]")
ax.axhline(sep_y, color="gray", lw=0.5, ls=":")

ax.set_xlim(0, 1)
ax.set_xlabel("c$_{FP}$ (company's FP cost weight)", fontsize=11)
ax.set_title("Constraints slicing the c$_{FP}$ axis", fontsize=12,
             fontweight="bold")
ax.set_yticks([])
ax.legend(fontsize=9, loc="upper center")
ax.grid(alpha=0.2, axis="x")

# --- Panel (1,0): Cost at each operating point as function of c_fp ---
ax = axes[1, 0]
cfp_grid = np.linspace(0.01, 0.99, 500)

for i, (fi, ti) in enumerate(ops):
    costs = cfp_grid * fi + (1 - cfp_grid) * (1 - ti)
    shade = 0.3 + 0.07 * i
    ax.plot(cfp_grid, costs, color=str(1 - shade), lw=1.5,
            label=f"Gen {i+1}" if i % 2 == 0 else None)

# Highlight feasible region
ax.axvspan(c_lo, c_hi, alpha=0.15, color="gold", zorder=0)
ax.axvline(true_cfp, color="red", lw=2, ls="--", label=f"True c_fp")

# In the feasible region, costs should be monotonically ordered
# (Gen 1 highest, Gen N lowest)
ax.set_xlabel("c$_{FP}$", fontsize=11)
ax.set_ylabel("Cost at operating point", fontsize=11)
ax.set_title("Cost curves — feasible region has correct ordering",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.2)
ax.set_xlim(0, 1)

# Add annotation showing where ordering breaks
ax.text(c_lo, ax.get_ylim()[1] * 0.95, "←ordering\n   breaks",
        fontsize=8, color="gray", ha="right", va="top")
ax.text(c_hi, ax.get_ylim()[1] * 0.95, "ordering→\nbreaks   ",
        fontsize=8, color="gray", ha="left", va="top")

# --- Panel (1,1): Same but in k-space (log scale), with interpretation ---
ax = axes[1, 1]

k_grid = np.logspace(-2, 3, 300)
cfp_grid2 = k_grid / (1 + k_grid)

for i, (fi, ti) in enumerate(ops):
    costs = cfp_grid2 * fi + (1 - cfp_grid2) * (1 - ti)
    shade = 0.3 + 0.07 * i
    ax.plot(k_grid, costs, color=str(1 - shade), lw=1.5)

ax.axvspan(k_lo, k_hi, alpha=0.15, color="gold", zorder=0)
ax.axvline(true_k, color="red", lw=2, ls="--")
ax.set_xscale("log")
ax.set_xlabel("k = c$_{FP}$/c$_{FN}$", fontsize=11)
ax.set_ylabel("Cost at operating point", fontsize=11)
ax.set_title("Same in k-space (log scale)", fontsize=12, fontweight="bold")
ax.grid(alpha=0.2)

# Interpretation box
midpoint_k = np.sqrt(k_lo * k_hi)  # geometric midpoint
width_factor = k_hi / k_lo

interp = (
    f"True k = {true_k:.1f}\n"
    f"Inferred: k ∈ [{k_lo:.2f}, {k_hi:.2f}]\n"
    f"Width: {width_factor:.1f}× (= {np.log10(width_factor):.2f} decades)\n"
    f"Geometric midpoint: {midpoint_k:.2f}\n"
    f"\n"
    f"Interpretation:\n"
    f"  The company values not annoying users\n"
    f"  {k_lo:.1f}–{k_hi:.1f}× more than blocking harm.\n"
    f"  (True: {true_k:.1f}×)"
)
ax.text(0.02, 0.02, interp, transform=ax.transAxes, fontsize=9,
        va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                  ec="steelblue", alpha=0.95))

plt.tight_layout()
plt.savefig(os.path.join(_HERE, "revealed_preference_graphical.png"),
            dpi=200, bbox_inches="tight")
print("\nSaved revealed_preference_graphical.png")
plt.close()

print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
print(f"""
With {n_gen} model generations spanning AUC {aucs[0]:.2f} to {aucs[-1]:.2f}:

  True cost ratio k = {true_k:.1f}
  Inferred interval: k ∈ [{k_lo:.2f}, {k_hi:.2f}]
  Width factor: {width_factor:.1f}× ({np.log10(width_factor):.2f} decades)

  In plain language: from observing 8 model deployments,
  an outsider could conclude the company weights usefulness
  {k_lo:.1f}–{k_hi:.1f}× more than safety.

  Quality of estimate:
    < 0.3 decades (2×)  = "pinned down"
    0.3-0.7 decades     = "useful ballpark"
    0.7-1.0 decades     = "right order of magnitude"
    1.0-2.0 decades     = "directional only"
    > 2.0 decades       = "uninformative"

  This example: {np.log10(width_factor):.2f} decades → {"useful ballpark" if np.log10(width_factor) < 0.7 else "right order of magnitude" if np.log10(width_factor) < 1.0 else "directional only"}
""")
