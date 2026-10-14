#!/usr/bin/env python3
"""
FINAL: How well can you infer k = c_fp/c_fn from operating points?

Method: trivial-classifier bounds (k ≤ TPR/FPR, k ≥ (1-TPR)/(1-FPR))
+ pairwise dominance. Point estimate = geometric midpoint.

Varied: true k (1e-10 to 1e10), AUC_min (0.55–0.95), n_gen (2–20).
AUC_max = 0.99 always. Gaussian ROC curves.
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def auc_to_mu(auc):
    return np.sqrt(2) * stats.norm.ppf(auc)

def roc_from_mu(mu, n_pts=3000):
    th = np.linspace(-4, mu + 4, n_pts)
    fpr = 1 - stats.norm.cdf(th)
    tpr = 1 - stats.norm.cdf(th, loc=mu)
    o = np.argsort(fpr)
    return fpr[o], tpr[o]

def optimal_op(fpr, tpr, c_fp, c_fn):
    idx = np.argmin(c_fp * fpr + c_fn * (1 - tpr))
    return fpr[idx], tpr[idx]

def cfp_to_k(c):
    return c / max(1 - c, 1e-15)

def infer_interval(ops):
    k_lo, k_hi = 1e-12, 1e12
    n = len(ops)
    for f, t in ops:
        if f > 1e-12:
            k_hi = min(k_hi, t / f)
        if (1 - f) > 1e-12:
            k_lo = max(k_lo, (1 - t) / (1 - f))
    for i in range(n):
        for j in range(i + 1, n):
            fi, ti = ops[i]
            fj, tj = ops[j]
            dfpr = fj - fi
            dtpr = tj - ti
            if abs(dfpr) < 1e-14:
                continue
            if dfpr > 0:
                k_hi = min(k_hi, dtpr / dfpr)
            elif dfpr < 0:
                k_lo = max(k_lo, dtpr / dfpr)
    return max(k_lo, 1e-12), min(k_hi, 1e12)


# ── Parameters ────────────────────────────────────────────
np.random.seed(42)

true_k_values = np.logspace(-10, 10, 21)
n_gen_values = [2, 5, 10, 20]
auc_min_values = [0.55, 0.65, 0.75, 0.85, 0.95]
AUC_MAX = 0.99
N_TRIALS = 50

shape = (len(true_k_values), len(n_gen_values), len(auc_min_values))
res_logwidth = np.zeros(shape)
res_midpt_err = np.zeros(shape)

print(f"k: 1e-10 to 1e10 (21 values)")
print(f"n_gen: {n_gen_values}, AUC_min: {auc_min_values}, AUC_max={AUC_MAX}")
print(f"Trials: {N_TRIALS}\n")

for ik, true_k in enumerate(true_k_values):
    c_fp = true_k / (1 + true_k)
    c_fn = 1 - c_fp
    for ig, n_gen in enumerate(n_gen_values):
        for ia, auc_min in enumerate(auc_min_values):
            mu_lo, mu_hi = auc_to_mu(auc_min), auc_to_mu(AUC_MAX)
            lw_list, err_list = [], []
            for _ in range(N_TRIALS):
                base = np.linspace(mu_lo, mu_hi, n_gen)
                jit = np.random.normal(0, 0.1, n_gen)
                jit[0] = jit[-1] = 0
                mus = np.sort(np.clip(base + jit, mu_lo, mu_hi))
                ops = [optimal_op(*roc_from_mu(m), c_fp, c_fn) for m in mus]
                k_lo, k_hi = infer_interval(ops)
                lw_list.append(np.log10(k_hi / max(k_lo, 1e-12)))
                k_est = np.sqrt(max(k_lo, 1e-12) * min(k_hi, 1e12))
                err_list.append(abs(np.log10(k_est / true_k)))
            res_logwidth[ik, ig, ia] = np.median(lw_list)
            res_midpt_err[ik, ig, ia] = np.median(err_list)

    ig_ref = n_gen_values.index(10)
    ia_ref = auc_min_values.index(0.65)
    print(f"  k={true_k:10.2g}: width={res_logwidth[ik,ig_ref,ia_ref]:.2f} dec, "
          f"midpt_err={res_midpt_err[ik,ig_ref,ia_ref]:.2f} dec")

# ── Plot: 2×2 heatmaps ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Inferring cost ratio k from operating points alone\n"
             "(AUC$_{max}$ = 0.99, Gaussian classifiers)",
             fontsize=13, fontweight="bold")

log_k = np.log10(true_k_values)

for col, (metric, mtitle, cmax, cmap_name) in enumerate([
    (res_logwidth, "Interval width (decades)", 8, "RdYlGn_r"),
    (res_midpt_err, "Midpoint error |log₁₀(k̂/k)| (decades)", 8, "RdYlGn_r"),
]):
    for row, (ig, n_gen) in enumerate([(1, 5), (2, 10)]):
        ax = axes[row, col]
        hm = metric[:, ig, :].T  # (auc_min, k)

        im = ax.imshow(hm, aspect="auto", origin="lower", cmap=cmap_name,
                       vmin=0, vmax=cmax,
                       extent=[log_k[0] - 0.5, log_k[-1] + 0.5,
                               -0.5, len(auc_min_values) - 0.5])

        # Tick labels
        tick_k = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        ax.set_xticks(tick_k)
        ax.set_xticklabels([f"$10^{{{t}}}$" for t in tick_k], fontsize=7)
        ax.set_yticks(range(len(auc_min_values)))
        ax.set_yticklabels([f"{a:.2f}" for a in auc_min_values], fontsize=8)

        ax.set_xlabel("True k = c$_{FP}$/c$_{FN}$", fontsize=10)
        ax.set_ylabel("AUC$_{min}$", fontsize=10)
        ax.set_title(f"{mtitle}\n({n_gen} generations)", fontsize=10,
                     fontweight="bold")

        # Cell annotations
        for ia_idx in range(len(auc_min_values)):
            for ik_idx in range(len(true_k_values)):
                val = hm[ia_idx, ik_idx]
                ax.text(log_k[ik_idx], ia_idx, f"{val:.1f}",
                        ha="center", va="center", fontsize=4.5,
                        color="white" if val > cmax * 0.6 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8, label="decades")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(_HERE, "revealed_preference_simulation.png"),
            dpi=150, bbox_inches="tight")
print("\nSaved revealed_preference_simulation.png")
plt.close()
print("Done.")
