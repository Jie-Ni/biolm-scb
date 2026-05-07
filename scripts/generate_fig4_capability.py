"""Generate Figure 4: capability prediction panel (a-d per-task scatter + e bar).

Data: data_v46_collected/capability_prediction.csv (41 models x 8 benchmarks).
Output: figures/Figure_4_Capability.{pdf,png,svg}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data_v46_collected" / "capability_prediction.csv"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# NOTE (2026-05-06, lin_mei integrity fix):
# All R^2 values (panels a-d AND panel e) are now computed DIRECTLY from
# capability_prediction.csv via numpy least-squares fits. Previously panels
# read from capability_prediction_report.json which used a smaller n-basis
# and produced figure-text inconsistencies (e.g. WinoGrande +0.14 in panel e
# vs +0.09 in caption). report.json is no longer consulted.


def family_of(name: str) -> str:
    n = name.lower()
    for fam in [
        "pythia", "bloom", "opt", "gpt-neo", "gpt-2", "gpt2", "olmo2", "olmo",
        "smollm2", "smollm", "falcon", "tinyllama", "redpajama",
        "qwen", "mistral", "phi", "stablelm",
    ]:
        if fam.replace("-", "") in n.replace("-", "").replace("_", ""):
            return fam
    return "other"


# ---- load data ----
df = pd.read_csv(DATA_CSV)

# Identify columns
n_col_candidates = ["n_params", "param_count", "N", "num_params", "params"]
n_col = next((c for c in n_col_candidates if c in df.columns), df.columns[1])
name_col = next((c for c in ["tag", "model", "model_name", "name", "id"] if c in df.columns), df.columns[0])

bench_cols = [b for b in ["arc_challenge", "hellaswag", "mmlu", "winogrande", "gsm8k", "lambada", "piqa", "sciq"] if b in df.columns]

df["family"] = df[name_col].astype(str).map(family_of)
df["log_N"] = np.log10(df[n_col].astype(float).clip(lower=1.0))

# Family palette (Tab10-ish)
family_colors = {
    "pythia": "#440154",
    "bloom": "#1f77b4",
    "opt": "#ff7f0e",
    "gpt-neo": "#2ca02c",
    "gpt-2": "#8c564b",
    "gpt2": "#8c564b",
    "olmo": "#e377c2",
    "olmo2": "#d62728",
    "smollm": "#17becf",
    "smollm2": "#17becf",
    "falcon": "#bcbd22",
    "tinyllama": "#9467bd",
    "redpajama": "#7f7f7f",
    "qwen": "#0d9488",
    "mistral": "#dc2626",
    "phi": "#a16207",
    "stablelm": "#facc15",
    "other": "#374151",
}

# ---- panel a-d: per-task scatter (PIQA, SciQ, HellaSwag, WinoGrande) ----
panel_tasks = ["piqa", "sciq", "hellaswag", "winogrande"]
panel_tasks = [t for t in panel_tasks if t in df.columns][:4]

# ---- CSV-direct R^2 (univariate logN baseline + multivariate logN+alpha+log chi_S) ----
# chi_S = chi_sigma_range (singular-value spread). The paper caption / Section 4
# numbers correspond to this combination (log N + alpha + log chi_S), n = full
# CSV rows per task with dropna. chi_seed_var (chi_V) gives different gains and
# is NOT the canonical predictor in v45/v46.
ALPHA_COL = "alpha_mean"
CHIS_COL = "chi_sigma_range"


def _r2_ols(X: np.ndarray, y: np.ndarray) -> float:
    """OLS R^2 with intercept; X shape (n, k), y shape (n,)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def r2_logN(task: str) -> float:
    sub = df[["log_N", task]].dropna()
    if len(sub) < 3:
        return float("nan")
    return _r2_ols(sub[["log_N"]].values, sub[task].values)


def r2_full(task: str) -> float:
    """Multivariate R^2: log_N + alpha_mean + log10(chi_sigma_range)."""
    cols = ["log_N", ALPHA_COL, CHIS_COL, task]
    sub = df[cols].dropna().copy()
    sub = sub[sub[CHIS_COL] > 0]
    if len(sub) < 4:
        return float("nan")
    sub["log_chiS"] = np.log10(sub[CHIS_COL].astype(float))
    X = sub[["log_N", ALPHA_COL, "log_chiS"]].values
    return _r2_ols(X, sub[task].values)


# ---- multivariate gain bar (CSV-direct) ----
benches_for_e = ["piqa", "lambada", "hellaswag", "winogrande", "arc_challenge", "mmlu", "sciq", "gsm8k"]
delta_table = []
for b in benches_for_e:
    if b not in df.columns:
        continue
    r0 = r2_logN(b)
    r1 = r2_full(b)
    if np.isnan(r0) or np.isnan(r1):
        continue
    delta_table.append({
        "benchmark": b,
        "R2_logN": r0,
        "R2_logN+alpha+chiS": r1,
        "delta": r1 - r0,
    })


# ---- figure ----
fig = plt.figure(figsize=(13.0, 9.5), dpi=150)
gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.40, height_ratios=[1.0, 1.0, 1.1])

# panels a-d
labels = ["a", "b", "c", "d"]
for i, task in enumerate(panel_tasks):
    ax = fig.add_subplot(gs[i // 2, (i % 2) * 2:(i % 2) * 2 + 2])
    sub = df[["log_N", task, "family", name_col]].dropna()
    for fam, grp in sub.groupby("family"):
        ax.scatter(grp["log_N"], grp[task], s=55,
                   c=family_colors.get(fam, "#374151"), edgecolor="white",
                   linewidth=0.6, label=fam, zorder=3)
    if len(sub) >= 2:
        slope, intercept = np.polyfit(sub["log_N"], sub[task], 1)
        xr = np.linspace(sub["log_N"].min(), sub["log_N"].max(), 100)
        ax.plot(xr, slope * xr + intercept, "k--", lw=1.2, alpha=0.6, zorder=2)
    r2 = r2_logN(task)
    ax.text(0.04, 0.95, f"({labels[i]}) {task}\n$R^2={r2:.2f}$ ($\\log N$)",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))
    ax.set_xlabel(r"$\log_{10} N$ (parameters)", fontsize=10)
    ax.set_ylabel(f"{task} accuracy", fontsize=10)
    ax.grid(alpha=0.25)

# panel e: multivariate R^2 gain bar
ax_e = fig.add_subplot(gs[2, :])
order = [d for d in delta_table if d["benchmark"] in benches_for_e]
order = sorted(order, key=lambda d: -d["delta"])
labels_e = [d["benchmark"] for d in order]
r2_logN_vals = [d["R2_logN"] for d in order]
r2_full_vals = [d["R2_logN+alpha+chiS"] for d in order]
x = np.arange(len(labels_e))
w = 0.38
ax_e.bar(x - w / 2, r2_logN_vals, w, color="#9ca3af", edgecolor="black", lw=0.6,
         label=r"$\log N$ only")
ax_e.bar(x + w / 2, r2_full_vals, w, color="#dc2626", edgecolor="black", lw=0.6,
         label=r"$\log N + \alpha + \log\chi_\mathrm{S}$")
for i, d in enumerate(order):
    delta = d["delta"]
    ax_e.text(x[i] + w / 2, r2_full_vals[i] + 0.015, f"+{delta:.2f}",
              ha="center", fontsize=9, color="#dc2626")
ax_e.set_xticks(x)
ax_e.set_xticklabels(labels_e, rotation=12, fontsize=10)
ax_e.set_ylabel(r"$R^2$", fontsize=11)
ax_e.set_title(r"(e) Spectral covariates raise $R^2$ over $\log N$ baseline (41 models, 8 benchmarks)",
               fontsize=11, loc="left")
ax_e.set_ylim(0, max(max(r2_full_vals) + 0.15, 1.0))
ax_e.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax_e.grid(axis="y", alpha=0.25)

# tight legend across the figure
handles, labels_h = [], []
for fam, color in family_colors.items():
    if (df["family"] == fam).any():
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
                                  markeredgecolor="white", markersize=8))
        labels_h.append(fam)
fig.legend(handles, labels_h, loc="upper center", ncol=min(len(handles), 9),
           bbox_to_anchor=(0.5, 1.005), fontsize=9, frameon=False)

fig.suptitle("Figure 4. Spectral predictors of in-context-learning capability "
             "across 41 publicly released LMs", y=1.04, fontsize=13)

# save
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT_DIR / f"Figure_4_Capability.{ext}", bbox_inches="tight")
print(f"saved to {OUT_DIR / 'Figure_4_Capability.{pdf,png,svg}'}")
print("R^2 values:")
for t in panel_tasks:
    print(f"  {t}: log_N R^2 = {r2_logN(t):.3f}")
print("delta table preview:")
for d in order:
    print(f"  {d['benchmark']}: logN={d['R2_logN']:.3f} -> full={d['R2_logN+alpha+chiS']:.3f} (+{d['delta']:.3f})")
