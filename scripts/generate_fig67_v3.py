"""
Figures 6 + 7 v3 — Nature-grade.
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from nature_style import apply_style, panel_label, style_axes
apply_style()

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"
OUT  = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\figures_v2"
# cycle 75: dual-write to v45_CURRENT/figures/ to fix supp.pdf sync.
OUT2 = r"C:\Users\Jie\Desktop\Bio-LM_v45_CURRENT\figures"

# ============================================================
# Figure 6 — SCB-suppress on Pythia-1B (v2 verified data, 2026-04-30)
# Source: 12_data_aggregates/scb_suppress_v2/*.json (6 seeds, lr=1e-5, alpha=1.0, 3000 steps)
# Replaces previous fig6_ft.npz (lr=5e-7 too small, FT failed to suppress)
# ============================================================
import json, glob
V2_DIR = os.path.join(DATA, "scb_suppress_v2")
v2_runs = []
for fp in sorted(glob.glob(os.path.join(V2_DIR, "*.json"))):
    with open(fp) as f:
        v2_runs.append(json.load(f))
print(f"Fig 6 v2: loaded {len(v2_runs)} seeds")

baseline_n_col = np.array([r["baseline_scb"]["n_col"] for r in v2_runs])
final_n_col    = np.array([r["final_scb"]["n_col"]    for r in v2_runs])
baseline_l1l2  = np.array([r["baseline_scb"]["max_lambda_ratio"] for r in v2_runs])
final_l1l2     = np.array([r["final_scb"]["max_lambda_ratio"]    for r in v2_runs])
baseline_pr    = np.array([r["baseline_scb"]["all_pr"] for r in v2_runs])  # (6, 16)
final_pr       = np.array([r["final_scb"]["all_pr"]    for r in v2_runs])

print(f"  baseline n_col: {baseline_n_col.tolist()}, final: {final_n_col.tolist()}")
print(f"  baseline l1l2:  {baseline_l1l2.mean():.1f}, final: {final_l1l2.mean():.2f}")

fig6 = plt.figure(figsize=(7.2, 5.5))
gs6 = GridSpec(2, 2, figure=fig6, hspace=0.42, wspace=0.32,
               left=0.09, right=0.97, top=0.92, bottom=0.10)

baseline_color = "#a04030"   # warm = SCB present
final_color    = "#3050a0"   # cool = SCB destroyed

# (a) Per-layer PR profile: baseline vs final (the structural evidence)
ax6a = fig6.add_subplot(gs6[0, 0])
style_axes(ax6a)
panel_label(ax6a, "a", x=-0.18, y=1.06)
n_layers = baseline_pr.shape[1]
xs = np.arange(n_layers)
# Per-seed faint lines
for i in range(len(v2_runs)):
    ax6a.plot(xs, baseline_pr[i], '-', color=baseline_color, lw=0.5, alpha=0.30)
    ax6a.plot(xs, final_pr[i],    '-', color=final_color,    lw=0.5, alpha=0.30)
# Mean ± std heavy lines
ax6a.plot(xs, baseline_pr.mean(0), '-o', color=baseline_color, lw=1.6, markersize=4,
          label=f"baseline (pre-FT)", zorder=5)
ax6a.fill_between(xs, baseline_pr.mean(0)-baseline_pr.std(0),
                  baseline_pr.mean(0)+baseline_pr.std(0),
                  color=baseline_color, alpha=0.15)
ax6a.plot(xs, final_pr.mean(0), '-s', color=final_color, lw=1.6, markersize=4,
          label=f"after spectral SCB-suppress", zorder=5)
ax6a.fill_between(xs, final_pr.mean(0)-final_pr.std(0),
                  final_pr.mean(0)+final_pr.std(0),
                  color=final_color, alpha=0.15)
ax6a.axhline(5, color='#d62728', ls='--', lw=0.7, alpha=0.7,
             label="PR=5 threshold")
ax6a.set_yscale('log')
ax6a.set_xlabel("layer index $\\ell$")
ax6a.set_ylabel("participation ratio PR$_\\ell$")
# Legend lifted ABOVE axes so it doesn't sit on top of the PR<5 dashed
# threshold line (the centred placement covered the thing it was labelling).
ax6a.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02),
            ncol=3, fontsize=6, frameon=False,
            handlelength=1.4, handletextpad=0.5, columnspacing=1.4)

def paired_seedplot(ax, x_left, x_right, vals_left, vals_right,
                    color_left, color_right, jitter=0.18):
    """Slope plot for paired pre/post seed values.

    The 6 SCB-suppress seeds converge so tightly (std≈0) that without
    enough x-jitter the per-seed lines collapse into a single line and
    the reader cannot tell n=6 from n=1. We use larger jitter (±0.18),
    semi-transparent dots so overlap forms a darker bundle, and an
    explicit \"n=6 seeds, std≈0\" annotation so low variance reads as
    high reproducibility (not as a missing data issue).
    """
    n = len(vals_left)
    rng = np.random.default_rng(0)
    # Pair-preserving jitter: each seed gets the SAME x-jitter on both
    # sides so its line is roughly vertical (matches semantic of "paired")
    jx = rng.uniform(-jitter, jitter, n)
    jx_l = x_left + jx
    jx_r = x_right + jx
    # Per-seed connecting lines (semi-transparent so 6 stack into a band)
    for i in range(n):
        ax.plot([jx_l[i], jx_r[i]], [vals_left[i], vals_right[i]],
                '-', color='#666', lw=1.4, alpha=0.35, zorder=2)
    # Per-seed dots (semi-transparent)
    ax.scatter(jx_l, vals_left, s=70, color=color_left,
               edgecolor='black', linewidth=0.5, alpha=0.55, zorder=4)
    ax.scatter(jx_r, vals_right, s=70, color=color_right,
               edgecolor='black', linewidth=0.5, alpha=0.55, zorder=4)
    # Mean line (thick black, opaque)
    ax.plot([x_left, x_right], [vals_left.mean(), vals_right.mean()],
            '-', color='black', lw=2.4, alpha=0.95, zorder=5)
    # Mean markers (diamond, opaque, larger)
    ax.scatter([x_left, x_right], [vals_left.mean(), vals_right.mean()],
               s=90, color=[color_left, color_right],
               edgecolor='black', linewidth=1.2, zorder=6,
               marker='D')
    # n + std annotation removed — was overlapping with post-FT dot
    # cluster at y≈0; the "6/6 seeds" detail is already in each panel's
    # top annotation, and the visible 6-dot bundle conveys low variance
    # without needing a numerical std reminder.

# (b) n_col paired seed plot (6 seeds tracked individually + mean line)
ax6b = fig6.add_subplot(gs6[0, 1])
style_axes(ax6b)
panel_label(ax6b, "b", x=-0.18, y=1.06)
positions = [0, 1]
paired_seedplot(ax6b, 0, 1, baseline_n_col, final_n_col,
                baseline_color, final_color)
ax6b.set_xticks(positions)
ax6b.set_xticklabels(["pre-FT", "post-FT"])
ax6b.set_xlim(-0.4, 1.4)
ax6b.set_ylabel(r"$n_{\mathrm{col}}$  (SCB length)")
ax6b.text(0.5, 0.92,
          f"$\\Delta n_{{\\mathrm{{col}}}} = -11$  (6/6 seeds)",
          transform=ax6b.transAxes, ha='center', fontsize=7,
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=2))
ax6b.set_ylim(-0.5, 14)

# (c) λ₁/λ₂ paired seed plot (log scale)
ax6c = fig6.add_subplot(gs6[1, 0])
style_axes(ax6c)
panel_label(ax6c, "c", x=-0.18, y=1.06)
paired_seedplot(ax6c, 0, 1, baseline_l1l2, final_l1l2,
                baseline_color, final_color)
ax6c.set_yscale('log')
ax6c.set_xticks(positions)
ax6c.set_xticklabels(["pre-FT", "post-FT"])
ax6c.set_xlim(-0.4, 1.4)
ax6c.set_ylabel(r"max $\lambda_1/\lambda_2$  (SCB band)")
# Compute exact drop ratio (true value 682.6/2.40 ≈ 284.4×). Round to
# nearest 5 to match paper main.tex "~285×" wording. Was hard-coded
# "~300×" before — off by ~5%.
_l1l2_ratio = baseline_l1l2.mean() / final_l1l2.mean()
_l1l2_ratio_round5 = int(round(_l1l2_ratio / 5.0) * 5)
# cycle 75 (PI feedback for supp Fig S4c): white text-box was sitting at
# ha='center' x=0.5 and overlapping the pre-FT high cluster around y≈600.
# Move right to x=0.97 with right-anchor; the upper-right corner is empty
# (post-FT values collapse to ~2 in lower-right, pre-FT values cluster
# near upper-left at x=0).
ax6c.text(0.97, 0.92,
          f"{baseline_l1l2.mean():.0f} $\\to$ {final_l1l2.mean():.1f}  "
          f"($\\sim${_l1l2_ratio_round5}$\\times$ drop)",
          transform=ax6c.transAxes, ha='right', fontsize=7,
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=2))

# (d) Spectral penalty paired seed plot
ax6d = fig6.add_subplot(gs6[1, 1])
style_axes(ax6d)
panel_label(ax6d, "d", x=-0.18, y=1.06)
spec_first = np.array([r["spec_penalty_first50"] for r in v2_runs])
spec_last  = np.array([r["spec_penalty_last50"]  for r in v2_runs])
paired_seedplot(ax6d, 0, 1, spec_first, spec_last,
                baseline_color, final_color)
ax6d.set_xticks(positions)
ax6d.set_xticklabels(["first 50 steps", "last 50 steps"])
ax6d.set_xlim(-0.4, 1.4)
ax6d.set_ylabel(r"spectral penalty  $\lambda_1\,/\,\sum_j\lambda_j$")
ax6d.text(0.5, 0.92,
          f"{spec_first.mean():.2f} $\\to$ {spec_last.mean():.3f}",
          transform=ax6d.transAxes, ha='center', fontsize=7,
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=2))

# NMI convention: figure-number sentence + asymmetric-control caveat in
# LaTeX caption only; figure carries no internal headers.
for out_dir in (OUT, OUT2):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Figure_6_BiDirControl.png"), dpi=600)
    plt.savefig(os.path.join(out_dir, "Figure_6_BiDirControl.pdf"))
    plt.savefig(os.path.join(out_dir, "Figure_6_BiDirControl.svg"))
print(f"Fig 6 v3 (verified) saved ({os.path.getsize(os.path.join(OUT, 'Figure_6_BiDirControl.png'))//1024} KB)")
plt.close(fig6)

# ============================================================
# Figure 7 — Pfam Application B
# ============================================================
pf = np.load(os.path.join(DATA, "fig7_pfam.npz"), allow_pickle=True)
keys = list(pf["keys"])
def short(m): return m.split("/")[-1].replace("progen2-", "")
data = defaultdict(dict)
for i, (m, k) in enumerate(keys):
    data[short(str(m))][int(k)] = {
        "scb": float(pf["scb_means"][i]),
        "scb_std": float(pf["scb_stds"][i]),
        "full": float(pf["full_means"][i]),
        "random": float(pf["random_means"][i]),
        "n": int(pf["n_seeds"][i]),
    }
scales = ["small","medium","base","large","xlarge"]
scale_ns = {"small":1.5e8, "medium":7.6e8, "base":2.7e9, "large":7.8e9, "xlarge":6.4e9}

fig7 = plt.figure(figsize=(7.2, 5.5))
gs7 = GridSpec(2, 2, figure=fig7, hspace=0.42, wspace=0.32,
               left=0.08, right=0.97, top=0.94, bottom=0.10)

# (a) K=10 across 5 scales
ax7a = fig7.add_subplot(gs7[0, 0])
style_axes(ax7a)
panel_label(ax7a, "a", x=-0.16, y=1.06)
xpos = np.arange(len(scales))
scbs, scb_e, fulls, rands, chs = [], [], [], [], []
for sc in scales:
    if sc in data and 10 in data[sc]:
        d = data[sc][10]
        scbs.append(d["scb"]); scb_e.append(d["scb_std"])
        fulls.append(d["full"]); rands.append(d["random"]); chs.append(0.10)
    else:
        scbs.append(0); scb_e.append(0); fulls.append(0); rands.append(0); chs.append(0.10)
w = 0.20
ax7a.bar(xpos - 1.5*w, fulls, w, color="#9467bd", alpha=0.85,
         label="full state ($D$-dim)", edgecolor='black', linewidth=0.4)
ax7a.bar(xpos - 0.5*w, scbs, w, yerr=scb_e, capsize=1.5,
         color="#d62728", alpha=0.92, label="1D SCB amplitude",
         edgecolor='black', linewidth=0.4, error_kw={"linewidth":0.6})
ax7a.bar(xpos + 0.5*w, rands, w, color="#7f7f7f", alpha=0.7,
         label="random direction", edgecolor='black', linewidth=0.4)
ax7a.bar(xpos + 1.5*w, chs, w, color="#cccccc", alpha=0.7,
         label="chance (1/$K$)", edgecolor='black', linewidth=0.4)
ax7a.set_xticks(xpos)
ax7a.set_xticklabels([f"PG2-{s}" for s in scales], rotation=20, ha='right', fontsize=7)
ax7a.set_ylabel(r"Pfam $K$=10 accuracy")
ax7a.legend(loc='upper left', fontsize=6, ncol=2)
ax7a.set_ylim(0, 1.12)

# (b) 3-K at PG2-large
ax7b = fig7.add_subplot(gs7[0, 1])
style_axes(ax7b)
panel_label(ax7b, "b", x=-0.16, y=1.06)
ks = [5, 10, 20]
xpos = np.arange(len(ks))
m = "large"
if m in data:
    scbs = [data[m][k]["scb"] if k in data[m] else 0 for k in ks]
    scb_e = [data[m][k]["scb_std"] if k in data[m] else 0 for k in ks]
    fulls = [data[m][k]["full"] if k in data[m] else 0 for k in ks]
    rands = [data[m][k]["random"] if k in data[m] else 0 for k in ks]
    chs = [1.0/k for k in ks]
    w = 0.20
    ax7b.bar(xpos - 1.5*w, fulls, w, color="#9467bd", alpha=0.85, label="full")
    ax7b.bar(xpos - 0.5*w, scbs, w, yerr=scb_e, capsize=1.5,
             color="#d62728", alpha=0.92, label="1D SCB",
             edgecolor='black', linewidth=0.4, error_kw={"linewidth":0.6})
    ax7b.bar(xpos + 0.5*w, rands, w, color="#7f7f7f", alpha=0.7, label="random")
    ax7b.bar(xpos + 1.5*w, chs, w, color="#cccccc", alpha=0.7, label="chance")
    ax7b.set_xticks(xpos)
    ax7b.set_xticklabels([f"$K$={k}" for k in ks])
    ax7b.set_ylabel("accuracy (PG2-large)")
    ax7b.legend(loc='upper right', fontsize=6.5)

# (c) above-chance information ratio
ax7c = fig7.add_subplot(gs7[1, 0])
style_axes(ax7c)
panel_label(ax7c, "c", x=-0.16, y=1.06)
k_colors = {5: "#1f77b4", 10: "#d62728", 20: "#2ca02c"}
for k in ks:
    fracs = []
    for sc in scales:
        if sc in data and k in data[sc]:
            f = data[sc][k]["full"]; s = data[sc][k]["scb"]; ch = 1.0/k
            fracs.append(((s - ch) / (f - ch)) if f > ch else 0)
        else:
            fracs.append(0)
    ax7c.plot(scales, fracs, '-o', color=k_colors[k], lw=1.4,
              markersize=6, markeredgecolor='black', markeredgewidth=0.4,
              label=f"$K$={k}")
ax7c.set_xticklabels([f"PG2-{s}" for s in scales], rotation=20, ha='right', fontsize=7)
ax7c.set_ylabel(r"$(\mathrm{acc}_\mathrm{SCB}-\mathrm{chance})\,/\,(\mathrm{acc}_\mathrm{full}-\mathrm{chance})$")
ax7c.set_xlabel("ProGen2 scale")
ax7c.axhline(0.2, color='#aaa', ls=':', lw=0.5)
ax7c.legend(loc='upper right', fontsize=6.5)
ax7c.set_ylim(-0.05, 0.5)

# (d) above-chance margin: SCB vs random at K=10
ax7d = fig7.add_subplot(gs7[1, 1])
style_axes(ax7d)
panel_label(ax7d, "d", x=-0.16, y=1.06)
margins_scb = []; margins_rand = []
for sc in scales:
    if sc in data and 10 in data[sc]:
        margins_scb.append(data[sc][10]["scb"] - 0.1)
        margins_rand.append(data[sc][10]["random"] - 0.1)
    else:
        margins_scb.append(0); margins_rand.append(0)
xpos = np.arange(len(scales))
ax7d.bar(xpos - 0.18, margins_scb, 0.36, color="#d62728", alpha=0.92,
         label="1D SCB", edgecolor='black', linewidth=0.4)
ax7d.bar(xpos + 0.18, margins_rand, 0.36, color="#7f7f7f", alpha=0.75,
         label="random direction", edgecolor='black', linewidth=0.4)
ax7d.axhline(0, color='black', lw=0.6)
ax7d.set_xticks(xpos)
ax7d.set_xticklabels([f"PG2-{s}" for s in scales], rotation=20, ha='right', fontsize=7)
ax7d.set_ylabel(r"accuracy − chance ($K$=10)")
ax7d.legend(loc='upper right', fontsize=6.5)

for out_dir in (OUT, OUT2):
    plt.savefig(os.path.join(out_dir, "Figure_7_AppB_Pfam.png"), dpi=600)
    plt.savefig(os.path.join(out_dir, "Figure_7_AppB_Pfam.pdf"))
    plt.savefig(os.path.join(out_dir, "Figure_7_AppB_Pfam.svg"))
print(f"Fig 7 v3 saved ({os.path.getsize(os.path.join(OUT, 'Figure_7_AppB_Pfam.png'))//1024} KB)")
plt.close(fig7)

print("\nAll 5 v3 figures done.")
