"""
Figure 4 v3 — Nature-grade redo.
4 sub-panels (a/b/c/d) at proper Nature-figure styling.
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from nature_style import (apply_style, FAMILY_COLORS, PYTHIA_VIRIDIS,
                          panel_label, style_axes, pretty_name)
apply_style()

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"
OUT  = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\figures_v2"
# cycle 75: dual-write to v45_CURRENT/figures/ to fix supp.pdf sync.
OUT2 = r"C:\Users\Jie\Desktop\Bio-LM_v45_CURRENT\figures"

def load_iff(path):
    d = np.load(path, allow_pickle=True)
    out = {}
    for i, (tag, abl) in enumerate(d["keys"]):
        out[(str(tag), str(abl))] = {
            "R": list(d["R_lists"][i]),
            "ncol": float(d["ncol_means"][i]),
        }
    return out

iff = load_iff(os.path.join(DATA, "fig4_iff_lnz.npz"))
inn = load_iff(os.path.join(DATA, "fig4_iff_inn.npz"))
combined = defaultdict(lambda: {"R": [], "ncol": None})
for src in (iff, inn):
    for k, v in src.items():
        combined[k]["R"].extend(v["R"])
        if combined[k]["ncol"] is None:
            combined[k]["ncol"] = v["ncol"]

def clean(rs):
    return [r for r in rs if np.isfinite(r) and abs(r) < 1000]

def family(tag):
    if tag.startswith("py") and not tag.startswith("phi"): return "Pythia"
    if tag.startswith("bloom"): return "BLOOM"
    if tag.startswith("qwen") and "moe" in tag: return "MoE"
    if tag.startswith("qwen"): return "Qwen2.5"
    if tag.startswith("neo") or tag.startswith("gptneo"): return "GPT-Neo"
    if "mistral" in tag: return "Mistral"
    if tag.startswith("olmoe"): return "MoE"
    if tag.startswith("olmo"): return "OLMo"
    return "Other"

# ---- Build records ----
# cycle 74 fix: two separate panels with distinct R metrics.
#   SCB-bearing (17, paper-canonical): R = projection-out ratio,
#       ΔNLL_ICL_proj / ΔNLL_ZS_proj. Used for the regression.
#   IFF controls (5, n_col=0): R = baseline few-shot/zero-shot ratio
#       (no projection). Projection-out R is 0/0 undefined for SCB-free
#       models. NOT included in the Pearson r regression — shown as
#       distinct black diamonds, qualitative IFF confirmation.
SCB_PANEL = ["py70m","py160m","py410m","py1b","py14b","py28b","py69b","py12b",
             "bloom_3b","bloom7b","bloom_560m",
             "mistral7b","stablelm","tinyllama","smol_17","phi15","phi1"]  # 17 canonical
IFF_PANEL = ["bloom_1b1","qwen_05","qwen_15","neo_27b","olmo7b"]  # 5 ncol=0 controls

records, iff_records = [], []
for tag in SCB_PANEL:
    if (tag, "leading") not in combined: continue
    rs = clean(combined[(tag, "leading")]["R"])
    if len(rs) < 3: continue
    ncol = combined[(tag, "leading")]["ncol"]
    records.append({"tag": tag, "ncol": ncol, "R_mean": float(np.mean(rs)),
                    "R_std": float(np.std(rs)), "R_n": len(rs),
                    "R_lo": float(np.percentile(rs, 2.5)),
                    "R_hi": float(np.percentile(rs, 97.5)),
                    "fam": family(tag), "metric": "proj_out"})
for tag in IFF_PANEL:
    if (tag, "none") not in combined: continue
    rs = clean(combined[(tag, "none")]["R"])
    if len(rs) < 3: continue
    ncol = combined[(tag, "none")]["ncol"]
    iff_records.append({"tag": tag, "ncol": ncol, "R_mean": float(np.mean(rs)),
                        "R_std": float(np.std(rs)), "R_n": len(rs),
                        "fam": family(tag), "metric": "baseline"})
print(f"Panel: {len(records)} SCB-bearing (proj_out R) + {len(iff_records)} IFF controls (baseline R)")

n_cols  = np.array([r["ncol"]    for r in records])
R_means = np.array([r["R_mean"]  for r in records])
R_stds  = np.array([r["R_std"]   for r in records])
R_n     = np.array([r["R_n"]     for r in records])
tags    = [r["tag"] for r in records]

iff_n_cols = np.array([r["ncol"] for r in iff_records])
iff_R_means = np.array([r["R_mean"] for r in iff_records])
iff_R_stds  = np.array([r["R_std"] for r in iff_records])
iff_tags = [r["tag"] for r in iff_records]

# Pearson on SCB-bearing only (paper-canonical) — IFF cluster has different
# metric so combining would mix proj_out and baseline R.
r_p, p_p = stats.pearsonr(n_cols, R_means)
print(f"Pearson r = {r_p:.3f}, p = {p_p:.2e}, n = {len(records)} (SCB-bearing only)")
print(f"IFF cluster R values: {[f'{r:.3f}' for r in iff_R_means]}")

# ============================================================
# Figure layout: 2-column wide, ~180mm x 130mm
# ============================================================
fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=False)
gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32,
              left=0.08, right=0.98, top=0.95, bottom=0.10)

# ============================================================
# (a) BUBBLE chart — n_col vs R, bubble size = log(parameter count N),
#     color = family. Adds a third encoded dimension (model size) so
#     reader can see scaling pattern within each family at a glance.
# ============================================================
ax_a = fig.add_subplot(gs[0, 0])
style_axes(ax_a)
panel_label(ax_a, "a", x=-0.18, y=1.06)

# Tag → parameter count (M params), used to set bubble area
TAG_TO_N = {
    "py70m": 7e7, "py160m": 1.6e8, "py410m": 4.1e8, "py1b": 1e9,
    "py14b": 1.4e9, "py28b": 2.8e9, "py69b": 6.9e9, "py12b": 1.2e10,
    "bloom_560m": 5.6e8, "bloom_1b1": 1.1e9, "bloom_3b": 3e9, "bloom7b": 7e9,
    "mistral7b": 7e9, "mistral7": 7e9,
    "stablelm": 1.6e9, "tinyllama": 1.1e9, "smol_17": 1.7e9,
    "phi1": 1.3e9, "phi15": 1.4e9,
    "qwen_05": 5e8, "qwen_15": 1.5e9, "qwen_3": 3e9, "qwen_7": 7e9,
    "neo_27b": 2.7e9, "olmo7b": 7e9, "olmoe_1b_7b": 7e9,
}
def bubble_area(tag, default=1e9):
    n = TAG_TO_N.get(tag, default)
    # cycle 75 (PI feedback): bubbles were "太大互相重叠"; shrink range
    # from [40, 350] → [25, 180]. Visible scaling cue, but no longer
    # mass-overlapping at high-density n_col regions.
    s = 25 + (np.log10(n) - 7.8) / (10.1 - 7.8) * 155
    return float(np.clip(s, 20, 200))

# Plot family-colored bubbles with size = log(N)
for fam in FAMILY_COLORS:
    idx = [i for i, r in enumerate(records) if r["fam"] == fam]
    if not idx: continue
    sizes = [bubble_area(records[i]["tag"]) for i in idx]
    ax_a.errorbar([n_cols[i] for i in idx], [R_means[i] for i in idx],
                  yerr=[R_stds[i] for i in idx], fmt='none',
                  ecolor=FAMILY_COLORS[fam], alpha=0.45, capsize=1.5,
                  elinewidth=0.7, zorder=2)
    ax_a.scatter([n_cols[i] for i in idx], [R_means[i] for i in idx],
                 s=sizes, c=FAMILY_COLORS[fam], edgecolor='black',
                 linewidth=0.6, alpha=0.85, label=fam, zorder=3)

# Bubble-size legend (3 reference sizes for log N scale)
size_ref_N = [1e8, 1e9, 1e10]   # 100M, 1B, 10B
size_ref_lab = ["100M", "1B", "10B"]
size_ref_pos_x = [0.62, 0.74, 0.88]
for x, n, lab in zip(size_ref_pos_x, size_ref_N, size_ref_lab):
    # match shrunk bubble formula above
    s = 25 + (np.log10(n) - 7.8) / (10.1 - 7.8) * 155
    ax_a.scatter([], [], s=s, c='#bbb', edgecolor='black', linewidth=0.5,
                 label=f"$N$={lab}")

# IFF controls (n_col=0): black diamonds w/ family-color edges, slight x-jitter
# so the 5 overlapping points at n_col=0 stay visible.
np.random.seed(0)
iff_jitter = np.random.uniform(-0.25, 0.25, len(iff_n_cols))
for j, ir in enumerate(iff_records):
    ec = FAMILY_COLORS.get(ir["fam"], "#888")
    ax_a.errorbar(iff_n_cols[j] + iff_jitter[j], iff_R_means[j],
                  yerr=iff_R_stds[j], fmt='D', color='black',
                  markersize=7, markeredgecolor=ec, markeredgewidth=1.2,
                  ecolor=ec, alpha=0.95, capsize=1.5, elinewidth=0.7,
                  zorder=5)
# One legend entry for IFF cluster
ax_a.scatter([], [], s=55, marker='D', facecolor='black',
             edgecolor='#444', linewidth=1.0,
             label=r"IFF control ($n_{\mathrm{col}}\!=\!0$, $N=$" +
                   f"{len(iff_records)}, baseline $R$)")

# Best-fit (cycle 74: bumped lw 1.0 → 1.3, alpha 0.45 → 0.85 for print)
xfit = np.linspace(-0.5, max(n_cols)+1.5, 100)
slope, intc, _, _, _ = stats.linregress(n_cols, R_means)
ax_a.plot(xfit, slope*xfit + intc, '-', color='#666', lw=1.3, alpha=0.85, zorder=1)
ax_a.axhline(1.0, color='#888', ls=':', lw=0.6, zorder=1)

# Stats annotation: italic non-monospace, capital P/N (NMI typography)
# cycle 75: PI feedback — white box was covering one of the upper-left
# bubbles. Move stats annotation to bottom-right where there is no data.
ax_a.text(0.97, 0.04,
          f"SCB-bearing: $r$ = {r_p:.3f}\n$P$ = {p_p:.1e}, $N$ = {len(records)}\nIFF controls: $N$ = {len(iff_records)}, $R$ = 1.00",
          transform=ax_a.transAxes, va='bottom', ha='right',
          fontsize=6.5, style='italic',
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=3))
ax_a.set_xlabel(r"$n_{\mathrm{col}}$  (SCB length, layers)")
ax_a.set_ylabel(r"ICL amplification  $R$")
# Lift legend ABOVE axes so it doesn't sit on top of any data point
ax_a.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02),
            ncol=3, fontsize=6, columnspacing=1.4, frameon=False,
            handlelength=1.4, handletextpad=0.5)

# ============================================================
# (b) IFF zoom n_col∈[0,5]
# ============================================================
ax_b = fig.add_subplot(gs[0, 1])
style_axes(ax_b)
panel_label(ax_b, "b", x=-0.18, y=1.06)

# Lighter IFF shading (was 0.18 → 0.10) so the IFF n_col=0 markers
# don't disappear under the pink wash.
ax_b.axvspan(-0.5, 0.5, alpha=0.10, color='#fadcd9', zorder=0,
             label=r"$n_{\mathrm{col}}\!=\!0$ IFF cluster")
ax_b.axhline(1.0, color='#888', ls=':', lw=0.6)

# Plot SCB-bearing low-ncol models (open circles)
zoom = [r for r in records if r["ncol"] <= 5.5]
for r in zoom:
    col = FAMILY_COLORS[r["fam"]]
    ax_b.errorbar(r["ncol"], r["R_mean"], yerr=r["R_std"], fmt='o',
                  color=col, markersize=8, markeredgecolor='black',
                  markeredgewidth=0.6, ecolor=col, alpha=0.9, capsize=2,
                  elinewidth=0.7, zorder=3)
    ax_b.annotate(pretty_name(r["tag"]), (r["ncol"], r["R_mean"]),
                  xytext=(6, 4), textcoords='offset points',
                  fontsize=5.5, color='#222')

# Plot IFF cluster (5 black diamonds at n_col=0, R=1) with x-jitter so
# overlapping points stay visible. Stacked label on the right.
np.random.seed(0)
jitter = np.random.uniform(-0.18, 0.18, len(iff_records))
for j, ir in enumerate(iff_records):
    ec = FAMILY_COLORS.get(ir["fam"], "#888")
    ax_b.errorbar(0 + jitter[j], ir["R_mean"], yerr=ir["R_std"], fmt='D',
                  color='black', markersize=8, markeredgecolor=ec,
                  markeredgewidth=1.2, ecolor=ec, alpha=0.95, capsize=1.5,
                  elinewidth=0.7, zorder=5)
# IFF cluster info: name list with explicit family-edge mapping, but compact
# (one line) so it tucks into the empty lower-right corner without covering
# the SCB-bearing low-ncol data points (BLOOM-3B at n_col=2, R=1.55 etc.).
# Cluster names listed vertically along the right edge (one per line)
# — uses empty far-right space without covering any data point.
iff_lines = ["IFF cluster ($N$=5)",
             "(black diamond,",
             " edge = family colour)",
             ""] + [f"$\\bullet$ {pretty_name(ir['tag'])}"
                    for ir in iff_records]
ax_b.text(0.985, 0.985, "\n".join(iff_lines),
          transform=ax_b.transAxes,
          fontsize=5.0, color='#222', va='top', ha='right', linespacing=1.20,
          bbox=dict(facecolor='white', edgecolor='#aaa',
                    linewidth=0.3, pad=2))

ax_b.set_xlim(-0.6, 5.5)
ax_b.set_ylim(0.6, 1.7)
ax_b.set_xlabel(r"$n_{\mathrm{col}}$")
ax_b.set_ylabel(r"$R$")
# Match (a) legend style for consistency
ax_b.legend(loc='upper left', fontsize=6, frameon=False,
            handlelength=1.4, handletextpad=0.5)

# ============================================================
# (c) Bootstrap CI of Pearson r
# ============================================================
ax_c = fig.add_subplot(gs[1, 0])
style_axes(ax_c)
panel_label(ax_c, "c", x=-0.18, y=1.06)

np.random.seed(42)
n_boot = 5000
boot_rs = []
for _ in range(n_boot):
    idx = np.random.choice(len(n_cols), len(n_cols), replace=True)
    if len(np.unique(n_cols[idx])) < 2: continue
    rb, _ = stats.pearsonr(n_cols[idx], R_means[idx])
    boot_rs.append(rb)
boot_rs = np.array(boot_rs)
ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])

ax_c.hist(boot_rs, bins=50, color="#b3cde3", edgecolor="#3a4a6e",
          linewidth=0.4, alpha=0.95)
ax_c.axvline(r_p,    color='#d62728', lw=1.4, label=f"$r$ = {r_p:.3f}")
ax_c.axvline(ci_lo,  color='black',   lw=0.9, ls='--', label=f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
ax_c.axvline(ci_hi,  color='black',   lw=0.9, ls='--')
ax_c.axvline(0,      color='#888',    lw=0.5, ls=':', alpha=0.6)
ax_c.set_xlabel("Pearson $r$  (bootstrap, $n_{\\mathrm{boot}}$=5,000)")
ax_c.set_ylabel("frequency")
# cycle 75 (PI feedback): "r=0.836" + "95% CI" annotations at upper-left
# overlapped the bootstrap histogram bars (distribution mass spans 0.5–0.95).
# Move legend to upper-left BUT shifted into the empty x<0 region via
# bbox_to_anchor=(0.02, ...), AND add solid white bbox so any residual
# overlap with left-tail bars is hidden cleanly.
leg = ax_c.legend(loc='upper left', fontsize=6.5, frameon=True,
                  facecolor='white', edgecolor='#aaa', framealpha=1.0,
                  bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0,
                  handlelength=1.6, handletextpad=0.4)
leg.set_zorder(20)
ax_c.set_xlim(-0.1, 1.0)

# ============================================================
# (d) Per-ablation effect bars: leading vs 2nd vs 5th vs random
#     for top-6 SCB models
# ============================================================
ax_d = fig.add_subplot(gs[1, 1])
style_axes(ax_d)
panel_label(ax_d, "d", x=-0.18, y=1.06)

abls = ["leading", "second", "fifth", "random"]
abl_colors = {"leading": "#B64342", "second": "#E0A234",
              "fifth": "#0F4D92", "random": "#767676"}
top_models = sorted([(t, combined[(t, "leading")]["ncol"]) for t in SCB_PANEL
                     if (t, "leading") in combined],
                    key=lambda x: -x[1])[:6]
top_tags = [t for t, _ in top_models]
xpos = np.arange(len(top_tags))
w = 0.21

for j, abl in enumerate(abls):
    vals = []
    errs = []
    for t in top_tags:
        if (t, abl) in combined:
            # Tighter clean: drop |R|>5 to avoid wild outlier err bars
            rs = [r for r in combined[(t, abl)]["R"]
                  if np.isfinite(r) and -3 <= r <= 5]
            vals.append(np.mean(rs) if rs else 0.0)
            errs.append(np.std(rs) if len(rs) > 1 else 0.0)
        else:
            vals.append(0.0); errs.append(0.0)
    ax_d.bar(xpos + (j - 1.5)*w, vals, w, yerr=errs, capsize=1.5,
             color=abl_colors[abl], alpha=0.9, label=abl,
             edgecolor='black', linewidth=0.4,
             error_kw={"linewidth": 0.5, "alpha": 0.8})

ax_d.axhline(1.0, color='#888', ls=':', lw=0.6)
ax_d.set_xticks(xpos)
ax_d.set_xticklabels([pretty_name(t) for t in top_tags],
                     rotation=20, ha='right', fontsize=6.5)
ax_d.set_ylabel(r"$R$  (mean $\pm$ SD)")
ax_d.set_ylim(-0.5, 5)  # ground at -0.5 (small under-1 dip OK), drop -1
# Rename "fifth" → "5th" in legend to match NMI caption convention.
abl_label = {"leading": "leading", "second": "2nd", "fifth": "5th", "random": "random"}
handles, labels = ax_d.get_legend_handles_labels()
ax_d.legend(handles, [abl_label.get(l, l) for l in labels],
            loc='upper right', fontsize=6, ncol=2, frameon=False)

# NMI convention: figure-number sentence in LaTeX caption only.
for out_dir in (OUT, OUT2):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Figure_4_IFF.png"), dpi=600)
    plt.savefig(os.path.join(out_dir, "Figure_4_IFF.pdf"))
    plt.savefig(os.path.join(out_dir, "Figure_4_IFF.svg"))
print(f"Fig 4 saved → {OUT}/Figure_4_IFF.png and {OUT2}/Figure_4_IFF.png")
print(f"Size: {os.path.getsize(os.path.join(OUT, 'Figure_4_IFF.png'))//1024} KB")
