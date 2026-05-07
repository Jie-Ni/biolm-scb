"""
Figure 5 v3 — Nature-grade trajectory figure.
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from nature_style import apply_style, PYTHIA_VIRIDIS, panel_label, style_axes
apply_style()

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"
OUT  = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\figures_v2"
# cycle 75: dual-write to v45_CURRENT/figures/ so manuscript build picks up
# the latest version without an extra `cp` step (root cause of sun_hao
# sync issue: scripts wrote to figures_v2/ but build read from v45_CURRENT/figures/).
OUT2 = r"C:\Users\Jie\Desktop\Bio-LM_v45_CURRENT\figures"

traj = np.load(os.path.join(DATA, "fig5_traj.npz"), allow_pickle=True)
ncol = np.load(os.path.join(DATA, "fig5_ncol.npz"), allow_pickle=True)

data = defaultdict(dict)
for i, (s, st, abl) in enumerate(traj["keys"]):
    rs = [r for r in traj["R_lists"][i] if np.isfinite(r) and -5 <= r <= 10]
    if rs:
        data[(str(s), str(abl))][int(st)] = rs

ncol_d = {}
for i, (s, st) in enumerate(ncol["keys"]):
    ncol_d[(str(s), int(st))] = float(ncol["nco_means"][i])

scales_order = ["py70m","py160m","py410m","py1b","py14b","py28b","py69b"]
scale_params = {"py70m":7e7,"py160m":1.6e8,"py410m":4.1e8,"py1b":1e9,
                "py14b":1.4e9,"py28b":2.8e9,"py69b":6.9e9,"py12b":1.2e10}
scale_label = {"py70m":"Pythia-70M","py160m":"Pythia-160M","py410m":"Pythia-410M",
               "py1b":"Pythia-1B","py14b":"Pythia-1.4B","py28b":"Pythia-2.8B",
               "py69b":"Pythia-6.9B","py12b":"Pythia-12B"}

# ============================================================
fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=False)
gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32,
              left=0.08, right=0.97, top=0.94, bottom=0.10)

# ============================================================
# (a) Heatmap: R(scale, step) — replaces 7-line spaghetti plot
# ============================================================
ax_a = fig.add_subplot(gs[0, 0])
style_axes(ax_a)
panel_label(ax_a, "a", x=-0.18, y=1.06)

# Define log-spaced step bins covering [50, 2e5]
log_bin_edges = np.logspace(np.log10(50), np.log10(2e5), 21)
bin_centres = np.sqrt(log_bin_edges[:-1] * log_bin_edges[1:])

# Compute mature_R + N_lock first (still needed downstream)
mature_R = {}; N_lock = {}
heat = np.full((len(scales_order), len(bin_centres)), np.nan)
for i, s in enumerate(scales_order):
    if (s, "leading") not in data: continue
    sr = sorted(data[(s, "leading")].items())
    if len(sr) < 4: continue
    steps = np.array([x[0] for x in sr])
    Rs = np.array([np.mean(x[1]) for x in sr])
    if len(steps) >= 3:
        mature_R[s] = float(np.median(Rs[-min(5, len(Rs)):]))
        target = 0.5 * (mature_R[s] - 1) + 1
        for j, st in enumerate(steps):
            if Rs[j] >= target:
                N_lock[s] = int(st); break
    # Bin into log-step bins
    bin_idx = np.digitize(steps, log_bin_edges) - 1
    for b in range(len(bin_centres)):
        in_bin = bin_idx == b
        if in_bin.sum() > 0:
            heat[i, b] = float(np.mean(Rs[in_bin]))

# Plot heatmap (rows=scales sorted small→large, cols=log step bin)
# cycle 74: explicit interpolation='nearest' for crisp cell boundaries.
cmap = plt.get_cmap('RdYlBu_r')
im = ax_a.imshow(heat, aspect='auto', cmap=cmap,
                 vmin=0.5, vmax=3.5, origin='lower',
                 interpolation='nearest',
                 extent=[np.log10(50), np.log10(2e5),
                         -0.5, len(scales_order) - 0.5])
ax_a.set_yticks(np.arange(len(scales_order)))
ax_a.set_yticklabels([scale_label.get(s, s).replace("Pythia-", "") for s in scales_order],
                     fontsize=6)
# cycle 75 (PI feedback): pad y-tick labels right so "160M" no longer overlaps
# the y-axis spine; affects all Pythia-* tick labels uniformly.
ax_a.tick_params(axis='y', pad=8)
ax_a.set_xlabel("training step (log$_{10}$)")
ax_a.set_ylabel("Pythia scale")

# Phase markers as compact bracket annotations along the top edge of the heatmap
ax_a.axvline(np.log10(2000), color='#a04030', ls=':', lw=0.8, alpha=0.6)
ax_a.axvline(np.log10(4000), color='#3050a0', ls=':', lw=0.8, alpha=0.6)
ax_a.text(np.log10(2800), len(scales_order) - 0.55, "I",
          ha='center', va='top', fontsize=6.5, color='#a04030',
          fontweight='bold',
          bbox=dict(facecolor='white', edgecolor='none',
                    pad=0.4, alpha=0.85))
ax_a.text(np.log10(11000), len(scales_order) - 0.55, "II",
          ha='center', va='top', fontsize=6.5, color='#3050a0',
          fontweight='bold',
          bbox=dict(facecolor='white', edgecolor='none',
                    pad=0.4, alpha=0.85))

# Inline colourbar
cbar = fig.colorbar(im, ax=ax_a, fraction=0.04, pad=0.02,
                    aspect=15)
cbar.set_label("$R$", fontsize=6.5, labelpad=1)
cbar.ax.tick_params(labelsize=5.5)

# ============================================================
# (b) Lock-in zoom: n_col vs step (0–6500)
# ============================================================
ax_b = fig.add_subplot(gs[0, 1])
style_axes(ax_b)
panel_label(ax_b, "b", x=-0.18, y=1.06)

ax_b.axvspan(2000, 4000, alpha=0.18, color='#fcb1a4')
for s in scales_order:
    pts = sorted([(st, ncol_d[(s, st)]) for (sc, st) in ncol_d
                  if sc == s and st <= 6500])
    if len(pts) < 2: continue
    steps = np.array([p[0] for p in pts])
    ns = np.array([p[1] for p in pts])
    col = PYTHIA_VIRIDIS.get(s, "#888")
    ax_b.plot(steps, ns, '-', color=col, lw=1.2, alpha=0.85)
    ax_b.scatter(steps, ns, s=18, c=col, edgecolor='white',
                 linewidth=0.5, label=scale_label.get(s, s), zorder=3)
ax_b.set_xlabel("training step")
ax_b.set_ylabel(r"$n_{\mathrm{col}}$")
ax_b.set_xlim(0, 6500)
# Per-panel legend removed; figure-level legend at top covers all panels.

# ============================================================
# (c) Normalised growth (R-1)/(R_mature-1)
# ============================================================
ax_c = fig.add_subplot(gs[1, 0])
style_axes(ax_c)
panel_label(ax_c, "c", x=-0.18, y=1.06)

# Aggregate normalised growth across all 7 Pythia scales onto a shared
# log-step axis, then plot median + IQR band instead of 7 spaghetti lines.
all_norm = []  # list of (steps_array, R_norm_array) per scale
for s in scales_order:
    if (s, "leading") not in data or s not in mature_R: continue
    sr = sorted(data[(s, "leading")].items())
    steps_s = np.array([x[0] for x in sr])
    Rs = np.array([np.mean(x[1]) for x in sr])
    if mature_R[s] > 1:
        R_norm = (Rs - 1) / (mature_R[s] - 1)
    else:
        R_norm = Rs
    all_norm.append((steps_s, R_norm))

# Build a common log-step grid and interpolate each scale onto it
common_log_steps = np.linspace(np.log10(500), np.log10(2e5), 40)
common_steps = 10 ** common_log_steps
matrix = np.full((len(all_norm), len(common_steps)), np.nan)
for i, (st, rn) in enumerate(all_norm):
    if len(st) >= 3 and st.max() > common_steps.min():
        matrix[i] = np.interp(common_log_steps,
                              np.log10(st), rn,
                              left=np.nan, right=np.nan)

# Median + IQR
med = np.nanmedian(matrix, axis=0)
q25 = np.nanpercentile(matrix, 25, axis=0)
q75 = np.nanpercentile(matrix, 75, axis=0)
ax_c.fill_between(common_steps, q25, q75,
                  color="#3a4a6e", alpha=0.20, label="IQR (25–75%)")
ax_c.plot(common_steps, med, '-', color="#0F4D92", lw=1.6,
          label="median (7 Pythia scales)")
# Faint per-scale lines as background, alpha 0.25
for i, (st, rn) in enumerate(all_norm):
    s = scales_order[i]
    col = PYTHIA_VIRIDIS.get(s, "#888")
    ax_c.plot(st, rn, '-', color=col, lw=0.6, alpha=0.30, zorder=1)
# cycle 75 (PI feedback): legend moved to upper-right; previous lower-left
# placement was reported as covering one of the per-scale faint background
# data points. Anchored slightly inside top via bbox_to_anchor y=0.95.
ax_c.legend(loc='upper right', bbox_to_anchor=(0.99, 0.95),
            fontsize=5.5, frameon=True,
            facecolor='white', edgecolor='#aaa', framealpha=0.9,
            handlelength=1.5, handletextpad=0.4, borderaxespad=0.4)
ax_c.axhline(1.0, color='#888', ls=':', lw=0.5)
ax_c.axhline(0.5, color='black', ls='--', lw=0.5, alpha=0.4)
ax_c.set_xscale('log')
ax_c.set_xlabel("training step")
ax_c.set_ylabel(r"$(R-1)\,/\,(R_{\mathrm{mature}}-1)$")
# Annotation moved to bottom-right (lower-right empty region)
ax_c.text(0.96, 0.04,
          "common Phase II shape across scales",
          transform=ax_c.transAxes, va='bottom', ha='right', fontsize=6.5,
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=2))
# Headroom adjusted (1.4 → 1.7) so early-step extreme normalised values
# of small Pythia scales are visible instead of clipped at the top edge.
ax_c.set_ylim(-0.3, 1.7)

# ============================================================
# (d) Mature R vs log(N) — cross-scale scaling
# ============================================================
ax_d = fig.add_subplot(gs[1, 1])
style_axes(ax_d)
panel_label(ax_d, "d", x=-0.18, y=1.06)

xs, ys, labs = [], [], []
for s in scales_order + ["py12b"]:
    if s in mature_R and s in scale_params:
        xs.append(np.log10(scale_params[s]))
        ys.append(mature_R[s])
        labs.append(s)
# Add Pythia-12B if final-checkpoint final-R available
if "py12b" not in mature_R and ("py12b", "leading") in data:
    sr = sorted(data[("py12b", "leading")].items())
    if sr:
        ys_12b = [np.mean(x[1]) for x in sr]
        xs.append(np.log10(scale_params["py12b"]))
        ys.append(float(np.median(ys_12b[-min(3, len(ys_12b)):])))
        labs.append("py12b")
xs, ys = np.array(xs), np.array(ys)

if len(xs) >= 3:
    rl, pl = stats.pearsonr(xs, ys)
    slope, intc, _, _, _ = stats.linregress(xs, ys)
    xfit = np.linspace(min(xs)-0.2, max(xs)+0.2, 50)
    # cycle 74: bumped lw 1.0 → 1.3, alpha 0.45 → 0.85 for print legibility
    ax_d.plot(xfit, slope*xfit + intc, '-', color='#666', lw=1.3, alpha=0.85)

# Larger label offsets + connecting arrows; py70m label INSIDE axes (was clipping outside)
LABEL_OFFSET_5D = {
    "py70m":  (22, -22),   # below-RIGHT (was -22 left, clipped outside axes)
    "py160m": (22, 18),    # above-right
    "py410m": (-25, 5),    # above-left
    "py1b":   (25, -18),   # below-right
    "py14b":  (-28, 16),
    "py28b":  (28, -16),
    "py69b":  (-22, 18),
    "py12b":  (28, 12),
}
for x, y, lab in zip(xs, ys, labs):
    col = PYTHIA_VIRIDIS.get(lab, "#888")
    ax_d.scatter(x, y, s=80, c=col, edgecolor='black',
                 linewidth=0.6, zorder=3)
    dx, dy = LABEL_OFFSET_5D.get(lab, (15, 8))
    ha = 'right' if dx < 0 else 'left'
    ax_d.annotate(scale_label.get(lab, lab), (x, y),
                  xytext=(dx, dy), textcoords='offset points',
                  fontsize=6.5, ha=ha,
                  bbox=dict(facecolor='white', edgecolor='#bbb',
                            linewidth=0.4, pad=1.2, alpha=0.95),
                  arrowprops=dict(arrowstyle='-', color='#888',
                                   lw=0.5, alpha=0.7,
                                   shrinkA=0, shrinkB=2))

if len(xs) >= 3:
    # cycle 74: italic non-monospace, italic capital P (NMI typography)
    ax_d.text(0.04, 0.96,
              f"$r$ = {rl:.2f}\n$P$ = {pl:.1e}\n$N$ = {len(xs)}",
              transform=ax_d.transAxes, va='top', fontsize=6.5, style='italic',
              bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=3))

ax_d.set_xlabel(r"$\log_{10}$(parameter count $N$)")
ax_d.set_ylabel(r"mature $R$")

# Panel b legend: cycle 75 (PI re-feedback for supp Fig S3b) — PI now
# requests left-center placement instead of lower-right. Lock-in n_col
# trajectories rise sharply 2000–4000; the lower-left/center region
# (step<1000 with n_col<3) is empty for all scales, so left-center is
# clean. 2-col compact layout retained.
ax_b.legend(loc='center left', bbox_to_anchor=(0.01, 0.55),
            fontsize=5.5, ncol=2, frameon=True,
            facecolor='white', edgecolor='#aaa', framealpha=0.92,
            handlelength=1.2, markerscale=0.7,
            columnspacing=0.8, labelspacing=0.5,
            handletextpad=0.4, borderaxespad=0.4)

# NMI convention: figure-number sentence in LaTeX caption only.
for out_dir in (OUT, OUT2):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Figure_5_Trajectory.png"), dpi=600)
    plt.savefig(os.path.join(out_dir, "Figure_5_Trajectory.pdf"))
    plt.savefig(os.path.join(out_dir, "Figure_5_Trajectory.svg"))
print(f"Fig 5 v3 saved ({os.path.getsize(os.path.join(OUT, 'Figure_5_Trajectory.png'))//1024} KB)")
print(f"Mature R: {mature_R}")
