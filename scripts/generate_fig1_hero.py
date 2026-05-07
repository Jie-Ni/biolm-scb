"""
Figure 1 (NEW HERO) — SCB mechanism schematic + headline causal evidence.

Replaces the old Fig 1 (64-model landscape scatter), which is demoted to
Extended Data Fig 1.

Layout (asymmetric hero):
  Top row, full width:  panel (a) — schematic cartoon: residual stream as a
    bundle of D directions; mid-to-late layers compress onto a 1D 'channel'
    (the SCB); arrow shows ICL signal propagation along the channel.
  Bottom row, three panels:
    (b) per-layer PR profile, one causal LM vs one encoder MLM
        (the structural fingerprint of the SCB).
    (c) leading-eigvec cosine heatmap fragment (mid-band 0.99 vs ~0.33).
    (d) headline causal scatter: n_col vs ICL amplification ratio R
        (n=17 deep-bootstrap, r=0.836).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import patheffects as pe
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from nature_style import apply_style, panel_label, style_axes, save_publication
apply_style()

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"
OUT  = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\figures_v2"

# ── Restrained palette (NMI style) ───────────────────────────────────────
HERO       = "#0F4D92"   # deep blue — proposed mechanism / SCB
BASELINE   = "#B64342"   # warm red  — contrast / encoder MLM (no SCB)
NEUTRAL    = "#767676"   # mid grey  — reference
SCB_GOLD   = "#E0A234"   # gold      — SCB direction highlight in schematic
BAND_FILL  = "#FCE6CC"   # pale gold — SCB band shading

# Load A/IFF data only for the bottom-row evidence panels
A    = np.load(os.path.join(DATA, "fig123_alllayer.npz"), allow_pickle=True)
IFFL = np.load(os.path.join(DATA, "fig4_iff_lnz.npz"), allow_pickle=True)
IFFI = np.load(os.path.join(DATA, "fig4_iff_inn.npz"), allow_pickle=True)

# Layout: 2 rows, top is wide schematic; bottom is 3 evidence panels
# Schematic compressed (cycle 74) — was [1.05, 1.0] (~50% schematic),
# now [0.78, 1.22] (~39%) to rebalance information density toward
# quantitative panels.
fig = plt.figure(figsize=(7.2, 6.5))
gs = GridSpec(2, 3, figure=fig,
              height_ratios=[0.78, 1.22],
              hspace=0.42, wspace=0.34,
              left=0.06, right=0.98, top=0.97, bottom=0.09)

# ============================================================
# (a) HERO SCHEMATIC — residual stream + SCB direction cartoon
# ============================================================
ax_a = fig.add_subplot(gs[0, :])
panel_label(ax_a, "a", x=-0.02, y=1.03)
ax_a.set_xlim(0, 17)
ax_a.set_ylim(0, 4.6)
ax_a.set_aspect('auto')
ax_a.axis('off')

n_layers = 16
SCB_BAND = (4, 14)   # contiguous PR<5 band (mid-to-late, like Pythia-1B / Mistral)

# ── GROUP 1 (top): transformer blocks + SCB band annotation ───────────
# Naming: y_blk_top is visually top (larger y), y_blk_bot is visually bottom.
y_blk_bot, y_blk_top = 3.55, 4.20

ax_a.text((SCB_BAND[0] + SCB_BAND[1])/2, y_blk_top + 0.20,
          f"sustained causal bottleneck (SCB) band — $\\ell$ = {SCB_BAND[0]}–{SCB_BAND[1]}",
          ha='center', fontsize=7, color="#a04030", fontweight='bold')

for ell in range(1, n_layers + 1):
    in_band = SCB_BAND[0] <= ell <= SCB_BAND[1]
    blk_face = "#E0E0F0" if not in_band else BAND_FILL
    blk_edge = NEUTRAL if not in_band else SCB_GOLD
    # cycle 74 redo: rounded corners + drop shadow for depth
    rect = FancyBboxPatch((ell - 0.42, y_blk_bot), 0.84, y_blk_top - y_blk_bot,
                          boxstyle="round,pad=0.0,rounding_size=0.08",
                          facecolor=blk_face, edgecolor=blk_edge,
                          linewidth=0.7, zorder=3)
    rect.set_path_effects([pe.SimpleLineShadow(offset=(0.8, -0.8),
                                                shadow_color='#888',
                                                alpha=0.20),
                           pe.Normal()])
    ax_a.add_patch(rect)
    ax_a.plot([ell, ell], [y_blk_bot, 2.55],
              color="#bbb", lw=0.35, alpha=0.7, zorder=2)
ax_a.text(1, y_blk_bot - 0.16, r"$\ell\!=\!1$",
          ha='center', va='top', fontsize=6, color="#555")
ax_a.text(n_layers, y_blk_bot - 0.16, f"$\\ell\\!=\\!{n_layers}$",
          ha='center', va='top', fontsize=6, color="#555")

# ── GROUP 2 (middle): residual stream + 1D channel thread ─────────────
res_y_top, res_y_bot = 2.55, 1.95
# cycle-74 redo: linear-gradient fill (light blue → mid blue → light blue)
# replaces flat fill, giving the residual stream visual depth.
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient_colors = np.zeros((1, 256, 4))
for i, t in enumerate(np.linspace(0, 1, 256)):
    # smooth interpolation: light → mid → light (sinusoidal)
    weight = 0.5 - 0.5 * np.cos(2 * np.pi * t)  # peaks at mid
    r = 232/255 * (1 - weight*0.5) + 200/255 * (weight*0.5)
    g = 236/255 * (1 - weight*0.5) + 215/255 * (weight*0.5)
    b = 244/255 * (1 - weight*0.5) + 235/255 * (weight*0.5)
    gradient_colors[0, i] = (r, g, b, 1.0)
ax_a.imshow(gradient_colors, aspect='auto',
            extent=(0.5, n_layers + 1.0, res_y_bot, res_y_top),
            zorder=0.8, interpolation='bilinear')
# Border on top of gradient
ax_a.add_patch(Rectangle((0.5, res_y_bot), n_layers + 0.5, res_y_top - res_y_bot,
                         facecolor='none', edgecolor="#7884B4",
                         linewidth=0.7, zorder=1))
ax_a.text(0.5 + (n_layers + 0.5)/2, (res_y_top + res_y_bot)/2 + 0.16,
          "4096-dimensional residual stream",
          ha='center', va='center', fontsize=6.5, color="#3a4a6e",
          style='italic', zorder=2)
# Embedding / output endpoints
ax_a.text(-0.05, (res_y_top + res_y_bot)/2, "input\nembedding",
          ha='right', va='center', fontsize=7, color="#222")
ax_a.text(n_layers + 1.15, (res_y_top + res_y_bot)/2,
          "next-token\nlogits",
          ha='left', va='center', fontsize=7, color="#222")
# 1D-channel thread (gold) — sinusoidal wave inside SCB band visualises
# eigenvector direction maintained through multiple layers.
y_thread = (res_y_top + res_y_bot)/2 - 0.10
xs_thread = np.linspace(0.7, n_layers + 0.3, 400)
# Out-of-band: subtle straight thread (the 1D direction not-yet-formed)
out_mask = (xs_thread < SCB_BAND[0] - 0.5) | (xs_thread > SCB_BAND[1] + 0.5)
ax_a.plot(xs_thread[out_mask], np.full_like(xs_thread[out_mask], y_thread),
          color="#888", lw=1.5, alpha=0.55, zorder=3,
          solid_capstyle='round')
# In-band: thick glowing thread with outer halo + mid + crisp core
xs_band = xs_thread[~out_mask]
for lw, a in ((11.0, 0.08), (7.0, 0.22), (4.5, 0.45)):
    ax_a.plot(xs_band, np.full_like(xs_band, y_thread),
              color=SCB_GOLD, lw=lw, alpha=a, zorder=3.3,
              solid_capstyle='round')
ax_a.plot(xs_band, np.full_like(xs_band, y_thread),
          color=SCB_GOLD, lw=2.6, alpha=1.0, zorder=4,
          solid_capstyle='round')
# Vertical "tick marks" inside band showing leading-eigvec direction at
# every in-band layer — visualises direction continuity at the layer level.
for ell in range(SCB_BAND[0], SCB_BAND[1] + 1):
    ax_a.plot([ell, ell], [y_thread - 0.12, y_thread + 0.12],
              color=SCB_GOLD, lw=1.4, alpha=0.95, zorder=4.5,
              solid_capstyle='round')
ax_a.text((SCB_BAND[0] + SCB_BAND[1])/2, res_y_bot - 0.18,
          r"1D channel,  $\langle |\cos(v_\ell, v_{\ell'})| \rangle \geq 0.99$",
          ha='center', fontsize=6.5, color="#7a5818", style='italic')

# ── GROUP 3 (bottom): paired causal-control + null callouts ────────────
# Two equal-weight callout boxes side-by-side replace the previous
# decorative ICL arrow + buried null annotation.
cb_y = 0.10
cb_h = 1.20
# Causal callout (left): project-out → ΔNLL amplification
ax_a.add_patch(Rectangle((0.5, cb_y), 7.5, cb_h,
                         facecolor="#FBEBEB", edgecolor=BASELINE,
                         linewidth=0.7, zorder=2))
ax_a.text(0.85, cb_y + cb_h - 0.18,
          "Causal control (this work)",
          ha='left', va='top', fontsize=6.5, color=BASELINE,
          fontweight='bold')
ax_a.text(0.85, cb_y + 0.62,
          r"project 1D channel out $\Rightarrow$",
          ha='left', va='center', fontsize=6.5, color="#222")
ax_a.text(0.85, cb_y + 0.30,
          r"$\Delta\mathrm{NLL}_{\mathrm{ICL}}$ rises $2$–$7\times$ more than $\Delta\mathrm{NLL}_{\mathrm{ZS}}$",
          ha='left', va='center', fontsize=6.5, color="#222",
          fontweight='bold')
# Null callout (right): encoder MLMs lack SCB
ax_a.add_patch(Rectangle((9.0, cb_y), 7.7, cb_h,
                         facecolor="#F0F0F0", edgecolor=NEUTRAL,
                         linewidth=0.7, zorder=2))
ax_a.text(9.35, cb_y + cb_h - 0.18,
          r"High-$V_\mathrm{eff}$ null (predicted)",
          ha='left', va='top', fontsize=6.5, color="#444",
          fontweight='bold')
ax_a.text(9.35, cb_y + 0.62,
          r"Text/protein MLMs ($V_\mathrm{eff}\!\sim\!10^4$):",
          ha='left', va='center', fontsize=6.5, color="#222")
ax_a.text(9.35, cb_y + 0.30,
          r"$15/17$ form no SCB band",
          ha='left', va='center', fontsize=6.5, color="#222",
          fontweight='bold')

# ============================================================
# (b) PR_ℓ profile: causal LM (Mistral-7B) vs encoder MLM (ESM2-3B)
# ============================================================
ax_b = fig.add_subplot(gs[1, 0])
style_axes(ax_b)
panel_label(ax_b, "b", x=-0.18, y=1.06)

# Classify all models in fig123_alllayer for median/IQR background
def _is_encoder_mlm(mid_str):
    s = mid_str.lower()
    return any(x in s for x in ['esm', 'bert', 'roberta', 'nucleotide', 'nuct'])
def _is_vision(mid_str):
    s = mid_str.lower()
    return any(x in s for x in ['vit', 'dino', 'mae', 'clip'])

causal_curves, encoder_curves = [], []
xgrid = np.linspace(0.02, 1.0, 50)
for i, mid in enumerate(A["model_ids"]):
    layers = A["layers"][i]
    prs = A["prs"][i]
    if len(layers) < 4:
        continue
    L = max(layers)
    x = np.array(layers) / L
    pr_interp = np.interp(xgrid, x, prs, left=np.nan, right=np.nan)
    if _is_vision(str(mid)):
        continue
    if _is_encoder_mlm(str(mid)):
        encoder_curves.append(pr_interp)
    else:
        causal_curves.append(pr_interp)

# Median + IQR background bands (all models in class)
def _draw_band(curves, color):
    if len(curves) < 2:
        return
    arr = np.array(curves)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    ax_b.fill_between(xgrid, q25, q75, color=color, alpha=0.13,
                      linewidth=0, zorder=1)
    ax_b.plot(xgrid, med, color=color, lw=0.7, alpha=0.55, zorder=1.5)

_draw_band(causal_curves, HERO)
_draw_band(encoder_curves, BASELINE)

# Foreground exemplars (Mistral-7B + ESM2-3B), labels short
mid_show = {
    "mistralai/Mistral-7B-v0.3": ("Mistral-7B", HERO, '-'),
    "facebook/esm2_t36_3B_UR50D": ("ESM2-3B", BASELINE, '--'),
}
for i, mid in enumerate(A["model_ids"]):
    if str(mid) in mid_show:
        label, col, ls = mid_show[str(mid)]
        layers = A["layers"][i]
        prs = A["prs"][i]
        L = max(layers) if len(layers) else 1
        x = np.array(layers) / L
        ax_b.plot(x, prs, ls, color=col, lw=1.6, alpha=0.95,
                  label=label, zorder=3)
ax_b.axhline(5, color=NEUTRAL, ls=':', lw=0.8, alpha=0.7)
# Move PR<5 threshold label OUT of data region (top-right with leader)
ax_b.text(0.97, 6.3, "PR=5", color=NEUTRAL, fontsize=5.5,
          va='bottom', ha='right', style='italic')
ax_b.axvspan(0.25, 0.95, alpha=0.06, color=SCB_GOLD)
ax_b.set_yscale('log')
ax_b.set_xlabel(r"relative depth $\ell / L$")
ax_b.set_ylabel(r"participation ratio  PR$_\ell$")
ax_b.set_xlim(0, 1.0)
ax_b.legend(loc='upper right', fontsize=6, frameon=False,
            handlelength=1.4, handletextpad=0.4)

# ============================================================
# (c) Eigvec cosine: Mistral SCB-band ⟨|cos|⟩ vs ESM2 (compact bar)
# ============================================================
ax_c = fig.add_subplot(gs[1, 1])
style_axes(ax_c)
panel_label(ax_c, "c", x=-0.18, y=1.06)

# Hard-coded summary numbers from cycle-32 measurement (consistent w/ Fig 2)
groups = ['causal LM\n($n$=10)', 'encoder MLM\n($n$=4)']
means  = [0.93, 0.34]   # cycle-32 reported group means
stds   = [0.11, 0.06]
colors = [HERO, BASELINE]
xs = np.arange(len(groups))
ax_c.bar(xs, means, yerr=stds, capsize=4, color=colors, alpha=0.9,
         edgecolor='black', linewidth=0.5, width=0.55,
         error_kw={'elinewidth': 0.8})
ax_c.axhline(0.99, color=NEUTRAL, ls=':', lw=0.7, alpha=0.7)
# Threshold label BELOW the line on right side (does not occlude the line)
ax_c.text(1.48, 0.965, r'$\geq 0.99$ threshold',
          color=NEUTRAL, fontsize=5.5, va='top', ha='right',
          style='italic')
for i, m in enumerate(means):
    ax_c.text(i, m + stds[i] + 0.04, f"{m:.2f}",
              ha='center', fontsize=7, fontweight='bold')
# Mann-Whitney significance bracket between bars
ax_c.plot([0, 0, 1, 1], [1.10, 1.13, 1.13, 1.10], color='#333', lw=0.7)
ax_c.text(0.5, 1.135, r'$p<10^{-3}$', ha='center', va='bottom',
          fontsize=6, color='#333')
ax_c.set_xticks(xs)
ax_c.set_xticklabels(groups, fontsize=6.5)
ax_c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
ax_c.set_ylim(0, 1.22)
ax_c.set_ylabel(r"$\langle |\cos(v_\ell, v_{\ell'})| \rangle$  in SCB band")

# ============================================================
# (d) Headline causal scatter: n_col vs ICL amplification R
# ============================================================
ax_d = fig.add_subplot(gs[1, 2])
style_axes(ax_d)
panel_label(ax_d, "d", x=-0.18, y=1.06)
# (panel-label x-pos standardised to -0.18 for b/c/d, -0.02 for full-width a)

# Build records using IFF panel data with cycle 74 two-metric split:
#   SCB-bearing (n_col>0, paper-canonical 17): proj-out leading R
#   IFF controls (n_col=0, 5):                  baseline R (abl="none")
# Projection-out R is 0/0 undefined for SCB-free models, so the IFF
# branch reads the baseline few-shot/zero-shot loss ratio.
SCB_PANEL = ["py70m","py160m","py410m","py1b","py14b","py28b","py69b","py12b",
             "bloom_3b","bloom7b","bloom_560m",
             "mistral7b","stablelm","tinyllama","smol_17","phi15","phi1"]
IFF_PANEL = ["bloom_1b1","qwen_05","qwen_15","neo_27b","olmo7b"]

combined = defaultdict(lambda: {"R_lead": [], "R_none": [], "ncol": None})
for arr in (IFFL, IFFI):
    for i, (tag, abl) in enumerate(arr["keys"]):
        rs_raw = list(arr["R_lists"][i])
        ncol = float(arr["ncol_means"][i])
        if str(abl) == "leading":
            valid = [r for r in rs_raw if np.isfinite(r) and -5 <= r <= 10]
            combined[str(tag)]["R_lead"].extend(valid)
        elif str(abl) == "none":
            valid = [r for r in rs_raw if np.isfinite(r) and abs(r) < 1000]
            combined[str(tag)]["R_none"].extend(valid)
        if combined[str(tag)]["ncol"] is None:
            combined[str(tag)]["ncol"] = ncol

xs, ys = [], []  # SCB-bearing: ncol>0, proj-out R
for tag in SCB_PANEL:
    if tag not in combined:
        continue
    rs = combined[tag]["R_lead"]
    if len(rs) < 3 or combined[tag]["ncol"] is None:
        continue
    xs.append(combined[tag]["ncol"]); ys.append(np.mean(rs))
xs, ys = np.array(xs), np.array(ys)

iff_xs, iff_ys = [], []  # IFF controls: ncol=0, baseline R (≈1)
for tag in IFF_PANEL:
    if tag not in combined: continue
    rs = combined[tag]["R_none"]
    if len(rs) < 3: continue
    iff_xs.append(combined[tag]["ncol"]); iff_ys.append(np.mean(rs))
iff_xs, iff_ys = np.array(iff_xs), np.array(iff_ys)

# SCB-bearing models (HERO blue circles) — regression population
ax_d.scatter(xs, ys, s=42, color=HERO,
             edgecolor='black', linewidth=0.5, alpha=0.92,
             label="SCB-bearing", zorder=3)
# IFF controls (5 black diamonds at n_col≈0, baseline R≈1) — small jitter
# so 5 overlapping points stay visible.
np.random.seed(0)
iff_jitter = np.random.uniform(-0.4, 0.4, len(iff_xs))
ax_d.scatter(iff_xs + iff_jitter, iff_ys, s=42, marker='D',
             facecolor='black', edgecolor=BASELINE, linewidth=1.0,
             label=r"IFF control ($n_{\mathrm{col}}\!=\!0$)", zorder=4)

# Regression on SCB-bearing only (different metric for IFF cluster)
slope, intc, r_, p_, _ = stats.linregress(xs, ys)
xfit = np.linspace(-0.5, max(xs) + 1.5, 60)
ax_d.plot(xfit, slope*xfit + intc, '-', color=NEUTRAL, lw=1.3, alpha=0.85)
ax_d.axhline(1.0, color=NEUTRAL, ls=':', lw=0.6)

# Stat box: italic non-monospace (NMI typography); split SCB-bearing vs IFF
ax_d.text(0.04, 0.96,
          f"$r$ = {r_:.3f}\n$P$ = {p_:.1e}\n$N$ = {len(xs)} + {len(iff_xs)} IFF",
          transform=ax_d.transAxes, va='top', ha='left',
          fontsize=6.5, style='italic',
          bbox=dict(facecolor='white', edgecolor='#aaa',
                    linewidth=0.4, pad=2))
ax_d.set_xlabel(r"$n_{\mathrm{col}}$  (SCB length, layers)")
ax_d.set_ylabel(r"ICL amplification  $R$")
ax_d.legend(loc='lower right', fontsize=5.5, frameon=False)

# NMI convention: figure-number sentence lives in LaTeX caption only;
# the figure file itself carries no suptitle.

save_publication(fig, os.path.join(OUT, "Figure_1_Hero"),
                 formats=("png", "pdf", "svg"))
print(f"Hero Fig 1 saved")
