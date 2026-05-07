"""
Figure 2 v4 — Nature-grade with REAL NxN cosine heatmaps.

cycle 74 audit fixes (2026-05-02):
  - removed in-figure suptitle (caption-only convention)
  - heatmap titles: panel-label letter only, descriptor moved to caption
  - bottom-row class detection extended (was missing several architectures)
  - histogram switched to histtype='step' (causal+encoder no longer occlude)
  - added Mann-Whitney U p-value annotation in panel (d)
  - exemplar names go through pretty_name() for cross-figure consistency
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy import stats as _scstats

sys.path.insert(0, os.path.dirname(__file__))
from nature_style import (apply_style, panel_label, style_axes,
                          save_publication, pretty_name)
apply_style()

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"
OUT  = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\figures_v2"

D = np.load(os.path.join(DATA, "fig2_nxn.npz"), allow_pickle=True)
results = {}
for i, tag in enumerate(D["tags"]):
    results[str(tag)] = {
        "cos": np.array(D["cos_means"][i]),
        "n_seeds": int(D["n_seeds"][i]),
        "L": int(D["L"][i]),
        "modality": str(D["modality"][i]),
        "arch": str(D["arch"][i]),
    }

print(f"Loaded {len(results)} models from fig2_nxn.npz")
for t in sorted(results):
    r = results[t]
    print(f"  {t}: L={r['L']} n_seeds={r['n_seeds']} ({r['modality']}/{r['arch']})")

fig = plt.figure(figsize=(7.2, 6.0))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
              left=0.08, right=0.96, top=0.94, bottom=0.10)

cmap_seq = "RdBu_r"  # red-white-blue diverging at 0; values in [-1,1]
norm = Normalize(vmin=-1, vmax=1)

def heatmap(ax, cos, L, panel_letter, descriptor):
    """Draw NxN heatmap. Bold panel letter only; descriptor goes inside as
    a small white-bbox annotation in the upper-left corner."""
    panel_label(ax, panel_letter, x=-0.18, y=1.06)
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap=cmap_seq, aspect='auto',
                   origin='upper', interpolation='nearest')
    lo, hi = int(0.25 * L), int(0.95 * L)
    ax.add_patch(plt.Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
                               fill=False, edgecolor='black', lw=1.0,
                               linestyle='--'))
    ax.set_xlabel(r"layer $\ell'$")
    ax.set_ylabel(r"layer $\ell$")
    ax.text(0.04, 0.95, descriptor, transform=ax.transAxes,
            fontsize=6.5, va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='#aaa',
                      linewidth=0.4, pad=2))
    return im

# (a) Mistral-7B heatmap
ax_a = fig.add_subplot(gs[0, 0])
style_axes(ax_a)
m = results["mistral7_full"]
mid_off = m["cos"][int(0.25*m["L"]):int(0.95*m["L"])+1, int(0.25*m["L"]):int(0.95*m["L"])+1]
mask = ~np.eye(mid_off.shape[0], dtype=bool)
mistral_mean = float(np.mean(np.abs(mid_off[mask])))
im_a = heatmap(ax_a, m["cos"], m["L"], "a",
               f"Mistral-7B (causal LM)\n$L$={m['L']},  $\\langle|\\cos|\\rangle$={mistral_mean:.3f}\nSCB band: dashed")

# (b) ESM2-3B heatmap (control)
ax_b = fig.add_subplot(gs[0, 1])
style_axes(ax_b)
e = results["esm2_3b_full"]
mid_off = e["cos"][int(0.25*e["L"]):int(0.95*e["L"])+1, int(0.25*e["L"]):int(0.95*e["L"])+1]
mask = ~np.eye(mid_off.shape[0], dtype=bool)
esm_mean = float(np.mean(np.abs(mid_off[mask])))
im_b = heatmap(ax_b, e["cos"], e["L"], "b",
               f"ESM2-3B (encoder MLM)\n$L$={e['L']},  $\\langle|\\cos|\\rangle$={esm_mean:.3f}")

# Colorbar (shared)
cbar_ax = fig.add_axes([0.97, 0.55, 0.013, 0.35])
cb = fig.colorbar(im_b, cax=cbar_ax)
cb.set_label("cos$(v_\\ell, v_{\\ell'})$", fontsize=7)
cb.ax.tick_params(labelsize=6)

# (c) Cross-model summary: mid-band off-diagonal cosine per model
ax_c = fig.add_subplot(gs[1, 0])
style_axes(ax_c)
panel_label(ax_c, "c", x=-0.18, y=1.06)
# Class detection — extended (cycle 74) to include nuct, vit, dino which
# previously fell into "other" silently.
def class_of(tag):
    tl = tag.lower()
    arch_field = results[tag]["arch"].lower()
    mod_field = results[tag]["modality"].lower()
    if arch_field == "encoder" or any(k in tl for k in ['esm', 'bert', 'roberta', 'nucleotide', 'nuct', 'electra', 'albert', 'distilbert', 'deberta']):
        return "encoder MLM"
    if mod_field == "vision" or any(k in tl for k in ['vit', 'dino', 'mae', 'clip']):
        return "vision"
    return "causal LM"

rows = []
for t, r in results.items():
    L = r["L"]
    lo, hi = int(0.25 * L), int(0.95 * L)
    if hi - lo < 3: continue
    mid = r["cos"][lo:hi+1, lo:hi+1]
    mask = ~np.eye(mid.shape[0], dtype=bool)
    rows.append({
        "tag": t, "class": class_of(t),
        "mean": float(np.mean(np.abs(mid[mask]))),
        "L": L,
    })
rows.sort(key=lambda x: -x["mean"])
xs = np.arange(len(rows))
heights = [r["mean"] for r in rows]
classes = [r["class"] for r in rows]
class_colors = {"causal LM": "#0F4D92", "encoder MLM": "#B64342",
                "vision": "#8c564b"}
cols = [class_colors[c] for c in classes]
ax_c.bar(xs, heights, color=cols, alpha=0.92,
         edgecolor='black', linewidth=0.4)
ax_c.set_xticks(xs)
ax_c.set_xticklabels([pretty_name(r["tag"]) for r in rows],
                     rotation=35, ha='right', fontsize=6)
ax_c.set_ylabel(r"SCB-band $\langle|\cos|\rangle$ (off-diag)")
ax_c.axhline(0.99, color='#0F4D92', ls=':', lw=0.6, alpha=0.7)
ax_c.text(len(rows) - 0.3, 0.985, r"$\geq 0.99$ continuity",
          color='#0F4D92', fontsize=5.5, ha='right', va='top',
          style='italic')
import matplotlib.patches as mp
n_causal = sum(1 for r in rows if r['class']=='causal LM')
n_enc = sum(1 for r in rows if r['class']=='encoder MLM')
ax_c.legend(handles=[mp.Patch(color="#0F4D92",
                              label=f"causal LM ($n$={n_causal})"),
                     mp.Patch(color="#B64342",
                              label=f"encoder MLM ($n$={n_enc})")],
            loc='lower left', bbox_to_anchor=(0.0, 1.02),
            ncol=2, fontsize=6.5, frameon=False,
            handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
ax_c.set_ylim(0, 1.05)

# (d) Distribution histogram: within-band off-diag cosines, causal vs encoder
ax_d = fig.add_subplot(gs[1, 1])
style_axes(ax_d)
panel_label(ax_d, "d", x=-0.18, y=1.06)
causal_offs, encoder_offs = [], []
for t, r in results.items():
    L = r["L"]
    lo, hi = int(0.25 * L), int(0.95 * L)
    if hi - lo < 3: continue
    mid = r["cos"][lo:hi+1, lo:hi+1]
    mask = ~np.eye(mid.shape[0], dtype=bool)
    vals = np.abs(mid[mask])
    cls = class_of(t)
    if cls == "causal LM": causal_offs.extend(vals)
    elif cls == "encoder MLM": encoder_offs.extend(vals)

bins = np.linspace(0, 1, 24)
# Step histogram (cycle 74) — non-occluding view of overlapping distributions.
ax_d.hist(encoder_offs, bins=bins, histtype='step', color="#B64342", lw=1.4,
          label=f"encoder MLMs ($n_{{\\mathrm{{pairs}}}}$={len(encoder_offs)})")
ax_d.hist(causal_offs, bins=bins, histtype='step', color="#0F4D92", lw=1.4,
          label=f"causal LMs ($n_{{\\mathrm{{pairs}}}}$={len(causal_offs)})")
# Filled step underneath at low alpha for the shape cue.
ax_d.hist(encoder_offs, bins=bins, color="#B64342", alpha=0.18,
          edgecolor='none')
ax_d.hist(causal_offs, bins=bins, color="#0F4D92", alpha=0.18,
          edgecolor='none')

# Analytic random-direction expectation for |cos| in N=4096 dims:
# E[|cos|] ≈ sqrt(2/(πN)) ≈ 0.012 — rendered explicitly, not at zero.
rand_exp = float(np.sqrt(2 / (np.pi * 4096)))
ax_d.axvline(rand_exp, color='#444', ls='--', lw=0.7,
             label=fr"random ($\sqrt{{2/\pi N}}\!\approx\!{rand_exp:.3f}$)")

# Mann-Whitney U on the two distributions
if len(causal_offs) > 1 and len(encoder_offs) > 1:
    _, p_mw = _scstats.mannwhitneyu(causal_offs, encoder_offs,
                                    alternative='two-sided')
    p_str = (f"$P\\!=\\!{p_mw:.1e}$" if p_mw > 1e-100
             else r"$P\!<\!10^{-100}$")
    ax_d.text(0.40, 0.55, f"Mann–Whitney U\n{p_str}",
              transform=ax_d.transAxes, va='top', ha='left',
              fontsize=6, style='italic',
              bbox=dict(facecolor='white', edgecolor='#aaa',
                        linewidth=0.4, pad=2))

ax_d.set_xlabel(r"$|\cos(v_\ell, v_{\ell'})|$  (within SCB band, off-diagonal)")
ax_d.set_ylabel("count")
ax_d.legend(loc='upper center', fontsize=6, frameon=False)

# NMI convention: figure-number sentence lives in LaTeX caption only.
save_publication(fig, os.path.join(OUT, "Figure_2_Continuity"),
                 formats=("png", "pdf", "svg"))
print(f"Mistral-7B SCB-band mean cosine: {mistral_mean:.3f}")
print(f"ESM2-3B   SCB-band mean cosine: {esm_mean:.3f}")
