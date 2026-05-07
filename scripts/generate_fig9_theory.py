"""
Figure 9 (Path A): Theory vs data — autoregressive vs MLM null.
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from scipy import stats
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from nature_style import (apply_style, FAMILY_COLORS, PYTHIA_VIRIDIS,
                          panel_label, style_axes, pretty_name)
apply_style()

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"
OUT  = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\figures_v2"
# cycle 75: dual-write to v45_CURRENT/figures/ to fix supp.pdf sync.
OUT2 = r"C:\Users\Jie\Desktop\Bio-LM_v45_CURRENT\figures"

iff_lnz = np.load(os.path.join(DATA, "fig4_iff_lnz.npz"), allow_pickle=True)
iff_inn = np.load(os.path.join(DATA, "fig4_iff_inn.npz"), allow_pickle=True)
# cycle 74 redo: prefer the v3 full aggregate (LEO5 + INN merged, 60 models)
_v3 = os.path.join(DATA, "fig123_alllayer_v3_full.npz")
_ext = os.path.join(DATA, "fig123_alllayer_extended.npz")
_path = _v3 if os.path.exists(_v3) else (_ext if os.path.exists(_ext)
                                          else os.path.join(DATA, "fig123_alllayer.npz"))
A = np.load(_path, allow_pickle=True)

# Combine R per (tag, abl)
combined = defaultdict(lambda: {"R": [], "ncol": None})
for arr in (iff_lnz, iff_inn):
    for i, (tag, abl) in enumerate(arr["keys"]):
        if str(abl) != "leading": continue
        rs = [r for r in arr["R_lists"][i] if np.isfinite(r) and -5 <= r <= 10]
        combined[str(tag)]["R"].extend(rs)
        if combined[str(tag)]["ncol"] is None:
            combined[str(tag)]["ncol"] = float(arr["ncol_means"][i])

# Pythia data for r_sat fit
N_params = {"py70m":7e7, "py160m":1.6e8, "py410m":4.1e8, "py1b":1e9,
            "py14b":1.4e9, "py28b":2.8e9, "py69b":6.9e9, "py12b":1.2e10}
# Use mature R as r_sat proxy (already R - 1 is amplitude)
pythia_data = []
for s in N_params:
    if s in combined and combined[s]["R"]:
        pythia_data.append((N_params[s], float(np.mean(combined[s]["R"])), s))
pythia_data.sort()
print(f"Pythia r_sat data: {len(pythia_data)} scales")
for N, R, s in pythia_data:
    print(f"  {s} N={N:.1e} R={R:.3f}")

# Fit: r_sat = 1 - exp(-(N/N0)^alpha)
# But Pythia mature R goes 0.7 to 2.9 — not constrained to [0,1]. So fit to (R-1) vs N
def sat_model(N, N0, alpha, A):
    return A * (1 - np.exp(-(N / N0)**alpha))

Ns = np.array([d[0] for d in pythia_data])
Rs = np.array([d[1] for d in pythia_data])
amplitude = Rs - 1.0  # offset (R=1 is no-amplification baseline)
amplitude_pos = np.where(amplitude < 0, 0.001, amplitude)

try:
    popt, pcov = curve_fit(sat_model, Ns, amplitude,
                           p0=[1e9, 0.4, 2.0], maxfev=10000,
                           bounds=([1e6, 0.05, 0.5], [1e12, 2.0, 10.0]))
    N0_fit, alpha_fit, A_fit = popt
    print(f"\nFit: N0={N0_fit:.2e}, alpha={alpha_fit:.3f}, A={A_fit:.3f}")
    Rs_pred = 1.0 + sat_model(Ns, *popt)
    r2 = 1 - np.sum((Rs - Rs_pred)**2) / np.sum((Rs - np.mean(Rs))**2)
    print(f"R^2 fit = {r2:.3f}")
except Exception as e:
    print(f"Fit failed: {e}")
    N0_fit, alpha_fit, A_fit = 1e9, 0.4, 2.0
    Rs_pred = 1.0 + sat_model(Ns, N0_fit, alpha_fit, A_fit)
    r2 = 0.0

# ============================================================
fig = plt.figure(figsize=(7.2, 5.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
              left=0.10, right=0.97, top=0.93, bottom=0.10)

# (a) Theory r_sat vs observed (Pythia)
ax_a = fig.add_subplot(gs[0, 0])
style_axes(ax_a)
panel_label(ax_a, "a", x=-0.18, y=1.06)
N_smooth = np.logspace(np.log10(min(Ns))-0.2, np.log10(max(Ns))+0.2, 100)
R_smooth = 1.0 + sat_model(N_smooth, *popt)
ax_a.plot(N_smooth, R_smooth, '-', color="#444", lw=1.2,
          label=f"theory: $r_{{sat}}$ = $A(1-e^{{-(N/N_0)^\\alpha}})$\n$N_0$={N0_fit:.1e}, $\\alpha$={alpha_fit:.2f}, $A$={A_fit:.2f}")
LABEL_OFFSET_9A = {
    "py70m":  (8, -12),
    "py160m": (10, -18),   # nudged so it doesn't bump py410m label
    "py410m": (-8, -16),
    # was (-12, 14): up-left collided with py14b label; flip to down-left
    "py1b":   (-14, -16),
    "py14b":  (12, 18),
    "py28b":  (8, -16),
    "py69b":  (-12, 14),
    "py12b":  (10, -16),   # down-right to clear py69b above
}
for N, R, s in pythia_data:
    col = PYTHIA_VIRIDIS.get(s, "#888")
    ax_a.scatter(N, R, s=80, c=col, edgecolor='black', linewidth=0.5, zorder=3)
    dx, dy = LABEL_OFFSET_9A.get(s, (5, 4))
    ha = 'right' if dx < 0 else 'left'
    ax_a.annotate(pretty_name(s), (N, R), xytext=(dx, dy),
                  textcoords='offset points', fontsize=6, ha=ha,
                  bbox=dict(facecolor='white', edgecolor='none',
                            pad=0.6, alpha=0.85))
ax_a.axhline(1.0, color='#888', ls=':', lw=0.5)
ax_a.set_xscale('log')
ax_a.set_xlabel(r"parameter count $N$")
ax_a.set_ylabel(r"mature $R$")
# cycle 74: italic non-monospace
ax_a.text(0.04, 0.96, f"Pythia fit $R^2$ = {r2:.3f}",
          transform=ax_a.transAxes, va='top', fontsize=6.5, style='italic',
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=3))
ax_a.legend(loc='lower right', fontsize=6, frameon=False)

# (b) Encoder MLM null prediction (theory predicts n_col=0; observed across 22 encoders)
ax_b = fig.add_subplot(gs[0, 1])
style_axes(ax_b)
panel_label(ax_b, "b", x=-0.18, y=1.06)
# cycle 74 v44: panel (b) now shows ALL non-causal-text models (encoder MLMs
# + vision Transformers + DNA encoders) so the V_eff cross-modality story
# is visible at-a-glance. We classify by modality, not architecture, since
# vision/DNA encoders ARE expected to form SCBs per V_eff theory.
enc_records = []
for i, mid in enumerate(A["model_ids"]):
    mod = str(A["modality"][i])
    arch = str(A["architecture"][i])
    L = int(A["n_layers_total"][i])
    if L < 5: continue  # skip too-short probes (smoke tests)
    # Show all non-causal-text models on this panel
    if mod in ("text",) and arch != "encoder":
        continue
    raw_id = str(mid).split("/")[-1]
    short_map = {
        "nucleotide-transformer-v2-500m-multi-species": "NucT-500M",
        "nucleotide-transformer-v2-2.5b-multi-species": "NucT-2.5B",
        "nucleotide-transformer-2.5b-multi-species": "NucT-2.5B",
        "nucleotide-transformer-500m-1000g": "NucT-500M",
        "vit-base-patch16-224": "ViT-B/16",
        "vit_base": "ViT-B/16",
        "dinov2-large": "DINOv2-L",
        "dinov2_large": "DINOv2-L",
        "dinov2-base": "DINOv2-B",
        "dinov2_base": "DINOv2-B",
        "dinov2-small": "DINOv2-S",
        "convnextv2-tiny-22k-224": "ConvNeXtV2-T",
        "convnextv2_tiny": "ConvNeXtV2-T",
        "progen2-xlarge": "ProGen2-xlarge",
        "esm2_t36_3B_UR50D": "ESM2-3B",
        "esm2_t33_650M_UR50D": "ESM2-650M",
        "esm2_t30_150M_UR50D": "ESM2-150M",
        "esm2_t12_35M_UR50D": "ESM2-35M",
        "esm2_t6_8M_UR50D": "ESM2-8M",
    }
    short = short_map.get(raw_id, raw_id[:24])
    short = pretty_name(short, fallback=short)
    enc_records.append({
        "tag": short,
        "n_col": int(A["n_cols"][i]),
        "n_params": float(A["n_params"][i]),
        "modality": mod or "text",
    })
# cycle 74 redo: now that fig123_alllayer_extended has 17 encoder MLMs with
# real PR data, NO hardcoded fill-in is needed. Comment out for transparency.
# (legacy hardcoded list removed — all encoder n_col values come from
#  the LEO5 _leo5_pr_probe.py runs.)
# dedupe by tag
seen = set(); unique_enc = []
for r in enc_records:
    if r["tag"] not in seen and r["n_params"]:
        seen.add(r["tag"]); unique_enc.append(r)

# Theory prediction: low V_eff regimes form SCB; high V_eff text/protein MLM
# do not. Plot: x = log_N, y = n_col, color by modality.
mod_colors = {"text": "#1f77b4", "protein": "#2ca02c", "DNA": "#ff7f0e",
              "vision": "#9467bd"}
for r in unique_enc:
    col = mod_colors.get(r["modality"], "#7f7f7f")
    ax_b.scatter(r["n_params"], r["n_col"], s=60, c=col,
                 edgecolor='black', linewidth=0.4, alpha=0.9, zorder=3)
    if r["n_col"] > 0 or len(unique_enc) <= 12:
        ax_b.annotate(r["tag"], (r["n_params"], r["n_col"]),
                      xytext=(4, 3), textcoords='offset points', fontsize=5.5)
# Theory line
ax_b.axhline(0, color='#d62728', ls='--', lw=1.0, label="theory: $n_{col}$ = 0")
# Bands for causal-LM cluster
ax_b.axhspan(8, 32, alpha=0.08, color='#1f77b4',
             label="causal LM range\n($n_{col}$ = 8-30)")
ax_b.set_xscale('log')
ax_b.set_xlabel(r"parameter count $N$")
ax_b.set_ylabel(r"$n_{\mathrm{col}}$")
ax_b.set_ylim(-2, 35)
# cycle-74 v44: split annotation by modality so V_eff theory's multi-class
# prediction is visible. Text/protein MLM should be ~all n_col=0; DNA +
# vision encoders should form SCBs.
text_protein_zero = sum(1 for r in unique_enc
                        if r["modality"] in ("text", "protein") and r["n_col"] == 0)
text_protein_total = sum(1 for r in unique_enc if r["modality"] in ("text", "protein"))
vis_dna_with_scb = sum(1 for r in unique_enc
                       if r["modality"] in ("vision", "DNA") and r["n_col"] >= 3)
vis_dna_total = sum(1 for r in unique_enc if r["modality"] in ("vision", "DNA"))
ax_b.text(0.04, 0.95,
          f"text/protein MLMs:\n{text_protein_zero}/{text_protein_total} at $n_{{col}}$=0\n"
          f"vision+DNA encoders:\n{vis_dna_with_scb}/{vis_dna_total} form SCB",
          transform=ax_b.transAxes, va='top', fontsize=6.5,
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=3))
# Legend lifted ABOVE axes so it doesn't sit on top of NucT-500M annotation
ax_b.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02),
            ncol=2, fontsize=6, frameon=False,
            handlelength=1.4, handletextpad=0.5)

# (c) Cross-modality scaling: theory predicts r_sat depends on V_eff (vocab entropy)
ax_c = fig.add_subplot(gs[1, 0])
style_axes(ax_c)
panel_label(ax_c, "c", x=-0.18, y=1.06)
# Plot: causal LMs from text vs protein on the same r_sat-vs-N plot
text_pts = [(N, R, s) for s, (N, R) in zip(["py70m","py160m","py410m","py1b","py14b","py28b","py69b","py12b"],
                                            [(N_params[s], combined[s]["R"]) for s in N_params if combined[s]["R"]])
            if R]
# protein ProGen2 — IFF panel data tag is "progen2_large" (singular).
# cycle 74 (redo): the prior pg2_* tags silently dropped 4/5 ProGen2 scales
# because they don't match data tags. Honest fix: only ProGen2-large was
# probed for deep-bootstrap mature R; the other 4 ProGen2 scales appear
# in the architectural scan (fig123_alllayer) but not in the IFF panel.
pg_meta = {"progen2_large": 2.7e9}
pg_data = []
for s, n in pg_meta.items():
    if s in combined and combined[s]["R"]:
        pg_data.append((n, float(np.mean(combined[s]["R"])), s))
print(f"Fig 9 (c): {len(pg_data)} ProGen2 scale(s) with deep-bootstrap R data: {[d[2] for d in pg_data]}")

text_xs = [t[0] for t in text_pts]
text_ys = [float(np.mean(t[1])) if isinstance(t[1], list) else t[1] for t in text_pts]
pg_xs = [t[0] for t in pg_data]
pg_ys = [t[1] for t in pg_data]

ax_c.scatter(text_xs, text_ys, s=60, c="#1f77b4", edgecolor='black',
             linewidth=0.4, alpha=0.9, label="text (Pythia)", zorder=3)
ax_c.scatter(pg_xs, pg_ys, s=60, c="#2ca02c", edgecolor='black',
             linewidth=0.4, alpha=0.9, label="protein (ProGen2)", zorder=3)
ax_c.plot(N_smooth, R_smooth, '-', color="#1f77b4", lw=1.0, alpha=0.5,
          label="theory (Pythia fit)")
ax_c.set_xscale('log')
ax_c.set_xlabel(r"parameter count $N$")
ax_c.set_ylabel(r"mature $R$")
ax_c.legend(loc='upper left', fontsize=6.5)
ax_c.text(0.96, 0.04,
          "ProGen2 sits BELOW Pythia\nfit at matched $N$ → larger $V_{eff}$\nfor protein vocab\n($\\alpha$ data-dependent)",
          transform=ax_c.transAxes, va='bottom', ha='right', fontsize=6,
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=2))

# (d) Slope-(−1) prediction from V_eff theory
ax_d = fig.add_subplot(gs[1, 1])
style_axes(ax_d)
panel_label(ax_d, "d", x=-0.18, y=1.06)
# Theory predicts slope(log V_eff, log Δ NLL_ICL) = -1
# Empirical from §M.5: slope = -0.92 ± 0.18
slopes_obs = np.array([-0.92, -0.89, -0.94, -0.93, -0.91, -0.84])
labels_obs = ["headline\n(canonical)", "alt $\\ell_{deep}$", "alt $K$ range",
              "exclude IFF", "alt 23-model", "ProGen2-only"]
sigma = 0.18
ypos = np.arange(len(slopes_obs))
ax_d.errorbar(slopes_obs, ypos, xerr=sigma, fmt='o', color="#1f77b4",
              markersize=7, markeredgecolor='black', markeredgewidth=0.4,
              capsize=2, ecolor="#888", elinewidth=0.7)
ax_d.axvline(-1.0, color='#d62728', ls='--', lw=1.0, label="theory: slope = −1")
ax_d.axvline(0, color='#000', lw=0.8, label="zero-effect reference")
ax_d.set_yticks(ypos)
ax_d.set_yticklabels(labels_obs, fontsize=6.5)
ax_d.set_xlim(-1.4, 0.2)
ax_d.set_xlabel(r"empirical slope $d \log \Delta\mathrm{NLL}_{\mathrm{ICL}} / d \log V_{\mathrm{eff}}$")
# Legend at lower-right (shifted right per PI feedback)
ax_d.legend(loc='lower right', fontsize=6, frameon=True,
            facecolor='white', edgecolor='#aaa', framealpha=0.9,
            ncol=1, handlelength=1.5, handletextpad=0.4,
            borderaxespad=0.4)

# NMI convention: figure-number sentence in LaTeX caption only.
for out_dir in (OUT, OUT2):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Figure_9_Theory.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "Figure_9_Theory.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "Figure_9_Theory.svg"))
print(f"\nFig 9 saved ({os.path.getsize(os.path.join(OUT, 'Figure_9_Theory.png'))//1024} KB)")
