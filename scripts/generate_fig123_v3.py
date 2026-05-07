"""
Figures 1, 2, 3 v3 — Nature-grade.
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from nature_style import apply_style, FAMILY_COLORS, PYTHIA_VIRIDIS, panel_label, style_axes, pretty_name
apply_style()

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"
OUT  = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\figures_v2"

A   = np.load(os.path.join(DATA, "fig123_alllayer_v3_full.npz"), allow_pickle=True)
PH  = np.load(os.path.join(DATA, "fig3_phase2.npz"),    allow_pickle=True)
COS = np.load(os.path.join(DATA, "leo5_consec_cos.npz"), allow_pickle=True)
IFFL = np.load(os.path.join(DATA, "fig4_iff_lnz.npz"), allow_pickle=True)
IFFI = np.load(os.path.join(DATA, "fig4_iff_inn.npz"), allow_pickle=True)

# Same META as v2
META = {
    "py70m":{"N":7e7,"L":6,"mod":"text","fam":"Pythia"},
    "py160m":{"N":1.6e8,"L":12,"mod":"text","fam":"Pythia"},
    "py410m":{"N":4.1e8,"L":24,"mod":"text","fam":"Pythia"},
    "py1b":{"N":1e9,"L":17,"mod":"text","fam":"Pythia"},
    "py14b":{"N":1.4e9,"L":24,"mod":"text","fam":"Pythia"},
    "py28b":{"N":2.8e9,"L":32,"mod":"text","fam":"Pythia"},
    "py69b":{"N":6.9e9,"L":32,"mod":"text","fam":"Pythia"},
    "py12b":{"N":1.2e10,"L":36,"mod":"text","fam":"Pythia"},
    "bloom_1b1":{"N":1.1e9,"L":24,"mod":"text","fam":"BLOOM"},
    "bloom_3b":{"N":3e9,"L":30,"mod":"text","fam":"BLOOM"},
    "bloom7b":{"N":7e9,"L":30,"mod":"text","fam":"BLOOM"},
    "bloom_560m":{"N":5.6e8,"L":24,"mod":"text","fam":"BLOOM"},
    "qwen_05":{"N":5e8,"L":24,"mod":"text","fam":"Qwen2.5"},
    "qwen_15":{"N":1.5e9,"L":28,"mod":"text","fam":"Qwen2.5"},
    "qwen_3":{"N":3e9,"L":36,"mod":"text","fam":"Qwen2.5"},
    "qwen_7":{"N":7e9,"L":29,"mod":"text","fam":"Qwen2.5"},
    "gptneo_125":{"N":1.25e8,"L":12,"mod":"text","fam":"GPT-Neo"},
    "gptneo_13":{"N":1.3e9,"L":24,"mod":"text","fam":"GPT-Neo"},
    "gptneo_27":{"N":2.7e9,"L":32,"mod":"text","fam":"GPT-Neo"},
    "mistral7b":{"N":7e9,"L":33,"mod":"text","fam":"Mistral"},
    "stablelm":{"N":1.6e9,"L":24,"mod":"text","fam":"Other"},
    "tinyllama":{"N":1.1e9,"L":22,"mod":"text","fam":"Other"},
    "smol_17":{"N":1.7e9,"L":24,"mod":"text","fam":"Other"},
    "smol_360":{"N":3.6e8,"L":32,"mod":"text","fam":"Other"},
    "phi15":{"N":1.3e9,"L":24,"mod":"text","fam":"Other"},
    "phi1":{"N":1.3e9,"L":24,"mod":"text","fam":"Other"},
    "esm2_3b":{"N":3e9,"L":36,"mod":"protein","fam":"ESM2(MLM)"},
    "esm2_650m":{"N":6.5e8,"L":33,"mod":"protein","fam":"ESM2(MLM)"},
    "esm2_150m":{"N":1.5e8,"L":30,"mod":"protein","fam":"ESM2(MLM)"},
    "esm2_35m":{"N":3.5e7,"L":12,"mod":"protein","fam":"ESM2(MLM)"},
    "esm2_8m":{"N":8e6,"L":6,"mod":"protein","fam":"ESM2(MLM)"},
    "bert_base":{"N":1.1e8,"L":12,"mod":"text","fam":"BERT(MLM)"},
    "bert_large":{"N":3.4e8,"L":24,"mod":"text","fam":"BERT(MLM)"},
    "roberta_base":{"N":1.25e8,"L":12,"mod":"text","fam":"BERT(MLM)"},
    "nt_500m":{"N":5e8,"L":24,"mod":"DNA","fam":"NucT(MLM)"},
    "pg2_small":{"N":1.5e8,"L":12,"mod":"protein","fam":"ProGen2"},
    "pg2_medium":{"N":7.6e8,"L":27,"mod":"protein","fam":"ProGen2"},
    "pg2_large":{"N":2.7e9,"L":32,"mod":"protein","fam":"ProGen2"},
    "pg2_xlarge":{"N":6.4e9,"L":32,"mod":"protein","fam":"ProGen2"},
    "qwen_moe_a27":{"N":1.4e10,"L":24,"mod":"text","fam":"MoE"},
    "olmoe_1b_7b":{"N":7e9,"L":16,"mod":"text","fam":"MoE"},
    "vit_b":{"N":8.6e7,"L":12,"mod":"vision","fam":"Vision"},
    "vit_l":{"N":3e8,"L":24,"mod":"vision","fam":"Vision"},
    "dino_b":{"N":8.6e7,"L":12,"mod":"vision","fam":"Vision"},
    "dino_l":{"N":3e8,"L":24,"mod":"vision","fam":"Vision"},
}

# n_col per tag from IFF panel
ncol_map = {}
for arr in (IFFL, IFFI):
    for i, (tag, abl) in enumerate(arr["keys"]):
        if str(abl) == "leading" and str(tag) not in ncol_map:
            ncol_map[str(tag)] = float(arr["ncol_means"][i])
for t in META:
    if "MLM" in META[t]["fam"] or t.startswith("vit") or t.startswith("dino"):
        ncol_map.setdefault(t, 0)

records = [{"tag": t, "N": m["N"], "n_col": ncol_map.get(t, 0),
            "mod": m["mod"], "fam": m["fam"]} for t, m in META.items()]

# Cycle-75 fix: pull DNA / vision / protein encoder n_col directly from
# fig123_alllayer_v3_full.npz so panel (c) reflects the V_eff cross-modality
# finding (NucT 18,24; ViT 8; DINOv2-L 6, etc.) instead of hardcoded zeros.
for i in range(len(A["model_ids"])):
    mid = str(A["model_ids"][i])
    mod = str(A["modality"][i])
    nc = int(A["n_cols"][i])
    Np = float(A["n_params"][i])
    if mod in ("DNA", "vision", "protein"):
        # Use a stable internal tag (last path component, lowercase)
        tag = mid.split("/")[-1].lower().replace("-", "_")[:30]
        if tag not in ncol_map:
            ncol_map[tag] = nc
            records.append({"tag": tag, "N": Np, "n_col": nc,
                            "mod": mod,
                            "fam": "Vision" if mod == "vision"
                                   else ("NucT(MLM)" if mod == "DNA"
                                         else "ESM2(MLM)" if "esm" in tag
                                         else "ProGen2")})

# ============================================================
# Figure 1 — Cross-arch landscape
# ============================================================
fig1 = plt.figure(figsize=(7.2, 5.5))
gs1 = GridSpec(2, 2, figure=fig1, hspace=0.42, wspace=0.32,
               left=0.08, right=0.97, top=0.94, bottom=0.10)

# (a) param vs n_col
ax1a = fig1.add_subplot(gs1[0, 0])
style_axes(ax1a)
panel_label(ax1a, "a", x=-0.16, y=1.06)
for fam in FAMILY_COLORS:
    pts = [(r["N"], r["n_col"]) for r in records if r["fam"] == fam]
    if not pts: continue
    xs, ys = zip(*pts)
    ax1a.scatter(xs, ys, s=42, color=FAMILY_COLORS[fam],
                 edgecolor='black', linewidth=0.5, alpha=0.88,
                 label=fam, zorder=3)
ax1a.set_xscale('log')
ax1a.axhline(0, color='#aaa', ls=':', lw=0.5)
ax1a.set_xlabel(r"parameter count $N$")
ax1a.set_ylabel(r"$n_{\mathrm{col}}$  (SCB length, layers)")
ax1a.legend(loc='upper left', fontsize=5.5, ncol=2,
            handletextpad=0.3, columnspacing=0.6, frameon=True,
            facecolor='white', edgecolor='#aaa', framealpha=0.9)
# Move "n=45 models" annotation OUT of plot area to title-line right corner
ax1a.set_title(f"$n$ = {len(records)} models", loc='right',
               fontsize=6.5, color='#444', pad=2)

# (b) family histogram
ax1b = fig1.add_subplot(gs1[0, 1])
style_axes(ax1b)
panel_label(ax1b, "b", x=-0.20, y=1.06)
fams = sorted(set(r["fam"] for r in records),
              key=lambda f: -np.mean([r["n_col"] for r in records if r["fam"]==f] or [0]))
ypos = np.arange(len(fams))
xs_max  = [max(r["n_col"] for r in records if r["fam"]==f) for f in fams]
xs_mean = [np.mean([r["n_col"] for r in records if r["fam"]==f]) for f in fams]
xs_n    = [sum(1 for r in records if r["fam"]==f) for f in fams]
cols = [FAMILY_COLORS[f] for f in fams]
ax1b.barh(ypos, xs_max, color=cols, alpha=0.30, label="max", edgecolor='none')
ax1b.barh(ypos, xs_mean, color=cols, alpha=0.95, label="mean",
          edgecolor='black', linewidth=0.4)
# n= annotations: place INSIDE long bars (white text), OUTSIDE short bars
xmax_axis = max(xs_max) * 1.15
for i, n in enumerate(xs_n):
    if xs_max[i] > xmax_axis * 0.5:
        # long bar — put label inside near right edge
        ax1b.text(xs_max[i] - 0.5, i, f"$n$={n}", va='center', ha='right',
                  fontsize=6, color='white', fontweight='bold')
    else:
        # short bar — put label outside right
        ax1b.text(xs_max[i] + 0.4, i, f"$n$={n}", va='center', ha='left',
                  fontsize=6, color='#444')
ax1b.set_yticks(ypos)
ax1b.set_yticklabels(fams, fontsize=6.5)
ax1b.set_xlabel(r"$n_{\mathrm{col}}$")
ax1b.set_xlim(0, xmax_axis)
ax1b.legend(loc='upper right', fontsize=6, frameon=True,
            facecolor='white', edgecolor='#aaa', framealpha=0.9)

# (c) modality bar
ax1c = fig1.add_subplot(gs1[1, 0])
style_axes(ax1c)
panel_label(ax1c, "c", x=-0.16, y=1.06)
mods = ["text", "protein", "DNA", "vision"]
mod_data = {}
for m in mods:
    rs = [r for r in records if r["mod"] == m]
    if rs: mod_data[m] = (sum(1 for r in rs if r["n_col"] > 0), len(rs))
xpos = np.arange(len(mod_data))
fracs  = [mod_data[m][0]/mod_data[m][1] for m in mod_data]
totals = [mod_data[m][1] for m in mod_data]
mod_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#8c564b"]
bars = ax1c.bar(xpos, fracs, color=mod_colors, alpha=0.88,
                edgecolor='black', linewidth=0.5, width=0.62)
for i, (m, (s, t)) in enumerate(mod_data.items()):
    ax1c.text(i, fracs[i] + 0.04, f"{s}/{t}", ha='center', fontsize=7)
ax1c.set_xticks(xpos)
ax1c.set_xticklabels(list(mod_data.keys()))
ax1c.set_ylabel(r"fraction with $n_{\mathrm{col}} > 0$")
ax1c.set_ylim(0, 1.15)

# (d) representative PR profiles
ax1d = fig1.add_subplot(gs1[1, 1])
style_axes(ax1d)
panel_label(ax1d, "d", x=-0.16, y=1.06)
mid_show = {
    "EleutherAI/pythia-6.9b": ("Pythia-6.9B", "#1f77b4"),
    "mistralai/Mistral-7B-v0.3": ("Mistral-7B", "#d62728"),
    "Qwen/Qwen2.5-7B": ("Qwen-7B", "#ff7f0e"),
    "facebook/esm2_t36_3B_UR50D": ("ESM2-3B (encoder)", "#7f7f7f"),
    "google-bert/bert-large-uncased": ("BERT-large (encoder)", "#555"),
}
for i, mid in enumerate(A["model_ids"]):
    if str(mid) in mid_show:
        label, col = mid_show[str(mid)]
        layers = A["layers"][i]
        prs = A["prs"][i]
        L = max(layers) if len(layers) else 1
        x = np.array(layers) / L
        ax1d.plot(x, prs, '-', color=col, lw=1.4, alpha=0.9, label=label)
ax1d.axhline(5, color='#d62728', ls='--', lw=0.7, alpha=0.7, label="PR<5 SCB threshold")
ax1d.axvspan(0.25, 0.95, alpha=0.04, color='#888')
ax1d.set_yscale('log')
ax1d.set_xlabel(r"relative depth $\ell/L$")
ax1d.set_ylabel(r"$\mathrm{PR}_\ell$")
# Legend ABOVE the panel (per PI: 2 rows x 3 cols, no overlap with curves)
ax1d.legend(loc='lower center', bbox_to_anchor=(0.5, 1.04),
            fontsize=5.5, ncol=3, frameon=False,
            handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)

plt.savefig(os.path.join(OUT, "Figure_1_Landscape.png"), dpi=600)
plt.savefig(os.path.join(OUT, "Figure_1_Landscape.pdf"))
plt.savefig(os.path.join(OUT, "Figure_1_Landscape.svg"))
print(f"Fig 1 v3 saved ({os.path.getsize(os.path.join(OUT, 'Figure_1_Landscape.png'))//1024} KB)")
plt.close(fig1)

# ============================================================
# Figure 2 — Eigvec continuity
# ============================================================
fig2 = plt.figure(figsize=(7.2, 5.5))
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.32,
               left=0.08, right=0.97, top=0.94, bottom=0.10)

cos_map = {}
for i, mid in enumerate(COS["model_ids"]):
    cos_map[str(mid)] = (np.array(COS["cos_mean"][i]),
                         np.array(COS["cos_std"][i]),
                         int(COS["n_seeds"][i]))

# (a) Pythia-1B (proxy SCB-bearing — Mistral not in COS data, use highest n_seeds Pythia)
ax2a = fig2.add_subplot(gs2[0, 0])
style_axes(ax2a)
panel_label(ax2a, "a", x=-0.16, y=1.06)
for k in cos_map:
    if "pythia-1b" in k.lower():
        cm, cs, n = cos_map[k]
        x = np.arange(len(cm))
        ax2a.fill_between(x, cm-cs, cm+cs, alpha=0.20, color="#1f77b4")
        ax2a.plot(x, cm, '-o', color="#1f77b4", lw=1.4, markersize=4,
                  label=f"{k.split('/')[-1]}\n($n_{{\\mathrm{{seeds}}}}$={n})")
        break
ax2a.axhline(0.99, color='#aaa', ls=':', lw=0.6)
ax2a.axhline(0,    color='#aaa', lw=0.4)
ax2a.set_ylim(-0.3, 1.05)
ax2a.set_xlabel(r"layer pair index $\ell, \ell+1$")
ax2a.set_ylabel(r"cosine($v_\ell, v_{\ell+1}$)")
ax2a.legend(loc='lower center', fontsize=6.5)

# (b) ESM2 control
ax2b = fig2.add_subplot(gs2[0, 1])
style_axes(ax2b)
panel_label(ax2b, "b", x=-0.16, y=1.06)
esm = [k for k in cos_map if "esm2" in k.lower()]
if esm:
    k = max(esm, key=lambda kk: cos_map[kk][2])
    cm, cs, n = cos_map[k]
    x = np.arange(len(cm))
    ax2b.fill_between(x, cm-cs, cm+cs, alpha=0.20, color="#7f7f7f")
    ax2b.plot(x, cm, '-o', color="#7f7f7f", lw=1.4, markersize=4,
              label=f"{k.split('/')[-1]}\n($n_{{\\mathrm{{seeds}}}}$={n})")
ax2b.axhline(0.99, color='#aaa', ls=':', lw=0.6, label="0.99 reference")
ax2b.axhline(0,    color='#aaa', lw=0.4)
ax2b.set_ylim(-0.3, 1.05)
ax2b.set_xlabel(r"layer pair index")
ax2b.set_ylabel(r"cosine")
ax2b.legend(loc='lower center', fontsize=6.5)

# (c) cross-model summary — median + IQR per class on a common depth grid
ax2c = fig2.add_subplot(gs2[1, 0])
style_axes(ax2c)
panel_label(ax2c, "c", x=-0.16, y=1.06)
SCB_KEYS = ["pythia", "mistral", "qwen", "bloom", "gpt-neo", "tinyllama",
            "smol", "phi", "stablelm", "olmo", "yi"]
ENC_KEYS = ["bert", "roberta", "albert", "esm2", "scibert", "distilbert"]

# Resample each model onto common depth grid, then aggregate by class
common_x = np.linspace(0, 1, 30)
causal_mat, encoder_mat = [], []
for k, (cm, cs, n) in cos_map.items():
    if n < 3 or len(cm) < 3: continue
    kl = k.lower()
    x_src = np.arange(len(cm)) / max(1, len(cm) - 1)
    interp = np.interp(common_x, x_src, cm)
    if any(s in kl for s in SCB_KEYS):
        causal_mat.append(interp)
    elif any(s in kl for s in ENC_KEYS):
        encoder_mat.append(interp)
causal_mat = np.array(causal_mat)
encoder_mat = np.array(encoder_mat)
n_scb = len(causal_mat)
n_enc = len(encoder_mat)

# Faint per-model background (alpha 0.15)
for row in causal_mat:
    ax2c.plot(common_x, row, '-', color="#1f77b4", lw=0.4, alpha=0.15, zorder=1)
for row in encoder_mat:
    ax2c.plot(common_x, row, '-', color="#7f7f7f", lw=0.4, alpha=0.20, zorder=1)

# Median + IQR band per class
if n_scb > 0:
    med_c = np.median(causal_mat, axis=0)
    q25_c = np.percentile(causal_mat, 25, axis=0)
    q75_c = np.percentile(causal_mat, 75, axis=0)
    ax2c.fill_between(common_x, q25_c, q75_c, color="#1f77b4", alpha=0.25, zorder=2)
    ax2c.plot(common_x, med_c, '-', color="#0F4D92", lw=1.8,
              label=f"causal LMs ($n$={n_scb}, median + IQR)", zorder=3)
if n_enc > 0:
    med_e = np.median(encoder_mat, axis=0)
    q25_e = np.percentile(encoder_mat, 25, axis=0)
    q75_e = np.percentile(encoder_mat, 75, axis=0)
    ax2c.fill_between(common_x, q25_e, q75_e, color="#7f7f7f", alpha=0.25, zorder=2)
    ax2c.plot(common_x, med_e, '-', color="#3a3a3a", lw=1.8,
              label=f"encoder MLMs ($n$={n_enc}, median + IQR)", zorder=3)

ax2c.axhline(0.99, color='black', ls=':', lw=0.6, alpha=0.7)
ax2c.axhline(0, color='#aaa', lw=0.4)
ax2c.set_ylim(-0.3, 1.05)
ax2c.set_xlabel(r"relative layer pair $\ell/L$")
ax2c.set_ylabel(r"cosine continuity")
ax2c.legend(loc='lower center', fontsize=6, frameon=False)

# (d) distribution histogram
ax2d = fig2.add_subplot(gs2[1, 1])
style_axes(ax2d)
panel_label(ax2d, "d", x=-0.16, y=1.06)
scb_means, enc_means = [], []
for k, (cm, cs, n) in cos_map.items():
    if n < 3: continue
    mn = float(np.mean(cm))
    kl = k.lower()
    if any(s in kl for s in SCB_KEYS): scb_means.append(mn)
    elif any(s in kl for s in ENC_KEYS): enc_means.append(mn)
bins = np.linspace(0, 1, 22)
ax2d.hist(scb_means, bins=bins, alpha=0.85, color="#1f77b4",
          edgecolor='black', linewidth=0.4,
          label=f"causal LMs ($n$={len(scb_means)})")
if enc_means:
    ax2d.hist(enc_means, bins=bins, alpha=0.85, color="#7f7f7f",
              edgecolor='black', linewidth=0.4,
              label=f"encoder MLMs ($n$={len(enc_means)})")
ax2d.axvline(0, color='#d62728', ls='--', lw=0.8, label="random-dir expectation")
ax2d.set_xlabel(r"mean consecutive cosine")
ax2d.set_ylabel("count")
ax2d.legend(loc='upper left', fontsize=6.5)

# cycle 74: Fig 2 is now generated by generate_fig2_v4.py (NxN heatmap
# version that matches the caption); v3's consec-cosine 4-panel here is
# kept as a sanity check but written to a separate filename so it cannot
# overwrite the heatmap version under figures_v2/Figure_2_Continuity.*.
plt.savefig(os.path.join(OUT, "Figure_2_ConsecCos_DEPRECATED.png"), dpi=600)
print(f"Fig 2 (consec-cos legacy) saved as DEPRECATED filename — heatmap version is canonical")
plt.close(fig2)

# ============================================================
# Figure 3 — Phase 2 ablation
# ============================================================
fig3 = plt.figure(figsize=(7.2, 5.5))
gs3 = GridSpec(2, 2, figure=fig3, hspace=0.42, wspace=0.32,
               left=0.08, right=0.97, top=0.94, bottom=0.12)

ph_keys = list(PH["keys"])
ph_data = {(str(k[0]), str(k[1])): {
    "rel": float(PH["rel_means"][i]),
    "rel_std": float(PH["rel_stds"][i]),
    "delta": float(PH["delta_means"][i]),
    "n": int(PH["delta_n"][i]),
    "prs": PH["prs_avg"][i],
} for i, k in enumerate(ph_keys)}

def canon_tag(t):
    for p in ["LNRD4_","LNRD3_","LNRD2_","LNRD_","LNIFF_","INRD_","INRD2_","INRD3_","INRD4_","INRD5_"]:
        if t.startswith(p): return t[len(p):]
    return t

canon = defaultdict(lambda: defaultdict(list))
for (tag, abl), v in ph_data.items():
    ct = canon_tag(tag)
    canon[ct][abl].append(v)

# (a) Pythia-1B PR profile
ax3a = fig3.add_subplot(gs3[0, 0])
style_axes(ax3a)
panel_label(ax3a, "a", x=-0.18, y=1.06)
# Shade the SCB band (ℓ=4–14 in Pythia-1B) for cross-figure consistency
# with Hero Fig 1 panel (b).
ax3a.axvspan(4, 14, alpha=0.10, color="#E0A234", zorder=0)
if "py1b" in canon and "leading" in canon["py1b"]:
    p1 = canon["py1b"]["leading"][0].get("prs")
    if p1 is not None:
        x = np.arange(len(p1))
        ax3a.plot(x, p1, '-o', color="#0F4D92", lw=1.4, markersize=4,
                  label="few-shot ICL probe", zorder=3)
for i, mid in enumerate(A["model_ids"]):
    if "pythia-1b" in str(mid).lower():
        layers = A["layers"][i]; prs = A["prs"][i]
        ax3a.plot(layers, prs, '-s', color="#9467bd", lw=1.2, markersize=3.5,
                  alpha=0.7, label="general continuation probe", zorder=2)
        break
ax3a.axhline(5, color='#888', ls=':', lw=0.7)
ax3a.text(15.5, 5.5, "PR=5", color='#888', fontsize=5.5, ha='right',
          va='bottom', style='italic')
ax3a.set_yscale('log')
ax3a.set_xlabel(r"layer index $\ell$")
ax3a.set_ylabel(r"$\mathrm{PR}_\ell$")
ax3a.set_title("Pythia-1B (probe-corpus contrast)", fontsize=7,
               color="#444", pad=2, loc='left')
ax3a.legend(loc='upper center', fontsize=5.5, frameon=False,
            bbox_to_anchor=(0.5, 1.02), ncol=2, handletextpad=0.4,
            columnspacing=1.0)

# (b) bar chart leading vs 2nd vs random for top SCB models
ax3b = fig3.add_subplot(gs3[0, 1])
style_axes(ax3b)
panel_label(ax3b, "b", x=-0.18, y=1.06)
# Drop Mistral-7B-v0.1 (kept v0.3 which is the headline anchor model);
# trim to 10 representative SCB-bearing causal LMs covering family + scale.
panel_models = ["py70m","py160m","py410m","py1b","py14b","py28b","py69b","py12b",
                "mistral7b","smol_17","stablelm","tinyllama","gptneo_27","gptneo_13",
                "phi15","bloom7b","bloom_3b","qwen_3","qwen_7"]
abl_show = ["leading", "second", "random"]
# Use SCB project palette: leading=BASELINE red, second=gold, random=neutral
abl_colors = {"leading": "#B64342", "second": "#E0A234", "random": "#767676"}
existing = []
for tag in panel_models:
    if tag in canon and all(a in canon[tag] and canon[tag][a] for a in abl_show):
        existing.append(tag)
existing = existing[:10]
xpos = np.arange(len(existing))
w = 0.27
for j, abl in enumerate(abl_show):
    vals = [canon[t][abl][0]["rel"] for t in existing]
    errs = [canon[t][abl][0]["rel_std"] for t in existing]
    ax3b.bar(xpos + j*w - w, vals, w, yerr=errs, capsize=1.5,
             color=abl_colors[abl], label=abl, alpha=0.9,
             edgecolor='black', linewidth=0.4, error_kw={"linewidth": 0.6})
ax3b.set_yscale('symlog', linthresh=1)
# Constrain y-axis to actual data range (max ~700%) — was showing 10³ tick
# above max-data, which misled the eye.
ax3b.set_ylim(0, 1500)
ax3b.set_xticks(xpos)
ax3b.set_xticklabels([pretty_name(t) for t in existing],
                     rotation=35, ha='right', fontsize=5.5)
ax3b.set_ylabel(r"$\Delta$NLL (% baseline)")
ax3b.legend(loc='upper right', fontsize=6, ncol=3, frameon=False,
            handletextpad=0.4, columnspacing=0.8)

# (c) n_col vs ΔNLL leading scatter
ax3c = fig3.add_subplot(gs3[1, 0])
style_axes(ax3c)
panel_label(ax3c, "c", x=-0.18, y=1.06)
xs, ys, labs = [], [], []
for tag in canon:
    if tag in META and "leading" in canon[tag] and canon[tag]["leading"]:
        nc = ncol_map.get(tag, 0)
        rel = canon[tag]["leading"][0]["rel"]
        if 0 < rel < 1e4:
            xs.append(nc); ys.append(rel); labs.append(tag)
xs, ys = np.array(xs), np.array(ys)
# Mark IFF cluster (n_col<=0.5) with same black-diamond shape used in
# Fig 1 panel (d) and Fig 4 — cross-figure marker consistency.
# Restrict (c) to SCB-bearing causal LMs only — IFF cluster is already
# shown in Fig 1 panel (d) / Fig 4 / Hero with the proper baseline-R metric;
# mixing it here with the leading-PC ΔNLL metric confused readers.
# Restrict to SCB-bearing causal LMs only
keep = xs > 0.5
xs, ys, labs = xs[keep], ys[keep], [labs[i] for i in range(len(labs)) if keep[i]]

# Horizontal lollipop chart: rows = models (sorted by n_col ascending), bar
# length = leading-PC projection-out ΔNLL, color = n_col (viridis gradient).
# This format lets every reader identify each model, see the n_col→ΔNLL
# trend visually (rows fill rightward as we move down high n_col), and has
# zero label/marker overlap because every model has its own row.
order = np.argsort(xs)
xs_s = xs[order]; ys_s = ys[order]
names_s = [pretty_name(labs[i]) for i in order]
n_models = len(xs_s)
y_pos = np.arange(n_models)

# Color stems & dots by n_col via viridis
import matplotlib.cm as cm
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=min(xs_s), vmax=max(xs_s))
colors = [cm.viridis(norm(v)) for v in xs_s]

# Stems
for yi, val, col in zip(y_pos, ys_s, colors):
    ax3c.plot([0, val], [yi, yi], '-', color=col, lw=1.6, alpha=0.85, zorder=2)
# Dots at the end (size encodes n_col, color also encodes n_col via viridis)
ax3c.scatter(ys_s, y_pos, s=46, c=colors, edgecolor='black',
             linewidth=0.5, zorder=3)

ax3c.set_yticks(y_pos)
ax3c.set_yticklabels([f"{n}  ({int(c)})" for n, c in zip(names_s, xs_s)],
                     fontsize=5.5)
ax3c.set_xscale('symlog', linthresh=10)
ax3c.set_xlim(0, max(ys_s) * 1.4)
ax3c.set_ylim(-0.7, n_models - 0.3)
ax3c.set_xlabel(r"leading-PC projection-out $\Delta$NLL (%)")
ax3c.set_ylabel(r"model  ($n_{\rm col}$)", fontsize=6)
ax3c.invert_yaxis()  # smallest n_col at top, largest at bottom

# n_col already shown in y-tick labels "ModelName (n_col)" so no colorbar
# is needed; the viridis stem colors function as a visual sorting cue.

if len(xs) >= 3:
    rp, pp = stats.pearsonr(xs, ys)
    # Stats live ABOVE the panel as title — no in-axes overlap with stems
    # or y-tick labels.
    ax3c.set_title(
        f"$n$={len(xs)} SCB-bearing causal LMs    "
        f"Pearson $r$ = {rp:.2f}, $P$ = {pp:.1e}",
        fontsize=6.5, style='italic', loc='left', pad=3)

# (d) Pythia scaling
ax3d = fig3.add_subplot(gs3[1, 1])
style_axes(ax3d)
panel_label(ax3d, "d", x=-0.18, y=1.06)
pyth_scales = [("py70m",7e7),("py160m",1.6e8),("py410m",4.1e8),("py1b",1e9),("py14b",1.4e9),
               ("py28b",2.8e9),("py69b",6.9e9),("py12b",1.2e10)]
pxs, pys, perrs, plab = [], [], [], []
for tag, n in pyth_scales:
    if tag in canon and "leading" in canon[tag] and canon[tag]["leading"]:
        pxs.append(n); pys.append(canon[tag]["leading"][0]["rel"])
        perrs.append(canon[tag]["leading"][0]["rel_std"]); plab.append(tag)
# Connector + visible error bars (was alpha=0.4, capsize=0 → invisible).
ax3d.errorbar(pxs, pys, yerr=perrs, fmt='-', color='#666',
              lw=0.9, alpha=0.7, capsize=2.5, ecolor='#888',
              elinewidth=0.7, zorder=2)
# Larger label offsets + connector arrows; py70m label INSIDE axes (was clipping outside left edge)
# py160m moved away from py410m (dark green circle) overlap.
LABEL_OFFSET = {
    "py70m":  (15, 18),     # right-above (inside axes; previous left-offset clipped)
    "py160m": (15, -22),    # right-below (away from py410m green dot above)
    "py410m": (-22, 18),
    "py1b":   (22, -20),
    "py14b":  (-28, 18),
    "py28b":  (28, -22),
    "py69b":  (-25, 22),
    "py12b":  (28, 8),
}
for x, y, lab in zip(pxs, pys, plab):
    col = PYTHIA_VIRIDIS.get(lab, "#888")
    ax3d.scatter(x, y, s=70, color=col, edgecolor='black',
                 linewidth=0.5, zorder=3)
    dx, dy = LABEL_OFFSET.get(lab, (15, 8))
    ha = 'right' if dx < 0 else 'left'
    ax3d.annotate(pretty_name(lab), (x, y), xytext=(dx, dy),
                  textcoords='offset points', fontsize=6, ha=ha,
                  bbox=dict(facecolor='white', edgecolor='#bbb',
                            linewidth=0.4, pad=1.2, alpha=0.95),
                  arrowprops=dict(arrowstyle='-', color='#888',
                                   lw=0.5, alpha=0.7,
                                   shrinkA=0, shrinkB=2))
ax3d.set_xscale('log')
ax3d.set_xlabel(r"parameter count $N$")
ax3d.set_ylabel(r"$\Delta$NLL leading (%)")

# NMI convention: figure-number sentence lives in LaTeX caption only.
plt.savefig(os.path.join(OUT, "Figure_3_Phase2.png"), dpi=600)
plt.savefig(os.path.join(OUT, "Figure_3_Phase2.pdf"))
plt.savefig(os.path.join(OUT, "Figure_3_Phase2.svg"))
print(f"Fig 3 v3 saved ({os.path.getsize(os.path.join(OUT, 'Figure_3_Phase2.png'))//1024} KB)")
plt.close(fig3)
