"""
Figure 8 (Path F): SCB-augmented scaling-law beats log(N)-only.
LOO prediction of mature R using log_N + SCB features across 8 Pythia + cross-arch.
"""
import os, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

sys.path.insert(0, os.path.dirname(__file__))
from nature_style import apply_style, FAMILY_COLORS, PYTHIA_VIRIDIS, panel_label, style_axes, pretty_name
apply_style()

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"
OUT  = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\11_manuscript\figures_v2"
# cycle 75: dual-write to v45_CURRENT/figures/ to fix supp.pdf sync.
OUT2 = r"C:\Users\Jie\Desktop\Bio-LM_v45_CURRENT\figures"

# Load IFF panel (final-checkpoint R + n_col)
iff_lnz = np.load(os.path.join(DATA, "fig4_iff_lnz.npz"), allow_pickle=True)
iff_inn = np.load(os.path.join(DATA, "fig4_iff_inn.npz"), allow_pickle=True)
combined = defaultdict(lambda: {"R": [], "ncol": None})
for arr in (iff_lnz, iff_inn):
    for i, (tag, abl) in enumerate(arr["keys"]):
        if str(abl) != "leading": continue
        rs = [r for r in arr["R_lists"][i] if np.isfinite(r) and -5 <= r <= 10]
        combined[str(tag)]["R"].extend(rs)
        if combined[str(tag)]["ncol"] is None:
            combined[str(tag)]["ncol"] = float(arr["ncol_means"][i])

# Param counts
# cycle 74 (redo) — match Fig 4 SCB_PANEL canonical 17-model panel exactly:
#   - removed legacy mistral7 (Mistral-7B-v0.1 duplicate of mistral7b v0.3)
#   - removed qwen_3/qwen_7 (data tags are qwen_3b/qwen_7b — but they're not
#     in the 17-canonical SCB panel anyway, so excluded)
#   - removed gptneo_* (data tags are neo_* AND not in canonical 17 panel)
#   - added phi1 (was missing)
#   - kept bloom_1b1/qwen_05/qwen_15 to filter as IFF (n_col=0)
# Result: this script IN-records will produce exactly 17 SCB-bearing models,
# matching Fig 4 paper-canonical regression panel.
META = {
    "py70m":7e7,"py160m":1.6e8,"py410m":4.1e8,"py1b":1e9,"py14b":1.4e9,
    "py28b":2.8e9,"py69b":6.9e9,"py12b":1.2e10,
    "bloom_1b1":1.1e9,"bloom_3b":3e9,"bloom7b":7e9,"bloom_560m":5.6e8,
    "qwen_05":5e8,"qwen_15":1.5e9,
    "mistral7b":7e9,
    "stablelm":1.6e9,"tinyllama":1.1e9,"smol_17":1.7e9,"phi15":1.3e9,"phi1":1.3e9,
}
def family(t):
    if t.startswith("py") and not t.startswith("phi"): return "Pythia"
    if t.startswith("bloom"): return "BLOOM"
    if t.startswith("qwen"): return "Qwen2.5"
    if t.startswith("neo") or t.startswith("gptneo"): return "GPT-Neo"
    if "mistral" in t: return "Mistral"
    return "Other"

# Build records (skip IFF controls n_col=0 since R=1 trivially)
records = []
for tag, m in META.items():
    if tag not in combined: continue
    rs = combined[tag]["R"]
    nc = combined[tag]["ncol"]
    if not rs or nc is None or nc <= 0.5: continue
    records.append({
        "tag": tag, "fam": family(tag),
        "log_N": np.log10(m),
        "n_col": nc,
        "mature_R": float(np.mean(rs)),
        "R_n": len(rs),
    })
print(f"Records: {len(records)} models with n_col>0 and ≥1 seed")
for r in sorted(records, key=lambda x: x["log_N"]):
    print(f"  {r['tag']:12s} fam={r['fam']:8s} logN={r['log_N']:5.2f} ncol={r['n_col']:5.1f} mature_R={r['mature_R']:5.2f} (n={r['R_n']})")

X_logN = np.array([[r["log_N"]] for r in records])
X_full = np.array([[r["log_N"], r["n_col"]] for r in records])
y      = np.array([r["mature_R"] for r in records])

def loo_eval(X, y, label):
    if len(y) < X.shape[1] + 2: return None
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        preds.append(float(m.predict(X[te])[0]))
        trues.append(float(y[te][0]))
    preds, trues = np.array(preds), np.array(trues)
    r2 = 1 - np.sum((trues - preds)**2) / np.sum((trues - np.mean(trues))**2)
    rmse = np.sqrt(np.mean((trues - preds)**2))
    rp, pp = stats.pearsonr(trues, preds)
    return {"preds": preds, "trues": trues, "r2": r2, "rmse": rmse,
            "pearson_r": rp, "pearson_p": pp, "label": label}

res_logN = loo_eval(X_logN, y, "log(N) only")
res_full = loo_eval(X_full, y, "log(N) + n_col")
res_ncol = loo_eval(np.array([[r["n_col"]] for r in records]), y, "n_col only")

print(f"\nLOO results (n={len(records)}):")
for r in (res_logN, res_ncol, res_full):
    if r:
        print(f"  {r['label']:25s} R^2={r['r2']:.3f} RMSE={r['rmse']:.3f} Pearson r={r['pearson_r']:.3f} p={r['pearson_p']:.2g}")

# ============================================================
# Figure 8
# ============================================================
fig = plt.figure(figsize=(7.2, 5.8))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
              left=0.10, right=0.97, top=0.93, bottom=0.10)

# (a) LOO scatter for full model (log N + n_col)
ax_a = fig.add_subplot(gs[0, 0])
style_axes(ax_a)
panel_label(ax_a, "a", x=-0.18, y=1.06)
preds, trues = res_full["preds"], res_full["trues"]
diag = [min(min(preds), min(trues))-0.2, max(max(preds), max(trues))+0.2]
ax_a.plot(diag, diag, '--', color='#888', lw=0.7, label="$y=x$ ideal")
for i, r in enumerate(records):
    if r["fam"] == "Pythia":
        col = PYTHIA_VIRIDIS.get(r["tag"], "#1f77b4")
    else:
        col = FAMILY_COLORS.get(r["fam"], "#888")
    ax_a.scatter(trues[i], preds[i], s=55, c=col,
                 edgecolor='black', linewidth=0.4, alpha=0.9, zorder=3)
ax_a.set_xlabel(r"observed mature $R$")
ax_a.set_ylabel(r"LOO-predicted mature $R$")
# cycle 74: italic non-monospace, capital P/N, n_col with mathrm
# cycle 75 (PI feedback): moved stat box from upper-left to lower-right
# to stop covering the leftmost purple Pythia-70M circle near observed R≈0.7.
ax_a.text(0.96, 0.04,
          f"log $N$ + $n_\\mathrm{{col}}$\nLOO $R^2$ = {res_full['r2']:.3f}\nPearson $r$ = {res_full['pearson_r']:.3f}\nRMSE = {res_full['rmse']:.3f}\n$N$ = {len(records)} models",
          transform=ax_a.transAxes, va='bottom', ha='right', fontsize=6.5, style='italic',
          bbox=dict(facecolor='white', edgecolor='#aaa', linewidth=0.4, pad=3))
# cycle 75: legend was lower-right → moved to upper-left (vacated by stat box)
ax_a.legend(loc='upper left', fontsize=6.5, frameon=False)

# (b) Compare R^2 across feature sets
ax_b = fig.add_subplot(gs[0, 1])
style_axes(ax_b)
panel_label(ax_b, "b", x=-0.18, y=1.06)
labels_short = ["log $N$\nonly", "$n_{col}$\nonly", "log $N$\n+ $n_{col}$"]
r2s = [res_logN["r2"], res_ncol["r2"], res_full["r2"]]
colors = ["#7f7f7f", "#1f77b4", "#d62728"]
xpos = np.arange(3)
ax_b.bar(xpos, r2s, color=colors, alpha=0.85,
         edgecolor='black', linewidth=0.4)
for i, r2 in enumerate(r2s):
    ax_b.text(i, r2 + 0.02, f"{r2:.3f}", ha='center', fontsize=8)
ax_b.set_xticks(xpos)
ax_b.set_xticklabels(labels_short, fontsize=7)
ax_b.set_ylabel(r"LOO $R^2$")
ax_b.axhline(0, color='#888', lw=0.5)
ax_b.set_ylim(-0.1, 1.05)

# (c) Residuals vs log_N: is the SCB-augmented residual smaller?
ax_c = fig.add_subplot(gs[1, 0])
style_axes(ax_c)
panel_label(ax_c, "c", x=-0.18, y=1.06)
res_logN_only = np.abs(res_logN["preds"] - res_logN["trues"])
res_full_abs  = np.abs(res_full["preds"] - res_full["trues"])
xpos = np.arange(len(records))
order = np.argsort([r["log_N"] for r in records])
ax_c.plot(xpos, res_logN_only[order], 'o-', color="#7f7f7f", lw=1.2,
          markersize=5, label="log $N$ only")
ax_c.plot(xpos, res_full_abs[order], 's-', color="#d62728", lw=1.2,
          markersize=5, label="log $N$ + $n_{col}$")
ax_c.set_xticks(xpos)
ax_c.set_xticklabels([pretty_name(records[i]["tag"]) for i in order],
                     rotation=35, ha='right', fontsize=6)
ax_c.set_ylabel(r"|prediction error| in $R$")
ax_c.set_ylim(0, max(max(res_logN_only.max(), res_full_abs.max()) * 1.4, 1.0))
ax_c.legend(loc='upper left', fontsize=6.5, frameon=True,
            facecolor='white', edgecolor='#aaa', framealpha=0.95)

# (d) Same-family vs cross-family LOO R^2 comparison
ax_d = fig.add_subplot(gs[1, 1])
style_axes(ax_d)
panel_label(ax_d, "d", x=-0.18, y=1.06)

# Compute per-family same-family LOO R^2 (when at least 3 models in family)
fam_results = {}
all_fams = sorted(set(r["fam"] for r in records))
for fam in all_fams:
    fam_records = [r for r in records if r["fam"] == fam]
    if len(fam_records) < 3: continue
    X = np.array([[r["log_N"], r["n_col"]] for r in fam_records])
    y = np.array([r["mature_R"] for r in fam_records])
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(X):
        if len(np.unique(X[tr])) < 2: continue
        m = LinearRegression().fit(X[tr], y[tr])
        preds.append(float(m.predict(X[te])[0])); trues.append(float(y[te][0]))
    if len(preds) >= 3:
        preds, trues = np.array(preds), np.array(trues)
        if np.var(trues) > 1e-6:
            r2 = 1 - np.sum((trues - preds)**2) / np.sum((trues - np.mean(trues))**2)
            fam_results[fam] = {"r2": r2, "n": len(fam_records)}

# Cross-family: train on all-but-one-family, test on held-out family
cross_results = {}
for fam in all_fams:
    train_records = [r for r in records if r["fam"] != fam]
    test_records  = [r for r in records if r["fam"] == fam]
    if len(train_records) < 3 or len(test_records) < 1: continue
    X_tr = np.array([[r["log_N"], r["n_col"]] for r in train_records])
    y_tr = np.array([r["mature_R"] for r in train_records])
    X_te = np.array([[r["log_N"], r["n_col"]] for r in test_records])
    y_te = np.array([r["mature_R"] for r in test_records])
    m = LinearRegression().fit(X_tr, y_tr)
    preds = m.predict(X_te)
    if np.var(y_te) > 1e-6 and len(y_te) >= 2:
        r2 = 1 - np.sum((y_te - preds)**2) / np.sum((y_te - np.mean(y_te))**2)
    else:
        r2 = 1 - np.sum((y_te - preds)**2) / max(1e-3, np.sum((y_te - np.mean(y))**2))
    cross_results[fam] = {"r2": r2, "n": len(test_records), "rmse": float(np.sqrt(np.mean((y_te - preds)**2)))}

# Plot grouped bars
fams_with_both = [f for f in all_fams if f in cross_results and (f in fam_results or len(records) > 0)]
fams_with_both = [f for f in fams_with_both if cross_results.get(f) is not None]
if not fams_with_both:
    fams_with_both = list(cross_results.keys())[:5]

xpos = np.arange(len(fams_with_both))
w = 0.38
same_r2 = [fam_results[f]["r2"] if f in fam_results else np.nan for f in fams_with_both]
cross_r2 = [cross_results[f]["r2"] for f in fams_with_both]
cross_rmse = [cross_results[f]["rmse"] for f in fams_with_both]

# Plot RMSE instead of clipped R^2 (cleaner)
ax_d.bar(xpos, cross_rmse, color="#d62728", alpha=0.85,
         edgecolor='black', linewidth=0.4)
for i, (rmse, n) in enumerate(zip(cross_rmse, [cross_results[f]["n"] for f in fams_with_both])):
    ax_d.text(i, rmse + 0.015, f"$n$={n}", ha='center', fontsize=6.5)
ax_d.set_xticks(xpos)
ax_d.set_xticklabels(fams_with_both, rotation=20, ha='right', fontsize=6.5)
ax_d.set_ylabel(r"family-hold-out RMSE")
mean_rmse = float(np.mean(cross_rmse))
ax_d.axhline(mean_rmse, color='black', ls='--', lw=0.6,
             label=f"mean RMSE = {mean_rmse:.2f}")
ax_d.set_xlabel("held-out family (train on others)")
ax_d.set_ylim(0, max(cross_rmse) * 1.25 + 0.1)
ax_d.legend(loc='upper right', fontsize=6.5, frameon=True,
            facecolor='white', edgecolor='#aaa', framealpha=0.95)

# NMI convention: figure-number sentence in LaTeX caption only.
for out_dir in (OUT, OUT2):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Figure_8_Predictor.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "Figure_8_Predictor.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "Figure_8_Predictor.svg"))
print(f"\nFig 8 saved ({os.path.getsize(os.path.join(OUT, 'Figure_8_Predictor.png'))//1024} KB)")
