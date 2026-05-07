"""
Application A demo: SCB onset is an early-warning signal for ICL maturity.
For each Pythia scale, compute:
  - step at which n_col first reaches ≥5 (SCB onset)
  - step at which R first reaches 90% of mature R (ICL benchmark maturity)
  - lead factor = compute saved by relying on SCB onset
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
# cycle 75: dual-write to v45_CURRENT/figures/ to fix supp.pdf sync.
OUT2 = r"C:\Users\Jie\Desktop\Bio-LM_v45_CURRENT\figures"

traj = np.load(os.path.join(DATA, "fig5_traj.npz"), allow_pickle=True)
ncol = np.load(os.path.join(DATA, "fig5_ncol.npz"), allow_pickle=True)

# Build per-scale (step, R) and per-scale (step, n_col)
data_R = defaultdict(dict)
for i, (s, st, abl) in enumerate(traj["keys"]):
    rs = [r for r in traj["R_lists"][i] if np.isfinite(r) and -5 <= r <= 10]
    if rs and str(abl) == "leading":
        data_R[(str(s))][int(st)] = float(np.mean(rs))

ncol_d = defaultdict(dict)
for i, (s, st) in enumerate(ncol["keys"]):
    ncol_d[str(s)][int(st)] = float(ncol["nco_means"][i])

scales = ["py70m","py160m","py410m","py1b","py14b","py28b","py69b"]
labels = {"py70m":"Pythia-70M","py160m":"Pythia-160M","py410m":"Pythia-410M",
          "py1b":"Pythia-1B","py14b":"Pythia-1.4B","py28b":"Pythia-2.8B","py69b":"Pythia-6.9B"}

results = []
for s in scales:
    if s not in data_R or s not in ncol_d: continue
    R_steps = sorted(data_R[s].items())
    nc_steps = sorted(ncol_d[s].items())
    if len(R_steps) < 4 or len(nc_steps) < 4: continue
    # mature R = median of last 5 steps
    R_arr = np.array([x[1] for x in R_steps])
    mature_R = float(np.median(R_arr[-5:]))
    if mature_R - 1 < 0.1: continue  # skip non-amplifying models
    # ICL maturity step: first step where R >= mature_R - 0.1*(mature_R - 1)
    target_R = mature_R - 0.1 * (mature_R - 1)
    icl_mature_step = None
    for st, R in R_steps:
        if R >= target_R:
            icl_mature_step = st; break
    # SCB onset step: first step where n_col >= 5
    scb_onset_step = None
    for st, nc in nc_steps:
        if nc >= 5:
            scb_onset_step = st; break
    if icl_mature_step and scb_onset_step and scb_onset_step > 0:
        lead = icl_mature_step / scb_onset_step
        results.append({
            "scale": s, "label": labels[s],
            "scb_onset": scb_onset_step,
            "icl_mature": icl_mature_step,
            "lead": lead,
            "mature_R": mature_R,
        })
        print(f"  {s}: SCB onset @ step {scb_onset_step}, ICL @ step {icl_mature_step}, lead = {lead:.2f}x, mature_R = {mature_R:.2f}")

if not results:
    print("ERROR: no valid scales")
    sys.exit(0)

leads = [r["lead"] for r in results]
print(f"\nLead factor: median={np.median(leads):.2f}x, range [{min(leads):.2f}, {max(leads):.2f}]")

# ============================================================
fig = plt.figure(figsize=(7.2, 4.5))
gs = GridSpec(1, 2, figure=fig, hspace=0.32, wspace=0.34,
              left=0.13, right=0.97, top=0.86, bottom=0.13)

# (a) SLOPEGRAPH — per Pythia scale: SCB onset step (blue dot) -> ICL
#     mature step (red dot), connected by a horizontal line whose length
#     IS the lead factor. Visually superior to paired bars: reader sees
#     simultaneously (i) when SCB forms, (ii) when ICL matures, (iii)
#     the gap (lead factor) for each scale, all on one log-time axis.
ax_a = fig.add_subplot(gs[0, 0])
style_axes(ax_a)
panel_label(ax_a, "a", x=-0.20, y=1.06)
ypos = np.arange(len(results))
for i, r in enumerate(results):
    onset = r["scb_onset"]; mature = r["icl_mature"]
    # connecting line (gradient: blue at onset → red at mature via grey midpoint)
    ax_a.plot([onset, mature], [i, i], '-', color="#888", lw=1.2,
              alpha=0.6, zorder=1)
    # SCB onset dot (blue)
    ax_a.scatter([onset], [i], s=80, color="#1f77b4",
                 edgecolor='black', linewidth=0.6, zorder=3)
    # ICL mature dot (red)
    ax_a.scatter([mature], [i], s=80, color="#d62728",
                 edgecolor='black', linewidth=0.6, zorder=3)
    # Lead factor annotation at midpoint, above line
    mid_log = 10 ** ((np.log10(onset) + np.log10(mature)) / 2)
    ax_a.text(mid_log, i + 0.20, f"{r['lead']:.1f}×",
              ha='center', va='bottom', fontsize=6,
              color="#444", style='italic')
ax_a.set_xscale('log')
ax_a.set_yticks(ypos)
ax_a.set_yticklabels([r["label"].replace("Pythia-", "") for r in results],
                     fontsize=7)
ax_a.set_xlabel("training step")
ax_a.set_ylabel("Pythia scale")
ax_a.set_ylim(-0.7, len(results) - 0.3)
ax_a.invert_yaxis()  # smallest scale at top
# Legend ABOVE axes
import matplotlib.lines as mlines
h_onset = mlines.Line2D([], [], color="#1f77b4", marker='o',
                        markeredgecolor='black', linestyle='None',
                        markersize=8, label="SCB onset ($n_{\\mathrm{col}}\\geq 5$)")
h_mature = mlines.Line2D([], [], color="#d62728", marker='o',
                         markeredgecolor='black', linestyle='None',
                         markersize=8, label="ICL mature (within 10% of $R_{\\mathrm{mature}}$)")
ax_a.legend(handles=[h_onset, h_mature],
            loc='lower left', bbox_to_anchor=(0.0, 1.02),
            ncol=2, fontsize=6, frameon=False,
            handlelength=1.5, handletextpad=0.4, columnspacing=1.5)

# (b) Lead factor per scale
ax_b = fig.add_subplot(gs[0, 1])
style_axes(ax_b)
panel_label(ax_b, "b", x=-0.18, y=1.06)
for r in results:
    col = PYTHIA_VIRIDIS.get(r["scale"], "#888")
    ax_b.bar(r["scale"], r["lead"], color=col, alpha=0.9,
             edgecolor='black', linewidth=0.4)
    ax_b.text(r["scale"], r["lead"] + 0.1, f"{r['lead']:.1f}×",
              ha='center', fontsize=7)
ax_b.axhline(1, color='#888', ls=':', lw=0.5)
median_lead = np.median(leads)
ax_b.axhline(median_lead, color='#d62728', ls='--', lw=0.7,
             label=f"median = {median_lead:.1f}×")
ax_b.set_xticklabels([r["label"].replace("Pythia-", "") for r in results],
                     rotation=20, ha='right', fontsize=7)
ax_b.set_ylabel("lead factor (compute saved by SCB monitor)")
ax_b.legend(loc='upper right', fontsize=7)
ax_b.set_ylim(0, max(leads) * 1.3)

for out_dir in (OUT, OUT2):
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Figure_AppA_Lead.png"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "Figure_AppA_Lead.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "Figure_AppA_Lead.svg"))
print(f"\nApp A figure saved ({os.path.getsize(os.path.join(OUT, 'Figure_AppA_Lead.png'))//1024} KB)")
