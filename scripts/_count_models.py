"""Count unique models per panel + union total."""
import numpy as np
import os, json, glob

DATA = r"C:\Users\Jie\Desktop\BioLM_Scaling_Physics\12_data_aggregates"

# ── fig123_alllayer (architecture scan, used in Fig 1 / ED1) ─────────────
A = np.load(os.path.join(DATA, "fig123_alllayer.npz"), allow_pickle=True)
arch = set(str(m) for m in A["model_ids"])
print(f"Architecture scan (fig123_alllayer): {len(arch)} models")

# ── fig4 IFF panel ──────────────────────────────────────────────────────
iff_lnz = np.load(os.path.join(DATA, "fig4_iff_lnz.npz"), allow_pickle=True)
iff_inn = np.load(os.path.join(DATA, "fig4_iff_inn.npz"), allow_pickle=True)
iff_tags = set()
for arr in (iff_lnz, iff_inn):
    for tag, abl in arr["keys"]:
        iff_tags.add(str(tag))
print(f"IFF panel (fig4_iff_lnz+inn union): {len(iff_tags)} unique tags")

# ── fig3_phase2 ablation panel ───────────────────────────────────────────
phase = np.load(os.path.join(DATA, "fig3_phase2.npz"), allow_pickle=True)
phase_tags = set(str(k[0]) for k in phase["keys"])
print(f"Phase 2 ablation (fig3_phase2): {len(phase_tags)} tags")

# ── fig5 Pythia trajectory ──────────────────────────────────────────────
traj = np.load(os.path.join(DATA, "fig5_traj.npz"), allow_pickle=True)
traj_tags = set(str(k[0]) for k in traj["keys"])
print(f"Trajectory (fig5_traj): {len(traj_tags)} Pythia scales")

# ── Classifier ──────────────────────────────────────────────────────────
def classify(tag):
    t = tag.lower()
    if any(x in t for x in ['esm', 'bert', 'roberta', 'nucleotide', 'nuct']):
        return 'encoder_MLM'
    if any(x in t for x in ['vit', 'dino', 'mae', 'clip']):
        return 'vision'
    return 'causal_LM'

print()
print("=== Architecture-scan (fig123_alllayer) breakdown ===")
from collections import Counter
arch_breakdown = Counter(classify(m) for m in arch)
for k, v in arch_breakdown.most_common():
    print(f"  {k}: {v}")

print()
print("=== IFF panel breakdown (fig4 union) ===")
iff_breakdown = Counter(classify(m) for m in iff_tags)
for k, v in iff_breakdown.most_common():
    print(f"  {k}: {v}")
for m in sorted(iff_tags):
    print(f"    {m}")

print()
print("=== Phase 2 panel breakdown (fig3) ===")
phase_breakdown = Counter(classify(m) for m in phase_tags)
for k, v in phase_breakdown.most_common():
    print(f"  {k}: {v}")

# Union of everything probed
all_tags = arch | iff_tags | phase_tags | traj_tags
print()
print(f"=== UNION across all panels: {len(all_tags)} models ===")
all_breakdown = Counter(classify(m) for m in all_tags)
for k, v in all_breakdown.most_common():
    print(f"  {k}: {v}")
