"""Merge cycle-74 PR-profile JSONs into fig123_alllayer.npz.

Reads existing fig123_alllayer.npz + all pr_profile_*.json under
12_data_aggregates/pr_profile/ and writes fig123_alllayer_extended.npz
with deduped models (existing fig123 entries take precedence on duplicate
model_id).
"""
import json
import os
import numpy as np

DATA = r"C:/Users/Jie/Desktop/BioLM_Scaling_Physics/12_data_aggregates"

# Load existing fig123_alllayer
A = np.load(os.path.join(DATA, "fig123_alllayer.npz"), allow_pickle=True)
existing_ids = set(str(m) for m in A["model_ids"])
print(f"fig123_alllayer.npz: {len(existing_ids)} existing models")

model_ids = list(A["model_ids"])
layers = list(A["layers"])
prs = list(A["prs"])
n_cols = list(A["n_cols"])
n_params = list(A["n_params"])
n_layers_total = list(A["n_layers_total"])
modality = list(A["modality"])
architecture = list(A["architecture"])

added = 0
pr_dir = os.path.join(DATA, "pr_profile")
for fn in sorted(os.listdir(pr_dir)):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(pr_dir, fn)) as f:
        d = json.load(f)
    mid = d["model"]
    if mid in existing_ids:
        print(f"  skip duplicate: {mid}")
        continue
    model_ids.append(mid)
    layers.append(np.array(d["layers"], dtype=int))
    prs.append(np.array(d["prs"], dtype=float))
    n_cols.append(int(d["n_col"]))
    n_params.append(float(d["n_params"]))
    n_layers_total.append(int(d["n_layers_total"]))
    modality.append(d["modality"])
    architecture.append(d["architecture"])
    existing_ids.add(mid)
    added += 1
    print(f"  + {mid}: L={d['n_layers_total']} n_col={d['n_col']} arch={d['architecture']}")

print(f"\nAdded {added} new models, total now {len(model_ids)}")

out = os.path.join(DATA, "fig123_alllayer_extended.npz")
np.savez_compressed(
    out,
    model_ids=np.array(model_ids, dtype=object),
    layers=np.array(layers, dtype=object),
    prs=np.array(prs, dtype=object),
    n_cols=np.array(n_cols, dtype=int),
    n_params=np.array(n_params, dtype=float),
    n_layers_total=np.array(n_layers_total, dtype=int),
    modality=np.array(modality, dtype=object),
    architecture=np.array(architecture, dtype=object),
)
print(f"Wrote {out}")

# Breakdown by arch
from collections import Counter
arch_count = Counter(architecture)
mod_count = Counter(modality)
print(f"\nFinal breakdown:")
print(f"  by arch: {dict(arch_count)}")
print(f"  by modality: {dict(mod_count)}")
print(f"  models with n_col=0: {sum(1 for n in n_cols if n == 0)}")
print(f"  models with n_col>0: {sum(1 for n in n_cols if n > 0)}")
