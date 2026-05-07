"""Merge INN aggregated PR profiles into fig123_alllayer_extended.npz.

Strategy:
- Load existing local extended npz (LEO5 + cycle-32 = 36 models)
- Load INN aggregated npz (49 models, INN data)
- Dedupe by canonicalized model_id, prefer INN multi-seed averaged PR profile
  for shared models (better quality than single-seed)
- Add new INN-only models (vision Transformers, etc.)
- Output: fig123_alllayer_v3_full.npz with deduped + extended set
"""
import os
import numpy as np

DATA = r"C:/Users/Jie/Desktop/BioLM_Scaling_Physics/12_data_aggregates"


def canonicalize(mid: str) -> str:
    """Normalize various spellings to canonical key."""
    s = mid.strip().lower()
    # remove HF org prefix variations
    s = s.replace("eleutherai/", "").replace("facebookai/", "").replace("facebook/", "")
    s = s.replace("google-bert/", "").replace("distilbert/", "")
    s = s.replace("microsoft/", "").replace("bigscience/", "")
    s = s.replace("stabilityai/", "").replace("huggingfacetb/", "")
    s = s.replace("instadeepai/", "").replace("hugohrban/", "")
    # normalize separators
    s = s.replace("_", "-").replace(" ", "-")
    s = s.replace("--", "-")
    return s


def n_col_from_prs(prs):
    L = len(prs)
    if L < 4:
        return 0
    lo, hi = int(0.25 * L), int(0.95 * L)
    band = np.array(prs[lo:hi+1])
    in_band = band < 5
    max_run = cur = 0
    for x in in_band:
        if x:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return int(max_run)


def main():
    # Load existing
    A_local = np.load(os.path.join(DATA, "fig123_alllayer_extended.npz"), allow_pickle=True)
    A_inn = np.load(os.path.join(DATA, "inn_alllayer_v3.npz"), allow_pickle=True)
    print(f"Local extended: {len(A_local['model_ids'])} models")
    print(f"INN: {len(A_inn['model_ids'])} models")

    # Index local by canon key
    local_idx = {}
    for i, m in enumerate(A_local["model_ids"]):
        local_idx[canonicalize(str(m))] = i
    inn_idx = {}
    for i, m in enumerate(A_inn["model_ids"]):
        inn_idx[canonicalize(str(m))] = i

    # Build merged: for shared keys prefer INN (multi-seed); for local-only keep
    final_keys = set(local_idx) | set(inn_idx)
    merged = {
        "model_ids": [], "layers": [], "prs": [],
        "n_cols": [], "n_params": [], "n_layers_total": [],
        "modality": [], "architecture": [], "n_seeds": [], "source": [],
    }
    inn_n_seeds = A_inn["n_seeds"] if "n_seeds" in A_inn.files else None

    for k in sorted(final_keys):
        if k in inn_idx:
            i = inn_idx[k]
            mid = str(A_inn["model_ids"][i])
            prs = A_inn["prs"][i]
            # Recompute n_col with consistent L (skip if too short)
            if len(prs) < 4:
                continue
            n_col_clean = n_col_from_prs(prs)
            # Cap n_col at L (sanity check on aggregator bug)
            if n_col_clean > len(prs):
                n_col_clean = len(prs)
            merged["model_ids"].append(mid)
            merged["layers"].append(np.arange(len(prs), dtype=int))
            merged["prs"].append(np.asarray(prs, dtype=float))
            merged["n_cols"].append(n_col_clean)
            merged["n_params"].append(float(A_inn["n_params"][i]))
            merged["n_layers_total"].append(int(A_inn["n_layers_total"][i]))
            merged["modality"].append(str(A_inn["modality"][i]))
            merged["architecture"].append(str(A_inn["architecture"][i]) or "decoder")
            merged["n_seeds"].append(int(inn_n_seeds[i]) if inn_n_seeds is not None else 1)
            merged["source"].append("INN")
        elif k in local_idx:
            i = local_idx[k]
            mid = str(A_local["model_ids"][i])
            prs = A_local["prs"][i]
            merged["model_ids"].append(mid)
            merged["layers"].append(np.asarray(A_local["layers"][i], dtype=int))
            merged["prs"].append(np.asarray(prs, dtype=float))
            merged["n_cols"].append(int(A_local["n_cols"][i]))
            merged["n_params"].append(float(A_local["n_params"][i]))
            merged["n_layers_total"].append(int(A_local["n_layers_total"][i]))
            merged["modality"].append(str(A_local["modality"][i]))
            merged["architecture"].append(str(A_local["architecture"][i]))
            merged["n_seeds"].append(1)
            merged["source"].append("LEO5")

    out = os.path.join(DATA, "fig123_alllayer_v3_full.npz")
    np.savez_compressed(
        out,
        **{k: np.array(v, dtype=object if k in
                       ("model_ids", "layers", "prs", "modality", "architecture", "source")
                       else (int if k in ("n_cols", "n_layers_total", "n_seeds") else float))
           for k, v in merged.items()},
    )

    print(f"Wrote {out} with {len(merged['model_ids'])} models")
    from collections import Counter
    print(f"  by source: {dict(Counter(merged['source']))}")
    print(f"  by architecture: {dict(Counter(merged['architecture']))}")
    print(f"  by modality: {dict(Counter(merged['modality']))}")
    print(f"  n_col=0: {sum(1 for n in merged['n_cols'] if n == 0)}")
    print(f"  n_col>0: {sum(1 for n in merged['n_cols'] if n > 0)}")

    print(f"\n=== All models with n_col>0 ===")
    rows = sorted(zip(merged["model_ids"], merged["n_cols"],
                      merged["modality"], merged["architecture"],
                      merged["n_layers_total"], merged["source"]),
                  key=lambda x: -x[1])
    for mid, nc, mod, arch, L, src in rows:
        if nc > 0:
            print(f"  {mid:<60s} L={L:>3} n_col={nc:>3} {mod:<8} {arch:<8} {src}")

    print(f"\n=== All models with n_col=0 ===")
    for mid, nc, mod, arch, L, src in rows:
        if nc == 0:
            print(f"  {mid:<60s} L={L:>3} {mod:<8} {arch:<8} {src}")


if __name__ == "__main__":
    main()
