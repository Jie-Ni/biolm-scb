"""Aggregate MUSICA-INN per-layer JSON records into a comprehensive npz.

INN's alllayer JSONs are LISTS of per-layer dicts. Each dict has:
  model_id (or model), hidden_dim, layer_idx, n_layers_total,
  op2_repr_cov_pr (the participation ratio), modality, architecture,
  random_init, n_params, sigma, seed.

This aggregator dedupes (model_id, seed) pairs and assembles per-model
PR profiles compatible with the fig123_alllayer.npz schema:
  model_ids, layers, prs, n_cols, n_params, n_layers_total, modality,
  architecture.

Run on INN:
  python _inn_aggregator.py --root /scratch/fs201130/jn20658/TurboGNN/inn_polite/runs \
      --out /scratch/fs201130/jn20658/TurboGNN/inn_polite/aggregates/inn_alllayer_v3.npz
"""
import argparse
import json
import os
import re
from collections import defaultdict
import numpy as np


def n_col_from_prs(prs):
    """Maximum contiguous run of PR<5 in mid-band [0.25L, 0.95L]."""
    L = len(prs)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include", default="alllayer", help="Substring filter on subdir/filename")
    args = ap.parse_args()

    # per (model_id, seed) → list of (layer_idx, pr)
    per_run = defaultdict(list)
    meta = {}
    n_scanned = 0
    n_matched = 0
    skipped_reasons = defaultdict(int)

    for dirpath, _, files in os.walk(args.root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            if args.include and args.include not in (dirpath + "/" + fn):
                continue
            path = os.path.join(dirpath, fn)
            n_scanned += 1
            try:
                with open(path) as f:
                    d = json.load(f)
            except Exception as e:
                skipped_reasons["json_error"] += 1
                continue
            if not isinstance(d, list):
                skipped_reasons["not_list"] += 1
                continue
            for entry in d:
                if not isinstance(entry, dict):
                    continue
                pr = entry.get("op2_repr_cov_pr")
                if pr is None:
                    continue
                if isinstance(pr, dict):
                    pr = pr.get("value")
                    if pr is None:
                        continue
                mid = entry.get("model_id") or entry.get("model")
                if mid is None:
                    continue
                seed = entry.get("seed", 0)
                layer_idx = entry.get("layer_idx")
                n_layers_total = entry.get("n_layers_total")
                if layer_idx is None or n_layers_total is None:
                    continue
                key = (mid, seed)
                per_run[key].append((int(layer_idx), float(pr)))
                if mid not in meta:
                    meta[mid] = {
                        "n_params": entry.get("n_params"),
                        "hidden_dim": entry.get("hidden_dim"),
                        "modality": entry.get("modality"),
                        "architecture": entry.get("architecture"),
                        "n_layers_total": n_layers_total,
                    }
                n_matched += 1

    print(f"Scanned {n_scanned} files, found {n_matched} per-layer records, "
          f"{len(per_run)} (model, seed) pairs.")
    if skipped_reasons:
        print(f"Skipped reasons: {dict(skipped_reasons)}")

    # Aggregate per model: average across seeds
    per_model = defaultdict(list)
    for (mid, seed), pts in per_run.items():
        pts.sort()
        prs = np.array([p for _, p in pts], dtype=float)
        per_model[mid].append(prs)

    # Build output arrays
    model_ids = []
    layers_list = []
    prs_list = []
    n_cols = []
    n_params_list = []
    n_layers_total_list = []
    modality_list = []
    architecture_list = []
    n_seeds_list = []

    for mid, runs in per_model.items():
        L = min(len(r) for r in runs)
        if L < 4:
            continue  # skip too-short profiles
        # Truncate to common L, then average
        mat = np.stack([r[:L] for r in runs])
        prs_mean = mat.mean(axis=0)
        m = meta[mid]
        model_ids.append(mid)
        layers_list.append(np.arange(L).astype(int))
        prs_list.append(prs_mean)
        n_cols.append(n_col_from_prs(prs_mean))
        n_params_list.append(float(m.get("n_params") or 0))
        n_layers_total_list.append(int(m.get("n_layers_total") or L))
        modality_list.append(str(m.get("modality") or ""))
        architecture_list.append(str(m.get("architecture") or ""))
        n_seeds_list.append(len(runs))

    print(f"Aggregated {len(model_ids)} models with PR profiles")

    from collections import Counter
    arch = Counter(architecture_list)
    mod = Counter(modality_list)
    print(f"  by arch: {dict(arch)}")
    print(f"  by modality: {dict(mod)}")
    print(f"  models with n_col=0: {sum(1 for n in n_cols if n == 0)}")
    print(f"  models with n_col>0: {sum(1 for n in n_cols if n > 0)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        model_ids=np.array(model_ids, dtype=object),
        layers=np.array(layers_list, dtype=object),
        prs=np.array(prs_list, dtype=object),
        n_cols=np.array(n_cols, dtype=int),
        n_params=np.array(n_params_list, dtype=float),
        n_layers_total=np.array(n_layers_total_list, dtype=int),
        modality=np.array(modality_list, dtype=object),
        architecture=np.array(architecture_list, dtype=object),
        n_seeds=np.array(n_seeds_list, dtype=int),
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
