"""LEO5: per-layer PR profile probe for any HF model.

Computes the participation ratio PR_l = (sum lambda_i)^2 / sum lambda_i^2
of the residual-stream covariance at each layer, where lambda_i are the
eigenvalues of the d x d covariance matrix collected from N probe sequences.

Output JSON schema (matches what fig123_alllayer.npz expects):
  model: str
  seed: int
  layers: list[int]
  prs: list[float]    # PR per layer
  n_col: int          # max contiguous count of layers with PR<5 in [0.25L, 0.95L]
  n_params: int
  n_layers_total: int
  modality: str
  architecture: str   # "encoder" or "decoder"
  hidden: int
  wall_seconds: float

Usage:
  python _leo5_pr_probe.py --model bert-base-uncased --seed 0 \
      --out runs/pr_profile/ --task-class text --architecture encoder
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch

# 200 OpenWebText-like probe sequences
TEXT_PROBES = [
    "The capital of France is Paris, the largest city in the country.",
    "Quantum mechanics describes physical phenomena at atomic and subatomic scales.",
    "The patient presented with elevated blood glucose and HbA1c above 7.0 percent.",
    "Climate change refers to long-term shifts in global weather patterns.",
    "Machine learning is a subset of artificial intelligence focused on data.",
    "She walked into the room and noticed the lights were already on.",
    "The mitochondria is the powerhouse of the cell, producing ATP via oxidative phosphorylation.",
    "Reading a good book can be one of the most enjoyable ways to spend an afternoon.",
] * 25  # 200 sequences total (8 unique * 25)

PROTEIN_PROBES = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "MQTPQQRTPGQPGSRSLLAGAPVDAAATGGRSMLLLHGPGRPKVLAGGSGSLASVLR",
    "MAEEELDLQYVHHGAEAGAGSLGRGAAEAEAAEAAEAAFEEEHAAEAAVAAAEAAAARRRPPP",
    "MVDLENQQEELEDPNAIVPESRPGGTLGEKARERSVQGGYLPRDLRTRGGRPGNEGYV",
    "MASSSSGCRSAAAAAVRASGLAARRAVRSAVRQRPLLAGTAAAFAKVRGAAATKVLGTGRSPL",
    "MLLLAALLAGAVCAAVTSAFAALVAEGNRRSPLNFSCEDFEKAKKILQCLDGTPTDDP",
    "MTGGAEKEEDIGVPGGGFKKMKKLLDLLDDEGDEAERRREKLEKEKEEEERKKKKE",
    "MERTEALLRPLQIPVGEAVLVPVASGLALAALSEPAAAEAVQAERPGAPLALAALPVAAAQAEH",
] * 25

DNA_PROBES = [
    "ATGGGCAGCAGCCATCATCATCATCATCACAGCAGCGGCCTGGTGCCGCGCGGCAGCCAT",
    "GAATTCGGATCCAGTACTACGGTACGTACGTACGTAGCATGCATGCATGCATGCATGCATG",
    "TGCAGCAGGTGTACAACAACAACAACAACAACAACAACAACAACAACAACAACAACAACAAC",
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC",
    "GCGCGCGCATATATATGCGCGCGCATATATATGCGCGCGCATATATATGCGCGCGCATATAT",
    "AAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTT",
    "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATG",
    "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGAC",
] * 25


def get_probes(task_class):
    if task_class == "protein":
        return PROTEIN_PROBES
    if task_class == "DNA":
        return DNA_PROBES
    return TEXT_PROBES


def collect_hidden_states(model, tokenizer, probes, device, max_length=128):
    """Return list of per-layer hidden-state tensors stacked across all
    probes and token positions. Each layer gives an (N*T, D) matrix."""
    layer_states = None
    model.eval()
    with torch.no_grad():
        for txt in probes:
            inputs = tokenizer(txt, return_tensors="pt", max_length=max_length,
                               truncation=True, padding=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs, output_hidden_states=True)
            # tuple of (n_layers+1) tensors of shape (1, T, D)
            hs = out.hidden_states
            if layer_states is None:
                layer_states = [[] for _ in range(len(hs))]
            for i, h in enumerate(hs):
                # mean over token positions (or store all positions?)
                # Use ALL positions to match cycle-32 protocol
                layer_states[i].append(h.squeeze(0).float().cpu())
    return [torch.cat(ls, dim=0).numpy() for ls in layer_states]


def participation_ratio(states):
    """PR = (sum lambda)^2 / sum lambda^2 where lambda are eigvals of cov."""
    states_centered = states - states.mean(axis=0, keepdims=True)
    cov = states_centered.T @ states_centered / states_centered.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return float("nan")
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/pr_profile/")
    ap.add_argument("--task-class", default="text",
                    choices=["text", "protein", "DNA", "vision"])
    ap.add_argument("--architecture", default="auto",
                    choices=["auto", "encoder", "decoder"])
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--n-probes", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {args.model} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype,
                                       trust_remote_code=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    arch_str = args.architecture
    if arch_str == "auto":
        cls = type(model).__name__.lower()
        arch_str = "encoder" if "encoder" in cls or "bert" in cls or "esm" in cls else "decoder"

    probes = get_probes(args.task_class)[: args.n_probes]
    print(f"  task_class={args.task_class}, architecture={arch_str}, n_probes={len(probes)}")

    layer_states = collect_hidden_states(model, tokenizer, probes, device,
                                          max_length=args.max_length)
    n_layers_total = len(layer_states)
    print(f"  collected {n_layers_total} layer states")

    prs = []
    for i, states in enumerate(layer_states):
        pr = participation_ratio(states)
        prs.append(pr)
        if i % 5 == 0:
            print(f"    layer {i}: states={states.shape}, PR={pr:.2f}")

    # n_col: max contiguous run of PR<5 in [0.25L, 0.95L]
    L = n_layers_total
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
    n_col = int(max_run)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "__")
    out_path = out_dir / f"pr_profile_{safe_name}__seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "seed": args.seed,
            "task_class": args.task_class,
            "architecture": arch_str,
            "modality": args.task_class,
            "layers": list(range(n_layers_total)),
            "prs": prs,
            "n_col": n_col,
            "n_params": n_params,
            "n_layers_total": n_layers_total,
            "hidden": int(layer_states[0].shape[-1]),
            "wall_seconds": time.time() - t0,
        }, f)
    print(f"Wrote {out_path}, n_col={n_col}, wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
