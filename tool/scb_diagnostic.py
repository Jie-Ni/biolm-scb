"""SCB diagnostic — predict mature ICL amplification from any HuggingFace model.

Given a Hugging Face model identifier, this CLI:
  1. Loads the model + tokenizer
  2. Runs the 200-probe participation-ratio scan
  3. Computes n_col (sustained-causal-bottleneck length)
  4. Predicts mature ICL amplification R via the LOO-fit regression
     from the paper (slope=0.064, intercept=0.69 on n_col;
     LOO R^2=0.55 on the 17-model deep-bootstrap panel)
  5. Reports a one-line verdict with V_eff regime classification

Usage:
  scb-diagnostic --model bert-base-uncased
  scb-diagnostic --model EleutherAI/pythia-1b --task-class text
  scb-diagnostic --model facebook/esm2_t12_35M_UR50D --task-class protein

Source: Bio-LM Scaling Physics paper (NMI 2026).
GitHub: https://github.com/Jie-Ni/biolm-scb
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PAPER_REGRESSION = {
    "slope": 0.064,
    "intercept": 0.69,
    "loo_r2": 0.55,
    "n_models_fit": 17,
}

TEXT_PROBES = [
    "The capital of France is Paris.",
    "Quantum mechanics describes physical phenomena at atomic scales.",
    "Climate change refers to long-term shifts in global weather patterns.",
    "Machine learning is a subset of artificial intelligence.",
    "Reading a good book is one of the most enjoyable ways to spend an afternoon.",
    "She walked into the room and noticed the lights were already on.",
    "The mitochondria is the powerhouse of the cell.",
    "If a tree falls in the forest, does it make a sound?",
] * 25
PROTEIN_PROBES = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL",
    "MQTPQQRTPGQPGSRSLLAGAPVDAAATGGRSMLLLHGPGRPKVLAGGSGSLASVLR",
    "MAEEELDLQYVHHGAEAGAGSLGRGAAEAEAAEAAEAAFEEEHAAEAAVAAAEAAAARRR",
    "MVDLENQQEELEDPNAIVPESRPGGTLGEKARERSVQGGYLPRDLRTRGGRPGNEGYV",
    "MASSSSGCRSAAAAAVRASGLAARRAVRSAVRQRPLLAGTAAAFAKVRGAAATKVLG",
    "MLLLAALLAGAVCAAVTSAFAALVAEGNRRSPLNFSCEDFEKAKKILQCLDGTPTDDP",
    "MTGGAEKEEDIGVPGGGFKKMKKLLDLLDDEGDEAERRREKLEKEKEEEERKKKKE",
    "MERTEALLRPLQIPVGEAVLVPVASGLALAALSEPAAAEAVQAERPGAPLALAALPV",
] * 25
DNA_PROBES = [
    "ATGGGCAGCAGCCATCATCATCATCATCACAGCAGCGGCCTGGTGCCGCGCGGCAGC",
    "GAATTCGGATCCAGTACTACGGTACGTACGTACGTAGCATGCATGCATGCATGCATG",
    "TGCAGCAGGTGTACAACAACAACAACAACAACAACAACAACAACAACAACAACAACAAC",
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC",
    "GCGCGCGCATATATATGCGCGCGCATATATATGCGCGCGCATATATATGCGCGCGCAT",
    "AAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTTTAAACCCGGGTT",
    "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATG",
    "ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGAC",
] * 25


def get_probes(task: str):
    return {"text": TEXT_PROBES, "protein": PROTEIN_PROBES,
            "DNA": DNA_PROBES}.get(task, TEXT_PROBES)


def participation_ratio(states):
    sc = states - states.mean(axis=0, keepdims=True)
    cov = sc.T @ sc / sc.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return float("nan")
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


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


@torch.no_grad()
def collect_layer_states(model, tokenizer, probes, device, max_length=128):
    layer_states = None
    for txt in probes:
        inputs = tokenizer(txt, return_tensors="pt", max_length=max_length,
                           truncation=True, padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        if layer_states is None:
            layer_states = [[] for _ in range(len(hs))]
        for i, h in enumerate(hs):
            layer_states[i].append(h.squeeze(0).float().cpu().numpy())
    return [np.concatenate(ls, axis=0) for ls in layer_states]


def predict_R(n_col):
    """Predict mature R from regression on the 17-model panel."""
    return PAPER_REGRESSION["intercept"] + PAPER_REGRESSION["slope"] * n_col


def classify_v_eff(model_id, n_col, L, modality):
    """Heuristic: estimate V_eff regime from n_col + modality."""
    if n_col >= 8:
        regime = "low V_eff (rich SCB)"
    elif n_col >= 3:
        regime = "intermediate V_eff (partial SCB)"
    else:
        regime = "high V_eff (no SCB)"
    return regime


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scb-diagnostic",
                                  description="Predict mature ICL R from any HF model")
    ap.add_argument("--model", required=True, help="HuggingFace model identifier")
    ap.add_argument("--task-class", default="text",
                    choices=["text", "protein", "DNA", "vision"])
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer
    print(f"\n=== SCB diagnostic: {args.model} ===")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype,
                                       trust_remote_code=True).to(device)
    model.eval()

    probes = get_probes(args.task_class)[: args.n_probes]
    print(f"  modality={args.task_class}, n_probes={len(probes)}, device={device}")
    print(f"  probing {len(probes)} sequences ...")

    layer_states = collect_layer_states(model, tokenizer, probes, device,
                                         max_length=args.max_length)
    L = len(layer_states)
    prs = [participation_ratio(s) for s in layer_states]
    n_col = n_col_from_prs(prs)
    R_pred = predict_R(n_col)
    regime = classify_v_eff(args.model, n_col, L, args.task_class)

    elapsed = time.time() - t0
    print(f"\n=== RESULT ===")
    print(f"  L (layers including embedding): {L}")
    print(f"  PR profile: {[f'{p:.1f}' for p in prs]}")
    print(f"  n_col (max contiguous PR<5 in [0.25L, 0.95L]):  {n_col}")
    print(f"  V_eff regime: {regime}")
    print(f"  predicted mature ICL amplification R:  {R_pred:.2f}")
    print(f"  (regression: slope=0.064, intercept=0.69, "
          f"LOO R^2=0.55, fit on 17 SCB-bearing causal LMs)")
    print(f"  elapsed: {elapsed:.1f} sec")

    result = {
        "model": args.model,
        "task_class": args.task_class,
        "L": L,
        "prs": prs,
        "n_col": n_col,
        "predicted_R": R_pred,
        "v_eff_regime": regime,
        "elapsed_seconds": elapsed,
        "regression_source": PAPER_REGRESSION,
        "schema_version": "scb-diagnostic-1.0",
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved to {args.output_json}")
    return result


if __name__ == "__main__":
    main()
