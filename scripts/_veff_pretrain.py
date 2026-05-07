"""V_eff sweep pretraining for SCB causal proof.

Trains a Pythia-160M-style decoder-only Transformer from scratch under
six tokenizer configurations differing only in vocabulary size, while
periodically measuring n_col + ICL amplification R.

Each config: 160M params, 1024 ctx, ~5B tokens of OpenWebText subset.

Outputs (per config) under OUT/{tag}/:
  - tokenizer/                   trained from corpus
  - probe_step{step}.json        n_col + R + per-layer PR every PROBE_INTERVAL steps
  - ckpt_step{step}.pt           model state every CKPT_INTERVAL steps
  - final.json                   summary + final eval
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# 6 V_eff configurations (paper Section 8 sweep)
CONFIGS = {
    "v_256":    {"vocab_size": 256,    "tokenizer": "byte"},     # char/byte level
    "v_1k":     {"vocab_size": 1024,   "tokenizer": "bpe"},
    "v_8k":     {"vocab_size": 8192,   "tokenizer": "bpe"},
    "v_50k":    {"vocab_size": 50257,  "tokenizer": "bpe"},      # Pythia default-like
    "v_100k":   {"vocab_size": 100000, "tokenizer": "bpe"},
    "v_200k":   {"vocab_size": 200000, "tokenizer": "bpe"},
}


def train_tokenizer(corpus_iter, vocab_size: int, kind: str, save_dir: Path):
    """Train a tokenizer of the given vocab size on the corpus iterator."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if kind == "byte":
        # Trivial byte-level: use raw bytes 0-255 as tokens
        tk = Tokenizer(models.BPE(byte_fallback=True))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        # Train minimal alphabet
        trainer = trainers.BpeTrainer(vocab_size=256, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tk.train_from_iterator([" ".join([chr(i) for i in range(256)])], trainer=trainer)
    else:  # bpe
        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tk.train_from_iterator(corpus_iter, trainer=trainer)
    tk.save(str(save_dir / "tokenizer.json"))
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(save_dir / "tokenizer.json"),
        unk_token="<unk>", pad_token="<pad>",
        bos_token="<bos>", eos_token="<eos>",
    )
    fast.save_pretrained(save_dir)
    return fast


class StreamingOpenWebText(IterableDataset):
    """Stream OpenWebText shards as tokenized sequences of length seq_len."""
    def __init__(self, hf_dataset, tokenizer, seq_len=1024, max_tokens=None):
        super().__init__()
        self.ds = hf_dataset
        self.tok = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens

    def __iter__(self):
        buf = []
        n_yielded_tokens = 0
        for ex in self.ds:
            text = ex.get("text", "")
            if not text:
                continue
            ids = self.tok.encode(text)
            if isinstance(ids, list):
                buf.extend(ids)
            else:
                buf.extend(ids.ids)
            while len(buf) >= self.seq_len + 1:
                seq = buf[:self.seq_len + 1]
                buf = buf[self.seq_len:]
                n_yielded_tokens += self.seq_len
                yield torch.tensor(seq, dtype=torch.long)
                if self.max_tokens and n_yielded_tokens >= self.max_tokens:
                    return


# ── Probe: per-layer participation ratio + ICL amplification R ─────────
PROBE_TEXTS = [
    "The capital of France is Paris, the largest city in the country.",
    "Quantum mechanics describes physical phenomena at atomic and subatomic scales.",
    "Climate change refers to long-term shifts in global weather patterns.",
    "Machine learning is a subset of artificial intelligence focused on data.",
    "Reading a good book can be one of the most enjoyable ways to spend an afternoon.",
    "She walked into the room and noticed the lights were already on.",
    "The mitochondria is the powerhouse of the cell, producing ATP via oxidative phosphorylation.",
    "If a tree falls in the forest and no one is around to hear it, does it make a sound?",
] * 25  # 200 probes


def participation_ratio(states):
    sc = states - states.mean(axis=0, keepdims=True)
    cov = sc.T @ sc / sc.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return float("nan")
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


@torch.no_grad()
def probe_model(model, tokenizer, device):
    """Compute per-layer PR + ICL amplification R."""
    model.eval()
    layer_states = None
    for txt in PROBE_TEXTS[:50]:  # smaller probe during training
        ids = tokenizer(txt, return_tensors="pt", max_length=128, truncation=True).to(device)
        out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple of (1, T, D) per layer
        if layer_states is None:
            layer_states = [[] for _ in range(len(hs))]
        for i, h in enumerate(hs):
            layer_states[i].append(h.squeeze(0).float().cpu().numpy())

    prs = []
    for ls in layer_states:
        states = np.concatenate(ls, axis=0)
        prs.append(participation_ratio(states))
    L = len(prs)
    lo, hi = int(0.25 * L), int(0.95 * L)
    band = np.array(prs[lo:hi+1]) if hi >= lo else np.array([])
    in_band = band < 5
    max_run = cur = 0
    for x in in_band:
        if x:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    n_col = int(max_run)

    # ICL amplification R: simple proxy via few-shot vs zero-shot continuation NLL
    R = compute_R(model, tokenizer, device)
    model.train()
    return {"prs": prs, "n_col": n_col, "R": R, "L": L}


@torch.no_grad()
def compute_R(model, tokenizer, device):
    """ICL amplification R = ΔNLL_ICL / ΔNLL_ZS via projection-out hook.
    For training-time probe we use a lighter version: just compare few-shot
    to zero-shot continuation NLL directly (no projection-out)."""
    fewshot_prompt = (
        "Q: The capital of France is\nA: Paris\nQ: The capital of Germany is\nA: Berlin\n"
        "Q: The capital of Japan is\nA: Tokyo\nQ: The capital of Italy is\nA: "
    )
    zs_prompt = "The capital of Italy is "
    target = "Rome"
    target_ids = tokenizer(target, return_tensors="pt").input_ids.to(device)

    def nll(prefix):
        ids = tokenizer(prefix + target, return_tensors="pt").input_ids.to(device)
        out = model(ids, labels=ids)
        return float(out.loss)

    # 5 prompt variants
    pairs = [
        (fewshot_prompt, zs_prompt),
        (fewshot_prompt.replace("Italy", "Spain"), "The capital of Spain is "),
    ]
    Rs = []
    for fs, zs in pairs:
        try:
            nll_fs = nll(fs)
            nll_zs = nll(zs)
            if nll_zs > 1e-3:
                Rs.append(nll_zs / max(nll_fs, 1e-3))
        except Exception:
            pass
    return float(np.mean(Rs)) if Rs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=int(5e9), help="total training tokens")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--probe-interval", type=int, default=2000)
    ap.add_argument("--ckpt-interval", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local-cache", default="/local/hf_cache")
    ap.add_argument("--hf-dataset", default="HuggingFaceFW/fineweb-edu",
                    help="HF dataset for training corpus")
    ap.add_argument("--hf-subset", default="sample-10BT",
                    help="HF subset name (10B tokens of FineWeb-edu)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = CONFIGS[args.config]
    out = Path(args.out) / args.config
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.config}: vocab_size={cfg['vocab_size']}, tokenizer={cfg['tokenizer']}")
    print(f"  out={out}")

    # ── 1) Stream HF dataset ───────────────────────────────────────────
    os.environ["HF_HOME"] = args.local_cache
    Path(args.local_cache).mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    ds = load_dataset(args.hf_dataset, name=args.hf_subset, split="train",
                      streaming=True)

    # ── 2) Train tokenizer on first 10k texts ──────────────────────────
    tok_dir = out / "tokenizer"
    if not (tok_dir / "tokenizer.json").exists():
        print("  training tokenizer on first 10k texts ...")
        sample_texts = []
        for i, ex in enumerate(ds):
            sample_texts.append(ex.get("text", ""))
            if i >= 10000:
                break
        tokenizer = train_tokenizer(iter(sample_texts), cfg["vocab_size"],
                                     cfg["tokenizer"], tok_dir)
    else:
        print(f"  loading existing tokenizer from {tok_dir}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))

    # ── 3) Build Pythia-160M architecture with this vocab ───────────────
    config = GPTNeoXConfig(
        vocab_size=cfg["vocab_size"] + 4,  # +special tokens
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=args.seq_len,
        rotary_pct=0.25,
        rotary_emb_base=10000,
        bos_token_id=tokenizer.bos_token_id or 2,
        eos_token_id=tokenizer.eos_token_id or 3,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    model = GPTNeoXForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device, dtype=torch.bfloat16)
    print(f"  model params: {n_params/1e6:.1f}M, device={device}")

    # ── 4) Streaming dataloader ─────────────────────────────────────────
    ds = load_dataset(args.hf_dataset, name=args.hf_subset, split="train",
                      streaming=True)
    train_ds = StreamingOpenWebText(ds, tokenizer, seq_len=args.seq_len,
                                     max_tokens=args.tokens)
    loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2,
                        pin_memory=True, prefetch_factor=2)

    # ── 5) Train ────────────────────────────────────────────────────────
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(args.tokens / (args.batch_size * args.seq_len)))

    log_path = out / "train_log.jsonl"
    summary_path = out / "summary.json"
    summary = {"config": args.config, "vocab_size": cfg["vocab_size"],
               "n_params": n_params, "tokens_target": args.tokens, "probes": []}

    model.train()
    optim.zero_grad()
    step = 0
    tokens_seen = 0
    t0 = time.time()
    log_f = open(log_path, "a")
    for batch in loader:
        if batch is None or batch.numel() == 0:
            continue
        batch = batch.to(device, non_blocking=True)
        out_m = model(batch, labels=batch)
        loss = out_m.loss / args.grad_accum
        loss.backward()
        tokens_seen += batch.numel()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad()

        if step % 200 == 0:
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            print(f"  step {step}: loss={float(out_m.loss):.4f}, "
                  f"tokens={tokens_seen/1e6:.1f}M, tps={tps/1e3:.1f}k")
            log_f.write(json.dumps({"step": step, "loss": float(out_m.loss),
                                     "tokens": tokens_seen,
                                     "tps": tps}) + "\n")
            log_f.flush()

        if step % args.probe_interval == 0 and step > 0:
            probe = probe_model(model, tokenizer, device)
            probe["step"] = step
            probe["tokens"] = tokens_seen
            summary["probes"].append(probe)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  step {step}: PROBE n_col={probe['n_col']}, R={probe['R']:.3f}")

        if step % args.ckpt_interval == 0 and step > 0:
            torch.save(model.state_dict(), out / f"ckpt_step{step}.pt")

        step += 1
        if tokens_seen >= args.tokens:
            break

    log_f.close()

    # ── 6) Final probe + save ───────────────────────────────────────────
    final = probe_model(model, tokenizer, device)
    summary["final"] = final
    summary["wall_seconds"] = time.time() - t0
    summary["tokens_actual"] = tokens_seen
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), out / "final.pt")
    print(f"\n=== {args.config} DONE ===")
    print(f"  wall: {(time.time()-t0)/3600:.2f} hr, tokens: {tokens_seen/1e9:.2f}B")
    print(f"  final n_col={final['n_col']}, R={final['R']:.3f}")


if __name__ == "__main__":
    main()
