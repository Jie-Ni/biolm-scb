# biolm-scb

**The sustained causal bottleneck of in-context learning across modalities**

---

## One-line abstract

We show that mature in-context learning (ICL) in autoregressive Transformers is governed by a **sustained one-dimensional channel** in the residual stream — a contiguous block of mid-to-late layers whose activations collapse to participation ratio < 5 — and that the length of this channel `n_col` is set by the **effective vocabulary entropy** `V_eff` of the training distribution, not by the encoder/decoder dichotomy.

---

## Repository layout

```
biolm-scb/
├── paper/              LaTeX source + compiled PDFs (main, supplementary)
│   ├── main.tex / main.pdf
│   ├── supplementary.tex / supplementary.pdf
│   └── references.bib
├── figures/            Main figures + Extended Data, in PDF / PNG / SVG
├── scripts/            Figure-generation scripts (generate_fig*.py)
├── data/               npz aggregates used by the figure scripts
├── source_data/        CSV / JSON reproducibility files (one per main figure)
├── tool/               scb-diagnostic CLI (Application C, pip-installable)
├── README.md
├── LICENSE
└── .gitignore
```

---

## Reproducing the figures

```bash
git clone https://github.com/Jie-Ni/biolm-scb.git
cd biolm-scb
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas

python scripts/generate_fig123_v3.py
python scripts/generate_fig4_capability.py
python scripts/generate_fig5_v3.py
python scripts/generate_fig7_veff_sweep.py
python scripts/generate_fig9_theory.py
```

Output lands in `figures/`. A snapshot is already shipped.

---

## Application C — `scb-diagnostic` CLI

A 1-minute probe that classifies any pretrained Transformer by its sustained-causal-bottleneck length and predicts its mature ICL amplification.

```bash
pip install -e tool/
scb-diagnostic --model bert-base-uncased
scb-diagnostic --model EleutherAI/pythia-1.4b --task-class text
```

See [`tool/README.md`](tool/README.md) for full usage.

---

## License

MIT for code and reproducibility data. See [`LICENSE`](LICENSE).
