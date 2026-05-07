# biolm-scb

Code and reproducibility data for **"The sustained causal bottleneck of in-context learning across modalities"**.

## Layout

```
biolm-scb/
├── scripts/        Figure-generation scripts (generate_fig*.py)
├── data/           npz aggregates used by the figure scripts
├── source_data/    CSV / JSON reproducibility files
├── tool/           scb-diagnostic CLI (Application C)
├── README.md
├── LICENSE
└── .gitignore
```

## Reproducing figures

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas

python scripts/generate_fig123_v3.py
python scripts/generate_fig4_capability.py
python scripts/generate_fig5_v3.py
python scripts/generate_fig7_veff_sweep.py
python scripts/generate_fig9_theory.py
```

## Application C — `scb-diagnostic` CLI

```bash
pip install -e tool/
scb-diagnostic --model EleutherAI/pythia-1.4b --task-class text
```

## License

MIT — see [`LICENSE`](LICENSE).
