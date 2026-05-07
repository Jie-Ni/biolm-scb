# biolm-scb

**A sustained one-dimensional channel underlies in-context learning in autoregressive Transformers**

Ni Jie and Adam Jatowt &middot; University of Innsbruck, Digital Science Center

> arXiv preprint: **TBA** &middot; target venue: *Nature Machine Intelligence*, 2026

---

## One-line abstract

We show that mature in-context learning (ICL) in autoregressive Transformers is governed by a **sustained one-dimensional channel** in the residual stream — a contiguous block of mid-to-late layers whose activations collapse to participation ratio &lt; 5 — and that the length of this channel `n_col` is set by the **effective vocabulary entropy** `V_eff` of the training distribution, not by the encoder/decoder dichotomy.

---

## Repository layout

```
biolm-scb/
├── paper/              LaTeX source + compiled PDFs of main, supplementary, cover letter
│   ├── main.tex
│   ├── supplementary.tex
│   ├── references.bib
│   ├── cover_letter.md / cover_letter.pdf
│   ├── main.pdf
│   └── supplementary.pdf
├── figures/            Figs 1-9 + AppA + ED1, in PDF / PNG / SVG
├── scripts/            Figure-generation scripts (generate_fig*.py) + probe + aggregators
├── data/               npz aggregates and per-model JSON probes used by the figure scripts
├── source_data/        CSV / JSON reproducibility files (one per main figure)
├── tool/               scb-diagnostic CLI (Application C, pip-installable)
├── README.md
├── LICENSE
└── .gitignore
```

---

## Reproducing the figures

All eight main figures and the appendix figures are generated from the npz files in `data/`. No GPU is needed for the figure-generation step.

```bash
# 1. clone
git clone https://github.com/Jie-Ni/biolm-scb.git
cd biolm-scb

# 2. minimal env
python -m venv .venv && source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install numpy scipy matplotlib pandas

# 3. regenerate any figure
python scripts/generate_fig123_v3.py        # Fig 1, 2, 3 (Hero / Continuity / Phase2)
python scripts/generate_fig4_v3.py          # Fig 4 IFF
python scripts/generate_fig4_capability.py  # Fig 4 Capability
python scripts/generate_fig5_v3.py          # Fig 5 Trajectory
python scripts/generate_fig67_v3.py         # Fig 6, 7 (BiDirControl, Pfam)
python scripts/generate_fig7_veff_sweep.py  # Fig 7 V_eff sweep
python scripts/generate_fig8_predict.py     # Fig 8 Predictor
python scripts/generate_fig9_theory.py      # Fig 9 Theory
python scripts/generate_appA_lead.py        # Appendix A
```

Output PDFs/PNGs land in `figures/`. The repository ships a snapshot already.

### Re-running the upstream HPC probes (optional)

The npz files in `data/` are aggregates of per-model PR / SCB / ICL probes that were run on UIBK LEO5 and ASC MUSICA. To regenerate them from scratch you need access to those clusters and the model weights from HuggingFace.

```bash
# on LEO5
sbatch scripts/_leo5_pr_probe.sbatch         # rounds 1-3 launchers also included
# on MUSICA (V_eff pretraining sweep)
sbatch scripts/_veff_pretrain.sbatch
```

After staging the resulting JSONs back to your laptop under `data/pr_profile/`, `data/icl_verify/` and `data/scb_suppress_v2/`, run the aggregators in `scripts/_*.py` (`_inn_aggregator.py`, `_merge_pr_into_alllayer.py`, `_merge_inn_into_extended.py`) to rebuild the npz files.

---

## Application C — `scb-diagnostic` CLI

A 1-minute CPU-or-GPU probe that classifies any pretrained Transformer by its sustained-causal-bottleneck length and predicts its mature ICL amplification `R`. Two to three orders of magnitude cheaper than a full ICL benchmark.

```bash
pip install -e tool/
scb-diagnostic --model bert-base-uncased
scb-diagnostic --model EleutherAI/pythia-1.4b --task-class text
```

See [`tool/README.md`](tool/README.md) for full usage, output format, and the cross-modality calibration table.

---

## Citation

If you use any of the data, scripts, or the SCB diagnostic tool, please cite:

```bibtex
@article{ni2026scb,
  title   = {A sustained one-dimensional channel underlies in-context learning
             in autoregressive Transformers},
  author  = {Ni, Jie and Jatowt, Adam},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {arXiv:TBA}
}
```

---

## License

- **Code** (`scripts/`, `tool/`) and **reproducibility data** (`data/`, `source_data/`): MIT, see [`LICENSE`](LICENSE).
- **Manuscript** (`paper/*.pdf`, figures): &copy; 2026 the authors. Redistribution of derivatives permitted under CC-BY 4.0 once the paper is published; until then, please cite the arXiv preprint instead of redistributing the PDF.

---

## Contact

Ni Jie &mdash; `njie@seu.edu.cn`
PhD student, Digital Science Center, University of Innsbruck
Supervisor: Prof. Adam Jatowt
