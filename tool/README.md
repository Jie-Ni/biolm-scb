# scb-diagnostic — predict mature ICL amplification from any Hugging Face model

A 1-minute CPU-or-GPU probe that classifies any pretrained Transformer
according to its sustained-causal-bottleneck (SCB) length and predicts
its mature in-context-learning amplification R.

Source: companion to *"A sustained one-dimensional channel underlies
in-context learning in autoregressive Transformers"* (Nature Machine
Intelligence, 2026).

## Quick start

```bash
pip install torch transformers numpy
python scb_diagnostic.py --model bert-base-uncased
```

Sample output:
```
=== SCB diagnostic: bert-base-uncased ===
  modality=text, n_probes=200, device=cuda
  probing 200 sequences ...

=== RESULT ===
  L (layers including embedding): 13
  n_col (max contiguous PR<5 in [0.25L, 0.95L]):  0
  V_eff regime: high V_eff (no SCB)
  predicted mature ICL amplification R:  0.69
  elapsed: 18.4 sec
```

## What it measures

For any Transformer, the diagnostic computes:
- **L** — number of layers (including embedding)
- **PR profile** — per-layer participation ratio of the residual-stream
  covariance ($\mathrm{PR}_\ell = (\sum \lambda_i)^2 / \sum \lambda_i^2$)
- **n_col** — the sustained-causal-bottleneck length: the maximum
  contiguous block of mid-to-late layers (relative depth in
  [0.25, 0.95]) where PR < 5
- **V_eff regime** — qualitative classification: low / intermediate / high
- **Predicted mature R** — the few-shot/zero-shot loss amplification
  ratio, predicted via a linear regression
  (slope=0.064, intercept=0.69, LOO R²=0.55) fit on the 17-model
  deep-bootstrap panel of the paper

The probe is **modality-aware**: pass `--task-class text/protein/DNA`
to use the appropriate probe corpus.

## Why use it

A full ICL benchmark on a new model takes ~hours of GPU time across
multiple downstream evaluations. **SCB diagnostic predicts ICL
amplification in <1 minute** with a single forward pass per probe
sequence — two to three orders of magnitude cheaper. Validation:
n_col explains 55% of variance in mature R via leave-one-out
regression, far better than parameter count alone (LOO R² = -0.02).

## Cross-modality calibration

Per the V_eff theory, models in low effective vocabulary regimes
(DNA k-mer tokenizers, vision patches) form SCBs even under
masked-LM training. The tool reports the V_eff regime so users can
interpret the prediction:

| n_col | V_eff regime | Typical models |
|-------|--------------|----------------|
| 0 | high V_eff | text/protein encoder MLMs (BERT, ESM2) |
| 1–7 | intermediate | XLM-RoBERTa, vision Transformer outliers |
| 8+ | low V_eff | causal LMs, NucT, deep ViTs |

## Citing

```
@article{ni2026scb,
  title  = {A sustained one-dimensional channel underlies in-context
            learning in autoregressive Transformers},
  author = {Ni, Jie and Jatowt, Adam},
  journal = {Nature Machine Intelligence},
  year    = {2026},
}
```
