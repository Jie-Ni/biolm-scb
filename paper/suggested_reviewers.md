# Suggested reviewers (v4 — SCB submission)

For *Nature* main-journal submission of "A sustained one-dimensional channel underlies in-context learning in autoregressive Transformers" (Jie & Jatowt, 2026).

## Suggestion strategy

We deliberately avoid suggesting authors aligned with the heavy-tailed / rank-collapse / outlier-dimension lineage (Mahoney, Hodgkinson, Skean, Yang, Sun, Timkey) to pre-empt scope-overlap concerns: their prior work is closest in topic but operates at strictly different levels of analysis (head, component, token-level outlier, eigenvalue exponent). Instead we suggest five reviewers from adjacent communities — interpretability, ICL theory, computational geometry, scaling laws — whose expertise covers the paper's content but who have no direct stake in the SCB framing.

## 1. Trenton Bricken (Anthropic)

- **Affiliation**: Anthropic, San Francisco, CA, USA
- **Expertise**: Mechanistic interpretability of LMs, sparse autoencoders, monosemanticity, dictionary learning. Co-author of "Towards Monosemanticity" (Anthropic 2023).
- **Justification**: Closest interpretability work to ours operationally; informed perspective on residual-stream geometry without being invested in the heavy-tail lineage.

## 2. Christopher Olah (Anthropic)

- **Affiliation**: Anthropic, San Francisco, CA, USA
- **Expertise**: Mechanistic interpretability, induction heads, circuits framework. Co-author of Olsson *et al.* 2022.
- **Justification**: Olsson 2022 is the foundational paper our work builds on for the ICL mechanism. He can evaluate whether our SCB story is consistent with the induction-head/function-vector mechanism literature.

## 3. Jacob Andreas (MIT CSAIL)

- **Affiliation**: Massachusetts Institute of Technology, Cambridge, MA, USA
- **Expertise**: In-context learning theory, language model probing, semantic parsing. Group studies the latent representation underlying ICL.
- **Justification**: Well-positioned to evaluate the projection-out hook and IFF claim from a probing-methodology standpoint.

## 4. Klaus-Robert Müller (TU Berlin)

- **Affiliation**: Technische Universität Berlin, Berlin, Germany
- **Expertise**: Computational geometry of deep networks, kernel methods, spectral analysis of feature spaces.
- **Justification**: Beyond the ML interpretability community; can evaluate whether our spectral / participation-ratio methodology is statistically sound.

## 5. Atticus Geiger (Pr(Ai)²R Group)

- **Affiliation**: Pr(Ai)²R Group (independent research collective), Stanford alumnus.
- **Expertise**: Causal abstraction and mediation analysis for neural networks; rigorous methodology for evaluating intervention-based mechanistic claims.
- **Justification**: His causal-abstraction framework provides the gold standard for evaluating projection-out interventions like our SCB ablation; can independently audit whether the if-and-only-if control we report constitutes a clean causal claim.

## Reviewer roster summary

| Reviewer | Affiliation | Topical fit |
|---|---|---|
| T. Bricken | Anthropic | Sections 2-4 (residual-stream geometry) |
| C. Olah | Anthropic | Section 5, 7 (ICL circuits) |
| J. Andreas | MIT CSAIL | Methods §M.4, §M.5 (probing) |
| K.-R. Müller | TU Berlin | Methods §M.2, §M.3 (spectral) |
| A. Geiger | Pr(Ai)²R Group | Section 4 (causal projection-out), §M.4 (intervention auditing) |

## Reviewers to AVOID (potential pre-empt or scope-overlap)

- **Michael Mahoney** (UC Berkeley) — heavy-tailed self-regularisation; prior critic of low-rank claims
- **Liam Hodgkinson** — co-author of HTMP, our Section 7 primary contrast
- **Oscar Skean / Yann LeCun group** — entropy-bimodal / single-layer-information papers in same space
- **Mingze Yang / Tianyi Zhao** ("Hidden Geometry") — closest concurrent work on ICL geometry
- **Mingjie Sun** — massive activations paper, token-level outlier-dimension framing
- **William Timkey / Marten van Schijndel** — rogue dimensions paper

These researchers are valuable readers and we cite their work; we believe a reviewer outside their direct lineage will give a more balanced assessment of the SCB-as-substrate framing.

## Conflict declaration

The authors declare no current or prior collaboration with the suggested reviewers and no funding overlap. NJ is a PhD student at the Digital Science Center, University of Innsbruck (supervisor: A. Jatowt). Compute infrastructure: LEO5 (University of Innsbruck) + MUSICA-Linz / MUSICA-Innsbruck (Austrian Scientific Computing). No external funding.

---

Email addresses for tracking are available on the institutional websites and will be confirmed at submission.
