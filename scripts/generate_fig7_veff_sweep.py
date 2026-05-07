"""Generate Figure 7 (Application B: from-scratch V_eff saturation test).

Data sources (verified 2026-05-06 from MUSICA-INN/LNZ pool runs;
per-seed (vocab, n_col, R) extracted from
data_v46_collected/consolidated.json -- v46 final probe values at
the 5-B-token horizon for all six tokenizer vocabularies):

    INN  v_256   seeds 42-45  (4)  n_col=[0,0,0,0]
                                   R    =[1.107,1.108,1.087,1.108]
    INN  v_1k    seeds 42-45  (4)  n_col=[3,1,2,3]
                                   R    =[1.805,1.497,1.786,1.523]
    INN  v_8k    seeds 42-45  (4)  n_col=[7,7,7,6]
                                   R    =[2.510,2.866,2.343,2.243]
    INN  v_50k   seeds 42-49  (8)  n_col=[7,8,8,7,6,7,6,6]
                                   R    =[2.406,2.761,2.374,2.612,
                                          2.611,2.364,2.478,2.402]
    INN  v_100k  seeds 42-46,48-49 (7) n_col=[7,7,7,7,6,6,6]
                                       R    =[2.876,2.751,2.327,2.680,
                                              2.523,2.575,2.611]
                                   (seed 47 missing in v46 collection)
    INN  v_200k  seeds 42-49  (8)  n_col=[7,8,7,6,7,7,7,8]
                                   R    =[2.895,2.620,2.795,2.680,
                                          2.864,2.797,2.515,2.696]

L = 12 (Pythia-160M arch), so saturation prediction n_col = L * 0.6 = 7.2.
ICL maturity threshold R = 1.5 (Section appB main text).

The fused-cross-entropy implementation that lifts the single-H100
logits-memory limit landed before v46; v_100k/v_200k now have full
final 5-B-token probes alongside v_256/v_1k/v_8k/v_50k.

Run from project root:
    python scripts/generate_fig7_veff_sweep.py
Produces:
    figures/Figure_7_AppB_VeffSweep.pdf
    figures/Figure_7_AppB_VeffSweep.png
    figures/Figure_7_AppB_VeffSweep.svg
    data/fig7_veff_sweep.npz
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

L_LAYERS = 12
SAT_NCOL = L_LAYERS * 0.6
R_MATURE = 1.5

# All six vocabularies are now final (5-B-token horizon).
VOCABS_FINAL = np.array(
    [256, 1024, 8192, 50257, 100000, 200000], dtype=float
)

NCOL_FINAL = {
    256:    np.array([0, 0, 0, 0]),
    1024:   np.array([3, 1, 2, 3]),
    8192:   np.array([7, 7, 7, 6]),
    50257:  np.array([7, 8, 8, 7, 6, 7, 6, 6]),
    100000: np.array([7, 7, 7, 7, 6, 6, 6]),
    200000: np.array([7, 8, 7, 6, 7, 7, 7, 8]),
}
R_FINAL = {
    256:    np.array([1.1068, 1.1084, 1.0866, 1.1081]),
    1024:   np.array([1.8050, 1.4973, 1.7863, 1.5235]),
    8192:   np.array([2.5101, 2.8662, 2.3435, 2.2427]),
    50257:  np.array([2.4062, 2.7613, 2.3735, 2.6120,
                      2.6112, 2.3637, 2.4783, 2.4023]),
    100000: np.array([2.8757, 2.7512, 2.3265, 2.6801,
                      2.5229, 2.5749, 2.6109]),
    200000: np.array([2.8952, 2.6197, 2.7953, 2.6795,
                      2.8643, 2.7971, 2.5152, 2.6959]),
}


def _save_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(
        DATA_DIR / "fig7_veff_sweep.npz",
        vocabs_final=VOCABS_FINAL,
        ncol_256=NCOL_FINAL[256],
        ncol_1k=NCOL_FINAL[1024],
        ncol_8k=NCOL_FINAL[8192],
        ncol_50k=NCOL_FINAL[50257],
        ncol_100k=NCOL_FINAL[100000],
        ncol_200k=NCOL_FINAL[200000],
        R_256=R_FINAL[256],
        R_1k=R_FINAL[1024],
        R_8k=R_FINAL[8192],
        R_50k=R_FINAL[50257],
        R_100k=R_FINAL[100000],
        R_200k=R_FINAL[200000],
        L_layers=L_LAYERS,
        sat_ncol=SAT_NCOL,
        R_mature=R_MATURE,
    )


def _scatter_seeds(ax, x_center, ys, *, color, marker, edge, label=None,
                    alpha=1.0, jitter=0.04):
    if ys.size == 0:
        return
    log_x = np.log10(x_center)
    rng = np.random.default_rng(0)
    if ys.size > 1:
        offsets = rng.uniform(-jitter, jitter, size=ys.size) * log_x
    else:
        offsets = np.zeros(1)
    xs = np.full_like(ys, x_center, dtype=float) * np.power(10.0, offsets)
    ax.scatter(
        xs, ys, color=color, marker=marker, edgecolor=edge,
        s=70, alpha=alpha, linewidths=1.2, label=label, zorder=3,
    )


def _plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
    })

    fig, ax_n = plt.subplots(figsize=(7.0, 4.4))
    ax_r = ax_n.twinx()
    ax_r.spines["top"].set_visible(False)

    NCOL_COLOR = "#1f77b4"
    R_COLOR = "#d95f02"
    SAT_COLOR = "#666666"

    x_min, x_max = 180, 270000
    ax_n.set_xscale("log")
    ax_n.set_xlim(x_min, x_max)

    sat_band = mpatches.Rectangle(
        (x_min, SAT_NCOL - 0.4), x_max - x_min, 0.8,
        facecolor=SAT_COLOR, alpha=0.18, zorder=0, edgecolor="none",
    )
    ax_n.add_patch(sat_band)
    ax_n.axhline(SAT_NCOL, color=SAT_COLOR, linestyle=":", linewidth=1.0,
                 alpha=0.7, zorder=1)

    ax_r.axhline(R_MATURE, color=R_COLOR, linestyle="--", linewidth=0.9,
                 alpha=0.55, zorder=1)

    for v in VOCABS_FINAL:
        ys_n = NCOL_FINAL[int(v)]
        ys_r = R_FINAL[int(v)]
        _scatter_seeds(ax_n, v, ys_n, color=NCOL_COLOR, marker="o",
                       edge="white")
        _scatter_seeds(ax_r, v, ys_r, color=R_COLOR, marker="^",
                       edge="white")

    # annotate placed in the FREE SPACE between v_256 (n_col=0) and v_1k
    # (n_col=1-3) at y≈1.5 — clear of all markers (v_256 triangles sit at
    # right-axis R≈1.10 but on left-axis n_col scale that is below y=0;
    # v_256 dots sit at y=0; v_1k dots sit at y=1-3). y=1.5 between them
    # is empty. zorder=6 + white bbox guarantees legibility regardless.
    import numpy as _np
    _x_mid = float(_np.sqrt(VOCABS_FINAL[0] * VOCABS_FINAL[1]))  # log midpoint
    ax_n.text(_x_mid, 1.5,
              "R=1.10\n(undertrained,\n$|V|$=256)",
              ha="center", va="center", fontsize=7.5, color="#555555",
              style="italic", zorder=6,
              bbox=dict(facecolor="white", edgecolor="#cccccc",
                        linewidth=0.4, alpha=0.92, pad=2))
    ax_n.text(VOCABS_FINAL[1], 3.6,
              "rising edge\nof ICL maturity",
              ha="center", va="bottom", fontsize=8, color="#555555",
              style="italic", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                        pad=1.2))

    ax_n.set_xlabel(r"Tokenizer vocabulary $|V|$  ($V_{\mathrm{eff}}$ at this scale)")
    ax_n.set_ylabel(r"$n_{\mathrm{col}}$  (per-seed markers)", color=NCOL_COLOR)
    ax_r.set_ylabel(r"$R = \mathcal{L}_{\mathrm{ZS}}/\mathcal{L}_{\mathrm{FS}}$  (per-seed)",
                    color=R_COLOR)

    ax_n.tick_params(axis="y", labelcolor=NCOL_COLOR)
    ax_r.tick_params(axis="y", labelcolor=R_COLOR)

    ax_n.set_ylim(-0.6, 12.6)
    ax_n.set_yticks([0, 2, 4, 6, 7, 8, 10, 12])
    ax_r.set_ylim(0.8, 3.2)

    tick_v = [256, 1024, 8192, 50257, 100000, 200000]
    ax_n.set_xticks(tick_v)
    ax_n.set_xticklabels(["256", "1k", "8k", "50k", "100k", "200k"])
    ax_n.minorticks_off()

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=NCOL_COLOR,
                   markersize=7, markeredgecolor="white",
                   label=r"$n_{\mathrm{col}}$ (final, 5 B tok)"),
        plt.Line2D([0], [0], marker="^", linestyle="", color=R_COLOR,
                   markersize=7, markeredgecolor="white",
                   label=r"$R$ (final, 5 B tok)"),
        mpatches.Patch(facecolor=SAT_COLOR, alpha=0.18,
                       label=r"saturation prediction $L\cdot 0.6 = 7.2$"),
        plt.Line2D([0], [0], color=R_COLOR, linestyle="--", linewidth=0.9,
                   alpha=0.7,
                   label=r"ICL maturity threshold $R = 1.5$"),
    ]
    ax_n.legend(handles=legend_handles, loc="upper left", frameon=False,
                fontsize=8.5, handlelength=1.4)

    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out_pdf = FIG_DIR / "Figure_7_AppB_VeffSweep.pdf"
    out_png = FIG_DIR / "Figure_7_AppB_VeffSweep.png"
    out_svg = FIG_DIR / "Figure_7_AppB_VeffSweep.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")


def main() -> None:
    _save_data()
    _plot()


if __name__ == "__main__":
    main()
