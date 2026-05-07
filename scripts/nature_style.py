"""
Nature-grade matplotlib style preset.
Apply via: from nature_style import apply_style; apply_style()

v2 (2026-04-30): merged with Yuan1z0825/nature-skills (nature-figure)
    - mandatory svg.fonttype='none' (editable SVG text)
    - Arial as first sans-serif fallback
    - PALETTE_NMI_PASTEL for secondary/delta elements
    - save_publication() helper writes PNG + PDF + SVG together
    - tight_layout default pad=2
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_style():
    mpl.rcParams.update({
        # ── MANDATORY (nature-skills, non-negotiable) ──────────────────────
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"],
        "svg.fonttype": "none",            # editable <text> nodes in SVG
        # ── Typography (compact multi-panel, journal-final 7-9pt regime) ──
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.labelweight": "regular",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.title_fontsize": 7,
        # ── Spines (top/right off, frameless legend) ──────────────────────
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.9,             # bumped 0.7 → 0.9 per nature-skills
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Lines
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.6,
        # Legend (frameless mandatory)
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,
        # Figure / Save
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "savefig.facecolor": "white",
        # Math text
        "mathtext.fontset": "dejavusans",
    })


# ── Semantic palettes (kept from v1, project-specific) ───────────────────
FAMILY_COLORS = {
    "Pythia":    "#1f77b4",
    "BLOOM":     "#2ca02c",
    "Qwen2.5":   "#ff7f0e",
    "GPT-Neo":   "#9467bd",
    "Mistral":   "#d62728",
    "MoE":       "#e377c2",
    "ProGen2":   "#17becf",
    "Other":     "#bcbd22",
    "OLMo":      "#7f7f7f",
    "ESM2(MLM)": "#8c8c8c",
    "BERT(MLM)": "#5c5c5c",
    "NucT(MLM)": "#a5a5a5",
    "Vision":    "#8c564b",
}

PYTHIA_VIRIDIS = {
    "py70m":  "#440154",
    "py160m": "#414487",
    "py410m": "#2a788e",
    "py1b":   "#22a884",
    "py14b":  "#7ad151",
    "py28b":  "#fde725",
    "py69b":  "#f0a73a",
    "py12b":  "#d62728",
}


# ── nature-skills NMI pastel palette (for secondary/delta elements) ──────
PALETTE_NMI_PASTEL = {
    "baseline_dark": "#484878",
    "baseline_mid":  "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny":  "#E4E4F0",
    "ours_base":  "#E4CCD8",
    "ours_large": "#F0C0CC",
    "neutral_light": "#D8D8D8",
    "neutral_mid":   "#A8A8A8",
    "neutral_dark":  "#606060",
    "delta_up":   "#2E9E44",
    "delta_down": "#E53935",
}

# Generic semantic accents (nature-skills core)
ACCENTS = {
    "hero":       "#0F4D92",       # blue_main — proposed/key method
    "ref":        "#767676",       # neutral_mid — reference / baseline
    "callout":    "#B64342",       # red_strong — warning / IFF control / negative
    "positive":   "#8BCF8B",       # green_3 — improvement
    "teal":       "#42949E",
    "violet":     "#9A4D8E",
    "gold":       "#FFD700",
}


def panel_label(ax, label, x=-0.10, y=1.04, fontsize=11, color="black"):
    """Bold lowercase panel letter, top-left edge (Nature convention)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left",
            color=color)


def style_axes(ax):
    """Apply Nature-grade tick + spine style."""
    ax.tick_params(which="both", top=False, right=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#333")


def save_publication(fig, out_path_no_ext, formats=("png", "pdf", "svg"),
                     pad=2, dpi=600, close=True):
    """Save figure in multiple formats per nature-skills export policy.

    SVG primary (editable text), PDF for LaTeX include, PNG@dpi for raster preview.
    Calls tight_layout(pad=pad) before save.

    Parameters
    ----------
    fig : matplotlib Figure
    out_path_no_ext : str  — path WITHOUT file extension
    formats : tuple of str — any subset of ('png','pdf','svg','eps','jpg')
    pad : float — tight_layout pad (default 2 per nature-skills)
    dpi : int   — for raster outputs
    close : bool — plt.close(fig) after save (free memory)

    Returns
    -------
    list[str] of written file paths
    """
    base_dir = os.path.dirname(out_path_no_ext)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    try:
        fig.tight_layout(pad=pad)
    except Exception:
        pass  # constrained_layout figures will skip
    written = []
    for fmt in formats:
        p = f"{out_path_no_ext}.{fmt}"
        if fmt in ("png", "jpg", "jpeg", "tif", "tiff"):
            fig.savefig(p, dpi=dpi)
        else:
            fig.savefig(p)
        written.append(p)
    if close:
        plt.close(fig)
    return written


def is_dark(hex_color, threshold=128):
    """Return True if hex color is dark enough to need white text on top."""
    c = hex_color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


# ── Pretty model-name map for figure axis labels ──────────────────────────
# Maps internal aggregator tag → publication-grade display name.
MODEL_PRETTY = {
    # Pythia
    "py70m": "Pythia-70M", "py160m": "Pythia-160M", "py410m": "Pythia-410M",
    "py1b": "Pythia-1B", "py14b": "Pythia-1.4B", "py28b": "Pythia-2.8B",
    "py69b": "Pythia-6.9B", "py12b": "Pythia-12B",
    # BLOOM
    "bloom_560m": "BLOOM-560M", "bloom_1b1": "BLOOM-1B1",
    "bloomp__1b": "BLOOM-1B1", "bloom_3b": "BLOOM-3B",
    "bloom7b": "BLOOM-7B1", "bloom7": "BLOOM-7B1",
    # Mistral
    "mistral7": "Mistral-7B-v0.1", "mistral7b": "Mistral-7B-v0.3",
    # Qwen
    "qwen_05": "Qwen2.5-0.5B", "qwen_15": "Qwen2.5-1.5B",
    "qwen_3": "Qwen2.5-3B", "qwen_7": "Qwen2.5-7B",
    "qwen7": "Qwen2.5-7B",
    "qwen_moe_a27": "Qwen1.5-MoE-A2.7B",
    "olmoe_1b_7b": "OLMoE-1B-7B",
    # GPT-Neo
    "gptneo_125": "GPT-Neo-125M", "gptneo_13": "GPT-Neo-1.3B",
    "gptneo_27": "GPT-Neo-2.7B", "neo_27b": "GPT-Neo-2.7B",
    "neo_13b": "GPT-Neo-1.3B",
    # Phi / Smol / Stable / TinyLlama
    "phi1": "Phi-1", "phi15": "Phi-1.5",
    "smol_17": "SmolLM-1.7B",
    "stablelm": "StableLM-2-1.6B",
    "tinyllama": "TinyLlama-1.1B",
    # ProGen2
    "pg2_small": "ProGen2-small", "pg2_medium": "ProGen2-medium",
    "pg2_base": "ProGen2-base", "pg2_large": "ProGen2-large",
    "pg2_xlarge": "ProGen2-xlarge",
    # ESM2 (encoder MLM)
    "esm2_8M": "ESM2-8M", "esm2_35M": "ESM2-35M",
    "esm2_150M": "ESM2-150M", "esm2_650M": "ESM2-650M",
    "esm2_650m": "ESM2-650M",
    "esm2_3B": "ESM2-3B", "esm2_3b": "ESM2-3B",
    "esm2_15B": "ESM2-15B", "esm2_15b": "ESM2-15B",
    "esmL_27": "ESM2-650M", "esmL_3b": "ESM2-3B", "esmL_15b": "ESM2-15B",
    # BERT family
    "bert_base": "BERT-base", "bert_large": "BERT-large",
    "roberta_base": "RoBERTa-base", "roberta_large": "RoBERTa-large",
    # Vision
    "vit_b": "ViT-B/16", "vit_l": "ViT-L/16",
    "dino_b": "DINOv2-B", "dino_l": "DINOv2-L",
    # OLMo / Yi / Falcon
    "olmo_2_7b": "OLMo-2-7B", "yi_6b": "Yi-6B",
    "falcon_7b": "Falcon-7B",
    "olmo7b": "OLMo-7B", "olmo_7b": "OLMo-7B",
    "opt_27b": "OPT-2.7B", "opt_67b": "OPT-6.7B",
    # Nucleotide Transformer
    "nuct_25b": "NucT-2.5B",
}


def pretty_name(tag, fallback=None):
    """Map internal tag to publication display name. Returns tag (or fallback) if unknown."""
    s = str(tag).strip()
    if s in MODEL_PRETTY:
        return MODEL_PRETTY[s]
    s2 = s.replace("_full", "")
    if s2 in MODEL_PRETTY:
        return MODEL_PRETTY[s2]
    return fallback if fallback is not None else s
