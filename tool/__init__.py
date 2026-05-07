"""scb-diagnostic — predict mature ICL amplification from any HF model.

Companion tool to *Bio-LM Scaling Physics paper* (NMI 2026).
"""
from .scb_diagnostic import (
    main,
    participation_ratio,
    n_col_from_prs,
    predict_R,
    classify_v_eff,
    PAPER_REGRESSION,
)
__version__ = "1.0.0"
