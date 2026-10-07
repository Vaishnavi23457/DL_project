"""
mb.correlation
==============
Pearson + Spearman correlation between XAI attribution scores and MBS.

This module answers RQ1:
  "Can XAI attribution scores be used to identify modality bias?"

Two correlation analyses:

1. XAI ↔ MBS (rho)
   Correlate per-run/epoch XAI audio_share with rho discrepancy ratio.
   Strong positive correlation -> XAI identifies the same bias as rho.

2. XAI ↔ Ablation reliance
   Correlate XAI audio_share with ablation-based audio reliance score.
   Strong positive correlation -> XAI reflects actual modality dependence.

Usage
-----
    from mb.correlation import compute_correlation, summarize_rq1

    result = compute_correlation(xai_shares, mbs_scores)
    print(result)  # {'pearson_r': ..., 'pearson_p': ..., 'spearman_r': ..., ...}
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def compute_correlation(
    x: List[float],
    y: List[float],
    label_x: str = "XAI_share",
    label_y: str = "MBS",
) -> Dict[str, object]:
    """
    Compute Pearson and Spearman correlation between two lists of values.

    Parameters
    ----------
    x       : first variable (e.g. XAI audio attribution shares)
    y       : second variable (e.g. MBS / rho / ablation reliance)
    label_x : name of x variable (for reporting)
    label_y : name of y variable (for reporting)

    Returns
    -------
    dict with:
        pearson_r  : Pearson correlation coefficient
        pearson_p  : Pearson p-value
        spearman_r : Spearman correlation coefficient
        spearman_p : Spearman p-value
        n          : number of samples
        significant: True if both p-values < 0.05
        interpretation: human-readable interpretation
    """
    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)

    if len(x_arr) < 3:
        return {
            "pearson_r" : float("nan"), "pearson_p" : float("nan"),
            "spearman_r": float("nan"), "spearman_p": float("nan"),
            "n"         : len(x_arr),
            "significant": False,
            "interpretation": "Not enough samples (need >= 3)",
        }

    pearson_r,  pearson_p  = stats.pearsonr(x_arr, y_arr)
    spearman_r, spearman_p = stats.spearmanr(x_arr, y_arr)

    significant = bool(pearson_p < 0.05 and spearman_p < 0.05)

    # interpretation
    r_mean = (abs(pearson_r) + abs(spearman_r)) / 2
    if r_mean >= 0.7:
        strength = "strong"
    elif r_mean >= 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if (pearson_r + spearman_r) > 0 else "negative"
    sig_str   = "significant" if significant else "not significant"

    interp = (
        f"{strength} {direction} {sig_str} correlation between "
        f"{label_x} and {label_y} "
        f"(Pearson r={pearson_r:.3f} p={pearson_p:.4f}, "
        f"Spearman ρ={spearman_r:.3f} p={spearman_p:.4f})"
    )

    return {
        "pearson_r"     : float(pearson_r),
        "pearson_p"     : float(pearson_p),
        "spearman_r"    : float(spearman_r),
        "spearman_p"    : float(spearman_p),
        "n"             : len(x_arr),
        "significant"   : significant,
        "interpretation": interp,
        "label_x"       : label_x,
        "label_y"       : label_y,
    }


def compute_xai_mbs_correlation(
    xai_shares: List[float],
    rho_values: List[float],
) -> Dict[str, object]:
    """
    Correlate XAI attribution shares with rho discrepancy ratios.
    Answers RQ1 Part A: does XAI identify the same bias as rho?
    """
    return compute_correlation(
        xai_shares, rho_values,
        label_x="XAI_audio_share",
        label_y="rho_discrepancy",
    )


def compute_xai_ablation_correlation(
    xai_shares: List[float],
    ablation_reliance: List[float],
) -> Dict[str, object]:
    """
    Correlate XAI attribution shares with ablation-based reliance scores.
    Answers RQ1 Part B: does XAI reflect actual modality dependence?
    """
    return compute_correlation(
        xai_shares, ablation_reliance,
        label_x="XAI_audio_share",
        label_y="ablation_reliance",
    )


def summarize_rq1(
    xai_mbs_corr: Dict[str, object],
    xai_ablation_corr: Dict[str, object],
    dataset: str = "",
) -> str:
    """
    Print a human-readable RQ1 summary.

    Parameters
    ----------
    xai_mbs_corr      : output of compute_xai_mbs_correlation()
    xai_ablation_corr : output of compute_xai_ablation_correlation()
    dataset           : dataset name for reporting

    Returns
    -------
    Multi-line summary string
    """
    lines = [
        f"\n{'='*60}",
        f"  RQ1 ANSWER {'— ' + dataset if dataset else ''}",
        f"{'='*60}",
        "",
        "Part A: Can XAI identify modality bias? (XAI ↔ MBS)",
        f"  {xai_mbs_corr['interpretation']}",
        f"  → {'YES ✓' if xai_mbs_corr['significant'] else 'INCONCLUSIVE ✗'}",
        "",
        "Part B: Does XAI reflect actual dependence? (XAI ↔ Ablation)",
        f"  {xai_ablation_corr['interpretation']}",
        f"  → {'YES ✓' if xai_ablation_corr['significant'] else 'INCONCLUSIVE ✗'}",
        "",
        "Overall RQ1:",
        "  XAI attribution scores CAN identify modality bias"
        if (xai_mbs_corr["significant"] and xai_ablation_corr["significant"])
        else "  XAI attribution scores show LIMITED ability to identify modality bias",
        f"{'='*60}\n",
    ]
    return "\n".join(lines)
