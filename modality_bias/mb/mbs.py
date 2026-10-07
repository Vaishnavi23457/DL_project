"""
mb.mbs
======
Modality Bias Score (MBS) computation.

Two complementary bias measures:

1. rho (ρ) — optimisation-side discrepancy ratio (from balanced_mm/bml/modulation.py)
   Measures how much more discriminative one modality is vs the other during training.
   rho > 1 means that modality is dominant.

2. MBS — input-side attribution-based bias score
   MBS = audio_share from IG attribution (0=fully video, 1=fully audio, 0.5=balanced)

Together they answer:
   rho  : is the model training in a biased way? (optimisation view)
   MBS  : does the model actually use one modality more? (attribution view)

Usage
-----
    from mb.mbs import compute_mbs, compute_rho

    mbs = compute_mbs(audio_shares)          # from ig_xai_av output
    rho = compute_rho(model, loader, device) # discrepancy ratio
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

EPS = 1e-8


def compute_mbs(audio_shares: List[float]) -> Dict[str, float]:
    """
    Compute Modality Bias Score from per-sample audio attribution shares.

    MBS = mean(audio_share)
    A balanced model has MBS ≈ 0.5.
    MBS > 0.5 means audio-dominant; MBS < 0.5 means video-dominant.

    Parameters
    ----------
    audio_shares : list of per-sample audio attribution shares (0-1)

    Returns
    -------
    dict with:
        mbs       : mean audio attribution share
        std       : standard deviation
        n_samples : number of samples
        dominant  : 'audio' | 'video' | 'balanced'
        imbalance : abs(mbs - 0.5), higher = more biased
    """
    import numpy as np
    arr = np.array(audio_shares, dtype=np.float32)
    mbs = float(arr.mean())
    std = float(arr.std())

    if mbs > 0.55:
        dominant = "audio"
    elif mbs < 0.45:
        dominant = "video"
    else:
        dominant = "balanced"

    return {
        "mbs"      : mbs,
        "std"      : std,
        "n_samples": len(audio_shares),
        "dominant" : dominant,
        "imbalance": abs(mbs - 0.5),
    }


@torch.no_grad()
def compute_rho(
    model,
    dataloader: DataLoader,
    device: torch.device,
    score_mode: str = "auto",
) -> Dict[str, float]:
    """
    Compute discrepancy ratio ρ for each modality over a dataloader.

    Reuses the discrepancy_ratios function from balanced_mm/bml/modulation.py.
    ρ > 1 for modality m means m is more discriminative (dominant).

    Parameters
    ----------
    model      : LateFusionModel with encode(), fuse(), unimodal_logits()
    dataloader : DataLoader
    device     : torch device
    score_mode : 'auto' | 'linear' | 'zero_out'

    Returns
    -------
    dict mapping modality name -> mean rho over the dataset
    """
    import sys, os
    # try to import from teammate's repo
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../balanced_mm"))
        from bml.modulation import discrepancy_ratios, unimodal_scores_from_logits
    except ImportError:
        # fallback: inline implementation
        def unimodal_scores_from_logits(unimodal_logits, labels):
            labels = labels.view(-1, 1)
            return torch.stack([F.softmax(lg.float(), dim=-1).gather(1, labels).squeeze(1)
                                 for lg in unimodal_logits], dim=0)

        def discrepancy_ratios(scores):
            tot = scores.float().sum(dim=1)
            M = tot.numel()
            if M < 2:
                return torch.ones_like(tot)
            mat = tot.view(-1, 1) / (tot.view(1, -1) + EPS)
            mat.fill_diagonal_(0.0)
            return mat.sum(dim=1) / (M - 1)

    model.eval()
    names = model.modality_names
    rho_sums = torch.zeros(len(names))
    n_batches = 0

    for inputs, labels in dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()} if isinstance(inputs, dict) else inputs
        labels = labels.to(device)
        feats   = model.encode(inputs)
        ulogits = model.unimodal_logits(feats, mode=score_mode)
        scores  = unimodal_scores_from_logits(ulogits, labels)
        rho     = discrepancy_ratios(scores).cpu()
        rho_sums += rho
        n_batches += 1

    rho_mean = rho_sums / max(n_batches, 1)
    return {names[m]: float(rho_mean[m]) for m in range(len(names))}


def compute_bias_summary(
    mbs_result: Dict[str, float],
    rho_result: Dict[str, float],
) -> Dict[str, object]:
    """
    Combine MBS and rho into a single bias summary.

    Parameters
    ----------
    mbs_result : output of compute_mbs()
    rho_result : output of compute_rho()

    Returns
    -------
    Combined summary dict
    """
    summary = {**mbs_result}
    summary["rho"] = rho_result
    # agreement check: do both measures agree on dominant modality?
    rho_dominant = max(rho_result, key=rho_result.get) if rho_result else "unknown"
    summary["rho_dominant"] = rho_dominant
    summary["measures_agree"] = (
        (mbs_result["dominant"] == "audio" and rho_dominant == "audio") or
        (mbs_result["dominant"] == "video" and rho_dominant == "video") or
        (mbs_result["dominant"] == "balanced")
    )
    return summary
