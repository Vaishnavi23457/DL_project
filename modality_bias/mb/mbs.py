"""
mb.mbs
======

Modality Bias Score (MBS) and optimisation-side modality bias.

This module supports an arbitrary number of modalities.

For each sample and modality m:

    s_m = P(y_true | modality m)

The modality-wise MBS is:

    MBS_m = s_m / sum_j(s_j)

Therefore:

    sum_m MBS_m = 1

Interpretation
--------------
For M modalities, perfectly balanced attribution/scores would be:

    MBS_m = 1 / M

For example:

    2 modalities:
        balanced = 0.50 per modality

    3 modalities:
        balanced = 0.3333 per modality

    4 modalities:
        balanced = 0.25 per modality

A modality with a larger MBS value has a larger relative
contribution according to the modality-wise true-class scores.

This implementation is modality-agnostic and supports:
    audio + video
    image + text
    audio + video + text
    any other M-modality configuration.

The model is expected to expose:

    model.modalities
    model.encode(inputs)
    model.unimodal_logits(features)

For models with a different architecture, use an adapter that
provides this interface.

rho
---
rho is an optimisation-side modality discrepancy measure.

It is separate from MBS and does not use Integrated Gradients.

For each modality, rho > 1 indicates that the modality has a
higher true-class score than the other modalities on average.
"""


from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


EPS = 1e-8


# ============================================================
# 1. TRUE-CLASS SCORES
# ============================================================

def true_class_scores(
    unimodal_logits: Dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Convert unimodal logits into true-class probabilities.

    Parameters
    ----------
    unimodal_logits:
        Dictionary:

            {
                "audio":  Tensor[B, C],
                "visual": Tensor[B, C],
                "text":   Tensor[B, C],
                ...
            }

    labels:
        Ground-truth class labels, shape [B].

    Returns
    -------
    Dict[str, Tensor]
        True-class probability for every modality.

        Each tensor has shape [B].
    """

    scores = {}

    labels = labels.view(-1, 1)

    for modality, logits in unimodal_logits.items():

        probs = F.softmax(
            logits.float(),
            dim=-1
        )

        scores[modality] = (
            probs
            .gather(1, labels)
            .squeeze(1)
        )

    return scores


# ============================================================
# 2. GENERIC N-MODALITY MBS
# ============================================================

def compute_mbs_per_sample_from_scores(
    scores: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute modality-wise MBS for an arbitrary number of modalities.

    For each sample:

        MBS_m = s_m / sum_j(s_j)

    Parameters
    ----------
    scores:
        Dictionary containing true-class scores.

        Example for 3 modalities:

            {
                "audio":  Tensor[B],
                "visual": Tensor[B],
                "text":   Tensor[B]
            }

    Returns
    -------
    Dict[str, Tensor]
        Per-sample MBS for every modality.

        Example:

            {
                "audio":  Tensor[B],
                "visual": Tensor[B],
                "text":   Tensor[B]
            }

        The MBS values across modalities sum to approximately 1
        for every sample.
    """

    if not scores:
        raise ValueError(
            "No modality scores were provided."
        )

    modalities = list(scores.keys())

    # Stack:
    #
    # [num_modalities, batch_size]
    #
    stacked_scores = torch.stack(
        [scores[m] for m in modalities],
        dim=0
    )

    total_score = stacked_scores.sum(
        dim=0,
        keepdim=True
    )

    # Normalize across modalities
    stacked_mbs = (
        stacked_scores /
        (total_score + EPS)
    )

    return {
        modality: stacked_mbs[i]
        for i, modality in enumerate(modalities)
    }


# ============================================================
# 3. MBS FROM UNIMODAL LOGITS
# ============================================================

def compute_mbs_from_logits(
    unimodal_logits: Dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-sample MBS directly from unimodal logits.

    This is the main generic MBS function.

    Supports any number of modalities.

    Parameters
    ----------
    unimodal_logits:
        Dictionary mapping modality name to logits.

        Example:

            {
                "audio":  [B, C],
                "visual": [B, C],
                "text":   [B, C]
            }

    labels:
        Ground-truth labels [B].

    Returns
    -------
    Dict[str, Tensor]
        Per-sample MBS for every modality.
    """

    scores = true_class_scores(
        unimodal_logits,
        labels
    )

    return compute_mbs_per_sample_from_scores(
        scores
    )


# ============================================================
# 4. MODEL-BASED PER-SAMPLE MBS
# ============================================================

def compute_mbs_per_sample(
    model,
    loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-sample MBS for a multimodal model.

    The model must expose:

        model.modalities
        model.encode(inputs)
        model.unimodal_logits(features)

    Parameters
    ----------
    model:
        Multimodal model.

    loader:
        DataLoader whose batches contain:

            batch["label"]

        and one entry for each modality.

    device:
        Torch device.

    max_samples:
        Optional maximum number of samples.

    Returns
    -------
    Dict[str, np.ndarray]

        Example for audio + visual:

            {
                "audio":  np.ndarray,
                "visual": np.ndarray
            }

        Example for audio + visual + text:

            {
                "audio":  np.ndarray,
                "visual": np.ndarray,
                "text":   np.ndarray
            }
    """

    model.eval()

    modalities = list(model.modalities)

    mbs_values = {
        modality: []
        for modality in modalities
    }

    with torch.no_grad():

        for batch in loader:

            # ------------------------------------------------
            # Move inputs to device
            # ------------------------------------------------

            labels = batch["label"].to(device)

            inputs = {}

            for modality in modalities:

                if modality not in batch:
                    raise KeyError(
                        f"Modality '{modality}' was not found "
                        f"in the batch."
                    )

                value = batch[modality]

                if torch.is_tensor(value):
                    inputs[modality] = value.to(device)

                else:
                    inputs[modality] = value

            # ------------------------------------------------
            # Encode modalities
            # ------------------------------------------------

            feats = model.encode(inputs)

            # ------------------------------------------------
            # Obtain unimodal logits
            # ------------------------------------------------

            unimodal_logits = model.unimodal_logits(
                feats
            )

            # ------------------------------------------------
            # Calculate MBS
            # ------------------------------------------------

            batch_mbs = compute_mbs_from_logits(
                unimodal_logits,
                labels
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            batch_size = labels.size(0)

            for modality in modalities:

                values = (
                    batch_mbs[modality]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                mbs_values[modality].extend(
                    values
                )

            # ------------------------------------------------
            # Stop if max_samples reached
            # ------------------------------------------------

            if max_samples is not None:

                current_count = len(
                    mbs_values[modalities[0]]
                )

                if current_count >= max_samples:
                    break

    # --------------------------------------------------------
    # Convert to NumPy and truncate
    # --------------------------------------------------------

    result = {}

    for modality in modalities:

        values = mbs_values[modality]

        if max_samples is not None:
            values = values[:max_samples]

        result[modality] = np.asarray(
            values,
            dtype=np.float32
        )

    return result


# ============================================================
# 5. GLOBAL MBS SUMMARY
# ============================================================

def compute_mbs(
    mbs_per_sample: Dict[str, np.ndarray],
) -> Dict[str, object]:
    """
    Compute global MBS statistics from per-sample modality MBS.

    Parameters
    ----------
    mbs_per_sample:
        Dictionary containing per-sample MBS.

        Example:

            {
                "audio":  np.array([...]),
                "visual": np.array([...]),
                "text":   np.array([...])
            }

    Returns
    -------
    Dict[str, object]

        {
            "mean": {...},
            "std": {...},
            "median": {...},
            "balanced_value": ...,
            "dominant": ...,
            "imbalance": ...
        }
    """

    if not mbs_per_sample:
        raise ValueError(
            "No MBS values were provided."
        )

    modalities = list(
        mbs_per_sample.keys()
    )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    mean_mbs = {
        modality: float(
            np.mean(mbs_per_sample[modality])
        )
        for modality in modalities
    }

    std_mbs = {
        modality: float(
            np.std(mbs_per_sample[modality])
        )
        for modality in modalities
    }

    median_mbs = {
        modality: float(
            np.median(mbs_per_sample[modality])
        )
        for modality in modalities
    }

    # --------------------------------------------------------
    # Balanced value
    # --------------------------------------------------------

    n_modalities = len(modalities)

    balanced_value = 1.0 / n_modalities

    # --------------------------------------------------------
    # Deviation from perfectly balanced MBS
    # --------------------------------------------------------

    imbalance = {
        modality: abs(
            mean_mbs[modality]
            - balanced_value
        )
        for modality in modalities
    }

    # --------------------------------------------------------
    # Dominant modality
    # --------------------------------------------------------

    dominant = max(
        mean_mbs,
        key=mean_mbs.get
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "mean": mean_mbs,
        "std": std_mbs,
        "median": median_mbs,
        "balanced_value": balanced_value,
        "imbalance": imbalance,
        "dominant": dominant,
        "n_modalities": n_modalities,
    }


# ============================================================
# 6. RHO — DISCREPANCY RATIO
# ============================================================

def discrepancy_ratios(
    scores: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate discrepancy ratios between modalities.

    Parameters
    ----------
    scores:
        Tensor with shape:

            [num_modalities, batch_size]

        containing true-class scores.

    Returns
    -------
    Tensor
        Rho for each modality.

    Interpretation
    --------------
    rho > 1:
        modality has greater discriminative score.

    rho ≈ 1:
        balanced.

    rho < 1:
        modality is less discriminative than
        the other modalities.
    """

    scores = scores.float()

    n_modalities = scores.size(0)

    if n_modalities < 2:
        return torch.ones_like(
            scores.sum(dim=1)
        )

    # Total score for each modality
    total_scores = scores.sum(
        dim=1
    )

    # Pairwise ratios
    ratio_matrix = (
        total_scores.view(-1, 1)
        /
        (
            total_scores.view(1, -1)
            + EPS
        )
    )

    # Ignore self-comparison
    ratio_matrix.fill_diagonal_(0.0)

    # Average ratio against all other modalities
    rho = (
        ratio_matrix.sum(dim=1)
        /
        (n_modalities - 1)
    )

    return rho


# ============================================================
# 7. MODEL-BASED RHO
# ============================================================

@torch.no_grad()
def compute_rho(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute optimisation-side modality discrepancy ratio.

    The model must expose:

        model.modalities
        model.encode(inputs)
        model.unimodal_logits(features)

    Supports any number of modalities.

    Returns
    -------
    Dict[str, float]

        Example:

            {
                "audio": 1.84,
                "visual": 0.72,
                "text": 0.44
            }
    """

    model.eval()

    modalities = list(
        model.modalities
    )

    rho_sum = torch.zeros(
        len(modalities),
        dtype=torch.float32
    )

    n_batches = 0

    for batch in dataloader:

        labels = batch["label"].to(
            device
        )

        # ----------------------------------------------------
        # Prepare inputs
        # ----------------------------------------------------

        inputs = {}

        for modality in modalities:

            if modality not in batch:
                raise KeyError(
                    f"Modality '{modality}' "
                    f"not found in batch."
                )

            value = batch[modality]

            if torch.is_tensor(value):
                inputs[modality] = value.to(
                    device
                )
            else:
                inputs[modality] = value

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        feats = model.encode(
            inputs
        )

        # ----------------------------------------------------
        # Unimodal logits
        # ----------------------------------------------------

        unimodal_logits = (
            model.unimodal_logits(feats)
        )

        # ----------------------------------------------------
        # True-class scores
        # ----------------------------------------------------

        scores_dict = true_class_scores(
            unimodal_logits,
            labels
        )

        scores = torch.stack(
            [
                scores_dict[m]
                for m in modalities
            ],
            dim=0
        )

        # ----------------------------------------------------
        # Rho
        # ----------------------------------------------------

        rho = discrepancy_ratios(
            scores
        ).cpu()

        rho_sum += rho

        n_batches += 1

    rho_mean = (
        rho_sum /
        max(n_batches, 1)
    )

    return {
        modality: float(
            rho_mean[i]
        )
        for i, modality
        in enumerate(modalities)
    }


# ============================================================
# 8. COMPLETE BIAS SUMMARY
# ============================================================

def compute_bias_summary(
    mbs_result: Dict[str, object],
    rho_result: Dict[str, float],
) -> Dict[str, object]:
    """
    Combine MBS and rho into one bias summary.

    Parameters
    ----------
    mbs_result:
        Output of compute_mbs().

    rho_result:
        Output of compute_rho().

    Returns
    -------
    Dict[str, object]
    """

    summary = dict(mbs_result)

    summary["rho"] = rho_result

    # --------------------------------------------------------
    # MBS dominant modality
    # --------------------------------------------------------

    mbs_dominant = mbs_result.get(
        "dominant",
        "unknown"
    )

    # --------------------------------------------------------
    # Rho dominant modality
    # --------------------------------------------------------

    rho_dominant = (
        max(
            rho_result,
            key=rho_result.get
        )
        if rho_result
        else "unknown"
    )

    summary["mbs_dominant"] = (
        mbs_dominant
    )

    summary["rho_dominant"] = (
        rho_dominant
    )

    # --------------------------------------------------------
    # Whether the two measures agree
    # --------------------------------------------------------

    summary["measures_agree"] = (
        mbs_dominant == rho_dominant
    )

    return summary
