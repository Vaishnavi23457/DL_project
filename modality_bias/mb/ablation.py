"""
mb.ablation
===========
Modality ablation and confidence drop analysis.

Ablation provides the ground-truth measure of actual modality dependence:
  audio_reliance = acc_normal - acc_zero_audio
  video_reliance = acc_normal - acc_zero_video

If audio_reliance >> video_reliance, the model truly depends on audio.
This is used to validate whether XAI scores reflect actual dependence (RQ1 Part B).

Three ablation strategies:
  1. zero    : replace modality with all-zeros tensor
  2. shuffle : replace modality with a random sample from the batch
  3. noise   : replace modality with Gaussian noise matching input statistics

Usage
-----
    from mb.ablation import compute_ablation_reliance

    result = compute_ablation_reliance(model, val_loader, device)
    print(result['audio_reliance'], result['video_reliance'])
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

Tensor = torch.Tensor


@torch.no_grad()
def _accuracy(model, inputs: Dict[str, Tensor], labels: Tensor) -> float:
    logits = model(inputs)
    return float((logits.argmax(1) == labels).float().mean())


def _zero_modality(inputs: Dict[str, Tensor], modality: str) -> Dict[str, Tensor]:
    """Replace a modality tensor with zeros."""
    out = dict(inputs)
    out[modality] = torch.zeros_like(inputs[modality])
    return out


def _shuffle_modality(inputs: Dict[str, Tensor], modality: str) -> Dict[str, Tensor]:
    """Replace a modality tensor with a shuffled version (random permutation within batch)."""
    out = dict(inputs)
    perm = torch.randperm(inputs[modality].shape[0])
    out[modality] = inputs[modality][perm]
    return out


def _noise_modality(inputs: Dict[str, Tensor], modality: str) -> Dict[str, Tensor]:
    """Replace a modality tensor with Gaussian noise matching its mean/std."""
    out = dict(inputs)
    x = inputs[modality].float()
    out[modality] = torch.randn_like(x) * x.std() + x.mean()
    return out


@torch.no_grad()
def compute_ablation_reliance(
    model,
    dataloader: DataLoader,
    device: torch.device,
    modalities: Optional[List[str]] = None,
    strategy: str = "zero",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute ablation-based modality reliance scores.

    For each modality, measures how much accuracy drops when that modality
    is removed. Higher drop = model relies more on that modality.

    Parameters
    ----------
    model      : trained multimodal model
    dataloader : validation DataLoader
    device     : torch device
    modalities : list of modality keys to ablate (default: ['audio','video'])
    strategy   : 'zero' | 'shuffle' | 'noise'
    max_batches: stop after this many batches (None = full dataset)

    Returns
    -------
    dict with:
        acc_normal          : baseline accuracy
        acc_zero_{modality} : accuracy with modality ablated
        {modality}_reliance : acc_normal - acc_zero_{modality}
        dominant_modality   : modality with highest reliance
    """
    assert strategy in ("zero", "shuffle", "noise"), f"Unknown strategy: {strategy}"
    ablate_fn = {"zero": _zero_modality, "shuffle": _shuffle_modality, "noise": _noise_modality}[strategy]

    model.eval()
    if modalities is None:
        # infer from model or use defaults
        modalities = getattr(model, "modality_names", ["audio", "video"])

    acc_normal  = 0.0
    acc_ablated = {m: 0.0 for m in modalities}
    n_batches   = 0

    for i, (inputs, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        acc_normal += _accuracy(model, inputs, labels)
        for m in modalities:
            if m not in inputs:
                continue
            ablated = ablate_fn(inputs, m)
            acc_ablated[m] += _accuracy(model, ablated, labels)

        n_batches += 1

    if n_batches == 0:
        return {}

    acc_normal /= n_batches
    result = {"acc_normal": acc_normal, "strategy": strategy}

    reliances = {}
    for m in modalities:
        if m not in inputs:
            continue
        acc_abl = acc_ablated[m] / n_batches
        reliance = acc_normal - acc_abl
        result[f"acc_zero_{m}"] = acc_abl
        result[f"{m}_reliance"] = reliance
        reliances[m] = reliance

    if reliances:
        result["dominant_modality"] = max(reliances, key=reliances.get)

    return result


@torch.no_grad()
def compute_confidence_drop(
    model,
    dataloader: DataLoader,
    device: torch.device,
    modalities: Optional[List[str]] = None,
    strategy: str = "zero",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute confidence (probability of true class) drop when each modality is ablated.

    Confidence drop is a softer measure than accuracy drop — useful when
    accuracy is already high and drops are small.

    Returns
    -------
    dict with:
        conf_normal          : mean confidence on true class
        conf_zero_{modality} : mean confidence with modality ablated
        {modality}_conf_drop : conf_normal - conf_zero_{modality}
    """
    assert strategy in ("zero", "shuffle", "noise")
    ablate_fn = {"zero": _zero_modality, "shuffle": _shuffle_modality, "noise": _noise_modality}[strategy]

    model.eval()
    if modalities is None:
        modalities = getattr(model, "modality_names", ["audio", "video"])

    conf_normal  = 0.0
    conf_ablated = {m: 0.0 for m in modalities}
    n_batches    = 0

    for i, (inputs, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        logits = model(inputs)
        probs  = F.softmax(logits.float(), dim=-1)
        conf_normal += float(probs.gather(1, labels.view(-1,1)).mean())

        for m in modalities:
            if m not in inputs:
                continue
            ablated     = ablate_fn(inputs, m)
            abl_logits  = model(ablated)
            abl_probs   = F.softmax(abl_logits.float(), dim=-1)
            conf_ablated[m] += float(abl_probs.gather(1, labels.view(-1,1)).mean())

        n_batches += 1

    if n_batches == 0:
        return {}

    conf_normal /= n_batches
    result = {"conf_normal": conf_normal, "strategy": strategy}

    for m in modalities:
        if m not in inputs:
            continue
        c = conf_ablated[m] / n_batches
        result[f"conf_zero_{m}"] = c
        result[f"{m}_conf_drop"] = conf_normal - c

    return result
