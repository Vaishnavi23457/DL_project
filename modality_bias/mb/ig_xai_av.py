"""
mb.ig_xai_av
============
Integrated Gradients attribution for audio + video modalities.

Extends the image/text IG from balanced_mm/bml/xai.py to handle
audio spectrograms and video frame inputs used in CREMA-D and CMU-MOSEI.

Key design:
  - Audio branch  : IG on log-mel spectrogram  [B, 1, H, W]
  - Video branch  : IG on stacked frames        [B, N, 3, H, W] or [B, C, H, W]
  - Normalization : attribution scores divided by input size (element count)
                    so audio and video shares are dimension-agnostic and comparable
  - Share formula :
        audio_score = mean(|IG_audio|) / audio_size
        video_score = mean(|IG_video|) / video_size
        audio_share = audio_score / (audio_score + video_score)

Usage
-----
    from mb.ig_xai_av import compute_ig_attribution, compute_modality_shares

    audio_attr, video_attr = compute_ig_attribution(model, inputs, target)
    audio_share, video_share = compute_modality_shares(audio_attr, video_attr)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


def _freeze(model: nn.Module):
    snap = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    return snap


def _unfreeze(model: nn.Module, snap):
    for p, r in zip(model.parameters(), snap):
        p.requires_grad_(r)


def integrated_gradients_audio(
    model: nn.Module,
    inputs: Dict[str, Tensor],
    target: Tensor,
    steps: int = 50,
    baseline: Optional[Tensor] = None,
) -> Tensor:
    """
    Integrated Gradients for the audio modality.

    Interpolates audio from baseline (zeros) to input, sums gradients.

    Parameters
    ----------
    model   : multimodal model with encode() / fuse() or forward()
    inputs  : dict containing 'audio' key with tensor [B, C, H, W]
    target  : [B] long — class index to explain
    steps   : number of interpolation steps (higher = more accurate)
    baseline: optional baseline tensor, defaults to zeros

    Returns
    -------
    IG tensor same shape as inputs['audio']
    """
    model.eval()
    x = inputs["audio"]
    base = torch.zeros_like(x) if baseline is None else baseline.clone()

    snap = _freeze(model)
    try:
        total = torch.zeros_like(x)
        for k in range(steps + 1):
            alpha = k / steps
            xi = (base + alpha * (x - base)).detach().requires_grad_(True)
            inp = {**inputs, "audio": xi}
            logits = model(inp) if callable(getattr(model, "forward", None)) else model.fuse(model.encode(inp))
            score = logits.gather(1, target.view(-1, 1)).sum()
            grads = torch.autograd.grad(score, xi)[0]
            total = total + grads.detach()
        ig = (x - base) * total / (steps + 1)
    finally:
        _unfreeze(model, snap)
    return ig.detach()


def integrated_gradients_video(
    model: nn.Module,
    inputs: Dict[str, Tensor],
    target: Tensor,
    steps: int = 50,
    baseline: Optional[Tensor] = None,
) -> Tensor:
    """
    Integrated Gradients for the video modality.

    Parameters
    ----------
    model   : multimodal model
    inputs  : dict containing 'video' key with tensor [B, *, H, W]
    target  : [B] long — class index to explain
    steps   : interpolation steps
    baseline: optional baseline, defaults to zeros

    Returns
    -------
    IG tensor same shape as inputs['video']
    """
    model.eval()
    x = inputs["video"]
    base = torch.zeros_like(x) if baseline is None else baseline.clone()

    snap = _freeze(model)
    try:
        total = torch.zeros_like(x)
        for k in range(steps + 1):
            alpha = k / steps
            xi = (base + alpha * (x - base)).detach().requires_grad_(True)
            inp = {**inputs, "video": xi}
            logits = model(inp) if callable(getattr(model, "forward", None)) else model.fuse(model.encode(inp))
            score = logits.gather(1, target.view(-1, 1)).sum()
            grads = torch.autograd.grad(score, xi)[0]
            total = total + grads.detach()
        ig = (x - base) * total / (steps + 1)
    finally:
        _unfreeze(model, snap)
    return ig.detach()


def compute_modality_shares(
    audio_attr: Tensor,
    video_attr: Tensor,
) -> Tuple[float, float]:
    """
    Compute normalized modality attribution shares.

    Divides mean absolute attribution by input size so that
    audio [1,128,128] and video [3,3,224,224] are comparable.

    Parameters
    ----------
    audio_attr : IG tensor for audio modality
    video_attr : IG tensor for video modality

    Returns
    -------
    (audio_share, video_share) — floats that sum to 1.0
    """
    audio_size = audio_attr[0].numel()
    video_size = video_attr[0].numel()

    audio_score = float(audio_attr.abs().flatten(1).mean(1).mean()) / audio_size
    video_score = float(video_attr.abs().flatten(1).mean(1).mean()) / video_size

    total = audio_score + video_score + 1e-8
    return audio_score / total, video_score / total


def compute_ig_attribution(
    model: nn.Module,
    inputs: Dict[str, Tensor],
    target: Optional[Tensor] = None,
    steps: int = 50,
) -> Dict[str, object]:
    """
    Full modality attribution for audio+video model.

    Parameters
    ----------
    model   : multimodal model
    inputs  : dict with 'audio' and/or 'video' tensors
    target  : [B] class indices; if None uses model prediction
    steps   : IG interpolation steps

    Returns
    -------
    dict with keys:
        audio_attr  : IG tensor for audio
        video_attr  : IG tensor for video
        audio_share : float (0-1), normalized attribution share
        video_share : float (0-1)
        predicted   : list of predicted class indices
    """
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        pred = logits.argmax(1)

    tgt = pred if target is None else target

    result: Dict[str, object] = {"predicted": pred.cpu().tolist()}

    audio_attr = None
    video_attr = None

    if "audio" in inputs:
        audio_attr = integrated_gradients_audio(model, inputs, tgt, steps=steps)
        result["audio_attr"] = audio_attr

    if "video" in inputs:
        video_attr = integrated_gradients_video(model, inputs, tgt, steps=steps)
        result["video_attr"] = video_attr

    if audio_attr is not None and video_attr is not None:
        a_share, v_share = compute_modality_shares(audio_attr, video_attr)
        result["audio_share"] = a_share
        result["video_share"] = v_share

    return result


def compute_dataset_attribution_shares(
    model: nn.Module,
    dataloader,
    device: torch.device,
    steps: int = 50,
    max_samples: int = 200,
) -> Dict[str, List[float]]:
    """
    Compute per-sample attribution shares over a dataset.

    Parameters
    ----------
    model       : trained multimodal model
    dataloader  : DataLoader returning (inputs_dict, labels)
    device      : torch device
    steps       : IG steps per sample (reduce for speed)
    max_samples : stop after this many samples

    Returns
    -------
    dict with 'audio_shares' and 'video_shares' lists (one float per sample)
    """
    model.eval()
    audio_shares: List[float] = []
    video_shares: List[float] = []
    n = 0

    for inputs, _ in dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        B = next(iter(inputs.values())).shape[0]

        for i in range(B):
            if n >= max_samples:
                break
            single = {k: v[i:i+1] for k, v in inputs.items()}
            result = compute_ig_attribution(model, single, steps=steps)
            if "audio_share" in result:
                audio_shares.append(result["audio_share"])
                video_shares.append(result["video_share"])
            n += 1

        if n >= max_samples:
            break

    return {"audio_shares": audio_shares, "video_shares": video_shares}
