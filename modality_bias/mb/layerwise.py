"""
mb.layerwise
============
Per-layer bias analysis — identifies WHERE and WHY modality bias develops
through the network depth.

For each layer (or block), computes:
  - Gradient norm per modality encoder layer
  - Attribution share per layer (via layer-wise IG or GradCAM-style)
  - Bias score per layer: how dominant is one modality at each depth?

This answers the pipeline stage:
  "Layer-wise Analysis → Identify where/why bias develops"

Usage
-----
    from mb.layerwise import compute_layerwise_bias, plot_layerwise_bias

    results = compute_layerwise_bias(model, val_loader, device)
    plot_layerwise_bias(results, save_path="results/layerwise.png")
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

Tensor = torch.Tensor


def _get_named_layers(encoder: nn.Module, prefix: str = "") -> List[Tuple[str, nn.Module]]:
    """Get all leaf layers with their names."""
    layers = []
    for name, module in encoder.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            full_name = f"{prefix}.{name}" if prefix else name
            layers.append((full_name, module))
    return layers


def compute_gradient_norms_per_layer(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute gradient norms per encoder layer.

    Higher gradient norm = layer is actively learning / receiving strong signal.
    Comparing audio vs video encoder gradient norms reveals which modality
    the model is updating more aggressively.

    Parameters
    ----------
    model       : multimodal model with named encoders
    dataloader  : DataLoader
    device      : torch device
    max_batches : number of batches to average over

    Returns
    -------
    dict: {modality_name: {layer_name: mean_grad_norm}}
    """
    import torch.nn.functional as F

    model.train()  # need gradients
    modality_names = getattr(model, "modality_names", ["audio", "video"])

    # accumulate gradient norms per layer
    grad_norms: Dict[str, Dict[str, List[float]]] = {
        m: {} for m in modality_names
    }

    n = 0
    for inputs, labels in dataloader:
        if n >= max_batches:
            break
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        model.zero_grad()
        feats  = model.encode(inputs)
        logits = model.fuse(feats)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()

        # collect gradient norms from each encoder
        for m_idx, m_name in enumerate(modality_names):
            encoder = model.encoders[m_name]
            for layer_name, layer in encoder.named_modules():
                for p_name, p in layer.named_parameters(recurse=False):
                    if p.grad is not None:
                        full_name = f"{layer_name}.{p_name}" if layer_name else p_name
                        norm = float(p.grad.norm().item())
                        if full_name not in grad_norms[m_name]:
                            grad_norms[m_name][full_name] = []
                        grad_norms[m_name][full_name].append(norm)

        n += 1

    model.eval()

    # average across batches
    return {
        m: {layer: float(np.mean(norms)) for layer, norms in layers.items()}
        for m, layers in grad_norms.items()
    }


def compute_layerwise_attribution(
    model,
    inputs: Dict[str, Tensor],
    target: Tensor,
    modality: str = "audio",
    n_layers: int = 4,
) -> List[float]:
    """
    Compute attribution magnitude at each ResNet block for a modality.

    Uses gradient hooks to capture intermediate activations and gradients,
    giving a per-layer view of where the modality contributes.

    Parameters
    ----------
    model    : multimodal model
    inputs   : single-sample input dict
    target   : [1] target class
    modality : 'audio' or 'video'
    n_layers : number of ResNet blocks to probe (default 4 for ResNet18)

    Returns
    -------
    list of attribution scores, one per layer (length = n_layers)
    """
    model.eval()
    encoder = model.encoders[modality]

    # find layer4 blocks for ResNet18: layer1, layer2, layer3, layer4
    block_names = [f"layer{i}" for i in range(1, n_layers + 1)]
    activations: Dict[str, Tensor] = {}
    gradients:   Dict[str, Tensor] = {}
    hooks = []

    def make_fwd_hook(name):
        def hook(m, i, o):
            activations[name] = o.detach()
        return hook

    def make_bwd_hook(name):
        def hook(m, gi, go):
            gradients[name] = go[0].detach() if go[0] is not None else None
        return hook

    for bname in block_names:
        block = getattr(encoder.net if hasattr(encoder, "net") else encoder, bname, None)
        if block is not None:
            hooks.append(block.register_forward_hook(make_fwd_hook(bname)))
            hooks.append(block.register_full_backward_hook(make_bwd_hook(bname)))

    try:
        inp = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        logits = model(inp)
        model.zero_grad()
        score = logits[0, int(target.item())]
        score.backward()
    finally:
        for h in hooks:
            h.remove()

    layer_scores = []
    for bname in block_names:
        if bname in activations and bname in gradients and gradients[bname] is not None:
            # GradCAM-style: global average pool of grad * activation
            g = gradients[bname]
            a = activations[bname]
            weights = g.mean(dim=[2, 3], keepdim=True) if g.dim() == 4 else g.mean(dim=1, keepdim=True)
            cam = (weights * a).sum(dim=1).relu()
            layer_scores.append(float(cam.mean()))
        else:
            layer_scores.append(0.0)

    return layer_scores


def compute_layerwise_bias(
    model,
    dataloader,
    device: torch.device,
    max_samples: int = 50,
    n_layers: int = 4,
) -> Dict[str, object]:
    """
    Full layer-wise bias analysis.

    For each layer, computes audio vs video attribution share,
    showing where the imbalance develops in the network.

    Returns
    -------
    dict with:
        layer_names        : list of layer names
        audio_layer_scores : mean attribution per layer for audio
        video_layer_scores : mean attribution per layer for video
        layer_bias         : audio_share per layer (0=video, 1=audio, 0.5=balanced)
        gradient_norms     : per-modality gradient norms
    """
    model.eval()
    modalities = getattr(model, "modality_names", ["audio", "video"])
    layer_names = [f"layer{i}" for i in range(1, n_layers + 1)]

    audio_scores_all: List[List[float]] = []
    video_scores_all: List[List[float]] = []
    n = 0

    for inputs, labels in dataloader:
        inputs_dev = {k: v.to(device) for k, v in inputs.items()}
        B = labels.shape[0]

        for i in range(min(B, max_samples - n)):
            single = {k: v[i:i+1] for k, v in inputs_dev.items()}
            tgt    = labels[i:i+1].to(device)

            if "audio" in modalities:
                a_scores = compute_layerwise_attribution(model, single, tgt, "audio", n_layers)
                audio_scores_all.append(a_scores)

            if "video" in modalities:
                v_scores = compute_layerwise_attribution(model, single, tgt, "video", n_layers)
                video_scores_all.append(v_scores)

            n += 1
            if n >= max_samples:
                break

        if n >= max_samples:
            break

    audio_mean = np.mean(audio_scores_all, axis=0).tolist() if audio_scores_all else [0.0] * n_layers
    video_mean = np.mean(video_scores_all, axis=0).tolist() if video_scores_all else [0.0] * n_layers

    # per-layer bias: audio_share at each layer
    layer_bias = []
    for a, v in zip(audio_mean, video_mean):
        total = a + v + 1e-8
        layer_bias.append(a / total)

    grad_norms = compute_gradient_norms_per_layer(model, dataloader, device, max_batches=5)

    return {
        "layer_names"       : layer_names,
        "audio_layer_scores": audio_mean,
        "video_layer_scores": video_mean,
        "layer_bias"        : layer_bias,   # 0=video, 1=audio, 0.5=balanced
        "gradient_norms"    : grad_norms,
        "n_samples"         : n,
    }


def plot_layerwise_bias(
    results: Dict[str, object],
    save_path: Optional[str] = None,
    dataset: str = "",
):
    """
    Plot per-layer audio vs video attribution and bias score.

    Parameters
    ----------
    results   : output of compute_layerwise_bias()
    save_path : path to save figure (None = show inline)
    dataset   : dataset name for plot title
    """
    import matplotlib.pyplot as plt

    layers      = results["layer_names"]
    audio_scores = results["audio_layer_scores"]
    video_scores = results["video_layer_scores"]
    layer_bias   = results["layer_bias"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # plot 1: attribution scores per layer
    x = range(len(layers))
    axes[0].plot(x, audio_scores, "o-", label="Audio", color="steelblue")
    axes[0].plot(x, video_scores, "s-", label="Video", color="coral")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(layers)
    axes[0].set_ylabel("Mean Attribution Score")
    axes[0].set_title(f"Layer-wise Attribution{' — ' + dataset if dataset else ''}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # plot 2: bias score per layer
    colors = ["coral" if b > 0.55 else "steelblue" if b < 0.45 else "green"
              for b in layer_bias]
    axes[1].bar(x, layer_bias, color=colors)
    axes[1].axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Balance (0.5)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(layers)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Audio Attribution Share")
    axes[1].set_title(f"Layer-wise Bias{' — ' + dataset if dataset else ''}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
