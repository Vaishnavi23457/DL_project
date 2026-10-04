"""
bml.xai
=======
XAI side of the proposal — *detecting* which modality the model leans on,
complementing the optimisation-side mitigation (OPM / OGM) of `bml.modulation`.

Tools
-----
1. ``integrated_gradients_image``     pixel-level IG (Sundararajan et al., 2017)
2. ``token_importance_text``          gradient x embedding score per token
3. ``modality_attribution``           per-modality attribution share of the
                                      predicted-class score (the "who did the
                                      work?" number, XAI-style)
4. ``head_block_contributions``       exact decomposition f = sum_m W^m phi^m + b
                                      for a linear head (Eq. 2 of the paper) —
                                      zero approximation error, costs nothing

Typical review flow:
    attrib = modality_attribution(model, inputs, target)
    -> shares["image"] >> shares["text"]   => image dominates (matches rho > 1)
    -> train with --modulation opm/ogm     => shares move closer to 0.5
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .models import LateFusionModel

Tensor = torch.Tensor


# --------------------------------------------------------------------------- 1. IG (images)
def _snapshot_requires_grad(model: torch.nn.Module):
    return [p.requires_grad for p in model.parameters()]


def _restore_requires_grad(model: torch.nn.Module, snap) -> None:
    for p, r in zip(model.parameters(), snap):
        p.requires_grad_(r)


def integrated_gradients_image(model: LateFusionModel, inputs: Dict[str, object], target: Tensor,
                               steps: int = 32, baseline: Optional[Tensor] = None) -> Tensor:
    """
    IG of the *image* input w.r.t. the logit of class `target` (per sample).

    Keeps the text branch fixed, straight-line interpolates the image from
    `baseline` (default: zeros) to the input and sums the gradients:

        IG_i(x) = (x_i - x'_i) * 1/K * sum_k dF(x' + k/K*(x-x'))/dx_i

    Returns [B, C, H, W]. Completeness: sum(IG) ~= F(x) - F(baseline).
    """
    if "image" not in model.encoders:
        raise KeyError("model has no image modality")
    model.eval()
    x = inputs["image"]
    assert torch.is_tensor(x) and x.dim() == 4, "image input must be [B,C,H,W]"
    bsz = x.shape[0]
    base = torch.zeros_like(x) if baseline is None else baseline.expand_as(x).clone()

    snap = _snapshot_requires_grad(model)
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        x = x.detach().clone().requires_grad_(True)
        total = torch.zeros_like(x)
        for k in range(steps + 1):
            alpha = k / steps
            xi = base + alpha * (x - base)
            xi = xi.detach().requires_grad_(True)
            logits = model({**inputs, "image": xi})
            grads = torch.autograd.grad(logits.gather(1, target.view(-1, 1)).sum(), xi)[0]
            total = total + grads
        ig = (x - base) * total / (steps + 1)
    finally:
        _restore_requires_grad(model, snap)
    return ig.detach()


# --------------------------------------------------------------------------- 2. token scores (text)
def _embedding_module(model: LateFusionModel) -> torch.nn.Module:
    """Our TextTransformerEncoder exposes `.tok`; HF DistilBERT via `.model.embeddings.word_embeddings`."""
    enc = model.encoders["text"]
    if hasattr(enc, "tok"):
        return enc.tok
    if hasattr(enc, "model") and hasattr(enc.model, "embeddings"):
        return enc.model.embeddings.word_embeddings
    raise AttributeError("cannot locate the token-embedding module of this text encoder")


def token_importance_text(model: LateFusionModel, inputs: Dict[str, object], target: Tensor) -> Tensor:
    """
    gradient x embedding token scores: dF/dE . E for the embedding output E of
    each token (the standard relevance score for discrete inputs, where IG on
    the input ids is not defined). Returns [B, L] (padded positions = 0).
    """
    if "text" not in model.encoders:
        raise KeyError("model has no text modality")
    model.eval()
    txt = inputs["text"]
    mask = txt["attention_mask"]
    captured: List[Tensor] = []
    # NOTE: params are deliberately left trainable here — the captured embedding
    # output must stay part of a grad-tracking graph for autograd.grad to work.
    h = _embedding_module(model).register_forward_hook(lambda m, i, o: captured.clear() or captured.append(o))
    try:
        logits = model(inputs)
        emb = captured[0]
        grads = torch.autograd.grad(logits.gather(1, target.view(-1, 1)).sum(), emb,
                                    retain_graph=False, allow_unused=False)[0]
        scores = (grads * emb).sum(-1) * mask.to(emb.dtype)      # [B, L]
    finally:
        h.remove()
    return scores.detach()


# --------------------------------------------------------------------------- 3. modality attribution
def modality_attribution(model: LateFusionModel, inputs: Dict[str, object],
                         target: Optional[Tensor] = None, steps: int = 16) -> Dict[str, object]:
    """
    How much did each modality contribute to the predicted-class score?

    `target=None` (default) explains the model's own prediction.

    Returns dict:
      attribution : {modality: mean |attribution| over the batch}
      share       : {modality: attribution / sum(attribution)}  (adds to 1)
      predicted   : predicted class per sample (for sanity checks)

    image branch -> integrated gradients on pixels,
    text branch  -> gradient x embedding summed over tokens.
    """
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        pred = logits.argmax(1)
    tgt = pred.view(-1, 1) if target is None else target.view(-1, 1)

    attrib: Dict[str, float] = {}
    if "image" in model.encoders:
        ig = integrated_gradients_image(model, inputs, tgt, steps=steps)
        attrib["image"] = float(ig.abs().flatten(1).sum(1).mean())
    if "text" in model.encoders:
        s = token_importance_text(model, inputs, tgt)
        attrib["text"] = float(s.abs().sum(1).mean())
    total = sum(attrib.values()) or 1.0
    return {
        "attribution": attrib,
        "share": {k: v / total for k, v in attrib.items()},
        "predicted": pred.detach().cpu().tolist(),
        "target": tgt.view(-1).detach().cpu().tolist(),
    }


# --------------------------------------------------------------------------- 4. exact head decomposition
@torch.no_grad()
def head_block_contributions(model: LateFusionModel, feats: Sequence[Tensor],
                             target: Optional[Tensor] = None) -> Dict[str, object]:
    """
    Eq. 2, exact for a linear head: the fused logit of class c decomposes as

        f_c = sum_m (W^m phi^m)_c + b_c   and   block_m := (W^m phi^m)_c + b_c / M

    so sum_m block_m == f_c exactly (bias is counted M times of b/M).

    Returns {"blocks": [B, M] tensor, "fused": [B] tensor, "delta": max |sum-fused|}
    — `delta` is ~1e-7 and is what the unit test asserts.
    """
    if model.head_type != "linear":
        raise ValueError("exact block decomposition requires head='linear' (use modality_attribution otherwise)")
    blocks = model.unimodal_logits(feats, mode="linear")           # list of M x [B, C]
    if target is None:
        bm = torch.stack(blocks, dim=1)                            # [B, M, C]
        fused = model.fuse(feats)                                  # [B, C]
        return {"blocks": bm, "fused": fused, "delta": float((bm.sum(1) - fused).abs().max())}
    bm = torch.stack([b.gather(1, target.view(-1, 1)) for b in blocks], dim=1).squeeze(-1)  # [B, M]
    fused = model.fuse(feats).gather(1, target.view(-1, 1)).squeeze(1)                     # [B]
    return {"blocks": bm, "fused": fused, "delta": float((bm.sum(1) - fused).abs().max())}
