"""
bml.modulation
==============
Pure-PyTorch, framework-agnostic implementation of the two on-the-fly modulation
strategies for balanced multimodal learning:

  * Discrepancy monitoring  ......  Eq. 6 & Eq. 7   (unimodal_scores_*, discrepancy_ratios)
  * OPM  (prediction modulation) ..  Eq. 8           (class OPM)
  * OGM-GE (gradient modulation) ..  Eq. 11, 12, 16, 17 (class OGMGE)
  * BalancedModulator  ...........  one-stop wrapper used by train.py / plug_in_example.py

References
  Wei, Hu, Du, Wen. "On-the-fly Modulation for Balanced Multimodal Learning", IEEE TPAMI 2024.
  Peng, Wei, Deng, Wang, Hu. "Balanced Multimodal Learning via On-the-fly Gradient Modulation", CVPR 2022.

Works for ANY number of modalities M >= 2 and any encoders. Nothing here depends on the
rest of this repo, so you can copy this single file into your own project.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
EPS = 1e-8


# --------------------------------------------------------------------------------------
# z(.) : monotonically increasing map of the discrepancy ratio into (0, 1)
# --------------------------------------------------------------------------------------
def make_z(name: str = "tanh") -> Callable[[Tensor], Tensor]:
    """z(rho). Paper default: tanh(rho - 1). Alternative used in ablation: sigmoid(rho)."""
    name = name.lower()
    if name == "tanh":
        return lambda r: torch.tanh(r - 1.0)
    if name == "sigmoid":
        return lambda r: torch.sigmoid(r)
    raise ValueError(f"unknown z function: {name}")


# --------------------------------------------------------------------------------------
# Eq. 6  --  uni-modal discriminative score  s_i^m = softmax(W^m phi^m_i + b/M)_{y_i}
# --------------------------------------------------------------------------------------
@torch.no_grad()
def unimodal_scores_from_logits(unimodal_logits: Sequence[Tensor], labels: Tensor) -> Tensor:
    """
    unimodal_logits : list (len M) of [B, C] tensors = each modality's own contribution to the
                      final logits (W^m phi^m + b/M for a linear head, or zero-out logits).
    labels          : [B] long
    returns         : scores [M, B]  (probability each modality alone gives to the true class)
    """
    labels = labels.view(-1, 1)
    out = []
    for lg in unimodal_logits:
        p = F.softmax(lg.float(), dim=-1)
        out.append(p.gather(1, labels).squeeze(1))
    return torch.stack(out, dim=0)


@torch.no_grad()
def unimodal_logits_zero_out(head_fn: Callable[[List[Tensor]], Tensor], feats: Sequence[Tensor]) -> List[Tensor]:
    """
    Zero-out strategy (paper Sec. 4.3.7) for NON-linear / multi-layer heads:
    keep modality m's feature, set all others to 0, run the head -> that is m's uni-modal prediction.
    """
    outs = []
    for m in range(len(feats)):
        masked = [f if j == m else torch.zeros_like(f) for j, f in enumerate(feats)]
        outs.append(head_fn(masked))
    return outs


# --------------------------------------------------------------------------------------
# Eq. 7  --  discrepancy ratio  rho^m_t = 1/(M-1) * sum_{j != m}  sum_i s^m_i / sum_i s^j_i
# --------------------------------------------------------------------------------------
@torch.no_grad()
def discrepancy_ratios(scores: Tensor) -> Tensor:
    """scores [M, B] -> rho [M].  rho^m > 1  => modality m is the more discriminative (dominant) one."""
    tot = scores.float().sum(dim=1)  # [M]
    M = tot.numel()
    if M < 2:
        return torch.ones_like(tot)
    mat = tot.view(-1, 1) / (tot.view(1, -1) + EPS)  # mat[m, j] = tot_m / tot_j
    mat.fill_diagonal_(0.0)
    return mat.sum(dim=1) / (M - 1)


# --------------------------------------------------------------------------------------
# OPM  --  On-the-fly Prediction Modulation  (feed-forward stage)   Eq. 8
# --------------------------------------------------------------------------------------
class OPM:
    """
    Adaptive modality-feature dropout.

        q^m_{t+1} = q_base * (1 + lam * z(rho^m_t))   if rho^m_t > 1   (dominant modality)
                  = 0                                 otherwise         (weak modality is never dropped)

    Usage per iteration (Algorithm 1):
        feats = opm.apply(feats)     # drop with the CURRENT q_t
        opm.update(rho_t)            # compute q_{t+1} from this batch's discrepancy
    """

    def __init__(self, num_modalities: int, q_base: float = 0.5, lam: float = 0.5,
                 z: str = "tanh", per_sample: bool = True, rescale: bool = False, q_max: float = 1.0):
        assert 0.0 < q_base < 1.0, "q_base must be in (0, 1)"
        assert lam > 0
        self.M = num_modalities
        self.q_base = q_base
        self.lam = lam
        self.z = make_z(z)
        self.per_sample = per_sample   # True: each sample drops independently (Fig. 2). False: whole batch.
        self.rescale = rescale         # inverted-dropout style rescale of kept features (paper: no)
        self.q_max = q_max
        self.q = torch.zeros(num_modalities)  # q_t  (starts at 0 -> first iteration drops nothing)

    @torch.no_grad()
    def update(self, rho: Tensor) -> Tensor:
        rho = rho.detach().float().cpu()
        q = self.q_base * (1.0 + self.lam * self.z(rho))
        q = torch.where(rho > 1.0, q, torch.zeros_like(q)).clamp_(0.0, self.q_max)
        self.q = q
        return q

    @torch.no_grad()
    def _keep_matrix(self, batch_size: int, device) -> Tensor:
        """[M, B] boolean, True = keep feature."""
        if self.per_sample:
            u = torch.rand(self.M, batch_size, device=device)
            keep = u >= self.q.to(device).view(-1, 1)
        else:
            u = torch.rand(self.M, device=device)
            keep = (u >= self.q.to(device)).view(-1, 1).expand(-1, batch_size).clone()
        # safety: never drop *all* modalities of a sample (cannot happen when some q == 0, but be safe)
        none_kept = ~keep.any(dim=0)
        if none_kept.any():
            m_min = int(torch.argmin(self.q))
            keep[m_min, none_kept] = True
        return keep

    def apply(self, feats: List[Tensor]) -> List[Tensor]:
        """Drop modality features (multiplicative 0/1 mask, gradient flows through kept ones)."""
        if float(self.q.max()) <= 0.0:
            return list(feats)
        B = feats[0].shape[0]
        keep = self._keep_matrix(B, feats[0].device)
        out = []
        for m, f in enumerate(feats):
            mask = keep[m].to(f.dtype).view(-1, *([1] * (f.dim() - 1)))
            if self.rescale and float(self.q[m]) > 0:
                mask = mask / (1.0 - float(self.q[m]))
            out.append(f * mask)
        return out

    def state(self) -> Dict[str, float]:
        return {f"q_{m}": float(self.q[m]) for m in range(self.M)}


# --------------------------------------------------------------------------------------
# OGM-GE  --  On-the-fly Gradient Modulation + Generalization Enhancement  Eq. 11/12/16/17
# --------------------------------------------------------------------------------------
class OGMGE:
    """
        k^m_t = 1 - alpha * z(rho^m_t)   if rho^m_t > 1   (dominant modality is slowed down)
              = 1                        otherwise
        theta^m <- theta^m - lr * ( k^m_t * g~  +  h ),   h ~ N(0, Cov(g~))      (GE noise)

    In practice (as in the official code) the noise for each parameter tensor is sampled as
    N(0, std(grad)) of that tensor's *un-modulated* gradient, so the resulting update noise has
    variance (k^2 + 1) x the original SGD noise (Eq. 17).

    Usage per iteration (Algorithm 2):
        loss.backward()
        # (if AMP)  scaler.unscale_(optimizer)
        k = ogm.coefficients(rho_t)
        ogm.apply_(model_modality_param_groups, k)
        optimizer.step()
    """

    def __init__(self, num_modalities: int, alpha: float = 0.5, z: str = "tanh", ge: bool = True,
                 ge_scope: str = "all", min_ndim_for_noise: int = 2, k_min: float = 0.0):
        assert alpha > 0
        assert ge_scope in ("all", "modulated")
        self.M = num_modalities
        self.alpha = alpha
        self.z = make_z(z)
        self.ge = ge                          # False -> plain OGM (no noise)
        self.ge_scope = ge_scope              # 'all': noise on every modality (Eq. 17 for all m, official code)
                                              # 'modulated': noise only where k < 1
        self.min_ndim = min_ndim_for_noise    # add noise only to weight tensors with >= this many dims
        self.k_min = k_min                    # paper: k is clamped at 0 minimum
        self.k = torch.ones(num_modalities)

    @torch.no_grad()
    def coefficients(self, rho: Tensor) -> Tensor:
        rho = rho.detach().float().cpu()
        k = 1.0 - self.alpha * self.z(rho)
        k = torch.where(rho > 1.0, k, torch.ones_like(k)).clamp_(self.k_min, 1.0)
        self.k = k
        return k

    @torch.no_grad()
    def apply_(self, modality_params: Sequence[Iterable[nn.Parameter]], k: Optional[Tensor] = None) -> None:
        """
        modality_params : list (len M) of iterables of parameters that belong to encoder m (theta^m).
                          The fusion head is NOT modulated (paper modulates theta^m only).
        Must be called after backward() and before optimizer.step().
        """
        k = self.k if k is None else k
        for m, params in enumerate(modality_params):
            km = float(k[m])
            add_noise = self.ge and (self.ge_scope == "all" or km < 1.0)
            for p in params:
                g = p.grad
                if g is None:
                    continue
                noise = None
                if add_noise and g.dim() >= self.min_ndim and g.numel() > 1:
                    std = g.float().std().item() + EPS      # std of the UN-modulated gradient
                    noise = torch.randn_like(g) * std
                if km != 1.0:
                    g.mul_(km)
                if noise is not None:
                    g.add_(noise)

    def state(self) -> Dict[str, float]:
        return {f"k_{m}": float(self.k[m]) for m in range(self.M)}


# --------------------------------------------------------------------------------------
# One-stop wrapper
# --------------------------------------------------------------------------------------
class BalancedModulator:
    """
    Combines discrepancy monitoring + OPM + OGM-GE behind two calls:

        feats, rho = mod.before_fusion(feats, labels, unimodal_logits_fn, epoch)
        logits = model.fuse(feats); loss = ce(logits, y); loss.backward()
        mod.after_backward(model.modality_parameters(), epoch)
        optimizer.step()

    mode: 'none' | 'opm' | 'ogm' | 'both'
    """

    def __init__(self, num_modalities: int, mode: str = "ogm",
                 # OPM
                 q_base: float = 0.5, lam: float = 0.5,
                 # OGM
                 alpha: float = 0.5, ge: bool = True, ge_scope: str = "all",
                 # shared
                 z: str = "tanh", start_epoch: int = 0, end_epoch: int = 10 ** 9):
        mode = mode.lower()
        assert mode in ("none", "opm", "ogm", "both")
        self.M = num_modalities
        self.mode = mode
        self.start_epoch, self.end_epoch = start_epoch, end_epoch
        self.opm = OPM(num_modalities, q_base=q_base, lam=lam, z=z) if mode in ("opm", "both") else None
        self.ogm = OGMGE(num_modalities, alpha=alpha, z=z, ge=ge, ge_scope=ge_scope) if mode in ("ogm", "both") else None
        self.rho = torch.ones(num_modalities)
        self.scores = None

    def active(self, epoch: int) -> bool:
        return self.mode != "none" and self.start_epoch <= epoch <= self.end_epoch

    @torch.no_grad()
    def monitor(self, feats: Sequence[Tensor], labels: Tensor,
                unimodal_logits_fn: Callable[[List[Tensor]], List[Tensor]]) -> Tensor:
        """Eq. 6 + 7 on the *undropped* features. Always safe to call (also in 'none' mode, for logging)."""
        ulogits = unimodal_logits_fn([f.detach() for f in feats])
        self.scores = unimodal_scores_from_logits(ulogits, labels)   # [M, B]
        self.rho = discrepancy_ratios(self.scores).cpu()             # [M]
        self.last_unimodal_logits = [u.detach() for u in ulogits]
        return self.rho

    def before_fusion(self, feats: List[Tensor], labels: Tensor,
                      unimodal_logits_fn: Callable[[List[Tensor]], List[Tensor]], epoch: int):
        rho = self.monitor(feats, labels, unimodal_logits_fn)
        if self.opm is not None and self.active(epoch):
            feats = self.opm.apply(feats)     # drop with q_t
            self.opm.update(rho)              # -> q_{t+1}
        return feats, rho

    def after_backward(self, modality_params: Sequence[Iterable[nn.Parameter]], epoch: int) -> None:
        if self.ogm is not None and self.active(epoch):
            k = self.ogm.coefficients(self.rho)
            self.ogm.apply_(modality_params, k)

    def state(self) -> Dict[str, float]:
        d = {f"rho_{m}": float(self.rho[m]) for m in range(self.M)}
        if self.opm is not None:
            d.update(self.opm.state())
        if self.ogm is not None:
            d.update(self.ogm.state())
        return d


__all__ = [
    "make_z", "unimodal_scores_from_logits", "unimodal_logits_zero_out", "discrepancy_ratios",
    "OPM", "OGMGE", "BalancedModulator",
]
