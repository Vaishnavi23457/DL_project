"""Earlier learning strategies, adapted to this project's feature-level fusion.

These are project adaptations, not the original papers' pretrained architectures.
See results/BASELINE_COMPARISON.md for equations, sources and protocol differences.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import LateFusionModel


class FeatureDropout:
    """Fixed modality/unit dropout, annealing, curriculum, or multinomial dropout."""

    METHODS = ("moddrop", "dropout", "annealed", "evolutional", "curriculum")

    def __init__(self, method, total_steps, p=0.5, curriculum_rate=5.0):
        if method not in self.METHODS or not 0 <= p < 1:
            raise ValueError("unknown method or invalid drop probability")
        self.method, self.p = method, p
        self.total_steps = max(1, total_steps)
        self.curriculum_rate = curriculum_rate
        self.step = 0

    def probability(self):
        progress = min(self.step / max(1, self.total_steps - 1), 1.0)
        if self.method == "annealed":
            return self.p * (1 - progress)
        if self.method == "curriculum":
            return self.p * (1 - math.exp(-self.curriculum_rate * progress))
        return self.p

    def __call__(self, feats):
        p = self.probability()
        self.step += 1
        if self.method == "moddrop":
            # One independent Bernoulli mask per sample and modality.
            return [f * (torch.rand(f.shape[0], 1, device=f.device) >= p) / (1 - p)
                    for f in feats]
        x = torch.cat(feats, dim=1)
        if self.method == "evolutional":
            # Li et al.: k multinomial draws, with p_j proportional to RMS(x_j).
            rms = x.detach().square().mean(0).sqrt()
            probs = rms / rms.sum() if rms.sum() > 0 else torch.ones_like(rms) / rms.numel()
            k = max(1, round(x.shape[1] * (1 - p)))
            draws = torch.multinomial(probs.expand(x.shape[0], -1), k, replacement=True)
            counts = torch.zeros_like(x).scatter_add_(1, draws, torch.ones_like(draws, dtype=x.dtype))
            x = x * counts / (k * probs.clamp_min(torch.finfo(x.dtype).eps))
        else:
            x = F.dropout(x, p=p, training=True)
        return list(x.split([f.shape[1] for f in feats], dim=1))


class GBlendingModel(LateFusionModel):
    """Independent auxiliary classifiers; only the fusion head is used at test."""

    def __init__(self, base):
        super().__init__(base.encoders, base.num_classes)
        self.head.load_state_dict(base.head.state_dict())
        self.aux_heads = nn.ModuleList([nn.Linear(d, self.num_classes) for d in self.dims])
        # Start each independent head at the base fusion head's corresponding block.
        start = 0
        with torch.no_grad():
            for head, d in zip(self.aux_heads, self.dims):
                head.weight.copy_(base.head.weight[:, start:start + d])
                head.bias.copy_(base.head.bias / self.M)
                start += d

    def component_logits(self, feats):
        return [head(f) for head, f in zip(self.aux_heads, feats)] + [self.fuse(feats)]

    def head_parameters(self):
        return list(self.head.parameters()) + list(self.aux_heads.parameters())


def blending_weights(gains, overfit_changes, eps=1e-6):
    """Offline GBlending G/O^2; retain positive validation gains, guard O=0.

    A nonpositive gain receives zero weight. If no pilot improves validation loss,
    use uniform weights and record that fallback in the run's pilot evidence.
    """
    scores = [max(0.0, g) / max(o * o, eps * eps) for g, o in zip(gains, overfit_changes)]
    total = sum(scores)
    return [s / total for s in scores] if total > 0 else [1 / len(scores)] * len(scores)


class FeatureMMTM(nn.Module):
    """One squeeze/excitation module on the final pooled feature vectors."""

    def __init__(self, dims, ratio=4):
        super().__init__()
        hidden = max(1, int(2 * sum(dims) / ratio))
        self.squeeze = nn.Linear(sum(dims), hidden)
        self.excite = nn.ModuleList([nn.Linear(hidden, d) for d in dims])
        for i, d in enumerate(dims):
            self.register_buffer(f"mean_gate_{i}", torch.zeros(d))
        self.register_buffer("regular_steps", torch.tensor(0, dtype=torch.long))

    def forward(self, feats, caring=None):
        hidden = F.relu(self.squeeze(torch.cat(feats, dim=1)))
        gates = [torch.sigmoid(head(hidden)) for head in self.excite]
        if self.training and caring is None:
            n = int(self.regular_steps)
            with torch.no_grad():
                for i, gate in enumerate(gates):
                    mean = getattr(self, f"mean_gate_{i}")
                    mean.mul_(n / (n + 1)).add_(gate.detach().mean(0) / (n + 1))
                self.regular_steps.add_(1)
        if self.training and caring is not None:
            # Cut the caring modality's cross-modal gate gradient during curation.
            gates[caring] = getattr(self, f"mean_gate_{caring}").detach().clone().expand_as(gates[caring])
        return [f * gate for f, gate in zip(feats, gates)]


class MMTMModel(LateFusionModel):
    def __init__(self, base):
        super().__init__(base.encoders, base.num_classes)
        self.head.load_state_dict(base.head.state_dict())
        self.mmtm = FeatureMMTM(self.dims)
        self.caring = None

    def encode(self, inputs):
        return self.mmtm(super().encode(inputs), caring=self.caring if self.training else None)

    def head_parameters(self):
        return list(self.head.parameters()) + list(self.mmtm.parameters())


class GreedyController:
    """Accumulate conditional learning speeds, then curate Q steps (Wu et al.)."""

    def __init__(self, alpha=0.01, window=5, warmup_epochs=2):
        self.alpha, self.window, self.warmup_epochs = alpha, window, warmup_epochs
        self.main, self.bypass = [0.0, 0.0], [0.0, 0.0]
        self.remaining = 0
        self.caring = None
        self.last_speed = 0.0
        self.curation_steps = 0

    @staticmethod
    def relative_gradient(params):
        params = list(params)
        wn = sum(p.detach().square().sum().item() for p in params)
        gn = sum(p.grad.detach().square().sum().item() for p in params if p.grad is not None)
        return gn / max(wn, 1e-12)

    def after_backward(self, model, epoch):
        if self.remaining:
            self.curation_steps += 1
            self.remaining -= 1
            if not self.remaining:
                self.caring = None
        else:
            for i, name in enumerate(model.modality_names):
                self.main[i] += self.relative_gradient(model.encoders[name].parameters())
                ps = list(model.mmtm.squeeze.parameters()) + list(model.mmtm.excite[i].parameters())
                self.bypass[i] += self.relative_gradient(ps)
            ratios = [max(b, 1e-12) / max(m, 1e-12) for b, m in zip(self.bypass, self.main)]
            self.last_speed = math.log10(ratios[0]) - math.log10(ratios[1])
            if epoch >= self.warmup_epochs and abs(self.last_speed) > self.alpha:
                self.caring = 0 if self.last_speed > 0 else 1
                self.remaining = self.window
        model.caring = self.caring
