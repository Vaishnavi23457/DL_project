"""
Unit tests for bml.modulation  (run:  python -m pytest -q   or   python tests/test_modulation.py)
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bml.modulation import OGMGE, OPM, BalancedModulator, discrepancy_ratios, unimodal_scores_from_logits  # noqa
from bml.models import LateFusionModel  # noqa


def test_scores_and_ratios_two_modalities():
    # paper's numeric example: audio strong [4,0,0], visual weak [0.5,0.3,0.2], true class 0
    la = torch.tensor([[4.0, 0.0, 0.0]])
    lv = torch.tensor([[0.5, 0.3, 0.2]])
    y = torch.tensor([0])
    s = unimodal_scores_from_logits([la, lv], y)
    assert s.shape == (2, 1)
    assert abs(s[0, 0].item() - 0.965) < 1e-2 and abs(s[1, 0].item() - 0.391) < 1e-2
    rho = discrepancy_ratios(s)
    assert abs(rho[0].item() - 2.47) < 0.02 and abs(rho[1].item() - 1 / 2.47) < 0.02
    assert rho[0] > 1 > rho[1]


def test_ratios_three_modalities_min_is_never_dominant():
    s = torch.tensor([[3.0], [2.0], [1.0]])
    rho = discrepancy_ratios(s)
    assert abs(rho[0].item() - 2.25) < 1e-6      # (3/2 + 3/1)/2
    assert abs(rho[1].item() - (2 / 3 + 2) / 2) < 1e-6
    assert rho[2] < 1                           # weakest modality always rho < 1


def test_opm_probabilities_and_dropping():
    opm = OPM(2, q_base=0.5, lam=0.5)
    q = opm.update(torch.tensor([2.47, 0.41]))
    assert abs(q[0].item() - 0.5 * (1 + 0.5 * math.tanh(1.47))) < 1e-6   # ~0.725
    assert q[1].item() == 0.0
    torch.manual_seed(0)
    f0, f1 = torch.ones(1000, 8), torch.ones(1000, 4)
    d0, d1 = opm.apply([f0, f1])
    frac_dropped = (d0.abs().sum(1) == 0).float().mean().item()
    assert abs(frac_dropped - q[0].item()) < 0.05                       # empirical drop rate ~ q
    assert torch.equal(d1, f1)                                          # weak modality untouched
    # no modulation when rho <= 1 for everyone (e.g. rho = 1, 1)
    opm.update(torch.tensor([1.0, 1.0]))
    assert float(opm.q.max()) == 0.0


def test_ogm_coefficients_and_gradients():
    ogm = OGMGE(2, alpha=0.5, ge=True, ge_scope="modulated")
    k = ogm.coefficients(torch.tensor([2.47, 0.41]))
    assert abs(k[0].item() - (1 - 0.5 * math.tanh(1.47))) < 1e-6      # ~0.55
    assert k[1].item() == 1.0
    p0 = nn.Parameter(torch.randn(16, 8)); p0.grad = torch.randn(16, 8)
    p1 = nn.Parameter(torch.randn(16, 8)); p1.grad = torch.randn(16, 8)
    g0, g1 = p0.grad.clone(), p1.grad.clone()
    ogm.apply_([[p0], [p1]], k)
    assert torch.equal(p1.grad, g1)                                     # weak modality untouched
    assert not torch.allclose(p0.grad, g0 * k[0])                       # noise added on top of scaling
    # plain OGM (no GE): exactly scaled
    ogm2 = OGMGE(2, alpha=0.5, ge=False)
    p0.grad = g0.clone()
    ogm2.apply_([[p0], [p1]], k)
    assert torch.allclose(p0.grad, g0 * k[0])


def test_ge_noise_variance_recovers():
    """Eq 17: Var(update noise) with GE = (k^2 + 1) * Var(original)."""
    torch.manual_seed(0)
    k = 0.5
    ogm = OGMGE(1, alpha=0.5, ge=True)
    g = torch.randn(400, 500)                      # 2-D weight, pretend gradient noise ~ N(0,1)
    p = nn.Parameter(torch.zeros(400, 500)); p.grad = g.clone()
    ogm.apply_([[p]], torch.tensor([k]))
    var = p.grad.var().item()
    assert abs(var - (k ** 2 + 1)) < 0.05          # 1.25 expected


def test_late_fusion_block_logits_match_full_head():
    torch.manual_seed(0)

    class Enc(nn.Module):
        def __init__(self, d):
            super().__init__(); self.out_dim = d; self.l = nn.Linear(d, d)

        def forward(self, x): return self.l(x)

    model = LateFusionModel({"a": Enc(6), "b": Enc(4)}, num_classes=5, head="linear")
    inputs = {"a": torch.randn(3, 6), "b": torch.randn(3, 4)}
    feats = model.encode(inputs)
    full = model.fuse(feats)
    blocks = model.unimodal_logits(feats, mode="linear")
    assert torch.allclose(full, blocks[0] + blocks[1], atol=1e-5)     # Eq 2: sum of blocks == full logits
    zo = model.unimodal_logits(feats, mode="zero_out")
    assert torch.allclose(zo[0], blocks[0] + model.head.bias / 2, atol=1e-5)  # zero-out keeps full bias


def test_balanced_modulator_end_to_end():
    torch.manual_seed(0)

    class Enc(nn.Module):
        def __init__(self, d):
            super().__init__(); self.out_dim = d; self.l = nn.Linear(d, d)

        def forward(self, x): return self.l(x)

    model = LateFusionModel({"a": Enc(6), "b": Enc(4)}, num_classes=3)
    mod = BalancedModulator(2, mode="both")
    x = {"a": torch.randn(8, 6), "b": torch.randn(8, 4)}
    y = torch.randint(0, 3, (8,))
    feats = model.encode(x)
    feats, rho = mod.before_fusion(feats, y, model.unimodal_logits, epoch=0)
    loss = F.cross_entropy(model.fuse(feats), y)
    loss.backward()
    mod.after_backward(model.modality_parameters(), epoch=0)
    st = mod.state()
    assert set(st) == {"rho_0", "rho_1", "q_0", "q_1", "k_0", "k_1"}
    assert all(p.grad is not None for p in model.head.parameters())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok  ", name)
    print("all tests passed")
