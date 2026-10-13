"""Check algorithm distinctions and the gradients used by comparison baselines."""
import math

import pytest
import torch
from torch import nn

from bml.baselines import (FeatureDropout, FeatureMMTM, GBlendingModel,
                           GreedyController, MMTMModel, blending_weights)
from bml.models import LateFusionModel


def base_model():
    class Encoder(nn.Linear):
        out_dim = 4
    return LateFusionModel({"audio": Encoder(3, 4), "text": Encoder(3, 4)}, 3)


def test_moddrop_drops_whole_modalities_independently():
    torch.manual_seed(1)
    feats = [torch.ones(1024, 6), torch.ones(1024, 4)]
    out = FeatureDropout("moddrop", 10)(feats)
    for f in out:
        assert torch.all(f == f[:, :1])
        assert set(f.unique().tolist()) == {0.0, 2.0}
    assert ((out[0][:, 0] == 0) & (out[1][:, 0] == 0)).any()


def test_schedules_have_opposite_direction():
    annealed = FeatureDropout("annealed", 101)
    curriculum = FeatureDropout("curriculum", 101)
    assert annealed.probability() == 0.5
    assert curriculum.probability() == 0.0
    annealed.step = curriculum.step = 100
    assert annealed.probability() == 0.0
    assert curriculum.probability() == pytest.approx(0.5 * (1 - math.exp(-5)))


def test_evolutional_dropout_is_unbiased_and_backpropagates():
    torch.manual_seed(4)
    x = torch.tensor([1.0, 2.0, 4.0, 0.0]).expand(20000, -1).clone().requires_grad_()
    out = FeatureDropout("evolutional", 10)([x])[0]
    assert torch.allclose(out.mean(0), x[0], atol=0.09)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    zeros = FeatureDropout("evolutional", 10)([torch.zeros(2, 4)])[0]
    assert torch.equal(zeros, torch.zeros_like(zeros))


def test_blending_weights_use_generalization_over_squared_overfitting():
    assert blending_weights([0.2, 0.2], [0.1, 0.2]) == pytest.approx([0.8, 0.2])
    assert blending_weights([-0.1, 0.2], [0.1, 0.2]) == [0.0, 1.0]
    assert blending_weights([0.0, -0.2], [0.0, 0.0]) == [0.5, 0.5]


def test_auxiliary_classifiers_do_not_change_fused_initial_prediction():
    base = base_model()
    adapted = GBlendingModel(base)
    feats = [torch.randn(5, 4), torch.randn(5, 4)]
    assert torch.equal(base.fuse(feats), adapted.fuse(feats))
    assert torch.allclose(sum(adapted.component_logits(feats)[:-1]), adapted.fuse(feats), atol=1e-6)
    # Optimizer must include independent heads, without duplicated parameters.
    ids = [id(p) for p in adapted.head_parameters()]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in adapted.head.parameters()} | {id(p) for p in adapted.aux_heads.parameters()}


def test_mmtm_curation_cuts_only_selected_gate_gradient_and_eval_is_frozen():
    module = FeatureMMTM([4, 6])
    feats = [torch.randn(5, 4, requires_grad=True), torch.randn(5, 6, requires_grad=True)]
    module(feats)  # gather training-only gate means
    means = [module.mean_gate_0.clone(), module.mean_gate_1.clone()]
    out = module(feats, caring=0)
    sum(x.sum() for x in out).backward()
    assert module.excite[0].weight.grad is None
    assert module.excite[1].weight.grad is not None
    assert feats[0].grad is not None and feats[1].grad is not None
    assert torch.equal(module.mean_gate_0, means[0]) and torch.equal(module.mean_gate_1, means[1])
    steps = module.regular_steps.clone()
    module.eval()
    # Evaluation always uses learned gates and cannot modify running means.
    assert all(torch.equal(a, b) for a, b in zip(module(feats), module(feats, caring=0)))
    assert torch.equal(module.regular_steps, steps)


def test_greedy_enters_five_step_curation_after_warmup():
    model = MMTMModel(base_model())
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    # Deliberately impose different bypass/main relative speeds.
    for p in model.mmtm.excite[0].parameters():
        p.grad *= 100
    controller = GreedyController()
    controller.after_backward(model, epoch=0)
    assert model.caring is None
    controller.after_backward(model, epoch=2)
    assert model.caring == 0 and controller.remaining == 5
    for _ in range(5):
        controller.after_backward(model, epoch=2)
    assert model.caring is None and controller.curation_steps == 5
