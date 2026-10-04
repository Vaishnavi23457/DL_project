import torch

from bml.models import LateFusionModel, SmallCNN, TextTransformerEncoder
from bml.xai import head_block_contributions, integrated_gradients_image, token_importance_text


def tiny_model(head="linear"):
    torch.manual_seed(0)
    encoders = {"image": SmallCNN(width=8), "text": TextTransformerEncoder(vocab_size=50, d_model=32,
                                                                            nhead=2, num_layers=1, dim_ff=64)}
    return LateFusionModel(encoders, num_classes=4, head=head)


def tiny_inputs(B=3, L=6):
    torch.manual_seed(1)
    return {
        "image": torch.randn(B, 3, 32, 32),
        "text": {"input_ids": torch.randint(1, 50, (B, L)),
                 "attention_mask": torch.ones(B, L, dtype=torch.long)},
    }, torch.randint(0, 4, (B,))


def test_unimodal_blocks_sum_to_fused_logit():
    model = tiny_model()
    inputs, _ = tiny_inputs()
    with torch.no_grad():
        feats = model.encode(inputs)
        blocks = model.unimodal_logits(feats, mode="linear")
        fused = model.fuse(feats)
    assert len(blocks) == 2 and blocks[0].shape == fused.shape
    assert torch.allclose(blocks[0] + blocks[1], fused, atol=1e-5)


def test_head_block_contributions_delta_near_zero():
    model = tiny_model()
    inputs, y = tiny_inputs()
    with torch.no_grad():
        feats = model.encode(inputs)
        out = head_block_contributions(model, feats, target=y)
    assert out["blocks"].shape == (3, 2)
    assert out["delta"] < 1e-5
    assert torch.allclose(out["blocks"].sum(1), out["fused"], atol=1e-5)


def test_integrated_gradients_shape_and_completeness():
    model = tiny_model()
    inputs, _ = tiny_inputs(B=2)
    with torch.no_grad():
        target = model(inputs).argmax(1)
    ig = integrated_gradients_image(model, inputs, target, steps=24)
    assert ig.shape == inputs["image"].shape
    # completeness: sum(IG) ~= F(x) - F(0)
    model.eval()
    with torch.no_grad():
        f_x = model(inputs).gather(1, target.view(-1, 1)).squeeze(1)
        zero = {**inputs, "image": torch.zeros_like(inputs["image"])}
        f_0 = model(zero).gather(1, target.view(-1, 1)).squeeze(1)
    s = ig.flatten(1).sum(1)
    assert torch.allclose(s, f_x - f_0, rtol=0.15, atol=0.5)


def test_token_importance_shape_and_padding():
    model = tiny_model()
    inputs, _ = tiny_inputs(B=3, L=6)
    inputs["text"]["attention_mask"][:, -2:] = 0            # last two tokens are padding
    with torch.no_grad():
        target = model(inputs).argmax(1)
    scores = token_importance_text(model, inputs, target)
    assert scores.shape == (3, 6)
    assert (scores[:, -2:] == 0).all()                      # pads contribute nothing
    assert torch.isfinite(scores).all()
