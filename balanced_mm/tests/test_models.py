import torch

from bml.models import LateFusionModel, SmallCNN, TextTransformerEncoder, build_image_encoder, build_text_encoder


def build(head="linear"):
    torch.manual_seed(0)
    encoders = {"image": SmallCNN(width=8),
                "text": TextTransformerEncoder(vocab_size=40, d_model=32, nhead=2, num_layers=1, dim_ff=64)}
    return LateFusionModel(encoders, num_classes=5, head=head)


def inputs(B=4, L=7):
    torch.manual_seed(2)
    return {"image": torch.randn(B, 3, 32, 32),
            "text": {"input_ids": torch.randint(1, 40, (B, L)),
                     "attention_mask": torch.ones(B, L, dtype=torch.long)}}


def test_forward_shapes():
    m = build()
    out = m(inputs())
    assert out.shape == (4, 5)
    feats = m.encode(inputs())
    assert len(feats) == 2 and feats[0].shape == (4, 128) and feats[1].shape == (4, 32)


def test_unimodal_linear_blocks_sum_to_fused():
    m = build(head="linear")
    x = inputs()
    with torch.no_grad():
        feats = m.encode(x)
        blocks = m.unimodal_logits(feats, mode="linear")
        fused = m.fuse(feats)
    assert torch.allclose(blocks[0] + blocks[1], fused, atol=1e-5)


def test_zero_out_mode_runs_for_mlp_head():
    m = build(head="mlp")
    x = inputs()
    with torch.no_grad():
        feats = m.encode(x)
        blocks = m.unimodal_logits(feats, mode="auto")   # -> zero_out for mlp
        fused = m.fuse(feats)
    assert len(blocks) == 2 and blocks[0].shape == fused.shape


def test_modality_parameter_groups():
    m = build()
    groups = m.modality_parameters()
    assert len(groups) == 2
    assert all(len(g) > 0 for g in groups)
    assert len(m.head_parameters()) >= 2   # weight + bias


def test_factories():
    img = build_image_encoder("smallcnn")
    txt = build_text_encoder("transformer", vocab_size=40, d_model=32, layers=1)
    with torch.no_grad():
        assert img(torch.randn(2, 3, 32, 32)).shape == (2, 128)
        out = txt({"input_ids": torch.randint(0, 40, (2, 5)),
                   "attention_mask": torch.ones(2, 5, dtype=torch.long)})
        assert out.shape == (2, 32)


def test_vector_mlp_encoder():
    from bml.models import VectorMLPEncoder, build_vector_encoder
    enc = build_vector_encoder("mlp", in_dim=300, out_dim=64, layers=2)
    assert isinstance(enc, VectorMLPEncoder)
    out = enc(torch.randn(5, 300))
    assert out.shape == (5, 64) and enc.out_dim == 64
