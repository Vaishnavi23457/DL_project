"""
bml.models
==========
Encoders + late-fusion model with block-wise uni-modal logits (needed for Eq. 6).

  ImageEncoderResNet18  : torchvision ResNet-18 (paper default), fc removed -> 512-d
  SmallCNN              : tiny CNN for fast CPU tests -> 128-d
  TextTransformerEncoder: embedding + 2-layer Transformer encoder (trained from scratch) -> 256-d
  HFTextEncoder         : optional HuggingFace model (e.g. distilbert-base-uncased) -> hidden-d
  LateFusionModel       : concat features -> linear (or MLP) head; exposes
                              encode(), fuse(), unimodal_logits(), modality_parameters()
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modulation import unimodal_logits_zero_out

Tensor = torch.Tensor


# ----------------------------------------------------------------------------- image encoders
class ImageEncoderResNet18(nn.Module):
    out_dim = 512

    def __init__(self, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        if in_channels != 3:  # e.g. 1-channel spectrogram like the paper's audio branch
            w = net.conv1.weight.data
            net.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            if pretrained:
                net.conv1.weight.data = w.mean(1, keepdim=True).repeat(1, in_channels, 1, 1)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x: Tensor) -> Tensor:  # [B, C, H, W] -> [B, 512]
        return self.net(x)


class SmallCNN(nn.Module):
    """4 conv blocks + GAP. Good enough to test the pipeline on CPU in a minute."""
    out_dim = 128

    def __init__(self, in_channels: int = 3, width: int = 32):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.net = nn.Sequential(block(in_channels, width), block(width, width * 2), block(width * 2, width * 4), block(width * 4, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.net(x)).flatten(1)


# ----------------------------------------------------------------------------- text encoders
class TextTransformerEncoder(nn.Module):
    """Input: dict(input_ids [B, L] long, attention_mask [B, L] {0,1}). Output: [B, out_dim]."""

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 512, dropout: float = 0.1, max_len: int = 512, pad_id: int = 0):
        super().__init__()
        self.out_dim = d_model
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        ids, mask = x["input_ids"], x["attention_mask"]
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        h = self.drop(self.tok(ids) + self.pos(pos))
        h = self.enc(h, src_key_padding_mask=(mask == 0))
        h = self.norm(h)
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)   # masked mean pooling


class HFTextEncoder(nn.Module):
    """Optional: pretrained HuggingFace encoder (needs `pip install transformers`)."""

    def __init__(self, model_name: str = "distilbert-base-uncased", freeze: bool = False):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        out = self.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp_min(1.0)


# ----------------------------------------------------------------------------- vector encoders (audio features etc.)
class VectorMLPEncoder(nn.Module):
    """Pre-extracted feature vectors (e.g. MELD 300-d audio embeddings) -> [B, out_dim]."""
    out_dim = 128

    def __init__(self, in_dim: int = 300, out_dim: int = 128, hidden: int = 256,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(max(1, layers - 1)):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: Tensor) -> Tensor:   # [B, D] -> [B, out_dim]
        return self.net(x)


# ----------------------------------------------------------------------------- late fusion
class LateFusionModel(nn.Module):
    """
    f(x) = W [phi^1; ...; phi^M] + b  = sum_m W^m phi^m + b          (Eq. 1 / Eq. 2)

    encoders : ordered dict  name -> nn.Module (each must expose .out_dim and take inputs[name])
    head     : 'linear' (paper setting; uni-modal logits = W^m phi^m + b/M)
               'mlp'    (multi-layer head; uni-modal logits via zero-out strategy)
    """

    def __init__(self, encoders: Dict[str, nn.Module], num_classes: int, head: str = "linear",
                 hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        assert head in ("linear", "mlp")
        self.modality_names: List[str] = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.dims: List[int] = [encoders[n].out_dim for n in self.modality_names]
        self.num_classes = num_classes
        self.head_type = head
        total = sum(self.dims)
        if head == "linear":
            self.head = nn.Linear(total, num_classes)
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(total, hidden), nn.ReLU(inplace=True),
                                      nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    @property
    def M(self) -> int:
        return len(self.modality_names)

    # ---- forward pieces -------------------------------------------------------------
    def encode(self, inputs: Dict[str, object]) -> List[Tensor]:
        return [self.encoders[n](inputs[n]) for n in self.modality_names]

    def fuse(self, feats: Sequence[Tensor]) -> Tensor:
        return self.head(torch.cat(list(feats), dim=1))

    def forward(self, inputs: Dict[str, object]) -> Tensor:
        return self.fuse(self.encode(inputs))

    # ---- uni-modal logits for Eq. 6 -------------------------------------------------
    def unimodal_logits(self, feats: Sequence[Tensor], mode: str = "auto") -> List[Tensor]:
        """
        mode 'linear'  : W^m phi^m + b/M  (exact block decomposition of the linear head)
        mode 'zero_out': keep m, zero the others, run the head (works for any head)
        mode 'auto'    : linear if the head is linear else zero_out
        """
        if mode == "auto":
            mode = "linear" if self.head_type == "linear" else "zero_out"
        if mode == "linear":
            assert self.head_type == "linear", "block-wise logits need a linear head; use mode='zero_out'"
            W, b = self.head.weight, self.head.bias
            outs, start = [], 0
            for m, f in enumerate(feats):
                d = self.dims[m]
                Wm = W[:, start:start + d]
                outs.append(F.linear(f, Wm) + b / self.M)
                start += d
            return outs
        return unimodal_logits_zero_out(self.fuse, feats)

    # ---- parameter groups per modality (theta^m) for OGM ------------------------------
    def modality_parameters(self) -> List[List[nn.Parameter]]:
        return [[p for p in self.encoders[n].parameters() if p.requires_grad] for n in self.modality_names]

    def head_parameters(self) -> List[nn.Parameter]:
        return list(self.head.parameters())


# ----------------------------------------------------------------------------- factory
def build_image_encoder(kind: str = "resnet18", pretrained: bool = False, in_channels: int = 3) -> nn.Module:
    if kind == "resnet18":
        return ImageEncoderResNet18(pretrained=pretrained, in_channels=in_channels)
    if kind == "smallcnn":
        return SmallCNN(in_channels=in_channels)
    raise ValueError(kind)


def build_text_encoder(kind: str = "transformer", vocab_size: int = 30000, hf_model: str = "distilbert-base-uncased",
                       max_len: int = 128, d_model: int = 256, layers: int = 2, freeze_hf: bool = False) -> nn.Module:
    if kind == "transformer":
        return TextTransformerEncoder(vocab_size, d_model=d_model, num_layers=layers, max_len=max(max_len, 512))
    if kind == "hf":
        return HFTextEncoder(hf_model, freeze=freeze_hf)
    raise ValueError(kind)


def build_vector_encoder(kind: str = "mlp", in_dim: int = 300, out_dim: int = 128, **kw) -> nn.Module:
    if kind == "mlp":
        return VectorMLPEncoder(in_dim, out_dim, **kw)
    raise ValueError(kind)
