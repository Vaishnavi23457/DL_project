#!/usr/bin/env python
"""
train.py - Text + Image late-fusion training with OPM / OGM-GE modulation.

Examples
--------
# 1) smoke test on synthetic data (CPU friendly)
python train.py --data synthetic --image_encoder smallcnn --img_size 64 --epochs 5 --modulation ogm --out_dir runs/syn_ogm

# 2) your own CSV data (columns: image,text,label), ResNet-18 + Transformer text encoder
python train.py --data csv --train_csv data/train.csv --val_csv data/val.csv --img_root data/images \
       --modulation both --epochs 40 --out_dir runs/my_both

# 3) same but pretrained image encoder + HuggingFace text encoder (pip install transformers)
python train.py --data csv --train_csv data/train.csv --val_csv data/val.csv --img_root data/images \
       --pretrained_image --text_encoder hf --hf_model distilbert-base-uncased --lr 1e-3 --lr_text 2e-5 \
       --modulation ogm --alpha 0.5 --out_dir runs/my_hf_ogm
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bml.data import build_dataloaders                                  # noqa: E402
from bml.engine import evaluate, probe_unimodal_encoders, train_one_epoch  # noqa: E402
from bml.models import LateFusionModel, build_image_encoder, build_text_encoder, build_vector_encoder  # noqa: E402
from bml.modulation import BalancedModulator                             # noqa: E402
from bml.utils import Logger, count_params, fmt, get_device, save_checkpoint, save_json, set_seed  # noqa: E402


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Balanced multimodal learning (OPM / OGM-GE) - Text + Image")
    # ---------------- data
    p.add_argument("--data", choices=["synthetic", "csv"], default="synthetic")
    p.add_argument("--train_csv"); p.add_argument("--val_csv"); p.add_argument("--test_csv")
    p.add_argument("--img_root", default="")
    p.add_argument("--col_image", default="image"); p.add_argument("--col_text", default="text"); p.add_argument("--col_label", default="label")
    p.add_argument("--col_audio", default="audio", help="optional CSV column with .npy feature-vector path (e.g. MELD)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--min_freq", type=int, default=2); p.add_argument("--max_vocab", type=int, default=30000)
    p.add_argument("--num_classes", type=int, default=6, help="synthetic only (csv: inferred)")
    p.add_argument("--syn_train_n", type=int, default=1920); p.add_argument("--syn_val_n", type=int, default=480)
    p.add_argument("--syn_text_p", type=float, default=0.7, help="synthetic: prob. that text is informative (has the class keyword)")
    p.add_argument("--syn_img_p", type=float, default=0.7, help="synthetic: prob. that image is informative (has the class patch)")
    p.add_argument("--syn_img_noise", type=float, default=1.0, help="synthetic: image noise std (higher = harder)")
    p.add_argument("--syn_synonyms", type=int, default=1, help="synthetic: keyword synonyms per class (1 = default demo; 6 = severe imbalance stress test)")
    # ---------------- model
    p.add_argument("--image_encoder", choices=["resnet18", "smallcnn"], default="resnet18")
    p.add_argument("--pretrained_image", action="store_true")
    p.add_argument("--text_encoder", choices=["transformer", "hf"], default="transformer")
    p.add_argument("--hf_model", default="distilbert-base-uncased"); p.add_argument("--freeze_hf", action="store_true")
    p.add_argument("--text_dim", type=int, default=256); p.add_argument("--text_layers", type=int, default=2)
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--only", choices=["none", "image", "text", "audio"], default="none",
                   help="train a uni-modal baseline (paper Fig.1 reference lines)")
    p.add_argument("--modalities", default="auto",
                   help="'auto' = infer from the CSV columns (image/audio/text); or comma list e.g. 'text,audio'")
    p.add_argument("--audio_encoder", choices=["mlp"], default="mlp")
    p.add_argument("--audio_dim", type=int, default=0, help="0 = infer from the first .npy in the CSV")
    p.add_argument("--audio_out_dim", type=int, default=128)
    # ---------------- modulation
    p.add_argument("--modulation", choices=["none", "opm", "ogm", "both"], default="ogm")
    p.add_argument("--q_base", type=float, default=0.5); p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--no_ge", action="store_true", help="OGM without the Gaussian-noise generalization enhancement")
    p.add_argument("--ge_scope", choices=["all", "modulated"], default="all")
    p.add_argument("--z", choices=["tanh", "sigmoid"], default="tanh")
    p.add_argument("--score_mode", choices=["auto", "linear", "zero_out"], default="auto")
    p.add_argument("--mod_start", type=int, default=0); p.add_argument("--mod_end", type=int, default=10 ** 9)
    # ---------------- optimisation
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--keep_last_batch", action="store_true", help="use every training row, including the final partial batch")
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--lr_text", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9); p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--sched", choices=["cos", "step", "none"], default="cos")
    p.add_argument("--lr_step", type=int, default=30); p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--amp", action="store_true")
    # ---------------- misc
    p.add_argument("--num_workers", type=int, default=2); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_name", default="", help="tag saved in config.json (e.g. food101, meld) for grouped reports")
    p.add_argument("--device", default="auto"); p.add_argument("--out_dir", default="runs/exp")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--probe", action="store_true", help="paper-style linear probing of each frozen encoder at the end")
    p.add_argument("--eval_every", type=int, default=1)
    return p.parse_args(argv)


def build_model(args, num_classes: int, vocab_size: int, modalities=None, audio_dim=None) -> LateFusionModel:
    if modalities is None:
        spec = getattr(args, "modalities", "auto")
        modalities = ["image", "text"] if spec == "auto" \
            else [m.strip() for m in spec.split(",") if m.strip()]
    encoders = {}
    for m in modalities:
        if m == "image":
            encoders[m] = build_image_encoder(args.image_encoder, pretrained=args.pretrained_image)
        elif m == "text":
            encoders[m] = build_text_encoder(args.text_encoder, vocab_size=vocab_size, hf_model=args.hf_model,
                                             max_len=args.max_len, d_model=args.text_dim, layers=args.text_layers,
                                             freeze_hf=args.freeze_hf)
        elif m == "audio":
            d = audio_dim or getattr(args, "audio_dim", 0)
            assert d, "audio feature dim unknown: pass --audio_dim (or use a CSV whose .npy files expose it)"
            encoders[m] = build_vector_encoder(getattr(args, "audio_encoder", "mlp"), in_dim=d,
                                               out_dim=getattr(args, "audio_out_dim", 128))
        else:
            raise ValueError(f"unknown modality: {m!r}")
    if args.only != "none":
        encoders = {args.only: encoders[args.only]}
    return LateFusionModel(encoders, num_classes=num_classes, head=args.head)


def build_optimizer(args, model: LateFusionModel):
    groups = [{"params": model.head_parameters(), "lr": args.lr}]
    for name in model.modality_names:
        ps = [p for p in model.encoders[name].parameters() if p.requires_grad]
        lr = args.lr_text if (name == "text" and args.lr_text is not None) else args.lr
        groups.append({"params": ps, "lr": lr})
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args, opt):
    if args.sched == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.sched == "step":
        return torch.optim.lr_scheduler.StepLR(opt, args.lr_step, args.lr_gamma)
    return None


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    log = Logger(os.path.join(args.out_dir, "train.log"))
    set_seed(args.seed)
    device = get_device(args.device)
    log(f"device={device}  args={vars(args)}")

    data = build_dataloaders(args, out_dir=args.out_dir)
    loaders, C, V = data["loaders"], data["num_classes"], data["vocab_size"]
    log(f"classes={C} vocab={V} train_batches={len(loaders['train'])} val_batches={len(loaders['val'])}")

    model = build_model(args, C, V, modalities=data["modalities"], audio_dim=data.get("audio_dim")).to(device)
    log("model params: " + " ".join(f"{n}={count_params(e):,}" for n, e in model.encoders.items()) + f" head={count_params(model.head):,}")
    opt = build_optimizer(args, model)
    sched = build_scheduler(args, opt)
    scaler = torch.amp.GradScaler(device.type) if (args.amp and device.type == "cuda") else None

    modulator = BalancedModulator(model.M, mode=args.modulation, q_base=args.q_base, lam=args.lam, alpha=args.alpha,
                                  ge=not args.no_ge, ge_scope=args.ge_scope, z=args.z,
                                  start_epoch=args.mod_start, end_epoch=args.mod_end)
    save_json(vars(args), os.path.join(args.out_dir, "config.json"))

    history, best_acc, best_ep = [], -1.0, -1
    for ep in range(args.epochs):
        tr = train_one_epoch(model, loaders["train"], opt, device, ep, modulator, scaler=scaler, grad_clip=args.grad_clip,
                             log=log, log_interval=args.log_interval, score_mode=args.score_mode)
        if sched is not None:
            sched.step()
        rec = {"epoch": ep, **{f"train_{k}": v for k, v in tr.items()}}
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            va = evaluate(model, loaders["val"], device, score_mode=args.score_mode)
            rec.update({f"val_{k}": v for k, v in va.items()})
            if va["acc"] > best_acc:
                best_acc, best_ep = va["acc"], ep
                save_checkpoint(os.path.join(args.out_dir, "best.pt"), model, epoch=ep,
                                extra={"val_acc": best_acc, "num_classes": C, "vocab_size": V})
        history.append(rec)
        save_json(history, os.path.join(args.out_dir, "history.json"))
        names = model.modality_names
        log(f"[ep {ep:03d}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
            + " ".join(f"rho_{n}={tr[f'rho_{n}']:.2f}" for n in names) + " | "
            + " ".join(f"k_{n}={tr[f'k_{n}']:.2f} q_{n}={tr[f'q_{n}']:.2f}" for n in names)
            + (f" || val acc={rec['val_acc']:.4f} " + " ".join(f"uni_{n}={rec[f'val_uni_acc_{n}']:.4f}" for n in names)
               if "val_acc" in rec else "") + f" | {tr['time_s']:.0f}s")
    save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, epoch=args.epochs - 1, extra={"num_classes": C, "vocab_size": V})
    log(f"best val acc={best_acc:.4f} @ epoch {best_ep}")

    summary = {"best_val_acc": best_acc, "best_epoch": best_ep, "final": history[-1]}
    if args.probe:
        log("linear-probing frozen encoders (paper footnote-1 protocol)...")
        summary["probe_best_ckpt"] = None
        ck = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        pr = probe_unimodal_encoders(model, loaders["train"], loaders["val"], C, device)
        summary["probe_best_ckpt"] = pr
        log("probe (best ckpt): " + fmt(pr))
    save_json(summary, os.path.join(args.out_dir, "summary.json"))
    return summary


if __name__ == "__main__":
    main()
