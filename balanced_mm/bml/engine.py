"""
bml.engine
==========
Training / evaluation loops with the modulation hooks in the right places.

Per-iteration order (matches Algorithm 1 & 2 of the paper):
    feats  = model.encode(inputs)                                  # forward encoders once
    feats, rho = modulator.before_fusion(feats, y, model.unimodal_logits, epoch)
            # -> Eq 6/7 on undropped feats; OPM drops dominant features with q_t and updates q_{t+1}
    logits = model.fuse(feats); loss = CE(logits, y); loss.backward()
    modulator.after_backward(model.modality_parameters(), epoch)   # OGM: grad *= k^m (+ GE noise)
    optimizer.step()
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import move_to
from .modulation import BalancedModulator, unimodal_scores_from_logits, discrepancy_ratios
from .utils import AverageMeter


def train_one_epoch(model, loader, optimizer, device, epoch: int, modulator: BalancedModulator,
                    scaler=None, grad_clip: Optional[float] = None, log=print, log_interval: int = 50,
                    score_mode: str = "auto") -> Dict[str, float]:
    model.train()
    names = model.modality_names
    meters = {k: AverageMeter() for k in ["loss", "acc"] + [f"uni_acc_{n}" for n in names]
              + [f"rho_{n}" for n in names] + [f"k_{n}" for n in names] + [f"q_{n}" for n in names]}
    t0 = time.time()
    uni_fn = lambda feats: model.unimodal_logits(feats, mode=score_mode)

    for it, (inputs, y) in enumerate(loader):
        inputs, y = move_to(inputs, device), y.to(device)
        B = y.size(0)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            feats = model.encode(inputs)
            feats, rho = modulator.before_fusion(feats, y, uni_fn, epoch)
            logits = model.fuse(feats)
            loss = F.cross_entropy(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)              # grads must be real-valued before modulation
        else:
            loss.backward()

        modulator.after_backward(model.modality_parameters(), epoch)

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        # ---- logging
        with torch.no_grad():
            meters["loss"].update(loss.item(), B)
            meters["acc"].update((logits.argmax(1) == y).float().mean().item(), B)
            for m, n in enumerate(names):
                meters[f"uni_acc_{n}"].update((modulator.last_unimodal_logits[m].argmax(1) == y).float().mean().item(), B)
                meters[f"rho_{n}"].update(float(rho[m]), B)
            st = modulator.state()
            for m, n in enumerate(names):
                meters[f"k_{n}"].update(st.get(f"k_{m}", 1.0), B)
                meters[f"q_{n}"].update(st.get(f"q_{m}", 0.0), B)
        if log_interval and (it + 1) % log_interval == 0:
            log(f"  ep{epoch} it{it + 1}/{len(loader)} loss={meters['loss'].avg:.4f} acc={meters['acc'].avg:.4f} "
                + " ".join(f"rho_{n}={meters[f'rho_{n}'].avg:.2f}" for n in names)
                + " " + " ".join(f"k_{n}={meters[f'k_{n}'].avg:.2f} q_{n}={meters[f'q_{n}'].avg:.2f}" for n in names))

    out = {k: v.avg for k, v in meters.items()}
    out["time_s"] = time.time() - t0
    return out


@torch.no_grad()
def evaluate(model, loader, device, score_mode: str = "auto") -> Dict[str, float]:
    """Multimodal accuracy + cheap uni-modal proxies (block logits / zero-out) + batch-averaged rho."""
    model.eval()
    names = model.modality_names
    meters = {k: AverageMeter() for k in ["loss", "acc"] + [f"uni_acc_{n}" for n in names] + [f"rho_{n}" for n in names]}
    for inputs, y in loader:
        inputs, y = move_to(inputs, device), y.to(device)
        feats = model.encode(inputs)
        logits = model.fuse(feats)
        ulog = model.unimodal_logits(feats, mode=score_mode)
        rho = discrepancy_ratios(unimodal_scores_from_logits(ulog, y))
        B = y.size(0)
        meters["loss"].update(F.cross_entropy(logits, y).item(), B)
        meters["acc"].update((logits.argmax(1) == y).float().mean().item(), B)
        for m, n in enumerate(names):
            meters[f"uni_acc_{n}"].update((ulog[m].argmax(1) == y).float().mean().item(), B)
            meters[f"rho_{n}"].update(float(rho[m]), B)
    return {k: v.avg for k, v in meters.items()}


# --------------------------------------------------------------------------- paper-style probing
@torch.no_grad()
def extract_features(model, loader, device):
    """Frozen encoders -> per-modality feature matrices + labels."""
    model.eval()
    feats = [[] for _ in model.modality_names]
    ys = []
    for inputs, y in loader:
        inputs = move_to(inputs, device)
        fs = model.encode(inputs)
        for m, f in enumerate(fs):
            feats[m].append(f.float().cpu())
        ys.append(y)
    return [torch.cat(f) for f in feats], torch.cat(ys)


def linear_probe(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor,
                 num_classes: int, epochs: int = 200, lr: float = 1e-2, wd: float = 1e-4, device="cpu") -> float:
    """
    Paper's protocol (footnote 1): freeze the uni-modal encoder taken from the multimodal model and
    fit a NEW linear classifier on its features; the resulting accuracy = quality of that encoder.
    """
    mu, sd = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True) + 1e-6
    tx, vx = ((train_x - mu) / sd).to(device), ((val_x - mu) / sd).to(device)
    ty, vy = train_y.to(device), val_y.to(device)
    clf = nn.Linear(tx.size(1), num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(tx.size(0), device=device)
        for i in range(0, tx.size(0), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(clf(tx[idx]), ty[idx]).backward()
            opt.step()
    with torch.no_grad():
        return (clf(vx).argmax(1) == vy).float().mean().item()


def probe_unimodal_encoders(model, train_loader, val_loader, num_classes: int, device, epochs: int = 200) -> Dict[str, float]:
    trx, try_ = extract_features(model, train_loader, device)
    vax, vay = extract_features(model, val_loader, device)
    return {f"probe_acc_{n}": linear_probe(trx[m], try_, vax[m], vay, num_classes, epochs=epochs, device=device)
            for m, n in enumerate(model.modality_names)}
