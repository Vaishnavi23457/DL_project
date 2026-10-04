#!/usr/bin/env python
"""
plug_in_example.py
==================
Minimal example of adding OPM / OGM-GE to an EXISTING PyTorch training loop.
Only `bml/modulation.py` is needed (copy that one file into your repo).

Your model must give you three things:
  1. per-modality features            feats = [f_1, ..., f_M]            (before fusion)
  2. a fusion head                    logits = head(concat(feats))
  3. a way to get uni-modal logits    ulogits = [W^1 f_1 + b/M, ...]       (linear head)
                                      or zero-out: head([f_1, 0, 0]), ... (any head)
Then the loop is:

    feats, rho = mod.before_fusion(feats, y, unimodal_logits_fn, epoch)   # Eq 6,7 (+ OPM Eq 8)
    loss = CE(head(cat(feats)), y); loss.backward()
    mod.after_backward([enc1.parameters(), enc2.parameters()], epoch)     # OGM Eq 11,12 + GE Eq 16
    optimizer.step()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bml.modulation import BalancedModulator, unimodal_logits_zero_out

torch.manual_seed(0)
C, D1, D2 = 5, 32, 16                      # classes, feature dims of the two modalities

# ---- 1. your existing encoders + fusion head (anything you like) ----------------------------
enc_a = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, D1))     # e.g. image / audio branch
enc_b = nn.Sequential(nn.Linear(50, 64), nn.ReLU(), nn.Linear(64, D2))      # e.g. text branch
head = nn.Linear(D1 + D2, C)                                                # linear fusion head (paper setting)
params = list(enc_a.parameters()) + list(enc_b.parameters()) + list(head.parameters())
opt = torch.optim.SGD(params, lr=1e-2, momentum=0.9)                        # SGD recommended for OGM (see README)


def unimodal_logits(feats):
    """Eq 6 helper: W^m phi^m + b/M from the linear head's weight blocks."""
    W, b = head.weight, head.bias
    return [F.linear(feats[0], W[:, :D1]) + b / 2, F.linear(feats[1], W[:, D1:]) + b / 2]
    # for an MLP head use instead:  return unimodal_logits_zero_out(lambda fs: head(torch.cat(fs, 1)), feats)


# ---- 2. one modulator object; mode = 'none' | 'opm' | 'ogm' | 'both' -----------------------
mod = BalancedModulator(num_modalities=2, mode="both", q_base=0.5, lam=0.5, alpha=0.5, ge=True)

# ---- 3. training loop (fake data) ----------------------------------------------------------
for epoch in range(3):
    for it in range(20):
        xa, xb = torch.randn(64, 100), torch.randn(64, 50)
        y = torch.randint(0, C, (64,))
        xa[torch.arange(64), y] += 3.0                                     # make modality A the "dominant" one

        opt.zero_grad()
        feats = [enc_a(xa), enc_b(xb)]                                     # (a) features
        feats, rho = mod.before_fusion(feats, y, unimodal_logits, epoch)   # (b) monitor + OPM drop
        loss = F.cross_entropy(head(torch.cat(feats, 1)), y)
        loss.backward()
        mod.after_backward([enc_a.parameters(), enc_b.parameters()], epoch)  # (c) OGM-GE on encoder grads
        opt.step()

    st = mod.state()
    print(f"epoch {epoch}: loss={loss.item():.3f} rho_A={st['rho_0']:.2f} rho_B={st['rho_1']:.2f} "
          f"q_A={st['q_0']:.2f} q_B={st['q_1']:.2f} k_A={st['k_0']:.2f} k_B={st['k_1']:.2f}")
