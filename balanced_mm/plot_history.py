#!/usr/bin/env python
"""
plot_history.py - paper-style curves (Fig. 1 / Fig. 5 / Fig. 7) from one or more runs.

    python plot_history.py runs/syn_none runs/syn_opm runs/syn_ogm runs/syn_both --out runs/curves.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="curves.png")
    a = ap.parse_args()

    hists = {}
    for r in a.runs:
        with open(os.path.join(r, "history.json")) as f:
            hists[os.path.basename(r.rstrip("/"))] = json.load(f)
    any_h = next(iter(hists.values()))
    names = sorted({k[len("val_uni_acc_"):] for k in any_h[-1] if k.startswith("val_uni_acc_")})

    ncols = 3 + len(names)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.6))
    for run, h in hists.items():
        ev = [x for x in h if "val_acc" in x]
        ep = [x["epoch"] for x in ev]
        axes[0].plot(ep, [x["val_acc"] for x in ev], label=run)
        axes[1].plot([x["epoch"] for x in h], [x["train_loss"] for x in h], label=run)
        if names:
            n0 = names[0]
            axes[2].plot([x["epoch"] for x in h], [x[f"train_rho_{n0}"] for x in h], label=run)
        for i, n in enumerate(names):
            axes[3 + i].plot(ep, [x[f"val_uni_acc_{n}"] for x in ev], label=run)
    axes[0].set_title("val accuracy (multimodal)"); axes[1].set_title("train loss")
    axes[2].set_title(f"discrepancy ratio rho_{names[0] if names else ''} (train)"); axes[2].axhline(1.0, ls="--", c="gray")
    for i, n in enumerate(names):
        axes[3 + i].set_title(f"val uni-modal acc: {n}")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
