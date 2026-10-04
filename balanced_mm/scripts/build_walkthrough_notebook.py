#!/usr/bin/env python
"""
Build + execute `notebooks/01_walkthrough.ipynb` so the committed notebook
contains REAL outputs (visible in GitHub's Preview tab).

    python scripts/build_walkthrough_notebook.py

Runs two 4-epoch synthetic trainings (none vs OGM-GE), plots accuracy/rho
curves, and computes XAI modality attribution on a trained checkpoint.
Scratch output goes to notebooks/_walkthrough_tmp/ (git-ignored).
"""
from __future__ import annotations

import os
import sys
import time

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "notebooks", "01_walkthrough.ipynb")

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [
        MD("""# `balanced_mm` — OPM / OGM-GE walkthrough (Text + Image)

Is notebook me (sab kuch **CPU** pe, chhote settings):

1. synthetic Text + Image data pe **bina modulation** training (`none`)
2. wahi **OGM-GE** ke saath — discrepancy ratio `ρ` 1 ki taraf kaise aata hai
3. accuracy + ρ curves ka comparison plot
4. **XAI modality attribution** (Integrated Gradients + gradient×input) —
   training ke baad kaunsi modality kaam kar rahi hai

> GitHub pe isi page ka **Preview** tab me code + output dono dikhte hain.
> Khud chalane ke liye: `balanced_mm/` me `pip install -r requirements.txt`,
> phir `python scripts/build_walkthrough_notebook.py` (ya Jupyter me khol kar Run).
> Pura benchmark: `bash scripts/run_synthetic_compare.sh` + `bash scripts/run_seed_sweep.sh`,
> report: `python scripts/make_report.py` → `results/RESULTS.md`."""),

        CODE("""%matplotlib inline
import json, os, sys
import matplotlib.pyplot as plt
import torch

# project root chaho (repo ke balanced_mm/ se chalao, ya notebooks/ se bhi chalega)
ROOT = os.path.abspath(os.getcwd())
if os.path.basename(ROOT) == "notebooks":
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
TMP = os.path.join(ROOT, "notebooks", "_walkthrough_tmp")
os.makedirs(TMP, exist_ok=True)

from train import main as train_main
print("root:", ROOT)
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())"""),

        MD("""## 1–2. Train: `none` vs `ogm`

Chhota setup: 800 train / 300 val, 4 epochs, SmallCNN (64×64) + 2-layer Transformer text encoder,
SGD lr 0.01 — dono runs **ek hi seed** (0) se, taaki sirf modulation ka farq dikhe."""),

        CODE("""def run(modulation, out_name):
    out_dir = os.path.join(TMP, out_name)
    summary = train_main([
        "--data", "synthetic", "--image_encoder", "smallcnn", "--img_size", "64",
        "--epochs", "4", "--batch_size", "32",
        "--syn_train_n", "800", "--syn_val_n", "300",
        "--lr", "0.01", "--num_workers", "0", "--log_interval", "0", "--seed", "0",
        "--modulation", modulation, "--out_dir", out_dir,
    ])
    return out_dir, summary

dir_none, sum_none = run("none", "none")
dir_ogm, sum_ogm = run("ogm", "ogm")
print()
print(f"best val acc | none: {sum_none['best_val_acc']:.3f}   ogm: {sum_ogm['best_val_acc']:.3f}")"""),

        MD("""## 3. Curves: accuracy upar, ρ (bias) neeche

`ρ_image` = discrepancy ratio (Eq 7). **1 = balanced**, >1 = image dominant."""),

        CODE("""def hist(d):
    return json.load(open(os.path.join(d, "history.json")))

h0, h1 = hist(dir_none), hist(dir_ogm)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for name, h, c in [("none", h0, "#888888"), ("OGM-GE", h1, "#d62728")]:
    xs = [r["epoch"] for r in h if "val_acc" in r]
    ys = [r["val_acc"] for r in h if "val_acc" in r]
    ax[0].plot(xs, ys, marker="o", label=name, color=c, lw=2)
    ax[1].plot([r["epoch"] for r in h], [r["train_rho_image"] for r in h],
               marker="o", label=name, color=c, lw=2)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val accuracy")
ax[0].set_title("validation accuracy"); ax[0].grid(alpha=0.3); ax[0].legend()
ax[1].axhline(1.0, ls=":", c="k", label="balanced (ρ=1)")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("ρ image")
ax[1].set_title("discrepancy ratio ρ_image (Eq 7)"); ax[1].grid(alpha=0.3); ax[1].legend()
plt.tight_layout(); plt.show()"""),

        MD("""## 4. XAI side — modality attribution (`bml/xai.py`)

Trained checkpoint le kar val batch par:

- **image** → Integrated Gradients (pixels par, completeness property ke saath)
- **text** → gradient × embedding (tokens ke liye standard relevance score)

Dono ko `share` me normalise karo → "kaunsi modality ne kitna contribute kiya"."""),
        CODE("""from argparse import Namespace
from bml.data import build_dataloaders
from bml.xai import modality_attribution
from train import build_model

cfg = Namespace(**json.load(open(os.path.join(dir_none, "config.json"))))
data = build_dataloaders(cfg, out_dir=dir_none)
model = build_model(cfg, data["num_classes"], data["vocab_size"])
ck = torch.load(os.path.join(dir_none, "best.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()

inputs, y = next(iter(data["loaders"]["val"]))
att = modality_attribution(model, inputs, steps=16)
print("prediction vs truth :", att["predicted"][:8], y[:8].tolist())
print("attribution shares   :", {k: round(v, 3) for k, v in att["share"].items()})

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ks = list(att["share"])
ax.bar(ks, [att["share"][k] for k in ks], color=["#1f77b4", "#9467bd"])
ax.set_ylim(0, 1); ax.set_ylabel("share of |attribution|")
ax.set_title("Who did the work? (IG / grad×input)")
ax.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()"""),

        MD("""## Aage kya dekhna hai

- **Pura benchmark + mean±std (3 seeds):** `results/RESULTS.md` (plots `results/*.png`)
- **Compare table CLI:** `python compare.py runs/syn_none runs/syn_opm runs/syn_ogm runs/syn_both`
- **Apne CSV data pe:** `python train.py --data csv --train_csv ... --modulation both` (`plug_in_example.py` dekho)
- **Modulation ke 7 unit tests + pipeline tests:** `python -m pytest tests/` (27 tests)
- **Review prep (likely Q&A):** [`docs/REVIEW_QA.md`](../docs/REVIEW_QA.md)

`--modulation {none|opm|ogm|both}` — bas yahi flag lagta hai; module `bml/modulation.py`
standalone hai (paper: Wei et al., TPAMI 2024)."""),
    ]
    return nb


def execute(nb: nbf.NotebookNode) -> None:
    from nbclient import NotebookClient

    os.chdir(ROOT)  # kernel inherits this cwd
    client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=False)
    client.execute()


def main() -> None:
    t0 = time.time()
    nb = build()
    print("executing notebook ...")
    execute(nb)
    os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
    nbf.write(nb, NB_PATH)
    size = os.path.getsize(NB_PATH)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_with_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.outputs)
    print(f"wrote {NB_PATH}  ({size/1e6:.2f} MB, {n_with_out}/{n_code} code cells with output, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
