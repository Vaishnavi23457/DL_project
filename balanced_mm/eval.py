#!/usr/bin/env python
"""
eval.py - evaluate a trained run on a CSV split (or the synthetic val split) and optionally run
the paper-style linear probing of each frozen encoder.

    python eval.py --run_dir runs/my_both --test_csv data/test.csv --img_root data/images --probe
    python eval.py --run_dir runs/syn_ogm --probe                      # synthetic run
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bml.data import build_dataloaders, move_to                # noqa: E402
from bml.engine import evaluate, probe_unimodal_encoders   # noqa: E402
from bml.metrics import classification_report, expected_calibration_error, format_confusion_md  # noqa: E402
from bml.utils import fmt, get_device, load_json, save_json  # noqa: E402
from train import build_model                              # noqa: E402


@torch.no_grad()
def collect_preds(model, loader, device):
    """All (logits, labels) of a loader -> numpy, for confusion / per-class / ECE."""
    model.eval()
    ys, outs = [], []
    for inputs, y in loader:
        outs.append(model(move_to(inputs, device)).cpu())
        ys.append(y.cpu())
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--test_csv", default=None, help="CSV to evaluate on (default: the run's val split)")
    ap.add_argument("--img_root", default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    cfg = load_json(os.path.join(a.run_dir, "config.json"))
    args = SimpleNamespace(**cfg)
    args.batch_size = a.batch_size
    args.num_workers = 0
    if a.test_csv:
        args.val_csv = a.test_csv
    if a.img_root is not None:
        args.img_root = a.img_root
    if args.data == "csv":  # reuse the saved vocab / label map
        args.vocab_path = os.path.join(a.run_dir, "vocab.json")
        args.label_map_path = os.path.join(a.run_dir, "label_map.json")

    device = get_device(a.device)
    data = build_dataloaders(args, out_dir=None)
    ck = torch.load(os.path.join(a.run_dir, a.ckpt), map_location=device, weights_only=True)
    model = build_model(args, ck.get("num_classes", data["num_classes"]), ck.get("vocab_size", data["vocab_size"]),
                        modalities=data["modalities"], audio_dim=data.get("audio_dim")).to(device)
    model.load_state_dict(ck["model"])

    res = evaluate(model, data["loaders"]["val"], device, score_mode=args.score_mode)
    print("eval:", fmt(res))

    # classification-quality metrics (bml.metrics): confusion, per-class acc, macro-F1, ECE
    logits_np, y_np = collect_preds(model, data["loaders"]["val"], device)
    class_names = None
    lm_path = os.path.join(a.run_dir, "label_map.json")
    if os.path.isfile(lm_path):
        lm = load_json(lm_path)
        class_names = [k for k, _ in sorted(lm.items(), key=lambda kv: kv[1])] if isinstance(lm, dict) else None
    rep = classification_report(y_np, logits_np.argmax(1), class_names=class_names)
    ece = expected_calibration_error(y_np, logits_np)
    print("confusion (rows = true):\n" + format_confusion_md(
        np.asarray(rep["confusion_matrix"]), class_names=class_names))
    print(f"macro_f1={rep['macro_f1']:.4f}  ece={ece:.4f}  per_class={rep['per_class_accuracy']}")
    res.update({"macro_f1": rep["macro_f1"], "ece": ece, "per_class_accuracy": rep["per_class_accuracy"],
                "confusion_matrix": rep["confusion_matrix"], "support": rep["support"]})

    if a.probe:
        pr = probe_unimodal_encoders(model, data["loaders"]["train"], data["loaders"]["val"], model.num_classes, device, epochs=a.probe_epochs)
        print("probe:", fmt(pr))
        res.update(pr)
    save_json(res, os.path.join(a.run_dir, "eval_" + (os.path.basename(a.test_csv) if a.test_csv else "val") + ".json"))
    return res


if __name__ == "__main__":
    main()
