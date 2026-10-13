#!/usr/bin/env python
"""Run earlier methods and OPM/OGM under one fresh, explicitly matched protocol.

Example (from balanced_mm):
    python scripts/unpack_demo_data.py --dataset all --out data/verified_demo
    python scripts/run_baseline_comparison.py --datasets food101 meld --seeds 0 1 2

Full CSVs can be used with --data-root data, after preparing data/<dataset>/*.csv.
Results are generated from measured predictions, never from paper scores.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import build_model, build_optimizer, build_scheduler, get_args  # noqa: E402
from bml.baselines import (FeatureDropout, GBlendingModel, GreedyController,  # noqa: E402
                          MMTMModel, blending_weights)
from bml.data import (SyntheticTextImageDataset, build_dataloaders, build_transforms,  # noqa: E402
                      collate_fn, move_to)
from bml.metrics import confusion_matrix, macro_f1  # noqa: E402
from bml.modulation import BalancedModulator  # noqa: E402
from bml.utils import save_json, set_seed  # noqa: E402

METHODS = ["none", "gblending", "moddrop", "dropout", "annealed", "evolutional",
           "curriculum", "opm", "ogm", "both", "mmtm", "greedy"]


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def measure(model, loader, classes):
    model.eval()
    targets, preds, loss = [], [], 0.0
    for inputs, y in loader:
        logits = model(move_to(inputs, "cpu"))
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        targets.extend(y.tolist())
        preds.extend(logits.argmax(1).tolist())
    cm = confusion_matrix(targets, preds, classes)
    return {"accuracy": sum(a == b for a, b in zip(targets, preds)) / len(targets),
            "macro_f1": macro_f1(cm), "loss": loss / len(targets),
            "n": len(targets), "confusion_matrix": cm.tolist(),
            "targets": targets, "predictions": preds}


def component_logits(model, inputs, component):
    if component == model.M:
        return model(inputs)
    name = model.modality_names[component]
    return model.aux_heads[component](model.encoders[name](inputs[name]))


@torch.no_grad()
def component_loss(model, loader, component):
    model.eval()
    loss, n = 0.0, 0
    for inputs, y in loader:
        logits = component_logits(model, move_to(inputs, "cpu"), component)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
    return loss / n


def fit_pilots(initial, loaders, train_eval, args):
    """Offline GBlending pilot estimates use training/validation only."""
    evidence, gains, overfit_changes = [], [], []
    start = time.perf_counter()
    for component in range(initial.M + 1):
        set_seed(args.seed)
        loaders["train"].generator.manual_seed(args.seed)
        model = copy.deepcopy(initial)
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)
        before_tr = component_loss(model, train_eval, component)
        before_va = component_loss(model, loaders["val"], component)
        curve = []
        for epoch in range(args.epochs):
            model.train()
            for inputs, y in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(component_logits(model, inputs, component), y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            curve.append({"epoch": epoch, "train_loss": component_loss(model, train_eval, component),
                          "val_loss": component_loss(model, loaders["val"], component)})
        after_tr, after_va = curve[-1]["train_loss"], curve[-1]["val_loss"]
        gain = before_va - after_va
        overfit = (after_va - after_tr) - (before_va - before_tr)
        gains.append(gain)
        overfit_changes.append(overfit)
        evidence.append({"component": (initial.modality_names + ["fusion"])[component],
                         "initial_train_loss": before_tr, "initial_val_loss": before_va,
                         "gain": gain, "overfit_change": overfit, "history": curve})
    weights = blending_weights(gains, overfit_changes)
    return weights, {"weights": weights, "components": evidence,
                     "fallback_uniform": all(g <= 0 for g in gains),
                     "estimation_time_s": time.perf_counter() - start,
                     "pilot_epochs_per_component": args.epochs,
                     "numerical_guard_eps": 1e-6, "nonpositive_gain_weight": 0.0}


def run_one(options, dataset, method, seed):
    out = Path(options.out) / "baseline_runs" / f"{dataset}_s{seed}_{method}"
    out.mkdir(parents=True, exist_ok=True)
    argv = ["--data", "synthetic" if dataset == "synthetic" else "csv",
            "--image_encoder", "smallcnn", "--text_dim", str(options.text_dim),
            "--text_layers", str(options.text_layers), "--img_size", "64", "--max_len", "48",
            "--epochs", str(options.epochs), "--batch_size", str(options.batch_size),
            "--lr", "0.01", "--num_workers", "0", "--device", "cpu", "--keep_last_batch",
            "--seed", str(seed), "--modulation", method if method in ("opm", "ogm", "both") else "none",
            "--dataset_name", dataset, "--out_dir", str(out)]
    if dataset == "synthetic":
        argv += ["--syn_train_n", str(options.synthetic_train_n), "--syn_val_n", str(options.synthetic_eval_n)]
    else:
        data_dir = Path(options.data_root) / dataset
        argv += ["--train_csv", str(data_dir / "train.csv"), "--val_csv", str(data_dir / "val.csv"),
                 "--test_csv", str(data_dir / "test.csv"), "--img_root", str(data_dir)]
    args = get_args(argv)
    set_seed(seed)
    data = build_dataloaders(args, out_dir=str(out))
    loaders, classes = data["loaders"], data["num_classes"]
    if dataset == "synthetic":
        # Non-training splits share an offset in the existing dataset class, so
        # use a distinct base seed for test; never reuse the validation examples.
        test = SyntheticTextImageDataset(options.synthetic_eval_n, args.num_classes, args.img_size,
                                         text_p=args.syn_text_p, img_p=args.syn_img_p,
                                         img_noise=args.syn_img_noise, seed=seed + 1_000_000, split="test")
        loaders["test"] = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_fn)
    train_eval_data = copy.copy(loaders["train"].dataset)
    if hasattr(train_eval_data, "transform"):
        train_eval_data.transform = build_transforms(args.img_size, train=False)
        train_eval_data._npz_cache = None
    train_eval = DataLoader(train_eval_data, batch_size=args.batch_size, collate_fn=collate_fn)
    loaders["train"].generator = torch.Generator().manual_seed(seed)
    # DataLoader.generator and the already constructed RandomSampler.generator
    # are separate attributes; both must share this isolated RNG.
    loaders["train"].sampler.generator = loaders["train"].generator
    base = build_model(args, classes, data["vocab_size"], data["modalities"], data.get("audio_dim"))
    initial_hash = state_hash(base.state_dict())
    model = GBlendingModel(base) if method == "gblending" else MMTMModel(base) if method in ("mmtm", "greedy") else base
    modulator = BalancedModulator(model.M, mode=args.modulation, q_base=args.q_base, lam=args.lam,
                                  alpha=args.alpha, ge=not args.no_ge, ge_scope=args.ge_scope, z=args.z)
    feature_dropout = FeatureDropout(method, args.epochs * len(loaders["train"])) if method in FeatureDropout.METHODS else None
    greedy = GreedyController() if method == "greedy" else None
    config = {**vars(args), "baseline_method": method, "cpu_threads": options.cpu_threads,
              "train_n": len(loaders["train"].dataset), "val_n": len(loaders["val"].dataset),
              "test_n": len(loaders["test"].dataset), "num_classes_actual": classes,
              "data_scope": "verified-demo" if Path(options.data_root).name == "verified_demo" and dataset != "synthetic" else "custom-or-synthetic",
              "initial_base_state_sha256": initial_hash,
              "parameters": sum(p.numel() for p in model.parameters()),
              "dropout_p": 0.5, "curriculum_rate_per_horizon": 5.0,
              "greedy_alpha": 0.01, "greedy_window": 5, "greedy_warmup_epochs": 2,
              "mmtm_ratio": 4, "synthetic_test_base_seed": seed + 1_000_000 if dataset == "synthetic" else None}
    save_json(config, str(out / "config.json"))
    started = time.perf_counter()
    pilot = None
    weights = None
    if method == "gblending":
        weights, pilot = fit_pilots(model, loaders, train_eval, args)
        save_json(pilot, str(out / "pilot.json"))
    # Pilot/extra-head allocation must not advance the main fit's RNG/order.
    set_seed(seed)
    loaders["train"].generator.manual_seed(seed)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    history, best_accuracy, best_epoch, best_state = [], -1.0, -1, None
    fit_started = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for inputs, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            feats = model.encode(inputs)
            if feature_dropout:
                feats = feature_dropout(feats)
            elif method in ("opm", "ogm", "both"):
                feats, _ = modulator.before_fusion(feats, y, model.unimodal_logits, epoch)
            logits = model.fuse(feats)
            if weights is None:
                loss = F.cross_entropy(logits, y)
            else:
                losses = [F.cross_entropy(pred, y) for pred in model.component_logits(feats)]
                loss = sum(w * value for w, value in zip(weights, losses))
            loss.backward()
            if method in ("opm", "ogm", "both"):
                modulator.after_backward(model.modality_parameters(), epoch)
            if greedy:
                greedy.after_backward(model, epoch)
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        scheduler.step()
        val = measure(model, loaders["val"], classes)
        if val["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = val["accuracy"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "train_acc": correct / n,
                        "val_loss": val["loss"], "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "time_s": time.perf_counter() - epoch_start,
                        "greedy_curation_steps": greedy.curation_steps if greedy else 0})
        save_json(history, str(out / "history.json"))
    fit_time = time.perf_counter() - fit_started
    model.load_state_dict(best_state)
    model.caring = None if isinstance(model, MMTMModel) else getattr(model, "caring", None)
    test = measure(model, loaders["test"], classes)
    save_json(test, str(out / "test_metrics.json"))
    torch.save({"model": best_state, "epoch": best_epoch}, out / "best.pt")
    result = {"dataset": dataset, "method": method, "seed": seed, "best_val_accuracy": best_accuracy,
              "best_epoch": best_epoch, "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
              "test_n": test["n"], "fit_time_s": fit_time, "total_time_s": time.perf_counter() - started,
              "pilot_time_s": pilot["estimation_time_s"] if pilot else 0.0,
              "parameters": config["parameters"], "initial_base_state_sha256": initial_hash,
              "best_state_sha256": state_hash(best_state), "evidence_path": str(out.relative_to(Path(options.out))),
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(result, str(out / "summary.json"))
    print(f"{dataset} seed={seed} {method:11s} test={test['accuracy']:.4f} F1={test['macro_f1']:.4f} time={result['total_time_s']:.1f}s", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["food101", "meld", "synthetic"], default=["food101", "meld"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-root", default="data/verified_demo")
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--text-layers", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--synthetic-train-n", type=int, default=600)
    parser.add_argument("--synthetic-eval-n", type=int, default=150)
    options = parser.parse_args(argv)
    if options.epochs < 1 or options.batch_size < 2:
        parser.error("epochs >= 1 and batch size >= 2 are required")
    torch.set_num_threads(options.cpu_threads)
    rows = []
    for dataset in options.datasets:
        for seed in options.seeds:
            for method in options.methods:
                rows.append(run_one(options, dataset, method, seed))
                save_json(rows, str(Path(options.out) / "baseline_comparison.json"))
    save_json({"python": platform.python_version(), "torch": torch.__version__,
               "platform": platform.platform(), "options": vars(options),
               "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__), ROOT / "bml" / "baselines.py", ROOT / "bml" / "modulation.py"]},
               "demo_archive_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((ROOT / "demo_data").glob("*.zip"))}},
              str(Path(options.out) / "baseline_environment.json"))
    return rows


if __name__ == "__main__":
    main()
