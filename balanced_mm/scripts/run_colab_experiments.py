#!/usr/bin/env python
"""Train comparisons; save held-out metrics, curves, and confusion matrices."""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from train import main as train_model
from eval import main as evaluate_model


def plot_run(directory, history, test, names):
    epochs = [r["epoch"] + 1 for r in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for prefix, label in [("train", "Train"), ("val", "Validation")]:
        axes[0].plot(epochs, [r[f"{prefix}_acc"] for r in history], label=label)
        axes[1].plot(epochs, [r[f"{prefix}_loss"] for r in history], label=label)
    for key in history[0]:
        if key.startswith("train_rho_"):
            axes[2].plot(
                epochs, [r[key] for r in history], label=key.removeprefix("train_rho_")
            )
    axes[2].axhline(1, color="gray", linestyle="--", linewidth=1)
    for axis, title in zip(
        axes, ["Accuracy", "Cross-entropy loss", "Training modality ratio"]
    ):
        axis.set(title=title, xlabel="Epoch")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(directory / "training_curves.png", dpi=150)
    plt.close(figure)
    matrix = np.asarray(test["confusion_matrix"])
    normalised = np.divide(
        matrix,
        matrix.sum(1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(1, keepdims=True) > 0,
    )
    figure, axis = plt.subplots(figsize=(max(6, len(names) * 0.25),) * 2)
    image = axis.imshow(normalised, vmin=0, vmax=1, cmap="Blues")
    axis.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Held-out test confusion matrix (row-normalised)",
    )
    size = 9 if len(names) <= 12 else 4
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right", fontsize=size)
    plt.setp(axis.get_yticklabels(), fontsize=size)
    if len(names) <= 12:
        for row in range(len(names)):
            for col in range(len(names)):
                axis.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalised[row, col] > 0.5 else "black",
                )
    figure.colorbar(image, ax=axis, label="Recall share")
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    with (directory / "confusion_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["true / predicted", *names])
        writer.writerows([label, *values] for label, values in zip(names, matrix))


def write_report(output, records, args):
    lines = [
        "# Reproducible training demo",
        "",
        f"Executed at {datetime.now(timezone.utc).isoformat()}",
        "",
        "New models on the selected data; separate from the older three-seed benchmark in results/RESULTS.md.",
        "",
        f"Epochs: {args.epochs}; text dimension/layers: {args.text_dim}/{args.text_layers}; image encoder: {args.image_encoder}.",
        "",
        "Best checkpoint selected using validation accuracy; test evaluated after training.",
        "",
        "| Dataset | Method | Seed | Best val accuracy | Test accuracy | Test macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['dataset']} | {r['modulation']} | {r['seed']} | {r['best_val_acc']:.4f} | {r['test_acc']:.4f} | {r['test_macro_f1']:.4f} |"
        )
    lines += ["", "## Data and scope", ""]
    for dataset in sorted({r["dataset"] for r in records}):
        manifest = json.loads((Path(args.data) / dataset / "manifest.json").read_text())
        counts = ", ".join(
            f"{name}={info['count']}" for name, info in manifest["splits"].items()
        )
        lines.append(f"- **{dataset}**: {counts}. {manifest['split_protocol']}")
    lines += [
        "",
        "Food-101 text is generated from labels: a controlled shortcut/bias experiment, not a natural image-and-caption benchmark.",
        "MELD uses real text and official 300-dimensional audio features. Positive per-class limits create a balanced demo sample; zero retains the full official distribution.",
        "Small samples, one seed, and short training verify the pipeline, not a publishable improvement.",
        "",
        "## Saved evidence",
        "",
    ]
    for r in records:
        name = r["run"]
        lines += [
            f"### {name}",
            "",
            f"![Training curves]({name}/training_curves.png)",
            "",
            f"![Test confusion matrix]({name}/confusion_matrix.png)",
            "",
            f"[Training log]({name}/train.log) · [History]({name}/history.json) · [Test metrics]({name}/eval_test.csv.json)",
            "",
        ]
    (output / "RESULTS.md").write_text("\n".join(lines))
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "runs": records,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "cuda_available": torch.cuda.is_available(),
                },
            },
            indent=2,
        )
    )
    figure, axis = plt.subplots(figsize=(max(7, len(records)), 4))
    positions = np.arange(len(records))
    axis.bar(
        positions - 0.18, [r["test_acc"] for r in records], 0.36, label="Test accuracy"
    )
    axis.bar(
        positions + 0.18,
        [r["test_macro_f1"] for r in records],
        0.36,
        label="Test macro-F1",
    )
    axis.set(
        xticks=positions,
        xticklabels=[
            f"{r['dataset']}\n{r['modulation']} s{r['seed']}" for r in records
        ],
        ylim=(0, 1),
        title="New runs: held-out test metrics",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [
        ("dataset", "all"),
        ("data", "data/demo"),
        ("out", "demo_runs"),
        ("methods", "none,opm,ogm,both"),
        ("seeds", "0"),
        ("device", "auto"),
        ("image-encoder", "smallcnn"),
    ]:
        parser.add_argument("--" + name, default=default)
    for name, default in [
        ("epochs", 5),
        ("text-dim", 64),
        ("text-layers", 1),
        ("batch-size", 16),
        ("cpu-threads", 2),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--pretrained-image", action="store_true")
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods or set(methods) - {"none", "opm", "ogm", "both"}:
        parser.error("Unknown modulation method")
    if args.dataset not in {"food101", "meld", "all"}:
        parser.error("Unknown dataset")
    if args.image_encoder not in {"smallcnn", "resnet18"}:
        parser.error("Unknown image encoder")
    if min(args.epochs, args.batch_size, args.cpu_threads) < 1:
        parser.error("Epochs, batch size, and CPU threads must be positive")
    seeds = [int(s) for s in args.seeds.split(",")]
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in ["food101", "meld"]:
        if args.dataset not in (dataset, "all"):
            continue
        root = Path(args.data) / dataset
        manifest = json.loads((root / "manifest.json").read_text())
        for split in ["train", "val", "test"]:
            if manifest["splits"][split]["count"] < (
                args.batch_size if split == "train" else 1
            ):
                raise ValueError(
                    "Training needs at least one full batch; validation/test must be nonempty"
                )
        for seed in seeds:
            for method in methods:
                run = output / f"{dataset}_s{seed}_{method}"
                run.mkdir(exist_ok=True)
                (run / "train.log").unlink(missing_ok=True)
                print(f"\n=== Training {run.name} ===", flush=True)
                train_args = [
                    "--data",
                    "csv",
                    "--train_csv",
                    str(root / "train.csv"),
                    "--val_csv",
                    str(root / "val.csv"),
                    "--img_root",
                    str(root),
                    "--dataset_name",
                    dataset + "_demo",
                    "--modulation",
                    method,
                    "--epochs",
                    str(args.epochs),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--out_dir",
                    str(run),
                    "--batch_size",
                    str(args.batch_size),
                    "--num_workers",
                    "0",
                    "--log_interval",
                    "0",
                    "--lr",
                    str(args.lr),
                    "--image_encoder",
                    args.image_encoder,
                    "--img_size",
                    "64",
                    "--text_dim",
                    str(args.text_dim),
                    "--text_layers",
                    str(args.text_layers),
                    "--max_len",
                    "48",
                    "--keep_last_batch",
                ]
                if args.pretrained_image:
                    train_args.append("--pretrained_image")
                summary = train_model(train_args)
                test = evaluate_model(
                    [
                        "--run_dir",
                        str(run),
                        "--test_csv",
                        str(root / "test.csv"),
                        "--img_root",
                        str(root),
                        "--device",
                        args.device,
                        "--batch_size",
                        str(args.batch_size),
                    ]
                )
                labels = json.loads((run / "label_map.json").read_text())
                names = [
                    label for label, index in sorted(labels.items(), key=lambda p: p[1])
                ]
                plot_run(
                    run, json.loads((run / "history.json").read_text()), test, names
                )
                records.append(
                    {
                        "dataset": dataset,
                        "modulation": method,
                        "seed": seed,
                        "run": run.name,
                        "best_val_acc": summary["best_val_acc"],
                        "test_acc": test["acc"],
                        "test_macro_f1": test["macro_f1"],
                    }
                )
                write_report(output, records, args)
    print(f"\nFinished: {output / 'RESULTS.md'}", flush=True)
    return records


if __name__ == "__main__":
    main()
