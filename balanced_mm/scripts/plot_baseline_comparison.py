#!/usr/bin/env python
"""Classification reports, training graphs and confusion matrices from saved tests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_baseline_report import LABELS, ORDER, validate

SHORT = {"none": "No modulation", "gblending": "GBlending", "moddrop": "ModDrop",
         "dropout": "Dropout", "annealed": "Annealed dropout", "evolutional": "Evolutional dropout",
         "curriculum": "Curriculum dropout", "opm": "OPM", "ogm": "OGM-GE",
         "both": "OPM + OGM-GE", "mmtm": "MMTM control", "greedy": "Greedy + MMTM"}
COLORS = {method: ("#16847A" if method in ("opm", "ogm", "both") else "#54739C") for method in ORDER}


def extended_metrics(cm, names):
    cm = np.asarray(cm, dtype=float)
    true_pos = cm.diagonal()
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(true_pos, predicted, out=np.zeros_like(true_pos), where=predicted > 0)
    recall = np.divide(true_pos, support, out=np.zeros_like(true_pos), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(true_pos), where=precision + recall > 0)
    supported = support > 0
    classes = [{"class": name, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
               for name, p, r, f, n in zip(names, precision, recall, f1, support)]
    n = int(support.sum())
    return {"accuracy": float(true_pos.sum() / n), "macro_precision": float(precision[supported].mean()),
            "macro_recall": float(recall[supported].mean()), "macro_f1": float(f1[supported].mean()),
            "weighted_f1": float((f1 * support).sum() / n), "micro_f1": float(true_pos.sum() / n),
            "balanced_accuracy": float(recall[supported].mean()), "test_n": n,
            "zero_division": 0, "classes": classes}


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None, "Creator": "DL_project measured baseline comparison"})
    plt.close(fig)
    # Matplotlib adds spaces to multiline SVG paths; keep generated Git diffs clean.
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def confusion_figure(cm, names, title, path):
    cm = np.asarray(cm)
    support = cm.sum(1, keepdims=True)
    normalized = np.divide(cm, support, out=np.zeros_like(cm, dtype=float), where=support > 0)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    for ax, values, label, vmax in zip(axes, [cm, normalized], ["Counts", "Row-normalized (%)"], [max(1, cm.max()), 1]):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(len(names)):
            for j in range(len(names)):
                text = str(int(values[i, j])) if label == "Counts" else f"{100*values[i,j]:.0f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > 0.55 * vmax else "#18283B")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(label)
    fig.suptitle(title, fontsize=13, weight="bold")
    savefig(fig, path)


def main(root):
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "dl-baseline-comparison-v1",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = json.loads((root / "baseline_comparison.json").read_text())
    validate(rows, root)
    plots = root / "baseline_plots"
    plots.mkdir(exist_ok=True)
    overall_rows, class_rows = [], []
    for row in rows:
        out = root / row["evidence_path"]
        test = json.loads((out / "test_metrics.json").read_text())
        label_path = out / "label_map.json"
        mapping = json.loads(label_path.read_text()) if label_path.exists() else {str(i): i for i in range(len(test["confusion_matrix"]))}
        names = [name for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
        report = extended_metrics(test["confusion_matrix"], names)
        if abs(report["macro_f1"] - row["test_macro_f1"]) > 1e-12:
            raise ValueError("Saved F1 does not match confusion matrix")
        (out / "classification_report.json").write_text(json.dumps(report, indent=2) + "\n")
        overall_rows.append({"dataset": row["dataset"], "method": row["method"], "seed": row["seed"],
                             **{k: v for k, v in report.items() if k not in ("classes", "zero_division")}})
        class_rows += [{"dataset": row["dataset"], "method": row["method"], "seed": row["seed"], **c} for c in report["classes"]]
        lines = [f"# {row['dataset']} — {LABELS[row['method']]} — seed {row['seed']}", "",
                 f"Held-out test n={report['test_n']}; best validation checkpoint at epoch {row['best_epoch']} (0-based).", "",
                 "| Metric | Score (%) |", "|---|---:|"]
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "micro_f1", "balanced_accuracy"):
            lines.append(f"| {metric.replace('_', ' ')} | {report[metric] * 100:.2f} |")
        lines += ["", "| Class | Precision (%) | Recall (%) | F1 (%) | Support |", "|---|---:|---:|---:|---:|"]
        for c in report["classes"]:
            lines.append(f"| {c['class']} | {c['precision']*100:.2f} | {c['recall']*100:.2f} | {c['f1']*100:.2f} | {c['support']} |")
        lines += ["", "Undefined precision/recall uses zero. For single-label multiclass classification, "
                  "micro-F1 equals accuracy and balanced accuracy equals macro recall.", "",
                  "![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)", "",
                  "[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)", ""]
        (out / "classification_report.md").write_text("\n".join(lines))
        confusion_figure(test["confusion_matrix"], [n.replace("_", " ") for n in names],
                         f"{row['dataset']} | {SHORT[row['method']]} | seed {row['seed']} | held-out test", out / "confusion_matrix.svg")
    for filename, records in [("baseline_classification_metrics.csv", overall_rows), ("baseline_classwise_metrics.csv", class_rows)]:
        with (root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    for dataset in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True, constrained_layout=True)
        for ax, key, title in zip(axes, ["test_accuracy", "test_macro_f1"], ["Held-out test accuracy", "Held-out test macro-F1"]):
            values = [[100*r[key] for r in ds_rows if r["method"] == method] for method in ORDER]
            means, stds = [np.mean(v) for v in values], [np.std(v, ddof=1) for v in values]
            ax.barh(range(len(ORDER)), means, xerr=stds, color=[COLORS[m] for m in ORDER], capsize=3, alpha=0.9)
            ax.set_yticks(range(len(ORDER)), [SHORT[m] for m in ORDER])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Mean ± sample SD over 3 seeds (%)")
            ax.set_title(title, weight="bold")
            ax.grid(axis="x", alpha=0.18)
            for i, mean in enumerate(means):
                ax.text(mean + stds[i] + 1.5, i, f"{mean:.1f}", va="center", fontsize=9)
        axes[0].invert_yaxis()
        fig.suptitle(f"{dataset}: earlier methods and project OPM/OGM — fresh short protocol", weight="bold")
        savefig(fig, plots / f"metrics_{dataset}.svg")
        for kind in ("training", "validation"):
            fig, axes = plt.subplots(4, 3, figsize=(13.6, 12.5), constrained_layout=True)
            for ax, method in zip(axes.flat, ORDER):
                history = [json.loads((root/r["evidence_path"]/"history.json").read_text()) for r in ds_rows if r["method"] == method]
                keys = [("train_loss", "Training"), ("val_loss", "Validation")] if kind == "training" else [("val_acc", "Validation accuracy")]
                for key, label in keys:
                    values = np.array([[ep[key] for ep in h] for h in history])
                    if key == "val_acc":
                        values *= 100
                    epochs = np.arange(1, values.shape[1] + 1)
                    mean, sd = values.mean(0), values.std(0, ddof=1)
                    ax.plot(epochs, mean, marker="o", markersize=3, label=label)
                    ax.fill_between(epochs, mean - sd, mean + sd, alpha=0.13)
                ax.set_title(SHORT[method], fontsize=11, weight="bold")
                ax.set_xlabel("Main-fit epoch")
                ax.set_ylabel("Loss" if kind == "training" else "Accuracy (%)")
                ax.grid(alpha=0.18)
                ax.legend(fontsize=8)
            fig.suptitle(f"{dataset}: {kind} curves — mean ± sample SD, seeds 0/1/2", weight="bold")
            savefig(fig, plots / f"{kind}_{dataset}.svg")
        aggregate_cms, names = [], None
        for method in ORDER:
            runs = [r for r in ds_rows if r["method"] == method]
            matrices = [json.loads((root/r["evidence_path"]/"test_metrics.json").read_text())["confusion_matrix"] for r in runs]
            aggregate_cms.append(np.asarray(matrices).sum(0))
            if names is None:
                report = json.loads((root/runs[0]["evidence_path"]/"classification_report.json").read_text())
                names = [c["class"].replace("_", " ") for c in report["classes"]]
        fig, axes = plt.subplots(4, 3, figsize=(17.5, 20.5), constrained_layout=True)
        for ax, method, cm in zip(axes.flat, ORDER, aggregate_cms):
            support = cm.sum(1, keepdims=True)
            normalized = np.divide(cm, support, out=np.zeros_like(cm, dtype=float), where=support > 0)
            ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7)
            ax.set_yticks(range(len(names)), names, fontsize=7)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{100*normalized[i,j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if normalized[i,j] > 0.55 else "#18283B")
            ax.set_title(SHORT[method], fontsize=11, weight="bold")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.suptitle(f"{dataset}: pooled three-seed test predictions, row-normalized percentages\n"
                     "Food-101/MELD reuse the same held-out cases across seeds; pooled counts are not unique sample counts.", fontsize=12, weight="bold")
        savefig(fig, plots / f"confusions_{dataset}.svg")
    print(f"Rendered {len(rows)} individual confusion matrices, "
          f"{4 * len({r['dataset'] for r in rows})} summary graphs, and classification metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    main(parser.parse_args().results)
