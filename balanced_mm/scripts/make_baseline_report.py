#!/usr/bin/env python
"""Generate comparison tables and paired improvements from measured run evidence."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

LABELS = {"none": "Concatenation / no modulation", "gblending": "GBlending (offline adaptation)",
          "moddrop": "ModDrop (feature adaptation)", "dropout": "Dropout",
          "annealed": "Annealed dropout", "evolutional": "Evolutional dropout",
          "curriculum": "Curriculum dropout", "opm": "OPM (project)", "ogm": "OGM-GE (project)",
          "both": "OPM + OGM-GE (project)", "mmtm": "MMTM feature-level control",
          "greedy": "Greedy + MMTM (project adaptation)"}
ORDER = list(LABELS)
OLDER = ["gblending", "greedy", "moddrop", "dropout", "annealed", "evolutional", "curriculum"]


def stats(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def validate(rows, root):
    """Refuse incomplete/mismatched comparisons and reconcile saved predictions."""
    seen = set()
    initial = {}
    protocols = {}
    test_targets = {}
    for row in rows:
        key = row["dataset"], row["seed"], row["method"]
        if key in seen:
            raise ValueError(f"duplicate run: {key}")
        seen.add(key)
        run_dir = root / row["evidence_path"]
        config = json.loads((run_dir / "config.json").read_text())
        test = json.loads((run_dir / "test_metrics.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        if [ep["epoch"] for ep in history] != list(range(config["epochs"])):
            raise ValueError(f"incomplete training history: {key}")
        selected = max(history, key=lambda ep: ep["val_acc"])
        if selected["epoch"] != row["best_epoch"] or selected["val_acc"] != row["best_val_accuracy"]:
            raise ValueError(f"validation/checkpoint-selection mismatch: {key}")
        truth, pred = test["targets"], test["predictions"]
        accuracy = sum(a == b for a, b in zip(truth, pred)) / len(truth)
        if len(truth) != len(pred) or len(truth) != row["test_n"] or abs(accuracy - row["test_accuracy"]) > 1e-12:
            raise ValueError(f"prediction/metric mismatch: {key}")
        comparable = tuple(config[k] for k in ("epochs", "batch_size", "lr", "optimizer", "momentum",
                           "weight_decay", "sched", "image_encoder", "text_encoder", "audio_encoder",
                           "text_dim", "text_layers", "img_size", "train_n", "val_n", "test_n",
                           "train_csv", "val_csv", "test_csv", "num_classes_actual"))
        ds_key = row["dataset"], row["seed"]
        if ds_key in initial and initial[ds_key] != row["initial_base_state_sha256"]:
            raise ValueError(f"base initialization mismatch: {key}")
        if ds_key in protocols and protocols[ds_key] != comparable:
            raise ValueError(f"protocol mismatch: {key}")
        if ds_key in test_targets and test_targets[ds_key] != truth:
            raise ValueError(f"test examples/label ordering mismatch: {key}")
        initial[ds_key], protocols[ds_key] = row["initial_base_state_sha256"], comparable
        test_targets[ds_key] = truth
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        seeds = sorted({r["seed"] for r in rows if r["dataset"] == ds})
        if seeds != [0, 1, 2]:
            raise ValueError(f"published report requires seeds 0,1,2: {ds} has {seeds}")
        expected = {(ds, seed, method) for seed in seeds for method in ORDER}
        missing = expected - seen
        if missing:
            raise ValueError(f"incomplete comparison: {sorted(missing)}")


def make_report(root):
    rows = json.loads((root / "baseline_comparison.json").read_text())
    validate(rows, root)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"], row["method"]].append(row)
    datasets = sorted({row["dataset"] for row in rows})
    synthetic_scope = "Synthetic uses a new 600/150/150 train/validation/test protocol. " if "synthetic" in datasets else ""
    lines = ["# Earlier-method comparison on the project models", "",
             "Fresh measured runs for the earlier learning strategies in **Tables V and VI** of "
             "[On-the-fly Modulation for Balanced Multimodal Learning](https://arxiv.org/html/2410.11582v1). "
             "These tables use the project's encoders, with the explicitly documented adaptations below.", "",
             "**Scope:** Food-101 and MELD use the verified small archives committed in `demo_data/`. "
             "The full CSVs behind the older [benchmark report](RESULTS.md) are absent from the checkout. "
             + synthetic_scope + "These fresh five-epoch "
             "results must not be merged numerically with the older 12/15-epoch benchmark or the paper's datasets.", "",
             "All percentages are **mean ± sample standard deviation over seeds 0, 1, 2**. "
             "Differences are absolute percentage points (pp). Checkpoints are selected using validation "
             "accuracy; the held-out test set is evaluated once after selection. No test metrics are used "
             "for training, pilot weights, schedules, or checkpoint selection.", "",
             "[Aggregate CSV](baseline_comparison.csv) · [Per-run JSON](baseline_comparison.json) · "
             "[Runtime and data hashes](baseline_environment.json) · "
             "[All classification metrics](baseline_classification_metrics.csv) · "
             "[All class-wise precision/recall/F1](baseline_classwise_metrics.csv)", ""]
    csv_rows, deltas = [], []
    for ds in datasets:
        first = grouped[ds, "none"][0]
        config = json.loads((root / first["evidence_path"] / "config.json").read_text())
        baseline_mean = statistics.mean(r["test_accuracy"] for r in grouped[ds, "none"])
        lines += [f"## {ds}", "",
                  f"Train/validation/test: **{config['train_n']}/{config['val_n']}/{config['test_n']}**; "
                  f"classes: {config['num_classes_actual']}; epochs: {config['epochs']}; batch size: {config['batch_size']}.", "",
                  "| Method | Test accuracy (%) | Test macro-F1 (%) | Δ vs no modulation (pp) | Measured run time (s) |",
                  "|---|---:|---:|---:|---:|"]
        for method in ORDER:
            runs = grouped[ds, method]
            acc, acc_sd = stats([r["test_accuracy"] * 100 for r in runs])
            f1, f1_sd = stats([r["test_macro_f1"] * 100 for r in runs])
            elapsed = statistics.mean(r["total_time_s"] for r in runs)
            delta = acc - baseline_mean * 100
            lines.append(f"| {LABELS[method]} | {acc:.2f} ± {acc_sd:.2f} | {f1:.2f} ± {f1_sd:.2f} | {delta:+.2f} | {elapsed:.1f} |")
            csv_rows.append({"dataset": ds, "method": method, "seeds": "0;1;2", "train_n": config["train_n"],
                             "val_n": config["val_n"], "test_n": config["test_n"], "epochs": config["epochs"],
                             "test_accuracy_mean_pct": acc, "test_accuracy_sample_sd_pct": acc_sd,
                             "test_macro_f1_mean_pct": f1, "test_macro_f1_sample_sd_pct": f1_sd,
                             "delta_vs_none_pp": delta, "total_training_time_mean_s": elapsed,
                             "parameters": runs[0]["parameters"]})
        lines += ["", "Greedy and its MMTM-only control have an additional cross-modal module. "
                  "Their comparison isolates the Greedy training policy; comparisons with plain "
                  "concatenation also include the architecture change. GBlending has auxiliary heads "
                  "and three pilot fits; their cost is included in the time column.", "",
                  "### Project methods minus each older method", "",
                  "| Earlier method | OPM Δ (pp) | OGM-GE Δ (pp) | OPM + OGM-GE Δ (pp) |",
                  "|---|---:|---:|---:|"]
        for older in OLDER:
            older_by_seed = {r["seed"]: r for r in grouped[ds, older]}
            values = []
            for ours in ("opm", "ogm", "both"):
                paired = [(r["test_accuracy"] - older_by_seed[r["seed"]]["test_accuracy"]) * 100
                          for r in grouped[ds, ours]]
                mean, sd = stats(paired)
                values.append(mean)
                deltas.append({"dataset": ds, "older_method": older, "project_method": ours,
                               "paired_accuracy_delta_mean_pp": mean, "paired_accuracy_delta_sample_sd_pp": sd})
            lines.append(f"| {LABELS[older]} | {values[0]:+.2f} | {values[1]:+.2f} | {values[2]:+.2f} |")
        greedy_delta = (statistics.mean(r["test_accuracy"] for r in grouped[ds, "greedy"])
                        - statistics.mean(r["test_accuracy"] for r in grouped[ds, "mmtm"])) * 100
        lines += ["", f"Greedy minus MMTM-only control: **{greedy_delta:+.2f} pp** test accuracy.", ""]
        lines += ["### Graphs and confusion matrices", "",
                  f"![{ds}: test accuracy and macro-F1](baseline_plots/metrics_{ds}.svg)", "",
                  f"![{ds}: training and validation loss](baseline_plots/training_{ds}.svg)", "",
                  f"![{ds}: validation accuracy by epoch](baseline_plots/validation_{ds}.svg)", "",
                  f"![{ds}: three-seed pooled row-normalized confusion matrices](baseline_plots/confusions_{ds}.svg)", "",
                  "The evidence links below include an individual **count and normalized confusion "
                  "matrix for every run**, plus accuracy, macro/micro/weighted F1, precision, recall, "
                  "balanced accuracy and per-class support. Pooled Food-101/MELD matrices combine "
                  "three models' predictions on the same test cases, rather than three independent "
                  "test samples. Training loss for GBlending is its weighted auxiliary/fusion objective, "
                  "while validation loss uses only its fusion classifier.", ""]
    lines += ["## Training protocol and implementation", "",
              "- The common architecture is the existing SmallCNN (128-dimensional image features) "
              "or MELD audio MLP (300-dimensional input, 128-dimensional output), plus the existing "
              "one-layer, 64-dimensional text Transformer and linear concatenation classifier.",
              "- All main fits use five epochs, batch size 16, SGD learning rate 0.01, momentum 0.9, "
              "weight decay 0.0001, cosine scheduling, two CPU threads, and all training rows. "
              "Encoders and fusion head start from exactly the same hashed initialization per dataset/seed. "
              "The loader has a separately seeded shuffle generator. Stochastic masks and augmentations "
              "can still differ as methods consume random numbers differently.",
              "- Run times are CPU wall-clock observations including main fitting, validation, "
              "test evaluation, evidence/checkpoint writing and GBlending pilot work, after initial "
              "data/model construction. Some dataset sweeps ran concurrently. Treat these as "
              "execution evidence rather than a controlled speed benchmark.",
              "- Food-101 has ten classes and label-derived prompt text: this measures a controlled "
              "shortcut/bias setting, not natural image-caption recognition. MELD has seven classes, "
              "real text, and official audio embeddings; official split boundaries are preserved. "
              "Both committed demo samples are balanced and small.",
              "- Synthetic has six classes, informative-modality probabilities 0.7, image size 64 "
              "and noise standard deviation 1.0. The test base seed is `seed + 1_000_000` so it cannot "
              "duplicate the validation stream in the existing synthetic dataset class.",
              "- OPM, OGM-GE and their combination use the existing project implementation unchanged "
              "(`q_base=lam=alpha=0.5`, Gaussian enhancement enabled). The paper labels its gradient "
              "method OGM; these project rows explicitly include GE.", "",
              "| Earlier strategy | Exact project adaptation |",
              "|---|---|",
              "| GBlending | Offline independent unimodal/fusion pilot fits, each five epochs, from "
              "the initial weights. `G = val_loss_before - val_loss_after`; "
              "`O = (val-train)_after - (val-train)_before`; weights proportional to `max(G,0)/max(O²,1e-12)`. "
              "Uniform fallback if no pilot improves validation loss. Train-loss probes use evaluation "
              "transforms. Restore the initial model for the main weighted auxiliary/fusion CE fit. "
              "Positive-gain clipping and guards are explicit numerical adaptations; all pilot evidence is saved. |",
              "| Greedy | One MMTM module on the final pooled vectors, bottleneck `2*(d1+d2)/4`, "
              "sigmoid gates. Accumulate squared-gradient/squared-parameter norms separately for "
              "encoder and shared-plus-modality gate parameters. Trigger curation when the difference "
              "of log10 bypass/main ratios exceeds 0.01; use five curation steps after two warmup epochs. "
              "Replace the selected modality's gate with its detached training-only running mean. "
              "Each modality has its own correct mean; evaluation never updates it. This final-feature "
              "module and project optimizer differ from the original multi-layer video architecture. |",
              "| ModDrop | Independent per-sample, whole-feature-vector masks, fixed drop probability "
              "0.5 and inverted scaling. Both modalities can be dropped. This is the modality-dropout "
              "component, without the original gesture model's pretraining/fusion curriculum. |",
              "| Dropout | Independent feature-unit Bernoulli dropout, probability 0.5, before fusion. |",
              "| Annealed dropout | Feature-unit dropout; probability decreases linearly from 0.5 "
              "to 0 over the main fit's optimization steps. |",
              "| Evolutional dropout | Mini-batch feature RMS defines multinomial probabilities; "
              "draw `k=round(D*(1-0.5))` units with replacement per sample and scale counts by `1/(k*p_j)`. "
              "Detached probability estimates, zero-feature fallback, and denominator guards. |",
              "| Curriculum dropout | Feature-unit dropout; probability increases as "
              "`0.5*(1-exp(-5*t/(T-1)))`, from zero toward 0.5. The horizon rate is fixed in advance. |", "",
              "These settings are fixed before evaluating test results. They are transparent project "
              "adaptations, not exact reproductions of the original authors' architectures or hyperparameter searches.", "",
              "## Coverage of the reference paper", "",
              "| Paper comparison | Status in this project |",
              "|---|---|",
              "| Tables V/VI: GBlending, Greedy, ModDrop and four neuron dropout methods | Fresh measured "
              "project adaptations above, alongside concatenation, OPM, OGM-GE and the combination. |",
              "| Table II: Summation and FiLM | Architecture comparisons; not implemented/evaluated in "
              "this update. Concatenation is the existing project's fusion architecture. |",
              "| Table III: CentralNet, VATT, MMTM | A final-feature MMTM control is measured for Greedy. "
              "Original CentralNet, VATT and multi-layer MMTM architectures are not reproduced. |",
              "| Tables VIII/IX: I-vector, X-vector, MWTSM, TSN, TSM | Require the paper's audio/video "
              "inputs and task-specific architectures; not evaluated on these stored image/text and embedding samples. |",
              "| Table VII: AVGA, AVSDN, PSP, AVSD, PanoAVQA | Event-localization or question-answering "
              "models; their tasks/data/metrics do not match these classification experiments. |", "",
              "No unavailable score is filled in as an experimental result. The main table covers "
              "the paper's directly comparable learning strategies; it does not claim every architecture "
              "in the paper has been trained here.", "",
              "## Paper-reported reference values (different datasets)", "",
              "Accuracy/mAP percentages from Tables V, VI and XII, provided as a literature reference. "
              "They are **reported by the paper**, not measured by this repository. No improvement "
              "is computed between these cells and Food-101, MELD or synthetic results.", "",
              "| Method | CREMA-D Acc / mAP | Kinetics-Sounds Acc / mAP | UCF-101 Acc / mAP | Source table |",
              "|---|---:|---:|---:|---|"]
    paper_rows = [("Concatenation", "66.9 / 72.1", "63.0 / 67.3", "80.5 / 86.4", "V/VI"),
                  ("GBlending", "72.0 / 78.6", "67.5 / 72.1", "80.9 / 86.6", "V"),
                  ("Greedy", "68.4 / 76.8", "65.2 / 69.6", "80.9 / 86.9", "V"),
                  ("ModDrop", "69.9 / 77.6", "65.9 / 71.8", "81.4 / 87.2", "VI"),
                  ("Dropout", "66.5 / 71.8", "62.8 / 66.1", "80.3 / 85.8", "VI"),
                  ("Annealed dropout", "65.3 / 70.9", "62.1 / 65.2", "80.1 / 84.7", "VI"),
                  ("Evolutional dropout", "65.9 / 72.0", "62.4 / 65.7", "80.4 / 86.1", "VI"),
                  ("Curriculum dropout", "64.9 / 70.2", "61.9 / 64.9", "79.9 / 84.3", "VI"),
                  ("OPM", "75.1 / 81.2", "67.0 / 72.5", "81.9 / 88.0", "V/VI"),
                  ("OGM", "74.6 / 81.4", "65.4 / 71.4", "81.5 / 87.6", "V/VI"),
                  ("OPM + OGM", "76.7 / 83.6", "68.0 / 74.7", "81.5 / 87.8", "XII")]
    lines += ["| " + " | ".join(row) + " |" for row in paper_rows]
    lines += ["", "## Reproduce", "", "Run from `balanced_mm/` with the repository requirements installed:", "",
              "```bash", "python scripts/unpack_demo_data.py --dataset all --out data/verified_demo",
              "python scripts/run_baseline_comparison.py --datasets food101 meld synthetic --seeds 0 1 2",
              "python scripts/plot_baseline_comparison.py --results results",
              "python scripts/make_baseline_report.py --results results", "```", "",
              "For larger prepared CSV datasets, change `--data-root`, training budget, and output "
              "directory explicitly; rerun all methods under that shared protocol. Best checkpoints "
              "are kept locally as ignored `.pt` files. Git contains configs, histories, pilot-loss "
              "evidence, predictions, confusion matrices, metric summaries and hashes.", "",
              "## Primary sources", "",
              "- [Reference paper, arXiv v1](https://arxiv.org/html/2410.11582v1) and "
              "[author implementation](https://github.com/GeWu-Lab/BML_TPAMI2024/tree/4b3aa4fd841856c4acbebf48549e6c59fbf7635e).",
              "- [Gradient Blending, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html), "
              "[Algorithm 1/2 PDF](https://arxiv.org/pdf/1905.12681).",
              "- [Greedy multimodal learning, ICML 2022](https://proceedings.mlr.press/v162/wu22d.html), "
              "[author code inspected at a190e03](https://github.com/nyukat/greedy_multimodal_learning/tree/a190e03755bd8d0883b2c5c0cd7921d2561ce2ee).",
              "- [ModDrop author paper](https://arxiv.org/abs/1501.00102).",
              "- [Dropout, JMLR](https://jmlr.org/papers/v15/srivastava14a.html).",
              "- [Annealed dropout, IBM Research](https://research.ibm.com/publications/annealed-dropout-training-of-deep-networks).",
              "- [Improved Dropout for Shallow and Deep Learning](https://arxiv.org/abs/1602.02220).",
              "- [Curriculum dropout, ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Morerio_Curriculum_Dropout_ICCV_2017_paper.html), "
              "[author code](https://github.com/pmorerio/curriculum-dropout).", "",
              "## Run evidence", "",
              "| Dataset | Seed | Method | Selected epoch (0-based) | Evidence |",
              "|---|---:|---|---:|---|"]
    for row in rows:
        path = row["evidence_path"]
        links = f"[Metrics + confusion matrix]({path}/classification_report.md) · [Config]({path}/config.json) · [History]({path}/history.json) · [Test predictions]({path}/test_metrics.json)"
        if row["method"] == "gblending":
            links += f" · [Pilot]({path}/pilot.json)"
        lines.append(f"| {row['dataset']} | {row['seed']} | {LABELS[row['method']]} | {row['best_epoch']} | {links} |")
    lines += ["", "Small samples, three seeds and five epochs support a pipeline comparison; "
              "they do not establish a statistically significant or publishable advantage. "
              "A method may improve one dataset and worsen another; the measured differences above "
              "are retained regardless of their direction.", ""]
    (root / "BASELINE_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    for name, records in (("baseline_comparison.csv", csv_rows), ("baseline_deltas.csv", deltas)):
        with (root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    print(f"Validated {len(rows)} runs; wrote comparison report and CSVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    make_report(parser.parse_args().results)
