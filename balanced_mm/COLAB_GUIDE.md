# Run Food-101 and MELD from a fresh clone

[Open the training notebook in Colab](https://colab.research.google.com/github/Nishant21092004/DL_project/blob/main/balanced_mm/notebooks/02_colab_training.ipynb)

Run the cells from top to bottom. The notebook downloads/prepares data, trains fresh models, selects the best checkpoint on validation, and evaluates a separate test split. It displays and saves accuracy/loss curves, modality ratios, test confusion matrices, macro-F1, class-wise accuracy, calibration error, and checkpoints.

## Choose a data mode

| Mode | Data location and sampling |
|---|---|
| `repo_sample` | Unpacks actual Food-101 images and MELD text/audio examples from `demo_data/`. No large download. |
| `custom_sample` | Downloads original data and lets you choose Food classes, samples per class in each split, image resolution, and sampling seed. |
| `full_data` | All 101 Food classes and all official MELD splits. Food's original training split supplies training and validation; its official test split is kept for test. |

Bundled sample sizes (train / validation / test): Food-101 **200 / 50 / 50**, MELD **350 / 70 / 70**. MELD samples have equal counts per emotion; their distribution differs from the full dataset.

Full mode uses Food-101 **68,175 training / 7,575 validation / 25,250 test** images and MELD **9,989 / 1,109 / 2,610** utterances. Every selected training row is used, including the final partial batch.

Parameters also control epochs, methods (`none`, `opm`, `ogm`, `both`), training seeds, image encoder, text width/layers, batch size, and learning rate. Set `SEEDS='0,1,2'` for a three-seed comparison. Colab GPU availability and runtime limits vary; start with the bundled sample.

## Local commands

From `balanced_mm/`:

```bash
pip install -r requirements.txt
python scripts/unpack_demo_data.py --out data/demo
python scripts/run_colab_experiments.py --data data/demo --epochs 5 --out demo_runs_new
```

For custom samples or full data:

```bash
python scripts/download_datasets.py --dataset all --out data/raw
python scripts/prepare_experiment_data.py --source data/raw --out data/experiment --food-classes all --food-train-per-class 0 --food-val-per-class 75 --food-test-per-class 0 --meld-train-per-class 0 --meld-val-per-class 0 --meld-test-per-class 0
python scripts/run_colab_experiments.py --data data/experiment --epochs 12 --seeds 0,1,2 --out full_runs
```

Zero sample counts mean all available data (Food validation zero reserves 10% of its original training split). Positive counts take a seeded sample within each class. Increase sample sizes/epochs gradually. Only Food's test images come from its original test split: validation is reserved from original training images.

## Results and evidence

- `demo_runs/RESULTS.md`: newly executed sample results and figures.
- `results/RESULTS.md`: the older 42-run benchmark; different data sizes, model settings, and split protocol.
- Each new run saves configuration, history, training log, `best.pt`/`last.pt`, test metrics, PNG plots, and the numeric confusion matrix as CSV. New checkpoints are created during execution; large weight files are not committed.
- The notebook's final cell downloads a ZIP of the new results and checkpoints. Save it before the Colab session ends; `/content` is temporary. Mount Drive and set `DATA_ROOT` there if you want to retain raw data.

## Interpretation

Food-101 itself has images and labels. Our text inputs are label-derived prompts, including the correct food name with probability `text_p` (default 0.7). This is a controlled shortcut/bias experiment, not a natural caption benchmark or standard Food-101 leaderboard comparison. More classes do not remove the shortcut.

MELD uses official utterance text and 300-dimensional pre-extracted audio embeddings. Visual inputs are not included. Report accuracy alongside macro-F1 and class-wise errors. A lower discrepancy ratio need not mean better classification accuracy.

The bundled sample and short training demonstrate reproducibility. Use larger experiments, multiple seeds, a fixed evaluation protocol, and a clear contribution for a paper.
