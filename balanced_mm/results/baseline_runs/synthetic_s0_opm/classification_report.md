# synthetic — OPM (project) — seed 0

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 67.33 |
| macro precision | 72.52 |
| macro recall | 67.33 |
| macro f1 | 66.79 |
| weighted f1 | 66.79 |
| micro f1 | 67.33 |
| balanced accuracy | 67.33 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 48.78 | 80.00 | 60.61 | 25 |
| 1 | 80.00 | 48.00 | 60.00 | 25 |
| 2 | 78.57 | 88.00 | 83.02 | 25 |
| 3 | 86.67 | 52.00 | 65.00 | 25 |
| 4 | 80.00 | 48.00 | 60.00 | 25 |
| 5 | 61.11 | 88.00 | 72.13 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
