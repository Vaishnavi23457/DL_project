# synthetic — ModDrop (feature adaptation) — seed 2

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 64.67 |
| macro precision | 77.63 |
| macro recall | 64.67 |
| macro f1 | 65.50 |
| weighted f1 | 65.50 |
| micro f1 | 64.67 |
| balanced accuracy | 64.67 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 58.62 | 68.00 | 62.96 | 25 |
| 1 | 100.00 | 56.00 | 71.79 | 25 |
| 2 | 100.00 | 40.00 | 57.14 | 25 |
| 3 | 87.50 | 56.00 | 68.29 | 25 |
| 4 | 41.38 | 96.00 | 57.83 | 25 |
| 5 | 78.26 | 72.00 | 75.00 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
