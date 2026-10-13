# synthetic — Dropout — seed 1

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 78.00 |
| macro precision | 80.78 |
| macro recall | 78.00 |
| macro f1 | 77.87 |
| weighted f1 | 77.87 |
| micro f1 | 78.00 |
| balanced accuracy | 78.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 75.00 | 72.00 | 73.47 | 25 |
| 1 | 88.24 | 60.00 | 71.43 | 25 |
| 2 | 90.91 | 80.00 | 85.11 | 25 |
| 3 | 78.57 | 88.00 | 83.02 | 25 |
| 4 | 89.47 | 68.00 | 77.27 | 25 |
| 5 | 62.50 | 100.00 | 76.92 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
