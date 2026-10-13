# synthetic — OPM + OGM-GE (project) — seed 1

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 74.00 |
| macro precision | 77.93 |
| macro recall | 74.00 |
| macro f1 | 74.50 |
| weighted f1 | 74.50 |
| micro f1 | 74.00 |
| balanced accuracy | 74.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 62.07 | 72.00 | 66.67 | 25 |
| 1 | 93.75 | 60.00 | 73.17 | 25 |
| 2 | 90.48 | 76.00 | 82.61 | 25 |
| 3 | 79.17 | 76.00 | 77.55 | 25 |
| 4 | 85.71 | 72.00 | 78.26 | 25 |
| 5 | 56.41 | 88.00 | 68.75 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
