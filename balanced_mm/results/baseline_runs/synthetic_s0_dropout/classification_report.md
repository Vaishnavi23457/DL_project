# synthetic — Dropout — seed 0

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 70.67 |
| macro precision | 77.18 |
| macro recall | 70.67 |
| macro f1 | 71.01 |
| weighted f1 | 71.01 |
| micro f1 | 70.67 |
| balanced accuracy | 70.67 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 55.88 | 76.00 | 64.41 | 25 |
| 1 | 60.71 | 68.00 | 64.15 | 25 |
| 2 | 100.00 | 80.00 | 88.89 | 25 |
| 3 | 100.00 | 52.00 | 68.42 | 25 |
| 4 | 87.50 | 56.00 | 68.29 | 25 |
| 5 | 58.97 | 92.00 | 71.88 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
