# synthetic — Evolutional dropout — seed 2

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 67.33 |
| macro precision | 73.41 |
| macro recall | 67.33 |
| macro f1 | 67.42 |
| weighted f1 | 67.42 |
| micro f1 | 67.33 |
| balanced accuracy | 67.33 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 64.00 | 64.00 | 64.00 | 25 |
| 1 | 68.97 | 80.00 | 74.07 | 25 |
| 2 | 100.00 | 44.00 | 61.11 | 25 |
| 3 | 94.12 | 64.00 | 76.19 | 25 |
| 4 | 62.07 | 72.00 | 66.67 | 25 |
| 5 | 51.28 | 80.00 | 62.50 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
