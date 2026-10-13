# synthetic — Evolutional dropout — seed 0

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 69.33 |
| macro precision | 76.92 |
| macro recall | 69.33 |
| macro f1 | 69.89 |
| weighted f1 | 69.89 |
| micro f1 | 69.33 |
| balanced accuracy | 69.33 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 50.00 | 68.00 | 57.63 | 25 |
| 1 | 61.54 | 64.00 | 62.75 | 25 |
| 2 | 100.00 | 80.00 | 88.89 | 25 |
| 3 | 100.00 | 56.00 | 71.79 | 25 |
| 4 | 92.86 | 52.00 | 66.67 | 25 |
| 5 | 57.14 | 96.00 | 71.64 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
