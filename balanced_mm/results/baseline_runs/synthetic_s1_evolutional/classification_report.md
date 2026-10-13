# synthetic — Evolutional dropout — seed 1

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 74.00 |
| macro precision | 79.79 |
| macro recall | 74.00 |
| macro f1 | 74.38 |
| weighted f1 | 74.38 |
| micro f1 | 74.00 |
| balanced accuracy | 74.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 83.33 | 60.00 | 69.77 | 25 |
| 1 | 100.00 | 60.00 | 75.00 | 25 |
| 2 | 70.37 | 76.00 | 73.08 | 25 |
| 3 | 90.91 | 80.00 | 85.11 | 25 |
| 4 | 80.95 | 68.00 | 73.91 | 25 |
| 5 | 53.19 | 100.00 | 69.44 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
