# synthetic — ModDrop (feature adaptation) — seed 0

Held-out test n=150; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 71.33 |
| macro precision | 83.12 |
| macro recall | 71.33 |
| macro f1 | 72.13 |
| weighted f1 | 72.13 |
| micro f1 | 71.33 |
| balanced accuracy | 71.33 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| 0 | 53.49 | 92.00 | 67.65 | 25 |
| 1 | 94.12 | 64.00 | 76.19 | 25 |
| 2 | 100.00 | 76.00 | 86.36 | 25 |
| 3 | 100.00 | 52.00 | 68.42 | 25 |
| 4 | 100.00 | 52.00 | 68.42 | 25 |
| 5 | 51.11 | 92.00 | 65.71 | 25 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
