# meld — Greedy + MMTM (project adaptation) — seed 0

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 28.57 |
| macro precision | 31.74 |
| macro recall | 28.57 |
| macro f1 | 27.83 |
| weighted f1 | 27.83 |
| micro f1 | 28.57 |
| balanced accuracy | 28.57 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 22.22 | 40.00 | 28.57 | 10 |
| disgust | 16.67 | 10.00 | 12.50 | 10 |
| fear | 44.44 | 40.00 | 42.11 | 10 |
| joy | 26.32 | 50.00 | 34.48 | 10 |
| neutral | 50.00 | 30.00 | 37.50 | 10 |
| sadness | 12.50 | 10.00 | 11.11 | 10 |
| surprise | 50.00 | 20.00 | 28.57 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
