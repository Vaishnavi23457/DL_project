# meld — OGM-GE (project) — seed 2

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 27.14 |
| macro precision | 29.99 |
| macro recall | 27.14 |
| macro f1 | 22.93 |
| weighted f1 | 22.93 |
| micro f1 | 27.14 |
| balanced accuracy | 27.14 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 21.74 | 50.00 | 30.30 | 10 |
| disgust | 50.00 | 10.00 | 16.67 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 0.00 | 0.00 | 0.00 | 10 |
| neutral | 75.00 | 30.00 | 42.86 | 10 |
| sadness | 18.75 | 60.00 | 28.57 | 10 |
| surprise | 44.44 | 40.00 | 42.11 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
