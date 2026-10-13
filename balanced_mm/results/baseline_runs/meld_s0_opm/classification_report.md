# meld — OPM (project) — seed 0

Held-out test n=70; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 24.29 |
| macro precision | 34.82 |
| macro recall | 24.29 |
| macro f1 | 20.12 |
| weighted f1 | 20.12 |
| micro f1 | 24.29 |
| balanced accuracy | 24.29 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 17.50 | 70.00 | 28.00 | 10 |
| disgust | 20.00 | 10.00 | 13.33 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 50.00 | 20.00 | 28.57 | 10 |
| neutral | 31.25 | 50.00 | 38.46 | 10 |
| sadness | 25.00 | 10.00 | 14.29 | 10 |
| surprise | 100.00 | 10.00 | 18.18 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
