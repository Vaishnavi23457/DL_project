# meld — Concatenation / no modulation — seed 1

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 24.29 |
| macro precision | 21.61 |
| macro recall | 24.29 |
| macro f1 | 19.60 |
| weighted f1 | 19.60 |
| micro f1 | 24.29 |
| balanced accuracy | 24.29 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 37.50 | 30.00 | 33.33 | 10 |
| disgust | 0.00 | 0.00 | 0.00 | 10 |
| fear | 42.86 | 30.00 | 35.29 | 10 |
| joy | 33.33 | 10.00 | 15.38 | 10 |
| neutral | 24.24 | 80.00 | 37.21 | 10 |
| sadness | 13.33 | 20.00 | 16.00 | 10 |
| surprise | 0.00 | 0.00 | 0.00 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
