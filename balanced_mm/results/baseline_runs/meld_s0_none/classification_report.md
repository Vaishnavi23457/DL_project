# meld — Concatenation / no modulation — seed 0

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 27.14 |
| macro precision | 38.07 |
| macro recall | 27.14 |
| macro f1 | 26.07 |
| weighted f1 | 26.07 |
| micro f1 | 27.14 |
| balanced accuracy | 27.14 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 22.73 | 50.00 | 31.25 | 10 |
| disgust | 20.00 | 20.00 | 20.00 | 10 |
| fear | 100.00 | 10.00 | 18.18 | 10 |
| joy | 40.00 | 40.00 | 40.00 | 10 |
| neutral | 44.44 | 40.00 | 42.11 | 10 |
| sadness | 14.29 | 20.00 | 16.67 | 10 |
| surprise | 25.00 | 10.00 | 14.29 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
