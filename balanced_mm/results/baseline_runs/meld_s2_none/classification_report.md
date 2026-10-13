# meld — Concatenation / no modulation — seed 2

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 24.29 |
| macro precision | 35.30 |
| macro recall | 24.29 |
| macro f1 | 20.53 |
| weighted f1 | 20.53 |
| micro f1 | 24.29 |
| balanced accuracy | 24.29 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 20.83 | 50.00 | 29.41 | 10 |
| disgust | 100.00 | 10.00 | 18.18 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 0.00 | 0.00 | 0.00 | 10 |
| neutral | 66.67 | 20.00 | 30.77 | 10 |
| sadness | 15.15 | 50.00 | 23.26 | 10 |
| surprise | 44.44 | 40.00 | 42.11 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
