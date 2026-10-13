# meld — Evolutional dropout — seed 1

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 24.29 |
| macro precision | 24.45 |
| macro recall | 24.29 |
| macro f1 | 22.33 |
| weighted f1 | 22.33 |
| micro f1 | 24.29 |
| balanced accuracy | 24.29 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 23.08 | 60.00 | 33.33 | 10 |
| disgust | 22.22 | 20.00 | 21.05 | 10 |
| fear | 28.57 | 20.00 | 23.53 | 10 |
| joy | 20.00 | 10.00 | 13.33 | 10 |
| neutral | 25.00 | 20.00 | 22.22 | 10 |
| sadness | 25.00 | 10.00 | 14.29 | 10 |
| surprise | 27.27 | 30.00 | 28.57 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
