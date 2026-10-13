# meld — GBlending (offline adaptation) — seed 0

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 25.71 |
| macro precision | 29.40 |
| macro recall | 25.71 |
| macro f1 | 24.21 |
| weighted f1 | 24.21 |
| micro f1 | 25.71 |
| balanced accuracy | 25.71 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 16.67 | 20.00 | 18.18 | 10 |
| disgust | 33.33 | 10.00 | 15.38 | 10 |
| fear | 27.78 | 50.00 | 35.71 | 10 |
| joy | 25.00 | 30.00 | 27.27 | 10 |
| neutral | 36.36 | 40.00 | 38.10 | 10 |
| sadness | 50.00 | 10.00 | 16.67 | 10 |
| surprise | 16.67 | 20.00 | 18.18 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
