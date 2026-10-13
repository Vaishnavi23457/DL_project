# meld — GBlending (offline adaptation) — seed 1

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 18.57 |
| macro precision | 11.15 |
| macro recall | 18.57 |
| macro f1 | 12.89 |
| weighted f1 | 12.89 |
| micro f1 | 18.57 |
| balanced accuracy | 18.57 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 21.88 | 70.00 | 33.33 | 10 |
| disgust | 11.11 | 10.00 | 10.53 | 10 |
| fear | 33.33 | 30.00 | 31.58 | 10 |
| joy | 0.00 | 0.00 | 0.00 | 10 |
| neutral | 0.00 | 0.00 | 0.00 | 10 |
| sadness | 0.00 | 0.00 | 0.00 | 10 |
| surprise | 11.76 | 20.00 | 14.81 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
