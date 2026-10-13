# meld — GBlending (offline adaptation) — seed 2

Held-out test n=70; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 28.57 |
| macro precision | 32.62 |
| macro recall | 28.57 |
| macro f1 | 25.60 |
| weighted f1 | 25.60 |
| micro f1 | 28.57 |
| balanced accuracy | 28.57 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 21.43 | 30.00 | 25.00 | 10 |
| disgust | 28.57 | 20.00 | 23.53 | 10 |
| fear | 50.00 | 10.00 | 16.67 | 10 |
| joy | 25.00 | 50.00 | 33.33 | 10 |
| neutral | 53.33 | 80.00 | 64.00 | 10 |
| sadness | 50.00 | 10.00 | 16.67 | 10 |
| surprise | 0.00 | 0.00 | 0.00 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
