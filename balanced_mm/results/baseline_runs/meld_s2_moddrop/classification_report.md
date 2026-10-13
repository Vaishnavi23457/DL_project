# meld — ModDrop (feature adaptation) — seed 2

Held-out test n=70; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 15.71 |
| macro precision | 11.11 |
| macro recall | 15.71 |
| macro f1 | 8.31 |
| weighted f1 | 8.31 |
| micro f1 | 15.71 |
| balanced accuracy | 15.71 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 11.76 | 20.00 | 14.81 | 10 |
| disgust | 0.00 | 0.00 | 0.00 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 0.00 | 0.00 | 0.00 | 10 |
| neutral | 0.00 | 0.00 | 0.00 | 10 |
| sadness | 16.00 | 80.00 | 26.67 | 10 |
| surprise | 50.00 | 10.00 | 16.67 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
