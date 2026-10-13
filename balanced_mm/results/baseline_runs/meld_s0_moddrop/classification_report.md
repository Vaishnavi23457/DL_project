# meld — ModDrop (feature adaptation) — seed 0

Held-out test n=70; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 27.14 |
| macro precision | 24.61 |
| macro recall | 27.14 |
| macro f1 | 22.49 |
| weighted f1 | 22.49 |
| micro f1 | 27.14 |
| balanced accuracy | 27.14 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| anger | 26.32 | 50.00 | 34.48 | 10 |
| disgust | 22.73 | 50.00 | 31.25 | 10 |
| fear | 0.00 | 0.00 | 0.00 | 10 |
| joy | 37.50 | 30.00 | 33.33 | 10 |
| neutral | 35.71 | 50.00 | 41.67 | 10 |
| sadness | 0.00 | 0.00 | 0.00 | 10 |
| surprise | 50.00 | 10.00 | 16.67 | 10 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
