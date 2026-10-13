# food101 — OPM (project) — seed 0

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 26.00 |
| macro precision | 25.89 |
| macro recall | 26.00 |
| macro f1 | 21.97 |
| weighted f1 | 21.97 |
| micro f1 | 26.00 |
| balanced accuracy | 26.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 11.76 | 40.00 | 18.18 | 5 |
| chocolate_cake | 0.00 | 0.00 | 0.00 | 5 |
| french_fries | 42.86 | 60.00 | 50.00 | 5 |
| fried_rice | 40.00 | 40.00 | 40.00 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 50.00 | 80.00 | 61.54 | 5 |
| pizza | 100.00 | 20.00 | 33.33 | 5 |
| ramen | 0.00 | 0.00 | 0.00 | 5 |
| sushi | 14.29 | 20.00 | 16.67 | 5 |
| tacos | 0.00 | 0.00 | 0.00 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
