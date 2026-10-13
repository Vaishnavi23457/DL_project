# food101 — OPM + OGM-GE (project) — seed 1

Held-out test n=50; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 28.00 |
| macro precision | 28.31 |
| macro recall | 28.00 |
| macro f1 | 23.70 |
| weighted f1 | 23.70 |
| micro f1 | 28.00 |
| balanced accuracy | 28.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 20.00 | 40.00 | 26.67 | 5 |
| chocolate_cake | 28.57 | 80.00 | 42.11 | 5 |
| french_fries | 16.67 | 20.00 | 18.18 | 5 |
| fried_rice | 100.00 | 20.00 | 33.33 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 50.00 | 40.00 | 44.44 | 5 |
| pizza | 42.86 | 60.00 | 50.00 | 5 |
| ramen | 0.00 | 0.00 | 0.00 | 5 |
| sushi | 0.00 | 0.00 | 0.00 | 5 |
| tacos | 25.00 | 20.00 | 22.22 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
