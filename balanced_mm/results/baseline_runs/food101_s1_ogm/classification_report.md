# food101 — OGM-GE (project) — seed 1

Held-out test n=50; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 40.00 |
| macro precision | 44.50 |
| macro recall | 40.00 |
| macro f1 | 38.88 |
| weighted f1 | 38.88 |
| micro f1 | 40.00 |
| balanced accuracy | 40.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 33.33 | 60.00 | 42.86 | 5 |
| chocolate_cake | 57.14 | 80.00 | 66.67 | 5 |
| french_fries | 33.33 | 40.00 | 36.36 | 5 |
| fried_rice | 25.00 | 20.00 | 22.22 | 5 |
| hamburger | 50.00 | 20.00 | 28.57 | 5 |
| ice_cream | 33.33 | 40.00 | 36.36 | 5 |
| pizza | 100.00 | 40.00 | 57.14 | 5 |
| ramen | 20.00 | 20.00 | 20.00 | 5 |
| sushi | 42.86 | 60.00 | 50.00 | 5 |
| tacos | 50.00 | 20.00 | 28.57 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
