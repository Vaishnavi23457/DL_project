# food101 — OPM (project) — seed 1

Held-out test n=50; best validation checkpoint at epoch 3 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 32.00 |
| macro precision | 39.44 |
| macro recall | 32.00 |
| macro f1 | 27.84 |
| weighted f1 | 27.84 |
| micro f1 | 32.00 |
| balanced accuracy | 32.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 25.00 | 60.00 | 35.29 | 5 |
| chocolate_cake | 44.44 | 80.00 | 57.14 | 5 |
| french_fries | 25.00 | 40.00 | 30.77 | 5 |
| fried_rice | 0.00 | 0.00 | 0.00 | 5 |
| hamburger | 50.00 | 20.00 | 28.57 | 5 |
| ice_cream | 30.00 | 60.00 | 40.00 | 5 |
| pizza | 100.00 | 20.00 | 33.33 | 5 |
| ramen | 0.00 | 0.00 | 0.00 | 5 |
| sushi | 100.00 | 20.00 | 33.33 | 5 |
| tacos | 20.00 | 20.00 | 20.00 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
