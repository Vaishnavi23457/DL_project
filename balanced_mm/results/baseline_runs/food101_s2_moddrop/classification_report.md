# food101 — ModDrop (feature adaptation) — seed 2

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 36.00 |
| macro precision | 27.41 |
| macro recall | 36.00 |
| macro f1 | 29.25 |
| weighted f1 | 29.25 |
| micro f1 | 36.00 |
| balanced accuracy | 36.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 57.14 | 80.00 | 66.67 | 5 |
| chocolate_cake | 33.33 | 80.00 | 47.06 | 5 |
| french_fries | 0.00 | 0.00 | 0.00 | 5 |
| fried_rice | 0.00 | 0.00 | 0.00 | 5 |
| hamburger | 33.33 | 40.00 | 36.36 | 5 |
| ice_cream | 25.00 | 20.00 | 22.22 | 5 |
| pizza | 18.18 | 40.00 | 25.00 | 5 |
| ramen | 50.00 | 20.00 | 28.57 | 5 |
| sushi | 0.00 | 0.00 | 0.00 | 5 |
| tacos | 57.14 | 80.00 | 66.67 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
