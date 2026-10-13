# food101 — ModDrop (feature adaptation) — seed 1

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 30.00 |
| macro precision | 20.14 |
| macro recall | 30.00 |
| macro f1 | 23.64 |
| weighted f1 | 23.64 |
| micro f1 | 30.00 |
| balanced accuracy | 30.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 30.00 | 60.00 | 40.00 | 5 |
| chocolate_cake | 66.67 | 80.00 | 72.73 | 5 |
| french_fries | 25.00 | 40.00 | 30.77 | 5 |
| fried_rice | 0.00 | 0.00 | 0.00 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 0.00 | 0.00 | 0.00 | 5 |
| pizza | 40.00 | 40.00 | 40.00 | 5 |
| ramen | 0.00 | 0.00 | 0.00 | 5 |
| sushi | 27.27 | 60.00 | 37.50 | 5 |
| tacos | 12.50 | 20.00 | 15.38 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
