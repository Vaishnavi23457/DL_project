# food101 — ModDrop (feature adaptation) — seed 0

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 26.00 |
| macro precision | 30.66 |
| macro recall | 26.00 |
| macro f1 | 23.86 |
| weighted f1 | 23.86 |
| micro f1 | 26.00 |
| balanced accuracy | 26.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 0.00 | 0.00 | 0.00 | 5 |
| chocolate_cake | 50.00 | 40.00 | 44.44 | 5 |
| french_fries | 100.00 | 20.00 | 33.33 | 5 |
| fried_rice | 25.00 | 40.00 | 30.77 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 0.00 | 0.00 | 0.00 | 5 |
| pizza | 36.36 | 80.00 | 50.00 | 5 |
| ramen | 28.57 | 40.00 | 33.33 | 5 |
| sushi | 16.67 | 20.00 | 18.18 | 5 |
| tacos | 50.00 | 20.00 | 28.57 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
