# food101 — Annealed dropout — seed 0

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 46.00 |
| macro precision | 57.02 |
| macro recall | 46.00 |
| macro f1 | 43.55 |
| weighted f1 | 43.55 |
| micro f1 | 46.00 |
| balanced accuracy | 46.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 30.00 | 60.00 | 40.00 | 5 |
| chocolate_cake | 57.14 | 80.00 | 66.67 | 5 |
| french_fries | 44.44 | 80.00 | 57.14 | 5 |
| fried_rice | 28.57 | 40.00 | 33.33 | 5 |
| hamburger | 50.00 | 20.00 | 28.57 | 5 |
| ice_cream | 60.00 | 60.00 | 60.00 | 5 |
| pizza | 100.00 | 20.00 | 33.33 | 5 |
| ramen | 50.00 | 20.00 | 28.57 | 5 |
| sushi | 50.00 | 60.00 | 54.55 | 5 |
| tacos | 100.00 | 20.00 | 33.33 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
