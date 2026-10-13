# food101 — Annealed dropout — seed 1

Held-out test n=50; best validation checkpoint at epoch 2 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 28.00 |
| macro precision | 25.58 |
| macro recall | 28.00 |
| macro f1 | 24.25 |
| weighted f1 | 24.25 |
| micro f1 | 28.00 |
| balanced accuracy | 28.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 27.27 | 60.00 | 37.50 | 5 |
| chocolate_cake | 50.00 | 80.00 | 61.54 | 5 |
| french_fries | 50.00 | 20.00 | 28.57 | 5 |
| fried_rice | 28.57 | 40.00 | 33.33 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 25.00 | 40.00 | 30.77 | 5 |
| pizza | 0.00 | 0.00 | 0.00 | 5 |
| ramen | 0.00 | 0.00 | 0.00 | 5 |
| sushi | 50.00 | 20.00 | 28.57 | 5 |
| tacos | 25.00 | 20.00 | 22.22 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
