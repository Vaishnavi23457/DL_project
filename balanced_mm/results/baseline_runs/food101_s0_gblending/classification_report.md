# food101 — GBlending (offline adaptation) — seed 0

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 24.00 |
| macro precision | 25.40 |
| macro recall | 24.00 |
| macro f1 | 20.50 |
| weighted f1 | 20.50 |
| micro f1 | 24.00 |
| balanced accuracy | 24.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 10.00 | 20.00 | 13.33 | 5 |
| chocolate_cake | 30.00 | 60.00 | 40.00 | 5 |
| french_fries | 18.18 | 40.00 | 25.00 | 5 |
| fried_rice | 0.00 | 0.00 | 0.00 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 37.50 | 60.00 | 46.15 | 5 |
| pizza | 100.00 | 20.00 | 33.33 | 5 |
| ramen | 33.33 | 20.00 | 25.00 | 5 |
| sushi | 25.00 | 20.00 | 22.22 | 5 |
| tacos | 0.00 | 0.00 | 0.00 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
