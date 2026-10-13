# food101 — Greedy + MMTM (project adaptation) — seed 1

Held-out test n=50; best validation checkpoint at epoch 1 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 28.00 |
| macro precision | 22.65 |
| macro recall | 28.00 |
| macro f1 | 23.67 |
| weighted f1 | 23.67 |
| micro f1 | 28.00 |
| balanced accuracy | 28.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 18.18 | 40.00 | 25.00 | 5 |
| chocolate_cake | 40.00 | 80.00 | 53.33 | 5 |
| french_fries | 40.00 | 40.00 | 40.00 | 5 |
| fried_rice | 25.00 | 20.00 | 22.22 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 0.00 | 0.00 | 0.00 | 5 |
| pizza | 50.00 | 40.00 | 44.44 | 5 |
| ramen | 33.33 | 20.00 | 25.00 | 5 |
| sushi | 20.00 | 40.00 | 26.67 | 5 |
| tacos | 0.00 | 0.00 | 0.00 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
