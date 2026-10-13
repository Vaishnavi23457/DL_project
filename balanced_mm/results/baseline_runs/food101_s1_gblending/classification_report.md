# food101 — GBlending (offline adaptation) — seed 1

Held-out test n=50; best validation checkpoint at epoch 4 (0-based).

| Metric | Score (%) |
|---|---:|
| accuracy | 34.00 |
| macro precision | 33.81 |
| macro recall | 34.00 |
| macro f1 | 30.46 |
| weighted f1 | 30.46 |
| micro f1 | 34.00 |
| balanced accuracy | 34.00 |

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
|---|---:|---:|---:|---:|
| caesar_salad | 21.43 | 60.00 | 31.58 | 5 |
| chocolate_cake | 50.00 | 80.00 | 61.54 | 5 |
| french_fries | 25.00 | 40.00 | 30.77 | 5 |
| fried_rice | 25.00 | 20.00 | 22.22 | 5 |
| hamburger | 0.00 | 0.00 | 0.00 | 5 |
| ice_cream | 33.33 | 40.00 | 36.36 | 5 |
| pizza | 100.00 | 40.00 | 57.14 | 5 |
| ramen | 0.00 | 0.00 | 0.00 | 5 |
| sushi | 33.33 | 40.00 | 36.36 | 5 |
| tacos | 50.00 | 20.00 | 28.57 | 5 |

Undefined precision/recall uses zero. For single-label multiclass classification, micro-F1 equals accuracy and balanced accuracy equals macro recall.

![Test confusion matrix: counts and row-normalized percentages](confusion_matrix.svg)

[Raw test predictions](test_metrics.json) · [Training history](history.json) · [Config](config.json)
