# Real-data examples for the live demo

`food101.zip` and `meld.zip` contain actual examples, not generated substitutes. Unpack with `python scripts/unpack_demo_data.py`. Checksums in `manifest.json` verify the archives and their contents.

| Dataset | Train | Validation | Test | Inputs |
|---|---|---|---|---|
| Food-101 | 200 | 50 | 50 | Real images resized to a maximum side of 96 px, plus generated label-derived prompt text |
| MELD | 350 | 70 | 70 | Real utterance text and official 300-dimensional audio embeddings |

Food-101 uses the ten classes listed in its manifest. Validation is drawn from the original training split; the original test split is held out. MELD retains its official splits and samples 50/10/10 examples per emotion. Sampling seed: 0. CSVs record original split and source identifiers.

Sources:

- [Food-101, ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) — Lukas Bossard, Matthieu Guillaumin, Luc Van Gool, *Food-101 – Mining Discriminative Components with Random Forests*, ECCV 2014. [Original archive](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz).
- [MELD official project](https://affective-meld.github.io/) and [declare-lab feature release](https://huggingface.co/datasets/declare-lab/MELD) — Soujanya Poria et al., *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*, ACL 2019. The upstream repository's GPL-3.0 license is included in `MELD_LICENSE.txt`.

These are teaching/research samples. Food text contains the true class name in some examples; this is an intentionally controlled shortcut. Small-sample accuracy is not a full-dataset benchmark. Use the Colab notebook to prepare different samples or all data.
