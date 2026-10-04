#!/usr/bin/env python
"""
make_food101_csv.py - build a Food-101 subset as (image, text, label) CSVs.

Food-101 itself has no text, so we give each image a CLIP-style prompt as its
text modality: with probability --text_p the prompt names the true class
("a photo of sushi"), otherwise a generic distractor ("a photo of food").
That makes the text modality *imperfect by design* -> a real modality-imbalance
setup on top of a real dataset (image usually more reliable than the prompt).

Prerequisite: the dataset extracted under --root, i.e.
    <root>/food-101/images/<class>/*.jpg
    <root>/food-101/meta/train.txt, test.txt

    python scripts/make_food101_csv.py --root data/food101

Writes <root>/train.csv and <root>/val.csv (paths relative to --root,
so train with  --img_root <root>).
"""
from __future__ import annotations

import argparse
import csv
import os
import random

DEFAULT_CLASSES = ["pizza", "sushi", "hamburger", "french_fries", "ice_cream",
                   "chocolate_cake", "ramen", "tacos", "fried_rice", "caesar_salad"]

CORRECT = [
    "a photo of {c}", "a close-up photo of {c}", "a plate of {c}",
    "a plate of {c} served at a restaurant", "delicious {c} on a table",
    "someone ordered {c}", "{c}, a popular dish",
]
DISTRACTOR = [
    "a photo of food", "a meal on a plate", "something edible on a table",
    "food being served at a restaurant", "a dish on a table", "lunch on a plate",
]


def read_split(path: str, classes, per_class: int, seed: int):
    by_cls = {c: [] for c in classes}
    with open(path) as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            cls = rel.split("/")[0]
            if cls in by_cls:
                by_cls[cls].append(rel)
    rng = random.Random(seed)
    chosen = []
    for c in classes:
        items = by_cls[c]
        rng.shuffle(items)
        if len(items) < per_class:
            raise SystemExit(f"class {c}: only {len(items)} images available (need {per_class})")
        chosen.extend(items[:per_class])
    rng.shuffle(chosen)
    return chosen


def make_text(cls: str, p: float, rng: random.Random) -> str:
    name = cls.replace("_", " ")
    tpl = rng.choice(CORRECT) if rng.random() < p else rng.choice(DISTRACTOR)
    return tpl.format(c=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/food101")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--train_per_class", type=int, default=250)
    ap.add_argument("--val_per_class", type=int, default=100)
    ap.add_argument("--text_p", type=float, default=0.7, help="prob. that the prompt names the true class")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    meta = os.path.join(a.root, "food-101", "meta")
    for split in ("train", "test"):
        if not os.path.isfile(os.path.join(meta, f"{split}.txt")):
            raise SystemExit(f"missing {meta}/{split}.txt - extract Food-101 first (see module docstring)")
    rng = random.Random(a.seed)
    for split, n_per, out_name in (("train", a.train_per_class, "train.csv"),
                                   ("test", a.val_per_class, "val.csv")):
        rels = read_split(os.path.join(meta, f"{split}.txt"), classes, n_per, a.seed)
        out_path = os.path.join(a.root, out_name)
        missing = 0
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "text", "label"])
            for rel in rels:
                img_rel = os.path.join("food-101", "images", rel + ".jpg")
                if not os.path.isfile(os.path.join(a.root, img_rel)):
                    missing += 1
                    continue
                cls = rel.split("/")[0]
                w.writerow([img_rel, make_text(cls, a.text_p, rng), cls])
        print(f"{out_path}: {len(rels) - missing} rows ({len(classes)} classes, text_p={a.text_p})"
              + (f"  [{missing} missing files skipped]" if missing else ""))


if __name__ == "__main__":
    main()
