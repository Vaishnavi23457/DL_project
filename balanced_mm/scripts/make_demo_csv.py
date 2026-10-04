#!/usr/bin/env python
"""
Writes a small demo dataset in the exact CSV format train.py expects, so you can see the format
and test the CSV pipeline end-to-end:

    python scripts/make_demo_csv.py --out data/demo --n_train 600 --n_val 120
    python train.py --data csv --train_csv data/demo/train.csv --val_csv data/demo/val.csv \
                    --img_root data/demo --image_encoder smallcnn --img_size 64 --epochs 3 --modulation both

Format (header required):
    image,text,label
    images/train_00000.png,"some words here",cls_3
image = path relative to --img_root (or absolute), label = any string/int.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bml.data import SyntheticTextImageDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/demo")
ap.add_argument("--n_train", type=int, default=600)
ap.add_argument("--n_val", type=int, default=120)
ap.add_argument("--num_classes", type=int, default=6)
ap.add_argument("--img_size", type=int, default=64)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
tr = SyntheticTextImageDataset(a.n_train, a.num_classes, a.img_size, split="train").to_csv(a.out, "train")
va = SyntheticTextImageDataset(a.n_val, a.num_classes, a.img_size, split="val").to_csv(a.out, "val")
print("wrote", tr, "and", va)
with open(tr) as f:
    for _ in range(3):
        print("  ", f.readline().rstrip())
