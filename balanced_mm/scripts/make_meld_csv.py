#!/usr/bin/env python
"""
make_meld_csv.py - MELD (text + audio) -> CSVs + one packed audio .npz per split.

Uses the OFFICIAL MELD feature release (declare-lab):
    https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Features.Models.tar.gz
Extract it anywhere and point --features_dir at the folder containing
    audio_emotion.pkl   (per-dialogue audio embeddings, [max_utts, 300])
    data_emotion.p      (13,708 utterances: text / emotion / split / dialogue ids)

    python scripts/make_meld_csv.py --features_dir /path/to/features --out data/meld

Writes:
    <out>/audio.npz   keys '<split>_<utt idx>' -> 300-d audio embedding (all 13,708)
    <out>/train.csv  <out>/val.csv  <out>/test.csv
        columns: audio,text,label   where audio = 'audio.npz::<key>'

Train with:
    python train.py --data csv --train_csv data/meld/train.csv --val_csv data/meld/val.csv \
                    --img_root data/meld --dataset_name meld --modulation both ...
(the `audio` column is picked up automatically via --col_audio, default "audio")
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np

SPLIT_ORDER = ["train", "val", "test"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="data/meld_features",
                    help="folder with audio_emotion.pkl + data_emotion.p")
    ap.add_argument("--out", default="data/meld")
    ap.add_argument("--limit", type=int, default=0, help="0 = all utterances of each split")
    a = ap.parse_args()

    with open(os.path.join(a.features_dir, "data_emotion.p"), "rb") as f:
        uts = pickle.load(f)[0]
    with open(os.path.join(a.features_dir, "audio_emotion.pkl"), "rb") as f:
        audio = pickle.load(f)
    assert len(audio) == 3, "expected [train, val, test] audio dicts"

    os.makedirs(a.out, exist_ok=True)

    # sanity: for every dialogue, the audio rows must cover its utterances
    counts = {}
    for d in uts:
        key = (d["split"], d["dialog"])
        counts[key] = max(counts.get(key, 0), int(d["utterance"]) + 1)
    for si, s in enumerate(SPLIT_ORDER):
        for dk, arr in audio[si].items():
            need = counts.get((s, dk), 0)
            if arr.shape[0] < need:
                raise SystemExit(f"audio[{s}][{dk}] has {arr.shape[0]} rows but {need} utterances")

    feats, keys = {}, {}
    for si, s in enumerate(SPLIT_ORDER):
        keys[s] = []
    for idx, d in enumerate(uts):
        si = SPLIT_ORDER.index(d["split"])
        u = int(d["utterance"])
        k = f"{d['split']}_{idx:05d}"
        feats[k] = np.asarray(audio[si][d["dialog"]][u], dtype=np.float32)
        keys[d["split"]].append((idx, k))
    np.savez(os.path.join(a.out, "audio.npz"), **feats)

    for s in SPLIT_ORDER:
        path = os.path.join(a.out, f"{s}.csv")
        rows = 0
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["audio", "text", "label"])
            for idx, d in enumerate(uts):
                if d["split"] != s:
                    continue
                if a.limit and rows >= a.limit:
                    break
                k = f"{s}_{idx:05d}"
                w.writerow([f"audio.npz::{k}", d["text"], d["y"]])
                rows += 1
        print(f"{path}: {rows} rows")
    print(f"audio.npz: {len(feats)} vectors x {next(iter(feats.values())).shape[-1]} dims")
    print("classes:", sorted({d["y"] for d in uts}))


if __name__ == "__main__":
    main()
