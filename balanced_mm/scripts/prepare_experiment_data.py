#!/usr/bin/env python
"""Make configurable, disjoint train/validation/test data for Food-101 and MELD."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import pickle
import random
import shutil
import zipfile

import numpy as np
from PIL import Image

from download_datasets import FOOD_URL, MELD_URL, sha256
from make_food101_csv import DEFAULT_CLASSES, make_text

PROJECT = Path(__file__).resolve().parents[1]


def reset_destination(destination, source):
    destination, source = Path(destination), Path(source)
    if source.resolve().is_relative_to(
        destination.resolve()
    ) or destination.resolve().is_relative_to(source.resolve()):
        raise ValueError(
            "Prepared-data output must be separate from the original dataset folder"
        )
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def take(items, count, rng):
    items = list(items)
    rng.shuffle(items)
    if count < 0:
        raise ValueError("Sample counts must be zero (all) or positive")
    return items if count == 0 else items[:count]


def save_manifest(destination, dataset, source, splits, **extra):
    ids = {
        name: {(row["source_split"], row["source_id"]) for row in rows}
        for name, rows in splits.items()
    }
    for name, rows in splits.items():
        if len(ids[name]) != len(rows):
            raise ValueError(f"Duplicate source examples in {name}")
    if (
        ids["train"] & ids["val"]
        or ids["train"] & ids["test"]
        or ids["val"] & ids["test"]
    ):
        raise ValueError(
            "Training, validation, and test must not share source examples"
        )
    manifest = {
        "dataset": dataset,
        "source": source,
        "splits": {
            name: {
                "count": len(rows),
                "class_counts": dict(sorted(Counter(r["label"] for r in rows).items())),
            }
            for name, rows in splits.items()
        },
        **extra,
    }
    manifest["file_sha256"] = {
        str(path.relative_to(destination)): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(dataset, {name: len(rows) for name, rows in splits.items()}, flush=True)
    return manifest


def prepare_food(args):
    original = Path(args.source) / "food101/food-101"
    if not (original / "meta/train.txt").is_file():
        raise FileNotFoundError(
            "Download/extract Food-101 first with scripts/download_datasets.py"
        )
    classes = (
        sorted(p.name for p in (original / "images").iterdir() if p.is_dir())
        if args.food_classes == "all"
        else [name.strip() for name in args.food_classes.split(",") if name.strip()]
    )
    by_split = {}
    for name in ["train", "test"]:
        groups = defaultdict(list)
        for line in (original / f"meta/{name}.txt").read_text().splitlines():
            if line.strip():
                groups[line.split("/")[0]].append(line.strip())
        by_split[name] = groups
    destination = Path(args.out) / "food101"
    reset_destination(destination, Path(args.source) / "food101")
    splits = {name: [] for name in ["train", "val", "test"]}
    for index, label in enumerate(classes):
        train_pool = by_split["train"][label]
        if not train_pool:
            raise ValueError(f"Unknown or empty Food-101 class: {label}")
        rng = random.Random(args.seed + index)
        shuffled = take(train_pool, 0, rng)
        val_count = args.food_val_per_class or max(1, len(shuffled) // 10)
        if val_count >= len(shuffled):
            raise ValueError(
                "Validation count must leave some original training images for training"
            )
        selected = {
            "val": shuffled[:val_count],
            "train": (
                shuffled[val_count:]
                if args.food_train_per_class == 0
                else shuffled[val_count : val_count + args.food_train_per_class]
            ),
            "test": take(
                by_split["test"][label],
                args.food_test_per_class,
                random.Random(args.seed + 10000 + index),
            ),
        }
        for split, ids in selected.items():
            caption_rng = random.Random(
                args.seed + index + {"train": 0, "val": 10000, "test": 20000}[split]
            )
            for source_id in ids:
                relative = Path("images") / (source_id + ".jpg")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if args.food_resize:
                    with Image.open(original / relative) as image:
                        image = image.convert("RGB")
                        image.thumbnail(
                            (args.food_resize, args.food_resize),
                            Image.Resampling.LANCZOS,
                        )
                        image.save(target, quality=85)
                else:
                    shutil.copyfile(original / relative, target)
                splits[split].append(
                    {
                        "image": relative.as_posix(),
                        "text": make_text(label, args.text_p, caption_rng),
                        "label": label,
                        "source_split": "test" if split == "test" else "train",
                        "source_id": source_id,
                    }
                )
    for split, rows in splits.items():
        random.Random(args.seed).shuffle(rows)
        write_csv(
            destination / f"{split}.csv",
            rows,
            ["image", "text", "label", "source_split", "source_id"],
        )
    return save_manifest(
        destination,
        "food101",
        FOOD_URL,
        splits,
        seed=args.seed,
        classes=classes,
        text_p=args.text_p,
        image_max_side=args.food_resize,
        text_protocol="Label-derived prompts: controlled bias demonstration, not natural captions.",
        split_protocol="Validation reserved from original training images; original test images used only for test.",
    )


def prepare_meld(args):
    source = Path(args.source) / "meld"
    with (source / "data_emotion.p").open("rb") as stream:
        utterances = pickle.load(stream)[0]
    with (source / "audio_emotion.pkl").open("rb") as stream:
        audio = pickle.load(stream)
    destination = Path(args.out) / "meld"
    reset_destination(destination, source)
    features, splits = {}, {}
    for split_index, split in enumerate(["train", "val", "test"]):
        groups = defaultdict(list)
        for index, utterance in enumerate(utterances):
            if utterance["split"] == split:
                groups[utterance["y"]].append((index, utterance))
        count = getattr(args, f"meld_{split}_per_class")
        selected = []
        for label_index, label in enumerate(sorted(groups)):
            selected.extend(
                take(
                    groups[label],
                    count,
                    random.Random(args.seed + 100 * split_index + label_index),
                )
            )
        random.Random(args.seed).shuffle(selected)
        rows = []
        for index, utterance in selected:
            key = f"{split}_{index:05d}"
            features[key] = np.asarray(
                audio[split_index][utterance["dialog"]][int(utterance["utterance"])],
                dtype=np.float32,
            )
            rows.append(
                {
                    "audio": f"audio.npz::{key}",
                    "text": utterance["text"],
                    "label": utterance["y"],
                    "source_split": split,
                    "source_id": f"{utterance['dialog']}:{utterance['utterance']}",
                }
            )
        splits[split] = rows
        write_csv(
            destination / f"{split}.csv",
            rows,
            ["audio", "text", "label", "source_split", "source_id"],
        )
    np.savez_compressed(destination / "audio.npz", **features)
    return save_manifest(
        destination,
        "meld",
        MELD_URL,
        splits,
        seed=args.seed,
        audio_dim=300,
        sampling="Equal per-class sample when count is positive; official distribution when count is zero.",
        split_protocol="Official train, development, and test splits preserved.",
        source_sha256={
            name: sha256(source / name)
            for name in ["audio_emotion.pkl", "data_emotion.p"]
        },
    )


def pack_demo(root, dataset):
    destination = PROJECT / "demo_data"
    destination.mkdir(exist_ok=True)
    archive = destination / f"{dataset}.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as output:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                info = zipfile.ZipInfo(
                    file.relative_to(root).as_posix(), date_time=(2026, 10, 8, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                output.writestr(info, file.read_bytes())
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[dataset] = {
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "data": json.loads((root / "manifest.json").read_text()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Packed real-data sample: {archive.name} ({archive.stat().st_size / 1024:.0f} KiB)"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["food101", "meld", "all"], default="all")
    parser.add_argument("--source", default="data/raw")
    parser.add_argument("--out", default="data/experiment")
    parser.add_argument(
        "--food-classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated names, or all for 101 classes",
    )
    for split, default in [("train", 20), ("val", 5), ("test", 5)]:
        parser.add_argument(
            f"--food-{split}-per-class",
            type=int,
            default=default,
            help="0 = all available (val: reserve 10%%)",
        )
    for split, default in [("train", 50), ("val", 10), ("test", 10)]:
        parser.add_argument(
            f"--meld-{split}-per-class",
            type=int,
            default=default,
            help="0 = full official split",
        )
    parser.add_argument(
        "--food-resize",
        type=int,
        default=96,
        help="0 = preserve original image resolution",
    )
    parser.add_argument("--text-p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pack-demo", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.text_p <= 1:
        parser.error("--text-p must be between zero and one")
    counts = [
        getattr(args, f"{dataset}_{split}_per_class")
        for dataset in ["food", "meld"]
        for split in ["train", "val", "test"]
    ]
    if min(counts) < 0 or args.food_resize < 0:
        parser.error("Counts and image size cannot be negative")
    for dataset, prepare in [("food101", prepare_food), ("meld", prepare_meld)]:
        if args.dataset in (dataset, "all"):
            prepare(args)
            if args.pack_demo:
                pack_demo(Path(args.out) / dataset, dataset)


if __name__ == "__main__":
    main()
