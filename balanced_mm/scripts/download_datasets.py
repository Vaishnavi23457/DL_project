#!/usr/bin/env python
"""Download the original Food-101 archive and official MELD feature release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

FOOD_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
MELD_URL = "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Features.Models.tar.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    """Cache a complete download; never treat a partial file as a dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response:
        expected = int(response.headers.get("Content-Length", 0))
        if path.is_file() and expected and path.stat().st_size == expected:
            print(f"Using cached archive: {path}", flush=True)
            return
        received, next_notice = 0, 256 * 1024 * 1024
        with partial.open("wb") as output:
            while block := response.read(4 * 1024 * 1024):
                output.write(block)
                received += len(block)
                if received >= next_notice:
                    print(
                        f"Downloaded {received / 1024**2:.0f} MiB: {path.name}",
                        flush=True,
                    )
                    next_notice += 256 * 1024 * 1024
        if expected and received != expected:
            raise RuntimeError(f"Incomplete download: {received} / {expected} bytes")
    partial.replace(path)


def extract_food(archive: Path, destination: Path) -> None:
    marker = destination / ".extracted.json"
    if marker.exists() and (destination / "food-101/meta/train.txt").exists():
        print(f"Using extracted Food-101: {destination}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive entry: {member.name}")
            if not member.isfile():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
            if count % 20000 == 0:
                print(f"Extracted {count} Food-101 files", flush=True)
    if not (destination / "food-101/meta/train.txt").exists():
        raise RuntimeError("Food-101 metadata is missing after extraction")
    marker.write_text(
        json.dumps(
            {"source": FOOD_URL, "archive_sha256": sha256(archive), "files": count},
            indent=2,
        )
    )


def extract_meld(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {"audio_emotion.pkl", "data_emotion.p"}
    if all((destination / name).is_file() for name in wanted):
        print(f"Using extracted MELD features: {destination}", flush=True)
        return
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = Path(member.name).name
            if member.isfile() and name in wanted:
                with stream.extractfile(member) as source, (destination / name).open(
                    "wb"
                ) as output:
                    shutil.copyfileobj(source, output)
                wanted.remove(name)
                print(f"Extracted {name}", flush=True)
                if not wanted:
                    break
    if wanted:
        raise RuntimeError(f"Missing MELD features: {sorted(wanted)}")
    manifest = {
        "source": MELD_URL,
        "files": {
            name: sha256(destination / name)
            for name in ["audio_emotion.pkl", "data_emotion.p"]
        },
    }
    (destination / "source.json").write_text(json.dumps(manifest, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["food101", "meld", "all"], default="all")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument(
        "--food-archive", help="Use an already downloaded Food-101 tar.gz"
    )
    parser.add_argument("--meld-archive", help="Use an already downloaded MELD tar.gz")
    args = parser.parse_args(argv)
    root = Path(args.out)
    if args.dataset in ("food101", "all"):
        archive = (
            Path(args.food_archive)
            if args.food_archive
            else root / "downloads/food-101.tar.gz"
        )
        if not args.food_archive:
            download(FOOD_URL, archive)
        extract_food(archive, root / "food101")
    if args.dataset in ("meld", "all"):
        archive = (
            Path(args.meld_archive)
            if args.meld_archive
            else root / "downloads/MELD.Features.Models.tar.gz"
        )
        if not all(
            (root / "meld" / name).exists()
            for name in ["audio_emotion.pkl", "data_emotion.p"]
        ):
            if not args.meld_archive:
                download(MELD_URL, archive)
            extract_meld(archive, root / "meld")
    print("Original data is ready. Run scripts/prepare_experiment_data.py next.")


if __name__ == "__main__":
    main()
