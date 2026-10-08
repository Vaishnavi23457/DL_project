#!/usr/bin/env python
"""Unpack and verify the small real-data samples stored in this repository."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def unpack(dataset, destination):
    collection = json.loads((PROJECT / "demo_data/manifest.json").read_text())
    record = collection[dataset]
    archive = PROJECT / "demo_data" / record["archive"]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    target = Path(destination) / dataset
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            if (
                Path(member.filename).is_absolute()
                or ".." in Path(member.filename).parts
            ):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        stream.extractall(target)
    manifest = json.loads((target / "manifest.json").read_text())
    for relative, expected in manifest["file_sha256"].items():
        actual = hashlib.sha256((target / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Data checksum mismatch: {relative}")
    print(dataset, {name: info["count"] for name, info in manifest["splits"].items()})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["food101", "meld", "all"], default="all")
    parser.add_argument("--out", default="data/demo")
    args = parser.parse_args(argv)
    for dataset in ["food101", "meld"]:
        if args.dataset in (dataset, "all"):
            unpack(dataset, args.out)


if __name__ == "__main__":
    main()
