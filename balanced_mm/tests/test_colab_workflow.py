"""Checks that the live demo has real, disjoint data and retains all training rows."""

import csv
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from bml.data import build_dataloaders
from train import get_args
from prepare_experiment_data import reset_destination
from unpack_demo_data import unpack


@pytest.mark.parametrize("dataset,classes", [("food101", 10), ("meld", 7)])
def test_bundled_sources_are_disjoint_and_loadable(tmp_path, dataset, classes):
    manifest = unpack(dataset, tmp_path)
    root = tmp_path / dataset
    ids = {}
    for split in ["train", "val", "test"]:
        with (root / f"{split}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == manifest["splits"][split]["count"]
        assert len({row["label"] for row in rows}) == classes
        ids[split] = {(row["source_split"], row["source_id"]) for row in rows}
        assert len(ids[split]) == len(rows)
        if dataset == "meld":
            with np.load(root / "audio.npz") as features:
                assert all(
                    features[row["audio"].split("::")[1]].shape == (300,)
                    for row in rows
                )
        else:
            from PIL import Image

            for row in rows:
                with Image.open(root / row["image"]) as image:
                    assert max(image.size) <= 96
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_keep_last_batch_uses_every_training_example():
    args = get_args(
        [
            "--data",
            "synthetic",
            "--syn_train_n",
            "35",
            "--syn_val_n",
            "8",
            "--batch_size",
            "16",
            "--num_workers",
            "0",
            "--keep_last_batch",
        ]
    )
    loader = build_dataloaders(args)["loaders"]["train"]
    assert [len(labels) for inputs, labels in loader] == [16, 16, 3]


def test_preparation_cannot_erase_original_data(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    marker = source / "original.txt"
    marker.write_text("keep original data")
    with pytest.raises(ValueError):
        reset_destination(source, source)
    assert marker.read_text() == "keep original data"
