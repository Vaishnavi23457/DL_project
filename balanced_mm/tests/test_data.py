import os
from types import SimpleNamespace

import numpy as np
import torch

from bml.data import build_dataloaders, collate_fn


def write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_args(tmp, **over):
    base = dict(data="csv", train_csv=os.path.join(tmp, "train.csv"), val_csv=os.path.join(tmp, "val.csv"),
                test_csv=None, img_root=str(tmp), col_image="image", col_text="text", col_label="label",
                col_audio="audio", text_encoder="transformer", hf_model="x", min_freq=1, max_vocab=500,
                max_len=16, img_size=32, batch_size=2, num_workers=0, seed=0,
                vocab_path=None, label_map_path=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_audio_csv_no_image(tmp_path):
    """MELD-style CSV: audio from packed npz + text, NO image column."""
    vecs = {f"k{i}": np.random.RandomState(i).randn(300).astype(np.float32) for i in range(6)}
    np.savez(tmp_path / "audio.npz", **vecs)
    rows = [[f"audio.npz::k{i}", f"utterance number {i}", ["joy", "anger"][i % 2]] for i in range(6)]
    write_csv(tmp_path / "train.csv", ["audio", "text", "label"], rows[:4])
    write_csv(tmp_path / "val.csv", ["audio", "text", "label"], rows[4:])

    data = build_dataloaders(make_args(str(tmp_path)))
    assert data["modalities"] == ["audio", "text"]
    assert data["audio_dim"] == 300
    assert data["num_classes"] == 2

    inputs, y = next(iter(data["loaders"]["val"]))
    assert set(inputs.keys()) == {"audio", "text"}
    assert inputs["audio"].shape == (2, 300)
    assert inputs["text"]["input_ids"].shape[0] == 2
    assert y.shape == (2,)


def test_audio_npy_path(tmp_path):
    """Plain .npy path in the audio column works too."""
    aud = tmp_path / "audio"
    aud.mkdir()
    np.save(aud / "0.npy", np.random.RandomState(0).randn(16).astype(np.float32))
    rows = [["audio/0.npy", "hello", "x"], ["audio/0.npy", "world", "y"]]
    write_csv(tmp_path / "train.csv", ["audio", "text", "label"], rows)
    write_csv(tmp_path / "val.csv", ["audio", "text", "label"], rows)
    data = build_dataloaders(make_args(str(tmp_path)))
    assert data["audio_dim"] == 16
    inputs, _ = next(iter(data["loaders"]["train"]))
    assert inputs["audio"].shape == (2, 16)


def test_image_text_csv_unchanged(tmp_path):
    """Food-101 / classic CSV keeps producing image+text inputs."""
    from PIL import Image
    (tmp_path / "images").mkdir()
    rows = []
    for i in range(6):
        p = tmp_path / "images" / f"{i}.png"
        Image.new("RGB", (32, 32), (i * 30 % 255, 50, 60)).save(p)
        rows.append([f"images/{i}.png", f"a photo of thing {i}", ["a", "b"][i % 2]])
    write_csv(tmp_path / "train.csv", ["image", "text", "label"], rows[:4])
    write_csv(tmp_path / "val.csv", ["image", "text", "label"], rows[4:])

    data = build_dataloaders(make_args(str(tmp_path)))
    assert data["modalities"] == ["image", "text"]
    assert data["audio_dim"] is None
    inputs, y = next(iter(data["loaders"]["train"]))
    assert set(inputs.keys()) == {"image", "text"}
    assert inputs["image"].shape == (2, 3, 32, 32)


def test_text_only_csv(tmp_path):
    rows = [[f"line {i}", ["x", "y"][i % 2]] for i in range(6)]
    write_csv(tmp_path / "train.csv", ["text", "label"], rows[:4])
    write_csv(tmp_path / "val.csv", ["text", "label"], rows[4:])
    data = build_dataloaders(make_args(str(tmp_path), col_audio=None))
    assert data["modalities"] == ["text"]
    inputs, y = next(iter(data["loaders"]["train"]))
    assert set(inputs.keys()) == {"text"}


def test_collate_handles_missing_modalities():
    batch = [{"text_ids": [1, 2], "label": 0}, {"text_ids": [3], "label": 1}]
    inputs, y = collate_fn(batch)
    assert set(inputs.keys()) == {"text"}
    assert inputs["text"]["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert y.tolist() == [0, 1]

    batch[0]["image"] = torch.zeros(3, 8, 8)
    batch[1]["image"] = torch.ones(3, 8, 8)
    inputs, _ = collate_fn(batch)
    assert inputs["image"].shape == (2, 3, 8, 8)
