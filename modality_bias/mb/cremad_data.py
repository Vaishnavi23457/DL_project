"""
mb.cremad_data
==============
CREMA-D dataset loader for audio + video multimodal learning.

CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
  - 7,442 clips, 91 actors (IDs 1001–1091), 6 emotions
  - Filename pattern: 1001_DFA_ANG_XX.flv
    actor_id _ sentence _ emotion _ intensity

Emotions: ANG=0, DIS=1, FEA=2, HAP=3, NEU=4, SAD=5

Speaker-independent split (no actor appears in multiple splits):
  train : actors 1001–1073  (73 actors)
  val   : actors 1074–1082  ( 9 actors)
  test  : actors 1083–1091  ( 9 actors)

Expected directory layout after preprocessing:
    cremad_root/
        audio/       <- log-mel spectrograms as .npy files  [1, 128, 128]
        frames/      <- video frames as .npy files          [3, 3, 224, 224]
        filelist.csv <- columns: stem, actor_id, emotion, split

Usage
-----
    from mb.cremad_data import CREMADDataset, build_cremad_loaders
    train_ds = CREMADDataset(root, split="train")
    loaders  = build_cremad_loaders(root, batch_size=32)
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LABEL_MAP: Dict[str, int] = {
    "ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5
}
IDX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = 6

# Speaker-independent split boundaries
TRAIN_ACTORS = set(range(1001, 1074))   # 73 actors
VAL_ACTORS   = set(range(1074, 1083))   #  9 actors
TEST_ACTORS  = set(range(1083, 1092))   #  9 actors


def _actor_to_split(actor_id: int) -> str:
    if actor_id in TRAIN_ACTORS:
        return "train"
    if actor_id in VAL_ACTORS:
        return "val"
    return "test"


def parse_cremad_filename(stem: str) -> Tuple[int, str]:
    """
    Parse CREMA-D filename stem.
    '1001_DFA_ANG_XX' -> (actor_id=1001, emotion='ANG')
    """
    parts = stem.split("_")
    actor_id = int(parts[0])
    emotion  = parts[2]
    return actor_id, emotion


class CREMADDataset(Dataset):
    """
    CREMA-D dataset returning (audio_spec, video_frames, label) tuples.

    Parameters
    ----------
    root : str
        Path to preprocessed CREMA-D directory containing audio/, frames/, filelist.csv
    split : str
        One of 'train', 'val', 'test'
    audio_transform : callable, optional
        Transform applied to audio spectrogram tensor [1, 128, 128]
    video_transform : callable, optional
        Transform applied to each video frame tensor [3, 224, 224]
    """

    modalities = ["audio", "video"]

    def __init__(self, root: str, split: str = "train",
                 audio_transform=None, video_transform=None):
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split}"
        self.root   = root
        self.split  = split
        self.audio_dir  = os.path.join(root, "audio")
        self.frames_dir = os.path.join(root, "frames")
        self.audio_transform = audio_transform
        self.video_transform = video_transform

        self.samples: List[Dict] = self._load_filelist()

    def _load_filelist(self) -> List[Dict]:
        csv_path = os.path.join(self.root, "filelist.csv")
        if os.path.exists(csv_path):
            return self._from_csv(csv_path)
        return self._scan_directory()

    def _from_csv(self, csv_path: str) -> List[Dict]:
        samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == self.split:
                    samples.append({
                        "stem"     : row["stem"],
                        "actor_id" : int(row["actor_id"]),
                        "emotion"  : row["emotion"],
                        "label"    : LABEL_MAP[row["emotion"]],
                    })
        return samples

    def _scan_directory(self) -> List[Dict]:
        """Scan audio/ directory if no filelist.csv exists."""
        samples = []
        audio_dir = self.audio_dir
        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(
                f"Audio directory not found: {audio_dir}\n"
                "Run preprocessing first (see data/preprocess.py or notebook 01)."
            )
        for fname in sorted(os.listdir(audio_dir)):
            if not fname.endswith(".npy"):
                continue
            stem = fname[:-4]
            try:
                actor_id, emotion = parse_cremad_filename(stem)
            except (ValueError, IndexError):
                continue
            if emotion not in LABEL_MAP:
                continue
            if _actor_to_split(actor_id) != self.split:
                continue
            samples.append({
                "stem"    : stem,
                "actor_id": actor_id,
                "emotion" : emotion,
                "label"   : LABEL_MAP[emotion],
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        s = self.samples[idx]
        stem = s["stem"]

        # ── audio spectrogram  [1, 128, 128] ──────────────────────────────
        audio_path = os.path.join(self.audio_dir, f"{stem}.npy")
        audio = torch.from_numpy(np.load(audio_path).astype(np.float32))
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)   # [H, W] -> [1, H, W]
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        # ── video frames  [3, 3, 224, 224] or [N, 3, 224, 224] ───────────
        frames_path = os.path.join(self.frames_dir, f"{stem}.npy")
        video = torch.from_numpy(np.load(frames_path).astype(np.float32))
        if self.video_transform is not None:
            video = self.video_transform(video)

        inputs = {"audio": audio, "video": video}
        return inputs, s["label"]

    def get_split_stats(self) -> Dict[str, int]:
        """Return class distribution for this split."""
        counts: Dict[str, int] = {k: 0 for k in LABEL_MAP}
        for s in self.samples:
            counts[s["emotion"]] += 1
        return counts


def cremad_collate_fn(batch):
    """Collate (inputs_dict, label) pairs into batched tensors."""
    inputs = {
        "audio": torch.stack([b[0]["audio"] for b in batch]),
        "video": torch.stack([b[0]["video"] for b in batch]),
    }
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return inputs, labels


def build_cremad_loaders(root: str, batch_size: int = 32, num_workers: int = 0,
                         audio_transform=None, video_transform=None) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders for CREMA-D.

    Parameters
    ----------
    root : str
        Preprocessed CREMA-D root directory
    batch_size : int
    num_workers : int

    Returns
    -------
    dict with keys 'train', 'val', 'test'
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CREMADDataset(root, split=split,
                           audio_transform=audio_transform,
                           video_transform=video_transform)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=cremad_collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders
