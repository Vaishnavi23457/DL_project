"""
mb.cmu_mosei_data
=================
CMU-MOSEI dataset loader for audio + video + text trimodal learning.

CMU-MOSEI: CMU Multimodal Opinion Sentiment and Emotion Intensity
  - 23,453 utterances from 1,000 YouTube speakers
  - 6 emotion labels: happiness, sadness, anger, fear, disgust, surprise
  - Pre-aligned features available via CMU-MultimodalSDK

Expected directory layout (pre-extracted SDK features):
    mosei_root/
        audio.npy      <- [N, T_a, 74]  COVAREP acoustic features
        video.npy      <- [N, T_v, 35]  FACET visual features
        text.npy       <- [N, T_t, 300] GloVe word embeddings
        labels.npy     <- [N, 6]        multi-label emotion intensities
        split.npy      <- [N]           0=train, 1=val, 2=test
        word2id.json   <- vocabulary mapping (optional)

Alternatively, raw SDK loading is supported via CMUMultimodalSDK
if the raw .csd files are available.

Usage
-----
    from mb.cmu_mosei_data import CMUMOSEIDataset, build_mosei_loaders
    train_ds = CMUMOSEIDataset(root, split="train")
    loaders  = build_mosei_loaders(root, batch_size=32)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EMOTION_LABELS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
NUM_CLASSES = 6
SPLIT_MAP = {"train": 0, "val": 1, "test": 2}


class CMUMOSEIDataset(Dataset):
    """
    CMU-MOSEI dataset returning (inputs_dict, label) tuples.

    For each sample, the dominant emotion (highest intensity) is used
    as the class label for 6-way classification.

    Parameters
    ----------
    root : str
        Path to directory with pre-extracted .npy feature files
    split : str
        One of 'train', 'val', 'test'
    max_audio_len : int
        Pad/truncate audio sequence to this length
    max_video_len : int
        Pad/truncate video sequence to this length
    max_text_len : int
        Pad/truncate text sequence to this length
    """

    modalities = ["audio", "video", "text"]

    def __init__(self, root: str, split: str = "train",
                 max_audio_len: int = 200,
                 max_video_len: int = 100,
                 max_text_len: int = 50):
        assert split in ("train", "val", "test")
        self.root = root
        self.split = split
        self.max_audio_len = max_audio_len
        self.max_video_len = max_video_len
        self.max_text_len  = max_text_len

        self._load_data()

    def _load_data(self):
        split_idx = SPLIT_MAP[self.split]

        audio  = np.load(os.path.join(self.root, "audio.npy"),  allow_pickle=True)
        video  = np.load(os.path.join(self.root, "video.npy"),  allow_pickle=True)
        text   = np.load(os.path.join(self.root, "text.npy"),   allow_pickle=True)
        labels = np.load(os.path.join(self.root, "labels.npy"), allow_pickle=True)
        splits = np.load(os.path.join(self.root, "split.npy"),  allow_pickle=True)

        mask = splits == split_idx
        self.audio  = audio[mask]
        self.video  = video[mask]
        self.text   = text[mask]
        self.labels = labels[mask]   # [N, 6] emotion intensities

    def __len__(self) -> int:
        return len(self.labels)

    def _pad_or_truncate(self, arr: np.ndarray, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or truncate a sequence [T, D] to [max_len, D].
        Returns (tensor [max_len, D], mask [max_len]).
        """
        T, D = arr.shape
        out  = np.zeros((max_len, D), dtype=np.float32)
        mask = np.zeros(max_len, dtype=np.float32)
        L = min(T, max_len)
        out[:L]  = arr[:L]
        mask[:L] = 1.0
        return torch.from_numpy(out), torch.from_numpy(mask)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        audio_seq, audio_mask = self._pad_or_truncate(
            self.audio[idx].astype(np.float32), self.max_audio_len)
        video_seq, video_mask = self._pad_or_truncate(
            self.video[idx].astype(np.float32), self.max_video_len)
        text_seq,  text_mask  = self._pad_or_truncate(
            self.text[idx].astype(np.float32),  self.max_text_len)

        # dominant emotion as class label
        label = int(np.argmax(self.labels[idx]))

        inputs = {
            "audio"      : audio_seq,    # [max_audio_len, 74]
            "audio_mask" : audio_mask,   # [max_audio_len]
            "video"      : video_seq,    # [max_video_len, 35]
            "video_mask" : video_mask,
            "text"       : text_seq,     # [max_text_len, 300]
            "text_mask"  : text_mask,
        }
        return inputs, label

    def get_split_stats(self) -> Dict[str, int]:
        """Return class distribution for this split."""
        dominant = np.argmax(self.labels, axis=1)
        return {EMOTION_LABELS[i]: int((dominant == i).sum()) for i in range(NUM_CLASSES)}


def mosei_collate_fn(batch):
    """Collate (inputs_dict, label) pairs into batched tensors."""
    keys = list(batch[0][0].keys())
    inputs = {k: torch.stack([b[0][k] for b in batch]) for k in keys}
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return inputs, labels


def build_mosei_loaders(root: str, batch_size: int = 32, num_workers: int = 0,
                        max_audio_len: int = 200, max_video_len: int = 100,
                        max_text_len: int = 50) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders for CMU-MOSEI.

    Parameters
    ----------
    root : str
        Directory with pre-extracted .npy feature files
    batch_size : int
    num_workers : int

    Returns
    -------
    dict with keys 'train', 'val', 'test'
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CMUMOSEIDataset(root, split=split,
                             max_audio_len=max_audio_len,
                             max_video_len=max_video_len,
                             max_text_len=max_text_len)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=mosei_collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders
