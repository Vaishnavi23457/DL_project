"""
bml.data
========
Text + Image data pipeline.

  SimpleTokenizer            : word-level tokenizer trained on your CSV (saved as vocab.json)
  TextImageCSVDataset        : CSV with columns  image_path,text,label   (names configurable)
  SyntheticTextImageDataset  : on-the-fly synthetic data (text easy / image hard) for smoke tests & demos
  collate_fn                 : pads text, stacks images  -> inputs dict + labels
  build_dataloaders(args)    : everything wired together for train.py / eval.py

Batch format produced everywhere:
    inputs = {"image": FloatTensor[B,3,H,W],
              "text" : {"input_ids": LongTensor[B,L], "attention_mask": LongTensor[B,L]}}
    labels = LongTensor[B]
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PAD, UNK, CLS = 0, 1, 2
_WORD_RE = re.compile(r"[A-Za-z0-9_']+|[^\sA-Za-z0-9_']")


# ------------------------------------------------------------------------------ tokenizer
class SimpleTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None, lower: bool = True):
        self.lower = lower
        self.vocab: Dict[str, int] = vocab or {"[PAD]": PAD, "[UNK]": UNK, "[CLS]": CLS}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _split(self, text: str) -> List[str]:
        if self.lower:
            text = text.lower()
        return _WORD_RE.findall(text)

    def fit(self, texts: Sequence[str], min_freq: int = 1, max_vocab: int = 30000) -> "SimpleTokenizer":
        cnt = Counter()
        for t in texts:
            cnt.update(self._split(t))
        for w, c in cnt.most_common():
            if c < min_freq or len(self.vocab) >= max_vocab:
                break
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
        return self

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        ids = [CLS] + [self.vocab.get(w, UNK) for w in self._split(text)]
        return ids[:max_len]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"lower": self.lower, "vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(vocab=d["vocab"], lower=d.get("lower", True))


class HFTokenizerWrapper:
    """Same .encode()/.vocab_size interface backed by a HuggingFace tokenizer."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        return self.tok(text, truncation=True, max_length=max_len)["input_ids"]

    def save(self, path: str) -> None:  # nothing to save; model name is in config.json
        pass


# ------------------------------------------------------------------------------ transforms
def build_transforms(img_size: int = 224, train: bool = True, normalize: bool = True):
    from torchvision import transforms as T
    norm = [T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] if normalize else []
    if train:
        return T.Compose([T.RandomResizedCrop(img_size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), *norm])
    return T.Compose([T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size), T.ToTensor(), *norm])


# ------------------------------------------------------------------------------ CSV dataset
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TextImageCSVDataset(Dataset):
    """
    CSV columns (default names): image, text, label.  `label` may be a string (mapped via label_map).

    Any modality column can be omitted (or left empty) and is then simply not produced:
      * no `image` column  -> no image input   (e.g. MELD: text + audio)
      * `--col_audio audio` column with a .npy path -> adds an `audio` vector input
        (also supports `<npz file>::<key>` to read from one packed .npz per split)
    The dataset exposes `.modalities` / `.audio_dim` so train.py can build the matching encoders.
    """

    def __init__(self, csv_path: str, img_root: str, tokenizer, label_map: Dict[str, int], transform,
                 col_image: str = "image", col_text: str = "text", col_label: str = "label", max_len: int = 64,
                 col_audio: Optional[str] = None):
        self.rows = read_csv_rows(csv_path)
        self.img_root = img_root or ""
        self.tok = tokenizer
        self.label_map = label_map
        self.transform = transform
        self.ci, self.ct, self.cl = col_image, col_text, col_label
        self.ca = col_audio
        self.max_len = max_len
        fields = set(self.rows[0].keys()) if self.rows else set()
        self.has_image = col_image in fields and bool(self.rows) and str(self.rows[0].get(col_image, "")).strip() != ""
        self.has_audio = bool(col_audio) and col_audio in fields and bool(self.rows) \
            and str(self.rows[0].get(col_audio, "")).strip() != ""
        self.modalities = (["image"] if self.has_image else []) + (["audio"] if self.has_audio else []) + ["text"]
        self.audio_dim: Optional[int] = None
        self._npz_cache = None
        if self.has_audio:
            self.audio_dim = int(self._load_audio(self.rows[0][self.ca]).shape[-1])

    def _abs(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.img_root, p)

    def _load_audio(self, ref: str) -> np.ndarray:
        """'vec.npy'  or  'audio.npz::train_00437' (one packed npz per split)."""
        if "::" in ref:
            npz_path, key = ref.split("::", 1)
            if self._npz_cache is None or self._npz_cache[0] != npz_path:
                self._npz_cache = (npz_path, np.load(self._abs(npz_path)))
            return self._npz_cache[1][key]
        return np.load(self._abs(ref))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        out: Dict[str, object] = {}
        if self.has_image:
            from PIL import Image
            img = Image.open(self._abs(r[self.ci])).convert("RGB")
            out["image"] = self.transform(img)
        if self.has_audio:
            out["audio"] = torch.from_numpy(self._load_audio(r[self.ca]).astype(np.float32))
        out["text_ids"] = self.tok.encode(r[self.ct], self.max_len)
        out["label"] = self.label_map[str(r[self.cl])]
        return out

    def texts(self) -> List[str]:
        return [r[self.ct] for r in self.rows]

    def labels_raw(self) -> List[str]:
        return [str(r[self.cl]) for r in self.rows]


# ------------------------------------------------------------------------------ synthetic dataset
class SyntheticTextImageDataset(Dataset):
    """
    Deterministic synthetic Text+Image classification data (C classes) designed to show the
    imbalance phenomenon:

      * each modality is *informative* for a sample only with probability p (independently):
          - text : a class keyword is inserted into a bag of random tokens      (prob text_p)
          - image: a class-specific coloured patch is drawn on Gaussian noise    (prob img_p)
        otherwise the modality carries no label information at all.
      * ceilings:  uni-modal acc ~ p + (1-p)/C          (p=0.7, C=6  ->  0.75)
                   fused acc     ~ 1 - (1-p)^2 + (1-p)^2/C  (0.7, 6 ->  0.925)

    So the fused model can only reach ~0.92 if BOTH encoders are well learnt. Under plain joint
    training one modality (whichever the optimiser finds easier: here usually the CNN/image) is
    learnt first and dominates; the other stays under-optimised and fused accuracy stalls.
    OPM / OGM lift the weak modality and the fused accuracy.
    """
    vocab_size = 400
    seq_len = 16

    def __init__(self, n: int = 2000, num_classes: int = 6, img_size: int = 64, text_p: float = 0.7,
                 img_p: float = 0.7, img_noise: float = 1.0, patch_frac: float = 0.2, seed: int = 0, split: str = "train",
                 synonyms: int = 1):
        self.n, self.C, self.S = n, num_classes, img_size
        self.text_p, self.img_p, self.img_noise = text_p, img_p, img_noise
        self.K = max(1, synonyms)   # each class has K interchangeable keyword tokens -> text is slower to learn
        self.patch = max(4, int(img_size * patch_frac))
        self.seed = seed + (0 if split == "train" else 10_000_007)
        g = np.random.RandomState(1234)
        self.palette = g.uniform(0.3, 1.0, size=(num_classes, 3)).astype(np.float32)
        grid = max(2, int(math.ceil(math.sqrt(num_classes))))
        cells = [(r, c) for r in range(grid) for c in range(grid)]
        g.shuffle(cells)
        self.cells = cells[:num_classes]
        self.grid = grid

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.RandomState((self.seed * 100_003 + i) % (2 ** 32 - 1))
        y = i % self.C
        # ---- image: noise (+ class patch if informative)
        img = rng.normal(0.0, self.img_noise, size=(3, self.S, self.S)).astype(np.float32)
        if rng.rand() < self.img_p:
            r, c = self.cells[y]
            cell = self.S // self.grid
            r0 = r * cell + rng.randint(0, max(1, cell - self.patch))
            c0 = c * cell + rng.randint(0, max(1, cell - self.patch))
            img[:, r0:r0 + self.patch, c0:c0 + self.patch] += self.palette[y][:, None, None]
        # ---- text: random tokens (+ one of the K class keywords if informative)
        kw_first = 3                                   # keyword ids: 3 .. 3+C*K-1  (class y owns ids kw_first+y*K .. +K-1)
        ids = rng.randint(kw_first + self.C * self.K, self.vocab_size, size=self.seq_len).tolist()
        if rng.rand() < self.text_p:
            ids[rng.randint(0, self.seq_len)] = kw_first + y * self.K + rng.randint(0, self.K)
        ids = [CLS] + ids
        return {"image": torch.from_numpy(img), "text_ids": ids, "label": y}

    def to_csv(self, out_dir: str, name: str = "train") -> str:
        """Materialise into PNG files + CSV (to test the CSV pipeline on your machine)."""
        from PIL import Image
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        path = os.path.join(out_dir, f"{name}.csv")
        words = [f"w{i}" for i in range(self.vocab_size)]
        for y in range(self.C):
            for k in range(self.K):
                words[3 + y * self.K + k] = f"class{y}word{k}"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "text", "label"])
            for i in range(len(self)):
                s = self[i]
                arr = s["image"].numpy()
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8).transpose(1, 2, 0)
                rel = os.path.join("images", f"{name}_{i:05d}.png")
                Image.fromarray(arr).save(os.path.join(out_dir, rel))
                text = " ".join(words[t] for t in s["text_ids"][1:])
                w.writerow([rel, text, f"cls_{s['label']}"])
        return path


# ------------------------------------------------------------------------------ collate
def collate_fn(batch: List[Dict]) -> Tuple[Dict[str, object], torch.Tensor]:
    inputs: Dict[str, object] = {}
    if "image" in batch[0]:
        inputs["image"] = torch.stack([b["image"] for b in batch], 0)
    if "audio" in batch[0]:
        inputs["audio"] = torch.stack([torch.as_tensor(b["audio"], dtype=torch.float32) for b in batch], 0)
    L = max(len(b["text_ids"]) for b in batch)
    ids = torch.full((len(batch), L), PAD, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        t = torch.tensor(b["text_ids"], dtype=torch.long)
        ids[i, : len(t)] = t
        mask[i, : len(t)] = 1
    inputs["text"] = {"input_ids": ids, "attention_mask": mask}
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return inputs, labels


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to(v, device) for v in obj)
    return obj


# ------------------------------------------------------------------------------ builder
def build_dataloaders(args, out_dir: Optional[str] = None):
    """
    Returns dict(train=DataLoader, val=DataLoader, test=DataLoader|None, num_classes, vocab_size,
                 label_map, tokenizer)
    """
    if args.data == "synthetic":
        syn_kw = dict(text_p=args.syn_text_p, img_p=args.syn_img_p, img_noise=args.syn_img_noise, seed=args.seed,
                      synonyms=getattr(args, "syn_synonyms", 1))
        tr = SyntheticTextImageDataset(args.syn_train_n, args.num_classes, args.img_size, split="train", **syn_kw)
        va = SyntheticTextImageDataset(args.syn_val_n, args.num_classes, args.img_size, split="val", **syn_kw)
        te = None
        num_classes, vocab_size = args.num_classes, SyntheticTextImageDataset.vocab_size
        label_map = {str(i): i for i in range(num_classes)}
        tokenizer = None
        modalities, audio_dim = ["image", "text"], None
    else:
        # ---- tokenizer
        train_rows = read_csv_rows(args.train_csv)
        train_texts = [r[args.col_text] for r in train_rows]
        if args.text_encoder == "hf":
            tokenizer = HFTokenizerWrapper(args.hf_model)
        elif getattr(args, "vocab_path", None) and os.path.exists(args.vocab_path):
            tokenizer = SimpleTokenizer.load(args.vocab_path)
        else:
            tokenizer = SimpleTokenizer().fit(train_texts, min_freq=args.min_freq, max_vocab=args.max_vocab)
        # ---- labels
        if getattr(args, "label_map_path", None) and os.path.exists(args.label_map_path):
            with open(args.label_map_path) as f:
                label_map = json.load(f)
        else:
            classes = sorted({str(r[args.col_label]) for r in train_rows},
                             key=lambda s: (not s.lstrip("-").isdigit(), int(s) if s.lstrip("-").isdigit() else s))
            label_map = {c: i for i, c in enumerate(classes)}
        num_classes, vocab_size = len(label_map), tokenizer.vocab_size
        mk = lambda csv_path, train: TextImageCSVDataset(csv_path, args.img_root, tokenizer, label_map,
                                                         build_transforms(args.img_size, train),
                                                         args.col_image, args.col_text, args.col_label, args.max_len,
                                                         col_audio=getattr(args, "col_audio", None))
        base = mk(args.train_csv, True)
        modalities, audio_dim = list(base.modalities), base.audio_dim
        tr, va, te = base, None, None
        va = mk(args.val_csv, False) if args.val_csv else None
        te = mk(args.test_csv, False) if getattr(args, "test_csv", None) else None
        if va is None:  # no val csv -> hold out 10 % of train
            n_val = max(1, int(0.1 * len(tr)))
            g = torch.Generator().manual_seed(args.seed)
            tr, va = torch.utils.data.random_split(tr, [len(tr) - n_val, n_val], generator=g)
        if out_dir:
            tokenizer.save(os.path.join(out_dir, "vocab.json"))
            with open(os.path.join(out_dir, "label_map.json"), "w") as f:
                json.dump(label_map, f, indent=2)

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn,
              pin_memory=torch.cuda.is_available())
    loaders = {
        "train": DataLoader(tr, shuffle=True, drop_last=True, **kw),
        "val": DataLoader(va, shuffle=False, **kw),
        "test": DataLoader(te, shuffle=False, **kw) if te is not None else None,
    }
    return dict(loaders=loaders, num_classes=num_classes, vocab_size=vocab_size, label_map=label_map,
                tokenizer=tokenizer, modalities=modalities, audio_dim=audio_dim)
