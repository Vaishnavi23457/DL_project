"""Small helpers: seeding, meters, json io, logger, checkpointing."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.n)


class Logger:
    """print + append to a log file."""

    def __init__(self, path: str | None = None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def __call__(self, *args: Any) -> None:
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(time.strftime("[%H:%M:%S] ") + msg + "\n")


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def save_checkpoint(path: str, model: torch.nn.Module, optimizer=None, epoch: int = 0, extra: Dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ck = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        ck["optimizer"] = optimizer.state_dict()
    if extra:
        ck.update(extra)
    torch.save(ck, path)


def fmt(d: Dict[str, float], keys=None, nd: int = 4) -> str:
    keys = keys or list(d.keys())
    return " ".join(f"{k}={d[k]:.{nd}f}" if isinstance(d[k], float) else f"{k}={d[k]}" for k in keys if k in d)
