from __future__ import annotations

import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def select_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and prefer in {"auto", "cuda"}:
        return torch.device("cuda")
    return torch.device("cpu")


def atomic_save(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(Path(path), map_location=map_location)
    return payload


def run_backward(loss: torch.Tensor) -> None:
    cast(Any, loss).backward()
