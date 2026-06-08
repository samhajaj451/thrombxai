from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from thrombxai.blending import roving
from thrombxai.carding import pulse, thermal
from thrombxai.order import DataConfig
from thrombxai.yarn import Split


def subject_level_split(subjects: list[int], fracs: tuple[float, float, float], seed: int) -> Split:
    unique = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled = [unique[int(i)] for i in order]
    n = len(shuffled)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return Split(train=sorted(train), val=sorted(val), test=sorted(test))


def record_to_features(ecg: np.ndarray, temp: np.ndarray, cfg: DataConfig) -> torch.Tensor:
    hrv = pulse.hrv_matrix(ecg.astype(np.float64), cfg.signal)
    steps = hrv.shape[0] if hrv.shape[0] > 0 else cfg.time_steps
    temp_matrix = thermal.temperature_matrix(temp.astype(np.float64), cfg.signal, steps)
    fused = roving.integrate(hrv, temp_matrix)
    tensor = torch.tensor(fused, dtype=torch.float32)
    if tensor.shape[0] < cfg.time_steps:
        pad = cfg.time_steps - tensor.shape[0]
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad))
    return tensor[: cfg.time_steps]


def read_manifest(cfg: DataConfig) -> list[dict[str, object]]:
    manifest_path = Path(cfg.root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "MIMIC-IV Waveform manifest not found; set data.synthetic=true for offline runs "
            "or run scripts/prepare_data.sh to build the manifest."
        )
    with manifest_path.open() as handle:
        records: list[dict[str, object]] = json.load(handle)
    return records
