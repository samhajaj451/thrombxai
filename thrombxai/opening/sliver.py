from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from thrombxai.blending import roving
from thrombxai.opening.bale import subject_level_split
from thrombxai.order import DataConfig
from thrombxai.yarn import Batch


def synthesize(
    num_subjects: int, time_steps: int, num_features: int, positive_rate: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(num_subjects) < positive_rate).astype(np.float64)
    time = np.linspace(0.0, 1.0, time_steps)
    features = np.zeros((num_subjects, time_steps, num_features), dtype=np.float64)
    for i in range(num_subjects):
        base = rng.normal(0.0, 1.0, size=(time_steps, num_features))
        drift = np.outer(time, rng.normal(0.0, 0.3, size=num_features))
        signal = base + drift
        if labels[i] > 0.5:
            ramp = np.clip((time - 0.4) * 2.5, 0.0, 1.0)
            signal[:, 6] += 1.6 * ramp
            signal[:, 1] -= 1.2 * ramp
            signal[:, 10] += 0.9 * ramp
            signal[:, 14] += 1.1 * ramp
            signal[:, 17] += 0.8 * ramp
        features[i] = signal
    subjects = np.arange(num_subjects, dtype=np.int64)
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(subjects, dtype=torch.long),
    )


class CohortDataset(Dataset[Batch]):
    def __init__(
        self, features: torch.Tensor, labels: torch.Tensor, subjects: torch.Tensor
    ) -> None:
        self.features = features
        self.labels = labels
        self.subjects = subjects

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Batch:
        return Batch(
            signal=self.features[index],
            label=self.labels[index],
            subject=self.subjects[index],
        )


@dataclass
class Cohort:
    train: DataLoader[Batch]
    val: DataLoader[Batch]
    test: DataLoader[Batch]
    standardizer: roving.Standardizer
    mask: torch.Tensor


def _subset(
    features: torch.Tensor, labels: torch.Tensor, subjects: torch.Tensor, ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = torch.tensor([int(s) in set(ids) for s in subjects.tolist()], dtype=torch.bool)
    return features[keep], labels[keep], subjects[keep]


def build_cohort(cfg: DataConfig, seed: int) -> Cohort:
    features, labels, subjects = synthesize(
        cfg.num_subjects, cfg.time_steps, cfg.num_features, cfg.positive_rate, seed
    )
    split = subject_level_split(
        subjects.tolist(), (cfg.train_frac, cfg.val_frac, cfg.test_frac), seed
    )
    tr = _subset(features, labels, subjects, split.train)
    va = _subset(features, labels, subjects, split.val)
    te = _subset(features, labels, subjects, split.test)

    standardizer = roving.Standardizer.fit(tr[0])
    mask = torch.tensor(
        roving.modality_mask(cfg.modalities, cfg.hrv_features, cfg.temp_features),
        dtype=torch.float32,
    )

    def _prepare(block: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> CohortDataset:
        normalized = roving.apply_mask(standardizer.transform(block[0]), mask)
        return CohortDataset(normalized, block[1], block[2])

    train_set = _prepare(tr)
    sampler: WeightedRandomSampler | None = None
    shuffle = True
    if cfg.oversample_minority and train_set.labels.numel() > 0:
        positive = float(train_set.labels.mean().item())
        if 0.0 < positive < 1.0:
            weights = torch.where(
                train_set.labels > 0.5,
                torch.tensor(1.0 / positive),
                torch.tensor(1.0 / (1.0 - positive)),
            )
            sampler = WeightedRandomSampler(
                weights.tolist(), num_samples=len(train_set), replacement=True
            )
            shuffle = False

    train_loader: DataLoader[Batch] = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader: DataLoader[Batch] = DataLoader(
        _prepare(va), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader: DataLoader[Batch] = DataLoader(
        _prepare(te), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return Cohort(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        standardizer=standardizer,
        mask=mask,
    )
