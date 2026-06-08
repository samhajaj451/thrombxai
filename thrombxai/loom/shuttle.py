from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from thrombxai.dyeing import vat
from thrombxai.dyeing.mordant import relevance_vector
from thrombxai.fabric import ThrombXAI
from thrombxai.finishing import objective
from thrombxai.inspection import gauge
from thrombxai.loom.warp_beam import build_optimizer, build_scheduler
from thrombxai.order import ExperimentConfig
from thrombxai.spindle import atomic_save, get_logger, run_backward
from thrombxai.yarn import Batch

LOGGER = get_logger("thrombxai.loom")


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)
    best_auroc: float = 0.0
    best_epoch: int = 0


class Trainer:
    def __init__(self, model: ThrombXAI, cfg: ExperimentConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.relevance = relevance_vector(cfg.crga.degree_smoothing).to(device)
        self.optimizer = build_optimizer(model, cfg.train.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.train.epochs, cfg.train.cosine_eta_min)

    def _loss(self, batch: Batch, epoch: int) -> torch.Tensor:
        signal = batch["signal"].to(self.device)
        label = batch["label"].to(self.device)
        output = self.model(signal)
        loss = objective.prediction_loss(output.logit, label, self.cfg.objective)
        if self.cfg.crga.enabled:
            weight = vat.kl_warmup(self.cfg.crga.lambda_kl, epoch, self.cfg.crga.warmup_epochs)
            loss = loss + vat.crga_loss(
                output.spatial_attention,
                output.temporal_attention,
                self.relevance,
                weight,
                self.cfg.crga.lambda_smooth,
            )
        loss = loss + objective.l2_penalty(self.model, self.cfg.objective.l2_reg)
        return loss

    def _run_epoch(self, loader: DataLoader[Batch], epoch: int) -> float:
        self.model.train()
        accum = max(1, self.cfg.train.grad_accum)
        running = 0.0
        steps = 0
        self.optimizer.zero_grad()
        for index, batch in enumerate(loader):
            loss = self._loss(batch, epoch) / accum
            run_backward(loss)
            if (index + 1) % accum == 0:
                if self.cfg.train.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
            running += float(loss.item()) * accum
            steps += 1
        self.optimizer.step()
        self.optimizer.zero_grad()
        return running / max(1, steps)

    @torch.no_grad()
    def predict(self, loader: DataLoader[Batch]) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        scores: list[float] = []
        labels: list[float] = []
        for batch in loader:
            output = self.model(batch["signal"].to(self.device))
            scores.extend(output.risk.cpu().tolist())
            labels.extend(batch["label"].tolist())
        return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.float64)

    def evaluate(self, loader: DataLoader[Batch]) -> dict[str, float]:
        scores, labels = self.predict(loader)
        if scores.size == 0:
            return {"auroc": 0.0}
        return gauge.report(scores, labels, self.cfg.train.threshold)

    def fit(self, train_loader: DataLoader[Batch], val_loader: DataLoader[Batch]) -> History:
        history = History()
        patience = 0
        ckpt = Path(self.cfg.train.ckpt_dir) / f"{self.cfg.name}_best.pt"
        for epoch in range(self.cfg.train.epochs):
            loss = self._run_epoch(train_loader, epoch)
            self.scheduler.step()
            metrics = self.evaluate(val_loader)
            auroc = metrics.get("auroc", 0.0)
            history.train_loss.append(loss)
            history.val_auroc.append(auroc)
            if epoch % self.cfg.train.log_every == 0:
                LOGGER.info("epoch=%d loss=%.4f val_auroc=%.4f", epoch, loss, auroc)
            if auroc > history.best_auroc:
                history.best_auroc = auroc
                history.best_epoch = epoch
                patience = 0
                atomic_save(
                    {
                        "model": self.model.state_dict(),
                        "config_name": self.cfg.name,
                        "seed": self.cfg.train.seed,
                        "epoch": epoch,
                        "val_auroc": auroc,
                    },
                    ckpt,
                )
            else:
                patience += 1
                if patience >= self.cfg.train.early_stop_patience:
                    LOGGER.info("early stop at epoch=%d", epoch)
                    break
        return history
