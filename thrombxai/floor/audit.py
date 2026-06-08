from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from thrombxai.dyeing.mordant import clinical_rank
from thrombxai.fabric import build_model
from thrombxai.floor import hydra_entry, to_config
from thrombxai.inspection import burling, gauge
from thrombxai.loom.shuttle import Trainer
from thrombxai.opening import sliver
from thrombxai.spindle import get_logger, load_checkpoint, select_device, set_seed

LOGGER = get_logger("thrombxai.floor.audit")


@hydra_entry
def main(cfg: DictConfig) -> None:
    exp = to_config(cfg)
    set_seed(exp.train.seed)
    device = select_device()
    cohort = sliver.build_cohort(exp.data, exp.train.seed)
    model = build_model(exp.model)
    ckpt = Path(exp.train.ckpt_dir) / f"{exp.name}_best.pt"
    if ckpt.exists():
        model.load_state_dict(load_checkpoint(ckpt)["model"])
    trainer = Trainer(model, exp, device)
    scores, labels = trainer.predict(cohort.test)
    metrics = gauge.report(scores, labels, exp.train.threshold)
    lower, upper = gauge.bootstrap_ci(scores, labels, gauge.roc_auc, 200, 0.05, exp.train.seed)
    for name, value in metrics.items():
        LOGGER.info("%s=%.4f", name, value)
    LOGGER.info("auroc_ci=[%.4f, %.4f]", lower, upper)

    signal = next(iter(cohort.test))["signal"].to(device)
    with torch.no_grad():
        weights = model(signal).spatial_attention.mean(dim=0).cpu().numpy().astype(np.float64)
    ranks = clinical_rank()
    LOGGER.info("comprehensiveness@20=%.4f", burling.comprehensiveness(model, signal, 0.2))
    LOGGER.info("sufficiency@20=%.4f", burling.sufficiency(model, signal, 0.2))
    LOGGER.info("attention_overlap@5=%.4f", burling.attention_overlap(weights, ranks, 5))
    LOGGER.info("ranking_spearman=%.4f", burling.ranking_correlation(weights, ranks))


if __name__ == "__main__":
    main()
