from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from thrombxai.dyeing.mordant import FEATURE_NAMES
from thrombxai.fabric import build_model
from thrombxai.floor import hydra_entry, to_config
from thrombxai.opening import sliver
from thrombxai.spindle import get_logger, load_checkpoint, select_device, set_seed

LOGGER = get_logger("thrombxai.floor.pick")


@hydra_entry
def main(cfg: DictConfig) -> None:
    exp = to_config(cfg)
    set_seed(exp.train.seed)
    device = select_device()
    cohort = sliver.build_cohort(exp.data, exp.train.seed)
    model = build_model(exp.model).to(device)
    ckpt = Path(exp.train.ckpt_dir) / f"{exp.name}_best.pt"
    if ckpt.exists():
        model.load_state_dict(load_checkpoint(ckpt)["model"])
    model.eval()
    batch = next(iter(cohort.test))
    with torch.no_grad():
        output = model(batch["signal"][:1].to(device))
    risk = float(output.risk.item())
    band = "high" if risk > 0.7 else "moderate" if risk >= 0.3 else "low"
    LOGGER.info("risk=%.4f band=%s", risk, band)
    spatial = output.spatial_attention.squeeze(0)
    top = torch.argsort(spatial, descending=True)[:3].tolist()
    for rank, index in enumerate(top, start=1):
        LOGGER.info("driver_%d=%s weight=%.4f", rank, FEATURE_NAMES[index], float(spatial[index]))


if __name__ == "__main__":
    main()
