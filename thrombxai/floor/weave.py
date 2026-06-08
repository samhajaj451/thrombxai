from __future__ import annotations

from omegaconf import DictConfig

from thrombxai.fabric import build_model, count_parameters
from thrombxai.floor import hydra_entry, to_config
from thrombxai.loom.shuttle import Trainer
from thrombxai.opening import sliver
from thrombxai.spindle import get_logger, select_device, set_seed

LOGGER = get_logger("thrombxai.floor.weave")


@hydra_entry
def main(cfg: DictConfig) -> None:
    exp = to_config(cfg)
    set_seed(exp.train.seed)
    device = select_device()
    cohort = sliver.build_cohort(exp.data, exp.train.seed)
    model = build_model(exp.model)
    LOGGER.info("experiment=%s parameters=%d device=%s", exp.name, count_parameters(model), device)
    trainer = Trainer(model, exp, device)
    history = trainer.fit(cohort.train, cohort.val)
    LOGGER.info("done best_val_auroc=%.4f best_epoch=%d", history.best_auroc, history.best_epoch)


if __name__ == "__main__":
    main()
