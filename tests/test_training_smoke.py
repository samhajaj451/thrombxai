from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import thrombxai
from thrombxai.fabric import build_model
from thrombxai.loom.shuttle import Trainer
from thrombxai.opening import sliver
from thrombxai.order import ExperimentConfig, register_configs

CONF_DIR = str(Path(thrombxai.__file__).resolve().parent.parent / "conf")


def _smoke_config() -> ExperimentConfig:
    register_configs()
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=["experiment=_smoke"])
    return cast(ExperimentConfig, OmegaConf.to_object(cfg))


def test_smoke_training_reduces_loss() -> None:
    exp = _smoke_config()
    assert exp.name == "_smoke"
    cohort = sliver.build_cohort(exp.data, exp.train.seed)
    model = build_model(exp.model)
    trainer = Trainer(model, exp, torch.device("cpu"))
    history = trainer.fit(cohort.train, cohort.val)
    assert len(history.train_loss) == exp.train.epochs
    assert history.train_loss[-1] <= history.train_loss[0] + 1e-3
