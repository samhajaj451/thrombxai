from __future__ import annotations

from collections.abc import Callable
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from thrombxai.order import ExperimentConfig, register_configs

register_configs()

__all__ = ["hydra_entry", "to_config"]


def hydra_entry(fn: Callable[[DictConfig], None]) -> Callable[[], None]:
    decorated = hydra.main(version_base=None, config_path="../../conf", config_name="config")(fn)
    return cast(Callable[[], None], decorated)


def to_config(cfg: DictConfig) -> ExperimentConfig:
    return cast(ExperimentConfig, OmegaConf.to_object(cfg))
