from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig

from thrombxai.fabric import build_edge_model
from thrombxai.floor import hydra_entry, to_config
from thrombxai.pressing import seam
from thrombxai.spindle import get_logger, set_seed

LOGGER = get_logger("thrombxai.floor.ship")


@hydra_entry
def main(cfg: DictConfig) -> None:
    exp = to_config(cfg)
    set_seed(exp.train.seed)
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    student = build_edge_model().eval()
    example = torch.randn(1, exp.data.time_steps, exp.data.num_features)
    onnx_path = out_dir / f"{exp.name}_edge.onnx"

    class Wrapped(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = student

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logit: torch.Tensor = self.inner(x).logit
            return logit

    torch.onnx.export(
        Wrapped(), (example,), str(onnx_path), input_names=["signal"], output_names=["logit"]
    )

    plan = {
        "target": "ARM Cortex-M7",
        "runtime": "TensorFlow Lite Micro",
        "quant_bits": exp.edge.quant_bits,
        "operator_fusion": ["conv+bn+relu"],
        "size_mb": round(seam.model_size_mb(student, exp.edge.quant_bits), 4),
        "est_latency_ms": round(
            seam.estimate_latency_ms(student, exp.data.time_steps, exp.edge.quant_bits), 1
        ),
    }
    (out_dir / f"{exp.name}_deployment.json").write_text(json.dumps(plan, indent=2))
    LOGGER.info("exported onnx=%s plan=%s", onnx_path.name, plan)


if __name__ == "__main__":
    main()
