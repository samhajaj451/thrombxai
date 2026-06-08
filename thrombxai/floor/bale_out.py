from __future__ import annotations

from omegaconf import DictConfig

from thrombxai.fabric import build_edge_model, build_model
from thrombxai.floor import hydra_entry, to_config
from thrombxai.loom.shuttle import Trainer
from thrombxai.opening import sliver
from thrombxai.pressing import baling, calender, seam
from thrombxai.spindle import get_logger, select_device, set_seed

LOGGER = get_logger("thrombxai.floor.bale_out")


@hydra_entry
def main(cfg: DictConfig) -> None:
    exp = to_config(cfg)
    set_seed(exp.train.seed)
    device = select_device()
    cohort = sliver.build_cohort(exp.data, exp.train.seed)

    teacher = build_model(exp.model)
    Trainer(teacher, exp, device).fit(cohort.train, cohort.val)

    student = build_edge_model()
    student = calender.distill(teacher, student, cohort.train, exp.edge, device)

    quantized = baling.quantize_model(student, exp.edge.quant_bits, exp.edge.quant_symmetric)
    trainer = Trainer(student, exp, device)
    student_metrics = trainer.evaluate(cohort.test)
    quant_trainer = Trainer(quantized, exp, device)
    quant_metrics = quant_trainer.evaluate(cohort.test)

    size = seam.model_size_mb(quantized, exp.edge.quant_bits)
    latency = seam.estimate_latency_ms(quantized, exp.data.time_steps, exp.edge.quant_bits)
    LOGGER.info("student_test_auroc=%.4f", student_metrics["auroc"])
    LOGGER.info("quant_bits=%d quant_test_auroc=%.4f", exp.edge.quant_bits, quant_metrics["auroc"])
    LOGGER.info("edge_size_mb=%.3f est_latency_ms=%.1f", size, latency)


if __name__ == "__main__":
    main()
