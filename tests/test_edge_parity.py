from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from thrombxai.fabric import build_edge_model
from thrombxai.opening.sliver import CohortDataset, synthesize
from thrombxai.order import EdgeConfig
from thrombxai.pressing import baling, calender, seam


def test_quantization_preserves_shape_and_shrinks() -> None:
    torch.manual_seed(0)
    model = build_edge_model().eval()
    x = torch.randn(4, 40, 19)
    with torch.no_grad():
        base = model(x).logit
    quantized = baling.quantize_model(model, 8, True)
    with torch.no_grad():
        out8 = quantized(x).logit
    assert out8.shape == base.shape
    assert torch.allclose(out8, base, atol=0.5)
    assert seam.model_size_mb(model, 8) < seam.model_size_mb(model, 32)


def test_int4_quantization_stays_finite() -> None:
    model = build_edge_model().eval()
    x = torch.randn(2, 40, 19)
    quantized = baling.quantize_model(model, 4, True)
    with torch.no_grad():
        out = quantized(x).logit
    assert torch.isfinite(out).all()


def test_distillation_runs_a_step() -> None:
    torch.manual_seed(0)
    features, labels, subjects = synthesize(12, 40, 19, 0.5, seed=2)
    loader: DataLoader = DataLoader(CohortDataset(features, labels, subjects), batch_size=4)
    teacher = build_edge_model()
    student = build_edge_model()
    cfg = EdgeConfig(distill_epochs=1, student_hidden=64)
    trained = calender.distill(teacher, student, loader, cfg, torch.device("cpu"))
    assert sum(p.numel() for p in trained.parameters()) > 0
