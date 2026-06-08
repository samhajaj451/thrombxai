# Deviations

Every intentional departure from the manuscript, with its paper anchor and reason.

## D1. INT4 edge path is simulated
- Anchor: Table VI (INT4 row), §V.D.
- PyTorch has no native 4-bit tensor type. `thrombxai/pressing/baling.py` applies symmetric 4-bit
  fake-quantization to the weights (round-to-grid with a per-tensor scale) so the INT4 configuration
  is exercisable. The 8-bit path is real symmetric quantization.

## D2. L2 term realised through the optimizer
- Anchor: Eq. (24) lambda-reg term; §IV.D weight decay 1e-4.
- The combined objective in Eq. (24) carries an explicit L2 term, and §IV.D also reports AdamW weight
  decay 1e-4. To avoid double counting, the L2 penalty is realised through AdamW weight decay by
  default (`objective.l2_reg = 0.0`). An explicit penalty remains available in
  `thrombxai/finishing/objective.py` (`l2_penalty`) for runs that prefer it.

## D3. Single-modality ablations via masking
- Anchor: Table V (Single Modality rows).
- The HRV-only and temperature-only ablations keep the 19-feature tensor shape and zero the excluded
  modality through `thrombxai/blending/roving.py` (`modality_mask`) rather than rebuilding a smaller
  input, so one architecture serves every ablation.

## D4. Offline data substitute
- Anchor: §IV.A, Table I.
- MIMIC-IV Waveform records are credentialed and not redistributable. Offline runs use a
  deterministic synthetic cohort (`thrombxai/opening/sliver.py`); the real records load through the
  manifest path in `thrombxai/opening/bale.py` when `data.synthetic=false`.

## D5. Edge latency and power are estimates
- Anchor: Table VI (latency, power columns).
- On-device latency and power require the ARM Cortex-M7 target. `thrombxai/pressing/seam.py` reports
  a parameter-count-based latency estimate and a bit-width-scaled size; absolute on-device numbers
  must be measured on hardware.

## D6. Feature-embedding and spatial-attention widths
- Anchor: Eq. (16)-(18), Table II.
- Table II fixes the hidden dimension, head count, and MLP width but not the spatial-attention inner
  width or the per-feature embedding width. These are set to 64 and 256 (full) and 32 and 64 (edge)
  in `thrombxai/order.py`; total parameter counts therefore approximate the Table II figures.
