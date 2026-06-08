# Project Context

This release accompanies the ThrombXAI study: an explainable temporal-spatial attention
framework that reads heart-rate-variability and skin-temperature streams from wearable sensors and
issues early warnings of thrombotic events for connected-health deployment.

```
project_name       : thrombxai                                  [HIGH]
domain             : wearable physiological time-series —        [HIGH]
                     thrombotic-event early warning + explainability
framework          : PyTorch 2.1 + Hydra structured configs      [HIGH]
venue              : IEEE Transactions on Consumer Electronics   [HIGH]
primary_datasets   : 1 dataset (MIMIC-IV Waveform, see §6)       [HIGH]
compute_target     : 1x NVIDIA A100 80GB; edge ARM Cortex-M7     [HIGH]
hparams_reference  : Methods §III.E, §IV.D, Table II             [HIGH]
extra_signals      : INT4 path simulated; N=156 prospective set
NEEDS_USER_DECISION: 0
```

## 1. project_name
`thrombxai`. Derived from the model name ThrombXAI used throughout the manuscript (Methods §III).

## 2. supplementary materials
None found alongside the manuscript. All numbers below come from the main text and its tables.

## 3. domain
Wearable physiological signal modelling for thrombotic-event early warning, with an explicit
explainability objective (attention attribution plus a clinical-knowledge prior). Source: Abstract;
Methods §III.A.

## 4. framework
PyTorch 2.1 (Implementation Details §IV.D states the model "is executed using PyTorch 2.1"). The
edge deployment is described as TensorFlow Lite Micro on an ARM Cortex-M7 (§III.D, §IV.D). This
release implements the full training and evaluation stack in PyTorch and emits an ONNX graph plus a
TFLite-Micro deployment descriptor for the edge target. Configuration and command-line handling use
Hydra structured configs.

## 5. venue
IEEE Transactions on Consumer Electronics. Inferred from the Index Terms, the consumer-electronics
framing, and the dense set of `IEEE Trans. Consum. Electron.` references.

## 6. primary_datasets
- **MIMIC-IV Waveform Database (mimic4wdb)** — ICU ECG waveforms and vital signs from Beth Israel
  Deaconess Medical Center, hosted by PhysioNet (https://physionet.org/content/mimic4wdb/).
  Cohort: 2,847 patients, 4,271 four-hour windows, 21.9% positive (Table I). Subject-level split
  70/15/15 → 1,993 / 427 / 427 patients (§IV.A.2).
  Access: PhysioNet credentialed access; license is the PhysioNet Credentialed Health Data License
  (verify the current version on the dataset landing page). The records are restricted, so this
  release ships a deterministic synthetic cohort for offline runs and a manifest-driven loader for
  the real records.

## 7. compute_target
Training: a single NVIDIA A100 80GB GPU; about 4 hours for the full model and about 1 hour to
distill the edge model (§IV.D). Five random seeds are reported. Edge inference: ARM Cortex-M7 with
512KB SRAM, TensorFlow Lite Micro, target latency below 850 ms (§III.D, §V.D).

## 8. hparams_reference
Methods §III.E and Implementation Details §IV.D, with architecture sizes in Table II. AdamW,
lr 1e-3, weight decay 1e-4, cosine annealing, 100 epochs, batch size 64, gradient accumulation 4,
early stopping patience 15 on validation AUROC, focal loss with focusing parameter 2 plus minority
oversampling. TSAAN: 4 dilated conv layers, hidden dimension 256, kernel size 3, dropout 0.3, 8
attention heads, attention dimension 32. CRGA: KL weight 0.1, smoothness weight 0.05, warmup 20
epochs. Edge: distillation temperature 4, student hidden dimension 64, 8-bit symmetric quantization.

## 9. extra_signals
- Table VI lists an INT4 deployment; PyTorch has no native INT4 tensor type, so the INT4 path is a
  simulated 4-bit fake-quantization. Recorded in `deviations.md`.
- A prospective observational study (N=156, IRB Protocol #2023-0847, Empatica E4 monitors, 90-day
  follow-up) backs Table IX. It is an evaluation cohort, not part of the training pipeline.
- No released checkpoints. The manuscript carries a Reproducibility Statement pointing to PhysioNet
  rather than a verbatim code-availability block.
