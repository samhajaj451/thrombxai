# ThrombXAI — Exhibition Guide

> A walk-through gallery for the wearable thrombotic-event early-warning model. Each room holds one
> stage of the pipeline: what is on display, the command that operates it, and the reading you should
> expect. Follow the rooms in order or jump to whichever exhibit you need.

```
   FOYER ──> R1 Intake ──> R2 Carding ──> R3 Blending ──> R4 Spinning ──> R5 Warp & Weave
                                                                              │
   R9 Edge Wing <── R8 Inspection <── R7 Finishing <── R6 Dyeing <───────────┘
```

The collection is arranged as a textile mill. Raw sensor fibre is opened and carded into aligned
features, blended into one cloth, spun on a temporal convolutional encoder, warped and woven by the
attention stages, dyed against a clinical-knowledge prior, finished into a risk score, and finally
pressed for the microcontroller.

---

## Plan your visit (installation)

Three ways through the door.

**pip**
```
python -m pip install -e ".[dev]"
```

**conda**
```
conda env create -f environment.yml
conda activate thrombxai
pip install -e .
```

**Docker**
```
docker build -t thrombxai:latest .
docker run --rm thrombxai:latest experiment=_smoke
```

Configuration is handled by Hydra. Every command accepts dotted overrides, for example
`train.seed=1` or `data.batch_size=128`, and selects an experiment with `experiment=<name>`.

---

## Room 1 — Intake

*On display:* subject-level partitioning and the offline cohort. The credentialed MIMIC-IV Waveform
records are not redistributable, so the gallery keeps a deterministic synthetic stand-in and a
manifest loader for the real data (`thrombxai/opening`).

*Operate it:* `python -m thrombxai.floor.weave experiment=_smoke` builds the cohort and trains two
quick steps. Point at real records with `data.synthetic=false data.root=<path>` after running
`scripts/prepare_data.sh`.

---

## Room 2 — Carding

*On display:* the move from raw waveform to aligned features. The HRV bench runs R-peak detection, a
0.5–40 Hz band-pass, an inter-beat-interval constraint of 300–2000 ms, and time, frequency, and
non-linear indices over 300-second windows with a 60-second stride. The thermal bench median-filters
skin temperature and derives statistical and dynamic descriptors. Twelve HRV columns and seven
temperature columns leave this room (`thrombxai/carding`).

---

## Room 3 — Blending

*On display:* the two feature streams concatenated into a single 19-channel cloth and standardized
with training-set statistics. Single-modality studies mask one stream here rather than rebuilding the
input (`thrombxai/blending`).

---

## Room 4 — Spinning

*On display:* the dilated causal temporal convolutional encoder. Four layers with dilation doubling
each step and kernel size three give a receptive field of 31 steps while keeping the time axis
causal. Residual connections and layer normalization wrap each block (`thrombxai/spinning`).

---

## Room 5 — Warp and Weave

*On display:* attention. Eight-head temporal self-attention learns long-range dependencies and yields
a per-step importance reading; spatial feature attention scores each of the 19 channels and pools
them into a context vector. The two readings are the model's own explanation
(`thrombxai/warping`, `thrombxai/weaving`).

---

## Room 6 — Dyeing

*On display:* the Clinical Relevance-Guided Attention prior. A small knowledge graph over the 19
features yields a relevance vector from node degree; a Kullback–Leibler term pulls the spatial
attention toward it and a smoothness term keeps temporal attention gradual. The pull anneals to zero
over a warmup window so the model is free to find new patterns later in training
(`thrombxai/dyeing`).

---

## Room 7 — Finishing

*On display:* the prediction head and the loss bench. The head maps the pooled representation and the
context vector to a risk probability; the objective combines focal loss with the relevance prior and
weight decay (`thrombxai/finishing`).

---

## Room 8 — Inspection

*On display:* the gauges. Predictive metrics, bootstrap confidence intervals, and DeLong variance sit
alongside the explanation-quality measures — comprehensiveness, sufficiency, and the attention
overlap score — and the rank agreement between attention and clinical importance
(`thrombxai/inspection`).

*Operate it:* `python -m thrombxai.floor.audit experiment=main`.

### Reading the labels (expected results)

The figures below are the manuscript readings on the MIMIC-IV Waveform test set. Reproduce them by
supplying the credentialed records (`data.synthetic=false`); the offline synthetic cohort exists only
to keep the commands runnable.

| Reading | Command | Expected |
| --- | --- | --- |
| Accuracy | `thrombxai-audit experiment=main` | 91.3 ± 0.5 % |
| AUROC | `thrombxai-audit experiment=main` | 0.936 ± 0.005 |
| AUPRC | `thrombxai-audit experiment=main` | 0.887 ± 0.007 |
| Sensitivity | `thrombxai-audit experiment=main` | 89.1 ± 0.7 % |
| Specificity | `thrombxai-audit experiment=main` | 92.0 ± 0.6 % |
| F1 | `thrombxai-audit experiment=main` | 0.897 ± 0.006 |
| NPV | `thrombxai-audit experiment=main` | 96.8 ± 0.4 % |
| Attention overlap score | `thrombxai-audit experiment=main` | 0.598 ± 0.024 |
| Sufficiency @ 20% | `thrombxai-audit experiment=main` | 0.812 ± 0.014 |
| Attention vs clinical rank | `thrombxai-audit experiment=main` | Spearman 0.871 |

The ablation rooms map to `experiment=ablation_no_crga`, `ablation_no_temporal`,
`ablation_no_spatial`, `ablation_no_dilation`, `ablation_no_focal`, `ablation_hrv_only`,
`ablation_temp_only`.

---

## Room 9 — Edge Wing

*On display:* the route to a microcontroller. Knowledge distillation transfers the full model into a
64-dimension student, 8-bit symmetric quantization and operator fusion shrink it, and an ONNX graph
plus a TensorFlow Lite Micro descriptor are emitted (`thrombxai/pressing`, `thrombxai/floor/ship.py`).

*Operate it:* `python -m thrombxai.floor.bale_out experiment=edge_distilled_int8` then
`python -m thrombxai.floor.ship experiment=edge_distilled_int8`.

| Edge reading | Expected (manuscript) |
| --- | --- |
| INT8 accuracy | 89.8 % |
| INT8 model size | 0.13 MB |
| INT8 latency (ARM Cortex-M7) | 850 ms |
| INT8 power | 145 mW |

The INT4 configuration (`experiment=edge_distilled_int4`) is a simulated 4-bit path; see
`docs/deviations.md`.

---

## Collection provenance (data)

| Dataset | Source | Version | License | Cohort |
| --- | --- | --- | --- | --- |
| MIMIC-IV Waveform | https://physionet.org/content/mimic4wdb/ | mimic4wdb | PhysioNet Credentialed Health Data License | 2,847 patients · 4,271 four-hour windows · 21.9% positive · subject-level 70/15/15 |

Preprocessing: `scripts/prepare_data.sh` describes the manifest format. Each record needs an ECG lead
and a skin-temperature channel; features are computed in Room 2.

---

## Logistics (compute budget)

| Stage | Hardware | Time |
| --- | --- | --- |
| Full model training | 1 × NVIDIA A100 80GB | ~4 hours, 5 seeds reported separately |
| Edge distillation | 1 × NVIDIA A100 80GB | ~1 hour |
| Edge inference | ARM Cortex-M7, 512KB SRAM, TFLite Micro | target < 850 ms |

---

## Access and ethics

The dataset is de-identified ICU data under PhysioNet credentialed access; obtain credentials before
downloading. Inference runs on-device and transmits no personal health information, consistent with
the privacy framing in the manuscript. The prospective validation cohort (N=156) was collected under
IRB Protocol #2023-0847. Performance is reported across demographic subgroups; the authors note that
age-based calibration and modelling of prior cancer are open items, and post-market monitoring is
expected to track drift.

---

## Tour desk (developer tasks)

| Task | Command |
| --- | --- |
| Lint | `make lint` |
| Format | `make format` |
| Type-check | `make type` |
| Tests | `make test` |
| Smoke train | `make smoke` |

The source modules carry no comments; paper provenance lives in `docs/implementation-map.md`.
