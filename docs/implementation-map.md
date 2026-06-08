# Implementation Map

Paper provenance for every numbered equation, algorithm, table, and reported result. The source
files carry no comments or docstrings by design; this table is the authoritative cross-reference.

The package is organised as a textile mill: raw fibre is opened, carded into aligned features,
blended, spun on the temporal convolutional encoder, warped and woven by the attention stages, dyed
against the clinical prior, finished into a risk score, then pressed for the edge target.

## Equations

| Paper item | Description | File | Symbol |
| --- | --- | --- | --- |
| Eq. (1) | HRV time-domain indices (SDNN, RMSSD, pNN50, HRmean) | `thrombxai/carding/pulse.py` | `time_domain` |
| Eq. (2) | HRV frequency-domain indices (LF, HF, LF/HF, VLF) | `thrombxai/carding/pulse.py` | `frequency_domain` |
| Eq. (3) | HRV non-linear indices (SD1, SD2, SampEn, DFA-alpha) | `thrombxai/carding/pulse.py` | `nonlinear`, `sample_entropy`, `dfa_alpha`, `poincare` |
| Eq. (4) | Skin-temperature statistical features | `thrombxai/carding/thermal.py` | `statistical_features` |
| Eq. (5) | Skin-temperature dynamic features | `thrombxai/carding/thermal.py` | `dynamic_features` |
| Eq. (6) | Modality concatenation to 19 features | `thrombxai/blending/roving.py` | `integrate` |
| Eq. (7) | Training-set standardization | `thrombxai/blending/roving.py` | `Standardizer` |
| Eq. (8) | Dilated causal convolution | `thrombxai/spinning/twist.py` | `DilatedCausalConv1d` |
| Eq. (9) | Receptive field = 31 | `thrombxai/spinning/twist.py` | `receptive_field` |
| Eq. (10) | Residual + LayerNorm block | `thrombxai/spinning/twist.py` | `TwistBlock` |
| Eq. (11)-(13) | Multi-head temporal self-attention | `thrombxai/warping/heddle.py` | `TemporalSelfAttention` |
| Eq. (14) | Temporal attention aggregation | `thrombxai/warping/heddle.py` | `aggregate_temporal` |
| Eq. (15) | Time-average pooling | `thrombxai/fabric.py` | `ThrombXAI.forward` |
| Eq. (16)-(17) | Spatial feature attention | `thrombxai/weaving/shed.py` | `SpatialFeatureAttention` |
| Eq. (18) | Context vector from spatial weights | `thrombxai/weaving/shed.py` | `SpatialFeatureAttention.forward` |
| Eq. (19) | Prediction head | `thrombxai/finishing/press.py` | `PredictionHead` |
| Eq. (20) | Clinical relevance vector from knowledge-graph degree | `thrombxai/dyeing/mordant.py` | `relevance_vector` |
| Eq. (21) | CRGA regularizer (KL + smoothness) | `thrombxai/dyeing/vat.py` | `crga_loss` |
| Eq. (22) | Temporal smoothness penalty | `thrombxai/dyeing/vat.py` | `temporal_smoothness` |
| Eq. (23) | Adaptive KL warmup schedule | `thrombxai/dyeing/vat.py` | `kl_warmup` |
| Eq. (24) | Combined training objective | `thrombxai/loom/shuttle.py` | `Trainer._loss` |
| Eq. (25) | Binary cross-entropy loss | `thrombxai/finishing/objective.py` | `bce_loss` |
| Eq. (26) | Focal loss (gamma = 2) | `thrombxai/finishing/objective.py` | `focal_loss` |

## Pipeline stages and edge engine

| Paper section | Description | File |
| --- | --- | --- |
| §III.B.1 | R-peak detection, band-pass, IBI constraint | `thrombxai/carding/pulse.py` |
| §III.B (windowing) | 300s window / 60s stride HRV matrix | `thrombxai/carding/pulse.py` (`hrv_matrix`) |
| §III.D | Knowledge distillation (temperature 4) | `thrombxai/pressing/calender.py` |
| §III.D | 8-bit symmetric quantization (+ simulated INT4) | `thrombxai/pressing/baling.py` |
| §III.D | Operator fusion + size/latency accounting | `thrombxai/pressing/seam.py` |
| §IV.A | Subject-level split + MIMIC manifest loader | `thrombxai/opening/bale.py` |
| §IV.A | Offline synthetic cohort | `thrombxai/opening/sliver.py` |

## Tables

| Table | Description | File |
| --- | --- | --- |
| Table I | Dataset characteristics | `docs/project-context.md`; `conf/data` defaults in `thrombxai/order.py` |
| Table II | Full vs edge architecture specs | `thrombxai/order.py` (`ModelConfig`, `edge_model_config`) |
| Table III | Predictive metrics (ACC/AUROC/AUPRC/SEN/SPE/F1/PPV/NPV) | `thrombxai/inspection/gauge.py` |
| Table IV | Explanation quality (Comp, Suff, AOS) | `thrombxai/inspection/burling.py` |
| Table V | Ablations | `conf/experiment/ablation_*.yaml` |
| Table VI | Edge deployment (size, latency, power) | `thrombxai/pressing/seam.py`; `conf/experiment/edge_distilled_*.yaml` |
| Table VII | Top features by attention + Spearman | `thrombxai/inspection/burling.py` (`ranking_correlation`); `thrombxai/dyeing/mordant.py` |
| Table VIII | Temporal attention patterns | `thrombxai/warping/heddle.py` (`aggregate_temporal`) |
| Table IX | Prospective validation cohort | evaluation-only; `thrombxai/inspection/gauge.py` |
| Table X | Performance across wearable devices | evaluation-only; `thrombxai/inspection/gauge.py` |

## Statistics

| Paper item | Description | File |
| --- | --- | --- |
| Bootstrap CI | Percentile confidence intervals | `thrombxai/inspection/gauge.py` (`bootstrap_ci`) |
| DeLong | AUROC variance | `thrombxai/inspection/gauge.py` (`delong_variance`) |
| Spearman | Attention vs clinical rank | `thrombxai/inspection/burling.py` (`ranking_correlation`) |
