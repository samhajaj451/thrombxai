# Repository Plan

## Layout

```
thrombxai/
  order.py            structured Hydra configs + ConfigStore registration
  yarn.py             typed batch and forward-output contracts
  spindle.py          seeding, logging, atomic checkpoint IO, device selection
  opening/            data intake: subject-level split, manifest loader, synthetic cohort
  carding/            raw signal to aligned features: pulse (HRV), thermal (skin temperature)
  blending/           feature concatenation, standardization, modality masking
  spinning/           dilated causal temporal convolutional encoder
  warping/            multi-head temporal self-attention and aggregation
  weaving/            spatial feature attention
  dyeing/             clinical knowledge graph, relevance vector, CRGA regularizers
  finishing/          prediction head and loss terms
  fabric.py           assembled ThrombXAI and edge model
  loom/               optimizer/scheduler builders and the trainer
  pressing/           distillation, quantization, operator fusion, size/latency accounting
  inspection/         predictive metrics, statistics, explanation-quality metrics
  floor/              Hydra command-line entry points
conf/                 config.yaml + model group + experiment group
tests/                shape, physiology, causality, attention, CRGA, focal, metrics,
                      overfit, gradient-flow, edge-parity, no-comment, forbidden-phrase, smoke
docs/                 project-context, implementation-map, deviations, repo-plan
scripts/              launch_train.sh, launch_eval.sh, prepare_data.sh
```

## Dependencies (pinned floors)

Runtime: `torch>=2.1`, `numpy>=1.24`, `scipy>=1.10`, `hydra-core>=1.3`, `omegaconf>=2.3`.
Dev: `ruff>=0.6`, `black>=24.0`, `isort>=5.13`, `mypy>=1.10`, `pytest>=8.0`, `scikit-learn>=1.3`.

## Command surface

| Console script | Module | Purpose |
| --- | --- | --- |
| `thrombxai-weave` | `thrombxai.floor.weave` | train a configuration |
| `thrombxai-audit` | `thrombxai.floor.audit` | evaluate metrics and explanation quality |
| `thrombxai-pick` | `thrombxai.floor.pick` | single-window risk and drivers |
| `thrombxai-bale-out` | `thrombxai.floor.bale_out` | distill and quantize the edge model |
| `thrombxai-ship` | `thrombxai.floor.ship` | export ONNX and the edge deployment descriptor |

## Test coverage intent

Every science module has at least one behavioural test: feature shapes (12/7/19), the IBI
physiological constraint and the receptive-field identity, temporal causality, attention rows summing
to one, the clinical relevance distribution and CRGA properties, focal-vs-BCE behaviour, metric
agreement against scikit-learn, single-batch overfitting, full gradient flow, edge quantization
parity, the no-comment/no-docstring guard, the forbidden-phrase guard, and the end-to-end smoke run.
