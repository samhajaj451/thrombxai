#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
SEED="${2:-0}"

python -m thrombxai.floor.weave experiment="${EXPERIMENT}" train.seed="${SEED}"
