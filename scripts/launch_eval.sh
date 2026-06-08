#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"

python -m thrombxai.floor.audit experiment="${EXPERIMENT}"
