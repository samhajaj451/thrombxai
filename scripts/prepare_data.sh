#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/mimic4wdb}"

echo "Place credentialed MIMIC-IV Waveform records under ${ROOT}."
echo "Each record needs an ECG lead and a skin-temperature channel resampled per conf/."
echo "Write ${ROOT}/manifest.json as a list of {record, subject, label, ecg_path, temp_path}."
echo "Then run training with data.synthetic=false data.root=${ROOT}."
