#!/usr/bin/env bash
set -euo pipefail

# run.sh - simple wrapper to call CLI inside container
# Example:
#   docker run ... pipeline --config configs/pipeline_config.yaml

if [ "$#" -eq 0 ]; then
  echo "No args provided. Showing CLI help..."
  exec python -m src.yolo_augmentor.cli --help
else
  exec python -m src.yolo_augmentor.cli "$@"
fi
