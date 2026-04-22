#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/scrubbed/yunbos/video_datasets/molmospaces/molmospaces

mkdir -p logs
LOG_FILE="logs/datagen_$(date +%Y%m%d_%H%M%S).log"
echo "Logging stdout/stderr to: $(pwd)/$LOG_FILE"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_INSTALL_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/assets/

python -m molmo_spaces.data_generation.mixture_main FrankaPickPointTrackOnly 2>&1 | tee "$LOG_FILE"
