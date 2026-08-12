#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/projects/raivn/yunbos/molmospaces

mkdir -p logs
LOG_FILE="logs/datagen_$(date +%Y%m%d_%H%M%S).log"
echo "Logging stdout/stderr to: $(pwd)/$LOG_FILE"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MLSPACES_SKIP_CACHE_VERIFY=1
export MLSPACES_CACHE_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/cache-2
export MLSPACES_ASSETS_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/symlink-2
python -m molmo_spaces.data_generation.mixture_main FrankaPickPointTrackAnimatedCamOnly 2>&1 | tee "$LOG_FILE"
