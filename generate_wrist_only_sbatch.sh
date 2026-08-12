#!/bin/bash
#SBATCH --job-name=pt_wrist_only
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd /gpfs/projects/raivn/yunbos/molmospaces
mkdir -p logs

export FILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB="${FILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB:-1024}"
export FILAMENT_PER_RENDER_PASS_ARENA_SIZE_IN_MB="${FILAMENT_PER_RENDER_PASS_ARENA_SIZE_IN_MB:-1024}"
export MLSPACES_CACHE_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/cache-2
export MLSPACES_ASSETS_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/symlink-2

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MLSPACES_SKIP_CACHE_VERIFY=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/gpfs/home/yunbos/.conda/envs/mlspaces/bin/python \
    -m molmo_spaces.data_generation.mixture_main \
    FrankaPickPointTrackWristOnly
