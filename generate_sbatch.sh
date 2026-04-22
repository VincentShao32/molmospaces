#!/bin/bash
#SBATCH --job-name=pt_datagen
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

cd /gpfs/scrubbed/yunbos/video_datasets/molmospaces/molmospaces

export FILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB="${FILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB:-1024}"
export FILAMENT_PER_RENDER_PASS_ARENA_SIZE_IN_MB="${FILAMENT_PER_RENDER_PASS_ARENA_SIZE_IN_MB:-1024}"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_INSTALL_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/assets/ \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/gpfs/home/yunbos/.conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.mixture_main FrankaPickPointTrackOnly
# Override per-component house counts as needed, e.g.:
#   ... mixture_main PointTrackTrioMixture \
#       --override FrankaPickPointTrackDebug=2000 \
#       --override RBY1PickPointTrack=500
