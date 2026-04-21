#!/bin/bash
#SBATCH --job-name=pt_datagen
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

cd /gpfs/scrubbed/yunbos/video_datasets/molmospaces/molmospaces
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_INSTALL_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/assets/ \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/gpfs/home/yunbos/.conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.mixture_main RBY1PickPointTrackOnly
# Override per-component house counts as needed, e.g.:
#   ... mixture_main PointTrackTrioMixture \
#       --override FrankaPickPointTrackDebug=2000 \
#       --override RBY1PickPointTrack=500
