cd /gpfs/scrubbed/yunbos/video_datasets/molmospaces/molmospaces
conda activate mlspaces
export MLSPACES_CACHE_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/cache
export MLSPACES_ASSETS_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/molmospaces/assets
export MUJOCO_GL=egl
python generate_point_tracking.py \
  --dataset ithor \
  --split train \
  --num_episodes 10 \
  --num_frames 150 \
  --num_points 256