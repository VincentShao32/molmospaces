cd /gpfs/scrubbed/yunbos/video_datasets/molmospaces/molmospaces && \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_INSTALL_DIR=/gpfs/scrubbed/yunbos/video_datasets/molmospaces/assets/ \
python -m molmo_spaces.data_generation.main FrankaPickDroidDataGenConfig