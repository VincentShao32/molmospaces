cd /gpfs/projects/raivn/yunbos/molmospaces

MLSPACES_DOWNLOAD_EXTRACT_ALL_SCENES_OBJECTS_GRASPS=true \
MLSPACES_DOWNLOAD_EXCLUDE_SOURCES="scenes:holodeck-objaverse-train,scenes:holodeck-objaverse-val,scenes:procthor-10k-train,scenes:procthor-10k-test,scenes:procthor-10k-val,scenes:procthor-objaverse-val,scenes:procthor-objaverse-test" \
  python -m molmo_spaces.molmo_spaces_constants 2>&1 | tee logs/download_$(date +%Y%m%d_%H%M%S).log
