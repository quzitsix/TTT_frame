# Source from Bash: source scripts/new_server_paths_20261031.sh
# Paths supplied by the user on 2026-10-31; verify mounts on the new server.
# This file only exports paths. It never installs, copies, or starts training.
export TTT_PROJECT_ROOT="${TTT_PROJECT_ROOT:-$HOME/TTT_frame}"
export TTT_RUN_ROOT="${TTT_RUN_ROOT:-$TTT_PROJECT_ROOT/runs/rl_forecasting_v1}"
export TTT_MODEL_ROOT="${TTT_MODEL_ROOT:-$HOME/models}"
export TTT_QWEN_MODEL="${TTT_QWEN_MODEL:-$TTT_MODEL_ROOT/Qwen3-VL-2B-Instruct}"
export TTT_SPATIAL_CHECKPOINT="${TTT_SPATIAL_CHECKPOINT:-$TTT_MODEL_ROOT/Spatial-TTT-nano/model.safetensors}"
export TTT_EGOLIFE_ROOT="/mnt/gpu02/disk1/wuyuhang/data/EgoLife/EgoLife"
export TTT_NEXTME_ROOT="/mnt/gpu02/disk1/wuyuhang/data/EgoLife/NextMe-meta"
export TTT_NEXTME_CAPTIONS="$TTT_NEXTME_ROOT/captions/2_egolife"
export TTT_BENCH_V0="/mnt/gpu02/disk1/wuyuhang/data/EgoLife/bench_v0"
export TTT_BENCH_MIXED="/mnt/gpu02/disk1/wuyuhang/data/EgoLife/bench_mixed_v1"
export TTT_CONTEXT_DATA="/mnt/nas/users/wuyuhang/forecasting_mcq50_200_v1/data"
export TTT_VQA_POOL="/mnt/gpu02/disk1/wuyuhang/workspace/TTT/outputs/case_lora/20261022T174317Z/data"
export TTT_OPEN_GRPO_DATA="/mnt/nas/users/wuyuhang/forecasting_open_grpo_v1/20261029_v1/data"
