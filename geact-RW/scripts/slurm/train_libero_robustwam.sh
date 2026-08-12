#!/bin/bash
# Robust-WAM (GE-Act + future-DINO alignment) LIBERO action post-training —
# 50K steps, 8x H100, global batch 128 (16 x 8 x 1).
#
# Required env var:
#   CONDA_ENV   path to (or name of) the conda env to activate
# Optional (can also be set inside the yaml via ${oc.env:...}):
#   LTX_MODEL_PATH, GEACT_CKPT_PATH, DINOV3_MODEL_PATH, DATA_ROOT, OUTPUT_DIR
#
#SBATCH --job-name=robustwam_bidir_50k
#SBATCH --partition=gpu   # <-- set to your cluster's GPU partition
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=96:00:00
#SBATCH --output=train_bidir_50k_%j.log

set -x
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${CONDA_ENV:?set CONDA_ENV to your conda env path/name}/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export MASTER_ADDR=127.0.0.1
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    main.py --config_file configs/ltx_model/libero/action_model_libero_fastwam_mix_align_bidir.yaml
