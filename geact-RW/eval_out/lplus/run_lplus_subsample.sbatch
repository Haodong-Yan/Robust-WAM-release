#!/bin/bash
#SBATCH --job-name=gelp
#SBATCH --partition=gpu   # <-- set to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=lplus_%x_%j.log

set -x
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${CONDA_ENV:?set CONDA_ENV to your conda env path/name}/bin/activate"
cd "$PROJECT_ROOT"
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:?set LIBERO_PLUS_ROOT to your LIBERO-plus checkout}"
export LIBERO_CONFIG_PATH=$HOME/.libero_plus
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# small jitter to avoid many jobs initializing CUDA at the exact same instant
sleep $(( (RANDOM % 20) + SHARD * 2 ))

# CFG / CKPT / OUT / SHARD / NUM_SHARDS injected by caller via --export.
# Retry up to 3x: the eval has GLOBAL cross-shard resume, so a retry only runs
# tasks not yet recorded -> safe against transient CUDA-init failures on bad nodes.
# Preflight: if this node cannot init CUDA (a bad-GPU node), requeue elsewhere.
if ! python -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and (torch.zeros(1).cuda().sum().item()==0)) else 1)" 2>/dev/null; then
  echo "[preflight] CUDA init FAILED on $(hostname); requeuing off this node"
  scontrol update JobId=$SLURM_JOB_ID ExcNodeList=$(hostname) 2>/dev/null
  scontrol requeue $SLURM_JOB_ID 2>/dev/null
  sleep 20
  exit 1
fi

rc=1
for attempt in 1 2 3; do
  echo "[attempt $attempt] shard $SHARD/$NUM_SHARDS on $(hostname) CVD=$CUDA_VISIBLE_DEVICES"
  python experiments/eval_libero_plus.py \
      --config_file "$CFG" \
      --ckpt_path   "$CKPT" \
      --out_dir     "$OUT" \
      --device 0 --shard "$SHARD" --num_shards "$NUM_SHARDS" \
      --tasklist "$PROJECT_ROOT/eval_out/lplus/tasklists/subsample.jsonl"
  rc=$?
  if [ $rc -eq 0 ]; then break; fi
  echo "[attempt $attempt] exited rc=$rc; sleeping 30s before retry"
  sleep 30
done
echo "GELP_EXIT=$rc OUT=$OUT SHARD=$SHARD/$NUM_SHARDS"
