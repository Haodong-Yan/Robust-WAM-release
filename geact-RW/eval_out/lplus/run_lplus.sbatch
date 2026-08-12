#!/bin/bash
#SBATCH --job-name=gelp
#SBATCH --partition=gpu   # <-- set to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=14:00:00
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

# Progress-based restart loop: native aborts (SIGABRT) kill the process, but crashguard
# makes each restart SKIP the prior crasher, so a crash still yields +1 recorded task.
# Keep restarting while forward progress continues; only give up after N consecutive
# ZERO-progress restarts (genuinely stuck), or rc=0 (shard finished). Grinds through any
# number of render-crashing tasks within walltime.
# Progress is tracked via the APPEND-ONLY crashguard file (monotonic, never rewritten ->
# no mid-write read race, unlike the results JSON which is fully rewritten per task).
# Every task ATTEMPTED appends one crashguard line, so crashguard growth == forward
# progress even when the rollout natively aborts (crashguard skips that task on resume).
CG="$OUT/crashguard_shard${SHARD}.jsonl"
cg_lines() { [ -f "$CG" ] && wc -l < "$CG" 2>/dev/null || echo 0; }
rc=1; stall=0; attempt=0
while true; do
  attempt=$((attempt+1))
  before=$(cg_lines)
  echo "[attempt $attempt] shard $SHARD/$NUM_SHARDS on $(hostname) CVD=$CUDA_VISIBLE_DEVICES cg_attempts=$before"
  python experiments/eval_libero_plus.py \
      --config_file "$CFG" \
      --ckpt_path   "$CKPT" \
      --out_dir     "$OUT" \
      --device 0 --shard "$SHARD" --num_shards "$NUM_SHARDS" \
      --tasklist "${TASKLIST:-$PROJECT_ROOT/eval_out/lplus/tasklists/full.jsonl}"
  rc=$?
  if [ $rc -eq 0 ]; then echo "[attempt $attempt] rc=0 (shard complete)"; break; fi
  after=$(cg_lines)
  if [ "$after" -gt "$before" ]; then
    stall=0
    echo "[attempt $attempt] exited rc=$rc but progressed (crashguard $before->$after, skipping crashers); restarting"
  else
    stall=$((stall+1))
    echo "[attempt $attempt] exited rc=$rc, NO crashguard progress ($before->$after); stall=$stall"
    if [ "$stall" -ge 5 ]; then echo "[giving up after $stall stalled restarts]"; break; fi
  fi
  sleep 15
done
echo "GELP_EXIT=$rc OUT=$OUT SHARD=$SHARD/$NUM_SHARDS cg_attempts=$(cg_lines)"
