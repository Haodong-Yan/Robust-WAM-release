#!/usr/bin/env bash
# Evaluate a Robust-WAM GE-Act checkpoint on LIBERO-Plus (all perturbation axes).
#
#   source scripts/env.sh && bash scripts/eval_libero_plus.sh
#   NUM_SHARDS=4 SHARD=2 DEVICE=2 bash scripts/eval_libero_plus.sh    # one shard per GPU
#
# Requires (normally set by scripts/env.sh): LTX_MODEL_PATH, GEACT_CKPT_PATH, LIBERO_PLUS_ROOT.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${LTX_MODEL_PATH:?set LTX_MODEL_PATH (source scripts/env.sh)}"
: "${GEACT_CKPT_PATH:?set GEACT_CKPT_PATH (source scripts/env.sh)}"
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-$REPO_ROOT/third_party/LIBERO-plus}"

CKPT="${CKPT:-$GEACT_CKPT_PATH/diffusion_pytorch_model.safetensors}"
CONFIG="${CONFIG:-eval_out/lplus/eval_lplus_bidir.yaml}"
OUT_DIR="${OUT_DIR:-outputs/eval_lplus}"
DEVICE="${DEVICE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"  # e.g. EXTRA_ARGS="--num_trails_per_task 1" for a quick smoke run
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
[ -d "$LIBERO_PLUS_ROOT" ] || { echo "LIBERO-Plus sources not found: $LIBERO_PLUS_ROOT (run scripts/setup_env.sh)" >&2; exit 1; }

# LIBERO-Plus replaces standard LIBERO: keep only this one on PYTHONPATH.
export PYTHONPATH="$LIBERO_PLUS_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero_plus}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# LIBERO's init-state files are numpy pickles and it calls torch.load() without weights_only;
# torch >= 2.6 defaults that to True and refuses to load them.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

echo "==> checkpoint: $CKPT"
echo "==> shard:      $SHARD / $NUM_SHARDS (device $DEVICE)"
echo "==> output:     $OUT_DIR"

python experiments/eval_libero_plus.py \
    --config_file "$CONFIG" \
    --ckpt_path "$CKPT" \
    --out_dir "$OUT_DIR" \
    --device "$DEVICE" \
    --shard "$SHARD" \
    --num_shards "$NUM_SHARDS" \
    $EXTRA_ARGS

echo "==> done, results under $OUT_DIR"
