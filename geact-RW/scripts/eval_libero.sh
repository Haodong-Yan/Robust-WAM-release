#!/usr/bin/env bash
# Evaluate a Robust-WAM GE-Act checkpoint on the four clean LIBERO suites.
#
#   source scripts/env.sh && bash scripts/eval_libero.sh
#   SUITES="libero_goal" DEVICE=1 bash scripts/eval_libero.sh
#
# Requires (normally set by scripts/env.sh): LTX_MODEL_PATH, GEACT_CKPT_PATH, LIBERO_ROOT.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${LTX_MODEL_PATH:?set LTX_MODEL_PATH (source scripts/env.sh)}"
: "${GEACT_CKPT_PATH:?set GEACT_CKPT_PATH (source scripts/env.sh)}"
LIBERO_ROOT="${LIBERO_ROOT:-$REPO_ROOT/third_party/LIBERO}"

CKPT="${CKPT:-$GEACT_CKPT_PATH/diffusion_pytorch_model.safetensors}"
CONFIG="${CONFIG:-configs/ltx_model/libero/action_model_libero_fastwam_eval_align_bidir.yaml}"
OUT_DIR="${OUT_DIR:-outputs/eval_libero}"
DEVICE="${DEVICE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"  # e.g. EXTRA_ARGS="--num_trails_per_task 1" for a quick smoke run
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
[ -d "$LIBERO_ROOT" ] || { echo "LIBERO sources not found: $LIBERO_ROOT (run scripts/setup_env.sh)" >&2; exit 1; }

# Clean LIBERO only: LIBERO-Plus must stay off PYTHONPATH, both expose `libero`.
export PYTHONPATH="$LIBERO_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# LIBERO's init-state files are numpy pickles and it calls torch.load() without weights_only;
# torch >= 2.6 defaults that to True and refuses to load them.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

echo "==> checkpoint: $CKPT"
echo "==> suites:     $SUITES"
echo "==> output:     $OUT_DIR"

for suite in $SUITES; do
  echo "==> [$suite]"
  python experiments/eval_libero.py \
      --config_file "$CONFIG" \
      --ckpt_path "$CKPT" \
      --task_suite_name "$suite" \
      --output_dir "$OUT_DIR" \
      --device "$DEVICE" \
      $EXTRA_ARGS
done

echo "==> done, results under $OUT_DIR"
