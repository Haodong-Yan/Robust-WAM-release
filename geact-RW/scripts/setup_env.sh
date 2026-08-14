#!/usr/bin/env bash
# Set up the environment needed to evaluate Robust-WAM on GE-Act.
#
#   bash scripts/setup_env.sh              # conda env + deps + LIBERO / LIBERO-Plus sources
#   ENV_NAME=my-env bash scripts/setup_env.sh
#
# Clean LIBERO and LIBERO-Plus both ship a top-level `libero` module, so they are
# checked out as sources under third_party/ and selected per run via PYTHONPATH.
# Neither is pip-installed; only their shared dependencies are.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-robustwam-geact}"
THIRD_PARTY="${THIRD_PARTY:-$REPO_ROOT/third_party}"
PY_VER="${PY_VER:-3.10}"

echo "==> repo:        $REPO_ROOT"
echo "==> conda env:   $ENV_NAME (python $PY_VER)"
echo "==> third_party: $THIRD_PARTY"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Install miniforge/miniconda first." >&2
  exit 1
fi

# ---------------------------------------------------------------- conda env
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> conda env '$ENV_NAME' already exists, reusing it"
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

CONDA_RUN=(conda run -n "$ENV_NAME" --no-capture-output)

echo "==> installing python dependencies"
"${CONDA_RUN[@]}" pip install --upgrade pip
"${CONDA_RUN[@]}" pip install -r "$REPO_ROOT/requirements.txt"
# Evaluation extras: simulator, benchmark assets, checkpoint download.
"${CONDA_RUN[@]}" pip install \
  "robosuite==1.4.1" "bddl==1.0.1" "mujoco==3.2.3" "imageio[ffmpeg]" \
  "huggingface_hub>=0.34" "hydra-core>=1.3" "omegaconf>=2.3" "easydict" "termcolor" \
  "future" "cloudpickle" "gym==0.25.2"
# NOTE: do not `pip install -r third_party/LIBERO/requirements.txt` -- it pins numpy 1.22 /
# transformers 4.21 and would break the GE-Act stack. The packages above are the ones
# libero.libero.envs actually imports (`future` is an undeclared dependency of bddl 1.0.1).

# --------------------------------------------------------- benchmark sources
mkdir -p "$THIRD_PARTY"

clone_or_update () {  # $1 url, $2 target dir
  if [ -d "$2/.git" ]; then
    echo "==> $2 already checked out, skipping clone"
    return 0
  fi
  for attempt in 1 2 3 4 5; do
    if git clone --depth 1 "$1" "$2"; then
      return 0
    fi
    echo "==> clone attempt $attempt/5 failed, retrying in $((attempt * 5))s"
    rm -rf "$2"
    sleep $((attempt * 5))
  done
  echo "could not clone $1 -- clone it manually into $2 and re-run" >&2
  return 1
}

clone_or_update https://github.com/Lifelong-Robot-Learning/LIBERO.git "$THIRD_PARTY/LIBERO"
clone_or_update https://github.com/sylvestf/LIBERO-plus.git          "$THIRD_PARTY/LIBERO-plus"

# Each benchmark resolves its assets through a config.yaml. LIBERO only writes defaults when
# the file is absent, so a stale config from an earlier project silently wins -- write both
# explicitly, pointing at the checkouts above.
write_libero_config () {  # $1 config dir, $2 libero package root
  mkdir -p "$1"
  cat > "$1/config.yaml" <<CFG
benchmark_root: $2
bddl_files: $2/bddl_files
init_states: $2/init_files
datasets: $2/../datasets
assets: $2/assets
CFG
  echo "==> wrote $1/config.yaml -> $2"
}

write_libero_config "${LIBERO_CONFIG_DIR:-$HOME/.libero}"           "$THIRD_PARTY/LIBERO/libero/libero"
write_libero_config "${LIBERO_PLUS_CONFIG_DIR:-$HOME/.libero_plus}" "$THIRD_PARTY/LIBERO-plus/libero/libero"

cat <<EOF

==> environment ready

Next steps:

  conda activate $ENV_NAME
  python scripts/download_assets.py            # weights (see --help for options)
  source scripts/env.sh                        # written by download_assets.py
  bash scripts/eval_libero.sh                  # clean LIBERO, four suites
  bash scripts/eval_libero_plus.sh             # LIBERO-Plus, all perturbation axes

Notes:
  * LIBERO sources:      $THIRD_PARTY/LIBERO
  * LIBERO-Plus sources: $THIRD_PARTY/LIBERO-plus
    Both provide a 'libero' module; the eval wrappers put exactly one of them on
    PYTHONPATH, so never pip-install them into the environment.
  * Headless rendering needs MUJOCO_GL=egl (set by the eval wrappers).
EOF
