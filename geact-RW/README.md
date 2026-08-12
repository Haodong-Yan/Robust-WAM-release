# Robust-WAM on GE-Act

Robust-WAM applied to the GE-Act backbone (LTX-Video DiT + action expert), evaluated on LIBERO / LIBERO-Plus and a real-robot setup.

## 1. Installation

This folder is the GE-Act codebase with the Robust-WAM alignment integrated. Follow GE-Act's environment (LTX-Video stack, PyTorch, DeepSpeed):

```bash
conda create -n robustwam-geact python=3.10 -y
conda activate robustwam-geact
pip install -r requirements.txt
```

## 2. Weights & Data

- **Video backbone** — LTX-Video (28 layers, dim 2048) + its stage-1 action-expert init.
- **Semantic teacher** — DINOv3 ViT-B/16 (`hidden_size = 768`). Frozen; used only to precompute targets.
- **Data** — LIBERO in the lerobot-like format with per-frame DINOv3 CLS targets precomputed offline:

```bash
python scripts/extract_dino_perframe_libero.py --root $DATA_ROOT/lerobot_libero_fw512 --dino /path/to/dinov3_vitb16
# targets are written in place under <root>/<domain>/dino_targets_perframe/ (add --shard/--nshard/--gpu to parallelize)
```

## 3. Training

Single node, 8 GPUs (global batch 128 = 16 × 8), 50K steps. Alignment is the `..._align_bidir` config:

```bash
# single node, 8 GPUs, global batch 128. Wrapper sets PROJECT_ROOT / PYTHONPATH
# and requires CONDA_ENV to be set to your conda environment path.
export CONDA_ENV=/path/to/conda/env
bash scripts/slurm/train_libero_robustwam.sh

# equivalent underlying command:
torchrun --nnodes=1 --nproc_per_node=8 main.py \
    --config_file configs/ltx_model/libero/action_model_libero_fastwam_mix_align_bidir.yaml
```

Key knobs (`configs/ltx_model/libero/action_model_libero_fastwam_mix_align_bidir.yaml`):

| Field | Value | Meaning |
|---|---|---|
| `align_num_future_frames` | 8 | `T_f` future frames per camera |
| `align_num_cameras` | 2 | `C` camera views |
| `align_num_prefix_tokens` | 16 | `K = T_f × C` query tokens |
| `prefix_stride` | 4 | shared temporal PE stride (Δ = `action_chunk // chunk` = 36 / 9) |
| `future_alignment.loss_weight` | 0.1 | `λ_align` |
| `future_alignment.warmup_steps` | 5000 | alignment-loss warmup |

## 4. Inference / Evaluation

At inference the DINOv3 teacher and the `align_proj` head are dropped, but the **K = 16 query tokens are kept prepended** (`align_enabled: true`); the action slice is offset by K before the action head. The logs print `[future-align] eval forward: prefix_len=16`, confirming the queries are retained.

Config paths are read from env vars (resolved by the eval loader). Common to both:

```bash
export LTX_MODEL_PATH=/path/to/ltx-video
export GEACT_CKPT_PATH=/path/to/step_50000        # dir holding diffusion_pytorch_model.safetensors
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

Clean LIBERO and LIBERO-Plus use **different `libero` packages** (both expose the `libero` module), so run them separately with the right one on `PYTHONPATH`.

### Clean LIBERO (four standard suites)

Standard [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO); `experiments/eval_libero.py` runs one suite per call (`--num_trails_per_task` defaults to 50):

```bash
export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH      # standard LIBERO on the path
export LIBERO_CONFIG_PATH=$HOME/.libero
for suite in libero_spatial libero_object libero_goal libero_10; do
  python experiments/eval_libero.py \
      --config_file configs/ltx_model/libero/action_model_libero_fastwam_eval_align_bidir.yaml \
      --ckpt_path $GEACT_CKPT_PATH/diffusion_pytorch_model.safetensors \
      --task_suite_name $suite --output_dir outputs/eval_libero --device 0
done
```

### LIBERO-Plus (10,030 perturbed tasks over the same four suites)

[LIBERO-Plus](https://github.com/fanwenke/LIBERO-plus) used from source; `experiments/eval_libero_plus.py` reads `LIBERO_PLUS_ROOT` and runs all four suites in one job (shardable across GPUs with `--shard` / `--num_shards`):

```bash
export LIBERO_PLUS_ROOT=/path/to/LIBERO-plus      # LIBERO-plus on the path (replaces standard libero)
export LIBERO_CONFIG_PATH=$HOME/.libero_plus
python experiments/eval_libero_plus.py \
    --config_file eval_out/lplus/eval_lplus_bidir.yaml \
    --ckpt_path $GEACT_CKPT_PATH/diffusion_pytorch_model.safetensors \
    --out_dir outputs/eval_lplus --device 0 --shard 0 --num_shards 4
```

## 5. Where the method lives

| Component | File |
|---|---|
| Query tokens + prepend + shared RoPE positions | `models/action_patches/patches.py` (`align_prefix_emb`, `prepend_align_prefix`, `prefix_stride`) |
| Prefix injected in the shared forward | `models/ltx_models/transformer_ltx_multiview.py` (gated by `align_enabled`) |
| Frozen DINOv3 CLS targets (offline) | `scripts/extract_dino_perframe_libero.py`, gathered in `data/lerobot_like_dataset.py` |
| Per-query cosine loss + linear head | linear head `align_proj` in `models/action_patches/patches.py`; per-query cosine loss in `runner/ge_trainer.py` (`cosine_similarity`) |
| Inference keeps the prefix | `models/ltx_models/transformer_ltx_multiview.py` (`align_enabled` gate; prefix prepended and stripped only at the head). `models/pipeline/custom_pipeline.py` orchestrates via `return_action=True` |

Acknowledgement: built on GE-Act / Genie and LTX-Video.
