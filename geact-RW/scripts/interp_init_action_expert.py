"""
Initialize the GE-Act action expert from the video backbone via linear interpolation,
following the FastWAM recipe (scripts/preprocess_action_dit_backbone.py):
  - per-dimension sequential 1D linear interpolation (align_corners=True)
  - alpha = sqrt(d_src/d_dst) scaling when the LAST dim of a >=2D tensor is resized
  - I/O-specific layers (action_proj_in / action_proj_out) are skipped (keep default init)

Mapping (LTXVideoTransformer3DModel, video inner=2048 -> action inner=512):
  action_blocks.{i}.*          <- transformer_blocks.{i}.*   (same submodule names)
  action_time_embed.*          <- time_embed.*               (AdaLayerNormSingle)
  action_scale_shift_table     <- scale_shift_table          ([2,2048] -> [2,512])

Usage:
  python scripts/interp_init_action_expert.py \
      --src  PATH/GE_base_fast_v0.1.safetensors \
      --out  PATH/GE_base_fast_actinterp.safetensors \
      [--action-inner 512] [--no-alpha]
"""
import argparse

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def interpolate_last_dim(t: torch.Tensor, new_size: int) -> torch.Tensor:
    if t.shape[-1] == new_size:
        return t
    flat = t.reshape(-1, 1, t.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*t.shape[:-1], new_size)


def resize_to_shape(src: torch.Tensor, target_shape) -> torch.Tensor:
    out = src.to(torch.float32)
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv = [0] * out.ndim
        for i, p in enumerate(perm):
            inv[p] = i
        out = interpolate_last_dim(out.permute(*perm).contiguous(), new_size).permute(*inv).contiguous()
    assert tuple(out.shape) == tuple(target_shape), (src.shape, target_shape, out.shape)
    return out


def target_shapes(video_sd, video_inner=2048, action_inner=512, num_layers=28):
    """Enumerate action-expert params and their init source key + target shape."""
    r = action_inner / video_inner
    plan = {}  # action_key -> (video_key, target_shape)

    def shrink(shape):
        return tuple(action_inner if s == video_inner else
                     (int(s * r) if s % video_inner == 0 and s // video_inner >= 4 else s)
                     for s in shape)

    for i in range(num_layers):
        vp, ap = f"transformer_blocks.{i}.", f"action_blocks.{i}."
        for k, v in video_sd.items():
            if not k.startswith(vp):
                continue
            suffix = k[len(vp):]
            vshape = tuple(v.shape)
            # attn2.to_k/to_v consume video features (2048) as KV: keep in-dim, shrink out-dim
            if suffix.startswith("attn2.to_k") or suffix.startswith("attn2.to_v"):
                if suffix.endswith("weight"):
                    tshape = (action_inner, video_inner)
                else:
                    tshape = (action_inner,)
            else:
                # ff inner dim is 4*inner in both towers: 8192 -> 2048
                tshape = tuple(action_inner if s == video_inner else
                               (4 * action_inner if s == 4 * video_inner else s) for s in vshape)
            plan[ap + suffix] = (k, tshape)

    # AdaLayerNormSingle: action_time_embed.* <- time_embed.*
    for k, v in video_sd.items():
        if k.startswith("time_embed."):
            tshape = tuple(action_inner if s == video_inner else
                           (6 * action_inner if s == 6 * video_inner else s) for s in v.shape)
            plan["action_" + k] = (k, tshape)

    # final modulation table [2, 2048] -> [2, 512]
    plan["action_scale_shift_table"] = ("scale_shift_table", (2, action_inner))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--action-inner", type=int, default=512)
    ap.add_argument("--video-inner", type=int, default=2048)
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--no-alpha", action="store_true")
    args = ap.parse_args()

    sd = load_file(args.src)
    plan = target_shapes(sd, args.video_inner, args.action_inner, args.num_layers)

    copied = interpolated = 0
    new_sd = dict(sd)
    for akey, (vkey, tshape) in sorted(plan.items()):
        src = sd[vkey]
        if tuple(src.shape) == tshape:
            val = src.clone()
            copied += 1
        else:
            val = resize_to_shape(src, tshape)
            # FastWAM alpha rule: only when a >=2D tensor's LAST dim was resized
            if (not args.no_alpha) and src.ndim >= 2 and src.shape[-1] != tshape[-1]:
                val = val * (float(src.shape[-1]) / float(tshape[-1])) ** 0.5
            interpolated += 1
        new_sd[akey] = val.to(src.dtype).contiguous()

    save_file(new_sd, args.out)
    print(f"saved {args.out}")
    print(f"video params kept: {len(sd)}, action params added: {len(plan)} "
          f"(copied={copied}, interpolated={interpolated})")
    print("intentionally left to default init: action_proj_in, action_proj_out")


if __name__ == "__main__":
    main()
