#!/usr/bin/env python
"""Offline: extract frozen DINOv3 CLS targets, PER FRAME, for future-DINO
alignment with GE-act's random-window sampler.

For every episode of every domain we decode ALL frames from each camera mp4,
run DINOv3 (224x224 bilinear, ImageNet norm — identical recipe to the
LingBot-VA / FastWAM extractions), and save

    <root>/<domain>/dino_targets_perframe/chunk-XXX/episode_XXXXXX.pth
        {"dino_target": float16 [T, V, hid], "num_cams": V, "hid": hid}

The training loader (data/lerobot_like_dataset.py, align_dino_targets=True)
gathers rows at the sampled future-frame ids x cameras at runtime.

Run (shardable across GPUs):
  python scripts/extract_dino_perframe_libero.py \
      --root "$DATA_ROOT/lerobot_libero_fw512" \
      --dino "$DINOV3_MODEL_PATH" \
      --gpu 0 --shard 0 --nshard 1
"""
import os, glob, json, argparse
import torch, av
import torch.nn.functional as F
from transformers import AutoModel

CAMS = ["observation.images.image", "observation.images.wrist_image"]
DOMAINS = [
    "libero_10_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_spatial_no_noops_lerobot",
]
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
OUT_TAG = "dino_targets_perframe"


def load_dino(path, device, dtype):
    m = AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval().to(device=device, dtype=dtype)
    hid = int(getattr(getattr(m, "config", None), "hidden_size", 0) or getattr(m, "embed_dim", 0))
    return m, hid


def extract_cls(out):
    for k in ("pooler_output", "x_norm_clstoken", "cls_token", "class_token"):
        v = getattr(out, k, None) if not isinstance(out, dict) else out.get(k)
        if v is not None:
            return v
    h = out.get("last_hidden_state") if isinstance(out, dict) else getattr(out, "last_hidden_state", None)
    return h[:, 0]


def decode_all_frames(mp4):
    """Decode every frame of an mp4 -> [T,3,H,W] float in [0,1]."""
    frames = []
    container = av.open(mp4)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    for frame in container.decode(stream):
        frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    container.close()
    return torch.stack(frames, 0).float() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dino", default=os.environ.get("DINOV3_MODEL_PATH", "/path/to/dinov3_model"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()

    dev = f"cuda:{a.gpu}"; dtype = torch.bfloat16
    dino, hid = load_dino(a.dino, dev, dtype)
    mean = torch.tensor(MEAN, device=dev, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev, dtype=dtype).view(1, 3, 1, 1)

    eps = []  # (domain, chunk, episode_index)
    for dom in DOMAINS:
        cam0_dir = os.path.join(a.root, dom, "videos")
        for mp4 in sorted(glob.glob(os.path.join(cam0_dir, "chunk-*", CAMS[0], "episode_*.mp4"))):
            chunk = os.path.basename(os.path.dirname(os.path.dirname(mp4)))
            ep = os.path.basename(mp4).split("_")[1].split(".")[0]
            eps.append((dom, chunk, ep))
    eps = [e for i, e in enumerate(eps) if i % a.nshard == a.shard]
    print(f"[gpu{a.gpu} shard {a.shard}/{a.nshard}] {len(eps)} episodes, dino hid={hid}", flush=True)

    done = 0; err = 0
    for dom, chunk, ep in eps:
        try:
            outdir = os.path.join(a.root, dom, OUT_TAG, chunk)
            os.makedirs(outdir, exist_ok=True)
            outf = os.path.join(outdir, f"episode_{ep}.pth")
            if os.path.exists(outf):
                done += 1
                continue
            cls_all = []
            n_frames = None
            for cam in CAMS:
                mp4 = os.path.join(a.root, dom, "videos", chunk, cam, f"episode_{ep}.mp4")
                fr = decode_all_frames(mp4)  # [T,3,H,W]
                if n_frames is None:
                    n_frames = fr.shape[0]
                else:
                    assert fr.shape[0] == n_frames, (mp4, fr.shape[0], n_frames)
                cls_cam = []
                for s in range(0, fr.shape[0], a.batch):
                    fb = fr[s:s + a.batch].to(dev, dtype)
                    fb = F.interpolate(fb, size=(a.img, a.img), mode="bilinear", align_corners=False)
                    fb = (fb - mean) / std
                    with torch.no_grad():
                        c = extract_cls(dino(fb))  # [b, hid]
                    cls_cam.append(c.float().cpu())
                cls_all.append(torch.cat(cls_cam, 0))  # [T, hid]
            tgt = torch.stack(cls_all, 1)  # [T, V, hid]
            torch.save({"dino_target": tgt.to(torch.float16), "num_cams": len(CAMS), "hid": hid}, outf)
            done += 1
            if done % 25 == 0:
                print(f"[gpu{a.gpu}] {done}/{len(eps)} (err={err})", flush=True)
        except Exception as e:
            err += 1
            if err <= 5:
                print(f"[gpu{a.gpu}] ERR {dom}/{chunk}/{ep}: {repr(e)[:160]}", flush=True)
    print(f"[gpu{a.gpu}] DONE {done}/{len(eps)} err={err}", flush=True)


if __name__ == "__main__":
    main()
