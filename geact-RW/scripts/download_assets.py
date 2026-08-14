#!/usr/bin/env python3
"""Download the weights needed to run Robust-WAM on GE-Act.

    python scripts/download_assets.py                  # eval assets (checkpoint + LTX-Video)
    python scripts/download_assets.py --with-dinov3    # also the training-only teacher
    python scripts/download_assets.py --assets-dir /data/robustwam_assets

Everything lands under --assets-dir (default: <repo>/assets) and the matching
environment variables are written to scripts/env.sh, so evaluation is:

    source scripts/env.sh && bash scripts/eval_libero_plus.sh
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Robust-WAM post-trained GE-Act checkpoint (dir holding diffusion_pytorch_model.safetensors).
CKPT_REPO = os.environ.get("ROBUSTWAM_CKPT_REPO", "Haodong082399/Robust-WAM-GE-Act")
# LTX-Video backbone the GE-Act stack is built on.
LTX_REPO = os.environ.get("LTX_REPO", "Lightricks/LTX-Video")
# The eval loads only the tokenizer / text encoder / VAE from LTX_MODEL_PATH -- the video DiT
# weights come from the Robust-WAM checkpoint. Fetching the whole repo would pull the standalone
# single-file releases as well (~254 GB); these patterns keep it to ~25 GB.
LTX_PATTERNS = ["tokenizer/*", "text_encoder/*", "vae/*", "scheduler/*", "*.json"]
# Frozen semantic teacher; only needed to precompute alignment targets for training.
DINOV3_REPO = os.environ.get("DINOV3_REPO", "facebook/dinov3-vitb16-pretrain-lvd1689m")


def safetensors_ok(path: Path) -> bool:
    """True if `path` is a complete safetensors file (header parses, size matches)."""
    import json
    import struct

    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(header_len))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        return 8 + header_len + end == size
    except Exception:
        return False


def drop_partial(path: Path) -> None:
    """Remove a truncated file and the hub bookkeeping that would let it be skipped."""
    meta = path.parent / ".cache" / "huggingface" / "download" / f"{path.name}.metadata"
    for p in (path, meta):
        try:
            p.unlink()
            print(f"    removed stale {p.name}")
        except FileNotFoundError:
            pass


def fetch(repo_id: str, dest: Path, allow_patterns=None, retries: int = 8) -> Path:
    """snapshot_download with retries; partial files are resumed, not restarted."""
    import time

    from huggingface_hub import snapshot_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> {repo_id}\n    -> {dest}")
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                allow_patterns=allow_patterns,
                max_workers=int(os.environ.get("HF_MAX_WORKERS", "4")),
            )
            return dest
        except Exception as exc:  # network hiccups are the norm on long transfers
            if attempt == retries:
                raise
            wait = min(60, 5 * attempt)
            print(f"    attempt {attempt}/{retries} failed ({type(exc).__name__}), retrying in {wait}s")
            time.sleep(wait)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-dir", default=os.environ.get("ROBUSTWAM_ASSETS", str(REPO_ROOT / "assets")))
    ap.add_argument("--ckpt-repo", default=CKPT_REPO, help=f"default: {CKPT_REPO}")
    ap.add_argument("--ltx-repo", default=LTX_REPO, help=f"default: {LTX_REPO}")
    ap.add_argument("--dinov3-repo", default=DINOV3_REPO, help=f"default: {DINOV3_REPO}")
    ap.add_argument("--with-dinov3", action="store_true", help="also download the training-only teacher")
    ap.add_argument("--skip-ltx", action="store_true", help="skip LTX-Video (already on disk)")
    ap.add_argument(
        "--full-ltx",
        action="store_true",
        help="download the entire LTX-Video repo (~254 GB) instead of just tokenizer/text_encoder/vae",
    )
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is missing: pip install 'huggingface_hub>=0.34'", file=sys.stderr)
        return 1

    # Xet/CAS transfers fail on some proxied networks; plain HTTPS is slower but reliable.
    # Set HF_HUB_DISABLE_XET=0 to opt back in.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    assets = Path(args.assets_dir).expanduser().resolve()
    ckpt_dir = assets / "robustwam-geact"
    weights = ckpt_dir / "diffusion_pytorch_model.safetensors"

    # An interrupted transfer leaves a full-size sparse file plus hub metadata that later runs
    # happily skip, so the download "succeeds" while the weights stay truncated. Verify, and if
    # the file is incomplete drop it and pull again.
    for round_ in range(1, 4):
        fetch(args.ckpt_repo, ckpt_dir)
        if safetensors_ok(weights):
            break
        print(f"==> {weights.name} is incomplete (round {round_}/3), re-fetching")
        drop_partial(weights)
    else:
        print(f"ERROR: {weights} still incomplete after 3 rounds", file=sys.stderr)
        return 1

    ltx_dir = assets / "ltx-video"
    if args.skip_ltx:
        print(f"==> skipping LTX-Video, expecting it at {ltx_dir}")
    else:
        fetch(args.ltx_repo, ltx_dir, allow_patterns=None if args.full_ltx else LTX_PATTERNS)

    dino_dir = assets / "dinov3-vitb16"
    if args.with_dinov3:
        fetch(args.dinov3_repo, dino_dir)

    print(f"==> checkpoint verified: {weights} ({weights.stat().st_size / 1e9:.2f} GB)")

    env_sh = REPO_ROOT / "scripts" / "env.sh"
    lines = [
        "# Written by scripts/download_assets.py -- source this before training or evaluation.",
        f'export LTX_MODEL_PATH="{ltx_dir}"',
        f'export GEACT_CKPT_PATH="{ckpt_dir}"',
        f'export LIBERO_ROOT="{REPO_ROOT / "third_party" / "LIBERO"}"',
        f'export LIBERO_PLUS_ROOT="{REPO_ROOT / "third_party" / "LIBERO-plus"}"',
        "export MUJOCO_GL=egl",
        "export PYOPENGL_PLATFORM=egl",
    ]
    if args.with_dinov3:
        lines.insert(3, f'export DINOV3_MODEL_PATH="{dino_dir}"')
    env_sh.write_text("\n".join(lines) + "\n")

    print(f"\n==> wrote {env_sh}\n")
    print("\n".join(lines[1:]))
    print("\nNext:  source scripts/env.sh && bash scripts/eval_libero_plus.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
