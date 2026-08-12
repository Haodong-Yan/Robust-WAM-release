#!/usr/bin/env python
"""Smoke assertions for the future-DINO alignment insertion (bidirectional).

Runs the standard assertion suite on LTXVideoTransformer3DModel + action expert
(assertion 7 — action/video losses in normal range — is covered by the 30-step
smoke TRAINING run, not this script):

  A1  action-expert sequence grows by exactly K=16 tokens
  A2  prefix RoPE rows == model's own RoPE at the future-step positions
      [4,4,8,8,...,32,32] (history token occupies grid index 0)
  A3  prefix split -> [B,16,action_hidden]; align output [B,16,768]
  A4  per-token align loss finite and active (nonzero, target-sensitive)
  A5  bidirectional prefix: action output finite, sane, and DIFFERENT from the
      stock (align-disabled) path; video stays stock-equal; the eval-mode
      forward keeps the prefix (+16) and logs prefix_len
  A6  grads: align loss -> nonzero on align_prefix_emb + align_proj, ZERO on
      action_proj_out; action loss -> ZERO on align_proj, nonzero on the prefix
      (the prefix participates in the action self-attention)

Usage (repo root):
  python scripts/smoke_align_assertions.py [--ckpt interp_init.safetensors]
      [--layers 28] [--device cuda:0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def build_model(layers, align, device):
    from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel
    cfg = dict(
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=64,
        attention_out_bias=True,
        caption_channels=4096,
        cross_attention_dim=2048,
        in_channels=128,
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        num_attention_heads=32,
        num_layers=layers,
        out_channels=128,
        patch_size=1,
        patch_size_t=1,
        qk_norm="rms_norm_across_heads",
        action_expert=True,
        action_in_channels=15,
        action_out_channels=15,
        action_num_attention_heads=16,
        action_attention_head_dim=32,
        align_enabled=align,
        align_num_prefix_tokens=16,
        align_num_future_frames=8,
        align_num_cameras=2,
        align_target_dim=768,
    )
    torch.manual_seed(0)
    model = LTXVideoTransformer3DModel(**cfg)
    return model.to(device=device, dtype=torch.float32)


def make_inputs(device, B=2, V=2, F_lat=6, H=8, W=8, action_chunk=36):
    g = torch.Generator().manual_seed(1234)
    def rnd(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=torch.float32)
    inp = dict(
        hidden_states=rnd(B * V, F_lat * H * W, 128),
        encoder_hidden_states=rnd(B, 77, 4096),
        timestep=(torch.rand(B * V, F_lat * H * W, generator=g) * 1000).long().to(device),
        encoder_attention_mask=torch.ones(B, 77, device=device),
        n_view=V,
        rope_interpolation_scale=[8.0 / 30.0, 32, 32],
        num_frames=F_lat, height=H, width=W,
        action_states=rnd(B, action_chunk, 15),
        action_timestep=(torch.rand(B, action_chunk, generator=g) * 1000).long().to(device),
        history_action_state=rnd(B, 1, 15),
        return_video=True,
        return_action=True,
        return_dict=False,
    )
    return inp


def forward(model, inp):
    return model(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()})[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="optional interp-init safetensors to load")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = a.device
    os.environ.pop("GEACT_ALIGN_DISABLE", None)

    from models.action_patches.patches import prepend_align_prefix, align_prefix_position_ids

    model = build_model(a.layers, align=True, device=device)
    if a.ckpt:
        from utils.model_utils import load_checkpoints
        load_checkpoints(model, a.ckpt)
        model = model.to(device=device, dtype=torch.float32)
    model.eval()
    inp = make_inputs(device)
    B, K = 2, 16
    action_seq = 1 + 36  # history + action chunk

    results = {}

    # Force one SDPA kernel for exact align-on/off comparability.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    def kernel_ctx():
        return sdpa_kernel([SDPBackend.MATH])

    # ---- A1: sequence length +K (hook on action_blocks[0]) ----
    seen = {}
    h = model.action_blocks[0].register_forward_pre_hook(
        lambda m, args: seen.__setitem__("shape", tuple(args[0].shape)) if args else None,
        with_kwargs=False,
    )
    def hook_kw(m, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        seen["shape"] = tuple(x.shape)
    h.remove()
    h = model.action_blocks[0].register_forward_pre_hook(hook_kw, with_kwargs=True)
    with torch.no_grad(), kernel_ctx():
        out_align = forward(model, inp)
    h.remove()
    assert seen["shape"] == (B, K + action_seq, model.action_inner_dim), seen
    results["A1 seq +16"] = f"PASS action_blocks[0] input {seen['shape']} = [B, 16+{action_seq}, {model.action_inner_dim}]"

    # ---- A2: prefix RoPE rows == model RoPE at future-step positions ----
    dummy = torch.zeros(B, action_seq, model.action_inner_dim, device=device)
    rope = model.action_rope(dummy, action_seq)  # ([1,L,D],[1,L,D])
    temb = torch.zeros(B, action_seq, 6 * model.action_inner_dim, device=device)
    ph, pt, (pcos, psin), bias = prepend_align_prefix(model, dummy, temb, rope, n_history=1)
    pos = align_prefix_position_ids(model, n_history=1, prefix_stride=4)
    assert pos == [4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28, 32, 32], pos
    assert torch.equal(pcos[:, :K], rope[0][:, pos]) and torch.equal(psin[:, :K], rope[1][:, pos])
    assert torch.equal(pcos[:, K:], rope[0]) and torch.equal(psin[:, K:], rope[1])
    results["A2 PE at future positions"] = f"PASS positions {pos}"

    # ---- A3: prefix split + align head shapes ----
    assert "align" in out_align, list(out_align.keys())
    align_feat = out_align["align"]
    assert align_feat.shape == (B, K, 768), align_feat.shape
    assert out_align["action"].shape == (B, 36, 15), out_align["action"].shape
    assert ph.shape == (B, K + action_seq, model.action_inner_dim)
    results["A3 prefix split/proj"] = f"PASS align {tuple(align_feat.shape)}, action {tuple(out_align['action'].shape)}"

    # ---- A4: align loss finite + target-sensitive ----
    tgt = torch.randn(B, K, 768, device=device)
    loss_a = (1.0 - F.cosine_similarity(align_feat.float(), tgt, dim=-1)).mean()
    loss_b = (1.0 - F.cosine_similarity(align_feat.float(), -tgt, dim=-1)).mean()
    assert torch.isfinite(loss_a) and loss_a.item() > 0 and abs(loss_a.item() - loss_b.item()) > 1e-6
    results["A4 align loss"] = f"PASS finite, active (rand-tgt {loss_a.item():.4f} vs flipped {loss_b.item():.4f})"

    # ---- A5: bidirectional prefix — action output finite, sane, DIFFERENT from
    #      the stock (align-disabled) path; video stays stock-equal; the
    #      eval-mode forward keeps the prefix (+16) and logs prefix_len ----
    os.environ["GEACT_ALIGN_DISABLE"] = "1"
    with torch.no_grad(), kernel_ctx():
        out_stock = forward(model, inp)
    del os.environ["GEACT_ALIGN_DISABLE"]
    assert "align" not in out_stock
    d_act = (out_align["action"] - out_stock["action"]).abs().max().item()
    d_vid = (out_align["video"] - out_stock["video"]).abs().max().item()
    act = out_align["action"]
    assert torch.isfinite(act).all(), "bidir action output has non-finite values"
    assert act.abs().max().item() < 1e3, f"bidir action output out of sane range: {act.abs().max().item()}"
    assert d_act > 0, "prefix inactive: action output identical to stock"
    assert d_vid < 1e-5, f"video output must stay stock (prefix lives in action expert): {d_vid}"
    model.eval()
    model._align_eval_logged = False
    seen_eval = {}
    def hook_eval(m, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        seen_eval["shape"] = tuple(x.shape)
    h2 = model.action_blocks[0].register_forward_pre_hook(hook_eval, with_kwargs=True)
    with torch.no_grad(), kernel_ctx():
        out_eval = forward(model, inp)
    h2.remove()
    assert seen_eval["shape"] == (B, K + action_seq, model.action_inner_dim), seen_eval
    assert torch.isfinite(out_eval["action"]).all(), "eval-path action output non-finite"
    assert getattr(model, "_align_eval_logged", False), "eval forward did not log prefix_len"
    results["A5 bidir behavior"] = (
        f"PASS action finite (max {act.abs().max().item():.3f}), differs from stock "
        f"(max|d|={d_act:.2e}), video stock-equal ({d_vid:.2e}); eval forward keeps "
        f"prefix {seen_eval['shape']} and logged prefix_len"
    )

    # ---- A6: gradient routing ----
    model.zero_grad(set_to_none=True)
    with kernel_ctx():
        out = forward(model, inp)
        align_loss = (1.0 - F.cosine_similarity(out["align"].float(), tgt, dim=-1)).mean()
        align_loss.backward()
    g_prefix = model.align_prefix_emb.grad
    g_proj = model.align_proj.weight.grad
    assert g_prefix is not None and g_prefix.abs().max() > 0, "no grad on align_prefix_emb"
    assert g_proj is not None and g_proj.abs().max() > 0, "no grad on align_proj"
    g_head = model.action_proj_out.weight.grad
    assert g_head is None or g_head.abs().max() == 0, "align loss leaked into action head"
    gp, gj = g_prefix.abs().max().item(), g_proj.abs().max().item()

    model.zero_grad(set_to_none=True)
    with kernel_ctx():
        out = forward(model, inp)
        action_loss = out["action"].float().pow(2).mean()
        action_loss.backward()
    g_prefix2 = model.align_prefix_emb.grad
    g_proj2 = model.align_proj.weight.grad
    assert g_proj2 is None or g_proj2.abs().max() == 0, "action loss reached align_proj"
    # bidirectional: the prefix participates in attention, so the action loss
    # legitimately reaches align_prefix_emb (reported, not asserted zero);
    # align_proj stays off the action path.
    gp2 = 0.0 if g_prefix2 is None else g_prefix2.abs().max().item()
    results["A6 grad routing"] = (
        f"PASS align->prefix {gp:.2e}, align->proj {gj:.2e}; action->proj = 0; "
        f"action->prefix = {gp2:.2e} (expected nonzero in bidir)"
    )

    print("\n===== SMOKE ASSERTIONS (A7 = losses in range, see smoke train run) =====")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
