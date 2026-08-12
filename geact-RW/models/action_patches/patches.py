import math
from typing import Any, Dict, Optional, Tuple


import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph


class ActionRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_seq_length: int = 57,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Always compute rope in fp32
        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0)

        grid = grid / self.base_seq_length

        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs




@maybe_allow_in_graph
class ActionTransformerBlock(nn.Module):
    r"""
    Modified from Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        attention_class,
        attention_args,
        dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        cross_attention_dim: int = 2048,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        attn3_cross_attention_dim = 2048,
        num_latent_downsample_block = 0,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = attention_class(
            **(attention_args[0]),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = attention_class(
            **(attention_args[1]),
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self.num_latent_downsample_block = num_latent_downsample_block
        if self.num_latent_downsample_block > 0:
            self.latent_downsample_block = nn.ModuleList()
            for _i in range(self.num_latent_downsample_block):
                self.latent_downsample_block.append(
                    downsampling_block()
                )


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attn3_hidden_states: torch.Tensor = None,
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=rotary_emb,
            n_view=1,
            attention_mask=self_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        
        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
            n_view=1,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        

        return hidden_states



def add_action_expert(
    self,
    num_layers: int = 28,
    inner_dim: int = 2048,
    activation_fn: str = "gelu",
    norm_eps: float = 1e-6,
    action_in_channels: int = 14,
    action_out_channels: int = None,
    action_num_attention_heads: int = 16,
    action_attention_head_dim: int = 32,
    action_rope_dim: int = None,
    action_final_embeddings: bool = True,
    learnable_action_state: bool = False,
    norm_elementwise_affine: bool = False,
    attention_bias: bool = True,
    attention_out_bias: bool = True,
    qk_norm: str = "rms_norm_across_heads",
    attention_class = None,
    attention_processor = None,
    align_enabled: bool = False,
    align_num_prefix_tokens: int = 16,
    align_num_future_frames: int = 8,
    align_num_cameras: int = 2,
    align_target_dim: int = 768,
    **kwargs,
):

    if action_out_channels is None:
        action_out_channels = action_in_channels

    self.action_inner_dim = action_num_attention_heads * action_attention_head_dim

    self.learnable_action_state = learnable_action_state
    if self.learnable_action_state:
        self.action_state = nn.Parameter(torch.randn(1, 1, action_in_channels))

    self.action_proj_in = nn.Linear(action_in_channels, self.action_inner_dim)
    self.action_scale_shift_table = nn.Parameter(torch.randn(2, self.action_inner_dim) / self.action_inner_dim**0.5)
    self.action_time_embed = AdaLayerNormSingle(self.action_inner_dim, use_additional_conditions=False)

    if action_rope_dim is None:
        action_rope_dim = self.action_inner_dim
    # set to a fixed value currently, should adjust according to the action length
    self.action_rope = ActionRotaryPosEmbed(
        dim=action_rope_dim,
        base_seq_length=57,
        theta=10000.0,
    )

    attention_args = []
    attention_args.append(dict(
        query_dim=self.action_inner_dim,
        heads=action_num_attention_heads,
        kv_heads=action_num_attention_heads,
        dim_head=action_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=None,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))
    attention_args.append(dict(
        query_dim=self.action_inner_dim,
        heads=action_num_attention_heads,
        kv_heads=action_num_attention_heads,
        dim_head=action_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=inner_dim,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))

    self.action_blocks = nn.ModuleList(
        [
            ActionTransformerBlock(
                attention_class = attention_class,
                attention_args = attention_args,
                dim=self.action_inner_dim,
                num_attention_heads=action_num_attention_heads,
                attention_head_dim=action_attention_head_dim,
                cross_attention_dim=inner_dim,
                qk_norm=qk_norm,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                attention_out_bias=attention_out_bias,
                eps=norm_eps,
                elementwise_affine=norm_elementwise_affine,
            )
            for _ in range(num_layers)
        ]
    )

    self.action_proj_out = nn.Linear(self.action_inner_dim, action_out_channels)
    self.action_final_embeddings = action_final_embeddings
    if not self.action_final_embeddings:
        self.action_proj_extra = nn.Linear(self.action_inner_dim, self.action_inner_dim)

    self.action_norm_out = nn.LayerNorm(self.action_inner_dim, eps=1e-6, elementwise_affine=False)

    ### ------------------------------------------------------------------
    ### Future-frame DINO alignment (faithful FastWAM-v6 port).
    ### K = align_num_future_frames * align_num_cameras learnable prefix
    ### tokens are prepended to the action-token sequence; after the last
    ### action block their hidden states are projected (single Linear) to
    ### the DINO-CLS dimension and cosine-aligned to frozen targets. The prefix
    ### participates bidirectionally in the action self-attention (no mask), so
    ### it must also be prepended at inference (enable align in the eval config).
    self.align_enabled = bool(align_enabled)
    self.align_num_prefix_tokens = int(align_num_prefix_tokens)
    self.align_num_future_frames = int(align_num_future_frames)
    self.align_num_cameras = int(align_num_cameras)
    if self.align_enabled:
        assert self.align_num_prefix_tokens == self.align_num_future_frames * self.align_num_cameras, (
            f"align_num_prefix_tokens ({self.align_num_prefix_tokens}) must equal "
            f"num_future_frames*num_cameras ({self.align_num_future_frames}*{self.align_num_cameras})"
        )
        # Learnable prefix (register) tokens used as alignment readout.
        # Init small (truncated normal std=0.02) — analogous to ViT register tokens.
        prefix = torch.zeros(self.align_num_prefix_tokens, self.action_inner_dim)
        nn.init.trunc_normal_(prefix, std=0.02)
        self.align_prefix_emb = nn.Parameter(prefix)
        # Per-token projection action_hidden -> DINO dim.
        self.align_proj = nn.Linear(self.action_inner_dim, int(align_target_dim))


def preprocessing_action_states(
    self,
    action_states: torch.Tensor = None,
    action_timestep: torch.LongTensor = None,
):

    assert self.action_expert == True
    assert action_states is not None and action_timestep is not None

    batch_size = action_states.shape[0]

    action_seq_length = action_states.shape[1]

    if getattr(self, "learnable_action_state") and self.learnable_action_state:
        action_states = self.action_state.repeat(batch_size, action_seq_length, 1).to(dtype=action_states.dtype, device=action_states.device)

    action_rotary_emb = self.action_rope(action_states, action_seq_length)
    action_hidden_states = self.action_proj_in(action_states)

    action_temb, action_embedded_timestep = self.action_time_embed(
        action_timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=action_hidden_states.dtype,
    )

    action_temb = action_temb.view(batch_size, -1, action_temb.size(-1))
    action_embedded_timestep = action_embedded_timestep.view(batch_size, -1, action_embedded_timestep.size(-1))

    return action_temb, action_embedded_timestep, action_rotary_emb, action_hidden_states


def align_prefix_position_ids(self, n_history: int, prefix_stride: int):
    """RoPE grid positions for the K align-prefix tokens.

    The token for future video frame j (j = 0..T_f-1) is placed at the SAME
    position the model's own positional encoding assigns to the action step
    that frame corresponds to. With the GE-act sampler, predicted frame j is
    action index (j+1)*stride - 1 within the action chunk
    (video_end[stride-1::stride]), and action index i sits at RoPE grid index
    n_history + i (history/state token occupies index 0 when present).
    Frame-major over cameras: [f0c0, f0c1, f1c0, f1c1, ...], both cameras of a
    frame share the position id (FastWAM v6: prefix_group_size = num_cameras).
    """
    positions = []
    for j in range(self.align_num_future_frames):
        positions.extend([n_history + (j + 1) * prefix_stride - 1] * self.align_num_cameras)
    return positions


def prepend_align_prefix(
    self,
    action_hidden_states: torch.Tensor,
    action_temb: torch.Tensor,
    action_rotary_emb,
    n_history: int = 0,
    prefix_stride: int = 4,
):
    """Prepend the K learnable align-prefix tokens to the action sequence.

    Returns (hidden, temb, rotary_emb, self_attention_bias) where every tensor
    is extended by K rows at the FRONT of the sequence. With
    The prefix participates bidirectionally in the action self-attention, so no
    attention bias is applied (bias is None).
    """
    assert self.align_enabled
    batch_size, action_seq_length, _ = action_hidden_states.shape
    K = self.align_num_prefix_tokens

    # 1) learnable prefix tokens, broadcast over batch
    prefix_tokens = self.align_prefix_emb.to(
        dtype=action_hidden_states.dtype, device=action_hidden_states.device
    )
    prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1, -1)
    hidden = torch.cat([prefix_tokens, action_hidden_states], dim=1)

    # 2) model's own RoPE at the future-step positions (rows gathered from the
    #    same grid the action tokens use)
    cos_freqs, sin_freqs = action_rotary_emb  # each [1, L, D]
    positions = align_prefix_position_ids(self, n_history=n_history, prefix_stride=prefix_stride)
    pos = torch.as_tensor(positions, device=cos_freqs.device, dtype=torch.long)
    assert int(pos.max()) < cos_freqs.shape[1], (
        f"align prefix position {int(pos.max())} out of action RoPE grid ({cos_freqs.shape[1]})"
    )
    cos_freqs = torch.cat([cos_freqs[:, pos], cos_freqs], dim=1)
    sin_freqs = torch.cat([sin_freqs[:, pos], sin_freqs], dim=1)

    # 3) timestep-0 (clean) AdaLN rows for the prefix — same treatment as the
    #    history/state token
    zero_t = torch.zeros(batch_size * K, device=hidden.device, dtype=torch.long)
    prefix_temb, _ = self.action_time_embed(
        zero_t, batch_size=batch_size, hidden_dtype=hidden.dtype
    )
    prefix_temb = prefix_temb.view(batch_size, K, -1)
    temb = torch.cat([prefix_temb, action_temb], dim=1)

    # 4) Bidirectional prefix: no attention bias — the prefix participates in
    #    the full action self-attention.
    bias = None

    return hidden, temb, (cos_freqs, sin_freqs), bias
