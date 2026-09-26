"""Standalone PyTorch reimplementation of the Nanosaur2-670M pipeline.

Faithful translation of the ComfyUI custom node code in the
`nanosaur2_support/` folder (plus the ComfyUI internals it leans on):
  - Gemma3-270M text encoder (second-to-last hidden layer, final RMSNorm)
  - Nanosaur2 diffusion transformer (DiT, adaLN-single, 2D RoPE, SPRINT)
  - Semantic VAE decoder (only decode is needed for text-to-image)

Every module is written to be export-friendly (no data-dependent control
flow, no in-place ops) so it can be traced / exported to OpenVINO.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_EMBED_DIM = 640
SPRINT_NUM_F = 2
SPRINT_NUM_H = 2

# Large finite negative used for additive attention masks; exp() underflows
# to exactly 0 while avoiding -inf edge cases in fp16.
NEG_INF = -30000.0


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

def rms_norm(x, weight=None, eps=1e-6, add_one=False):
    """RMSNorm computed in fp32 (ComfyUI: F.rms_norm with weight+1 for Gemma)."""
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    y = x32 * torch.rsqrt(var + eps)
    if weight is not None:
        w = weight.float()
        if add_one:
            w = w + 1.0
        y = y * w
    return y.to(x.dtype)


def timestep_embedding(timesteps, dim, max_period=10000.0):
    """Sinusoidal embedding, cos in the first half / sin in the second (LDM)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def apply_rope_split_half(x, cos, sin):
    """Split-half RoPE: pair k is (x[k], x[k + d/2]).

    x: (..., d); cos/sin: (..., d/2) broadcastable.
    out[:d/2]   = x[:d/2] * cos - x[d/2:] * sin
    out[d/2:]   = x[:d/2] * sin + x[d/2:] * cos
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def build_rope(head_dim, height, width, device):
    """2D RoPE over centered token coords: x frequencies first, then y.

    Returns (cos, sin), each (H*W, head_dim//2) — broadcast-ready for
    apply_rope_split_half over (b, n, heads, head_dim) tensors.
    """
    axis_dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim))
    y = torch.arange(height, dtype=torch.float32, device=device) - (height - 1) / 2
    x = torch.arange(width, dtype=torch.float32, device=device) - (width - 1) / 2
    y, x = torch.meshgrid(y, x, indexing="ij")
    angles = torch.cat([torch.outer(x.flatten(), inv_freq), torch.outer(y.flatten(), inv_freq)], dim=-1)
    return torch.cos(angles), torch.sin(angles)


def modulate(x, shift, scale):
    return torch.addcmul(shift, x, 1 + scale)


def sdpa(q, k, v, mask=None):
    """q: (b, h, nq, d), k/v: (b, h, nk, d), mask: additive broadcastable to (b, 1, nq, nk)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


# ---------------------------------------------------------------------------
# Text encoder: Gemma3-270M (ComfyUI TransformerBlockGemma2 semantics)
# ---------------------------------------------------------------------------

class GemmaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        inner = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, inner, bias=False)
        self.k_proj = nn.Linear(hidden, kv, bias=False)
        self.v_proj = nn.Linear(hidden, kv, bias=False)
        self.o_proj = nn.Linear(inner, hidden, bias=False)
        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))
        self.eps = cfg["rms_norm_eps"]

    def forward(self, x, cos, sin, attn_bias):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q, self.q_norm, self.eps, add_one=True)
        k = rms_norm(k, self.k_norm, self.eps, add_one=True)
        q = apply_rope_split_half(q, cos, sin)
        k = apply_rope_split_half(k, cos, sin)
        # GQA: repeat KV heads (4 q heads : 1 kv head)
        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = sdpa(q, k, v, mask=attn_bias)
        out = out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)
        return self.o_proj(out)


class GemmaMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg["hidden_size"]
        self.self_attn = GemmaAttention(cfg)
        self.mlp = GemmaMLP(cfg)
        self.input_layernorm = nn.Parameter(torch.empty(hidden))
        self.post_attention_layernorm = nn.Parameter(torch.empty(hidden))
        self.pre_feedforward_layernorm = nn.Parameter(torch.empty(hidden))
        self.post_feedforward_layernorm = nn.Parameter(torch.empty(hidden))
        self.eps = cfg["rms_norm_eps"]

    def forward(self, x, cos, sin, attn_bias):
        h = rms_norm(x, self.input_layernorm, self.eps, add_one=True)
        h = self.self_attn(h, cos, sin, attn_bias)
        h = rms_norm(h, self.post_attention_layernorm, self.eps, add_one=True)
        x = x + h
        h = rms_norm(x, self.pre_feedforward_layernorm, self.eps, add_one=True)
        h = self.mlp(h)
        h = rms_norm(h, self.post_feedforward_layernorm, self.eps, add_one=True)
        return x + h


class Gemma3TextEncoder(nn.Module):
    """Gemma3-270M encoder returning the layer (-2) hidden state after the final norm."""

    def __init__(self, cfg, seq_len=256, layer_idx=-2):
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.layer_idx = layer_idx
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([GemmaBlock(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = nn.Parameter(torch.empty(cfg["hidden_size"]))
        self.eps = cfg["rms_norm_eps"]
        # rope_theta = [global, local]; sliding layers (5 of every 6) use local
        self.theta_global, self.theta_local = cfg["rope_theta"]
        self.sliding_attention = cfg["sliding_attention"]
        self.embed_scale = cfg["hidden_size"] ** 0.5
        self.register_buffer("positions", torch.arange(seq_len, dtype=torch.float32), persistent=False)
        head_dim = cfg["head_dim"]
        inv_g = self.theta_global ** -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        inv_l = self.theta_local ** -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq_global", inv_g, persistent=False)
        self.register_buffer("inv_freq_local", inv_l, persistent=False)
        # causal additive mask, fixed seq_len
        self.register_buffer("causal_mask", NEG_INF * torch.triu(torch.ones(seq_len, seq_len), diagonal=1), persistent=False)

    def _rope(self, inv_freq, dtype):
        # (seq, head_dim/2) angles computed in fp32, cast for the graph dtype
        freqs = self.positions[:, None] * inv_freq[None, :]
        return freqs.cos()[None, None].to(dtype), freqs.sin()[None, None].to(dtype)

    def forward(self, token_ids, attn_mask):
        """token_ids: (b, N) int64; attn_mask: (b, N) float 1=real 0=pad."""
        x = self.embed_tokens(token_ids) * self.embed_scale
        pad_bias = (1.0 - attn_mask).to(x.dtype) * NEG_INF  # (b, N)
        attn_bias = (self.causal_mask[None, None] + pad_bias[:, None, None, :]).to(x.dtype)
        cos_g, sin_g = self._rope(self.inv_freq_global, x.dtype)
        cos_l, sin_l = self._rope(self.inv_freq_local, x.dtype)
        z = None
        take = self.layer_idx if self.layer_idx >= 0 else len(self.layers) + self.layer_idx
        for i, layer in enumerate(self.layers):
            sliding = self.sliding_attention[i % len(self.sliding_attention)]
            cos, sin = (cos_l, sin_l) if sliding else (cos_g, sin_g)
            x = layer(x, cos, sin, attn_bias)
            if i == take:
                z = x
        return rms_norm(z, self.norm, self.eps, add_one=True)


GEMMA3_270M_CFG = {
    "vocab_size": 262144,
    "hidden_size": 640,
    "intermediate_size": 2048,
    "num_hidden_layers": 18,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": [1000000.0, 10000.0],
    "sliding_attention": [512, 512, 512, 512, 512, False],
}


# ---------------------------------------------------------------------------
# Diffusion transformer (Nanosaur2Transformer2DModel)
# ---------------------------------------------------------------------------

class Embed(nn.Module):
    def __init__(self, in_dim, hidden_size, norm=False):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size, bias=True)
        self.has_norm = norm
        if norm:
            self.norm_weight = nn.Parameter(torch.empty(hidden_size))

    def forward(self, x):
        x = self.proj(x)
        if self.has_norm:
            x = rms_norm(x, self.norm_weight, eps=1e-6)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class DiTAttention(nn.Module):
    """Self-attention over image tokens with 2D RoPE, joint with (non-rotated) text KV."""

    def __init__(self, dim, num_heads, cross_attention):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=False)
        self.kv_y = nn.Linear(dim, dim * 2, bias=False) if cross_attention else None
        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, txt, cos, sin, txt_bias):
        b, n, c = x.shape
        q, k, v = self.qkv_x(x).view(b, n, 3, self.num_heads, self.head_dim).unbind(2)
        q = rms_norm(q, self.q_norm, eps=1e-6)
        k = rms_norm(k, self.k_norm, eps=1e-6)
        q = apply_rope_split_half(q, cos, sin)
        k = apply_rope_split_half(k, cos, sin)
        q = q.transpose(1, 2)  # (b, h, n, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.kv_y is not None:
            tn = txt.shape[1]
            ky, vy = self.kv_y(txt).view(b, tn, 2, self.num_heads, self.head_dim).unbind(2)
            ky = rms_norm(ky, self.k_norm, eps=1e-6)
            k = torch.cat([k, ky.transpose(1, 2)], dim=2)
            v = torch.cat([v, vy.transpose(1, 2)], dim=2)
            # image keys always visible; text keys carry log(emphasis), masked for padding
            mask = torch.cat([torch.zeros(b, 1, n, n, dtype=x.dtype, device=x.device), txt_bias[:, None, None, :].expand(b, 1, n, tn)], dim=-1)
        else:
            mask = None
        out = sdpa(q, k, v, mask=mask)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden, cross_attention):
        super().__init__()
        self.norm1 = nn.Parameter(torch.empty(hidden_size))
        self.attn = DiTAttention(hidden_size, num_heads, cross_attention)
        self.norm2 = nn.Parameter(torch.empty(hidden_size))
        self.mlp = FeedForward(hidden_size, mlp_hidden)

    def forward(self, x, txt, cos, sin, mod, txt_bias):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        h = modulate(rms_norm(x, self.norm1, eps=1e-6), shift_msa, scale_msa)
        h = self.attn(h, txt, cos, sin, txt_bias)
        x = torch.addcmul(x, gate_msa, h)
        h = self.mlp(modulate(rms_norm(x, self.norm2, eps=1e-6), shift_mlp, scale_mlp))
        return torch.addcmul(x, gate_mlp, h)


class TextRefineAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm = nn.Parameter(torch.empty(self.head_dim))
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, attn_bias):
        b, n, c = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).unbind(2)
        q = rms_norm(q, self.q_norm, eps=1e-6)
        k = rms_norm(k, self.k_norm, eps=1e-6)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = sdpa(q, k, v, mask=attn_bias)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden):
        super().__init__()
        self.norm1 = nn.Parameter(torch.empty(hidden_size))
        self.attn = TextRefineAttention(hidden_size, num_heads)
        self.norm2 = nn.Parameter(torch.empty(hidden_size))
        self.mlp = FeedForward(hidden_size, mlp_hidden)
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def forward(self, x, c, attn_bias, keep):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        h = modulate(rms_norm(x, self.norm1, eps=1e-6), shift_msa, scale_msa)
        h = self.attn(h, attn_bias)
        x = torch.addcmul(x, gate_msa, h)
        h = self.mlp(modulate(rms_norm(x, self.norm2, eps=1e-6), shift_mlp, scale_mlp))
        x = torch.addcmul(x, gate_mlp, h)
        return x * keep


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        # LayerNorm without affine, then modulate
        x = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6).to(x.dtype)
        return self.linear(modulate(x, shift, scale))


class Nanosaur2DiT(nn.Module):
    def __init__(self, in_channels=64, hidden_size=1536, num_heads=16, num_blocks=18,
                 num_text_blocks=2, mlp_hidden=4096):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.s_embedder = Embed(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = Embed(TEXT_EMBED_DIM, hidden_size, norm=True)
        self.shared_encoder_adaLN = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.encoder_adaLN_offsets = nn.ParameterList(
            [nn.Parameter(torch.empty(6 * hidden_size)) for _ in range(num_blocks)])
        self.y_pool_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sprint_out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_hidden, i % 2 == 0) for i in range(num_blocks)])
        self.text_refine_blocks = nn.ModuleList([
            TextRefineBlock(hidden_size, num_heads, mlp_hidden) for _ in range(num_text_blocks)])
        self.final_layer = FinalLayer(hidden_size, in_channels)

    def forward(self, x, timestep, context, token_weights, rope_cos, rope_sin):
        """x: (b, C, H, W) latent; timestep: (b,) sigma in [0,1]; context: (b, N, 640);
        token_weights: (b, N) >=0 (0 = padding); rope_cos/rope_sin: (1, H*W, 1, head_dim//2).
        Returns velocity v = (x - x0) / timestep."""
        b, _, h, w = x.shape
        token_weights = token_weights.to(x.dtype)
        keep = token_weights > 0
        txt_bias = torch.log(token_weights.clamp(min=1e-4)).masked_fill(~keep, NEG_INF).to(x.dtype)
        # a row with no text key keeps its first position and is zeroed after the block
        first = keep[:, :1] | ~keep.any(dim=1, keepdim=True)
        refine_keep = torch.cat([first, keep[:, 1:]], dim=1)
        refine_mask = torch.zeros_like(txt_bias).masked_fill(~refine_keep, NEG_INF)[:, None, None, :]
        keep_f = keep.unsqueeze(-1).to(x.dtype)

        t = self.t_embedder(timestep * 1000.0).unsqueeze(1)
        txt = self.y_embedder(context)
        time_condition = F.silu(t)
        for block in self.text_refine_blocks:
            txt = block(txt, time_condition, refine_mask, keep_f)

        weights = token_weights.unsqueeze(-1)
        pooled = (txt * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        condition = F.silu(t + self.y_pool_proj(pooled).unsqueeze(1))
        mod = self.shared_encoder_adaLN(condition)

        cos = rope_cos.unsqueeze(0).unsqueeze(2)  # (1, N, 1, head_dim//2)
        sin = rope_sin.unsqueeze(0).unsqueeze(2)
        s = self.s_embedder(x.flatten(2).transpose(1, 2))
        for i in range(SPRINT_NUM_F):
            s = self.blocks[i](s, txt, cos, sin, mod + self.encoder_adaLN_offsets[i], txt_bias)

        g = s
        for i in range(SPRINT_NUM_F, len(self.blocks) - SPRINT_NUM_H):
            g = self.blocks[i](g, txt, cos, sin, mod + self.encoder_adaLN_offsets[i], txt_bias)
        s = s + self.sprint_out_proj(g - s)

        for i in range(len(self.blocks) - SPRINT_NUM_H, len(self.blocks)):
            s = self.blocks[i](s, txt, cos, sin, mod + self.encoder_adaLN_offsets[i], txt_bias)

        x0 = self.final_layer(s, condition).transpose(1, 2).reshape(b, self.in_channels, h, w)
        return (x - x0) / timestep.view(-1, 1, 1, 1)


# ---------------------------------------------------------------------------
# VAE decoder (ComfyUI LDM Decoder: ch=128, ch_mult=(1,1,2,2,4), attn at 16)
# ---------------------------------------------------------------------------

class VaeResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        sc = self.nin_shortcut(x) if self.nin_shortcut is not None else x
        return sc + h


class VaeAttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h).flatten(2).transpose(1, 2)  # (b, hw, c)
        k = self.k(h).flatten(2).transpose(1, 2)
        v = self.v(h).flatten(2).transpose(1, 2)
        o = sdpa(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))  # single head
        o = o.squeeze(1).transpose(1, 2).reshape(x.shape)
        return x + self.proj_out(o)


class VaeUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # nearest-neighbor x2 == row/column repeat; friendlier to dynamo than
        # F.interpolate(scale_factor=...) with symbolic spatial dims
        x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return self.conv(x)


class Nanosaur2VaeDecoder(nn.Module):
    def __init__(self, latent_channels=64, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 attn_resolution=16, resolution=256):
        super().__init__()
        self.latent_channels = latent_channels
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(latent_channels, block_in, 3, padding=1)
        self.mid_block_1 = VaeResnetBlock(block_in, block_in)
        self.mid_attn_1 = VaeAttnBlock(block_in)
        self.mid_block_2 = VaeResnetBlock(block_in, block_in)
        # deepest level first in execution
        levels = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for i_level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[i_level]
            blocks = [VaeResnetBlock(block_in, block_out)]
            block_in = block_out
            for _ in range(num_res_blocks):
                blocks.append(VaeResnetBlock(block_in, block_out))
            # attention follows every resblock at the attn resolution level
            attn = nn.ModuleList([VaeAttnBlock(block_in) for _ in range(num_res_blocks + 1)]) \
                if curr_res == attn_resolution else None
            up = VaeUpsample(block_in) if i_level != 0 else None
            levels.append({"blocks": nn.ModuleList(blocks), "attn": attn, "up": up})
            if i_level != 0:
                curr_res *= 2
        self.levels = nn.ModuleList([
            nn.ModuleDict({k: v for k, v in lvl.items() if v is not None}) for lvl in levels])
        self.level_specs = [(len(lvl["blocks"]), lvl["attn"] is not None, lvl["up"] is not None) for lvl in levels]
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, 3, 3, padding=1)
        self.register_buffer("latent_mean", torch.zeros(1, latent_channels, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, latent_channels, 1, 1))

    def forward(self, z):
        z = z * self.latent_std.to(z.dtype) + self.latent_mean.to(z.dtype)
        h = self.conv_in(z)
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)
        for level, spec in zip(self.levels, self.level_specs):
            n_blocks, has_attn, has_up = spec
            for i in range(n_blocks):
                h = level["blocks"][i](h)
                if has_attn:
                    h = level["attn"][i](h)
            if has_up:
                h = level["up"](h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.tanh(h)
