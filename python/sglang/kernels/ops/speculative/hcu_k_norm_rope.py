"""HCU-only DFLASH K normalization/RoPE epilogue (opt-in at call sites)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hcu_k_norm_rope_kernel(
    K, W, Positions, Cache, Out,
    K_ROW: tl.constexpr, CACHE_ROW: tl.constexpr,
    HEADS: tl.constexpr, DIM: tl.constexpr, EPS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, DIM)
    raw = tl.load(K + token * K_ROW + head * DIM + d).to(tl.float32)
    weight = tl.load(W + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(raw * raw, axis=0) / DIM + EPS)
    # Match the narrow intermediate written by the original RMSNorm kernel.
    norm = (raw * inv * weight).to(K.dtype.element_ty).to(tl.float32)
    half: tl.constexpr = DIM // 2
    partner_idx = tl.where(d < half, d + half, d - half)
    partner = tl.gather(norm, partner_idx, axis=0)
    pos = tl.load(Positions + token)
    cos = tl.load(Cache + pos * CACHE_ROW + d % half).to(tl.float32)
    sin = tl.load(Cache + pos * CACHE_ROW + half + d % half).to(tl.float32)
    rotated = tl.where(d < half, -partner, partner)
    value = norm * cos + rotated * sin
    tl.store(Out + (token * HEADS + head) * DIM + d, value)


def hcu_k_norm_rope(k, weight, positions, cos_sin_cache, head_dim, eps):
    """Preserve strided QKV inputs and avoid a dummy Q allocation."""
    heads = k.shape[1] // head_dim
    out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _hcu_k_norm_rope_kernel[(k.shape[0], heads)](
        k, weight, positions, cos_sin_cache, out,
        k.stride(0), cos_sin_cache.stride(0), heads, head_dim, eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return out
