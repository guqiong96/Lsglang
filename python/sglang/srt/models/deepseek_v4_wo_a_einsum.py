"""BF16 grouped einsum for the DeepSeek-V4 attention ``wo_a`` low-rank.

``o [T, G, D] @ wo_a [G, R, D] -> [T, G, R]`` (equation ``tgd,grd->tgr``).

Ported from vllm-ds4 ``fp8_einsum.py`` but simplified: the SM80/SM86 decode
path dequantizes the ``wo_a`` weight to bf16 at load time (no FP8 tensor cores
on Ampere), so both operands are bf16 and no scales are needed. A Triton
grouped einsum is ~1.4x faster than the ``torch.einsum`` cuBLAS batched-bmm at
decode shapes. On prefill (large T) the cuBLAS batched bmm wins, so the plain
wrapper keeps a ``torch.einsum`` path there.

``wo_a_bf16_einsum_with_rope`` fuses the per-head inverse RoPE into the einsum
so the standalone inverse-roped kernel pass is skipped during decode.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _wo_a_bf16_einsum_kernel(
    o_ptr,
    wo_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    o_stride_t,
    o_stride_g,
    o_stride_h,
    wo_stride_g,
    wo_stride_r,
    wo_stride_h,
    out_stride_t,
    out_stride_g,
    out_stride_r,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t_block = tl.program_id(0)
    r_block = tl.program_id(1)
    g = tl.program_id(2)
    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    r_offs = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    h_offs = tl.arange(0, BLOCK_H)
    t_mask = t_offs < num_tokens
    r_mask = r_offs < out_rank
    accum = tl.zeros((BLOCK_T, BLOCK_R), tl.float32)
    for h_start in range(0, hidden_size, BLOCK_H):
        h = h_start + h_offs
        h_mask = h < hidden_size
        a = tl.load(
            o_ptr + t_offs[:, None] * o_stride_t + g * o_stride_g + h[None, :] * o_stride_h,
            mask=t_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            wo_ptr + g * wo_stride_g + r_offs[:, None] * wo_stride_r + h[None, :] * wo_stride_h,
            mask=r_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        accum = tl.dot(a, tl.trans(b), acc=accum, input_precision="tf32")
    tl.store(
        out_ptr + t_offs[:, None] * out_stride_t + g * out_stride_g + r_offs[None, :] * out_stride_r,
        accum,
        mask=t_mask[:, None] & r_mask[None, :],
    )


@triton.jit
def _wo_a_bf16_einsum_rope_kernel(
    o_ptr,
    wo_ptr,
    freqs_ptr,
    positions_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    o_stride_t,
    o_stride_g,
    o_stride_h,
    wo_stride_g,
    wo_stride_r,
    wo_stride_h,
    out_stride_t,
    out_stride_g,
    out_stride_r,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    FREQ_STRIDE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t_block = tl.program_id(0)
    r_block = tl.program_id(1)
    g = tl.program_id(2)
    t_offs = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    r_offs = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    h_offs = tl.arange(0, BLOCK_H)
    t_mask = t_offs < num_tokens
    r_mask = r_offs < out_rank
    positions = tl.load(positions_ptr + t_offs)  # [BLOCK_T]
    accum = tl.zeros((BLOCK_T, BLOCK_R), tl.float32)
    for h_start in range(0, hidden_size, BLOCK_H):
        h = h_start + h_offs
        h_mask = h < hidden_size
        a = tl.load(
            o_ptr + t_offs[:, None] * o_stride_t + g * o_stride_g + h[None, :] * o_stride_h,
            mask=t_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        # --- inverse RoPE on each head's rope tail (last ROPE_DIM of HEAD_DIM).
        # Layout after o.view(T, G, D): head h_local occupies columns
        # [h_local*HEAD_DIM, (h_local+1)*HEAD_DIM), rope tail at
        # [h_local*HEAD_DIM + NOPE_DIM, ...). Partners are adjacent (even,odd)
        # pairs; inverse rotation matches fused_rope_inplace(inverse=True) and
        # vllm-ds4 fused_inv_rope_fp8_quant (lines 81-97).
        local = h % HEAD_DIM
        k = local - NOPE_DIM  # rope sub-index 0..ROPE_DIM-1, negative on nope
        is_rope = (local >= NOPE_DIM) & h_mask
        pair = tl.maximum(k, 0) // 2
        cos_idx = pair * 2
        freq_idx = positions[:, None] * FREQ_STRIDE + cos_idx[None, :]
        f_mask = is_rope[None, :] & t_mask[:, None]
        cos_v = tl.load(freqs_ptr + freq_idx, mask=f_mask, other=0.0)
        sin_v = tl.load(freqs_ptr + freq_idx + 1, mask=f_mask, other=0.0)
        a_partner = tl.load(
            o_ptr + t_offs[:, None] * o_stride_t + g * o_stride_g + (h[None, :] ^ 1) * o_stride_h,
            mask=t_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        is_even = (k & 1) == 0
        rotated = tl.where(
            is_even[None, :],
            a * cos_v + a_partner * sin_v,
            a * cos_v - a_partner * sin_v,
        )
        # Round the rotated rope tail back to bf16, matching the standalone
        # CUDA fused_rope_inplace which casts fp32 -> bf16 before the einsum.
        a = tl.where(is_rope[None, :], rotated, a).to(tl.bfloat16)
        # ---------------------------------------------------------------
        b = tl.load(
            wo_ptr + g * wo_stride_g + r_offs[:, None] * wo_stride_r + h[None, :] * wo_stride_h,
            mask=r_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        accum = tl.dot(a, tl.trans(b), acc=accum, input_precision="tf32")
    tl.store(
        out_ptr + t_offs[:, None] * out_stride_t + g * out_stride_g + r_offs[None, :] * out_stride_r,
        accum,
        mask=t_mask[:, None] & r_mask[None, :],
    )


def _block_sizes(num_tokens: int, out_rank: int, hidden_size: int, is_decode: bool):
    # Decode (small T) favors latency: BLOCK_T=16 (minimum tf32 dot M) with the
    # extra lanes masked off. Prefill (large T) uses a wider token tile.
    if is_decode or num_tokens <= 16:
        block_t = 16
    else:
        block_t = 64
    return block_t, 128, 128


def wo_a_bf16_einsum(
    o: torch.Tensor,
    wo_a: torch.Tensor,
    is_decode: bool = False,
) -> torch.Tensor:
    """Grouped bf16 einsum ``o [T,G,D] @ wo_a [G,R,D] -> [T,G,R]``.

    On decode (small T) uses the Triton grouped einsum (~1.4x faster than
    torch.einsum cuBLAS batched bmm). On prefill (large T) falls back to
    ``torch.einsum`` where cuBLAS is better.
    """
    T, G, D = o.shape
    wo_G, R, wo_D = wo_a.shape
    assert (wo_G, wo_D) == (G, D), f"wo_a shape {tuple(wo_a.shape)} vs o {tuple(o.shape)}"
    assert o.dtype == torch.bfloat16 and wo_a.dtype == torch.bfloat16

    out = torch.empty((T, G, R), dtype=o.dtype, device=o.device)
    if T == 0:
        return out

    use_triton = is_decode or T <= 16
    if not use_triton:
        return torch.einsum("tgd,grd->tgr", o, wo_a)

    block_t, block_r, block_h = _block_sizes(T, R, D, is_decode)
    grid = (triton.cdiv(T, block_t), triton.cdiv(R, block_r), G)
    _wo_a_bf16_einsum_kernel[grid](
        o,
        wo_a,
        out,
        T,
        G,
        R,
        D,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        wo_a.stride(0),
        wo_a.stride(1),
        wo_a.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_T=block_t,
        BLOCK_R=block_r,
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=3,
    )
    return out


def wo_a_bf16_einsum_with_rope(
    o: torch.Tensor,
    wo_a: torch.Tensor,
    is_decode: bool = False,
    freqs_cis: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused inverse-RoPE + grouped bf16 einsum for decode.

    ``o`` is the attention output already in roped form ``[T, G, D]`` (each
    head's last ``ROPE_DIM`` dims still carry the forward RoPE). The inverse
    RoPE is applied in-register inside the einsum kernel so the standalone
    ``fused_rope_inplace`` pass is skipped. ``freqs_cis`` is the full complex
    table from ``precompute_freqs_cis`` (converted to the interleaved
    ``[pos, 64]`` cos||sin layout here); ``positions`` are the per-token
    positions ``[T]``.
    """
    T, G, D = o.shape
    wo_G, R, wo_D = wo_a.shape
    assert (wo_G, wo_D) == (G, D)
    assert o.dtype == torch.bfloat16 and wo_a.dtype == torch.bfloat16
    assert freqs_cis is not None and positions is not None
    assert positions.shape[0] == T

    rope_dim = 64
    head_dim = 512  # DSV4: qk_nope(448) + qk_rope(64)
    nope_dim = head_dim - rope_dim  # 448
    assert D % head_dim == 0, (
        f"grouped hidden dim {D} not a multiple of head_dim {head_dim}"
    )

    # Interleaved [cos0, sin0, cos1, sin1, ...] layout, matching what
    # fused_rope_inplace consumes (freqs_real = view_as_real(freqs_cis).flatten).
    if freqs_cis.is_complex():
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    else:
        freqs_real = freqs_cis.contiguous()
    freq_stride = freqs_real.shape[-1]  # rope_dim (64)
    assert freq_stride == rope_dim

    out = torch.empty((T, G, R), dtype=o.dtype, device=o.device)
    if T == 0:
        return out

    use_triton = is_decode or T <= 16
    if not use_triton:
        # Prefill: plain einsum over de-roped o (rope applied separately by
        # the caller in that case).
        return torch.einsum("tgd,grd->tgr", o, wo_a)

    block_t, block_r, block_h = _block_sizes(T, R, D, is_decode)
    grid = (triton.cdiv(T, block_t), triton.cdiv(R, block_r), G)
    _wo_a_bf16_einsum_rope_kernel[grid](
        o,
        wo_a,
        freqs_real,
        positions,
        out,
        T,
        G,
        R,
        D,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        wo_a.stride(0),
        wo_a.stride(1),
        wo_a.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        FREQ_STRIDE=freq_stride,
        BLOCK_T=block_t,
        BLOCK_R=block_r,
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=1,
    )
    return out
