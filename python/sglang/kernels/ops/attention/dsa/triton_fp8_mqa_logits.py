"""Portable Triton non-paged FP8 MQA logits for the DSA indexer.

This implements the semantics of ``deep_gemm.fp8_mqa_logits`` (which only has
SM90/SM100/SM120 cubins) so that the DSA kpool indexer can run on SM80/SM86/
SM89 (Ampere/Ada) where DeepGEMM is unavailable. The math mirrors the portable
AITER Triton reference (``aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py``):

    logits[row, kv] = sum_h relu(q[row, h, :] . k[kv, :]) * kv_scales[kv] * w[row, h]

for ``kv`` in ``[starts[row], ends[row])``. The indexer key ``k`` is shared
across heads (multi-query indexer), so it is a 2D ``(kv_len, head_dim)``
tensor. Only the valid ``[starts[row], ends[row])`` window is written; the
rest of each output row is left untouched (garbage, masked by the caller).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fp8_e4m3_to_f16(b):
    """Decode an NVIDIA fp8 e4m3 byte (as uint8/uint32) into fp16.

    SM80/SM86/SM89 (Ampere/Ada) expose no fp8 tensor cores and their ptxas
    rejects the fp8 ``cvt`` PTX, so we decode with pure integer/float math
    (E4M3: 1 sign + 4 exp + 3 mantissa, bias 7; e==0 is subnormal).
    """
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    exp2 = tl.exp2(e.to(tl.float32) - 7.0)
    val_normal = exp2 * (1.0 + m.to(tl.float32) / 8.0)
    val_sub = m.to(tl.float32) * 1.953125e-3  # 2^-9
    val = tl.where(e == 0, val_sub, val_normal)
    sign = tl.where((s & 1) == 1, -1.0, 1.0)
    return (sign * val).to(tl.float16)


@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,
    KV_ptr,
    kv_scales_ptr,
    weights_ptr,
    cu_start_ptr,
    cu_end_ptr,
    logits_ptr,
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    stride_q_s: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.constexpr,
    stride_logits_k: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )
    q_block = _fp8_e4m3_to_f16(tl.load(q_ptrs, cache_modifier=".cg").to(tl.uint32))

    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)
    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )
    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets
    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = _fp8_e4m3_to_f16(tl.load(kv_ptrs).to(tl.uint32))
        kv_scales = tl.load(kv_scales_ptrs)

        scores = tl.dot(q_block, kv_block)
        scores = scores * kv_scales[None, :]
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    kv_col_mask = kv_col_offsets < end_ind
    kv_block = _fp8_e4m3_to_f16(
        tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0).to(tl.uint32)
    )
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    scores = tl.dot(q_block, kv_block)
    scores = scores * kv_scales[None, :]
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    scores = tl.sum(scores, axis=0)

    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    Q_ptr,
    KV_u8_ptr,
    W_ptr,
    seq_lens_ptr,
    page_table_ptr,
    logits_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_w_s,
    stride_w_h,
    stride_pt_s,
    stride_pt_p,
    stride_logits_s,
    stride_logits_k,
    BLOCK_BYTES: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # One program per query row. KV is a page-granular pooled cache where each
    # page holds BLOCK_S=64 tokens; token t in page p has its fp8 key bytes at
    # KV[p][t*HEAD_SIZE : (t+1)*HEAD_SIZE] and its fp32 scale (4 bytes) at
    # KV[p][BLOCK_S*HEAD_SIZE + t*4 : +4]. logits[bx, s] for s in
    # [0, seq_lens[bx]) uses page = page_table[bx, s//BLOCK_S].
    bx = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + bx)
    n_pages = tl.cdiv(seq_len, BLOCK_S)

    h = tl.arange(0, NUM_HEADS)[:, None]
    d = tl.arange(0, HEAD_SIZE)
    q = _fp8_e4m3_to_f16(
        tl.load(
            Q_ptr + bx * stride_q_s + h * stride_q_h + d[None, :] * stride_q_d
        ).to(tl.uint32)
    )
    w = tl.load(W_ptr + bx * stride_w_s + h * stride_w_h).to(tl.float32)

    t = tl.arange(0, BLOCK_S)
    base = bx * stride_logits_s
    for pi in tl.range(0, n_pages):
        page = tl.load(page_table_ptr + bx * stride_pt_s + pi * stride_pt_p)
        page_bytes = page * BLOCK_BYTES

        kp = KV_u8_ptr + page_bytes + t[:, None] * HEAD_SIZE + d[None, :]
        key = _fp8_e4m3_to_f16(
            tl.load(kp, mask=(t[:, None] < BLOCK_S), other=0)
        )

        sb = KV_u8_ptr + page_bytes + SCALE_OFFSET + t * 4
        b0 = tl.load(sb, mask=(t < BLOCK_S), other=0)
        b1 = tl.load(sb + 1, mask=(t < BLOCK_S), other=0)
        b2 = tl.load(sb + 2, mask=(t < BLOCK_S), other=0)
        b3 = tl.load(sb + 3, mask=(t < BLOCK_S), other=0)
        s_u32 = (
            b0.to(tl.uint32)
            | (b1.to(tl.uint32) << 8)
            | (b2.to(tl.uint32) << 16)
            | (b3.to(tl.uint32) << 24)
        )
        scale = s_u32.to(tl.float32, bitcast=True)

        scores = tl.dot(key, tl.trans(q))
        scores = tl.maximum(scores, 0.0)
        scores = scores * tl.trans(w)
        scores = tl.sum(scores, axis=1)
        scores = scores * scale

        out_off = pi * BLOCK_S + t
        mask = out_off < seq_len
        tl.store(logits_ptr + base + out_off * stride_logits_k, scores, mask=mask)


def triton_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: object,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Portable Triton replacement for the kpool paged FP8 MQA logits.

    Mirrors ``tilelang_fp8_paged_mqa_logits`` semantics (which fails on SM8x
    because its CuTe codegen needs the SM89 F32 MMA): a page-granular pooled
    KV cache where each page holds 64 tokens of fp8 keys + fp32 scales, and the
    pooled page table indexes those pages.

    Args:
        q_fp8: (batch, 1, num_heads, head_dim) float8_e4m3fn.
        kvcache_fp8: (num_blocks, 64, 1, head_dim + 4) float8_e4m3fn.
        weights: (batch, num_heads) float32.
        seq_lens: (batch,) int32 pooled seq len per query.
        page_table: (batch, num_pages) int32 pooled page indices.
        deep_gemm_metadata: unused (API parity).
        max_seq_len: output row width.
        clean_logits: accepted for API parity (top-k is order-invariant).

    Returns:
        logits: (batch, max_seq_len) float32.
    """
    batch, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128
    assert block_size == 64
    assert q_fp8.shape == (batch, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weights.shape == (batch, num_heads)
    assert seq_lens.shape == (batch,)
    assert page_table.shape[0] == batch

    logits = torch.empty((batch, max_seq_len), dtype=torch.float32, device=q_fp8.device)
    if batch == 0 or max_seq_len == 0:
        return logits

    block_bytes = block_size * (head_dim + 4)
    kv_u8 = kvcache_fp8.contiguous().view(torch.uint8)
    q = q_fp8.contiguous().view(batch, num_heads, head_dim).view(torch.uint8)
    _fp8_paged_mqa_logits_kernel[(batch,)](
        q,
        kv_u8,
        weights,
        seq_lens.to(torch.int32),
        page_table,
        logits,
        stride_q_s=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_w_s=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_pt_s=page_table.stride(0),
        stride_pt_p=page_table.stride(1),
        stride_logits_s=logits.stride(0),
        stride_logits_k=logits.stride(1),
        BLOCK_BYTES=block_bytes,
        SCALE_OFFSET=block_size * head_dim,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_dim,
        BLOCK_S=block_size,
    )
    return logits


def triton_fp8_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Portable Triton replacement for ``deep_gemm.fp8_mqa_logits``.

    Args:
        q: (num_queries, num_heads, head_dim) float8_e4m3fn.
        k: (total_k_rows, head_dim) float8_e4m3fn, shared across heads.
        k_scale: (total_k_rows,) float32 per-row scale.
        weights: (num_queries, num_heads) float32 per-head weight.
        starts/ends: (num_queries,) int32; row i uses k rows [starts[i], ends[i]).
        clean_logits: accepted for API parity; the raw relu'd scores are
            produced either way (matches the portable AITER semantics, and the
            caller selects top-k which is order-invariant).

    Returns:
        logits: (num_queries, total_k_rows) float32.
    """
    assert q.dtype == torch.float8_e4m3fn, q.dtype
    assert k.dtype == torch.float8_e4m3fn, k.dtype
    assert k_scale.dtype == torch.float32, k_scale.dtype
    assert weights.dtype == torch.float32, weights.dtype
    assert q.ndim == 3 and k.ndim == 2
    num_queries, num_heads, head_dim = q.shape
    total_k_rows = k.shape[0]
    assert k.shape[1] == head_dim
    assert k_scale.shape[0] == total_k_rows
    assert weights.shape == (num_queries, num_heads)
    assert starts.shape == (num_queries,) and ends.shape == (num_queries,)
    assert head_dim % 16 == 0, "head_dim must be a multiple of 16 for tl.dot"

    logits = torch.empty(
        (num_queries, total_k_rows), dtype=torch.float32, device=q.device
    )
    if num_queries == 0 or total_k_rows == 0:
        return logits

    BLOCK_KV = 64
    grid = (num_queries,)
    _fp8_mqa_logits_kernel[grid](
        q.contiguous().view(torch.uint8),
        k.contiguous().view(torch.uint8),
        k_scale,
        weights,
        starts.to(torch.int32),
        ends.to(torch.int32),
        logits,
        num_queries,
        total_k_rows,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_dim,
        stride_q_s=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_kv_s=k.stride(0),
        stride_kv_d=k.stride(1),
        stride_w_s=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_logits_s=logits.stride(0),
        stride_logits_k=logits.stride(1),
        BLOCK_KV=BLOCK_KV,
    )
    return logits
