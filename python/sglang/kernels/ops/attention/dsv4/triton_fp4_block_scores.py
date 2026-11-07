"""Portable Triton fp4 block scores for the DSV4.1 low-ratio indexer.

The v4.1 indexer keeps its key rows as packed nvfp4 (two e2m1 nibbles per byte)
with four ue8m0 exponents per 128-value row, exactly what
``DeepSeekV4IndexerPool.get_low_ratio_index_k_fp4`` hands back. The torch
reference path dequantizes those rows to bf16 and scores every visible
candidate per query row, which is O(rows x lc) of *materialized* rows and
scores -- at a 250K-token prefill that is ~288 MB of K plus a [rows, lc] score
matrix, and it OOMs deep into the prompt (measured on dsv41-1017, 2x3090).

This kernel decodes the nibbles in registers and reduces each alignment block
of candidates to its max score, which is all ``select_candidate_block_ids``
needs. Nothing of size ``rows x lc`` is ever materialized: the payload is
``lc x 68`` bytes and the output is ``rows x ceil(lc / BLOCK)`` fp32.

Semantics mirror the reference math exactly:

    score[row, c] = sum_h relu(q[row, h, :] . k[c, :]) * w[row, h]
    k[c, :]       = fp4(nibble) * 2 ** (exp[c, dim // 32] - 127)

so a parity test against the dequantized path is a strict comparison, not an
approximation.

Rows of one request share the candidate buffer and differ only in how far
their window reaches (``starts`` equal, ``ends`` per row), so a tile of rows
fans out over the head axis of a single dot: the tile is
``[ROWS * heads, HEAD] x [HEAD, BLOCK_K]``. A per-row program would leave a
``[heads, HEAD]`` dot per launch and measured 3.5 TFLOP/s on a 3090.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_value(nib):
    """e2m1 nibble magnitude: 0, .5, 1, 1.5, 2, 3, 4, 6 (sign is bit 3)."""
    m = nib & 0x7
    small = m.to(tl.float32) * 0.5                       # m = 0, 1
    big = tl.exp2((m >> 1).to(tl.float32) - 1.0) * (
        1.0 + 0.5 * (m & 1).to(tl.float32)
    )                                                     # m = 2..7
    mag = tl.where(m < 2, small, big)
    neg = (nib & 0x8) != 0
    return tl.where(neg, -mag, mag)


@triton.jit
def _fp4_block_scores_kernel(
    Q_ptr,
    P_ptr,
    S_ptr,
    W_ptr,
    OUT_ptr,
    starts_ptr,
    ends_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_p_n,
    stride_s_n,
    stride_w_t,
    stride_w_h,
    stride_o_t,
    stride_o_b,
    rows,
    NUM_HEADS: tl.constexpr,
    HEAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    NUM_BLOCKS,
    SPLITS,
):
    """A ``[ROWS * heads, HEAD]`` row tile against one candidate block.

    Programs are ``(row tile, block split)``; each walks its split's blocks.
    Blocks a tile cannot reach (all rows' windows end earlier) are skipped and
    keep the buffer's ``-inf``.
    """
    r0 = tl.program_id(0) * ROWS
    split = tl.program_id(1)

    r = r0 + tl.arange(0, ROWS)
    rmask = r < rows
    start = tl.load(starts_ptr + r0)
    window = tl.load(ends_ptr + r, mask=rmask, other=0) - start
    maxwin = tl.max(window)

    m = tl.arange(0, ROWS * NUM_HEADS)
    mr = m // NUM_HEADS
    mh = m % NUM_HEADS
    d = tl.arange(0, HEAD)
    q = tl.load(
        Q_ptr
        + (r0 + mr)[:, None] * stride_q_t
        + mh[:, None] * stride_q_h
        + d[None, :] * stride_q_d,
        mask=((r0 + mr) < rows)[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    w = tl.load(
        W_ptr + (r0 + mr) * stride_w_t + mh * stride_w_h,
        mask=(r0 + mr) < rows,
        other=0.0,
    ).to(tl.float32)

    v = tl.arange(0, HEAD)
    vbyte = (v // 2).to(tl.int32)
    vshift = ((v % 2) * 4).to(tl.uint8)
    vgrp = (v // 32).to(tl.int32)

    for blk in tl.range(split, NUM_BLOCKS, SPLITS):
        c0 = blk * BLOCK_K
        if c0 < maxwin:
            k = c0 + tl.arange(0, BLOCK_K)
            live = k[None, :] < window[:, None]
            kmask = k < maxwin
            cand = start + k
            pbyte = tl.load(
                P_ptr + cand[:, None] * stride_p_n + vbyte[None, :],
                mask=kmask[:, None],
                other=0,
            )
            nib = (pbyte >> vshift[None, :]) & 0xF
            scale_byte = tl.load(
                S_ptr + cand[:, None] * stride_s_n + vgrp[None, :],
                mask=kmask[:, None],
                other=0,
            )
            kv = _fp4_value(nib) * tl.exp2(scale_byte.to(tl.float32) - 127.0)
            s = tl.dot(q, tl.trans(kv.to(tl.bfloat16)))       # [ROWS*H, BLOCK_K]
            s = tl.maximum(s, 0.0) * w[:, None]
            sc = tl.sum(tl.reshape(s, (ROWS, NUM_HEADS, BLOCK_K)), axis=1)
            sc = tl.where(live, sc, -float("inf"))
            tl.store(
                OUT_ptr + r * stride_o_t + blk * stride_o_b,
                tl.max(sc, axis=1),
                mask=rmask,
            )


def _row_tile(rows: int) -> int:
    """Rows per program: power of two, at most 8 (32->128 rows of q per tile)."""
    tile = 8
    while tile > 1 and tile > rows:
        tile //= 2
    return max(1, tile)


def triton_fp4_block_scores(
    q: torch.Tensor,
    payload: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    num_blocks: int,
    block_size: int = 128,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-(row, candidate block) max score.

    Args:
        q: ``(rows, heads, 128)`` bf16 query (the indexer's own output).
        payload: ``(n_candidates, 64)`` uint8 packed e2m1 nibbles.
        scales: ``(n_candidates, 4)`` uint8 ue8m0 exponents.
        weights: ``(rows, heads)`` fp32 head weights.
        starts/ends: ``(rows,)`` int32 candidate window (row-relative). All rows
            must share one ``start``: that is what makes the candidate block
            shared across the row tile.
        num_blocks: ``ceil(max_window / block_size)``; the output width.
        out: optional ``(rows, num_blocks)`` fp32 buffer.

    Returns:
        ``(rows, num_blocks)`` fp32 block maxima, ``-inf`` past each row's window.
    """
    rows, heads, head = q.shape
    device = q.device
    if out is None:
        out = torch.full(
            (rows, num_blocks), -float("inf"), dtype=torch.float32, device=device
        )
    if rows == 0 or num_blocks == 0:
        return out

    lo = int(starts.min())
    if lo != int(starts.max()):
        raise ValueError(
            "fp4 block scores need one shared window start across the batch; "
            f"got [{lo}, {int(starts.max())}]"
        )

    tile = _row_tile(rows)
    row_tiles = triton.cdiv(rows, tile)
    # Enough programs to fill the SMs a few times over, never more splits than
    # blocks (an empty split is a wasted launch).
    splits = max(1, min(num_blocks, triton.cdiv(4 * torch.cuda.get_device_properties(
        device).multi_processor_count, row_tiles)))

    _fp4_block_scores_kernel[(row_tiles, splits)](
        q,
        payload.view(torch.uint8),
        scales.view(torch.uint8),
        weights,
        out,
        starts.to(torch.int32),
        ends.to(torch.int32),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        payload.stride(0),
        scales.stride(0),
        weights.stride(0),
        weights.stride(1),
        out.stride(0),
        out.stride(1),
        rows,
        NUM_HEADS=heads,
        HEAD=head,
        BLOCK_K=block_size,
        ROWS=tile,
        NUM_BLOCKS=num_blocks,
        SPLITS=splits,
        num_warps=8,
    )
    return out
