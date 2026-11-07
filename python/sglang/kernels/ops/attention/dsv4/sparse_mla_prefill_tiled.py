"""Tiled Triton sparse-MLA prefill for the Ampere (SM8x) route.

The TileLang v1 kernel this replaces only fills two replicated CTAs on SM86, so
long prefills run it at a small fraction of the GPU. This is a port of the
vllm-ds4 tiled formulation: one program owns one query x ``HEAD_BLOCK`` heads,
each KV row is read once per head block instead of once per (query, head), and
the fp32 online-softmax state is chunked over queries and top-k candidates to
keep it in L2. No score matrix is materialized.

Contract matches ``tilelang_sparse_fwd`` as the prefill path calls it:

* ``q``        ``(T, H, D)`` -- D is the packed q/k width (qk_nope + qk_rope).
* ``kv_flat``  ``(R, D)``   -- the gathered bf16 workspace rows.
* ``indices``  ``(T, K)``   -- flat row ids into ``kv_flat``, valid entries
  first, ``-1`` for padding.
* ``lens``     ``(T,)``     -- valid candidate count per query.
* value dim ``d_v`` <= D    -- MLA values are the leading ``d_v`` columns of the
  same latent row, so the score uses all D and the accumulator only ``d_v``.
* ``sink``     ``(H,)``     -- attention sink, merged with zero value.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Same tiling as the vllm-ds4 Triton prefill path.
QUERY_CHUNK_SIZE = 256
TOPK_CHUNK_SIZE = 512
# 32 heads per program: one program owns a whole token, so a KV row is read once
# instead of four times (the 8-head mapping this replaced), and M=32 fills the
# tensor-core tile. Measured on a 3090 at T=32K/topk=2176: 2.46 s -> 0.29 s.
HEAD_BLOCK = 32
# Candidates per score/value tile. 32 keeps the [CAND, d_v] value tile and the
# fp32 accumulator inside one CTA's registers; 16 measured 0.37 s, 32 measured
# 0.29 s.
CAND_BLOCK = 32
# Reduction piece for q.k; 576 is 9 x 64 and not a power of two, so a 64-piece
# loop avoids padding the reduction to 1024.
D_CHUNK = 128

_state_buffers: dict = {}


def _get_state_buffers(device, num_heads: int, d_v: int):
    key = (device, num_heads, d_v)
    buf = _state_buffers.get(key)
    if buf is None:
        buf = (
            torch.empty(QUERY_CHUNK_SIZE, num_heads, dtype=torch.float32, device=device),
            torch.empty(QUERY_CHUNK_SIZE, num_heads, dtype=torch.float32, device=device),
            torch.empty(
                QUERY_CHUNK_SIZE, num_heads, d_v, dtype=torch.float32, device=device
            ),
        )
        _state_buffers[key] = buf
    return buf


@triton.jit
def _accumulate_multihead_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    lens_ptr,
    max_ptr,
    denom_ptr,
    acc_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_r,
    stride_kv_d,
    stride_idx_t,
    stride_idx_c,
    stride_st_t,
    stride_st_h,
    stride_acc_t,
    stride_acc_h,
    stride_acc_d,
    num_heads,
    head_dim,
    d_v: tl.constexpr,
    num_candidates,
    candidate_offset,
    scale,
    HEAD_BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CAND_BLOCK: tl.constexpr,
    D_CHUNK: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    heads = head_block_idx * HEAD_BLOCK_T + tl.arange(0, HEAD_BLOCK_T)
    vals = tl.arange(0, BLOCK_V)
    cands = tl.arange(0, CAND_BLOCK)
    dchunk = tl.arange(0, D_CHUNK)
    head_mask = heads < num_heads
    val_mask = vals < d_v

    st_base = token_idx * stride_st_t
    running_max = tl.load(
        max_ptr + st_base + heads * stride_st_h, mask=head_mask, other=float("-inf")
    )
    running_denom = tl.load(
        denom_ptr + st_base + heads * stride_st_h, mask=head_mask, other=0.0
    )
    acc_base = token_idx * stride_acc_t
    running_acc = tl.load(
        acc_ptr
        + acc_base
        + heads[:, None] * stride_acc_h
        + vals[None, :] * stride_acc_d,
        mask=head_mask[:, None] & val_mask[None, :],
        other=0.0,
    )

    valid_len = tl.load(lens_ptr + token_idx)
    local_eff = tl.minimum(num_candidates, tl.maximum(valid_len - candidate_offset, 0))

    # Candidates are taken CAND_BLOCK at a time. One-at-a-time left a single
    # dependent chain per program (load a row, score it, rescale the state) which
    # measured 133 GB/s on a 3090 -- 14% of peak -- and neither the tiling knobs
    # nor HEAD_BLOCK changed that. A block of rows loads in parallel and both the
    # score and the value product become tl.dot.
    for c0 in range(0, local_eff, CAND_BLOCK):
        c = c0 + cands
        keep = c < local_eff
        rows = tl.load(
            idx_ptr + token_idx * stride_idx_t + c * stride_idx_c,
            mask=keep,
            other=-1,
        )
        keep = keep & (rows >= 0)
        rows64 = tl.where(keep, rows, 0).to(tl.int64)

        # q.k over the packed width in D_CHUNK pieces; D is 576 here, which is not
        # a power of two, so this avoids padding the reduction to 1024 and doing
        # 1.8x the work.
        s = tl.zeros((HEAD_BLOCK_T, CAND_BLOCK), dtype=tl.float32)
        for d0 in range(0, head_dim, D_CHUNK):
            dd = d0 + dchunk
            dmask = dd < head_dim
            qc = tl.load(
                q_ptr
                + token_idx * stride_q_t
                + heads[:, None] * stride_q_h
                + dd[None, :] * stride_q_d,
                mask=head_mask[:, None] & dmask[None, :],
                other=0.0,
            )
            kc = tl.load(
                kv_ptr
                + rows64[:, None] * stride_kv_r
                + dd[None, :] * stride_kv_d,
                mask=keep[:, None] & dmask[None, :],
                other=0.0,
            )
            s = tl.dot(qc, tl.trans(kc), s)
        s = s * scale
        s = tl.where(keep[None, :], s, float("-inf"))

        tile_max = tl.max(s, axis=1)
        next_max = tl.maximum(running_max, tile_max)
        # A token whose visible window is still empty keeps -inf here; folding it
        # to 0 keeps exp() finite (the accumulator and normalizer are both 0).
        next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
        prev_w = tl.exp(running_max - next_max)
        cand_w = tl.exp(s - next_max[:, None])

        v = tl.load(
            kv_ptr + rows64[:, None] * stride_kv_r + vals[None, :] * stride_kv_d,
            mask=keep[:, None] & val_mask[None, :],
            other=0.0,
        )
        running_acc = running_acc * prev_w[:, None] + tl.dot(
            cand_w.to(tl.bfloat16), v
        )
        running_denom = running_denom * prev_w + tl.sum(cand_w, axis=1)
        running_max = next_max

    tl.store(max_ptr + st_base + heads * stride_st_h, running_max, mask=head_mask)
    tl.store(denom_ptr + st_base + heads * stride_st_h, running_denom, mask=head_mask)
    tl.store(
        acc_ptr
        + acc_base
        + heads[:, None] * stride_acc_h
        + vals[None, :] * stride_acc_d,
        running_acc,
        mask=head_mask[:, None] & val_mask[None, :],
    )


@triton.jit
def _finish_kernel(
    max_ptr,
    denom_ptr,
    acc_ptr,
    sink_ptr,
    out_ptr,
    stride_st_t,
    stride_st_h,
    stride_acc_t,
    stride_acc_h,
    stride_acc_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    num_heads,
    d_v: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % num_heads
    token_idx = row // num_heads
    vals = tl.arange(0, BLOCK_V)
    val_mask = vals < d_v

    st_off = token_idx * stride_st_t + head * stride_st_h
    m = tl.load(max_ptr + st_off)
    d = tl.load(denom_ptr + st_off)
    acc = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head * stride_acc_h
        + vals * stride_acc_d,
        mask=val_mask,
        other=0.0,
    )
    # Sink carries zero value: it only joins the normalizer.
    sink = tl.load(sink_ptr + head)
    sink_w = tl.exp(sink - m)
    z = d + sink_w
    out = acc / z
    tl.store(
        out_ptr + token_idx * stride_o_t + head * stride_o_h + vals * stride_o_d,
        out,
        mask=val_mask,
    )


def sparse_mla_prefill_tiled(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    d_v: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sink-aware sparse MLA over gathered bf16 rows; returns ``(T, H, d_v)``."""
    assert q.dim() == 3, q.shape
    # The workspace is allocated as (rows, 1, D); a flat (rows, D) view is what
    # the kernels index. Either layout is accepted, and a non-unit last stride
    # is carried through as a runtime stride (no copy).
    if kv_flat.dim() == 3:
        assert kv_flat.shape[1] == 1, kv_flat.shape
        kv_flat = kv_flat[:, 0]
    assert kv_flat.dim() == 2, kv_flat.shape
    assert indices.dim() == 2 and indices.dtype in (torch.int32, torch.int64)
    num_tokens, num_heads, head_dim = q.shape
    num_candidates = indices.shape[-1]
    if output is None:
        output = torch.empty(
            (num_tokens, num_heads, d_v), dtype=q.dtype, device=q.device
        )
    block_v = triton.next_power_of_2(d_v)
    sink = attn_sink[:num_heads] if attn_sink.dim() == 1 else attn_sink
    sink = sink.contiguous()
    max_s, denom, acc = _get_state_buffers(q.device, num_heads, d_v)

    for t0 in range(0, num_tokens, QUERY_CHUNK_SIZE):
        t1 = min(t0 + QUERY_CHUNK_SIZE, num_tokens)
        rows = t1 - t0
        q_c = q[t0:t1]
        idx_c = indices[t0:t1]
        lens_c = lens[t0:t1]
        st_max, st_den, st_acc = max_s[:rows], denom[:rows], acc[:rows]
        st_max.fill_(float("-inf"))
        st_den.zero_()
        st_acc.zero_()

        for c0 in range(0, num_candidates, TOPK_CHUNK_SIZE):
            c1 = min(c0 + TOPK_CHUNK_SIZE, num_candidates)
            cand = idx_c[:, c0:c1]
            grid = (rows, triton.cdiv(num_heads, HEAD_BLOCK))
            _accumulate_multihead_kernel[grid](
                q_c,
                kv_flat,
                cand,
                lens_c,
                st_max,
                st_den,
                st_acc,
                q_c.stride(0),
                q_c.stride(1),
                q_c.stride(2),
                kv_flat.stride(0),
                kv_flat.stride(1),
                cand.stride(0),
                cand.stride(1),
                st_max.stride(0),
                st_max.stride(1),
                st_acc.stride(0),
                st_acc.stride(1),
                st_acc.stride(2),
                num_heads,
                head_dim,
                d_v,
                c1 - c0,
                c0,
                scale,
                HEAD_BLOCK_T=HEAD_BLOCK,
                BLOCK_V=block_v,
                CAND_BLOCK=CAND_BLOCK,
                D_CHUNK=D_CHUNK,
                num_warps=8,
            )

        out_c = output[t0:t1]
        _finish_kernel[(rows * num_heads,)](
            st_max,
            st_den,
            st_acc,
            sink,
            out_c,
            st_max.stride(0),
            st_max.stride(1),
            st_acc.stride(0),
            st_acc.stride(1),
            st_acc.stride(2),
            out_c.stride(0),
            out_c.stride(1),
            out_c.stride(2),
            num_heads,
            d_v,
            BLOCK_V=block_v,
            num_warps=4,
        )
    return output
