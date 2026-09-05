"""Sink-aware sparse MLA attention over materialized bf16 KV.

Ported from vllm-ds4 ``vllm/v1/attention/backends/mla/sparse_mla_kernels.py``.
Computes scores with a batched bmm then finishes the sink-aware online-softmax
value reduction in Triton. Used on the SM80/SM86 decode path where the
TileLang ``sparse_attention_fwd_kernel_v1`` (2 replicated blocks) is
poorly-occupied; bmm + Triton finish is ~3x faster and numerically equivalent.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _finish_materialized_scores_with_sink_kernel(
    scores_ptr,
    kv_ptr,
    valid_tokens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_c: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_valid_t: tl.constexpr,
    stride_valid_c: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    running_max = tl.load(
        attn_sink_ptr + head_offsets, mask=head_mask, other=0.0
    ).to(tl.float32)
    running_denom = tl.full((HEAD_BLOCK,), 1.0, tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    for candidate_idx in range(0, num_candidates):
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidate_idx * stride_valid_c
        )
        if is_valid:
            score = tl.load(
                scores_ptr
                + token_idx * stride_scores_t
                + head_offsets * stride_scores_h
                + candidate_idx * stride_scores_c,
                mask=head_mask,
                other=-float("inf"),
            ).to(tl.float32)
            kv = tl.load(
                kv_ptr
                + token_idx * stride_kv_t
                + candidate_idx * stride_kv_c
                + dim_offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    result = running_acc / running_denom[:, None]
    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_offsets[:, None] * stride_out_h
        + dim_offsets[None, :] * stride_out_d,
        result,
        mask=matrix_mask,
    )


@triton.jit
def _finish_materialized_scores_with_sink_candidate_block_kernel(
    scores_ptr,
    kv_ptr,
    valid_tokens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_c: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_valid_t: tl.constexpr,
    stride_valid_c: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    candidate_offsets = tl.arange(0, BLOCK_K)
    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim

    max_score = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    for candidate_start in range(0, num_candidates, BLOCK_K):
        candidates = candidate_start + candidate_offsets
        candidate_mask = candidates < num_candidates
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidates * stride_valid_c,
            mask=candidate_mask,
            other=0,
        ).to(tl.int1)
        scores = tl.load(
            scores_ptr
            + token_idx * stride_scores_t
            + head_idx * stride_scores_h
            + candidates * stride_scores_c,
            mask=candidate_mask & is_valid,
            other=-float("inf"),
        ).to(tl.float32)
        max_score = tl.maximum(max_score, tl.max(scores, axis=0))

    denom = tl.exp(tl.load(attn_sink_ptr + head_idx).to(tl.float32) - max_score)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for candidate_start in range(0, num_candidates, BLOCK_K):
        candidates = candidate_start + candidate_offsets
        candidate_mask = candidates < num_candidates
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidates * stride_valid_c,
            mask=candidate_mask,
            other=0,
        ).to(tl.int1)
        scores = tl.load(
            scores_ptr
            + token_idx * stride_scores_t
            + head_idx * stride_scores_h
            + candidates * stride_scores_c,
            mask=candidate_mask & is_valid,
            other=-float("inf"),
        ).to(tl.float32)
        weights = tl.exp(scores - max_score)
        denom += tl.sum(weights, axis=0)
        kv = tl.load(
            kv_ptr
            + token_idx * stride_kv_t
            + candidates[:, None] * stride_kv_c
            + dim_offsets[None, :] * stride_kv_d,
            mask=(candidate_mask & is_valid)[:, None] & dim_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(kv.to(tl.float32) * weights[:, None], axis=0)

    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_idx * stride_out_h
        + dim_offsets * stride_out_d,
        acc / denom,
        mask=dim_mask,
    )


@triton.jit
def _finish_materialized_scores_with_sink_value_block_kernel(
    scores_ptr,
    kv_ptr,
    valid_tokens_ptr,
    attn_sink_ptr,
    output_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_c: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_valid_t: tl.constexpr,
    stride_valid_c: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)
    dim_offsets = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim

    running_max = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
    running_denom = tl.full((), 1.0, tl.float32)
    running_acc = tl.zeros((BLOCK_D,), tl.float32)

    for candidate_idx in range(0, num_candidates):
        is_valid = tl.load(
            valid_tokens_ptr
            + token_idx * stride_valid_t
            + candidate_idx * stride_valid_c
        )
        if is_valid:
            score = tl.load(
                scores_ptr
                + token_idx * stride_scores_t
                + head_idx * stride_scores_h
                + candidate_idx * stride_scores_c
            ).to(tl.float32)
            kv = tl.load(
                kv_ptr
                + token_idx * stride_kv_t
                + candidate_idx * stride_kv_c
                + dim_offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = running_acc * previous_weight + kv * candidate_weight
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    result = running_acc / running_denom
    tl.store(
        output_ptr
        + token_idx * stride_out_t
        + head_idx * stride_out_h
        + dim_offsets * stride_out_d,
        result,
        mask=dim_mask,
    )


def finish_materialized_sparse_mla_scores_with_sink(
    scores: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    num_heads: int | None = None,
    head_block_size: int = 1,
    value_block_size: int | None = None,
    candidate_block_size: int | None = None,
) -> None:
    assert scores.dim() == 3
    assert kv.dim() == 3
    assert valid_tokens.shape == kv.shape[:2]
    assert scores.shape[0] == kv.shape[0]
    assert scores.shape[2] == kv.shape[1]
    assert output.shape[0] == kv.shape[0]
    assert output.shape[2] == kv.shape[2]
    assert scores.dtype in (torch.float32, torch.bfloat16)
    assert head_block_size in (1, 2, 4)
    if value_block_size is not None:
        assert value_block_size in (64, 128, 256, 512)
    if candidate_block_size is not None:
        assert candidate_block_size in (16, 32, 64, 128)
    assert scores.is_cuda and kv.is_cuda and valid_tokens.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    active_heads = num_heads if num_heads is not None else output.shape[1]
    assert active_heads <= scores.shape[1]
    assert active_heads <= output.shape[1]
    assert active_heads <= attn_sink.shape[0]

    num_tokens, _, num_candidates = scores.shape
    head_dim = kv.shape[2]

    def _smallest_block_d_covering(hd: int) -> int:
        for cand in (64, 128, 256, 512):
            if cand >= hd:
                return cand
        return 512

    if candidate_block_size is not None:
        target_block_d = _smallest_block_d_covering(head_dim)
        if value_block_size is None:
            block_d = target_block_d
        else:
            block_d = min(value_block_size, target_block_d)
        candidate_grid = (num_tokens, active_heads, triton.cdiv(head_dim, block_d))
        _finish_materialized_scores_with_sink_candidate_block_kernel[candidate_grid](
            scores,
            kv,
            valid_tokens,
            attn_sink,
            output,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
            valid_tokens.stride(0),
            valid_tokens.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            head_dim,
            num_candidates,
            BLOCK_K=candidate_block_size,
            BLOCK_D=block_d,
            num_warps=8,
        )
        if output.shape[1] > active_heads:
            output[:, active_heads:].zero_()
        return

    if value_block_size is not None and value_block_size < head_dim:
        value_grid = (
            num_tokens,
            active_heads,
            triton.cdiv(head_dim, value_block_size),
        )
        _finish_materialized_scores_with_sink_value_block_kernel[value_grid](
            scores,
            kv,
            valid_tokens,
            attn_sink,
            output,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
            valid_tokens.stride(0),
            valid_tokens.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            head_dim,
            num_candidates,
            BLOCK_D=value_block_size,
            num_warps=4,
        )
        if output.shape[1] > active_heads:
            output[:, active_heads:].zero_()
        return

    block_d = min(1024, next_power_of_2(head_dim))
    head_grid = (num_tokens, triton.cdiv(active_heads, head_block_size))
    _finish_materialized_scores_with_sink_kernel[head_grid](
        scores,
        kv,
        valid_tokens,
        attn_sink,
        output,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        valid_tokens.stride(0),
        valid_tokens.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        active_heads,
        head_dim,
        num_candidates,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        num_warps=8,
    )
    if output.shape[1] > active_heads:
        output[:, active_heads:].zero_()


def matmul_sparse_mla_attention_with_sink(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    num_heads: int | None = None,
    score_buffer: torch.Tensor | None = None,
    head_block_size: int = 1,
    value_block_size: int | None = None,
    candidate_block_size: int | None = None,
) -> None:
    """Compute sink-aware sparse MLA over materialized BF16 KV.

    Dequantizes/gathers KV once, computes scores with a batched bmm, and
    finishes the sink-aware value reduction in Triton. Used on the SM80/SM86
    decode path (ported from vllm-ds4).
    """
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv.dim() == 3, f"Expected kv shape [T, K, D], got {kv.shape}"
    assert valid_tokens.shape == kv.shape[:2]
    assert q.shape[0] == kv.shape[0]
    assert q.shape[-1] == kv.shape[-1]
    assert output.shape[0] == q.shape[0]
    assert output.shape[2] == q.shape[-1]
    assert q.is_cuda and kv.is_cuda and valid_tokens.is_cuda
    assert attn_sink.is_cuda and output.is_cuda

    active_heads = num_heads if num_heads is not None else output.shape[1]
    assert active_heads <= q.shape[1]
    assert active_heads <= output.shape[1]
    assert active_heads <= attn_sink.shape[0]

    q_active = q[:, :active_heads]
    num_tokens = q.shape[0]
    num_candidates = kv.shape[1]
    if score_buffer is None:
        score_buffer = torch.empty(
            (num_tokens, active_heads, num_candidates),
            dtype=torch.float32,
            device=q.device,
        )
    assert score_buffer.shape == (num_tokens, active_heads, num_candidates)
    assert score_buffer.device == q.device
    assert score_buffer.dtype in (torch.float32, torch.bfloat16)
    if score_buffer.dtype == torch.float32:
        q_score = q_active.float()
        kv_score = kv.float()
    else:
        q_score = q_active.to(score_buffer.dtype)
        kv_score = kv.to(score_buffer.dtype)
    torch.bmm(q_score, kv_score.transpose(1, 2), out=score_buffer)
    score_buffer.mul_(scale)
    finish_materialized_sparse_mla_scores_with_sink(
        score_buffer,
        kv,
        valid_tokens,
        attn_sink,
        output,
        num_heads=active_heads,
        head_block_size=head_block_size,
        value_block_size=value_block_size,
        candidate_block_size=candidate_block_size,
    )
