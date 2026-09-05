from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# v4 KV cache layout (see dsv4.index_buf_accessor._set_k_and_s_triton_kernel):
#   per-token: 448 fp8 nope + 64 bf16 rope (= 576 contiguous bytes) +
#              7 ue8m0 scales padded to 8 bytes.
#   per-page:  [token 0..P-1 nope+rope (P*576 bytes)] [token 0..P-1 scale (P*8 bytes)]
#              padded up to a multiple of 576.
DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64  # one nope scale tile = 64 fp8 values
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE  # 7
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2  # 576
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1  # 8


@triton.jit
def _e4m3_uint8_to_f32(u):
    """Decode an ``e4m3fn`` byte (1-4-3, exp bias 7) to f32 without ever
    materializing Triton's ``fp8e4nv`` type, which Ampere (SM80/SM86) cannot
    represent. ``u`` is the raw uint8 bit pattern. NaN (S.1111.111) is not
    produced by quantized weights and is decoded as a finite value."""
    ui = u.to(tl.int32)
    sign = (ui >> 7) & 1
    exp = (ui >> 3) & 0xF
    man = ui & 0x7
    mant = man.to(tl.float32) * 0.125
    # normal: 2^(exp-7) * (1+mant); subnormal (exp==0): 2^-6 * mant
    val = tl.where(
        exp != 0,
        tl.exp2((exp - 7).to(tl.float32)) * (1.0 + mant),
        0.015625 * mant,
    )
    return tl.where(sign != 0, -val, val)


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize the DeepSeek v4 paged KV cache for a list of token IDs.

    Args:
        quant_k_cache: (num_pages, bytes_per_page_padded) uint8.
        page_table_1_flattened: (num_tokens,) int — token IDs into the cache.
        page_size: number of tokens per page.
        out: optional (num_tokens, 1, DIM_NOPE + DIM_ROPE) bf16 destination.
            May be a slice of a larger workspace; the kernel uses out.stride(0)
            so contiguous-along-dim-0 slices work.

    Returns:
        (num_tokens, 1, DIM_NOPE + DIM_ROPE) bfloat16.
    """
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)

    # The buffer's dtype is whatever the pool exposes (often bf16); the
    # underlying storage is uint8. Reinterpret to byte-space first.
    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    # Two typed views over the same underlying bytes.
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16

    _dequantize_k_cache_paged_kernel[(num_tokens,)](
        out,
        buf_bf16,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=s_offset_bytes,
    )
    return out


@triton.jit
def _dequantize_combined_kv_paged_kernel(
    combined_kv_ptr,
    swa_buf_bf16_ptr,
    swa_buf_uint8_ptr,
    swa_page_table_ptr,
    extra_buf_bf16_ptr,
    extra_buf_uint8_ptr,
    extra_page_table_ptr,
    swa_topk_lengths_ptr,
    extra_topk_lengths_ptr,
    valid_out_ptr,
    WRITE_VALID: tl.constexpr,
    SWA_PAGE_SIZE: tl.constexpr,
    SWA_BYTES_PER_PAGE: tl.constexpr,
    EXTRA_PAGE_SIZE: tl.constexpr,
    EXTRA_BYTES_PER_PAGE: tl.constexpr,
    N_SWA: tl.constexpr,
    TOPK: tl.constexpr,
    N_EXTRA: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    SWA_S_OFFSET_BYTES: tl.constexpr,
    EXTRA_S_OFFSET_BYTES: tl.constexpr,
):
    # One program per (query, candidate) pair. The SWA columns [0, N_SWA) come
    # from the SWA pool; the compressed columns [N_SWA, TOPK) come from the
    # extra pool (which may have a different page_size/bytes_per_page). The
    # e4m3 + ue8m0 nope and the bf16 rope tail are decoded once and written
    # directly into the gathered combined_kv[b, topk, 512] buffer.
    pid = tl.program_id(0)
    b_idx = pid // TOPK
    j = pid % TOPK
    is_extra = j >= N_SWA
    jj = tl.where(is_extra, j - N_SWA, j)

    # Fetch the token id; -1 (padded) is clamped to 0 so we never OOB-read,
    # and its row is marked invalid by the length mask below (never attended).
    loc = tl.where(
        is_extra,
        tl.load(extra_page_table_ptr + b_idx * N_EXTRA + jj),
        tl.load(swa_page_table_ptr + b_idx * N_SWA + jj),
    )
    loc = tl.maximum(loc, 0).to(tl.int64)
    page_idx = loc // tl.where(is_extra, EXTRA_PAGE_SIZE, SWA_PAGE_SIZE)
    in_page = loc % tl.where(is_extra, EXTRA_PAGE_SIZE, SWA_PAGE_SIZE)
    page_byte_base = page_idx * tl.where(
        is_extra, EXTRA_BYTES_PER_PAGE, SWA_BYTES_PER_PAGE
    )
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = page_byte_base + tl.where(
        is_extra, EXTRA_S_OFFSET_BYTES, SWA_S_OFFSET_BYTES
    ) + in_page * PADDED_SCALE_PER_TOKEN

    out_row_base = b_idx * TOPK * (DIM_NOPE + DIM_ROPE) + j * (DIM_NOPE + DIM_ROPE)

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_vals = _e4m3_uint8_to_f32(
            tl.load(
                tl.where(is_extra, extra_buf_uint8_ptr, swa_buf_uint8_ptr)
                + token_data_base
                + tile_id * TILE_SIZE
                + nope_offs
            )
        )
        scale_u8 = tl.load(
            tl.where(is_extra, extra_buf_uint8_ptr, swa_buf_uint8_ptr)
            + token_scale_base
            + tile_id
        ).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))
        tl.store(
            combined_kv_ptr + out_row_base + tile_id * TILE_SIZE + nope_offs,
            (fp8_vals * scale_pow2).to(combined_kv_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(
        tl.where(is_extra, extra_buf_bf16_ptr, swa_buf_bf16_ptr) + bf16_off
    )
    tl.store(
        combined_kv_ptr + out_row_base + DIM_NOPE + rope_offs,
        rope_data.to(combined_kv_ptr.dtype.element_ty),
    )

    if WRITE_VALID:
        is_valid = tl.where(
            is_extra,
            jj < tl.load(extra_topk_lengths_ptr + b_idx),
            jj < tl.load(swa_topk_lengths_ptr + b_idx),
        )
        tl.store(valid_out_ptr + b_idx * TOPK + j, is_valid.to(tl.int1))


def dequantize_combined_kv_paged(
    combined_kv: torch.Tensor,
    swa_quant_k_cache: torch.Tensor,
    swa_page_table: torch.Tensor,
    swa_page_size: int,
    extra_quant_k_cache: Optional[torch.Tensor] = None,
    extra_page_table: Optional[torch.Tensor] = None,
    extra_page_size: Optional[int] = None,
    swa_topk_lengths: Optional[torch.Tensor] = None,
    extra_topk_lengths: Optional[torch.Tensor] = None,
    valid_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize the SWA + compressed KV regions directly into a gathered
    ``(batch, topk, 512)`` bf16 buffer in a single Triton pass.

    Columns ``[0, n_swa)`` come from the SWA cache (token ids ``swa_page_table``)
    and columns ``[n_swa, topk)`` from the compressed cache (token ids
    ``extra_page_table``). The two pools may use different ``page_size`` /
    ``bytes_per_page``, so both are carried as kernel constants.

    ``-1`` padding token ids are clamped to 0 (never OOB-read); rows beyond the
    per-query ``*_topk_lengths`` are marked invalid in ``valid_out`` when
    ``WRITE_VALID`` is requested and are never attended to.

    Args:
        combined_kv: (batch, topk, 512) bf16 destination (in-place filled).
        swa_quant_k_cache: SWA pool (num_pages, bytes_per_page_padded) uint8.
        swa_page_table: (batch, n_swa) int32 token ids.
        swa_page_size: tokens per SWA page.
        extra_quant_k_cache: compressed pool or None (SWA-only).
        extra_page_table: (batch, n_extra) int32 token ids or None.
        extra_page_size: tokens per compressed page or None.
        swa_topk_lengths: (batch,) int32; optional, only used for the fused mask.
        extra_topk_lengths: (batch,) int32; optional, only used for the fused mask.
        valid_out: (batch, topk) bool; optional, filled with the fused valid mask.

    Returns:
        combined_kv (the filled buffer).
    """
    assert combined_kv.is_cuda
    assert combined_kv.dtype == torch.bfloat16
    b, topk, dim = combined_kv.shape
    assert dim == DIM_NOPE + DIM_ROPE
    assert combined_kv.is_contiguous()

    n_swa = swa_page_table.shape[-1]
    assert topk >= n_swa

    has_extra = extra_quant_k_cache is not None
    n_extra = 0
    if has_extra:
        assert extra_page_table is not None and extra_page_size is not None
        n_extra = extra_page_table.shape[-1]
        assert topk == n_swa + n_extra

    swa_u8 = swa_quant_k_cache.view(torch.uint8)
    swa_bytes_per_page = swa_u8.shape[-1]
    swa_bf16 = swa_u8.view(torch.bfloat16).reshape(-1)
    swa_u8_flat = swa_u8.reshape(-1)
    swa_s_offset = swa_page_size * NOPE_ROPE_BYTES

    if has_extra:
        extra_u8 = extra_quant_k_cache.view(torch.uint8)
        extra_bytes_per_page = extra_u8.shape[-1]
        extra_bf16 = extra_u8.view(torch.bfloat16).reshape(-1)
        extra_u8_flat = extra_u8.reshape(-1)
        extra_s_offset = extra_page_size * NOPE_ROPE_BYTES
    else:
        # unused sentinels
        extra_u8_flat = swa_u8_flat
        extra_bf16 = swa_bf16
        extra_bytes_per_page = swa_bytes_per_page
        extra_s_offset = swa_s_offset

    write_valid = valid_out is not None
    if write_valid:
        assert swa_topk_lengths is not None
        assert valid_out.shape == (b, topk)
        if has_extra:
            assert extra_topk_lengths is not None

    grid = (b * topk,)
    _dequantize_combined_kv_paged_kernel[grid](
        combined_kv,
        swa_bf16,
        swa_u8_flat,
        swa_page_table,
        extra_bf16,
        extra_u8_flat,
        extra_page_table if has_extra else swa_page_table,
        swa_topk_lengths if write_valid else swa_page_table,
        (extra_topk_lengths if write_valid and has_extra else swa_topk_lengths)
        if write_valid
        else swa_page_table,
        valid_out if write_valid else combined_kv,
        WRITE_VALID=write_valid,
        SWA_PAGE_SIZE=swa_page_size,
        SWA_BYTES_PER_PAGE=swa_bytes_per_page,
        EXTRA_PAGE_SIZE=extra_page_size if has_extra else swa_page_size,
        EXTRA_BYTES_PER_PAGE=extra_bytes_per_page,
        N_SWA=n_swa,
        TOPK=topk,
        N_EXTRA=n_extra,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        SWA_S_OFFSET_BYTES=swa_s_offset,
        EXTRA_S_OFFSET_BYTES=extra_s_offset,
    )
    return combined_kv


def gather_dequant_requant_fp8_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    extra_rows: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather DeepSeek-V4 paged KV cache into a flat FP8 workspace.

    This is the Q8KV8 sparse-prefill adapter for the DeepSeek-V4 packed layout.
    It gathers token IDs from the existing paged cache, dequantizes the 448-dim
    nope region with its UE8M0 per-64 scales, casts the 64-dim BF16 rope tail to
    FP8, and writes the result as ``(num_tokens + extra_rows, 1, 512)`` FP8.

    ``extra_rows`` appends zero rows for kernels that map masked sparse indices
    to a valid zero landing pad.
    """
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    assert extra_rows >= 0

    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    total_rows = num_tokens + extra_rows
    bytes_per_page = quant_k_cache_u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    buf_fp8 = quant_k_cache_u8.view(fp8_dtype).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    if out is None:
        out = torch.zeros(
            (total_rows, 1, DIM_NOPE + DIM_ROPE),
            dtype=fp8_dtype,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (total_rows, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == fp8_dtype
        if extra_rows:
            out[num_tokens:].zero_()

    if num_tokens == 0:
        return out

    _gather_dequant_requant_fp8_paged_kernel[(num_tokens,)](
        out,
        buf_fp8,
        buf_bf16,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=s_offset_bytes,
    )
    return out


def q8kv8_padded_num_heads(num_heads: int) -> int:
    """Return a Q-head count supported by the SM90 Q8KV8 kernel."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if num_heads <= 64:
        return 64
    if num_heads <= 128:
        return 128
    raise ValueError(
        "DeepSeek-V4 Q8KV8 sparse prefill supports at most 128 local "
        f"query heads, got {num_heads}"
    )


def cast_q_fp8_for_q8kv8_prefill(
    q: torch.Tensor,
    padded_num_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast DeepSeek-V4 sparse-prefill Q to the Q8KV8 kernel format.

    The incoming Q is the model-produced BF16/FP16 tensor already shaped as
    ``(num_tokens, num_heads, 512)`` after removing the singleton MQA axis.

    The SM90 kernel processes query heads in 64-head blocks. Tensor parallelism
    commonly leaves fewer than 64 local heads, so the active heads are copied
    into a zero-padded 64/128-head FP8 tensor.
    """
    assert q.ndim == 3
    assert q.shape[-1] == DIM_NOPE + DIM_ROPE

    num_tokens, num_heads, head_dim = q.shape
    if padded_num_heads is None:
        padded_num_heads = q8kv8_padded_num_heads(num_heads)

    if padded_num_heads not in (64, 128) or padded_num_heads < num_heads:
        raise ValueError(
            f"invalid padded_num_heads={padded_num_heads} for num_heads={num_heads}"
        )

    expected_shape = (num_tokens, padded_num_heads, head_dim)

    if out is None:
        q_fp8 = torch.zeros(
            expected_shape,
            dtype=fp8_dtype,
            device=q.device,
        )
    else:
        if (
            out.shape != expected_shape
            or out.dtype != fp8_dtype
            or out.device != q.device
        ):
            raise ValueError(
                "Q8KV8 Q output must have shape/dtype/device "
                f"{expected_shape}/{fp8_dtype}/{q.device}, got "
                f"{tuple(out.shape)}/{out.dtype}/{out.device}"
            )
        q_fp8 = out
        if padded_num_heads > num_heads:
            q_fp8[:, num_heads:].zero_()

    q_fp8[:, :num_heads].copy_(q)
    q_scale = torch.ones((), dtype=torch.float32, device=q.device)
    return q_fp8, q_scale


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # One program per token: load page_table[token_id] once and emit all
    # NUM_SCALE_TILES nope tiles + rope tail via tl.static_range.
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        # Load e4m3 bytes as uint8 and decode in-register. This works on every
        # arch (including Ampere, which cannot materialize Triton's fp8e4nv
        # type) and is numerically lossless: e4m3 widens exactly into fp32.
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = _e4m3_uint8_to_f32(tl.load(buf_uint8_ptr + fp8_off))

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)


@triton.jit
def _gather_dequant_requant_fp8_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(
        output_ptr + out_row_base + DIM_NOPE + rope_offs,
        rope_data.to(output_ptr.dtype.element_ty),
    )


def dequantize_k_cache_paged_ref(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Pure-torch reference for :func:`dequantize_k_cache_paged`.

    Decodes the same v4 paged layout with vectorized torch indexing instead of
    a Triton kernel. Used to validate the kernel (see the ``__main__`` block
    below); not on any hot path.
    """
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    flat_u8 = u8.reshape(-1)
    flat_fp8 = u8.view(fp8_dtype).reshape(-1)
    flat_bf16 = u8.view(torch.bfloat16).reshape(-1)

    loc = page_table_1_flattened.to(torch.int64)
    page_idx = loc // page_size
    in_page = loc % page_size
    page_byte_base = page_idx * bytes_per_page
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + s_offset_bytes + in_page * PADDED_SCALE_PER_TOKEN
    )

    device = quant_k_cache.device
    nope_byte = (
        token_data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
    )
    nope_fp8 = flat_fp8[nope_byte].to(torch.float32)
    scale_byte = (
        token_scale_base[:, None]
        + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
    )
    scale_u8 = flat_u8[scale_byte].to(torch.int32)
    scale_pow2 = torch.exp2((scale_u8 - 127).to(torch.float32))
    scale_pow2 = torch.where(
        scale_pow2 < (2.0**-126), torch.zeros_like(scale_pow2), scale_pow2
    )
    scale_full = scale_pow2.repeat_interleave(TILE_SIZE, dim=1)
    nope = nope_fp8 * scale_full

    rope_bf16_base = (token_data_base + DIM_NOPE) // 2
    rope_idx = rope_bf16_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
    rope = flat_bf16[rope_idx]

    out = torch.empty(
        (loc.shape[0], 1, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=device,
    )
    out[:, 0, :DIM_NOPE] = nope.to(torch.bfloat16)
    out[:, 0, DIM_NOPE:] = rope
    return out


def gather_dequant_requant_fp8_paged_ref(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    extra_rows: int = 0,
) -> torch.Tensor:
    """Torch reference for :func:`gather_dequant_requant_fp8_paged`."""
    active = dequantize_k_cache_paged_ref(
        quant_k_cache,
        page_table_1_flattened,
        page_size,
    ).to(fp8_dtype)
    if extra_rows == 0:
        return active
    out = torch.zeros(
        (active.shape[0] + extra_rows, 1, DIM_NOPE + DIM_ROPE),
        dtype=fp8_dtype,
        device=active.device,
    )
    out[: active.shape[0]] = active
    return out


if __name__ == "__main__":
    assert torch.cuda.is_available(), "this self-test needs a CUDA device"
    torch.manual_seed(0)
    device = "cuda"

    page_size = 64
    num_pages = 8
    num_tokens = 333
    raw_bytes = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (
        (raw_bytes + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    ) * NOPE_ROPE_BYTES

    quant_k_cache = torch.randint(
        0, 256, (num_pages, bytes_per_page), dtype=torch.uint8, device=device
    )
    page_table = torch.randint(
        0, num_pages * page_size, (num_tokens,), dtype=torch.int32, device=device
    )

    out_kernel = dequantize_k_cache_paged(quant_k_cache, page_table, page_size)
    out_ref = dequantize_k_cache_paged_ref(quant_k_cache, page_table, page_size)

    torch.testing.assert_close(out_kernel, out_ref, atol=0, rtol=0, equal_nan=True)
    print(
        f"OK: kernel matches torch ref for {num_tokens} tokens "
        f"(page_size={page_size}, bytes_per_page={bytes_per_page})"
    )
