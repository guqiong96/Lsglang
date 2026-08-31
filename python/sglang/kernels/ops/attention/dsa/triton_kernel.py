from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _f32_to_e4m3_uint8(x):
    """Encode f32 -> ``e4m3fn`` (1-4-3, exp bias 7, max 448, no inf) raw uint8
    bits, for Ampere (SM80/SM86/SM89) where Triton cannot materialize the
    ``fp8e4nv`` type in a kernel. Round-to-nearest-even (RNE), saturates
    |x| > 448, NaN/Inf -> +/-448.

    NOTE: the upstream vLLM reference (``fp8_utils._f32_to_e4m3_uint8``) derives
    the exponent via ``tl.log2``, which on our Triton rounds ``log2(v)`` up to
    the next integer for values just below a power of two (e.g. ``log2(127.99)``
    -> 7.0), wrongly routing them through the subnormal path and encoding to a
    tiny value (127.99 -> ~0.0156 instead of ~128). That is a real bug for
    activation quantization (activations routinely reach ~128). This version
    derives the exponent exactly from the fp32 bit pattern (no ``tl.log2``), so
    it is correct at every power-of-two boundary.
    """
    x = x.to(tl.float32)
    sign = tl.where(x < 0, 1, 0).to(tl.int32)
    a = tl.abs(x)
    a = tl.where(a != a, 0.0, a)  # NaN -> 0 magnitude
    a = tl.minimum(a, 448.0)
    is_zero = a == 0.0
    a_safe = tl.where(is_zero, 1.0, a)
    bits = a_safe.to(tl.int32, bitcast=True)
    exp8 = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF

    # ---- fp8 normal output (a >= 2^-6): exact via fp32 bit manipulation ----
    # value = 2^(exp8-127) * (1 + man/2^23).  e4m3 keeps 4 mantissa bits
    # (1 implicit + 3), so RNE-round the 24-bit mantissa to a multiple of 2^20.
    man24 = (1 << 23) | man  # in [2^23, 2^24)
    low20 = man24 & 0xFFFFF
    r = man24 + (1 << 19)  # add half ulp (2^19)
    is_tie = low20 == (1 << 19)
    bit20 = (man24 >> 20) & 1  # LSB of the kept 4-bit mantissa
    r = tl.where(is_tie & (bit20 == 0), r - 1, r)  # ties-to-even
    e4 = exp8 - 120  # e4m3 biased exponent
    m4 = (r >> 20) - 8  # strip implicit leading 1
    carry = r >= (1 << 24)
    m4 = tl.where(carry, 0, m4)
    e4 = e4 + carry
    e4 = tl.minimum(e4, 15)  # overflow -> max (a<=448 so e4<=15 normally)
    byte_n = (sign << 7) | (e4 << 3) | m4

    # ---- fp8 subnormal output (a < 2^-6): value = m * 2^-9, m in [0,7] ----
    # m = RNE(a * 512).  a*512 in [0, 8); RNE via round-half-to-even.
    t = a * 512.0
    ti = tl.floor(t)
    frac = t - ti
    rne = ti + tl.where(
        frac > 0.5,
        1.0,
        tl.where(frac == 0.5, tl.where((ti.to(tl.int32) & 1) == 1, 1.0, 0.0), 0.0),
    )
    m_s = rne.to(tl.int32)  # in [0, 8]
    is_promote = m_s >= 8  # rounds up to min normal 2^-6
    m_s2 = tl.where(is_promote, 0, m_s)
    byte_s = tl.where(
        is_promote, (sign << 7) | (1 << 3), (sign << 7) | m_s2
    )

    byte = tl.where(a >= 0.015625, byte_n, byte_s)  # 2^-6 boundary
    byte = tl.where(is_zero, (sign << 7), byte)  # +0 -> 0x00, -0 -> 0x80
    return byte.to(tl.uint8)


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_u8_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_u8_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, _f32_to_e4m3_uint8(y), mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors (fp8 produced as raw uint8 bytes for sm8x, where
    # Triton cannot represent the fp8e4nv type; returned as an fp8 view below).
    y = torch.empty_like(x, dtype=torch.uint8)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y.view(torch.float8_e4m3fn), s


@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  # [bs, topk]
    kv_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,  # [bs * topk] output buffer
    bs: tl.constexpr,
    topk: tl.constexpr,
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    batch_id = tl.program_id(0)

    # Get the start position for this batch in kv_indices
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    # Load all topk indices for this batch
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    # Count valid indices and compact them
    mask = indices != -1

    # Use prefix sum to compute destination positions for valid elements
    # For each position, count how many valid elements are before it
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    # Store valid indices to their compacted positions
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )
