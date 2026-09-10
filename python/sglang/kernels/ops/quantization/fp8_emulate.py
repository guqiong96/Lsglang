"""fp8 e4m3 bit helpers for GPUs whose Triton has no ``fp8e4nv`` type.

Triton before Ada (SM89) cannot even name ``tl.float8e4nv``: both a pointer to
it (``.to(tl.pointer_type(tl.float8e4nv))``) and a value cast (``x.to(tl.float8e4nv)``)
fail during ``to_ir`` with ``type fp8e4nv not supported in this architecture``.
These helpers reproduce the two uses those kernels need — decode a raw e4m3 byte,
and round a float to the nearest e4m3 value (round-half-to-even, the e4m3 cast) —
entirely in int/fp32 so the fp8 type is never materialized. SM89+ keeps its
native hardware path; this module is only reached when gated off by
:func:`fp8_native_supported`.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@lru_cache(maxsize=1)
def fp8_native_supported() -> bool:
    """Whether Triton can form ``fp8e4nv`` pointers / casts on this device
    (SM89 / Ada and newer)."""
    if not torch.cuda.is_available():
        return True
    return torch.cuda.get_device_capability() >= (8, 9)


@triton.jit
def e4m3fn_u8_to_f32(u):
    """Decode an ``e4m3fn`` byte (1-4-3, exp bias 7) to f32. NaN (S.1111.111)
    decodes to +/-480 (never stored by a quantizer that saturates at 448)."""
    ui = u.to(tl.int32)
    sign = (ui >> 7) & 1
    exp = (ui >> 3) & 0xF
    man = ui & 0x7
    mant = man.to(tl.float32) * 0.125
    val = tl.where(
        exp != 0,
        tl.exp2((exp - 7).to(tl.float32)) * (1.0 + mant),
        0.015625 * mant,
    )
    return tl.where(sign != 0, -val, val)


@triton.jit
def round_to_e4m3fn_f32(x):
    """Round ``x`` to the nearest ``e4m3fn`` value and return it as f32 — the
    value cast ``x.to(tl.float8e4nv).to(tl.float32)`` done without that type.

    Round-half-to-even via ``rint`` on the quantized mantissa. The exponent is
    read from the IEEE bits (exact at binade boundaries, unlike ``log2``) and
    clamped to e4m3's minimum normal exponent (-6) so the subnormal range uses a
    fixed 2**-9 grid. ``|x|`` is saturated to 448 first, matching the saturating
    fp8 cast (e4m3fn has no value in (448, 512)).
    """
    a = tl.minimum(tl.abs(x), 448.0)
    bits = a.to(tl.int32, bitcast=True)
    e32 = (bits >> 23) & 0xFF
    is_sub = bits < 0x00800000
    e = tl.where(is_sub, -126, e32 - 127)
    e_step = tl.maximum(e, -6)
    step = tl.exp2(e_step.to(tl.float32) - 3.0)
    q = libdevice.rint(a / step)
    mag = q * step
    return tl.where(x < 0, -mag, mag)
