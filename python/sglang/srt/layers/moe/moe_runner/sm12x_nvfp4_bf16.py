# SPDX-License-Identifier: Apache-2.0
"""SM12x (Blackwell GeForce / RTX PRO) NVFP4 W4A16 fused MoE runner.

Wraps flashinfer's CuTe DSL SM12x kernels (``flashinfer.fused_moe.cute_dsl.
blackwell_sm12x.moe_nvfp4_w4a16`` and its ``moe_nvfp4_w4a4`` sibling): BF16
activations are fed straight into fused NVFP4-weight-only grouped GEMMs (no
activation quantization), which is the numerics a W4A16 checkpoint is
calibrated for. Weight payloads are the canonical ModelOpt NVFP4 buffers --
packed E2M1 nibbles row-major ``[E, N13, K/2]``, 128x4-swizzled E4M3 block
scales and one FP32 global scale per expert -- so the same tensors serve both
arms. Ported from lovedheart/sglang ``qsa-on-main`` (4cea8de1c6, f7e2810fd7,
5a4be0e194); the difference here is that the kernel is imported from the
installed flashinfer instead of a vendored copy, with the two host-glue
choices the fork measured on SM120 re-applied on import (see
``_apply_sm120_tuning``).

Token counts are rounded up to a bounded bucket ladder (bounded kernel-compile
set, since a CuTe DSL compile is keyed on the token count and costs seconds);
batches above the prefill chunk size (--chunked-prefill-size, also the
ladder's horizon) launch as repeated chunk-sized blocks, which bounds the
workspace base as well as the compile set.

With ``SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS`` set, batches at or above that
token count instead run flashinfer's ``moe_nvfp4_w4a4`` sibling kernel on the
same weights (NVFP4-quantized activations, faster for prefill).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

# flashinfer's SM12x CuTe-DSL MoE modules (no TMA: every operand is reached by
# plain pointer arithmetic, so a 256B-aligned view into a larger buffer is legal).
_W4A16_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a16"
_W4A4_MODULE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4"


@dataclass
class Sm12xNvfp4W4A16MoeQuantInfo(MoeQuantInfo):
    """Weight payload for the SM12x W4A16 fused MoE kernel.

    w13 rows are [gate; up] (gate first), FP4-packed along K. ``*_gs`` are the
    per-expert FP4 block-scale global factors (FP32 [num_experts]): raw weight
    ``weight_scale_2`` values, *not* folded with an activation scale
    (activations stay BF16, so any static input scale would cancel). When the
    checkpoint gives gate and up separate global scales, their ratio rides on
    ``w2_gs`` -- see ``_sm12x_global_scales``.
    """

    w13: torch.Tensor  # uint8 [E, 2I, H/2], [gate; up] rows
    w13_sf: torch.Tensor  # e4m3/uint8 [E, pad128(2I), pad4(H/16)], 128x4 swizzled
    w13_gs: torch.Tensor  # fp32 [E]
    w2: torch.Tensor  # uint8 [E, H, I/2]
    w2_sf: torch.Tensor  # e4m3/uint8 [E, pad128(H), pad4(I/16)]
    w2_gs: torch.Tensor  # fp32 [E]
    # Optional W4A4 prefill arm (flashinfer moe_nvfp4_w4a4, sibling of the W4A16
    # kernel reading the same canonical weights). When set, batches with at
    # least SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS tokens quantize activations to
    # NVFP4 using these quantization multipliers (1 / input_scale); see
    # _prepare_sm12x_w4a16_weights for their derivation.
    a13_gs: Optional[torch.Tensor] = None
    a2_gs: Optional[torch.Tensor] = None
    # MoE models that scale the routed sum outside topk (routed_scaling_factor
    # on the runner config) expect a backend to multiply it into the output;
    # the quant method clears this flag when it already folds the factor into
    # topk_weights. Same convention as FlashInferCutlassMoeQuantInfo.
    apply_routed_scaling_factor: bool = True


# --------------------------------------------------------------------------------
# kernel source
# --------------------------------------------------------------------------------
# Scratch buffers are keyed by (bucket tokens, hidden, inter, N13, experts,
# top_k, device, w4a4); every layer shares the same geometry, so one entry
# per bucket serves the whole model. A workspace must not be shared by two
# *concurrent* launches, which per-layer sequential execution guarantees.
# The W4A4 arm carves its buckets DISJOINTLY out of one arena; the W4A16 arm
# ALIASES all buckets onto one base (see _w4a16_workspace for the contract
# that makes that sound) -- per-bucket residency used to cost Σ ≈ 2× the
# max bucket at 112 KB/tok, which is what the q36 OOM chain paid.
_WORKSPACE_CACHE: dict[tuple, tuple[list, bool]] = {}
# One flat scratch arena for all W4A4 buckets (workspace bytes scale linearly
# with the token count, so buckets carved out of one max-size buffer replace
# the per-bucket ladder; views are 256B-aligned so TMA descriptors stay legal,
# and the arena is allocated once and never resized, so captured graphs keep
# valid addresses).
_ARENA_ALIGN = 256
_ARENA_SLACK = 1 << 20
_W4A4_ARENAS: dict[torch.device, list] = {}  # device -> [uint8 buffer, next offset]
_W4A16_BASE: dict[torch.device, list] = {}   # device -> [uint8 base, component offsets]

_TUNE_NOTE = False


def _ws_bytes(shapes) -> int:
    return sum(
        (math.prod(s) * d.itemsize + _ARENA_ALIGN - 1) // _ARENA_ALIGN * _ARENA_ALIGN
        for s, d in shapes
    )


def _pick_bm_ladder(S: int, E: int) -> int:
    """Rows-per-CTA ladder measured on the Qwen3.8-Flash-Next geometry
    (E=512, H=2560, top-10) over token buckets 1..8192, cold-L2 event timing
    cross-checked with paired CUDA-graph ABBA on the runner path. Identical to
    flashinfer's next-pow2(mean slots/expert) rule except in the 129..896-token
    band, where BM=8 beats the old BM=16 choice by ~10% at T=512 (a taller
    M-tile mostly buys padding, not weight reuse). Still capped by
    next_pow2(mean slots/expert) so small-E shapes never sit on a tile taller
    than any expert actually fills.
    """
    if S <= 320:
        bm = 1
    elif S <= 640:
        bm = 2
    elif S <= 1280:
        bm = 4
    elif S <= 8192:
        bm = 8
    else:
        bm = 32
    r = S / max(1, min(E, S))
    cap = 1
    while cap < 32 and cap < r:
        cap *= 2
    if r > 32:
        # few experts, many rows each: weight reuse per CTA dominates and only
        # the widest tile keeps it (the analytic rule's choice; never tune down).
        bm = 32
    return min(bm, cap)


def _apply_sm120_tuning(kernel_module) -> None:
    """Re-apply the two SM120 host-glue measurements the installed flashinfer
    defaults miss. Both are private host-side planning hooks: touching them
    cannot change numerics, only which launch plan a shape gets, and a flashinfer
    that no longer has them simply keeps its own defaults (logged once).
    """
    global _TUNE_NOTE
    notes = []
    if envs.SGLANG_SM12X_W4A16_BM_LADDER.get() and hasattr(kernel_module, "_pick_bm"):
        kernel_module._pick_bm = _pick_bm_ladder
        notes.append("rows-per-CTA ladder")
    if not envs.SGLANG_SM12X_W4A16_PERSISTENT.get():
        # The fused single-launch (persistent) kernel measures slower than the
        # separate route/gemm/act/final launches at every token bucket swept on
        # SM120 (e.g. 19.7 vs 13.0 us at T=1 under graph replay), but stock
        # flashinfer takes it whenever BM == 1. Returning None from
        # _persistent_plan makes _entry fall through to the tiled plan.
        persistent_plan = getattr(kernel_module, "_persistent_plan", None)
        if persistent_plan is not None:

            def _tiled_only(*args, **kwargs):
                return None

            kernel_module._persistent_plan = _tiled_only
            notes.append("tiled plan (persistent off)")
    if notes and not _TUNE_NOTE:
        _TUNE_NOTE = True
        logger.info("sm12x_nvfp4_bf16: applied SM120 tuning (%s)", ", ".join(notes))


def w4a16_kernels():
    """flashinfer's W4A16 module with the SM120 launch-plan tuning applied."""
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
            moe_nvfp4_w4a16 as _m,
        )
    except ImportError as exc:  # pragma: no cover - depends on the wheel
        raise ImportError(
            "--moe-runner-backend sm12x_nvfp4_bf16 needs flashinfer's SM12x "
            f"CuTe DSL MoE kernels ({_W4A16_MODULE}); installed flashinfer does "
            f"not provide them ({exc})"
        ) from exc
    _apply_sm120_tuning(_m)
    return _m


# --------------------------------------------------------------------------------
# workspaces
# --------------------------------------------------------------------------------
def _ws_upper_tokens(bucket: int) -> int:
    """Token count the workspace bases are sized for: the server's prefill
    chunk size, which is also the launch block size (see
    ``_max_launch_tokens``); the max() keeps an oversized caller from
    silently aliasing onto an undersized base."""
    return max(_max_launch_tokens(), 1 << (bucket - 1).bit_length())


def _w4a4_workspace(
    bucket: int,
    h: int,
    i: int,
    n13: int,
    e: int,
    topk: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Per-bucket W4A4 workspace views carved from the shared arena."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 import (
        allocate_workspace,
    )

    shapes = None
    try:
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 import (
            _entry,
        )

        _, shapes, _ = _entry(bucket, h, i, n13, e, topk, torch.device(device))
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        logger.debug("W4A4 arena disabled (%s); per-bucket workspace", exc)
    if shapes is not None:
        need = _ws_bytes(shapes)
        arena = _W4A4_ARENAS.get(device)
        if arena is None:
            upper = _ws_upper_tokens(bucket)
            if upper is not None and upper >= bucket:
                cap = _ws_bytes(
                    _entry(upper, h, i, n13, e, topk, torch.device(device))[1]
                )
                if cap >= need:
                    arena = [
                        torch.empty(
                            cap + _ARENA_SLACK, dtype=torch.uint8, device=device
                        ),
                        0,
                    ]
                    _W4A4_ARENAS[device] = arena
        if arena is not None and arena[0].numel() - arena[1] >= need:
            off = arena[1]
            ws = []
            for shape, dtype in shapes:
                n = math.prod(shape)
                nbytes = (
                    n * dtype.itemsize + _ARENA_ALIGN - 1
                ) // _ARENA_ALIGN * _ARENA_ALIGN
                view = arena[0][off : off + n * dtype.itemsize].view(dtype)
                ws.append(view.view(tuple(shape)))
                off += nbytes
            arena[1] = off
            return ws
    return allocate_workspace(
        num_tokens=bucket,
        hidden_size=h,
        intermediate_size=i,
        num_experts=e,
        top_k=topk,
        is_gated=n13 == 2 * i,
        device=device,
    )


def _w4a16_workspace(bucket, h, i, n13, e, topk, device):
    """Per-bucket W4A16 workspace as VIEWS onto one aliasing base buffer,
    plus the pre-launch int32-ledger zero flag.

    Aliasing across buckets is sound because (i) the three slice sizes are
    strictly linear in the token count (probed through _entry: fp32 92 KB/tok
    dominates, bf16 20 KB/tok, int32 traces), so every bucket's component is
    an exact prefix of the base's -- components sit at the UPPER-sized
    component offsets, keeping each component's start bucket-invariant;
    (ii) PERSISTENT plans restore the counters and the FP32 combine slice
    they touched to zero on exit, the very contract per-bucket reuse rests on
    today; (iii) launches never run concurrently. TILED plans do NOT restore
    (tier-side probe: counters 99% dirty after one launch), so their int32
    ledger is zeroed before launch: same-bucket reuse gets away with the
    dirt because an identical T re-draws identical values, but under aliasing
    a bigger bucket's permutation offsets must not leak into a smaller
    bucket's tables. FP32/BF16 need no reset -- results are bit-stable under
    today's same-bucket dirty reuse, so those slices are rewritten from the
    inputs, not accumulated.
    """
    m = w4a16_kernels()  # the TUNED module: _pick_bm here decides persistent
    dt = (torch.float32, torch.int32, torch.bfloat16)

    def _per_bucket():
        return m.allocate_workspace(
            num_tokens=bucket, hidden_size=h, intermediate_size=i,
            num_experts=e, top_k=topk, is_gated=n13 == 2 * i,
            device=device)

    try:
        persistent, _, sizes, _ = m._entry(
            bucket, h, i, n13, e, topk, torch.device(device))
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        logger.debug("W4A16 aliasing disabled (%s); per-bucket workspace", exc)
        return _per_bucket(), False
    ent = _W4A16_BASE.get(device)
    if ent is None:
        upper = _ws_upper_tokens(bucket)
        usizes = None
        if upper is not None and upper >= bucket:
            try:
                usizes = m._entry(
                    upper, h, i, n13, e, topk, torch.device(device))[2]
            except (ImportError, AttributeError, TypeError, RuntimeError):
                usizes = None
        if usizes is None:
            return _per_bucket(), False
        offs, o = [], 0
        for s, t in zip(usizes, dt):
            offs.append(o)
            o += (s * t.itemsize + _ARENA_ALIGN - 1) // _ARENA_ALIGN * _ARENA_ALIGN
        # Zero-initialized ONCE: the launch-start invariant the contract
        # restores to; views keep the base alive as long as they are cached.
        ent = [torch.zeros(o, dtype=torch.uint8, device=device), offs]
        _W4A16_BASE[device] = ent
    base, offs = ent
    limits = offs[1:] + [base.numel()]
    for s, t, lo, hi in zip(sizes, dt, offs, limits):
        if s * t.itemsize > hi - lo:  # non-monotonic component: do not alias
            return _per_bucket(), False
    ws = [base[lo:lo + s * t.itemsize].view(t)
          for lo, s, t in zip(offs, sizes, dt)]
    return ws, (not persistent)


def _workspace(
    bucket: int,
    h: int,
    i: int,
    n13: int,
    e: int,
    topk: int,
    device: torch.device,
    w4a4: bool = False,
) -> tuple[list[torch.Tensor], bool]:
    """Cached (workspace, pre-launch int32 zero flag)."""
    key = (bucket, h, i, n13, e, topk, device, w4a4)
    hit = _WORKSPACE_CACHE.get(key)
    if hit is None:
        if w4a4:
            hit = (_w4a4_workspace(bucket, h, i, n13, e, topk, device), False)
        else:
            hit = _w4a16_workspace(bucket, h, i, n13, e, topk, device)
        _WORKSPACE_CACHE[key] = hit
    return hit


# --------------------------------------------------------------------------------
# token bucket ladder (bounds the per-token-count CuTe DSL compile set)
# --------------------------------------------------------------------------------
def _hybrid_buckets(max_num_tokens: int) -> tuple[int, ...]:
    """Bucket ladder mirroring flashinfer's hybrid policy: powers of 2 up to
    256, steps of 256 to 2048, steps of 512 to 4096, powers of 2 beyond."""
    buckets = []
    m = 1
    while m <= min(max_num_tokens, 256):
        buckets.append(m)
        m *= 2
    for start, end, step in ((256, 2048, 256), (2048, 4096, 512)):
        m = start + step
        while m <= min(max_num_tokens, end):
            buckets.append(m)
            m += step
    m = 8192
    while m <= max_num_tokens:
        buckets.append(m)
        m *= 2
    if not buckets or buckets[-1] != max_num_tokens:
        buckets.append(max_num_tokens)
    return tuple(sorted(set(buckets)))


# Fallback block size for callers without server args (standalone probes);
# same order as sglang's own chunked-prefill-size default, never a policy.
_NO_SERVER_TOKENS = 8192
_launch_tokens_cache: Optional[int] = None


def _max_launch_tokens() -> int:
    """Resident launch block size (tokens): the server's prefill chunk size
    (--chunked-prefill-size), the stack's one declared prefill-batch budget.

    The workspace scales linearly (~112 KB/tok probed), and the scheduler
    may still hand this arm a larger batch than the chunk size -- a
    super-chunk (LK_GPU_PREFILL_SUB_M) raises the MoE forward to
    max(chunked_prefill_size, super_chunk) while the chunk knob itself stays
    put. Batches above the chunk size are therefore block-split at it. The
    split is pure launch-side (zero transfers, no pool/ring contact), so the
    tier's per-chunk amortization -- the whole point of the super-chunk --
    is untouched, while this arm's workspace base and compile set scale
    with a visible server argument instead of a hidden constant.
    """
    global _launch_tokens_cache
    if _launch_tokens_cache is None:
        try:
            from sglang.srt.server_args import get_server_args

            c = int(get_server_args().chunked_prefill_size or 0)
        except (ImportError, RuntimeError, TypeError, ValueError):
            c = 0
        _launch_tokens_cache = c if c > 0 else _NO_SERVER_TOKENS
    return _launch_tokens_cache


def map_to_bucket(num_tokens: int, max_num_tokens: Optional[int] = None) -> int:
    """Token count -> the bucket (== the compiled shape) to launch at.

    Total over the ladder: callers block-split at ``_max_launch_tokens()``,
    so ``num_tokens`` never exceeds the horizon and the loop always answers;
    the power-of-two line is a defensive floor for a caller that forgot."""
    if max_num_tokens is None:
        max_num_tokens = _max_launch_tokens()
    for b in _hybrid_buckets(max_num_tokens):
        if num_tokens <= b:
            return b
    return 1 << (num_tokens - 1).bit_length()


def _pad_rows(x: torch.Tensor, rows: int, dtype: torch.dtype, pad_value: float):
    """Cast to ``dtype`` and pad to ``rows`` total rows along dim 0."""
    if x.shape[0] == rows:
        return x if x.dtype == dtype else x.to(dtype)
    out = torch.full((rows, *x.shape[1:]), pad_value, dtype=dtype, device=x.device)
    out[: x.shape[0]] = x.to(dtype)
    return out


# --------------------------------------------------------------------------------
# fused entry point
# --------------------------------------------------------------------------------
@register_fused_func("none", "sm12x_nvfp4_bf16")
def fused_experts_none_to_sm12x_nvfp4_bf16(
    dispatch_output: "StandardDispatchOutput",
    quant_info: Sm12xNvfp4W4A16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> "StandardCombineInput":
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(quant_info, Sm12xNvfp4W4A16MoeQuantInfo), (
        f"Unexpected quant_info type for sm12x_nvfp4_bf16: {type(quant_info)}"
    )
    cfg = runner_config
    if cfg.num_fused_shared_experts:
        raise NotImplementedError(
            "sm12x_nvfp4_bf16 expects shared experts to run outside the kernel."
        )
    if cfg.apply_router_weight_on_input:
        raise NotImplementedError(
            "apply_router_weight_on_input is not supported by sm12x_nvfp4_bf16."
        )

    hidden = dispatch_output.hidden_states
    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "sm12x_nvfp4_bf16 expects unquantized BF16 activations."
        )
    if hidden.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"sm12x_nvfp4_bf16 requires BF16 activations, got {hidden.dtype}."
        )

    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids

    t, h = hidden.shape
    e, n13, _ = quant_info.w13.shape
    i = quant_info.w2.shape[2] * 2
    topk = topk_ids.shape[1]
    w4a4_min = envs.SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS.get()
    # Large batches (typically prefill chunks) go through the W4A4 sibling
    # kernel, ~1.4x faster than W4A16 here, at the cost of quantizing
    # activations to NVFP4.
    use_w4a4 = w4a4_min > 0 and quant_info.a13_gs is not None and t >= w4a4_min

    # The kernel multiplies topk_weights into the routed sum; a model that
    # scales the sum outside topk rides that factor along on the weights
    # instead of in a second pass over the (BF16) output.
    factor = getattr(cfg, "routed_scaling_factor", None)
    if quant_info.apply_routed_scaling_factor and factor is not None and factor != 1.0:
        topk_weights = topk_weights * factor

    def _launch(x_b, ids_b, w_b, dst, bucket):
        """One ladder launch of ``bucket`` rows into ``dst``."""
        if use_w4a4:
            from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_nvfp4_w4a4 import (  # noqa: E501
                run_moe_w4a4,
            )

            workspace, _reset = _workspace(bucket, h, i, n13, e, topk,
                                           hidden.device, w4a4=True)
            run_moe_w4a4(
                x_b,
                ids_b,
                w_b,
                quant_info.w13,
                quant_info.w13_sf,
                quant_info.w13_gs,
                quant_info.w2,
                quant_info.w2_sf,
                quant_info.w2_gs,
                quant_info.a13_gs,
                quant_info.a2_gs,
                dst,
                workspace=workspace,
            )
        else:
            workspace, reset = _workspace(bucket, h, i, n13, e, topk,
                                          hidden.device)
            if reset:
                # TILED plan: the int32 ledger is not restored on exit, and
                # under aliasing a bigger bucket's offsets must not seed this
                # launch's work list.  One <=MB memset.
                workspace[1].zero_()
            w4a16_kernels().run_moe_w4a16(
                x_b,
                ids_b,
                w_b,
                quant_info.w13,
                quant_info.w13_sf,
                quant_info.w13_gs,
                quant_info.w2,
                quant_info.w2_sf,
                quant_info.w2_gs,
                dst,
                workspace=workspace,
            )

    # Always launch at a ladder bucket: an exact-count launch would JIT one
    # CuTe DSL specialization per prefill batch -- prefill counts never
    # repeat, so every batch would pay a cute.compile. Batches above the
    # chunk size are SPLIT at it rather than rounded up: the split is pure
    # launch-side (zero transfers, no pool/ring contact), so the scheduler's
    # super-chunk amortization of the staging tier's work is untouched while
    # this arm's workspace base and compile set stay ladder-bounded.
    cap = _max_launch_tokens()
    if t > cap:
        out = torch.empty((t, h), dtype=torch.bfloat16, device=hidden.device)
        pos = 0
        while pos < t:
            blk = min(cap, t - pos)
            sl = slice(pos, pos + blk)
            x_b = hidden[sl]
            ids_b = topk_ids[sl].to(torch.int32)
            w_b = topk_weights[sl].to(torch.float32)
            bucket = map_to_bucket(blk, cap)
            if bucket == blk:
                # full blocks land exactly on the cap: no padding at all
                _launch(x_b, ids_b, w_b, out[sl], bucket)
            else:
                # ragged tail: pad into a ladder-sized scratch, copy back
                dst = torch.empty((bucket, h), dtype=torch.bfloat16,
                                  device=hidden.device)
                _launch(_pad_rows(x_b, bucket, torch.bfloat16, 0.0),
                        _pad_rows(ids_b, bucket, torch.int32, float(e)),
                        _pad_rows(w_b, bucket, torch.float32, 0.0),
                        dst, bucket)
                out[sl] = dst[:blk]
            pos += blk
        return StandardCombineInput(hidden_states=out)

    bucket = map_to_bucket(t, cap)
    if bucket != t:
        ids = _pad_rows(topk_ids, bucket, torch.int32, float(e))
        weights = _pad_rows(topk_weights, bucket, torch.float32, 0.0)
        x = _pad_rows(hidden, bucket, torch.bfloat16, 0.0)
    else:
        ids = topk_ids.to(torch.int32)
        weights = topk_weights.to(torch.float32)
        x = hidden

    # Out rows past ``t`` (padding slots) are never written: routing slots
    # with an out-of-range expert id are skipped by the kernel.
    out = torch.empty((bucket, h), dtype=torch.bfloat16, device=hidden.device)
    _launch(x, ids, weights, out, bucket)
    return StandardCombineInput(hidden_states=out[:t])
