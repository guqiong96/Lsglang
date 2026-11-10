import logging
import os
"""lk_moe configuration surface: env parsing, layer predicates and the
expert-pool pump park helper.

Everything LK_* lives here so the knobs have a single source of truth; callers
keep importing the names from ``sglang.srt.utils.common`` (compat re-export).
"""

import warnings

import torch

from contextlib import contextmanager
from dataclasses import dataclass

from sglang.srt.utils.env_vars import (
    get_bool_env_var,
    get_int_env_var,
    get_str_env_var,
)


@dataclass(frozen=True)
class LkMoeConfig:
    """Immutable snapshot of the LK_* knobs for one call site.

    Built on demand (no global cache): several call sites mutate the
    environment at runtime (tests, migrate_lk_env_vars, set_int_env_var), so a
    cached singleton would go stale.
    """

    enabled: bool
    numa_interleave: bool
    pool_slots: int
    pool_enabled: bool
    pool_decode_enabled: bool
    gpu_prefill_min_tokens: int
    gpu_prefill_engine: str
    prefill_debug: bool
    layerwise_load: bool



def get_lk_moe_config() -> LkMoeConfig:
    return LkMoeConfig(
        enabled=is_lk_moe_feature_enabled(),
        numa_interleave=is_numa_interleave_enabled(),
        pool_slots=get_pool_slots(),
        pool_enabled=is_lk_pool_enabled(),
        pool_decode_enabled=is_lk_pool_decode_enabled(),
        gpu_prefill_min_tokens=get_gpu_prefill_min_tokens(),
        gpu_prefill_engine=get_gpu_prefill_engine(),
        prefill_debug=is_lk_prefill_debug(),
        layerwise_load=enabled_layerwise_load(),
    )



@contextmanager
def park_lk_moe_expert_pool():
    """Park lk_moe's expert-pool pump around CUDA-graph capture/replay.

    The pump issues cudaMemcpyAsync from its own thread; overlapping that with
    cuGraphLaunch (or with an in-flight capture) deadlocks inside libcuda on
    P2P-patched drivers.  No-op when lk_moe is off or too old to expose the
    pause/resume hooks.
    """
    if not is_lk_pool_enabled():
        yield
        return
    import lk_moe

    # Pool is on, so these hooks are part of the lk_moe contract: a build that
    # lacks them is incompatible and must fail loudly here instead of silently
    # skipping the park (which would bring the cuGraphLaunch/copy deadlock
    # back).
    pause = lk_moe.pool_capture_pause
    resume = lk_moe.pool_capture_resume
    pause()
    try:
        yield
    finally:
        resume()


def wait_lk_moe_pool_fences(stream_ptr: int) -> None:
    """Attach the expert-pool copy fences to ``stream_ptr`` (pre-replay).

    With the deterministic split on, ``publish_snapshot`` publishes CLAIMED
    slots as soon as the pump's batch fence event is recorded, while the
    copies may still be in flight on the pump stream.  A graph replay that
    reads those slots must therefore stream-wait on the recorded fence
    first; ``pool_wait_all_fences`` does exactly nothing unless det_split
    is active.  Old lk_moe builds without the hook simply keep the
    pre-feature behavior.
    """
    if not is_lk_pool_enabled():
        return
    import lk_moe

    wait = getattr(lk_moe, "pool_wait_all_fences", None)
    if wait is not None:
        wait(stream_ptr)


def is_lk_moe_feature_enabled() -> bool:
    return get_bool_env_var("LK_MOE_HYBRID_ENABLED")

def is_numa_interleave_enabled() -> bool:
    # Doctrine: tier power follows the master switch - default on; 0 opts out.
    return get_bool_env_var("LK_NUMA_INTERLEAVE", default="true")

def get_gpu_prefill_min_tokens() -> int:
    # Dual knob (designs/lk_switch_doctrine.md rule 2): >0 both enables the GPU
    # prefill tier and sets its batch gate. Default 1024 (2026-11-07): the tier
    # is on out of the box -- a recipe no longer has to opt in, and 1024 tokens
    # is the smallest prompt where the GPU engine wins over the CPU path.
    # Forwarded to lk_moe via cfg; lk_moe itself never reads the env.
    return get_int_env_var("LK_GPU_PREFILL_MIN_TOKENS", 1024)

def get_gpu_prefill_engine() -> str:
    # Pure selector switch (lk_switch_doctrine: a selector never flips the
    # tier itself -- the GPU-prefill tier still powers on only via
    # LK_GPU_PREFILL_MIN_TOKENS>0 under the master switch). Chooses WHICH
    # GPU-prefill implementation a powered tier uses:
    #   "cutlass" (default) = restored CUTLASS grouped-GEMM engine (bf16
    #                         intermediates; that extra bf16 rounding drifts
    #                         long-context logits and can flip borderline EOS
    #                         -- see journal 2026-10-22; set "triton" to opt out)
    #   "triton"            = staging group-GEMM over pool slots (legacy,
    #                         fp32 activations, matches the CPU reference)
    #   "flashinfer"        = SM120/SM121 only: stage FP4, bridge the rows to
    #                         flashinfer's CuTe DSL moe_nvfp4_w4a16 layout and
    #                         run the FP4 tensor cores directly (no dequant,
    #                         1/4 the staged bytes; same numerics as W4A16)
    # Unset and unrecognized values both select cutlass (2026-10-23: restored
    # as the default, reversing the 2026-10-22 hotfix; every production launch
    # script now pins the engine explicitly, so this default only matters for
    # direct python embedding). Inert unless the tier is on. Forwarded to
    # lk_moe as cfg.gpu_prefill_engine ("" when the tier is off -- the string
    # carries the tier bit by being non-empty; the C++ decodes it into the
    # staging-ring / cutlass-arena buffer builds at ctor time).
    val = get_str_env_var("LK_GPU_PREFILL_ENGINE", "cutlass").strip().lower()
    return val if val in ("triton", "flashinfer") else "cutlass"

_GPU_PREFILL_QUANT_MODES = ("auto", "w4a4", "w4a8", "w4a16", "w8a8")

# THE validity table: (engine, requested mode) -> requirements.
#
#   frozenset()            always implementable by that engine
#   {"sm120"}              SM120 family only
#   {"nvfp4"}              NVFP4 weights only (MXFP4 needs a swizzled SF mirror)
#
# A pair that is absent from the table is NOT implemented: asking for it warns
# and degrades (precision is a speed setting, the fallback is still correct).
# Keeping this as a table is the point -- the engine x mode decision was spread
# over capability probes and ad-hoc sets, which nobody could read.
#
# Two tables, one per device family -- the same pair resolves differently:
#
# SM120 (5060 Ti / RTX PRO / B-series GeForce)
#   engine \ mode   auto        w4a4            w4a8   w4a16   w8a8
#   cutlass          -> w4a16    nvfp4           -      yes     -
#   triton           -> w4a16    -               -      yes     -
#   flashinfer       -> w4a16    nvfp4           -      yes     -
#
# sm8x / SM90 and everything else
#   cutlass          -> w4a16    -               -      yes     -
#   triton           -> w4a16    -               -      yes     -
#   flashinfer       - (SM12x-only kernels; its probe fails elsewhere)
#
# "auto" deliberately does not pick w4a4 until _PREFILL_AUTO_PREFERS_W4A4 is
# flipped: the explicit request is the opt-in, and there is no second knob.
_PREFILL_QUANT_MATRIX = {
    ("cutlass", "w4a16"): frozenset(),
    # cutlass reads both 4-bit weight families: NVFP4 (ModelOpt/Qwen) and
    # MXFP4 (DSV4 / DSV4.1 experts). MXFP4 only became addressable once the
    # prefill engine learned the native panel -> dense repack for its
    # 32-element scale blocks (lk_moe repack_panel_to_dense, 2026-11-10).
    # flashinfer's block-scaled kernel is NVFP4-only and stays gated below.
    ("cutlass", "w4a4"): frozenset({"sm120"}),
    ("triton", "w4a16"): frozenset(),
    ("flashinfer", "w4a16"): frozenset({"sm120"}),
    ("flashinfer", "w4a4"): frozenset({"sm120", "nvfp4"}),
}
_PREFILL_AUTO_PREFERS_W4A4 = False
_PREFILL_QUANT_WARNED = set()
_PREFILL_QUANT_LEGAL = {
    "nvfp4": ("w4a4", "w4a8", "w4a16"),
    "mxfp4": ("w4a4", "w4a8", "w4a16"),
    "fp8": ("w8a8", "w8a16"),
    "bf16": ("w8a16",),
}
_PREFILL_QUANT_FALLBACK = {"nvfp4": "w4a16", "mxfp4": "w4a16",
                           "fp8": "w8a16", "bf16": "w8a16"}


def get_gpu_prefill_quant() -> str:
    """The requested prefill precision: one knob, resolved per layer."""
    if get_bool_env_var("LK_GPU_PREFILL_SM120_W4A4"):
        raise ValueError(
            "LK_GPU_PREFILL_SM120_W4A4 is retired: set "
            "LK_GPU_PREFILL_QUANT=w4a4 instead")
    quant = (get_str_env_var("LK_GPU_PREFILL_QUANT", "auto") or "auto").strip().lower()
    if quant not in _GPU_PREFILL_QUANT_MODES:
        raise ValueError(
            f"LK_GPU_PREFILL_QUANT={quant!r} is not one of "
            f"{_GPU_PREFILL_QUANT_MODES}")
    return quant


def engine_quant_supported(engine: str, mode: str, weight_format: str,
                           is_sm120: bool) -> bool:
    """Table lookup: does `engine` implement `mode` for these weights here?"""
    req = _PREFILL_QUANT_MATRIX.get((engine, mode))
    if req is None:
        return False
    return all(
        (need != "sm120" or is_sm120) and
        (need != "nvfp4" or weight_format == "nvfp4")
        for need in req
    )


def resolve_gpu_prefill_quant(requested: str, engine: str, weight_format: str,
                              is_sm120: bool) -> str:
    """The mode this layer runs. Pure table + policy, no capability probing."""
    fmt = (weight_format or "").lower()
    if fmt not in _PREFILL_QUANT_LEGAL:
        raise ValueError(f"unknown prefill weight format {weight_format!r}")
    legal = _PREFILL_QUANT_LEGAL[fmt]

    def usable(mode):
        if mode == "w4a4" and not _PREFILL_AUTO_PREFERS_W4A4 and requested == "auto":
            return False
        return engine_quant_supported(engine, mode, fmt, is_sm120)

    if requested == "auto":
        for mode in legal:
            if usable(mode):
                return mode
        return _PREFILL_QUANT_FALLBACK[fmt]

    if requested not in legal:
        raise ValueError(
            f"LK_GPU_PREFILL_QUANT={requested} is not valid for {fmt} weights "
            f"(legal: {legal})")
    if engine_quant_supported(engine, requested, fmt, is_sm120):
        return requested
    # An explicit request that the engine lacks degrades to the SAFE mode for
    # that weight format (w4a16), never sideways into another 4-bit experiment:
    # the operator asked for 8-bit activations, silently handing back 4-bit
    # would be a worse surprise than keeping the known-good path.
    fallback = _PREFILL_QUANT_FALLBACK[fmt]
    key = (requested, engine, fmt, fallback)
    if key not in _PREFILL_QUANT_WARNED:
        _PREFILL_QUANT_WARNED.add(key)
        warnings.warn(
            f"LK_GPU_PREFILL_QUANT={requested} is not implemented by the "
            f"{engine} engine for {fmt} weights on this device -> prefill runs "
            f"{fallback}; this is a speed mode, not a correctness setting",
            stacklevel=3)
    return fallback


def get_pool_window() -> int:
    # Tuned default (hit ~81%). The ONLY pool switch is LK_POOL_SLOTS: any
    # illegal value (incl. <=0) clamps to the minimum 8, never disables --
    # a perf knob must not secretly flip the membership switch.
    return max(8, get_int_env_var("LK_POOL_WINDOW", 32))

def is_lk_gpu_prefill_enabled() -> bool:
    return is_lk_moe_feature_enabled() and get_gpu_prefill_min_tokens() > 0

def get_super_chunk_size() -> int:
    if not is_lk_gpu_prefill_enabled():
        return 0
    return get_int_env_var("LK_GPU_PREFILL_SUB_M")

def get_pool_inflight() -> int:
    # In-flight DMA jobs / event-ring depth of the decode pump; the
    # backpressure cap is 8x this -- 12 false-positives at small batch.
    return get_int_env_var("LK_POOL_INFLIGHT", 16)

def get_verify_resident() -> int:
    # VR: draft-row pool-resident experts computed GPU-side. Default ON since the
    # ride-share split (verify_pump stays off, so the draft rows only read the
    # residency the real tokens built): measured on 0731/DSPARK at accept 4.25 vs
    # 4.29 without it -- the fidelity cost the old comment warned about did not
    # materialise there -- while the pooled hit rate (48%) took the step from
    # 130 ms to 109 ms. LK_VERIFY_RESIDENT=0 keeps the old CPU-only behaviour.
    return get_int_env_var("LK_VERIFY_RESIDENT", 1)


def get_hf_mbox() -> int:
    # mailbox early-start decode path (default on; 0 = pure host-node path).
    return get_int_env_var("LK_HF_MBOX", 1)


def get_pool_log() -> int:
    # dstep interval of the C++ [pool] stats line (0 = silent).
    return get_int_env_var("LK_POOL_LOG")

def is_lk_prefill_debug() -> bool:
    # Extra tracing inside the prefill staging tier.
    return get_bool_env_var("LK_PP_DEBUG")

def get_prefill_wave_experts() -> int:
    # Tier wave width, ONE geometry for all three engines: <=0 (default)
    # = every expert in one wave (wave = E); >0 = W experts per wave, a
    # per-wave gemm loop only the cutlass engine has. The working set occupies
    # 2*wave rows of the pool's row space (shared, zero extra VRAM).
    #
    # The knob is cutlass-only (LK_GPU_PREFILL_WAVE_EXPERTS in the README
    # table), so a positive width means nothing to the staging engines: they
    # gemm the whole layer per stage. Normalise it here, once, instead of
    # letting the tier refuse to start a valid staging deployment.
    wave = get_int_env_var("LK_GPU_PREFILL_WAVE_EXPERTS", -1)
    if wave > 0:
        engine = get_gpu_prefill_engine()
        if engine != "cutlass":
            warnings.warn(
                f"LK_GPU_PREFILL_WAVE_EXPERTS={wave} is cutlass-only; ignoring "
                f"it for engine={engine} (auto = one wave per layer)"
            )
            return -1
    return wave


def get_prefill_d_slots() -> int:
    # Tier act window in slot rows, ONE formula for all engines: 0/-1
    # (default) = sub_m*topk + 7*E; >0 = that many rows, clamped. The
    # cutlass act planes are the tier arena's tail sized from it; the
    # staging drivers keep their own fixed-size launch-block buffers.
    return get_int_env_var("LK_GPU_PREFILL_D_SLOTS", 0)


def get_model_type_from_layer_name(layer_name: str) -> str:
    if not layer_name:
        return "main"
    
    if layer_name.startswith('stages.'):
        return "dspark"
    if layer_name.startswith('mtp.') or '.mtp.' in layer_name:
        return "mtp"
    return "main"

def get_gpu_resident_env_var(model_type: str = "main") -> str | None:
    if model_type in ("dspark", "mtp"):
        env_value = get_str_env_var("LK_GPU_RESIDENT_MOE_LAYERS_SPEC", None)
        if env_value is not None:
            return env_value

        role_env = (
            "LK_GPU_RESIDENT_MOE_LAYERS_MTP"
            if model_type == "mtp"
            else "LK_GPU_RESIDENT_MOE_LAYERS_DSPARK"
        )
        env_value = get_str_env_var(role_env, None)
        if env_value is not None:
            return env_value

    return get_str_env_var("LK_GPU_RESIDENT_MOE_LAYERS", None)

def is_lk_ple_probe_enabled() -> bool:
    """PLE ngram-row locality/traffic probe master switch."""
    return get_bool_env_var("LK_PLE_PROBE")

def get_lk_ple_probe_dump_path() -> str | None:
    """Probe row-dump path, or None (empty string means off)."""
    return get_str_env_var("LK_PLE_PROBE_DUMP", None) or None

def is_lk_moe_gpu_resident_layer(layer_id: str, model_type: str = "main") -> bool:
    if not is_lk_moe_feature_enabled():
        return True
     
    disabled_layers_env = get_gpu_resident_env_var(model_type)
    if not disabled_layers_env:
        return False   
    
    disabled_layers_env = disabled_layers_env.strip()
    
    disabled_layers = set()
    for part in disabled_layers_env.split(','):
        part = part.strip()
        if not part:
            continue
        
        if '-' in part:
            try:
                start, end = map(int, part.split('-')) 
                if start <= end:
                    disabled_layers.update(range(start, end + 1))
            except ValueError: 
                continue
        else:
            try:
                disabled_layers.add(int(part))
            except ValueError: 
                continue
     
    return layer_id in disabled_layers

def get_lk_pool_layer_set() -> set | None:
    # Same grammar as the resident-layer env; None (unset) = all non-resident
    # layers get a pool. The engine never reads this env: membership lands on
    # MOEConfigV2.gpu_pool at construction time.
    layers_env = get_str_env_var("LK_POOL_LAYERS", None)
    if not layers_env:
        return None

    pool_layers = set()
    for part in layers_env.strip().split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start <= end:
                    pool_layers.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                pool_layers.add(int(part))
            except ValueError:
                continue
    return pool_layers


def get_pool_slots() -> int:
    # Expert-cache slots per layer. 0 = cache off (prefill is sized
    # independently: the arena is max(cache, prefill)).
    return get_int_env_var("LK_POOL_SLOTS", 0)

def is_lk_pool_enabled() -> bool:
    return is_lk_moe_feature_enabled() and get_pool_slots() > 0

def is_lk_pool_layer(layer_id: str, model_type: str = "main") -> bool:
    if model_type != "main":
        return False
    if is_lk_moe_gpu_resident_layer(layer_id, model_type):
        return False
    return is_lk_pool_enabled()

def lk_route_probe_enabled() -> bool:
    return is_lk_pool_enabled() and get_bool_env_var("LK_ROUTE_PROBE")

def is_lk_pool_decode_enabled() -> bool:
    if not is_lk_moe_feature_enabled():
        return False
    return get_bool_env_var("LK_POOL_DECODE",
                            "true" if is_lk_pool_enabled() else "false")

def is_pool_mbox_split() -> bool:
    # Split the mailbox early-start path into "ring the doorbell" (on the pool
    # stream, before the GPU half's kernels) and "join" (on the main stream,
    # after them), so the CPU half's compute overlaps the resident half's kernels
    # instead of the main stream standing in a host node: hf_wait p50 was 14us and
    # hf_dec p50 12us, i.e. the CPU work was fully exposed.
    #
    # Default OFF, because it measured as nothing and the reason is worth keeping
    # next to the code: q36/2x5060Ti, same build, 128 slots, 7300-step stream --
    # 7.43 vs 7.07-7.35 ms/step at one row (no better), 9.70 at two rows (same),
    # while hf_dec p50 went 12us -> 47-52us. The dependency did not move: the
    # layer cannot consume the expert output before the CPU half is done, so
    # reordering changes which wait the stall is *counted as*, and the CPU work
    # costs more when it runs beside the GPU work it was meant to hide. What the
    # block is really bounded by is max(CPU compute, resident-half kernels), so
    # shortening it means making one of those two smaller, not rearranging them.
    # Correctness of the split was checked while it was on: cmiss=cdup=0 (every
    # routed occurrence still owned by exactly one side) and no
    # "join without doorbell" fallback fired.
    return is_lk_moe_feature_enabled() and get_bool_env_var("LK_POOL_MBOX_SPLIT")


def enabled_layerwise_load() -> bool:
    return get_bool_env_var("LK_MOE_LAYERWISE_LOAD")

