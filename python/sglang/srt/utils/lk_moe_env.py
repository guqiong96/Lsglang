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
    # Dual knob (designs/lk_switch_doctrine.md rule 2): default 0 = tier off,
    # pure CPU prefill; >0 both enables the tier and sets the batch gate.
    # Forwarded to lk_moe via cfg; lk_moe itself never reads the env.
    return get_int_env_var("LK_GPU_PREFILL_MIN_TOKENS", 0)

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

def is_gpu_prefill_sm120_w4a4_enabled() -> bool:
    # Quantization-level switch for the LK GPU-prefill tier (2026-10-27
    # doctrine: three orthogonal factors, any one off degrades safely to
    # W4A16, whose numerics equal the BF16-activation paths). Two of them
    # are decided here, in one place: this env (pure switch, default off --
    # lk_switch_doctrine rule 2) and the device capability (SM120 family,
    # hard check; the three-key rule -- capability decides correctness,
    # never the name). The third is the engine's w4a4_support(), asked by
    # the caller against the driver it just created. There is NO second
    # token threshold inside the tier: the batch gate is
    # LK_GPU_PREFILL_MIN_TOKENS and nothing else (the retired
    # MIN_TOKENS-version threshold left no compatibility layer).
    if not get_bool_env_var("LK_GPU_PREFILL_SM120_W4A4"):
        return False
    if torch.cuda.get_device_capability()[0] != 12:  # SM120 family
        # warnings.warn dedups per call site: one line, not one per layer.
        warnings.warn(
            "LK_GPU_PREFILL_SM120_W4A4=1 but this device is not SM120-"
            "family -> prefill stays W4A16", stacklevel=2)
        return False
    return True

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
    # VR: draft-row pool-resident experts computed GPU-side; has an acceptance
    # fidelity tradeoff -> python-owned cfg value (LK_* env is the script UI).
    return get_int_env_var("LK_VERIFY_RESIDENT", 0)


def get_verify_skip_miss() -> int:
    # verify rows drop CPU-side misses (distribution declaration, not bitwise).
    return get_int_env_var("LK_VERIFY_SKIP_MISS", 0)


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


def get_pool_decode_engine() -> str:
    # Pure selector switch (same doctrine as get_gpu_prefill_engine): the pool
    # decode tier is powered by LK_POOL_DECODE, this only picks the
    # implementation of its RESIDENT half (the CPU half and the paired-skip
    # split are identical either way):
    #   "gemv"   (default) = the three Triton GEMV launches + two reduces
    #   "sm12x"            = SM120/SM121 only: one fused flashinfer W4A16 call
    #                        on the pool's canonical regions (zero copy). Needs
    #                        the canonical slot layout, i.e. the flashinfer
    #                        prefill engine and intermediate % 128 == 0; logs
    #                        the reason and stays on "gemv" when unavailable.
    # The kernel's intermediate is FP16 (measured 3e-3 against an fp32 oracle
    # where the GEMV path is 2e-7), so this is a numerics CLASS, not a free
    # speed switch -- see journal 2026-10-30. Gate that class with the unit test
    # against an fp32 oracle, NOT with model-level statistics: here the SAME
    # engine run twice disagreed on 8 of 50 gsm8k questions and differed by 3
    # questions of accuracy (43 vs 46), so nothing measured at that level can see
    # a 3e-3 kernel difference -- greedy drift and per-token logprobs are buried
    # the same way, because which half of the model produced a logit moves with
    # the resident set.
    #
    # What it buys, measured on q36/2x5060Ti with the same build and one long
    # stream per configuration: the resident half's kernels cost 2.1x less GPU
    # time (0.71 vs 1.51 ms/step) and the STEP moves +5.4% at one row per step
    # (7.21 ms mean of six runs against 7.60) and +3.3% at two rows (9.70 against
    # 10.03). It is not a bs=1 lever: at one row per step most of that
    # GPU saving was already being hidden by the attention projections, the
    # all-reduces and the CPU half, which is why a kernel-only benchmark
    # overpromises by ~2.5x.
    val = get_str_env_var("LK_POOL_DECODE_ENGINE", "gemv").strip().lower()
    return val if val in ("gemv", "sm12x") else "gemv"

def enabled_layerwise_load() -> bool:
    return get_bool_env_var("LK_MOE_LAYERWISE_LOAD")

