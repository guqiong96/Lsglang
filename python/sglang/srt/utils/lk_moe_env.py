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
    return get_int_env_var("LK_GPU_PREFILL_MIN_TOKENS", 1024)

def get_gpu_prefill_engine() -> str:
    if not is_lk_gpu_prefill_enabled():
        return ""
    return get_str_env_var("LK_GPU_PREFILL_ENGINE", "cutlass").lower()

_LK_PREFILL_QUANT_OK = {
    # (arch, engine, quant); present = legal, absent = raise.
    ("sm120", "cutlass",    "w4a4"),
    ("sm120", "cutlass",    "w4a16"),
    ("other", "cutlass",    "w4a16"),
    ("sm120", "triton",     "w4a16"),
    ("other", "triton",     "w4a16"),
    ("sm120", "flashinfer", "w4a16"),
    ("sm120", "flashinfer", "w4a4"),
}


def get_gpu_prefill_quant() -> str:
    return get_str_env_var("LK_GPU_PREFILL_QUANT", "auto").lower()


def resolve_prefill_quant() -> str:
    engine = get_gpu_prefill_engine()
    if not engine:
        return ""
    quant = get_gpu_prefill_quant()
    mode = "w4a16" if quant == "auto" else quant
    arch = "sm120" if torch.cuda.get_device_capability()[0] == 12 else "other"
    if (arch, engine, mode) not in _LK_PREFILL_QUANT_OK:
        raise ValueError(f"LK_GPU_PREFILL_QUANT={quant}: not in "
                         f"_LK_PREFILL_QUANT_OK on {arch} for engine {engine!r}")
    return mode


def is_prefill_w4a4(weight_format: str) -> bool:
    return resolve_prefill_quant() == "w4a4" and weight_format in ("nvfp4", "mxfp4")


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

def get_verify_pump() -> bool:
    return get_bool_env_var("LK_VERIFY_PUMP", "false")


def get_verify_resident() -> bool:
    # VR: draft-row pool-resident experts computed GPU-side. Default ON since the
    # ride-share split (verify_pump stays off, so the draft rows only read the
    # residency the real tokens built): measured on 0731/DSPARK at accept 4.25 vs
    # 4.29 without it -- the fidelity cost the old comment warned about did not
    # materialise there -- while the pooled hit rate (48%) took the step from
    # 130 ms to 109 ms. LK_VERIFY_RESIDENT=0 keeps the old CPU-only behaviour.
    return get_bool_env_var("LK_VERIFY_RESIDENT", "true")


def get_hf_mbox() -> bool:
    # mailbox early-start decode path (default on; 0 = pure host-node path).
    return get_bool_env_var("LK_HF_MBOX", "true")


def get_pool_log() -> int:
    # dstep interval of the C++ [pool] stats line (0 = silent).
    return get_int_env_var("LK_POOL_LOG")

def is_lk_prefill_debug() -> bool:
    # Extra tracing inside the prefill staging tier.
    return get_bool_env_var("LK_PP_DEBUG")


def get_pp_debug_every() -> int:
    return get_int_env_var("LK_PP_DEBUG_EVERY", 0)


def is_lk_pool_det_split() -> bool:
    return is_lk_moe_feature_enabled() and get_bool_env_var("LK_POOL_DET_SPLIT")


def is_lk_pool_split_probe() -> bool:
    return is_lk_moe_feature_enabled() and get_bool_env_var("LK_POOL_SPLIT_PROBE")

def get_prefill_wave_experts() -> int:
    # Tier wave width, ONE geometry for all three engines: <=0 (default)
    # = every expert in one wave (wave = E); >0 = W experts per wave, a
    # per-wave gemm loop only the cutlass engine has. The working set occupies
    # 2*wave rows of the pool's row space (shared, zero extra VRAM).
    #
    # The knob is cutlass-only (LK_GPU_PREFILL_WAVE_EXPERTS in the README
    # table), so a positive width means nothing to the whole-window engines: they
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
    # cutlass act planes are the tier global region's tail sized from it; the
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
    """Resident (never pooled) layers, per ROLE.

    `LK_GPU_RESIDENT_MOE_LAYERS` names MAIN layers only. A draft used to inherit
    that list through a fallback, and the consequence was not cosmetic: draft
    models restart their layer ids at 0, so `...=0-1,33-34` silently marked the
    draft's own layer 0 resident, the draft never registered as a pool member,
    and the "draft never joins the pool" question stayed open for days
    (2026-11-17). Drafts therefore read ONLY their own variables:
        LK_GPU_RESIDENT_MOE_LAYERS_SPEC        (all spec kinds, wins first)
        LK_GPU_RESIDENT_MOE_LAYERS_MTP         (mtp / NextN drafts)
        LK_GPU_RESIDENT_MOE_LAYERS_DSPARK      (dspark drafts)
    Unset draft variables mean "the draft JOINS the pool".
    """
    if model_type in ("dspark", "mtp"):
        env_value = get_str_env_var("LK_GPU_RESIDENT_MOE_LAYERS_SPEC", None)
        if env_value:
            return env_value
        role_env = (
            "LK_GPU_RESIDENT_MOE_LAYERS_MTP"
            if model_type == "mtp"
            else "LK_GPU_RESIDENT_MOE_LAYERS_DSPARK"
        )
        return get_str_env_var(role_env, None)

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
    # independently: the global region is max(cache, prefill)).
    return get_int_env_var("LK_POOL_SLOTS", 0)

def get_draft_pool_slots() -> int:
    # Slots DRAFT layers admit to (0 = all of them, like the main layers).
    # A quantity knob, not a structure knob: the row addressing is shared and
    # unchanged, only the LRU admission scan stops early. Keep it >= top_k.
    return get_int_env_var("LK_POOL_DRAFT_SLOTS", 0)

def is_lk_pool_enabled() -> bool:
    return is_lk_moe_feature_enabled() and get_pool_slots() > 0

_pool_main_layers: list = []
_pool_draft_layers: list = []
_pool_ids: dict = {}


def is_main_layer(model_type: str) -> bool:
    return model_type == "main"


def is_draft_layer(model_type: str) -> bool:
    return model_type != "main"


def regist_lk_pool_member(layer_name: str, model_type: str) -> None:
    # The key is the layer's NAME (prefix), never its model-local index: the
    # draft model restarts the index at 0, and two layers sharing a key would
    # hand one member id to both -- the C++ registry then declines the
    # second instance ("the registry gave one layer two numbers").
    lid = str(layer_name)
    if lid in _pool_ids:
        return
    _pool_ids[lid] = len(_pool_main_layers) + len(_pool_draft_layers)
    if get_pool_log() > 0:
        import logging as _lg
        _lg.getLogger(__name__).info(
            "[pool] registry-add %s type=%s id=%d (main=%d draft=%d)",
            lid, model_type, _pool_ids[lid],
            len(_pool_main_layers), len(_pool_draft_layers))
    (_pool_main_layers if is_main_layer(model_type)
     else _pool_draft_layers).append(lid)


def lk_pool_id(layer_name: str) -> int:
    # THE pool-registry key: the member id the registry handed out at
    # registration (keyed by layer name, unique across main and draft). Main
    # and draft layers take their numbers from this one counter, so ids are
    # dense, unique, and no consumer can tell the two kinds apart.
    # -1 = not a member.
    return _pool_ids.get(str(layer_name), -1)


_member_map_logged: tuple = ()


def log_pool_member_map() -> None:
    # The pump's per-layer lines name themselves with the CENSUS ORDINAL
    # ("[pool] L43 ..."), which only python can turn back into a layer. Dump
    # the ordinal -> layer map once the registry is complete (called after the
    # draft worker exists; re-dumped if it grew), and say which end of the
    # counter is whose: that split is exactly what an aggregate hit rate hides.
    # Gated on LK_POOL_LOG: it exists to be read together with the pump lines.
    global _member_map_logged
    if get_pool_log() <= 0:
        return
    snap = (len(_pool_main_layers), len(_pool_draft_layers))
    if snap == _member_map_logged:
        return
    _member_map_logged = snap
    logger.info(
        "[pool] registry: %d member layers = %d main (L0..L%d) + %d draft "
        "(L%d..L%d); draft layers are %s",
        sum(snap), snap[0], max(snap[0] - 1, 0), snap[1],
        snap[0], max(snap[0] + snap[1] - 1, 0),
        _pool_draft_layers or "none",
    )


_spec_draft_moe_layers: int = 0


def record_spec_draft_moe_layers(n: int) -> None:
    # The draft worker records its MoE layer count from ITS OWN draft config
    # (tp_worker's ModelConfig), because the first lk_moe ctor -- the one that
    # sizes the pool's global region -- can run before any draft FusedMoE
    # exists. Idempotent (recorded twice: draft config is parsed twice).
    global _spec_draft_moe_layers
    import os as _os
    if get_pool_log() > 0:
        import logging as _lg
        _lg.getLogger(__name__).info(
            "[pool] registry-spec record n=%d pid=%d (was %d)", int(n),
            _os.getpid(), _spec_draft_moe_layers)
    _spec_draft_moe_layers = max(_spec_draft_moe_layers, int(n))


def get_pool_member_layers() -> int:
    # How many layers have an expert cache. Draft layers are counted at their own
    # member id like any other member: adding a draft layer adds cache rows,
    # never a different kind of memory.
    _n = len(_pool_main_layers) + max(len(_pool_draft_layers),
                                      _spec_draft_moe_layers)
    if get_pool_log() > 0:
        import logging as _lg
        _lg.getLogger(__name__).info(
            "[pool] registry-count main=%d draft=%d spec_declared=%d -> %d",
            len(_pool_main_layers), len(_pool_draft_layers),
            _spec_draft_moe_layers, _n)
    return _n


def is_lk_pool_layer(layer_name: str) -> bool:
    # Membership by the SAME key as the registry: the layer name.
    return str(layer_name) in _pool_main_layers or \
        str(layer_name) in _pool_draft_layers

def lk_route_probe_enabled() -> bool:
    return is_lk_pool_enabled() and get_bool_env_var("LK_ROUTE_PROBE")

def is_lk_pool_decode_enabled() -> bool:
    if not is_lk_moe_feature_enabled():
        return False
    return get_bool_env_var("LK_POOL_DECODE",
                            "true" if is_lk_pool_enabled() else "false")

def is_lk_pool_decode_layer(layer_name: str) -> bool:
    return is_lk_pool_decode_enabled() and is_lk_pool_layer(layer_name)

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

