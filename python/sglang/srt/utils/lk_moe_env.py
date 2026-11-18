"""lk_moe configuration surface: env parsing, layer predicates and the
expert-pool pump park helper.

Everything LK_* lives here so the knobs have a single source of truth; callers
keep importing the names from ``sglang.srt.utils.common`` (compat re-export).
"""

import warnings

import torch

from contextlib import contextmanager

from sglang.srt.utils.env_vars import (
    get_bool_env_var,
    get_int_env_var,
    get_str_env_var,
)


@contextmanager
def park_lk_moe_expert_pool():
    if not is_lk_pool_enabled():
        yield
        return
    import lk_moe

    # The pool's side streams must be idle while a graph captures.
    pause = lk_moe.pool_capture_pause
    resume = lk_moe.pool_capture_resume
    pause()
    try:
        yield
    finally:
        resume()


def wait_lk_moe_pool_fences(stream_ptr: int) -> None:
    """Attach the expert-pool copy fences to ``stream_ptr`` (pre-replay).

    With det_split on, slots are published before their copies land, so a
    replay that reads them must wait on the fences first.
    """
    if not is_lk_pool_enabled():
        return
    import lk_moe

    wait = getattr(lk_moe, "pool_wait_all_fences", None)
    if wait is not None:
        wait(stream_ptr)


def is_lk_moe_feature_enabled() -> bool:
    return get_bool_env_var("LK_MOE_HYBRID_ENABLED", default="false")

def is_numa_interleave_enabled() -> bool:
    return get_bool_env_var("LK_NUMA_INTERLEAVE", default="true")

def get_gpu_prefill_min_tokens() -> int:
    return get_int_env_var("LK_GPU_PREFILL_MIN_TOKENS", 1024)

def get_gpu_prefill_engine() -> str:
    if not is_lk_gpu_prefill_enabled():
        return ""
    return get_str_env_var("LK_GPU_PREFILL_ENGINE", "cutlass").lower()

_LK_PREFILL_QUANT_OK = {
    # (arch, engine, quant)
    ("sm120", "cutlass",    "w4a4"),
    ("sm120", "cutlass",    "w4a16"),
    ("other", "cutlass",    "w4a16"),
    ("sm120", "triton",     "w4a16"),
    ("other", "triton",     "w4a16"),
    ("sm120", "flashinfer", "w4a16"),
    ("sm120", "flashinfer", "w4a4"),
}


def get_gpu_prefill_quant() -> str:
    engine = get_gpu_prefill_engine()
    if not engine:
        return ""
    mode = get_str_env_var("LK_GPU_PREFILL_QUANT", "w4a16").lower()
    arch = "sm120" if torch.cuda.get_device_capability()[0] == 12 else "other"
    if (arch, engine, mode) not in _LK_PREFILL_QUANT_OK:
        raise ValueError(f"LK_GPU_PREFILL_QUANT={mode}: not in "
                         f"_LK_PREFILL_QUANT_OK on {arch} for engine {engine!r}")
    return mode


def is_prefill_w4a4(weight_format: str) -> bool:
    return get_gpu_prefill_quant() == "w4a4" and weight_format in ("nvfp4", "mxfp4")


def get_pool_window() -> int:
    # Floor 8: a perf knob must never disable the pool.
    return max(8, get_int_env_var("LK_POOL_WINDOW", 32))

def is_lk_gpu_prefill_enabled() -> bool:
    return is_lk_moe_feature_enabled() and get_gpu_prefill_min_tokens() > 0

def get_super_chunk_size() -> int:
    if not is_lk_gpu_prefill_enabled():
        return 0
    return get_int_env_var("LK_GPU_PREFILL_SUB_M", default=0)

def get_pool_inflight() -> int:
    # Pump in-flight depth; backpressure caps at 8x it. 12 false-positived.
    return get_int_env_var("LK_POOL_INFLIGHT", 16)

def get_verify_pump() -> bool:
    return get_bool_env_var("LK_VERIFY_PUMP", "false")


def get_verify_resident() -> bool:
    # Draft rows ride the residency the real tokens built: 130 -> 109 ms.
    return get_bool_env_var("LK_VERIFY_RESIDENT", "true")


def get_hf_mbox() -> bool:
    # Early-start the CPU half of the decode over the mailbox.
    return get_bool_env_var("LK_HF_MBOX", "true")


def get_pool_log() -> int:
    # Step interval of the C++ [pool] stats line.
    return get_int_env_var("LK_POOL_LOG", default=0)

def is_lk_prefill_debug() -> bool:
    return get_bool_env_var("LK_PP_DEBUG", default="false")


def get_pp_debug_every() -> int:
    return get_int_env_var("LK_PP_DEBUG_EVERY", 0)


def is_lk_pool_det_split() -> bool:
    return is_lk_moe_feature_enabled() and get_bool_env_var(
        "LK_POOL_DET_SPLIT", default="false")


def is_lk_pool_split_probe() -> bool:
    return is_lk_moe_feature_enabled() and get_bool_env_var(
        "LK_POOL_SPLIT_PROBE", default="false")

def get_prefill_wave_experts() -> int:
    # Wave width is cutlass-only; normalise it here, not in the tier.
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
    # Act window in slot rows; default sub_m*topk + 7*E covers one pass.
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

    The main list applies to MAIN only: draft ids restart at 0, so inheriting
    it would mark the draft's layer 0 resident and keep the draft out of the
    pool. An unset draft variable means the draft joins.
        LK_GPU_RESIDENT_MOE_LAYERS_SPEC    (all spec kinds, wins first)
        LK_GPU_RESIDENT_MOE_LAYERS_MTP     (mtp / NextN drafts)
        LK_GPU_RESIDENT_MOE_LAYERS_DSPARK  (dspark drafts)
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
    return get_bool_env_var("LK_PLE_PROBE", default="false")

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

def get_pool_slots() -> int:
    # 0 = cache off; prefill still gets its own bytes (region = max of both).
    return get_int_env_var("LK_POOL_SLOTS", 0)

def get_draft_pool_slots() -> int:
    # Slots draft layers admit to (0 = all). Keep it >= top_k.
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
    # Key by layer NAME, not model-local index: draft restarts at 0.
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
    # Pool-registry key: one counter for main and draft. -1 = not a member.
    return _pool_ids.get(str(layer_name), -1)


_member_map_logged: tuple = ()


def log_pool_member_map() -> None:
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
    # The region is sized before any draft layer exists, so it declares its count.
    global _spec_draft_moe_layers
    import os as _os
    if get_pool_log() > 0:
        import logging as _lg
        _lg.getLogger(__name__).info(
            "[pool] registry-spec record n=%d pid=%d (was %d)", int(n),
            _os.getpid(), _spec_draft_moe_layers)
    _spec_draft_moe_layers = max(_spec_draft_moe_layers, int(n))


def get_pool_member_layers() -> int:
    # The draft layers may not have registered yet, so take the larger source.
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
    return str(layer_name) in _pool_main_layers or \
        str(layer_name) in _pool_draft_layers

def lk_route_probe_enabled() -> bool:
    return is_lk_pool_enabled() and get_bool_env_var(
        "LK_ROUTE_PROBE", default="false")

def is_lk_pool_decode_enabled() -> bool:
    if not is_lk_moe_feature_enabled():
        return False
    return get_bool_env_var("LK_POOL_DECODE",
                            "true" if is_lk_pool_enabled() else "false")

def is_lk_pool_decode_layer(layer_name: str) -> bool:
    return is_lk_pool_decode_enabled() and is_lk_pool_layer(layer_name)

def is_pool_mbox_split() -> bool:
    # Default off: overlapping the CPU half with the resident kernels gained nothing.
    return is_lk_moe_feature_enabled() and get_bool_env_var(
        "LK_POOL_MBOX_SPLIT", default="false")


def enabled_layerwise_load() -> bool:
    return get_bool_env_var("LK_MOE_LAYERWISE_LOAD", default="false")

