"""Group-wide architecture consensus for one TP group.

A heterogeneous TP group (SM86 + SM120, Hopper + Blackwell, ...) has to make
every decision that changes a forward's *collective footprint* -- how many
all-reduces a captured graph emits, in what order, over how many rows -- once,
for the whole group. A per-rank ``get_platform()`` read makes the ranks capture
different graphs, and the group then busy-waits in the first collective the
captures disagree about. That is one bug wearing two faces: during warmup it is
the 2026-10-18 TP4 capture deadlock, later on it is a decode that never finishes
with the GPUs at 100%.

``tp_all`` is the single place that decision is made: evaluate a predicate
locally, AND it across the TP group with one int32 all-reduce, cache the answer.
A homogeneous group gets its own local value back, so nothing changes there; a
mixed group gets the common denominator (one "no" is enough).

The rules (designs/hetero_tp_decision_doctrine.md):

* a decision that only picks a local kernel may stay per-rank;
* a decision about collective structure / count / order goes through here, and
  only here -- no model module keeps its own copy of this helper;
* every predicate in ``PREDICATES`` is resolved once at model init by
  :func:`prewarm_tp_arch_predicates`, so its collective never lands inside a
  CUDA graph. ``tp_all`` refuses to resolve an unknown-in-cache predicate while
  capturing instead of deadlocking the group, which is what turns a forgotten
  prewarm into one loud error rather than one un-debuggable hang.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import torch

logger = logging.getLogger(__name__)


def _blackwell() -> bool:
    from sglang.srt.runtime_context import get_platform

    return bool(get_platform().is_blackwell)


def _sm90() -> bool:
    from sglang.srt.runtime_context import get_platform

    return bool(get_platform().is_sm90)


def _hopper_or_blackwell() -> bool:
    return _sm90() or _blackwell()


def _sm8() -> bool:
    # SM80/SM86/SM89 (Ampere/Ada) only: the decode wo_a low-rank runs as a bf16
    # grouped einsum with the inverse RoPE folded in, which is a *forward shape*
    # difference, not a kernel choice -- a rank that fuses the rope while its
    # peers run the standalone rope would feed them a different tensor. Kept
    # lazy: the probes key on the current device, which is only bound per rank
    # after import on a mixed host.
    from sglang.srt.utils import is_sm80_supported, is_sm120_supported

    return bool(is_sm80_supported()) and not bool(is_sm120_supported())


# Insertion order is the prewarm order, and every rank walks it the same way:
# the consensus all-reduce carries no predicate name, so a fixed order is what
# keeps two ranks from pairing their first calls with different predicates.
PREDICATES: Dict[str, Callable[[], bool]] = {
    "blackwell": _blackwell,
    "sm90": _sm90,
    "hopper_or_blackwell": _hopper_or_blackwell,
    "sm8": _sm8,
}

_TP_ARCH_CACHE: Dict[str, bool] = {}


def _capturing() -> bool:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return True
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        return bool(get_is_capture_mode())
    except Exception:
        return False


def _flag_device() -> object:
    """Where to put the consensus flag (a plain CPU tensor when there is no GPU
    in the process, e.g. a CPU-only test run)."""
    return torch.cuda.current_device() if torch.cuda.is_available() else "cpu"


def tp_all(pred: str) -> bool:
    """Whether EVERY rank in the TP group satisfies ``pred``."""
    cached = _TP_ARCH_CACHE.get(pred)
    if cached is not None:
        return cached
    check = PREDICATES.get(pred)
    if check is None:
        raise KeyError(
            f"unknown TP arch predicate {pred!r}; add it to "
            f"sglang.srt.distributed.arch_consensus.PREDICATES (prewarm and the "
            f"group-consensus test follow from the table)"
        )
    if _capturing():
        raise RuntimeError(
            f"TP arch predicate {pred!r} was first resolved while a CUDA graph "
            "was being captured; its consensus all-reduce would deadlock a "
            "heterogeneous group. Resolve it at model init "
            "(prewarm_tp_arch_predicates) or from an earlier init path."
        )
    local = bool(check())
    from sglang.srt.runtime_context import get_parallel

    try:
        group = get_parallel().tp_group
    except RuntimeError:
        # Nothing has published a TP group: a bare process (tool, unit test) or
        # a single-rank run. No peers to agree with, so the local answer is the
        # group answer -- which is what a homogeneous group resolves to anyway.
        group = None
    world = getattr(group, "world_size", 1)
    if (
        world > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        flag = torch.tensor([1 if local else 0], dtype=torch.int32, device=_flag_device())
        torch.distributed.all_reduce(flag, group=group.device_group)
        result = int(flag.item()) == world
    else:
        result = local
    _TP_ARCH_CACHE[pred] = result
    if result != local:
        # Only a mixed group can get here: the local answer was overruled, i.e.
        # this rank loses an optimization its hardware would have used.
        logger.info(
            "TP group is architecturally mixed: %s resolved to %s on this rank "
            "(locally %s)",
            pred,
            result,
            local,
        )
    return result


def prewarm_tp_arch_predicates() -> None:
    """Resolve every predicate once, outside any CUDA graph.

    Called from model init. Adding a predicate to ``PREDICATES`` needs no other
    wiring: it is prewarmed here, and test_arch_consensus asserts the table and
    the cache agree.
    """
    for pred in PREDICATES:
        tp_all(pred)


def resolved_tp_arch_predicates() -> Dict[str, bool]:
    """The predicates resolved so far (for diagnostics and tests)."""
    return dict(_TP_ARCH_CACHE)


def clear_tp_arch_predicates() -> None:
    """Drop the cache. Test hook only."""
    _TP_ARCH_CACHE.clear()
