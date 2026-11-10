"""Memory budget for the fp32 MQA-logits scratch of the DeepSeek sparse
attention indexers (DSA, DSV4, DSV4.1).

The indexer scores every query row against every key column into one fp32
logits matrix that no pool sized by mem_fraction_static accounts for. These
helpers bound that matrix and slice it by query rows.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import get_device_module, is_hip, is_xpu
from sglang.srt.utils.common import ceil_div

MQA_LOGITS_BYTES_PER_ELEM = 4
MQA_LOGITS_STATIC_SKIP_ELEMS = 8_000_000
MQA_LOGITS_TOTAL_MEM_FRACTION = 0.3
# aiter's fp8_mqa_logits only compiles below 2 GiB of logits (buffer_store).
MQA_LOGITS_MAX_BYTES_ROCM = 2**31 - 1
# DeepGEMM pads the logits row stride to 1024 bytes, i.e. 256 fp32 columns.
MQA_LOGITS_ROW_ALIGN_ELEMS = 256


def mqa_logits_free_mem_fraction() -> float:
    return envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()


def mqa_logits_needs_budget_check(*, num_rows: int, num_cols: int) -> bool:
    return num_rows * num_cols >= MQA_LOGITS_STATIC_SKIP_ELEMS


def mqa_logits_row_bytes(num_cols: int) -> int:
    aligned_cols = (
        ceil_div(num_cols, MQA_LOGITS_ROW_ALIGN_ELEMS) * MQA_LOGITS_ROW_ALIGN_ELEMS
    )
    return aligned_cols * MQA_LOGITS_BYTES_PER_ELEM


def mqa_logits_static_budget_bytes(*, device_index: int) -> int:
    """Budget from configuration alone (no device query); safe during graph capture."""
    total_mem = get_device_module().get_device_properties(device_index).total_memory
    total_mem_budget = int(total_mem * MQA_LOGITS_TOTAL_MEM_FRACTION)
    mem_fraction_static = get_schedule().mem_fraction_static
    if mem_fraction_static is None:
        budget = total_mem_budget
    else:
        static_free_mem = int(total_mem * max(0.0, 1.0 - mem_fraction_static))
        budget = min(
            int(static_free_mem * mqa_logits_free_mem_fraction()), total_mem_budget
        )
    return max(1, budget)


def mqa_logits_budget_bytes(*, device_index: int, allow_sync: bool) -> int:
    """Static budget capped by current free memory; mem_get_info syncs, so
    callers pass allow_sync=False under CUDA graph capture."""
    budget = mqa_logits_static_budget_bytes(device_index=device_index)
    if allow_sync and not is_xpu():
        free_mem, _ = torch.cuda.mem_get_info(device_index)
        budget = min(int(free_mem * mqa_logits_free_mem_fraction()), budget)
    if is_hip():
        budget = min(budget, MQA_LOGITS_MAX_BYTES_ROCM)
    return max(1, budget)


# One persistent fp32 logits scratch per device, held by whoever first needs a
# row-chunked logits matrix.
_LOGITS_WORKSPACES: dict[int, torch.Tensor] = {}


def mqa_logits_workspace(device_index: int, *, want_bytes: int) -> torch.Tensor:
    """The persistent logits buffer, grown to ``want_bytes`` if it is smaller.

    A row-chunk plan is built when the forward's metadata is built, but the
    scores used to be allocated when each chunk ran. On a long prefill those two
    moments are minutes and gigabytes apart, and the budget the plan trusted had
    been spent meanwhile: the chunk then died on an allocation the plan had
    already accounted for. Holding the bytes here makes the plan's budget a fact
    -- the same fix the gfx950 fused indexer applies to its own workspace, which
    takes it while weights load so the pools are not sized against memory that is
    about to be taken.

    Grow-only, and never grown past the budget at the moment it grows: a forward
    that wants more rows than the buffer holds gets more, smaller chunks rather
    than an allocation that may no longer be there, which is what turns an
    over-budget 256K prefill from a crash into a slower run.
    """
    ws = _LOGITS_WORKSPACES.get(device_index)
    want_bytes = max(int(want_bytes), 0)
    have = 0 if ws is None else ws.numel() * MQA_LOGITS_BYTES_PER_ELEM
    if have >= want_bytes and want_bytes > 0:
        return ws
    budget = mqa_logits_budget_bytes(device_index=device_index, allow_sync=True)
    # Grow as far as the memory actually allows. When even what is already held
    # is past that, leave the buffer alone: a live buffer cannot shrink.
    target = min(max(want_bytes, MQA_LOGITS_BYTES_PER_ELEM), budget)
    if target <= have:
        target = max(have, MQA_LOGITS_BYTES_PER_ELEM)
    elems = max(target // MQA_LOGITS_BYTES_PER_ELEM, 1)
    grown = torch.empty(elems, dtype=torch.float32, device=f"cuda:{device_index}")
    if ws is not None and ws.numel():
        grown[: ws.numel()] = ws
    _LOGITS_WORKSPACES[device_index] = grown
    return grown


def mqa_logits_workspace_bytes(device_index: int) -> int:
    """Bytes currently held, so a planner can chunk against memory that exists."""
    ws = _LOGITS_WORKSPACES.get(device_index)
    return 0 if ws is None else ws.numel() * MQA_LOGITS_BYTES_PER_ELEM


def mqa_logits_should_chunk(
    *,
    num_rows: int,
    num_cols: int,
    get_budget_bytes: Callable[[], int],
    rocm: bool,
) -> Tuple[bool, int]:
    """Whether a [num_rows, num_cols] fp32 logits matrix must be row-chunked.

    Returns (need_chunk, effective_budget_bytes). Matrices below
    MQA_LOGITS_STATIC_SKIP_ELEMS never chunk and the budget is not queried
    (it may synchronize the host), hence the callable. On ROCm the budget is
    also capped by aiter's logits ceiling.
    """
    if not mqa_logits_needs_budget_check(num_rows=num_rows, num_cols=num_cols):
        return False, 0
    budget_bytes = get_budget_bytes()
    if rocm:
        budget_bytes = min(budget_bytes, MQA_LOGITS_MAX_BYTES_ROCM)
    logits_bytes = num_rows * num_cols * MQA_LOGITS_BYTES_PER_ELEM
    return logits_bytes > budget_bytes, budget_bytes


def mqa_logits_rows_per_chunk(
    *, num_rows: int, row_bytes: int, budget_bytes: int
) -> Optional[int]:
    """Query rows per chunk so one logits chunk fits the budget; None if all rows fit."""
    if num_rows * row_bytes <= budget_bytes:
        return None
    rows = max(budget_bytes // max(row_bytes, 1), 1)
    return int(rows) if rows < num_rows else None
