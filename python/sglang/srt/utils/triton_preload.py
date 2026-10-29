"""Rules and helpers for init-time Triton pre-loads.

A pre-load only covers serving when its launches produce the *same* JIT cache
keys the serving path will produce. Triton folds three things into a key that
are easy to miss when a pre-load is written "for the biggest shape":

1. An integer argument that equals 1 becomes a constexpr. ``bs=1`` -- a single
   request, the most common batch there is -- is therefore a different kernel
   from ``bs=2``. Warming only the largest batch misses it, and the first
   single-request batch then device-loads mid-serving. That is exactly how the
   DSpark accept pre-load failed its first two runs.
2. Whether an integer argument is divisible by 16 is a key attribute too, so a
   row/token count under 16 is not the kernel a larger count produces.
3. Pointer dtype is part of the key, and so is the "D" (divisible-by-16)
   alignment attribute of each pointer. Keep the dtypes identical to the
   serving call site, and hand the kernel storage-based tensors rather than
   tensors with odd element offsets.

So: replay every batch bucket the scheduler can form (``batch_buckets``) and
every token-count bucket the model can form (``token_buckets``), with the
serving dtypes. The helpers below exist so that rule is written down once
instead of being re-derived -- and re-forgotten -- at each pre-load site.
"""

from __future__ import annotations

from typing import Optional, Tuple


def batch_buckets(cap: int) -> range:
    """Batch sizes a scheduler with this cap can form: ``1..cap``.

    ``cap`` is ``max_running_requests``. Rule 1 in the module docstring is why
    this starts at 1 rather than at the first "interesting" batch size.
    """
    return range(1, max(1, int(cap)) + 1)


def token_buckets(*counts: int) -> Tuple[int, ...]:
    """Token (row) counts that cover Triton's integer-specialisation classes.

    Triton keys an integer argument three ways: ``== 1`` becomes a constexpr,
    otherwise it records whether the value is divisible by 16. So a pre-load
    needs one count per class it expects: ``1`` for the single-row case, an odd
    count above 1 for the generic class, and a multiple of 16. Callers pass
    exactly the classes their serving path can produce; duplicates collapse and
    order is preserved.
    """
    buckets = tuple(dict.fromkeys(int(x) for x in counts))
    assert buckets, "token_buckets needs at least one count"
    return buckets


def loaded_specializations(jit_fn) -> Optional[int]:
    """Number of specialisations of ``jit_fn`` loaded on the current device.

    Diagnostics only: lets a pre-load log how many kernels it actually warmed
    (``None`` when the Triton internals do not expose the table, in which case
    the caller should skip the count rather than fail).
    """
    try:
        import torch

        device_caches = getattr(jit_fn, "device_caches", None)
        if not device_caches:
            return None
        per_device = device_caches.get(torch.cuda.current_device())
        if not per_device:
            return None
        table = per_device[0]
        return len(table)
    except Exception:
        return None
