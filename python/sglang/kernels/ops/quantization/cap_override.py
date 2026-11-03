"""Test-only compute-capability override.

``LK_TEST_FAKE_CAP=8.9`` makes the *fp8 selection helpers* believe the device is
Ada while the real one is Ampere (or anything else). It exists for one reason:
the Ada-only fp8 routes cannot run on Ampere silicon (e4m3 MMA is an sm_89
instruction), so a 40-series deployment cannot be reproduced on a 30-series box.
What *can* be checked there is the decision itself -- after the Ada rollback the
same box must pick the Ampere kernels for "8.9" too, and those kernels do run.

Only the helpers in this fork consult this; torch and the drivers keep reporting
the truth, so no kernel is ever compiled for an arch the card cannot execute.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_warned = False


def fake_capability() -> Optional[Tuple[int, int]]:
    """(major, minor) from LK_TEST_FAKE_CAP, or None when unset/garbage."""
    global _warned
    raw = os.environ.get("LK_TEST_FAKE_CAP", "").strip()
    if not raw:
        return None
    try:
        major, minor = (int(part) for part in raw.split("."))
    except Exception:
        return None
    if not _warned:
        _warned = True
        logger.warning(
            "[test] LK_TEST_FAKE_CAP=%s: fp8 kernel selection will believe this "
            "device is %s while it really is %s. Test hook -- never deploy.",
            raw,
            (major, minor),
            torch.cuda.get_device_capability() if torch.cuda.is_available() else "cpu",
        )
    return (major, minor)


def device_capability() -> Tuple[int, int]:
    """Capability the fp8 selectors should use (override first)."""
    fake = fake_capability()
    if fake is not None:
        return fake
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()
