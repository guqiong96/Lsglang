"""The pool-ledger trace: one line per counter mutation, under one env flag.

Why this exists: the idle pool-leak invariant reports the SIZE of a drift
(`available + evictable + protected + session_held + uncached == total`) but not which
transition caused it, and twelve places across the tree components add to and subtract
from those two dicts. When it fired on DSV4.1 TP4 (2026-11-15, +1280 in both the full
and the swa counter) there was no way to tell which one, so the instrument is the
prerequisite for the fix.

The dict is wrapped rather than the call sites instrumented, because `d[ct] += x` is
the only way those counters move -- a call-site trace would rot with the next edit.
"""

import inspect
import logging
import os
import unittest
from unittest import mock

from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import _TracedSizes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

logger = logging.getLogger("sglang.srt.mem_cache.unified_cache.unified_tree_core")


class TestTracedSizes(unittest.TestCase):
    def _enable(self, value: str):
        return mock.patch.dict(os.environ, {"SGLANG_DEBUG_POOL_LEDGER": value})

    def test_a_mutation_is_reported_with_delta_and_caller(self):
        d = _TracedSizes("evictable", {ComponentType.FULL: 0})
        with self._enable("1"), self.assertLogs(logger, level="INFO") as cm:
            d[ComponentType.FULL] += 8
        self.assertEqual(len(cm.records), 1, cm.output)
        msg = cm.records[0].getMessage()
        self.assertIn("evictable[FULL]: 0 -> 8 (+8)", msg)
        # The caller is what makes the trace actionable: it names the transition.
        self.assertIn("test_pool_ledger_trace.py", msg)
        self.assertIn("test_a_mutation_is_reported_with_delta_and_caller", msg)

    def test_control_nothing_is_reported_without_the_flag(self):
        d = _TracedSizes("evictable", {ComponentType.FULL: 0})
        with self._enable("0"):
            with self.assertRaises(AssertionError):
                with self.assertLogs(logger, level="INFO"):
                    d[ComponentType.FULL] += 8
        # ...and the mutation still happened, i.e. the flag gates logging only.
        self.assertEqual(d[ComponentType.FULL], 8)

    def test_each_ledger_keeps_its_own_name(self):
        ev = _TracedSizes("evictable", {ComponentType.SWA: 0})
        pr = _TracedSizes("protected", {ComponentType.SWA: 0})
        with self._enable("1"), self.assertLogs(logger, level="INFO") as cm:
            ev[ComponentType.SWA] += 4
            pr[ComponentType.SWA] += 6
        joined = "\n".join(r.getMessage() for r in cm.records)
        self.assertIn("evictable[SWA]", joined)
        self.assertIn("protected[SWA]", joined)

    def test_it_still_behaves_like_a_dict(self):
        d = _TracedSizes("evictable", {ComponentType.FULL: 3, ComponentType.SWA: 4})
        self.assertEqual(d.get(ComponentType.FULL), 3)
        self.assertEqual(sum(d.values()), 7)
        self.assertEqual(sorted(d), sorted([ComponentType.FULL, ComponentType.SWA]))
        self.assertEqual(dict(d)[ComponentType.SWA], 4)
        self.assertEqual(d.get(ComponentType.MAMBA, 0), 0)

    def test_the_tree_core_uses_it_for_both_ledgers(self):
        # The wrapper only covers what is wrapped: if the core ever goes back to plain
        # dicts, the trace silently reports nothing while looking healthy.
        src = inspect.getsource(inspect.getmodule(_TracedSizes))
        self.assertIn('_TracedSizes(\n            "evictable"', src)
        self.assertIn('_TracedSizes(\n            "protected"', src)

    def test_control_the_source_guard_can_fail(self):
        offender = "self.component_evictable_size_ = {ct: 0 for ct in types}"
        self.assertNotIn('_TracedSizes(\n            "evictable"', offender)


if __name__ == "__main__":
    unittest.main()
