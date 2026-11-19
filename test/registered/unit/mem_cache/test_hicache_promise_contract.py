# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""HiCache load-back: what the match promises must be what the loader commits.

GitHub issue 40. On the DeepSeek-V4.1 recipe a request whose FULL prefix had been
written to host and then partially evicted killed the scheduler with

    HiCache load-back must commit all promised FULL tokens or none:
    promised=105216 loaded=101632

The two numbers are two walks over ONE node path:

* the promise, ``req.host_hit_length``, summed every node with a ``host_value``
  between ``best_match_node`` and ``last_device_node``;
* the commit is the contiguous run of device-evicted nodes below the first
  device-resident node (the FULL LOAD_BACK spec).

So a node that is host-backed AND still device-resident -- a write-through copy
whose device value was never reclaimed, here reached through host-only auxiliary
(SWA) state -- is promised but never transferred, and the all-or-none guard
aborts the request. The 3,584-token delta in the report was exactly such a
prefix (14 pages of 256). The fixture below reproduces the state on the CPU tree
core, with no SWA replay feature involved: the divergence needs only a second
component whose device values were evicted while FULL stayed resident.
"""

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import CacheTransferPhase
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_rust_tree_core_integration import (
    _internal_swa_write_back_case,
    _key,
    _mock_swa_write_back_io,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

FULL = ComponentType.FULL
SWA = ComponentType.SWA
SWA_WINDOW = 8


def _host_backed_prefix(backend):
    """The issue-40 tree: every node is host-backed for FULL; the deepest two are
    device-evicted; the head node is still resident. SWA was evicted first, so the
    component-consensus device prefix is 0 while FULL is resident below it.
    """
    case = _internal_swa_write_back_case(
        backend, page_size=4, window=SWA_WINDOW, prefix_nodes=2
    )
    _mock_swa_write_back_io(case, "success")
    core, cache = case.core, case.cache
    cache.dec_lock_ref(case.nodes[-1], case.leaf_lock)
    head = case.nodes[0]
    device_value, _ = core.build_backup_spec(head)
    core.mark_write_through_pending([head], ack_id=head)
    core.commit_backup(
        head,
        torch.arange(900, 900 + len(device_value), dtype=torch.int64),
        {FULL: []},
    )
    core.finish_write_through([head], head)
    total = case.sizes[-1]
    cache.evict(EvictParams(num_tokens=0, swa_num_tokens=total - SWA_WINDOW))
    cache.evict(EvictParams(num_tokens=total - SWA_WINDOW, swa_num_tokens=0))
    return case, total


def _promise_and_commit(case, total):
    matched = case.cache.match_prefix(MatchPrefixParams(key=_key(list(range(total)))))
    spec = case.core.build_hicache_transfers(
        FULL, matched.best_match_node, CacheTransferPhase.LOAD_BACK
    )[0]
    return matched, len(spec.host_indices)


class TestHiCacheLoadBackPromise(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _assert_promise_equals_commit(self, backend):
        case, total = _host_backed_prefix(backend)
        matched, commit = _promise_and_commit(case, total)
        self.assertGreater(commit, 0, "the fixture must leave real H2D work")
        self.assertLess(
            commit,
            matched.full_kv_hit_length,
            "some FULL has to stay resident, or the walks cannot diverge",
        )
        self.assertEqual(
            matched.host_hit_length,
            commit,
            "the guard compares these two: promising more than the FULL transfer "
            "spec commits aborts the request",
        )

    def test_promise_equals_commit_on_the_python_tree_core(self):
        self._assert_promise_equals_commit("python")

    def test_promise_equals_commit_on_the_rust_tree_core(self):
        self._assert_promise_equals_commit("rust")

    def test_control_the_fixture_state_really_diverges(self):
        """Counting every host-backed node -- what the promise side used to do --
        has to report MORE than the transfer spec on this state. Without this,
        the equality above could be passing on a tree that never diverges.
        """
        case, total = _host_backed_prefix("python")
        matched, commit = _promise_and_commit(case, total)
        node = case.core.node_by_id(matched.best_match_node)
        stop = case.core.node_by_id(matched.last_device_node)
        every_host_copy = 0
        resident_and_backed = 0
        while node is not stop and node.parent is not None:
            data = node.component_data[FULL]
            if data.host_value is not None:
                every_host_copy += len(data.host_value)
                if data.value is not None:
                    resident_and_backed += len(data.host_value)
            node = node.parent
        self.assertGreater(resident_and_backed, 0)
        self.assertEqual(every_host_copy - commit, resident_and_backed)
        self.assertGreater(every_host_copy, commit)

    def test_control_no_auxiliary_eviction_keeps_the_two_walks_equal(self):
        """Same tree with SWA left on device: nothing is stranded behind host-only
        auxiliary state, so the promise and the commit agree trivially."""
        case = _internal_swa_write_back_case(
            "python", page_size=4, window=SWA_WINDOW, prefix_nodes=2
        )
        _mock_swa_write_back_io(case, "success")
        total = case.sizes[-1]
        matched, commit = _promise_and_commit(case, total)
        self.assertEqual(commit, 0)
        self.assertEqual(matched.host_hit_length, commit)


if __name__ == "__main__":
    unittest.main()
