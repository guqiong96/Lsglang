"""The unified cache's per-pool token ledger, asserted on CPU after every transition.

The invariant the server checks while idle is, per pool,

    available + evictable + protected + session_held + uncached == total

and on DSV4.1 TP4 (2026-11-15) it once saw the full and the swa counter each exceed
their own total by exactly 1280 tokens. That message says by how much the ledger
drifted and nothing about which transition drifted it, so the useful thing to have is
a harness that runs the real transitions and asserts the invariant after each one:
a sequence that over-counts fails here, on CPU, in a second, instead of on a served
box two weeks later.

Two things this file is careful about, both learned by getting them wrong first:

  * a hybrid pool is TWO pools. The full-attention allocator and the SWA allocator are
    conserved separately (`_check_full_pool` and `_check_swa_pool` compare each counter
    against its own pool), so a single ledger summed over components reports a
    violation that does not exist -- and hides one that does.
  * every transition has to be driven for BOTH components. Driving only FULL leaves the
    SWA ledger behind, which looks exactly like the drift being hunted.

The control arm deliberately breaks the ledger and requires the checker to catch it;
without it, a harness that silently stopped asserting anything would still be green.
"""

import unittest
from array import array

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import CacheTransferPhase
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Ledger:
    """One pool's free-token count; `available_size` here is the ground truth."""

    def __init__(self, total: int):
        self.size = total
        self.page_size = 1
        self.device = torch.device("cpu")
        self._free = total

    def available_size(self) -> int:
        return self._free

    def alloc(self, need: int):
        n = min(int(need), self._free)
        self._free -= n
        return torch.arange(n, dtype=torch.int64)

    def free(self, indices) -> None:
        n = int(indices.numel()) if hasattr(indices, "numel") else int(indices)
        self._free = min(self.size, self._free + n)

    def free_segment(self, indices, start_pos: int = 0) -> None:
        self.free(indices)

    def free_group_(self, indices) -> None:
        self.free(indices)


class _LedgerKVCache:
    """The pool the allocator was built over, reduced to what the tree reads."""

    def __init__(self, total: int):
        self.full_to_swa_index_mapping = torch.zeros(total + 1, dtype=torch.int64)

    def translate_loc_from_full_to_swa(self, kv_indices):
        return self.full_to_swa_index_mapping[kv_indices]


class LedgerAllocator(SWATokenToKVPoolAllocator):
    """The real SWA allocator's type (components type-check it), without device pages.

    SWAComponent asserts it was handed a `SWATokenToKVPoolAllocator`, so the harness
    has to be one; what it must not do is allocate. Every alloc/free moves a ledger by
    exactly the tokens it reports, so the tree's counters can be checked against it.
    """

    def __init__(self, total: int, swa_tokens: int):
        self._size = total
        self._page = 1
        self.device = torch.device("cpu")
        self.full_attn_allocator = _Ledger(total)
        self.swa_attn_allocator = _Ledger(swa_tokens)
        self._swa_req_ring = False
        self._kvcache = _LedgerKVCache(total)

    @property
    def size(self) -> int:
        return self._size

    @property
    def page_size(self) -> int:
        return self._page

    def available_size(self) -> int:
        return self.full_attn_allocator.available_size()

    def alloc(self, need: int):
        idx = self.full_attn_allocator.alloc(need)
        self.swa_attn_allocator.alloc(int(idx.numel()))
        return idx

    def free(self, indices) -> None:
        n = int(indices.numel()) if hasattr(indices, "numel") else int(indices)
        self.full_attn_allocator.free(indices)
        self.swa_attn_allocator.free(n)

    def free_swa_segment(self, indices, *a, **k) -> None:
        self.swa_attn_allocator.free(indices)

    def set_full_to_swa_mapping(self, *a, **k) -> None:
        return None

    def clear_full_to_swa_mapping(self, *a, **k) -> None:
        return None

    def __getattr__(self, name):
        # Only public helpers are stubbed. A missing private attribute must fail
        # loudly: returning a callable for one turns a harness gap into a confusing
        # AttributeError several frames deeper.
        if name.startswith("_"):
            raise AttributeError(name)

        def _noop(*_a, **_k):
            return None

        return _noop


class TestUnifiedPoolLedger(unittest.TestCase):
    window = 4

    def _cache(self, total=64, swa=64):
        alloc = LedgerAllocator(total, swa)
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=alloc,
                page_size=1,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
                sliding_window_size=self.window,
                decoder_swa_bounded_replay=True,
                # The python core owns the counter dicts this file reads (and the
                # production recipe selects it explicitly). Asserted below, so a
                # backend default flip fails loudly instead of skipping the checks.
                tree_core_backend="python",
            )
        )
        self.assertEqual(
            type(cache.tree_core).__name__,
            "UnifiedTreeCore",
            "this harness reads the python core's counter dicts",
        )
        cache.tree_core.set_hicache_enabled()
        return cache, alloc

    def _check(self, cache, alloc, label):
        """Assert both pools independently, the way the server does."""
        drifts = []
        for name, ct, ledger in (
            ("full", ComponentType.FULL, alloc.full_attn_allocator),
            ("swa", ComponentType.SWA, alloc.swa_attn_allocator),
        ):
            ev = cache.tree_core.component_evictable_size_.get(ct, 0)
            pr = cache.tree_core.component_protected_size_.get(ct, 0)
            accounted = ev + pr + ledger.available_size()
            if accounted != ledger.size:
                drifts.append(
                    f"{label}: [{name}] evictable={ev} protected={pr} "
                    f"available={ledger.available_size()} total={ledger.size} "
                    f"drift={accounted - ledger.size:+d}"
                )
        self.assertEqual(drifts, [], "\n".join(drifts))

    def _insert(self, cache, alloc, tokens):
        return cache.insert(
            InsertParams(
                key=RadixKey(token_ids=array("q", tokens), extra_key=None),
                value=alloc.alloc(len(tokens)),
            )
        )

    def test_insert_lock_unlock_evict_conserves(self):
        cache, alloc = self._cache()
        result = self._insert(cache, alloc, range(8))
        self._check(cache, alloc, "insert")
        node = result.last_device_node
        receipt = cache.inc_lock_ref(node)
        self._check(cache, alloc, "inc_lock_ref")
        cache.dec_lock_ref(
            node,
            DecLockRefParams(
                node_id=receipt.node_id,
                component_lock_uuids=dict(receipt.component_lock_uuids),
                component_host_lock_uuids=dict(receipt.component_host_lock_uuids),
                skipped_lock_components=tuple(receipt.skipped_lock_components),
            ),
        )
        self._check(cache, alloc, "dec_lock_ref")
        cache.evict(EvictParams(num_tokens=8))
        self._check(cache, alloc, "evict")

    def test_a_split_insert_conserves(self):
        # A second key sharing a prefix splits the node and hands tokens between
        # parent and child -- the classic place a ledger loses count.
        cache, alloc = self._cache()
        self._insert(cache, alloc, range(8))
        self._insert(cache, alloc, list(range(4)) + list(range(100, 104)))
        self._check(cache, alloc, "split insert")

    def test_evict_for_alloc_conserves(self):
        # The refill path evicts to make room and then allocates; both sides of that
        # have to be credited or the pool reads over-full right afterwards.
        cache, alloc = self._cache()
        self._insert(cache, alloc, range(8))
        cache.evict_for_alloc(EvictParams(num_tokens=4))
        self._check(cache, alloc, "evict_for_alloc")
        cache.evict_for_alloc(EvictParams(num_tokens=4))
        self._check(cache, alloc, "evict_for_alloc (drained)")

    def test_backup_to_host_and_load_back_conserve(self):
        cache, alloc = self._cache()
        node_id = self._insert(cache, alloc, range(8)).last_device_node
        core = cache.tree_core
        node = core.node_by_id(node_id)
        pool_of = {ComponentType.FULL: PoolName.KV, ComponentType.SWA: PoolName.SWA}
        lens = {
            ct: len(node.component_data[ct].value)
            for ct in pool_of
            if node.component_data[ct].value is not None
        }
        host = {ct: torch.arange(n, dtype=torch.int64) for ct, n in lens.items()}
        for ct in lens:
            cache.components[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                [
                    PoolTransfer(
                        name=pool_of[ct], host_indices=host[ct], device_indices=None
                    )
                ],
                cache_actions=[],
            )
        self._check(cache, alloc, "backup_to_host")

        cache.evict(EvictParams(num_tokens=max(lens.values())))
        self._check(cache, alloc, "evict_to_host")

        # Size each load by what the host copy holds: a bounded-replay node keeps
        # less on the SWA side, and allocating for the full length leaves the other
        # ledger behind (a false positive this file used to report).
        ledger_of = {
            ComponentType.FULL: alloc.full_attn_allocator,
            ComponentType.SWA: alloc.swa_attn_allocator,
        }
        lens = {
            ct: len(node.component_data[ct].host_value)
            for ct in lens
            if node.component_data[ct].host_value is not None
        }
        device = {ct: ledger_of[ct].alloc(n) for ct, n in lens.items()}
        for ct in lens:
            cache.components[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.LOAD_BACK,
                [
                    PoolTransfer(
                        name=pool_of[ct],
                        host_indices=host[ct],
                        device_indices=device[ct],
                        nodes_to_load=[node_id],
                    )
                ],
                cache_actions=[],
            )
        self._check(cache, alloc, "load_back")

    def test_control_a_broken_ledger_is_caught(self):
        # If this passes, the assertions above are vacuous.
        cache, alloc = self._cache()
        self._insert(cache, alloc, range(8))
        cache.tree_core.component_evictable_size_[ComponentType.FULL] += 1280
        with self.assertRaises(AssertionError) as ctx:
            self._check(cache, alloc, "deliberate drift")
        self.assertIn("drift=+1280", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
