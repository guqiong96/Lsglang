"""Super-chunk token cuts must never straddle two requests.

The scheduler sizes prefill batches to CAP=LK_GPU_PREFILL_SUB_M, so one EXTEND
forward can pack several requests. Cutting the token axis purely at
multiples of inner_size puts a cut inside request A's rows while the child
also covers request B's head; that child view misdescribes its rows and the
attention kernels trap (Xid 43 on SM120, scheduler SIGABRT; repro: 2x ~33K
concurrent /generate with SUB_M >= 8192). The bounds builder therefore cuts
at {k*inner_size} UNION {sequence cum-boundaries}: every child lies inside
exactly one sequence (short remainder children are fine).

Pure-python: no GPU, CPU tensors only.
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.super_chunk_backend import (
    _super_chunk_unsupported_reason,
    build_super_chunk_children,
    compute_super_chunk_token_bounds,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_fb(seq_lens, prefix_lens=None, with_seq_lens=True, total=None):
    """Minimal ForwardBatch stand-in for the bounds/child builders."""
    if total is None:
        total = int(sum(seq_lens))
    if prefix_lens is None:
        prefix_lens = [0] * len(seq_lens)

    class _Mode:
        def is_extend(self):
            return True

        def is_mixed(self):
            return False

    fb = types.SimpleNamespace()
    fb.forward_mode = _Mode()
    fb.spec_info = None
    fb.out_cache_loc_dsv4 = None
    fb.global_num_tokens_cpu = None
    fb.extend_num_tokens = total
    fb.extend_seq_lens_cpu = list(seq_lens) if with_seq_lens else None
    fb.extend_prefix_lens_cpu = list(prefix_lens)
    fb.input_ids = torch.arange(total, dtype=torch.int64)
    fb.positions = torch.arange(total, dtype=torch.int64)
    fb.out_cache_loc = torch.arange(total, dtype=torch.int64)
    fb.out_cache_loc_virtual = None
    fb.out_cache_loc_dsv4 = None
    fb.req_pool_indices = torch.arange(len(seq_lens), dtype=torch.int64) + 100
    fb.seq_lens = torch.tensor(
        [p + s for p, s in zip(prefix_lens, seq_lens)], dtype=torch.int64
    )
    return fb


def _old_grid_bounds(total, inner_size):
    """The pre-fix algorithm: cuts only at multiples of inner_size."""
    bounds = []
    lo = 0
    while lo < total:
        hi = min(lo + inner_size, total)
        bounds.append((lo, hi))
        lo = hi
    return bounds


def _straddles(bounds, seq_lens):
    """First child that spans a sequence boundary, or None."""
    cum = [0]
    for s in seq_lens:
        cum.append(cum[-1] + s)
    for lo, hi in bounds:
        for b in cum[1:-1]:
            if lo < b < hi:
                return (lo, hi, b)
    return None


class TestSuperChunkBounds(CustomTestCase):
    def _assert_cover(self, bounds, total):
        self.assertTrue(bounds)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], total)
        for (lo0, hi0), (lo1, hi1) in zip(bounds, bounds[1:]):
            self.assertEqual(hi0, lo1)  # contiguous, ordered
            self.assertLess(lo0, hi0)  # never empty

    def test_two_seq_children_never_straddle(self):
        # The crash shape: 2-seq batch [33010, 5000], inner 1024. The old
        # grid-only cut has a 1024-multiple seam inside seq B's rows
        # (33010 is not a multiple of 1024 -> straddling child). New cuts
        # must add the boundary and keep every child in one sequence.
        seq_lens = [33010, 5000]
        total = sum(seq_lens)
        old = _old_grid_bounds(total, 1024)
        self.assertIsNotNone(
            _straddles(old, seq_lens)
        )  # regression anchor: old code was broken
        bounds = compute_super_chunk_token_bounds(_fake_fb(seq_lens), 1024)
        self._assert_cover(bounds, total)
        self.assertIsNone(_straddles(bounds, seq_lens))
        self.assertIn(33010, [lo for lo, _ in bounds])
        # grid multiples that stay inside one sequence are still cut points
        self.assertIn(32768, [lo for lo, _ in bounds])
        # seq A tail child: 32768 -> 33010 (242 rows, single seq)
        self.assertIn((32768, 33010), bounds)
        # seq B: first child 33010 -> next grid multiple 33792, ..., and the
        # total remainder tail 37888 -> 38010.
        self.assertIn((33010, 33792), bounds)
        self.assertIn((37888, 38010), bounds)

    def test_small_multi_seq_exact_cut(self):
        # 900+130 with inner 1024: boundary 900 must split the first 1024
        # window; the seam at 1024 stays inside seq B.
        bounds = compute_super_chunk_token_bounds(_fake_fb([900, 130]), 1024)
        self.assertEqual(bounds, [(0, 900), (900, 1024), (1024, 1030)])

    def test_single_seq_grid_cut_unchanged(self):
        # Single sequence: no interior boundaries, so the plain grid cut (the
        # empirically clean 32x1024 + 242 path) is preserved bit-for-bit.
        bounds = compute_super_chunk_token_bounds(_fake_fb([33010]), 1024)
        self.assertEqual(bounds, _old_grid_bounds(33010, 1024))
        self.assertEqual([hi - lo for lo, hi in bounds[:32]], [1024] * 32)
        self.assertEqual(bounds[32], (32768, 33010))

    def test_missing_seq_lens_falls_back_to_grid(self):
        # No extend_seq_lens_cpu: do not allocate/crash, keep old grid cuts
        # (the runtime gate is what disables the wrapper for such batches).
        fb = _fake_fb([33010, 5000], with_seq_lens=False)
        self.assertEqual(
            compute_super_chunk_token_bounds(fb, 1024),
            _old_grid_bounds(fb.extend_num_tokens, 1024),
        )

    def test_boundary_on_grid_multiple_has_no_empty_child(self):
        # 2048+5000: boundary 2048 coincides with a grid cut; the union must
        # deduplicate (no zero-length child).
        bounds = compute_super_chunk_token_bounds(_fake_fb([2048, 5000]), 1024)
        self._assert_cover(bounds, 7048)
        self.assertIsNone(_straddles(bounds, [2048, 5000]))
        self.assertEqual(bounds.count((2048, 3072)), 1)

    def test_children_views_lie_in_one_sequence(self):
        # Child views of the new bounds: batch_size==1 everywhere, and the
        # per-sequence (prefix, prefix+extend) windows tile the parent window
        # exactly once -- including mid-sequence slices with nonzero prefix
        # (the (900,1024) child of seq B starts at prefix 0 length 124; the
        # (1024,1030) child is mid-sequence with prefix 124).
        seq_lens, prefix_lens = [900, 130], [64, 0]
        fb = _fake_fb(seq_lens, prefix_lens)
        bounds = compute_super_chunk_token_bounds(fb, 1024)
        children = build_super_chunk_children(fb, bounds)
        self.assertEqual(len(children), 3)
        expected = [
            (1, [900], [64], [100]),  # seq A whole
            (1, [124], [0], [101]),   # seq B head
            (1, [6], [124], [101]),   # seq B mid-sequence tail
        ]
        for child, (bs, ext, pref, pool) in zip(children, expected):
            self.assertEqual(child.batch_size, bs)
            self.assertEqual(child.extend_seq_lens_cpu, ext)
            self.assertEqual(child.extend_prefix_lens_cpu, pref)
            self.assertEqual(child.req_pool_indices.tolist(), pool)

    def test_children_views_two_long_seqs_chain_prefixes(self):
        seq_lens, prefix_lens = [33010, 5000], [10, 20]
        fb = _fake_fb(seq_lens, prefix_lens)
        bounds = compute_super_chunk_token_bounds(fb, 1024)
        children = build_super_chunk_children(fb, bounds)
        covered = {0: [], 1: []}
        for (lo, _), child in zip(bounds, children):
            self.assertEqual(child.batch_size, 1)
            self.assertEqual(child.extend_num_tokens, sum(child.extend_seq_lens_cpu))
            seq = child.req_pool_indices.tolist()[0] - 100
            p0 = child.extend_prefix_lens_cpu[0]
            covered[seq].append((p0, p0 + child.extend_seq_lens_cpu[0]))
        # each sequence's rows are tiled exactly once, in order, from its
        # parent prefix to parent prefix + len
        for seq, (plen, slen) in enumerate(zip(prefix_lens, seq_lens)):
            wins = sorted(covered[seq])
            self.assertEqual(wins[0][0], plen)
            self.assertEqual(wins[-1][1], plen + slen)
            for (_, e0), (s1, _) in zip(wins, wins[1:]):
                self.assertEqual(e0, s1)


    def test_gate_disables_straddling_batches(self):
        # Defense-in-depth gate: a multi-seq batch whose token axis does not
        # agree with extend_seq_lens_cpu (so the builder cannot place the
        # cuts) must disable the wrapper instead of running a straddling
        # child. A coherent multi-seq batch must stay enabled.
        broken = _fake_fb([33010, 5000], total=37000)
        self.assertEqual(
            _super_chunk_unsupported_reason(broken, 1024),
            "child would straddle a sequence boundary",
        )
        good = _fake_fb([33010, 5000])
        self.assertIsNone(_super_chunk_unsupported_reason(good, 1024))


if __name__ == "__main__":
    unittest.main()
