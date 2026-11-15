"""The eager draft path has to be device-loaded at init, and it has to be COVERED.

Two things make this worth a test rather than a boot:

  * the first launch of these kernels lands in a served step — 510 ms measured
    against 0.03 ms warm — and a cold ``cuModuleLoadData`` inside a TP-synchronized
    step wedges the group (both schedulers at 100% CPU inside libcuda, GPUs idle);
  * coverage is only as good as its agreement with the served call site. Triton keys
    a binary on constexprs, int classes, dtypes and strides, so a warm-up that derives
    the window differently, or warms bs=1 only, compiles nothing the server will use
    and LOOKS covered — which is worse than not warming at all.

Part one tests the pure functions that decide the coverage set. Part two is the
structural guard that the served path and the warm-up go through the same launcher,
because a helper test passes no matter what the call sites do. Each part carries a
control that must fail.
"""

import ast
import inspect
import unittest
from unittest import mock

import torch

from sglang.srt.speculative import eagle_worker_common as common


class TestCoverageSet(unittest.TestCase):
    def test_batch_sizes_are_exhaustive_up_to_16(self):
        self.assertEqual(common.draft_prewarm_batch_sizes(2), (1, 2))
        self.assertEqual(common.draft_prewarm_batch_sizes(16), tuple(range(1, 17)))

    def test_control_bs4_is_not_missed(self):
        # bs=4 is the case that broke the {1, 2, 16} shortcut: the derived int
        # bs * window_cols is 16, so bs=4 takes the divisible binary while bs=2 takes
        # the generic one. A shortcut list serves bs=4 with a module nobody loaded.
        self.assertIn(4, common.draft_prewarm_batch_sizes(8))
        self.assertNotIn(4, (1, 2, 16))

    def test_unknown_cap_still_warms_bs1(self):
        self.assertEqual(common.draft_prewarm_batch_sizes(None), (1,))
        self.assertEqual(common.draft_prewarm_batch_sizes(0), (1,))

    def test_batch_sizes_beyond_16_fall_back_to_powers(self):
        got = common.draft_prewarm_batch_sizes(64)
        self.assertIn(32, got)
        self.assertIn(64, got)
        self.assertNotIn(33, got)

    def test_window_cols_follows_plans_window(self):
        # The served path picks num_draft_tokens only when it plans the whole window;
        # window_cols is a constexpr, so the two outcomes are two binaries.
        self.assertEqual(
            common.draft_window_cols(
                topk=1, num_steps=3, num_draft_tokens=4, plans_window=True
            ),
            4,
        )
        self.assertEqual(
            common.draft_window_cols(
                topk=1, num_steps=3, num_draft_tokens=4, plans_window=False
            ),
            3,
        )

    def test_control_window_cols_differs_when_both_are_live(self):
        self.assertNotEqual(
            common.draft_window_cols(
                topk=1, num_steps=3, num_draft_tokens=4, plans_window=True
            ),
            common.draft_window_cols(
                topk=1, num_steps=3, num_draft_tokens=4, plans_window=False
            ),
            "one binary would have to cover both layouts, and it does not",
        )


class TestTrackIntervals(unittest.TestCase):
    def test_no_page_size_means_only_zero(self):
        self.assertEqual(common.draft_track_intervals(None), (0,))

    def test_both_live_values_are_covered(self):
        # The served path passes mamba_track_grid(page_size) for a batch carrying
        # track indices and 0 for one that does not.
        with mock.patch.object(common, "mamba_track_grid", return_value=256):
            self.assertEqual(common.draft_track_intervals(1), (0, 256))

    def test_falls_back_to_server_args_when_the_context_is_absent(self):
        # During engine init the execution config is not entered, so mamba_track_grid
        # raises. Skipping the non-zero value there is what leaves the served call
        # cold, so the same value is derived from the server args instead:
        # lcm(lcm(chunk, page), mamba_track_interval).
        args = mock.Mock(mamba_track_interval=256)
        with (
            mock.patch.object(
                common, "mamba_track_grid", side_effect=RuntimeError("no ctx")
            ),
            mock.patch.object(common, "mamba_cache_chunk_size", return_value=64),
            mock.patch.object(common, "get_server_args", return_value=args),
        ):
            self.assertEqual(common.draft_track_intervals(1), (0, 256))

    def test_control_a_warm_of_only_zero_is_not_what_this_returns(self):
        with mock.patch.object(common, "mamba_track_grid", return_value=256):
            self.assertNotEqual(
                common.draft_track_intervals(1),
                (0,),
                "warming only interval 0 leaves the served call to device-load",
            )


class TestPrewarmIsInertWithoutCuda(unittest.TestCase):
    def test_returns_zero_and_touches_nothing(self):
        # The warm-up must never be able to stop an engine from starting, and a CPU
        # tensor has no Triton cache to fill either.
        self.assertEqual(
            common.prewarm_draft_step_path(
                torch.zeros((4, 8), dtype=torch.int64),
                topk=1,
                num_steps=3,
                num_draft_tokens=4,
                batch_sizes=(1, 2),
                track_intervals=(0,),
            ),
            0,
        )


class TestServedPathUsesTheSameLauncher(unittest.TestCase):
    """A helper test proves nothing about the call sites: this reads the real source."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(inspect.getsource(common))

    @staticmethod
    def _kernel_launches(tree):
        """Calls of the form `name[...](...)` and plain calls of `name(...)`."""
        found = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            func = n.func
            if isinstance(func, ast.Subscript):  # kernel[(grid)](...)
                func = func.value
            if isinstance(func, ast.Name):
                found.append((func.id, n))
        return found

    def test_the_kernel_is_launched_in_exactly_one_place(self):
        launches = [
            n
            for name, n in self._kernel_launches(self.tree)
            if name == "assign_draft_cache_locs_contiguous"
        ]
        self.assertEqual(
            len(launches),
            1,
            f"the kernel is launched from {len(launches)} places; the warm-up covers one",
        )
        launchers = [
            n
            for name, n in self._kernel_launches(self.tree)
            if name == "launch_assign_draft_cache_locs"
        ]
        self.assertGreaterEqual(
            len(launchers), 1, "the shared launcher is not called at all"
        )

    @staticmethod
    def _inline_window_assignments(tree):
        """`window_cols = a if cond else b` -- the copy of the formula to catch."""
        bad = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Assign):
                continue
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if "window_cols" in names and isinstance(n.value, ast.IfExp):
                bad.append(n)
        return bad

    def test_the_served_path_does_not_recompute_window_cols_inline(self):
        uses = [
            n
            for name, n in self._kernel_launches(self.tree)
            if name == "draft_window_cols"
        ]
        self.assertEqual(
            len(uses), 1, "the served path should take window_cols from the helper"
        )
        self.assertEqual(
            self._inline_window_assignments(self.tree),
            [],
            "window_cols is derived inline again, so the warm-up can disagree",
        )

    def test_control_an_inline_window_assignment_is_detected(self):
        offender = """
def prepare_for_draft(num_draft_tokens, num_steps, plans_window):
    window_cols = num_draft_tokens if plans_window else num_steps
"""
        self.assertEqual(
            len(self._inline_window_assignments(ast.parse(offender))),
            1,
            "the detector misses the inline form it exists to catch",
        )

    def test_control_a_reinlined_kernel_call_is_detected(self):
        offender = """
def prepare_for_draft(batch, bs, topk, window_cols):
    assign_draft_cache_locs_contiguous[(bs,)](
        batch.req_pool_indices, batch.req_to_token, batch.seq_lens,
        batch.out_cache_loc, 16, topk, window_cols, BS=bs,
    )
"""
        launches = [
            n
            for name, n in self._kernel_launches(ast.parse(offender))
            if name == "assign_draft_cache_locs_contiguous"
        ]
        self.assertEqual(
            len(launches), 1, "the detector misses the inline form it has to catch"
        )
        launchers = [
            name
            for name, _n in self._kernel_launches(ast.parse(offender))
            if name == "launch_assign_draft_cache_locs"
        ]
        self.assertEqual(launchers, [], "the guard would accept a re-inlined call")


class TestDraftGraphMetadataPrewarm(unittest.TestCase):
    """The replay-path kernel: capture does not load it, so init has to."""

    @staticmethod
    def _bare(backends):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseMultiStepDraftBackend,
        )

        obj = QwenSparseMultiStepDraftBackend.__new__(QwenSparseMultiStepDraftBackend)
        obj.attn_backends = backends
        return obj

    def test_no_backends_is_a_no_op(self):
        self.assertEqual(self._bare([]).prewarm_draft_graph_metadata(), 0)

    def test_a_cpu_backend_is_a_no_op(self):
        backend = mock.Mock()
        backend.req_to_token = torch.zeros((4, 8), dtype=torch.int64)  # CPU tensor
        self.assertEqual(self._bare([backend]).prewarm_draft_graph_metadata(), 0)

    def test_control_a_cpu_tensor_really_is_not_cuda(self):
        # If this were True the two tests above would be passing for the wrong reason.
        self.assertFalse(torch.zeros((4, 8)).is_cuda)

    def test_only_captured_decode_buckets_are_warmed(self):
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QwenSparseMultiStepDraftBackend,
        )

        src = inspect.getsource(
            QwenSparseMultiStepDraftBackend.prewarm_draft_graph_metadata
        )
        self.assertIn(
            "if mode == ForwardMode.DECODE",
            src,
            "a batch size with no captured graph cannot be served from the replay path",
        )
        self.assertIn("buffer.clone()", src, "the warm-up must not clobber those tables")


class TestTheWarmupRunsAtInitNotPerStep(unittest.TestCase):
    """Where the call sits is the whole point: per-step would be worse than nothing."""

    @classmethod
    def setUpClass(cls):
        import sglang.srt.speculative.eagle_worker_v2 as mod

        cls.src = inspect.getsource(mod)
        cls.tree = ast.parse(cls.src)

    def _segment(self, name):
        for n in ast.walk(self.tree):
            if isinstance(n, ast.FunctionDef) and n.name == name:
                return ast.get_source_segment(self.src, n) or ""
        return ""

    def test_the_segments_are_real(self):
        # Control for the assertion below: if segment extraction returned "" for
        # every method, `assertNotIn` would pass on nothing at all.
        self.assertIn("def _capture_cuda_graphs", self._segment("_capture_cuda_graphs"))
        self.assertIn("def draft(", self._segment("draft"))

    @staticmethod
    def _reads_page_size_from_server_args(src):
        """The defect this guards: ServerArgs has no page_size, so getattr returns
        None and the track-interval set silently collapses to {0}."""
        return 'getattr(server_args, "page_size"' in src

    def test_page_size_comes_from_the_allocator(self):
        capture = self._segment("_capture_cuda_graphs")
        self.assertIn(
            "token_to_kv_pool_allocator",
            capture,
            "page size must come from the allocator the served path reads",
        )
        self.assertFalse(
            self._reads_page_size_from_server_args(self.src),
            "ServerArgs has no page_size; this returns None and warms only interval 0",
        )

    def test_control_the_server_args_form_is_detected(self):
        offender = 'page_size = getattr(server_args, "page_size", None)'
        self.assertTrue(self._reads_page_size_from_server_args(offender))

    def test_capture_calls_it_and_the_served_step_does_not(self):
        self.assertIn(
            "prewarm_draft_graph_metadata",
            self._segment("_capture_cuda_graphs"),
            "the QSA metadata kernel is never device-loaded at init",
        )
        self.assertNotIn(
            "prewarm_draft_graph_metadata",
            self._segment("draft"),
            "a per-step warm-up would launch a dummy step inside a real one",
        )


if __name__ == "__main__":
    unittest.main()
