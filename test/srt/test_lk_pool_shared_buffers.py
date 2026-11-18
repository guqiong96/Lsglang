"""The shared decode buffers must cover the WIDEST pooled layer, not the first.

`sglang/srt/layers/moe/fused_moe_triton/layer.py::_pool_alloc_shared` sizes six
buffers that every pooled layer shares (hbuf / gout / hpart / gout_part /
slot_tab / gathered_*). Each layer reads them back with ITS OWN geometry, so the
only sound rule is "a later layer may raise any dimension, nobody may lower one".

Commit f17fbb94c4 compared `rows` alone. A draft is free to have a different
intermediate size (see lk_moe/tests/test_two_weight_types.py, DRAFT_I), and the
2 x RTX 3090 production pair happens to agree on every dimension, which is
exactly why the hole survived the gate suite: it only bites the day a release
pools two layers of different width. `_old_alloc` below is the rule from
f17fbb94c4's parent, kept verbatim as the must-fail control: this file fails
against it (asserted in `test_the_old_rule_is_really_blind`).

    CUDA_VISIBLE_DEVICES=<free gpu> python -m pytest test/srt/test_lk_pool_shared_buffers.py -q
"""

import unittest

import torch

from sglang.srt.layers.moe.fused_moe_triton import layer as L
from sglang.test.test_utils import CustomTestCase

H, I_NARROW, I_WIDE = 2048, 256, 512
ROWS, GOUT_ROWS, K = 128, 16, 8


def _ngk(hidden, iters=16):
    # same derivation the layer uses; any value works, the point is that the
    # buffers are compared per dimension
    return 16


def _old_alloc(max_real_tokens, rows, gout_rows, H_, I_, NGK):
    """f17fbb94c4's parent: only `rows` is compared. Kept as the control."""
    dev = torch.cuda.current_device()
    if not hasattr(L.FusedMoE, "pool_hbuf"):
        L.FusedMoE._max_real_tokens = max_real_tokens
        L.FusedMoE.pool_hbuf = torch.zeros((rows, 2 * I_), dtype=torch.float32, device=dev)
        L.FusedMoE.pool_gout = torch.zeros((gout_rows, H_), dtype=torch.float32, device=dev)
        L.FusedMoE.pool_slot_tab = torch.full((rows,), -1, dtype=torch.int32, device=dev)
        L.FusedMoE.pool_gathered_ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        L.FusedMoE.pool_gathered_weights = torch.zeros((rows,), dtype=torch.float32, device=dev)
        L.FusedMoE.pool_stream = torch.cuda.Stream()
    if (not hasattr(L.FusedMoE, "pool_hpart")
            or L.FusedMoE.pool_hpart.shape[1] < rows):
        L.FusedMoE.pool_hpart = torch.zeros((NGK, rows, 2 * I_), dtype=torch.float32, device=dev)
        L.FusedMoE.pool_gout_part = torch.zeros((rows, H_), dtype=torch.float32, device=dev)


def _forget():
    for name in ("pool_hbuf", "pool_gout", "pool_hpart", "pool_gout_part",
                 "pool_slot_tab", "pool_gathered_ids", "pool_gathered_weights",
                 "pool_stream"):
        if hasattr(L.FusedMoE, name):
            delattr(L.FusedMoE, name)


@unittest.skipUnless(torch.cuda.is_available(), "needs one GPU")
class TestSharedPoolBuffers(CustomTestCase):
    def tearDown(self):
        _forget()

    def test_a_wider_layer_grows_every_dimension(self):
        _forget()
        L._pool_alloc_shared(2, ROWS, GOUT_ROWS, H, I_NARROW, _ngk(H, I_NARROW))
        L._pool_alloc_shared(2, ROWS, GOUT_ROWS, H, I_WIDE, _ngk(H, I_WIDE))
        F = L.FusedMoE
        self.assertEqual(F.pool_hbuf.shape, (ROWS, 2 * I_WIDE))
        self.assertEqual(F.pool_hpart.shape, (16, ROWS, 2 * I_WIDE))
        self.assertEqual(F.pool_gout_part.shape, (ROWS, H))
        # same rows throughout: only the width changed, which the old rule missed
        self.assertEqual(F.pool_gout.shape, (GOUT_ROWS, H))

    def test_a_narrower_layer_never_shrinks_them(self):
        _forget()
        L._pool_alloc_shared(2, ROWS, GOUT_ROWS, H, I_WIDE, _ngk(H, I_WIDE))
        L._pool_alloc_shared(2, ROWS, GOUT_ROWS, H, I_NARROW, _ngk(H, I_NARROW))
        self.assertEqual(L.FusedMoE.pool_hbuf.shape, (ROWS, 2 * I_WIDE))
        self.assertEqual(L.FusedMoE.pool_hpart.shape, (16, ROWS, 2 * I_WIDE))

    def test_more_rows_grow_the_small_buffers_too(self):
        _forget()
        L._pool_alloc_shared(2, ROWS, GOUT_ROWS, H, I_NARROW, 16)
        L._pool_alloc_shared(4, 2 * ROWS, 2 * GOUT_ROWS, H, I_NARROW, 16)
        F = L.FusedMoE
        self.assertEqual(F.pool_slot_tab.shape, (2 * ROWS,))
        self.assertEqual(F.pool_gout.shape, (2 * GOUT_ROWS, H))
        self.assertEqual(F.pool_gathered_ids.shape, (2 * ROWS,))
        # re-created buffers must keep their sentinel: a stale 0 reads as "slot 0"
        self.assertTrue(bool((F.pool_slot_tab == -1).all()))

    def test_the_old_rule_is_really_blind(self):
        """Must-fail control: the parent's rule leaves the wide layer uncovered.

        Without this, the two assertions above could be passing for a reason
        unrelated to the fix (e.g. the buffers never being reused at all).
        """
        _forget()
        _old_alloc(2, ROWS, GOUT_ROWS, H, I_NARROW, 16)
        _old_alloc(2, ROWS, GOUT_ROWS, H, I_WIDE, 16)
        F = L.FusedMoE
        self.assertEqual(F.pool_hpart.shape, (16, ROWS, 2 * I_NARROW),
                         "the old rule should have kept the narrow plane")
        self.assertLess(F.pool_hpart.shape[2], 2 * I_WIDE,
                        "the old rule must be too small for the wider layer")


if __name__ == "__main__":
    unittest.main()
