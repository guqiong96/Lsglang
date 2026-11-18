"""The lk_moe pool registry: who counts as a member, and the count the region uses.

The 2026-11-17 machine test caught a real boot failure whose whole cause was this
bookkeeping: the Qwen4-Exp MTP draft is stamped with arch ``...ForCausalLMMTP``
(not ``...NextN``), so ``_record_draft_pool_members`` recorded nothing, the registry
was short by exactly the draft layer, and the first pool of the draft threw
``pool index 48 outside [0,48) of the counted members``. Python-side, GPU-free:
this file pins that down in milliseconds.

    python3 -m pytest test/srt/test_lk_draft_pool_registry.py -v
"""

import unittest

from sglang.srt.utils import lk_moe_env
from sglang.test.test_utils import CustomTestCase


class _Hf:
    """The few hf_config attributes the draft-count derivation reads."""

    def __init__(self, arch, nextn=None, layers=None):
        self.architectures = [arch]
        if nextn is not None:
            self.num_nextn_predict_layers = nextn
        if layers is not None:
            self.num_hidden_layers = layers


class _ModelConfig:
    def __init__(self, arch, nextn=None, layers=None):
        self.hf_config = _Hf(arch, nextn, layers)


class TestDraftPoolCensus(CustomTestCase):
    def setUp(self):
        # reset the module's process-wide registry state (it is process state by
        # design: one dense index space per process)
        lk_moe_env._pool_main_layers.clear()
        lk_moe_env._pool_draft_layers.clear()
        lk_moe_env._pool_ids.clear()
        lk_moe_env._spec_draft_moe_layers = 0
        self._env_backup = {
            k: __import__("os").environ.pop(k, None)
            for k in ("LK_GPU_RESIDENT_MOE_LAYERS",
                      "LK_GPU_RESIDENT_MOE_LAYERS_MTP",
                      "LK_GPU_RESIDENT_MOE_LAYERS_SPEC",
                      "LK_GPU_RESIDENT_MOE_LAYERS_DSPARK",
                      "LK_MOE_HYBRID_ENABLED")
        }
        # The registry only exists inside the hybrid path, and
        # `is_lk_moe_gpu_resident_layer` answers "resident" for EVERY layer when
        # the master switch is off (so nothing is ever a pool member). Leaving it
        # off made every count in this file read as "main only" -- the suite was
        # 4-red on the day it landed (2026-11-17). A gate has to turn on the thing
        # it measures.
        __import__("os").environ["LK_MOE_HYBRID_ENABLED"] = "1"

    def tearDown(self):
        import os
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_mtp_arch_records_one_draft_layer(self):
        from sglang.srt.managers.tp_worker import _record_draft_pool_members

        _record_draft_pool_members(
            _ModelConfig("Qwen4ExpForCausalLMMTP", nextn=None, layers=1))
        self.assertEqual(lk_moe_env._spec_draft_moe_layers, 1,
                         "the MTP draft arch must be counted, or the region is "
                         "one cell short and the first draft pool cannot register")

    def test_nextn_arch_counts_its_num_nextn_layers(self):
        from sglang.srt.managers.tp_worker import _record_draft_pool_members

        _record_draft_pool_members(_ModelConfig("SomeForCausalLMNextN", nextn=3))
        self.assertEqual(lk_moe_env._spec_draft_moe_layers, 3)

    def test_record_is_idempotent_and_monotone(self):
        from sglang.srt.managers.tp_worker import _record_draft_pool_members

        _record_draft_pool_members(_ModelConfig("Qwen4ExpForCausalLMMTP", layers=1))
        _record_draft_pool_members(_ModelConfig("Qwen4ExpForCausalLMMTP", layers=1))
        self.assertEqual(lk_moe_env._spec_draft_moe_layers, 1)
        # never goes back down: the draft config is parsed twice in one boot
        _record_draft_pool_members(_ModelConfig("SomeForCausalLMNextN", nextn=2))
        self.assertEqual(lk_moe_env._spec_draft_moe_layers, 2)

    def test_member_count_is_main_plus_declared_draft(self):
        for name in ("m0.mlp.experts", "m1.mlp.experts"):
            lk_moe_env.regist_lk_pool_member(name, "main")
        self.assertEqual(lk_moe_env.get_pool_member_layers(), 2)

        # a draft layer registers LATER (it is constructed after the target) --
        # the count must not race it: the declared number carries it.
        from sglang.srt.managers.tp_worker import _record_draft_pool_members
        _record_draft_pool_members(_ModelConfig("Qwen4ExpForCausalLMMTP", layers=1))
        self.assertEqual(lk_moe_env.get_pool_member_layers(), 3)

        # and the id handed out to the draft layer is inside [0, count)
        lk_moe_env.regist_lk_pool_member("mtp.layers.0.mlp.experts", "mtp")
        lid = lk_moe_env.lk_pool_id("mtp.layers.0.mlp.experts")
        self.assertEqual(lid, 2)
        self.assertLess(lid, lk_moe_env.get_pool_member_layers())

    def test_registration_is_idempotent_per_layer_name(self):
        lk_moe_env.regist_lk_pool_member("a.mlp.experts", "main")
        lk_moe_env.regist_lk_pool_member("a.mlp.experts", "main")
        self.assertEqual(len(lk_moe_env._pool_main_layers), 1)
        self.assertEqual(lk_moe_env.lk_pool_id("a.mlp.experts"), 0)
        # a second, different layer still gets the next dense id
        lk_moe_env.regist_lk_pool_member("b.mlp.experts", "main")
        self.assertEqual(lk_moe_env.lk_pool_id("b.mlp.experts"), 1)

    def test_resident_list_excludes_listed_layers(self):
        """`LK_GPU_RESIDENT_MOE_LAYERS_*` lists layers kept RESIDENT in VRAM.

        Listed layers keep their weights on the GPU and therefore do NOT join the
        expert pool. lsglang numbers mtp/dspark from layer 0 and `_SPEC` outranks
        `_MTP`/`_DSPARK`, so `LK_GPU_RESIDENT_MOE_LAYERS_SPEC=0` is the knob that
        pins the draft. (The earlier wording here, "disabled list", described the
        same effect from the pool's side and misled a reader once -- 2026-11-17.)
        """
        import os

        os.environ["LK_GPU_RESIDENT_MOE_LAYERS_MTP"] = "0"
        # the disabled list holds ints; the callers pass int layer ids
        self.assertTrue(lk_moe_env.is_lk_moe_gpu_resident_layer(0, "mtp"))
        self.assertFalse(lk_moe_env.is_lk_moe_gpu_resident_layer(1, "mtp"))
        os.environ.pop("LK_GPU_RESIDENT_MOE_LAYERS_MTP")
        self.assertFalse(lk_moe_env.is_lk_moe_gpu_resident_layer(0, "mtp"))

    def test_main_resident_list_never_reaches_a_draft(self):
        """The production list is `0-1,33-34`, and a draft restarts its ids at 0.

        A draft that inherited the MAIN variable would treat its own layer 0 as
        resident, never register as a pool member, and the region would still
        reserve a cell for it. This is the invariant the 2026-11-17 registry work
        exists for, so it is asserted directly -- `_MTP` alone passes on both the
        broken and the fixed code and therefore proves nothing.
        """
        import os

        os.environ["LK_GPU_RESIDENT_MOE_LAYERS"] = "0-1,33-34"
        res = lk_moe_env.is_lk_moe_gpu_resident_layer
        self.assertTrue(res(0, "main"))
        self.assertTrue(res(33, "main"))
        self.assertFalse(res(2, "main"))
        # the draft's own layer 0 is a pool member even though main's 0 is resident
        self.assertFalse(res(0, "mtp"))
        self.assertFalse(res(0, "dspark"))

    def test_a_resident_draft_reserves_no_cell(self):
        """Count == registrations: a draft pinned resident must not be counted.

        The region is ONE process-wide set of cells, so a cell reserved for a
        layer that never registers is bytes nobody can use -- and the reverse (a
        registering layer that was not counted) is a hard boot failure.
        """
        import os
        from sglang.srt.managers.tp_worker import _record_draft_pool_members

        for name in ("m0.mlp.experts", "m1.mlp.experts"):
            lk_moe_env.regist_lk_pool_member(name, "main")
        os.environ["LK_GPU_RESIDENT_MOE_LAYERS_SPEC"] = "0"
        _record_draft_pool_members(_ModelConfig("Qwen4ExpForCausalLMMTP", layers=1))
        self.assertEqual(lk_moe_env._spec_draft_moe_layers, 0)
        self.assertEqual(lk_moe_env.get_pool_member_layers(), 2)

        # and the joining draft (list removed) is counted exactly once
        os.environ.pop("LK_GPU_RESIDENT_MOE_LAYERS_SPEC")
        _record_draft_pool_members(_ModelConfig("Qwen4ExpForCausalLMMTP", layers=1))
        self.assertEqual(lk_moe_env._spec_draft_moe_layers, 1)
        self.assertEqual(lk_moe_env.get_pool_member_layers(), 3)


if __name__ == "__main__":
    unittest.main()
