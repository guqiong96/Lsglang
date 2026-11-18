"""The prefill (engine x precision) decision table, as an executable spec.

``LK_GPU_PREFILL_QUANT`` is the only precision knob and the README documents the
same matrix in prose; this test is the machine-checkable half so a table edit
cannot drift away from the docs unnoticed. An unsupported combination raises at
startup rather than degrading, so every rejection is asserted here too.

    python3 -m pytest test/srt/test_lk_prefill_quant.py -v
"""

import os
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from sglang.srt.utils.lk_moe_env import (
    _LK_PREFILL_QUANT_OK,
    get_gpu_prefill_quant,
    is_prefill_w4a4,
)
from sglang.test.test_utils import CustomTestCase

NV, MX = "nvfp4", "mxfp4"

# The resolver reads the device family from the CUDA capability, not from env.
CAP = {"sm120": (12, 0), "other": (8, 6)}


@contextmanager
def quant_env(quant=None, engine="cutlass", arch="sm120", prefill="1024"):
    # quant=None leaves the variable unset, which is the default-off state.
    env = {
        "LK_MOE_HYBRID_ENABLED": "1",
        "LK_GPU_PREFILL_MIN_TOKENS": prefill,
        "LK_GPU_PREFILL_ENGINE": engine,
    }
    if quant is not None:
        env["LK_GPU_PREFILL_QUANT"] = quant
    with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
        torch.cuda, "get_device_capability", return_value=CAP[arch]
    ):
        yield


class TestPrefillQuantTable(CustomTestCase):
    def test_table_grammar(self):
        # Only what the table carries may resolve, so the key shape is itself
        # part of the contract.
        self.assertEqual({q for _, _, q in _LK_PREFILL_QUANT_OK}, {"w4a4", "w4a16"})
        self.assertEqual({a for a, _, _ in _LK_PREFILL_QUANT_OK}, set(CAP))
        self.assertEqual(
            {e for _, e, _ in _LK_PREFILL_QUANT_OK},
            {"cutlass", "triton", "flashinfer"},
        )

    def test_sm120_accepts_the_documented_cells(self):
        for engine, quant in (
            ("cutlass", "w4a4"),
            ("cutlass", "w4a16"),
            ("triton", "w4a16"),
            ("flashinfer", "w4a4"),
            ("flashinfer", "w4a16"),
        ):
            with quant_env(quant, engine, "sm120"):
                self.assertEqual(get_gpu_prefill_quant(), quant)

    def test_sm120_rejects_w4a4_without_block_scaled_kernels(self):
        with quant_env("w4a4", "triton", "sm120"), self.assertRaises(ValueError):
            get_gpu_prefill_quant()

    def test_below_sm120_keeps_w4a16(self):
        for engine in ("cutlass", "triton"):
            with quant_env("w4a16", engine, "other"):
                self.assertEqual(get_gpu_prefill_quant(), "w4a16")
            with quant_env("w4a4", engine, "other"), self.assertRaises(ValueError):
                get_gpu_prefill_quant()

    def test_flashinfer_below_sm120_is_a_configuration_error(self):
        # Its kernels are SM12x-only. Running w4a16 there would hide the fact
        # that the requested engine is not the one that runs.
        for quant in (None, "w4a16", "w4a4"):
            with quant_env(quant, "flashinfer", "other"), self.assertRaises(ValueError):
                get_gpu_prefill_quant()

    def test_unset_never_picks_w4a4(self):
        # w4a4 is opt-in: the default must not pick it even where it is legal.
        for engine in ("cutlass", "flashinfer"):
            with quant_env(None, engine, "sm120"):
                self.assertEqual(get_gpu_prefill_quant(), "w4a16")

    def test_unset_quant_is_w4a16(self):
        os.environ.pop("LK_GPU_PREFILL_QUANT", None)
        with quant_env(engine="cutlass", arch="sm120"):
            self.assertEqual(get_gpu_prefill_quant(), "w4a16")
        with quant_env("w4a16", "cutlass", "sm120"):
            self.assertEqual(get_gpu_prefill_quant(), "w4a16")

    def test_quant_is_case_insensitive(self):
        with quant_env("W4A4", "cutlass", "sm120"):
            self.assertEqual(get_gpu_prefill_quant(), "w4a4")

    def test_unknown_quant_raises(self):
        for quant in ("w9a9", "auto"):
            with quant_env(quant, "cutlass", "sm120"), self.assertRaises(ValueError):
                get_gpu_prefill_quant()

    def test_disabled_prefill_has_no_mode(self):
        with quant_env("w4a4", "cutlass", "sm120", prefill="0"):
            self.assertEqual(get_gpu_prefill_quant(), "")

class TestPrefillW4A4(CustomTestCase):
    def test_needs_both_the_mode_and_4bit_weights(self):
        for fmt in (NV, MX):
            with quant_env("w4a4", "cutlass", "sm120"):
                self.assertTrue(is_prefill_w4a4(fmt))
            # The False below is only meaningful against the True above.
            with quant_env(None, "cutlass", "sm120"):
                self.assertFalse(is_prefill_w4a4(fmt))

    def test_16bit_weights_never_take_the_w4a4_path(self):
        with quant_env("w4a4", "cutlass", "sm120"):
            self.assertFalse(is_prefill_w4a4("bf16"))
            self.assertFalse(is_prefill_w4a4("fp8"))

    def test_illegal_combination_raises_instead_of_answering_false(self):
        with quant_env("w4a4", "triton", "sm120"), self.assertRaises(ValueError):
            is_prefill_w4a4(NV)


class TestPrefillW4A4Arming(CustomTestCase):
    """What FusedMoE.lk_moe_enable_w4a4 hands the engine, per loader."""

    class _Drv:
        def __init__(self, supports=True):
            self.supports = supports
            self.calls = []

        def w4a4_support(self):
            return self.supports

        def enable_w4a4(self, a13_gs, a2_gs):
            self.calls.append((float(a13_gs), float(a2_gs)))

    @staticmethod
    def _arm(fmt, drv, per_token=False, **scales):
        # The stub raises on any scale name it was not given, so a loader-specific
        # name or a guard slipping away surfaces as AttributeError instead of
        # silently reading the other loader's tensor.
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        class _Cfg:
            use_per_token_activation = per_token

        class _Method:
            quant_config = _Cfg

        class _Stub:
            lk_weight_format = fmt
            _gpu_prefill_driver = drv
            quant_method = _Method

            def __getattr__(self, name):
                raise AttributeError(name)

        for name, value in scales.items():
            setattr(_Stub, name, value)
        FusedMoE.lk_moe_enable_w4a4(_Stub())

    def test_mxfp4_arms_nothing_and_never_reads_the_nvfp4_scale_names(self):
        drv = self._Drv()
        self._arm(MX, drv)
        self.assertEqual(drv.calls, [])

    def test_an_engine_without_w4a4_is_never_asked_for_the_multipliers(self):
        drv = self._Drv(supports=False)
        self._arm(NV, drv)
        self.assertEqual(drv.calls, [])

    def test_no_driver_arms_nothing(self):
        self._arm(NV, None)

    def test_modelopt_hands_the_reciprocal_of_each_input_scale(self):
        # Ungated checkpoint: one global-scale column, so no gate/up ratio.
        drv = self._Drv()
        self._arm(
            NV,
            drv,
            w13_weight_scale_2=torch.full((2, 1), 1.0),
            w13_input_scale=torch.full((4, 2), 0.5),
            w2_input_scale=torch.full((4,), 0.25),
        )
        self.assertEqual(drv.calls, [(2.0, 4.0)])

    def test_modelopt_divides_the_gate_up_ratio_out_of_the_second_multiplier(
        self,
    ):
        drv = self._Drv()
        self._arm(
            NV,
            drv,
            w13_weight_scale_2=torch.tensor([[2.0, 1.0], [2.0, 1.0]]),
            w13_input_scale=torch.full((4, 2), 0.5),
            w2_input_scale=torch.full((4,), 0.25),
        )
        self.assertEqual(drv.calls, [(2.0, 8.0)])

    def test_compressed_tensors_names_are_used_and_only_stores_are_reciprocated(
        self,
    ):
        # Same layer, other loader: the weight global scale is stored inverted
        # (so the ratio it yields must match the case above) while the input
        # global scale already IS the multiplier.
        drv = self._Drv()
        self._arm(
            NV,
            drv,
            w13_weight_global_scale=torch.tensor([[0.5, 1.0], [0.5, 1.0]]),
            w13_input_global_scale=torch.full((2, 2), 0.5),
            w2_input_global_scale=torch.full((2,), 0.25),
        )
        self.assertEqual(drv.calls, [(0.5, 0.5)])

    def test_per_token_activation_never_reads_the_static_input_scale(self):
        # NaN is what an online-NVFP4 checkpoint leaves behind in w*_input_scale.
        drv = self._Drv()
        self._arm(
            NV,
            drv,
            per_token=True,
            w13_weight_scale_2=torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            w13_input_scale=torch.full((4, 2), float("nan")),
            w2_input_scale=torch.full((4,), float("nan")),
        )
        self.assertEqual(drv.calls, [(1.0, 0.5)])


if __name__ == "__main__":
    unittest.main()
