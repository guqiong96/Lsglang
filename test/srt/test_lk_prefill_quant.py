"""The prefill (engine x precision) decision table, as an executable spec.

``LK_GPU_PREFILL_QUANT`` is the only precision knob and the README documents the
same matrix in prose; this test is the machine-checkable half so a table edit
cannot drift away from the docs unnoticed. Two tables, one per device family:
the block-scaled CUTLASS collectives are SM120-only, flashinfer's CuTe kernels
are SM12x-only, and everything else falls back to W4A16.

    python3 -m pytest test/srt/test_lk_prefill_quant.py -v
"""

import os
import unittest

import torch

from sglang.srt.utils.lk_moe_env import (
    _GPU_PREFILL_QUANT_MODES,
    engine_quant_supported,
    get_gpu_prefill_quant,
    resolve_gpu_prefill_quant,
)
from sglang.test.test_utils import CustomTestCase

NV, MX = "nvfp4", "mxfp4"


class TestPrefillQuantMatrix(CustomTestCase):
    def _resolve(self, mode, engine, fmt, sm120):
        return resolve_gpu_prefill_quant(mode, engine, fmt, sm120)

    def test_only_known_modes(self):
        self.assertEqual(set(_GPU_PREFILL_QUANT_MODES),
                         {"auto", "w4a4", "w4a8", "w4a16", "w8a8"})

    def test_sm120_table(self):
        # w4a4 exists for the two engines that carry block-scaled kernels.
        # cutlass covers both 4-bit weight families; flashinfer's kernel is
        # NVFP4-only, so MXFP4 degrades to w4a16 there.
        for engine in ("cutlass", "flashinfer"):
            self.assertEqual(self._resolve("w4a4", engine, NV, True), "w4a4")
        self.assertEqual(self._resolve("w4a4", "cutlass", MX, True), "w4a4")
        self.assertEqual(self._resolve("w4a4", "flashinfer", MX, True), "w4a16")
        self.assertEqual(self._resolve("w4a4", "triton", NV, True), "w4a16")
        for engine in ("cutlass", "triton", "flashinfer"):
            self.assertEqual(self._resolve("w4a16", engine, NV, True), "w4a16")

    def test_sm8x_table(self):
        # No block-scaled kernels below SM120: every engine keeps W4A16, and
        # flashinfer is not even available there (its kernels are SM12x-only).
        for engine in ("cutlass", "triton"):
            self.assertEqual(self._resolve("w4a4", engine, NV, False), "w4a16")
            self.assertFalse(engine_quant_supported(engine, "w4a4", NV, False))
        self.assertEqual(self._resolve("auto", "flashinfer", NV, False), "w4a16")
        self.assertFalse(engine_quant_supported("flashinfer", "w4a4", NV, False))

    def test_auto_is_conservative(self):
        # The experiment is opt-in: "auto" must not pick w4a4 until the policy
        # constant graduates, even where the engine supports it.
        for engine in ("cutlass", "flashinfer"):
            self.assertEqual(self._resolve("auto", engine, NV, True), "w4a16")

    def test_reserved_modes_degrade(self):
        # w4a8 / w8a8 have kernels upstream but no engine wiring yet.
        self.assertEqual(self._resolve("w4a8", "cutlass", NV, True), "w4a16")
        self.assertEqual(self._resolve("w8a8", "cutlass", "fp8", True), "w8a16")

    def test_illegal_values_raise(self):
        with self.assertRaises(ValueError):
            self._resolve("w4a4", "cutlass", "fp8", True)   # 4-bit mode on fp8
        old = os.environ.pop("LK_GPU_PREFILL_QUANT", None)
        os.environ["LK_GPU_PREFILL_QUANT"] = "w9a9"
        try:
            with self.assertRaises(ValueError):
                get_gpu_prefill_quant()
        finally:
            os.environ.pop("LK_GPU_PREFILL_QUANT", None)
            if old is not None:
                os.environ["LK_GPU_PREFILL_QUANT"] = old

    def test_retired_flag_raises(self):
        old = os.environ.pop("LK_GPU_PREFILL_QUANT", None)
        os.environ["LK_GPU_PREFILL_SM120_W4A4"] = "1"
        try:
            with self.assertRaises(ValueError):
                get_gpu_prefill_quant()
        finally:
            os.environ.pop("LK_GPU_PREFILL_SM120_W4A4", None)
            if old is not None:
                os.environ["LK_GPU_PREFILL_QUANT"] = old


if __name__ == "__main__":
    unittest.main()
