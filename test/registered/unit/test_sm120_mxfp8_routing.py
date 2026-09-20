"""Backend selection is opt-in and does not change other shapes or platforms."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestB12xRouting(unittest.TestCase):
    def test_dispatch_matrix_and_bias_shape(self):
        for enabled in [False, True]:
            for sm120 in [False, True]:
                for rows in [1, 2, 3, 4, 5, 6, 8]:
                    for backend in ["cutlass", "trtllm"]:
                        with self.subTest(
                            enabled=enabled, sm120=sm120, rows=rows, backend=backend
                        ):
                            x = torch.zeros((1, rows, 32), dtype=torch.float8_e4m3fn)
                            w = torch.zeros((32, 32), dtype=torch.float8_e4m3fn)
                            bias = torch.ones(32, dtype=torch.bfloat16)
                            with (
                                envs.SGLANG_SM120_MXFP8_B12X_SMALL_BATCH.override(
                                    enabled
                                ),
                                patch.object(
                                    fp8_utils,
                                    "get_platform",
                                    return_value=SimpleNamespace(is_sm120=sm120),
                                ),
                                patch.object(
                                    fp8_utils,
                                    "flashinfer_mm_mxfp8",
                                    create=True,
                                    return_value=torch.zeros(
                                        rows, 32, dtype=torch.bfloat16
                                    ),
                                ) as mm,
                            ):
                                out = fp8_utils.flashinfer_mxfp8_blockscaled_linear(
                                    x,
                                    w,
                                    torch.zeros(32, 1, dtype=torch.uint8),
                                    input_scale=torch.zeros(rows, 1, dtype=torch.uint8),
                                    bias=bias,
                                    output_dtype=torch.bfloat16,
                                    backend=backend,
                                )
                            expected = (
                                "b12x"
                                if enabled
                                and sm120
                                and rows in [1, 4, 6]
                                and backend == "cutlass"
                                else backend
                            )
                            self.assertEqual(mm.call_args.kwargs["backend"], expected)
                            self.assertEqual(tuple(out.shape), (1, rows, 32))
                            self.assertTrue(torch.equal(out, torch.ones_like(out)))


if __name__ == "__main__":
    unittest.main()
