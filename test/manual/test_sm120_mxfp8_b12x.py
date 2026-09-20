"""Small SM120 opt-in kernel test. No checkpoint or inference server required."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_blockscaled_linear


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires SM120 CUDA",
)
class TestSm120B12x(unittest.TestCase):
    def test_wrapper_against_dequantized_reference(self):
        import flashinfer

        torch.manual_seed(271)
        torch.backends.cuda.matmul.allow_tf32 = False
        for rows in [1, 4, 6]:
            for n, k in [
                (256, 1280),
                (512, 2304),
                (256, 5120),
                (512, 6144),
                (256, 8192),
            ]:
                with self.subTest(rows=rows, n=n, k=k):
                    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
                    exp = torch.randint(
                        124, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8
                    )
                    scales = flashinfer.block_scale_interleave(
                        exp.repeat_interleave(32, dim=0)
                    )
                    with envs.SGLANG_SM120_MXFP8_B12X_SMALL_BATCH.override(True):
                        actual = flashinfer_mxfp8_blockscaled_linear(
                            x, weight, scales, backend="cutlass", pin_tactic=True
                        )
                    # Quantized activations, with unswizzled scales for a separate reference.
                    qx, xscale = flashinfer.mxfp8_quantize(
                        x, is_sf_swizzled_layout=False, alignment=32
                    )
                    qx = qx.reshape(-1, k)[:rows]
                    xscale = xscale.reshape(-1, k // 32)[:rows]
                    dequant_x = qx.float() * torch.exp2(
                        xscale.float() - 127
                    ).repeat_interleave(32, dim=1)
                    dequant_w = weight.float() * torch.exp2(
                        exp.float() - 127
                    ).repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
                    expected = dequant_x @ dequant_w.t()
                    error = (
                        actual.float() - expected
                    ).norm() / expected.norm().clamp_min(1e-12)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertLess(error.item(), 0.005)


if __name__ == "__main__":
    unittest.main()
