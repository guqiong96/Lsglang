# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for the SM12x NVFP4 fused MoE runner backend.

Sweeps token counts from decode to large prefill, and compares the two arms of
the backend: W4A16 (BF16 activations, the numerics a W4A16 checkpoint is
calibrated for) against W4A4 (NVFP4 activations, flashinfer's moe_nvfp4_w4a4
sibling kernel on the same weights). Ported from lovedheart/sglang
qsa-on-main; the geometry is overridable so the *per-TP-rank* shape can be
measured, not just the single-GPU one:

    SM12X_BENCH_E/H/I/TOPK=512/2560/320/10  # qwen3.8 under TP2

Run: python test/registered/kernels/benchmark/moe/bench_sm12x_nvfp4_w4a16.py
"""

import os

import pytest
import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_bf16 import (
    Sm12xNvfp4W4A16MoeQuantInfo,
    fused_experts_none_to_sm12x_nvfp4_bf16,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
if cc not in ((12, 0), (12, 1)):
    pytest.skip("sm12x_nvfp4_bf16 kernels require SM120/SM121", allow_module_level=True)

_Env = os.environ.get
E = int(_Env("SM12X_BENCH_E", 512))
H = int(_Env("SM12X_BENCH_H", 2560))
I = int(_Env("SM12X_BENCH_I", 640))
TOPK = int(_Env("SM12X_BENCH_TOPK", 10))


def _make_quant_info():
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    w13_sf = (torch.rand(E, 2 * I, H // 16, device="cuda") * 0.2).to(
        torch.float8_e4m3fn
    )
    gs13 = (10 ** torch.empty(E, device="cuda").uniform_(-3, -2)).float()
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device="cuda")
    w2_sf = (torch.rand(E, H, I // 16, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    gs2 = (10 ** torch.empty(E, device="cuda").uniform_(-3, -2)).float()
    # Synthetic activation multipliers: the W4A4 arm of a bench measures speed,
    # not quality (a checkpoint's real 1/input_scale ride on the quant info).
    unit = torch.ones((), dtype=torch.float32, device="cuda")
    return Sm12xNvfp4W4A16MoeQuantInfo(
        w13=w13,
        w13_sf=swizzle_blockscale(w13_sf),
        w13_gs=gs13,
        w2=w2,
        w2_sf=swizzle_blockscale(w2_sf),
        w2_gs=gs2,
        a13_gs=unit,
        a2_gs=unit,
    )


Q = _make_quant_info()


class _TopK:
    def __init__(self, ids, wts):
        self.topk_ids = ids
        self.topk_weights = wts
        self.router_logits = None


class _Dispatch:
    def __init__(self, x, ids, wts):
        self.hidden_states = x
        self.hidden_states_scale = None
        self.topk_output = _TopK(ids, wts)


class _Cfg:
    num_fused_shared_experts = 0
    apply_router_weight_on_input = False


def run_layer(x, ids, wts, arm="sm12x_w4a16"):
    """Same fused entry point the runner uses; the arm is picked by the
    production hybrid gate, so the bench cannot drift from the served path."""
    prev = os.environ.get("SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS")
    if arm == "sm12x_w4a4":
        os.environ["SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS"] = "1"
    else:
        os.environ["SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS"] = "0"
    try:
        return fused_experts_none_to_sm12x_nvfp4_bf16(
            _Dispatch(x, ids, wts), Q, _Cfg()
        ).hidden_states
    finally:
        if prev is None:
            os.environ.pop("SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS", None)
        else:
            os.environ["SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS"] = prev


@marker.parametrize(
    "T", [1, 4, 16, 64, 128, 512, 1024, 4096], [1, 16, 256]
)  # 128/512 = the tuned bucket band
@marker.benchmark("impl", ["sm12x_w4a16", "sm12x_w4a4"], unit="us")
def benchmark(T: int, impl: str):
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device="cuda")
    wts = torch.softmax(torch.rand(T, TOPK, device="cuda"), dim=-1)
    return marker.do_bench(
        lambda: run_layer(x, ids, wts, impl),
        memory_args=(x, Q.w13, Q.w2),
        memory_output="out",
        graph_clone_args="all",
    )


if __name__ == "__main__":
    benchmark.run()
