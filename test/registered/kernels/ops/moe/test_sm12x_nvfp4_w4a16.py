# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the SM12x NVFP4 W4A16 fused MoE runner backend
(flashinfer CuTe DSL kernels for SM120/SM121).

Synthetic weights: verifies kernel arithmetic, our swizzle_blockscale layout
compatibility, out-of-range expert routing, bucket padding, and CUDA-graph
replay determinism against a dequantized BF16 reference. Also covers
checkpoints that store independent gate/up global scales (the MTP draft
module does), which the quant method folds onto GEMM2's scale. Ported from
lovedheart/sglang qsa-on-main."""

import types

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.sm12x_nvfp4_bf16 import (
    Sm12xNvfp4W4A16MoeQuantInfo,
    fused_experts_none_to_sm12x_nvfp4_bf16,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")

cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
if cc not in ((12, 0), (12, 1)):
    pytest.skip("sm12x_nvfp4_bf16 kernels require SM120/SM121", allow_module_level=True)

DEV = "cuda"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)


def _dequant(w_u8, sf, gs):
    lo, hi = w_u8 & 0x0F, w_u8 >> 4
    vals = []
    for nib in (lo, hi):
        mag = E2M1[(nib & 0x07).long()]
        vals.append(mag * (1.0 - 2.0 * ((nib >> 3) & 1).float()))
    v = torch.stack(vals, dim=-1).flatten(-2).float()
    sf_f = sf.float().repeat_interleave(16, dim=-1)[:, :, : v.shape[-1]]
    return v * sf_f * gs.view(-1, 1, 1)


def _fold_scales(gate_gs, up_gs, w2_gs):
    """(gs13, gs2) exactly as the sm12x apply() path would compute them.

    Called unbound against a stub layer so the test guards the production fold
    instead of a copy of its arithmetic."""
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    layer = types.SimpleNamespace(
        w13_weight_scale_2=torch.stack([gate_gs, up_gs], dim=1),
        w2_weight_scale_2=w2_gs,
    )
    return ModelOptNvFp4FusedMoEMethod._sm12x_global_scales(None, layer)


def _make_layer(E, H, I, split_gs=False):
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=DEV)
    w13_sf = (torch.rand(E, 2 * I, H // 16, device=DEV) * 0.2).to(torch.float8_e4m3fn)
    gate_gs = (10 ** torch.empty(E, device=DEV).uniform_(-3, -2)).float()
    up_gs = (
        (10 ** torch.empty(E, device=DEV).uniform_(-3, -2)).float()
        if split_gs
        else gate_gs.clone()
    )
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=DEV)
    w2_sf = (torch.rand(E, H, I // 16, device=DEV) * 0.2).to(torch.float8_e4m3fn)
    w2_gs = (10 ** torch.empty(E, device=DEV).uniform_(-3, -2)).float()
    gs13, gs2 = _fold_scales(gate_gs, up_gs, w2_gs)
    q = Sm12xNvfp4W4A16MoeQuantInfo(
        w13=w13,
        w13_sf=swizzle_blockscale(w13_sf),
        w13_gs=gs13,
        w2=w2,
        w2_sf=swizzle_blockscale(w2_sf),
        w2_gs=gs2,
    )
    return q, (w13, w13_sf, gate_gs, w2, w2_sf, w2_gs, up_gs)


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
    routed_scaling_factor = None


def test_sm12x_w4a16_routed_scaling_factor():
    """A model that scales the routed sum outside topk must see that factor in
    the output, and a quant method that already folded it into topk_weights
    (apply_routed_scaling_factor=False) must not see it twice."""
    torch.manual_seed(4)
    E, H, I, TOPK = 8, 256, 128, 2
    q, _ = _make_layer(E, H, I)
    x = torch.randn(4, H, dtype=torch.bfloat16, device=DEV)
    ids = torch.randint(0, E, (4, TOPK), dtype=torch.int32, device=DEV)
    wts = torch.rand(4, TOPK, dtype=torch.float32, device=DEV)
    d = _Dispatch(x, ids, wts)
    base = fused_experts_none_to_sm12x_nvfp4_bf16(d, q, _Cfg()).hidden_states
    scaled_cfg = types.SimpleNamespace(
        num_fused_shared_experts=0,
        apply_router_weight_on_input=False,
        routed_scaling_factor=2.5,
    )
    got = fused_experts_none_to_sm12x_nvfp4_bf16(d, q, scaled_cfg).hidden_states
    ref = base.float() * 2.5
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 2e-2, f"rel_err {rel:.2e} too high"

    q.apply_routed_scaling_factor = False
    try:
        again = fused_experts_none_to_sm12x_nvfp4_bf16(d, q, scaled_cfg).hidden_states
    finally:
        q.apply_routed_scaling_factor = True
    torch.testing.assert_close(again, base)


def _reference(raw, x, ids, wts):
    w13, w13_sf, gate_gs, w2, w2_sf, w2_gs, up_gs = raw
    E = w13.shape[0]
    I = w13.shape[1] // 2
    w13_dq = torch.cat(
        [
            _dequant(w13[:, :I], w13_sf[:, :I], gate_gs),
            _dequant(w13[:, I:], w13_sf[:, I:], up_gs),
        ],
        dim=1,
    )
    w2_dq = _dequant(w2, w2_sf, w2_gs)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=DEV)
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = ids[t, j].item()
            if not 0 <= e < E:
                continue
            g = x[t].float() @ w13_dq[e, :I].t()
            u = x[t].float() @ w13_dq[e, I:].t()
            out[t] += wts[t, j] * ((torch.nn.functional.silu(g) * u) @ w2_dq[e].t())
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 8, 16])
def test_sm12x_w4a16_correctness(T):
    torch.manual_seed(0)
    E, H, I, TOPK = 8, 256, 128, 2
    q, raw = _make_layer(E, H, I)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV)
    ids = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=DEV)
    wts = torch.rand(T, TOPK, dtype=torch.float32, device=DEV)
    got = fused_experts_none_to_sm12x_nvfp4_bf16(
        _Dispatch(x, ids, wts), q, _Cfg()
    ).hidden_states
    ref = _reference(raw, x, ids, wts)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 2e-2, f"rel_err {rel:.2e} too high"


def test_sm12x_w4a16_out_of_range_expert():
    torch.manual_seed(1)
    E, H, I, TOPK = 8, 256, 128, 2
    q, raw = _make_layer(E, H, I)
    x = torch.randn(4, H, dtype=torch.bfloat16, device=DEV)
    ids = torch.randint(0, E, (4, TOPK), dtype=torch.int32, device=DEV)
    ids[1, 0] = E  # padding/non-local slot must contribute nothing
    wts = torch.rand(4, TOPK, dtype=torch.float32, device=DEV)
    got = fused_experts_none_to_sm12x_nvfp4_bf16(
        _Dispatch(x, ids, wts), q, _Cfg()
    ).hidden_states
    ref = _reference(raw, x, ids, wts)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 2e-2


def test_sm12x_w4a16_cuda_graph_replay():
    torch.manual_seed(2)
    E, H, I, TOPK = 8, 256, 128, 2
    q, raw = _make_layer(E, H, I)
    x = torch.randn(4, H, dtype=torch.bfloat16, device=DEV)
    ids = torch.randint(0, E, (4, TOPK), dtype=torch.int32, device=DEV)
    wts = torch.rand(4, TOPK, dtype=torch.float32, device=DEV)
    d = _Dispatch(x, ids, wts)
    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fused_experts_none_to_sm12x_nvfp4_bf16(d, q, _Cfg())
        torch.cuda.current_stream().wait_stream(s)
        fused_experts_none_to_sm12x_nvfp4_bf16(d, q, _Cfg())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fused_experts_none_to_sm12x_nvfp4_bf16(d, q, _Cfg())
        # mutate the (static) input buffers in place and replay
        x.copy_(torch.randn_like(x))
        ids.copy_(torch.randint(0, E, (4, TOPK), dtype=torch.int32, device=DEV))
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(raw, x, ids, wts)
        rel = (
            (out.hidden_states.float() - ref.float()).norm() / ref.float().norm()
        ).item()
        assert rel < 2e-2


def test_sm12x_global_scale_fold():
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    fold = ModelOptNvFp4FusedMoEMethod._sm12x_global_scales
    gate = torch.tensor([1.0, 3.0, 0.0, 2.0], device=DEV)
    up = torch.tensor([2.0, 3.0, 5.0, 2.0], device=DEV)
    w2 = torch.tensor([1e-3, 2e-3, 3e-3, 4e-3], device=DEV)
    layer = types.SimpleNamespace(
        w13_weight_scale_2=torch.stack([gate, up], dim=1),
        w2_weight_scale_2=w2,
    )
    gs13, gs2 = fold(None, layer)
    torch.testing.assert_close(gs13, gate)
    # gate keeps its own scale into silu; up's residual rides on GEMM2.
    torch.testing.assert_close(
        gs2, torch.tensor([2e-3, 2e-3, 3e-3, 4e-3], device=DEV)
    )
    assert torch.isfinite(gs2).all(), "a zero gate scale must not divide into inf"
    again = fold(None, layer)
    assert again[0] is gs13 and again[1] is gs2, "fold must be cached per layer"

    shared = types.SimpleNamespace(
        w13_weight_scale_2=torch.stack([gate, gate], dim=1),
        w2_weight_scale_2=w2,
    )
    torch.testing.assert_close(fold(None, shared)[1], w2)
    collapsed = types.SimpleNamespace(
        w13_weight_scale_2=gate.reshape(-1, 1), w2_weight_scale_2=w2
    )
    torch.testing.assert_close(fold(None, collapsed)[1], w2)


def test_sm12x_w4a16_split_gate_up_scale():
    torch.manual_seed(3)
    E, H, I, TOPK = 8, 256, 128, 2
    q, raw = _make_layer(E, H, I, split_gs=True)
    gate_gs, up_gs = raw[2], raw[6]
    assert not torch.equal(gate_gs, up_gs), "scales must actually be split"
    x = torch.randn(4, H, dtype=torch.bfloat16, device=DEV)
    ids = torch.randint(0, E, (4, TOPK), dtype=torch.int32, device=DEV)
    wts = torch.rand(4, TOPK, dtype=torch.float32, device=DEV)
    got = fused_experts_none_to_sm12x_nvfp4_bf16(
        _Dispatch(x, ids, wts), q, _Cfg()
    ).hidden_states
    ref = _reference(raw, x, ids, wts)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 2e-2, f"rel_err {rel:.2e} too high"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
