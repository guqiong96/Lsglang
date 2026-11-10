"""Prefill post/combine/RMSNorm with the original BF16 boundaries and norm tree."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _module():
    return load_jit(
        "mhc_post_combine_norm_prefill",
        cuda_files=["deepseek_v4/mhc_post_combine_norm_prefill.cuh"],
        cuda_wrappers=[("run", "MhcPostCombineNormPrefill<128>::run")],
    )


@register_custom_op(mutates_args=["updated", "normalized"])
def _mhc_post_combine_norm_prefill(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    updated: torch.Tensor,
    normalized: torch.Tensor,
    eps: float,
) -> None:
    _module().run(x, residual, post, comb, pre, weight, updated, normalized, eps)


def mhc_post_combine_norm_prefill(
    x, residual, post, comb, pre, weight, eps, *, residual_out=None
):
    """Fused prefill post/combine/RMSNorm.

    ``residual_out`` may be ``residual`` itself: the kernel reads each tile
    before writing the same tile from the same thread, and the norm reads the
    combined row back from shared memory rather than from the output, so the
    input is dead once its tile is read. Passing the caller's residual in makes
    the [tokens, 4, 5120] bf16 output free -- at the 8192-row super-chunk that
    is 320 MiB per forward, and on a 16 GB card it is what a long prefill after
    a logprob-heavy batch died asking for (2026-11-11, DeepSeek-V4.1 on
    2x5060 Ti, 147 MiB free). Same aliasing the sibling post/pre boundary
    already relies on (ee67618bad).
    """
    updated = torch.empty_like(residual) if residual_out is None else residual_out
    normalized = torch.empty_like(x)
    _mhc_post_combine_norm_prefill(
        x, residual, post, comb, pre, weight, updated, normalized, eps
    )
    return updated, normalized
