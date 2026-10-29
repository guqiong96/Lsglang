"""Init-time pre-load of the DSpark accept/sampling Triton kernels.

Why this lives here instead of in the model-level prewarm hook: the accept
arm runs *above* the model forward. ``accept_and_finalize`` is reached from
``dspark_worker_v2.forward_batch_generation -> _forward_decode`` (the
scheduler-side verify step), while ``BaseRunner._dummy_run`` -- the replay the
``prepare_before_cuda_graph_capture`` hook drives -- stops at
``model_runner.forward()``. So a dummy verify, greedy or not, never enters
``accept_draft_tokens`` and never warms these kernels; measured on the 3090
pair, a sampling replay still left all four accept kernels cold.

What the load costs when it is missed: the first *sampled* request takes a
Triton JIT compile plus a ``cuModuleLoadData`` with the pool/arena and the
CUDA graphs committed (~1.1-1.5 GiB free). That load spins on the driver
instead of failing -- GPU idle, one core at 100%, the batch wedged -- so it is
paid here, at warmup, while the allocator still has room.

Two details decide whether a pre-load actually covers serving:

* Batch size. Triton folds an integer argument that equals 1 into the kernel
  key as a constexpr, so ``bs=1`` (a single request) and ``bs=2`` are different
  specialisations. This loops over every bucket via ``batch_buckets``.
* Dtypes. They are the ones the serving path builds (``dflash_utils
  ._get_or_create_chain_verify_buffers``): int64 for the chain-retrieve buffers
  and the candidates, int32 for predicts / accept_index / accept_token_num, and
  int32 for the length/trim tensors. Keep these in sync with that function.

Greedy traffic never enters this arm (``AcceptGreedy`` handles it), so without
this pre-load a deployment that only sees greedy requests looks clean while the
first sampled request wedges.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    cap_correct_len_triton,
    finalize_accept_lens_triton,
    gather_row_bonus_triton,
    gather_two_level_bonus_triton,
    select_mixed_accept_triton,
    softmax_temp_triton,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.utils.triton_preload import batch_buckets


def preload_dspark_accept_kernels(
    device,
    batch_size: int,
    vocab_size: int,
    num_slots: int,
) -> None:
    """Compile and load the DSpark accept/sampling kernels for bs=1..cap.

    ``num_slots`` is the verified draft width (``verify_num_draft_tokens``);
    ``batch_size`` is the request cap, so every batch the scheduler can form
    gets its own specialisation loaded here.
    """
    max_bs = max(1, int(batch_size))
    vocab_size = int(vocab_size)
    num_slots = max(2, int(num_slots))
    dev = torch.device(device) if not isinstance(device, torch.device) else device

    with torch.inference_mode():
        for bs in batch_buckets(max_bs):

            def i32(*shape: int) -> torch.Tensor:
                return torch.zeros(shape, dtype=torch.int32, device=dev)

            def i64(*shape: int) -> torch.Tensor:
                return torch.zeros(shape, dtype=torch.int64, device=dev)

            # _gather_two_level_bonus_kernel: two-level accept (sampling path).
            gather_two_level_bonus_triton(
                accept_index=torch.full(
                    (bs, num_slots), -1, dtype=torch.int32, device=dev
                ),
                predicts=i32(bs * num_slots),
                correct_len=i32(bs),
            )

            # _mixed_accept_select_kernel: merges the greedy and sampling
            # results.
            select_mixed_accept_triton(
                greedy_mask=torch.zeros((bs,), dtype=torch.bool, device=dev),
                greedy_len=i32(bs),
                greedy_bonus=i64(bs),
                greedy_trim=i32(bs),
                sampling_len=i32(bs),
                sampling_bonus=i64(bs),
                sampling_trim=i32(bs),
            )

            # _gather_row_bonus_kernel: two-level accept (greedy path).
            gather_row_bonus_triton(
                table=i64(bs, num_slots),
                idx=i32(bs),
            )

            # _finalize_accept_lens_kernel: seq-lens commit. The prefix length
            # is a seq_len tensor, which serving passes as int32; the int64 arm
            # is covered too because a long-context config can hand it int64.
            for prefix_dtype in (torch.int32, torch.int64):
                finalize_accept_lens_triton(
                    correct_len=i32(bs),
                    cap_trim_lens=i32(bs),
                    prefix_lens=torch.zeros((bs,), dtype=prefix_dtype, device=dev),
                )

            # _cap_correct_len_kernel: caps the accepted length at the verify
            # width.
            cap_correct_len_triton(
                correct_len=i32(bs),
                verify_lens=i32(bs),
            )

            # _softmax_temp_kernel: temperature softmax of the verify logits.
            softmax_temp_triton(
                logits=torch.zeros(
                    (bs * num_slots, vocab_size), dtype=torch.float32, device=dev
                ),
                temperatures=torch.ones((bs,), dtype=torch.float32, device=dev),
                rows_per_request=num_slots,
            )

            # speculative_sampling_classic_kernel: chain rejection sampling.
            # Both block-verification arms are loaded because that flag changes
            # num_warps, i.e. it is part of the kernel key.
            for block_verification in (False, True):
                chain_speculative_sampling_triton(
                    predicts=torch.zeros(
                        (bs * num_slots,), dtype=torch.int32, device=dev
                    ),
                    accept_index=torch.full(
                        (bs, num_slots), -1, dtype=torch.int32, device=dev
                    ),
                    accept_token_num=i32(bs),
                    candidates=i64(bs, num_slots),
                    retrive_index=i64(bs, num_slots),
                    retrive_next_token=i64(bs, num_slots),
                    retrive_next_sibling=torch.full(
                        (bs, num_slots), -1, dtype=torch.int64, device=dev
                    ),
                    uniform_samples=torch.rand(
                        (bs, max(1, num_slots - 1)), dtype=torch.float32, device=dev
                    ),
                    uniform_samples_for_final_sampling=torch.rand(
                        (bs,), dtype=torch.float32, device=dev
                    ),
                    target_probs=torch.zeros(
                        (bs, num_slots, vocab_size), dtype=torch.float32, device=dev
                    ),
                    draft_probs=torch.zeros(
                        (bs, num_slots, vocab_size), dtype=torch.float32, device=dev
                    ),
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    deterministic=True,
                    block_verification=block_verification,
                )
