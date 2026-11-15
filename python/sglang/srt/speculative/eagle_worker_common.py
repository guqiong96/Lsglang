from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_draft_cache_locs_contiguous,
)
from sglang.kernels.ops.speculative.eagle import (
    build_chain_tree,
    fill_bonus_tokens_func,
    prepare_draft_extend_layout,
    prepare_draft_extend_lengths,
    prepare_verify_commit_outputs,
)
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import (
    get_server_args,
    mamba_cache_chunk_size,
    mamba_track_grid,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    commit_mamba_states_after_verify,
    move_accept_tokens_to_target_kvcache,
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.utils import is_cpu
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import is_npu

_is_cpu = is_cpu()
_is_npu = is_npu()

if _is_cpu:
    from sgl_kernel import assign_draft_cache_locs_contiguous_cpu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
    from sglang.srt.mem_cache.kv_loc_plan import KVLocPlan
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftExtendInput


def duplicate_prefix_tail_to_draft_branches(
    token_to_kv_pool,
    rows: torch.Tensor,
    prefix_base: torch.Tensor,
    last_page: torch.Tensor,
    num_new_pages: torch.Tensor,
    topk: int,
    page_size: int,
) -> None:
    """Copy the prefix partial-tail page into each branch's first-page holes (page>1 + topk>1).

    The draft-decode expand pass reads each branch's own draft page by block id
    (cache_loc // page_size), so branch b>=1's hole slots [0, last_page) must hold the
    real prefix tail (branch 0's first page already is it). Mirrors V1 #7725.
    """
    if topk <= 1:
        return
    bs = rows.shape[0]
    page_off = torch.arange(page_size, device=rows.device, dtype=torch.int64)
    branches = torch.arange(1, topk, device=rows.device, dtype=torch.int64).view(
        1, topk - 1, 1
    )
    # Source: the prefix tail page [prefix_base, prefix_base + page_size), one per branch.
    src_pos = (prefix_base.view(bs, 1, 1) + page_off.view(1, 1, page_size)).expand(
        bs, topk - 1, page_size
    )
    # Target: branch b's first page [prefix_base + b*num_new_pages*page, + page_size).
    tgt_pos = (
        prefix_base.view(bs, 1, 1)
        + branches * (num_new_pages.view(bs, 1, 1) * page_size)
        + page_off.view(1, 1, page_size)
    )
    # Only [0, last_page) holds real prefix KV; [last_page, page_size) are the branch's
    # own draft slots and must not be overwritten.
    vmask = (page_off.view(1, 1, page_size) < last_page.view(bs, 1, 1)).expand(
        bs, topk - 1, page_size
    )
    src_slots = torch.gather(rows, 1, src_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    tgt_slots = torch.gather(rows, 1, tgt_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    if src_slots.numel() > 0:
        token_to_kv_pool.move_kv_cache(tgt_slots, src_slots)


def prepare_for_draft_extend(
    draft_extend_input: EagleDraftExtendInput,
    batch: ScheduleBatch,
    predict: torch.Tensor,
    num_draft_tokens: int,
    draft_model_runner: Any,
    cuda_graph_runner: Any,
    *,
    return_hidden_states_before_norm: bool,
    widened_out_cache_loc: Optional[torch.Tensor] = None,
    widened_positions: Optional[torch.Tensor] = None,
    kv_loc_plan: Optional[KVLocPlan] = None,
):
    """``kv_loc_plan``: the iteration's plan, whose window the draft extend
    writes again (the verify window); a widened window plans its own."""
    bs = len(batch.seq_lens)
    # Optional window widening (num_front_tokens=0 -> off): prepend that many
    # rows below the boundary. Locs/positions arrive precomputed; token/hidden
    # buffers are zeroed placeholders the caller fills after the plan-stream join.
    num_front_tokens = draft_extend_input.num_front_tokens
    widen = num_front_tokens > 0 and not batch.forward_mode.is_idle()
    front_offset = num_front_tokens if widen else 0
    num_window_tokens = num_draft_tokens + front_offset
    extend_num_tokens = bs * num_window_tokens
    # When seq_lens_cpu is absent, stay on GPU-only path -- no .tolist()/.cpu().
    gpu_only = batch.seq_lens_cpu is None

    batch.spec_info = draft_extend_input
    # Do NOT cast predict dtype here. The caller (e.g., _draft_extend_for_decode)
    # may run this under a plan stream; casting inside the plan stream creates a
    # cross-stream dependency that can lead to data races and break MTP acceptance.
    # The caller should cast to int64 before entering the plan stream context.
    if widen:
        assert widened_out_cache_loc is not None and widened_positions is not None
        batch.input_ids = predict.new_zeros((extend_num_tokens,))
        batch.out_cache_loc = widened_out_cache_loc
        # init_new adopts spec_info.positions when present.
        draft_extend_input.positions = widened_positions
        # Placeholder for the widened hidden window, filled by the worker.
        if draft_extend_input.hidden_states is not None:
            draft_extend_input.hidden_states = (
                draft_extend_input.hidden_states.new_empty(
                    (extend_num_tokens, draft_extend_input.hidden_states.shape[1])
                )
            )
    else:
        batch.input_ids = predict
    maybe_detect_oob(
        batch.input_ids,
        0,
        batch.model_config.vocab_size,
        "v2 prepare_for_draft_extend input_ids",
    )
    # init_new requires both list or both Tensor;
    # gpu_only emits device tensors to skip H2D.
    fused_lengths = gpu_only and batch.seq_lens.is_cuda
    extend_position_info = None
    if fused_lengths and not widen and draft_extend_input.positions is None:
        batch.prefix_lens, batch.extend_lens, post_seq_lens, extend_position_info = (
            prepare_draft_extend_layout(
                batch.seq_lens,
                num_draft_tokens,
                draft_model_runner.model_config.model_is_mrope
                and all(x is None for x in batch.multimodal_inputs),
            )
        )
    elif fused_lengths:
        batch.prefix_lens, batch.extend_lens, post_seq_lens = (
            prepare_draft_extend_lengths(batch.seq_lens, num_draft_tokens, front_offset)
        )
    elif gpu_only:
        batch.prefix_lens = (batch.seq_lens - front_offset).clamp(min=0).to(torch.int32)
        batch.extend_lens = torch.full(
            (bs,), num_window_tokens, dtype=torch.int32, device=batch.seq_lens.device
        )
    else:
        batch.prefix_lens = [
            max(int(x) - front_offset, 0) for x in batch.seq_lens_cpu.tolist()
        ]
        batch.extend_lens = [num_window_tokens] * bs
    batch.extend_num_tokens = extend_num_tokens
    capture_mode = (
        CaptureHiddenMode.NULL
        if draft_model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.FULL
    )
    batch.forward_mode = (
        ForwardMode.IDLE
        if batch.forward_mode.is_idle()
        else ForwardMode.DRAFT_EXTEND_V2
    )
    forward_batch = ForwardBatch.init_new(
        batch,
        draft_model_runner,
        capture_hidden_mode=capture_mode,
        return_hidden_states_before_norm=return_hidden_states_before_norm,
        extend_position_info=extend_position_info,
        kv_loc_plan=None if widen else kv_loc_plan,
    )
    # Forward sees post-write length (draft extend writes num_draft_tokens
    # slots); mutation stays on forward_batch to preserve SB.seq_lens.
    forward_batch.seq_lens = (
        post_seq_lens if fused_lengths else forward_batch.seq_lens + num_draft_tokens
    )
    if not gpu_only:
        forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + num_draft_tokens
        forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
    else:
        # Supply CPU mirror (extend_seq_lens are all num_window_tokens) so
        # backend max() reads from list without a per-iter D2H sync.
        forward_batch.extend_seq_lens_cpu = [num_window_tokens] * bs
    can_run_decode_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(
        forward_batch
    )
    if not batch.forward_mode.is_idle() and not can_run_decode_cuda_graph:
        draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        # Planned pre-pad; do NOT opt into post-pad re-plan. DSA's indexer
        # cannot rebuild its deep_gemm schedule_meta on a DP-padded batch
        # (the `_batch_size == batch_size` assertion, see #27091); the
        # marked pre-pad metadata is used as-is, matching the proven
        # skip_attn_backend_init=True behavior.
        # On NPU with --disable-cuda-graph, block_table shape won't match
        # after prepare_mlp_sync_batch padding; defer re-init to
        # forward_extend (post-pad) instead.
        if not is_npu() or can_run_decode_cuda_graph:
            forward_batch.mark_forward_metadata_ready()
    return forward_batch


def draft_window_cols(
    *, topk: int, num_steps: int, num_draft_tokens: Optional[int], plans_window: bool
) -> int:
    """Columns the draft-cache-loc kernel copies for one draft iteration.

    One function because this number also selects the Triton specialization:
    `window_cols` is a `tl.constexpr`, so a warm-up that derives it differently from
    the served path compiles and loads a different binary, which is worse than not
    warming at all — it looks covered.
    """
    return int(num_draft_tokens) if plans_window else int(num_steps)


def launch_assign_draft_cache_locs(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    topk: int,
    window_cols: int,
    positions: Optional[torch.Tensor],
    mrope: Optional[torch.Tensor],
    bs: int,
) -> None:
    """The single call site of the draft-cache-loc Triton kernel.

    Kept as a function so the warm-up and the served step cannot disagree about
    which specialization gets compiled: the constexprs (`BS`, `window_cols`, and
    `pool_len` from the table's row width) and the int arguments Triton keys on all
    come from here.
    """
    assign_draft_cache_locs_contiguous[(bs,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        req_to_token.shape[1],
        topk,
        window_cols,
        positions=positions,
        mrope=mrope,
        BS=bs,
        WRITE_POSITIONS=positions is not None,
        WRITE_MROPE=mrope is not None,
    )


def draft_prewarm_batch_sizes(max_running_requests: Optional[int]) -> tuple[int, ...]:
    """Every batch size whose draft kernels Triton compiles separately.

    Triton specializes an int argument three ways: the value 1 becomes a constant, a
    multiple of 16 gets the divisibility hint, and everything else shares one binary.
    The draft kernels also take DERIVED ints — bs * window_cols, bs * row stride —
    whose class is not the class of bs: bs=2 gives 8 (the generic binary) while bs=4
    gives 16 (divisible, its own), so a {1, 2, 16} shortcut serves bs=4 with a module
    nobody loaded. Up to 16 this list is therefore exhaustive rather than clever;
    above 16 it falls back to powers of two, which is the honest limit of the
    argument above and the point where startup cost starts to matter.
    """
    cap = int(max_running_requests or 0)
    if cap <= 0:
        return (1,)
    if cap <= 16:
        return tuple(range(1, cap + 1))
    return tuple(
        sorted(set(range(1, 17)) | {p for p in (32, 64, 128, 256) if p <= cap})
    )


def draft_track_intervals(page_size: Optional[int]) -> tuple[int, ...]:
    """The `mamba_track_interval` values the verify-commit kernel is called with.

    `TRACK_INTERVAL` is a `tl.constexpr` there: the served path passes
    `mamba_track_grid(page_size)` for a batch carrying track indices and 0 for one
    that does not, so both are live specializations. `mamba_track_grid` reads the
    execution config, which is not entered during engine init, so when it is
    unavailable the same value is derived from the server args instead of being
    skipped: covering only 0 leaves the served call to device-load a module mid-step,
    which is the hazard this whole warm-up exists to remove.
    """
    if not page_size:
        return (0,)
    intervals = {0}
    try:
        intervals.add(int(mamba_track_grid(int(page_size))))
    except Exception:
        # mamba_track_grid(tree_page) = lcm(mamba_checkpoint_grid(tree_page),
        # mamba_track_interval), and mamba_checkpoint_grid needs only the server args.
        grid = math.lcm(int(mamba_cache_chunk_size()), int(page_size))
        args = get_server_args()
        intervals.add(
            int(math.lcm(grid, int(getattr(args, "mamba_track_interval", 0) or 0)))
        )
    return tuple(sorted(intervals))


def prewarm_draft_step_path(
    req_to_token: torch.Tensor,
    *,
    topk: int,
    num_steps: int,
    num_draft_tokens: Optional[int],
    batch_sizes: Sequence[int] = (1,),
    track_intervals: Sequence[int] = (0,),
) -> int:
    """Device-load every eager kernel one decode step launches, before serving.

    Graph capture covers the draft forward, but `prepare_for_draft`, the verify-input
    build and the verify-commit outputs run OUTSIDE the graph, and they specialize on
    the step count and draft width (`tl.constexpr`). Nothing runs them at init, so the
    first served step pays their first launch: measured 510 ms against 0.03 ms once
    warm. That is merely slow until it lands in a TP-synchronized step while the peer
    rank is parked in a collective — then `cuModuleLoadData` wedges the group, both
    schedulers at 100% CPU inside libcuda with the GPUs idle (this signature killed a
    qwen3.8 boot on 2x5060 Ti on 2026-10-18 and several boots on 2026-11-15).

    Coverage has to match what Triton actually keys on, not just call each kernel
    once: batch size is an int argument (warming bs=1 does not warm bs=2), and
    argument dtype and stride are part of the key, so every dummy below is built the
    way the served call site builds it — bonus tokens int32 from the fused commit
    path and int64 from sampling, draft tokens int64 from the drafter and int32 from
    the graph output, mrope on and off.

    Dummy inputs are zero-filled indices: the kernels read table row 0 and write only
    their own scratch, so nothing in the pools is disturbed.
    """
    if not req_to_token.is_cuda:
        return 0
    device = req_to_token.device
    width = int(num_draft_tokens) if num_draft_tokens else int(num_steps) + 1
    # Both layouts the served path can pick: the window plan (translator present and
    # num_draft_tokens > num_steps) or the plain num_steps copy. Warming one of them
    # is what a wrong assumption about `plans_window` would buy.
    window_variants = sorted({int(num_steps), width})
    launches = 0
    for bs in sorted({max(1, int(b)) for b in batch_sizes}):
        if req_to_token.shape[1] <= bs * max(window_variants) + 1:
            continue
        req_pool_indices = torch.zeros(bs, dtype=torch.int64, device=device)
        seq_lens = torch.zeros(bs, dtype=torch.int64, device=device)
        for window_cols in window_variants:
            out_cache_loc = torch.empty(
                (bs * topk * window_cols,), dtype=torch.int64, device=device
            )
            positions = torch.empty((bs * topk,), dtype=torch.int64, device=device)
            for mrope in (None, torch.zeros((3, bs), dtype=torch.int64, device=device)):
                launch_assign_draft_cache_locs(
                    req_pool_indices=req_pool_indices,
                    req_to_token=req_to_token,
                    seq_lens=seq_lens,
                    out_cache_loc=out_cache_loc,
                    topk=topk,
                    window_cols=window_cols,
                    positions=positions,
                    mrope=mrope,
                    bs=bs,
                )
                launches += 1
        # The chain-tree builder is the verify-input kernel of a chain draft. Its
        # dtypes follow the two producers named above, and it allocates its own mask
        # when given None, which is the same launch the served path makes first.
        for bonus_dtype in (torch.int32, torch.int64):
            for draft_dtype in (torch.int64, torch.int32):
                for with_mrope in (True, False):
                    build_chain_tree(
                        torch.zeros((bs,), dtype=bonus_dtype, device=device),
                        torch.zeros(
                            (bs, num_steps), dtype=draft_dtype, device=device
                        ),
                        seq_lens,
                        None,
                        req_pool_indices=req_pool_indices,
                        req_to_token=req_to_token,
                        with_mrope=bool(with_mrope),
                    )
                    launches += 1
        # Draft extend runs on the next prefill and specializes on the draft width.
        for with_mrope in (True, False):
            prepare_draft_extend_lengths(seq_lens, width, 0)
            prepare_draft_extend_layout(seq_lens, width, bool(with_mrope))
            launches += 2
        # The verify-commit kernel derives the accepted-token bookkeeping; its inputs
        # are eagle_sample's outputs: predict is the flattened int32 token row,
        # accept_index is (bs, width) int32 filled with -1, accept_lens is int32.
        predict = torch.zeros((bs * width,), dtype=torch.int32, device=device)
        accept_index = torch.full((bs, width), -1, dtype=torch.int32, device=device)
        accept_lens = torch.zeros((bs,), dtype=torch.int32, device=device)
        for interval in track_intervals:
            prepare_verify_commit_outputs(
                predict,
                accept_index,
                accept_lens,
                seq_lens,
                num_draft_tokens=width,
                mamba_track_interval=int(interval),
            )
            launches += 1
    return launches


def prepare_for_draft(
    draft_input: EagleDraftInput,
    req_to_token_pool: ReqToTokenPool,
    batch: ScheduleBatch,
    cuda_graph_runner: EAGLEDraftCudaGraphRunner,
    draft_model_runner: ModelRunner,
    topk: int,
    num_steps: int,
    *,
    target_translator: Optional[KVIndexTranslator] = None,
    num_draft_tokens: Optional[int] = None,
) -> tuple[ForwardBatch, bool, Optional[KVLocPlan]]:
    """The draft-decode batch. With ``target_translator``, a chain draft plans
    the iteration's whole write window (the ``num_draft_tokens`` slots past
    ``seq_lens`` that verify and draft extend also write) and writes its first
    ``num_steps`` columns; that plan is returned, else None."""
    prepared_positions = None
    prepared_mrope_positions = None
    kv_loc_plan = None
    if not batch.forward_mode.is_idle():
        bs = len(batch.seq_lens)

        # Assign cache locations (draft-write targets).
        page_size = batch.token_to_kv_pool_allocator.page_size
        plans_window = (
            target_translator is not None
            and topk == 1
            and num_draft_tokens is not None
            and num_draft_tokens > num_steps
            and not _is_cpu
        )
        # A chain draft's slots lead the iteration's window: gather the whole
        # window in the same launch.
        window_cols = draft_window_cols(
            topk=topk,
            num_steps=num_steps,
            num_draft_tokens=num_draft_tokens,
            plans_window=plans_window,
        )
        if page_size == 1 or topk == 1:
            batch.out_cache_loc = torch.empty(
                (bs * topk * window_cols,),
                dtype=torch.int64,
                device=batch.device,
            )
            if _is_cpu:
                assign_draft_cache_locs_contiguous_cpu(
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )
            else:
                if batch.seq_lens.is_cuda and topk == 1:
                    prepared_positions = torch.empty_like(batch.seq_lens)
                    if draft_model_runner.model_config.model_is_mrope and all(
                        x is None for x in batch.multimodal_inputs
                    ):
                        prepared_mrope_positions = torch.empty(
                            (3, bs), dtype=torch.int64, device=batch.device
                        )
                # FIXME(lsyin): align with the default code path
                launch_assign_draft_cache_locs(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=req_to_token_pool.req_to_token,
                    seq_lens=batch.seq_lens,
                    out_cache_loc=batch.out_cache_loc,
                    topk=topk,
                    window_cols=window_cols,
                    positions=prepared_positions,
                    mrope=prepared_mrope_positions,
                    bs=bs,
                )
            if plans_window:
                kv_loc_plan = target_translator.plan(
                    req_pool_indices=batch.req_pool_indices,
                    seq_lens=batch.seq_lens,
                    seq_lens_cpu=batch.seq_lens_cpu,
                    write_virtual=batch.out_cache_loc,
                    read_extent=num_draft_tokens,
                )
        else:
            # page_size > 1 + topk > 1: per-branch page-aligned draft pages.
            # Reduce out_cache_loc from the page-aligned tree region down to the
            # dense draft slots (skip each branch's duplicated prefix-tail slots
            # and trailing padding), matching generate_draft_decode_kv_indices'
            # paged read formula: prefix_base + t*num_new_pages*page + last_page + s.
            # base is batch.seq_lens (== KV-ready committed prefix at draft time;
            # the bonus is the tree root written by verify, not part of [0:seq_lens]).
            rows = req_to_token_pool.req_to_token[batch.req_pool_indices.long()]
            seq_lens = batch.seq_lens.to(torch.int64)
            last_page = seq_lens % page_size
            prefix_base = seq_lens - last_page
            num_new_pages = (last_page + num_steps + page_size - 1) // page_size
            topk_ids = torch.arange(topk, device=rows.device, dtype=torch.int64).view(
                1, topk
            )
            starts = (
                prefix_base.view(bs, 1)
                + topk_ids * (num_new_pages.view(bs, 1) * page_size)
                + last_page.view(bs, 1)
            )
            steps = torch.arange(num_steps, device=rows.device, dtype=torch.int64).view(
                1, 1, num_steps
            )
            pos = (starts.view(bs, topk, 1) + steps).reshape(bs, topk * num_steps)
            batch.out_cache_loc = torch.gather(rows, 1, pos).reshape(-1).contiguous()

            # Each branch's page-aligned region starts with `last_page` hole slots
            # overlapping the prefix tail page; duplicate the real prefix-tail KV
            # into them so whole-page reads stay coherent (see helper docstring).
            duplicate_prefix_tail_to_draft_branches(
                draft_model_runner.token_to_kv_pool,
                rows,
                prefix_base,
                last_page,
                num_new_pages,
                topk,
                page_size,
            )

    # Get a forward batch
    # Actual width of the next draft-decode forward: topk tokens per req.
    draft_input.num_tokens_per_req = topk
    draft_input.num_tokens_for_logprob_per_req = topk
    capture_mode = (
        CaptureHiddenMode.NULL
        if draft_model_runner.spec_algorithm.is_standalone()
        else CaptureHiddenMode.LAST
    )
    draft_input.positions = (
        prepared_positions
        if prepared_positions is not None
        else batch.seq_lens.repeat_interleave(topk, dim=0)
    )
    forward_batch = ForwardBatch.init_new(
        batch,
        draft_model_runner,
        capture_hidden_mode=capture_mode,
        return_hidden_states_before_norm=False,
        spec_mrope_positions=prepared_mrope_positions,
        kv_loc_plan=kv_loc_plan,
        write_cols=None if kv_loc_plan is None else slice(0, num_steps),
    )
    can_run_decode_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(
        forward_batch
    )
    return forward_batch, can_run_decode_cuda_graph, kv_loc_plan


def build_eagle_verify_input(
    batch: ScheduleBatch,
    draft_input: EagleDraftInput,
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    *,
    target_worker: TpModelWorker,
    topk: int,
    num_steps: int,
    num_draft_tokens: int,
    tree_mask_mode: TreeMaskMode,
    device: str,
) -> EagleVerifyInput:
    """Shared draft() tail: idle input, tree-mask build, EagleVerifyInput assembly.

    ``draft_probs`` is the caller's source of draft distributions (single-layer
    eagle: this round's draft_forward output; multi-layer eagle: the ones the
    draft input carried).
    """
    if batch.forward_mode.is_idle():
        return EagleVerifyInput.create_idle_input(
            topk,
            num_steps,
            num_draft_tokens,
            device,
        )

    # Write straight into the backend's buffer when it owns one and this batch
    # fits; an eager batch past the captured max_bs falls back to allocating.
    bs = batch.seq_lens.shape[0]
    target_attn_backend = target_worker.model_runner.attn_backend
    verify_mask = target_attn_backend.verify_mask
    if verify_mask is None:
        tree_mask_buf, mask_mode, fill_mask = None, tree_mask_mode, True
    else:
        mask_mode, fill_mask = verify_mask.mode, verify_mask.is_read
        tree_mask_buf = verify_mask.buffer if verify_mask.fits(bs) else None

    # build_tree_kernel uses seq_lens_sum only to size the (non-preallocated)
    # FULL_MASK tree mask; over-size is safe. Skip per-iter .sum().item() D2H via UB.
    seq_lens_sum = batch.seq_lens_sum
    if seq_lens_sum is None:
        if tree_mask_buf is not None or mask_mode == TreeMaskMode.QLEN_ONLY:
            # Preallocated, or a QLEN_ONLY allocation sized off bs alone.
            seq_lens_sum = 0
        else:
            seq_lens_sum = bs * target_attn_backend.max_context_len

    prepared_out_cache_loc = None
    prepared_mrope_positions = None
    if (
        batch.seq_lens.is_cuda
        and topk == 1
        and num_draft_tokens == num_steps + 1
        and draft_tokens.shape == (bs, num_steps)
        and mask_mode == TreeMaskMode.QLEN_ONLY
    ):
        tree_outputs = build_chain_tree(
            draft_input.bonus_tokens,
            draft_tokens,
            batch.seq_lens,
            tree_mask_buf,
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            with_mrope=target_worker.model_runner.model_config.model_is_mrope
            and all(x is None for x in batch.multimodal_inputs),
        )
        prepared_out_cache_loc, prepared_mrope_positions = tree_outputs[6:]
        tree_outputs = tree_outputs[:6]
    else:
        tree_outputs = build_tree_kernel_efficient(
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            seq_lens_sum,
            topk,
            num_steps,
            num_draft_tokens,
            mask_mode,
            tree_mask_buf,
            fill_prefix_mask=fill_mask,
        )
    (
        tree_mask,
        position,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    ) = tree_outputs

    return EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=tree_mask,
        positions=position,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_cum_len=None,
        spec_steps=num_steps,
        topk=topk,
        draft_token_num=num_draft_tokens,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
        draft_probs=draft_probs,
        prepared_out_cache_loc=prepared_out_cache_loc,
        prepared_mrope_positions=prepared_mrope_positions,
    )


def _finalize_accept_tree_path(
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    predict: torch.Tensor,
    logits_output: Any,
    bs: int,
    *,
    token_to_kv_pool_allocator: Any,
    num_draft_tokens: int,
) -> torch.Tensor:
    """Tree drafting (topk > 1): move the accepted path -- KV slots, predict,
    hidden_states -- to the contiguous front of each per-req block, which the
    downstream chain-layout code (draft-extend select_index, committed-KV reads)
    assumes. Returns compacted predict; mutates logits_output.hidden_states
    (moved only when present)."""
    move_accept_tokens_to_target_kvcache(
        batch, accept_index, accept_lens - 1, token_to_kv_pool_allocator
    )
    predict = _compact_accept_to_front(
        predict, accept_index, bs, num_draft_tokens=num_draft_tokens
    )
    if logits_output.hidden_states is not None:
        logits_output.hidden_states = _compact_accept_to_front(
            logits_output.hidden_states,
            accept_index,
            bs,
            num_draft_tokens=num_draft_tokens,
        )
    return predict


def _compact_accept_to_front(
    x: torch.Tensor,
    accept_index: torch.Tensor,
    bs: int,
    *,
    num_draft_tokens: int,
) -> torch.Tensor:
    """Gather the accepted tree path to the front of each per-req block.

    ``x`` is node-indexed over the whole tree (``[bs * num_draft_tokens, ...]``),
    ``accept_index`` is ``[bs, spec_steps + 1]`` global node indices (-1 padded).
    Padded entries clamp to node 0 but land past accept_lens (never read);
    trailing unaccepted slots stay and are freed as overshoot.
    """
    nd = num_draft_tokens
    s1 = accept_index.shape[1]  # spec_steps + 1
    safe = accept_index.to(torch.int64).clamp(min=0).reshape(-1)
    gathered = x[safe]
    out = x.clone()
    out.view(bs, nd, *x.shape[1:])[:, :s1] = gathered.view(bs, s1, *x.shape[1:])
    return out


def run_eagle_verify(
    batch: ScheduleBatch,
    *,
    target_worker: TpModelWorker,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: Any,
    plan_stream: Any,
    plan_stream_ctx: Any,
    topk: int,
    num_draft_tokens: int,
    device: str,
    metadata_ready_pre_pad: bool,
    finalize_tree_path: bool,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    grammar_barrier=None,
    uno_target_max_top_k: Optional[int] = None,
) -> GenerationBatchResult:
    """Shared verify step: target-verify forward, sampling, acceptance bookkeeping.

    The single-layer eagle verify body is the source of truth (superset). Two
    switches encode the multi-layer worker's preserved-verbatim differences:

    - ``metadata_ready_pre_pad``: multi-layer marks forward metadata ready
      pre-pad unconditionally; single-layer relies on eagle_prepare_for_verify
      marking it only when the cuda-graph path ran.
    - ``finalize_tree_path``: single-layer compacts the accepted tree path to
      the front of each per-req block for topk > 1; multi-layer has never run
      this compaction.
    """
    fwd_stream = torch.get_device_module(device).current_stream()
    verify_input: EagleVerifyInput = batch.spec_info
    record_stream_for_v2_verify(batch, verify_input, fwd_stream)

    bs = len(batch.seq_lens)

    # Batch 1: Target verify
    # Prepare for target verify in a separate stream
    with plan_stream_ctx:
        if plan_stream is not None:
            # Verify prep copies draft-produced tree metadata on the plan stream,
            # so it must not start before the draft frontier.
            plan_stream.wait_stream(fwd_stream)
        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input,
            req_to_token_pool,
            batch,
            target_worker,
        )

    # Cover post-prepare rebinds: draft_token, plan_stream-allocated out_cache_loc.
    record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

    # Correct some buffers due to the overlap plan
    if plan_stream:
        torch.get_device_module(device).current_stream().wait_stream(plan_stream)
        if (
            _is_npu
            and target_worker.model_runner.model_config.model_is_mrope
            and batch.spec_info is not None
            and getattr(batch.spec_info, "positions", None) is not None
            and not batch.forward_mode.is_idle()
        ):
            # mrope_position depends on draft output in default stream and is computed in plan stream,
            # causing errors. Compute it here for correct values.
            verify_forward_batch.compute_spec_mrope_positions(
                target_worker.model_runner, batch
            )

        # Some values such as custom_mask and position depend on the output of draft,
        # so the previous plan step used the wrong values. Here, we need to run the related
        # computation again to update them to the correct values.
        target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
            verify_input,
            (
                target_worker.model_runner.decode_cuda_graph_runner.bs
                if can_run_cuda_graph
                else None
            ),
        )

    # Must stay ahead of the target verify launch below.
    grammar_tree = (
        GrammarTree.from_device(
            verify_input.retrieve_next_token,
            verify_input.retrieve_next_sibling,
            verify_input.draft_token.view(verify_input.retrieve_next_token.shape),
        )
        if batch.has_grammar
        else None
    )

    if metadata_ready_pre_pad:
        # Multi-layer eagle preserved-verbatim behavior: metadata init is
        # skipped here unconditionally, although eagle_prepare_for_verify
        # only plans when cuda-graph load_batch ran. Single-layer eagle
        # re-inits the non-graph path instead (post-pad); multi-layer has
        # not adopted that fix. On NPU with --disable-cuda-graph, non-graph
        # verify needs metadata init in forward_extend (post-pad); only
        # mark ready for the cuda-graph path.
        if not _is_npu or can_run_cuda_graph:
            verify_forward_batch.mark_forward_metadata_ready()

    # Run target verify batch in the main compute stream (GPU compute).
    # Metadata init is skipped iff cuda-graph already ran load_batch —
    # eagle_prepare_for_verify marked the batch in exactly that case; the
    # non-cuda-graph path stays unmarked and gets forward_extend's init
    # (post-pad).
    forward_batch_output = target_worker.forward_batch_generation(
        batch=None,
        forward_batch=verify_forward_batch,
        is_verify=True,
        pp_proxy_tensors=pp_proxy_tensors,
    )
    logits_output = forward_batch_output.logits_output

    # Generate vocab mask for constrained decoding
    grammar_mask = None
    if batch.has_grammar:
        grammar_mask = build_grammar_vocab_mask(
            reqs=batch.reqs,
            tree=grammar_tree,
            sampling_info=batch.sampling_info,
            device=verify_input.retrieve_next_token.device,
            barrier=grammar_barrier,
        )

    # Sample
    maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
    maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
    (
        predict,
        accept_lens,
        accept_index,
    ) = eagle_sample(
        verify_input,
        batch,
        logits_output,
        grammar_mask,
        uno_target_max_top_k=uno_target_max_top_k,
        watermark_state=target_worker.model_runner.watermark_state,
        watermark_full_mask=(
            target_worker.model_runner.attn_backend.verify_mask is None
            or target_worker.model_runner.attn_backend.verify_mask.mode
            == TreeMaskMode.FULL_MASK
        ),
    )
    fused_commit_outputs = predict.is_cuda and not batch.forward_mode.is_idle()
    prepared_commit_steps = prepared_draft_inputs = None
    if fused_commit_outputs and topk == 1:
        track_interval = (
            mamba_track_grid(batch.tree_cache.page_size)
            if batch.mamba_track_indices is not None
            else 0
        )
        new_seq_lens, bonus_tokens, prepared_commit_steps, prepared_draft_inputs = (
            prepare_verify_commit_outputs(
                predict,
                accept_index,
                accept_lens,
                batch.seq_lens,
                num_draft_tokens=num_draft_tokens,
                mamba_track_interval=track_interval,
            )
        )
    elif fused_commit_outputs:
        new_seq_lens, bonus_tokens = prepare_verify_commit_outputs(
            predict, accept_index, accept_lens, batch.seq_lens
        )
    else:
        new_seq_lens = batch.seq_lens + accept_lens
    if batch.hisparse_coordinator is not None:
        batch.hisparse_coordinator.commit_spec_accept_tokens(
            batch=batch,
            accept_indices=accept_index,
        )
    clear_unaccepted_c128 = getattr(
        token_to_kv_pool_allocator.get_kvcache(),
        "clear_unaccepted_c128_draft_states",
        None,
    )
    if clear_unaccepted_c128 is not None and not batch.forward_mode.is_idle():
        clear_unaccepted_c128(
            batch.req_pool_indices,
            batch.seq_lens,
            accept_lens,
            num_draft_tokens,
        )

    # Update mamba state for hybrid GDN models after verification
    commit_mamba_states_after_verify(
        target_worker,
        batch,
        accept_lens,
        accept_index,
        num_draft_tokens,
        prepared_step_indices=prepared_commit_steps,
    )

    if not fused_commit_outputs and not batch.forward_mode.is_idle():
        accept_tokens = predict[accept_index]
        if (
            target_worker.model_runner.watermark_state is not None
            and batch.sampling_info.has_watermark_candidates
        ):
            target_worker.model_runner.watermark_state.append_speculative(
                batch.req_pool_indices,
                accept_tokens,
                accept_lens,
            )
        bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
        # stride = accept_tokens per-req width = accept_index.shape[1]
        # (spec_steps + 1); NOT num_draft_tokens, wrong for topk > 1 trees.
        fill_bonus_tokens_func(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_index.shape[1],
            bs,
        )
    elif batch.forward_mode.is_idle():
        bonus_tokens = torch.empty((0,), device=device, dtype=torch.int32)

    if batch.return_logprob and not batch.forward_mode.is_idle():
        compute_spec_logprobs(batch, logits_output, predict, accept_index=accept_index)

    if finalize_tree_path and not batch.forward_mode.is_idle() and topk > 1:
        # topk == 1 needs nothing here: the accepted path is already the front
        # chain, so the whole compaction is an identity transform.
        predict = _finalize_accept_tree_path(
            batch,
            accept_index,
            accept_lens,
            predict,
            logits_output,
            bs,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            num_draft_tokens=num_draft_tokens,
        )

    next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)

    # verify_forward_batch transitively holds verify-time GPU tensors
    # (draft_token / out_cache_loc / ...) that must outlive the imminent
    # batch.input_ids rebind in prepare_for_draft_extend.
    # Scheduler pins it in batch_record_buf for the 2-iter window.
    return GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=predict,
        can_run_cuda_graph=can_run_cuda_graph,
        speculative_num_draft_tokens=num_draft_tokens,
        next_draft_input=next_draft_input,
        accept_lens=accept_lens,
        accept_index=accept_index,
        new_seq_lens=new_seq_lens,
        routed_experts_output=forward_batch_output.routed_experts_output,
        indexer_topk_output=forward_batch_output.indexer_topk_output,
        extra_keep_alive_refs=[verify_forward_batch],
        prepared_draft_extend_inputs=prepared_draft_inputs,
    )
