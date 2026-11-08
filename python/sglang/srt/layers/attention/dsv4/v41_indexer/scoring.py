"""Index scores shared by the full top-k and both candidate schemes: the DeepGEMM
operands and flattened-K scores, the torch per-request scores, and the writes of
a selection back into pool slots."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    finish_paged_indexer_topk,
    fp4_index_logits_decode,
    fp4_index_logits_paged,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import index_q_rope_pack_weights
from sglang.kernels.ops.attention.dsv4.index_logits import flat_index_logits_tiles
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import async_h2d

from .types import DecodeInputs, PrefillInputs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

_DEEP_GEMM_SCORE_BUDGET_BYTES = 2 << 30


def _score_budget_bytes() -> int:
    """Per-tile budget for the DeepGEMM indexer logits, capped by free memory.

    Same rule the torch scorer already follows (mqa_logits_budget_bytes): the
    logits tile is the largest transient of a prefill, so a fixed 2 GiB budget
    - sized for 24-80 GB cards - is exactly the allocation that OOMs on a 16 GB
    card once the pool is nearly full. Measured on a 5060 Ti at 81% token
    usage: 420 MiB requested against 411 MiB free, killing the 250K prefill.
    """
    from sglang.srt.layers.attention.mqa_logits_utils import (
        mqa_logits_budget_bytes,
    )

    return min(
        _DEEP_GEMM_SCORE_BUDGET_BYTES,
        mqa_logits_budget_bytes(
            device_index=torch.cuda.current_device(), allow_sync=True
        ),
    )


class DeepGEMMDecodeData(msgspec.Struct, frozen=True):
    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool


class DeepGEMMPrefillData(msgspec.Struct, frozen=True):
    """The rows of one request are consecutive; a selection is a flattened-K
    column, request start plus compressed position."""

    k_slots: torch.Tensor  # [columns] index-K pool slot of each flattened-K column
    request_starts: torch.Tensor  # [rows] int32, first column of the row's request
    lens_per_request: List[int]  # compressed positions at each request's newest token
    rows_per_request: List[int]  # query rows of each request
    compress_lens: torch.Tensor  # [rows] int32, compressed positions the row sees
    q_fp4: torch.Tensor  # [rows, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, heads] int32, packed ue8m0
    weights: torch.Tensor  # [rows, heads] fp32 head weights

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]

    def empty_selection(self, topk: int) -> torch.Tensor:
        return self.request_starts.new_full(
            (self.num_rows, topk), -1, dtype=torch.int32
        )

    def write_page_indices(self, inputs: PrefillInputs) -> None:
        """The pool slots of the request-local ``inputs.out_raw_indices`` rows."""
        if inputs.out_page_indices is None:
            return
        raw = inputs.out_raw_indices[: self.num_rows]
        columns = (raw + self.request_starts[:, None]).clamp_min(0)
        inputs.out_page_indices[: self.num_rows, : raw.shape[1]] = torch.where(
            raw >= 0, self.k_slots[columns], -1
        )

    def write_selection(self, selected: torch.Tensor, inputs: PrefillInputs) -> None:
        """Write flattened-K columns ``selected`` into the selection buffers."""
        num_tokens, topk = selected.shape
        chosen = selected >= 0
        if inputs.out_page_indices is not None:
            inputs.out_page_indices[:num_tokens, :topk] = torch.where(
                chosen, self.k_slots[selected.clamp_min(0)], -1
            ).to(torch.int32)
        inputs.out_raw_indices[:num_tokens, :topk] = torch.where(
            chosen, selected - self.request_starts[:, None], -1
        )


def quantize_index_q(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor,
    )

    rows, heads = q.shape[0], q.shape[1]
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    return q_fp4.view(rows, heads, 64), q_sf.view(rows, heads)


def get_index_k_cache(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    layer_id: int,
    page_size: int,
) -> torch.Tensor:
    """The layer's index-K pool as ``[pages, page_size, 1, 68]`` uint8."""
    k_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
    assert k_cache.dim() == 2
    # fp4: 64 payload + 4 scale per slot
    return k_cache.view(k_cache.shape[0], page_size, 1, 68)


def _index_q_and_weights(
    *,
    indexer: DeepseekV41Indexer,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp4 index queries ``[rows, heads, 64]`` int8 with ``[rows, heads]`` packed
    ue8m0 scales, and fp32 head weights ``[rows, heads]``."""
    rows, heads = x.shape[0], indexer.n_local_heads
    q, _ = indexer.wq_b(q_lora)
    q_fp4, q_sf, weights = index_q_rope_pack_weights(
        q.view(rows, heads, indexer.index_head_dim),
        torch.view_as_real(freqs_cis).flatten(-2),
        positions,
        indexer.head_weights_raw(x),
        indexer.head_weight_scale,
    )
    return q_fp4.view(rows, heads, 64), q_sf.view(rows, heads), weights


def get_deep_gemm_decode_data(
    inputs: DecodeInputs, token_to_kv_pool: DeepSeekV4TokenToKVPool
) -> DeepGEMMDecodeData:
    indexer = inputs.indexer
    # The kernel sums head scores locally, so the indexer heads must be replicated.
    assert indexer.n_local_heads == indexer.n_heads
    q_fp4, q_sf, weights = _index_q_and_weights(
        indexer=indexer,
        x=inputs.x,
        q_lora=inputs.q_lora,
        freqs_cis=inputs.freqs_cis,
        positions=inputs.positions,
    )
    bs = q_fp4.shape[0]
    q_fp4 = q_fp4.view(bs, 1, indexer.n_local_heads, 64)
    q_sf = q_sf.view(bs, 1, indexer.n_local_heads)

    # Index pool page (64 slots); metadata.page_table is expanded to match.
    k_cache = get_index_k_cache(
        token_to_kv_pool=token_to_kv_pool,
        layer_id=inputs.layer_id,
        page_size=inputs.paged_metadata.compressed_page_size,
    )
    return DeepGEMMDecodeData(q_fp4, q_sf, weights, k_cache)


def get_deep_gemm_prefill_data(
    inputs: PrefillInputs, req_to_token: torch.Tensor
) -> Optional[DeepGEMMPrefillData]:
    """None when the chunk has no row or no visible compressed position."""
    ratio = inputs.compress_ratio
    pos = inputs.positions
    assert (
        inputs.seq_lens_cpu is not None
        and inputs.rows_per_request is not None
        and inputs.rows_per_request_device is not None
    ), "the DeepGEMM prefill indexer needs the CPU length vectors"
    device = pos.device
    # Visible compressed positions per request; (pos + 1) // ratio bounds each row.
    lens_per_request = [s // ratio for s in inputs.seq_lens_cpu]
    columns = sum(lens_per_request)
    num_tokens = pos.shape[0]
    if columns == 0 or num_tokens == 0:
        return None
    starts = list(itertools.accumulate(lens_per_request, initial=0))[:-1]
    lens = async_h2d(lens_per_request, dtype=torch.int64, device=device)
    starts_dev = async_h2d(starts, dtype=torch.int64, device=device)
    request = torch.repeat_interleave(
        torch.arange(len(lens_per_request), device=device), lens, output_size=columns
    )
    position = torch.arange(columns, device=device) - starts_dev[request]
    pool_rows = inputs.req_pool_indices.to(torch.int64)[request]
    q_fp4, q_sf, weights = _index_q_and_weights(
        indexer=inputs.indexer,
        x=inputs.x,
        q_lora=inputs.q_lora,
        freqs_cis=inputs.freqs_cis,
        positions=pos,
    )
    return DeepGEMMPrefillData(
        k_slots=req_to_token[pool_rows, position * ratio].to(torch.int64) // ratio,
        request_starts=torch.repeat_interleave(
            starts_dev.to(torch.int32),
            inputs.rows_per_request_device.to(torch.int64),
            output_size=num_tokens,
        ),
        lens_per_request=lens_per_request,
        rows_per_request=inputs.rows_per_request,
        compress_lens=((pos + 1) // ratio).to(torch.int32),
        q_fp4=q_fp4,
        q_sf=q_sf,
        weights=weights,
    )


def score_tiles(data: DeepGEMMPrefillData, kv: Tuple[torch.Tensor, torch.Tensor]):
    yield from flat_index_logits_tiles(
        q=(data.q_fp4, data.q_sf),
        kv=kv,
        weights=data.weights,
        starts=data.request_starts,
        lengths=data.compress_lens,
        context_lengths=data.lens_per_request,
        budget_bytes=_score_budget_bytes(),
    )


def topk_ragged(
    logits: torch.Tensor,
    lens: torch.Tensor,
    *,
    out: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    """Top-k of each row's first ``lens`` columns plus ``offsets`` into ``out``,
    ``-1`` padded, unordered: DeepSelect for bf16 logits, top-k v2 for fp32."""
    if logits.dtype == torch.bfloat16:
        from sglang.kernels.ops.attention.deep_select import topk

        topk(
            logits,
            out.shape[1],
            end=lens,
            output_idx=out,
            output_idx_offset=offsets,
            idx_oob_fill_value=-1,
        )
        return
    from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2

    topk_transform_ragged_v2(logits, lens, out_offsets=offsets, out_indices=out)


def dense_prefill_topk(
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    *,
    out: torch.Tensor,
) -> None:
    """Request-local compressed positions into ``out``, ``-1`` padded, unordered."""
    # Score columns are request-relative, so a zero offset emits request-local positions.
    zero_offsets = torch.zeros_like(data.request_starts)
    for tile, logits in score_tiles(data, kv):
        topk_ragged(
            logits, data.compress_lens[tile], out=out[tile], offsets=zero_offsets[tile]
        )
        # Free this tile's logits before the generator scores the next one.
        del logits


# ---------- torch ----------


# Hard cap on one bf16 [rows, heads, lc] score chunk; transients run ~3x this.
# The effective budget is also capped by the indexer's free-memory budget: the
# hardcoded 1 GiB tile is 768 MiB at lc~24K on a 24 GB card, where the KV pool
# owns most of the card and the transient peak (relu/mul copies) OOMs at 32K.
_TORCH_SCORE_BUDGET_BYTES = 1 << 30
# `scores` keeps more than the tile alive: the einsum result, an fp32 product of
# it, and the fp32 row sum. Sized 1:1 with the budget, a 1 GiB budget measured a
# 4.0x tile peak (3.1x with the in-place relu in `scores`) on a 3090 at lc 24K
# and 96K -- which is the OOM that only shows up deep into a long prompt. Rows
# are sized for the peak instead.
_SCORE_BUDGET_TO_TILE_DIVISOR = 3


def _torch_score_budget_bytes() -> int:
    from sglang.srt.layers.attention.mqa_logits_utils import mqa_logits_budget_bytes

    return min(
        _TORCH_SCORE_BUDGET_BYTES,
        mqa_logits_budget_bytes(
            device_index=torch.cuda.current_device(), allow_sync=True
        ),
    )


class RequestScores(msgspec.Struct, frozen=True):
    lc: int  # compressed positions visible at its newest row
    k: int
    columns: torch.Tensor  # arange(lc)
    tok: torch.Tensor  # [rows_b] its query rows in the chunk
    lens: torch.Tensor  # [rows_b] compressed positions each row sees
    slots: torch.Tensor  # [lc] index-K pool slot of each position


class ChunkScores(msgspec.Struct, frozen=True):
    rows: slice  # into the request's rows
    tok: torch.Tensor
    lens: torch.Tensor
    scores: torch.Tensor  # [rows, lc] bf16, -inf past each row's length


class DecodeScores(msgspec.Struct, frozen=True):
    bs: int
    lmax: int
    lens: torch.Tensor  # [bs]
    slots: torch.Tensor  # [bs, lmax]
    scores: torch.Tensor  # [bs, lmax]


def prefill_requests(
    *,
    inputs: PrefillInputs,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
) -> Iterator[tuple[RequestScores, Iterator[ChunkScores]]]:
    """Requests with nothing visible yet are skipped."""
    pool = token_to_kv_pool
    ratio = inputs.compress_ratio
    indexer = inputs.indexer
    req, pos = inputs.req_rows, inputs.positions
    assert req is not None, "the torch indexer needs the batch's rows"
    inputs.reset_outputs()
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    # A compressed position is visible once the query has passed its last token.
    compress_lens = (pos + 1) // ratio
    topk = indexer.index_topk
    for r in torch.unique_consecutive(req).tolist():
        tok = (req == r).nonzero().squeeze(1)
        lens = compress_lens[tok]
        lc = int(lens.max().item())
        if lc == 0:
            continue
        j = torch.arange(lc, device=pos.device)
        slots = req_to_token[r, j * ratio].to(torch.int64) // ratio
        # Dequantize only this request's visible K rows; the table is pool-sized.
        index_k = pool.get_low_ratio_index_k_dequant(inputs.layer_id, slots)
        request = RequestScores(
            lc=lc, k=min(topk, lc), columns=j, tok=tok, lens=lens, slots=slots
        )
        yield (
            request,
            _score_chunks(
                indexer=indexer, q=q, weights=weights, index_k=index_k, request=request
            ),
        )


def _score_chunks(
    *,
    indexer: DeepseekV41Indexer,
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    request: RequestScores,
) -> Iterator[ChunkScores]:
    lc, j = request.lc, request.columns
    # Chunk rows so the [rows, heads, lc] bf16 scores stay under the budget,
    # capped by current free memory (the KV pool already owns most of it). The
    # budget is what `scores` may hold, not the tile alone: `scores` keeps the
    # einsum result plus an fp32 product of it, so a tile budgeted 1:1 turns into
    # ~3x the budget in peak allocation -- measured 4.0x before the in-place relu
    # (3.1x after) at lc 24K and 96K on a 3090, and that is what OOMs deep into a
    # long prompt while the same shape fits at the start of it.
    rows_per_chunk = max(
        1,
        _torch_score_budget_bytes()
        // (q.shape[1] * lc * 2 * _SCORE_BUDGET_TO_TILE_DIVISOR),
    )
    for start in range(0, request.tok.numel(), rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        tok_c, lens_c = request.tok[rows], request.lens[rows]
        s = indexer.scores(q[tok_c], index_k, weights[tok_c])
        s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)
        yield ChunkScores(rows=rows, tok=tok_c, lens=lens_c, scores=s)


def write_prefill(
    inputs: PrefillInputs,
    request: RequestScores,
    chunk: ChunkScores,
    idx: torch.Tensor,
) -> None:
    k, lc = request.k, request.lc
    idx = idx.sort(dim=-1).values
    reach = idx < chunk.lens[:, None]
    if inputs.out_page_indices is not None:
        inputs.out_page_indices[chunk.tok, :k] = torch.where(
            reach, request.slots[idx.clamp_max(lc - 1)], -1
        ).to(torch.int32)
    inputs.out_raw_indices[chunk.tok, :k] = torch.where(reach, idx, -1).to(torch.int32)


class PagedDecodeScores(msgspec.Struct, frozen=True):
    """Paged decode scores with graph-stable capacity ``lmax``.

    Only positions below each device-side ``lens`` are initialized. Top-k
    consumers must respect those lengths. With ``has_candidate_mask``, masked
    positions have -inf scores and must be discarded during selection writeback.
    """

    bs: int
    lmax: int
    lens: torch.Tensor
    scores: torch.Tensor
    req: torch.Tensor
    req_to_token: torch.Tensor
    ratio: int
    plan: Optional[torch.Tensor]  # None when the caller skips the top-k v2 plan
    has_candidate_mask: bool


def decode_scores(
    *,
    inputs: DecodeInputs,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    compute_topk_plan: bool = True,
) -> Optional[DecodeScores | PagedDecodeScores]:
    """None when there is no row, or nothing visible yet."""
    pool = token_to_kv_pool
    ratio = inputs.compress_ratio
    indexer = inputs.indexer
    req, pos = inputs.req_rows, inputs.positions
    paged = get_platform().is_sm90
    bs = req.shape[0]
    assert pos.shape[0] == bs, (
        f"decode expects one token per request, {pos.shape=} {bs=}"
    )
    if bs == 0:
        inputs.reset_outputs()
        return None
    lens = (pos + 1) // ratio
    metadata = inputs.paged_metadata
    # V4 reserves the replay bound in metadata; visibility stays on device.
    # A capture-time length read would both synchronize and truncate replay.
    lmax = min(metadata.max_compressed_seq_len, req_to_token.shape[1] // ratio)
    if lmax == 0:
        inputs.reset_outputs()
        return None
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    table = pool.get_index_k_with_scale_buffer(inputs.layer_id)
    if paged:
        lens = lens.to(torch.int32)
        # Prepare the device-side plan before scoring, away from its top-k consumer.
        plan = plan_topk_v2(lens) if compute_topk_plan else None
        scores = fp4_index_logits_paged(
            q,
            weights,
            req,
            req_to_token,
            lens,
            table,
            table.shape[1] // 68,
            lmax,
            ratio,
            candidate_mask,
        )
        return PagedDecodeScores(
            bs=bs,
            lmax=lmax,
            lens=lens,
            scores=scores,
            req=req,
            req_to_token=req_to_token,
            ratio=ratio,
            plan=plan,
            has_candidate_mask=candidate_mask is not None,
        )
    inputs.reset_outputs()
    j = torch.arange(lmax, device=pos.device)
    valid = j[None, :] < lens[:, None]
    slots = req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
    slots = slots.masked_fill(~valid, 0)
    scores = fp4_index_logits_decode(
        q, weights, slots, lens, table, table.shape[1] // 68
    )
    return DecodeScores(bs=bs, lmax=lmax, lens=lens, slots=slots, scores=scores)


def select_decode(
    inputs: DecodeInputs, d: DecodeScores | PagedDecodeScores, topk: int
) -> None:
    k = min(topk, d.lmax)
    if isinstance(d, PagedDecodeScores):
        assert d.plan is not None, "paged select_decode needs the top-k v2 plan"
        idx = torch.empty((d.bs, k), dtype=torch.int32, device=d.scores.device)
        topk_transform_paged_v2(d.scores, d.lens, None, idx, 1, d.plan)
        finish_paged_indexer_topk(
            idx,
            d.scores,
            d.lens,
            d.req,
            d.req_to_token,
            inputs.out_page_indices,
            None,
            d.ratio,
            d.has_candidate_mask,
        )
    else:
        write_decode(inputs, d, d.scores.topk(k, dim=-1, sorted=False).indices)


def write_decode(inputs: DecodeInputs, d: DecodeScores, idx: torch.Tensor) -> None:
    k = idx.shape[1]
    idx = idx.sort(dim=-1).values
    reach = idx < d.lens[:, None]
    inputs.out_page_indices[: d.bs, :k] = torch.where(
        reach, d.slots.gather(1, idx.clamp_max(d.lmax - 1)), -1
    ).to(torch.int32)

# ---------------------------------------------------------------------------
# fp4 block path: block ids + the fine top-k, both inside a bounded window.
#
# The torch path above dequantizes every visible candidate and scores
# ``[rows, lc]``; the packed fp4 payload is 68 bytes per candidate, so the
# block maxima can come from a Triton kernel and only the selected blocks are
# ever decoded. Same math, no ``rows x lc`` tensor.
# ---------------------------------------------------------------------------

_FP4_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fp4_dequant_rows(payload_u8: torch.Tensor, scales_u8: torch.Tensor) -> torch.Tensor:
    """``(*, 64)`` uint8 + ``(*, 4)`` uint8 -> ``(*, 128)`` fp32, reference math.

    Decoded arithmetically from uint8 nibbles: a table indexed through int64
    codes costs 8x the bytes of the payload tile and was itself an OOM source.
    """
    lo = payload_u8 & 0x0F
    hi = (payload_u8 >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)          # uint8, 2 per byte
    m = (codes & 0x7).to(torch.float32)
    mag = torch.where(
        m < 2,
        m * 0.5,
        torch.exp2(torch.floor(m * 0.5) - 1.0) * (1.0 + 0.5 * (m - 2.0 * torch.floor(m * 0.5))),
    )
    val = torch.where((codes & 0x8) != 0, -mag, mag)
    scale = torch.exp2(scales_u8.to(torch.float32) - 127.0)
    return val * scale.repeat_interleave(32, dim=-1)


def _score_rows_against(q, k, w):
    """``sum_h relu(q[n,h,:] . k[n,c,:]) * w[n,h]`` -> ``[n, c]`` fp32."""
    s = torch.einsum("nhd,ncd->nhc", q.float(), k)
    return (s.clamp_min(0) * w[:, :, None]).sum(1)


def prefill_fp4_block_select(
    *,
    inputs,
    token_to_kv_pool,
    req_to_token,
    block_size: int,
    topk_blocks: int,
    rows_per_chunk: int = 128,
    blocks: Optional[torch.Tensor] = None,
):
    """Per request-row chunk: candidate block ids + the fine top-k indices.

    Yields ``(request, tok, lens, block_ids, fine_idx)``; ``block_ids`` is
    ``[rows, min(topk_blocks, n_blocks)]`` int32 with ``-1`` padding, ``fine_idx``
    is ``[rows, k]`` int32 candidate indices (already remapped, sorted by the
    caller's ``write_prefill``).

    With ``blocks`` the block selection is already published (a consumer layer
    reusing the source layer's metadata): the O(rows x lc) block scoring is
    skipped and only the fine top-k inside those blocks runs.
    """
    from sglang.kernels.ops.attention.dsv4.triton_fp4_block_scores import (
        triton_fp4_block_scores,
    )

    pool = token_to_kv_pool
    ratio = inputs.compress_ratio
    indexer = inputs.indexer
    req, pos = inputs.req_rows, inputs.positions
    assert req is not None, "the fp4 indexer needs the batch's rows"
    inputs.reset_outputs()
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    compress_lens = (pos + 1) // ratio
    topk = indexer.index_topk
    for r in torch.unique_consecutive(req).tolist():
        tok = (req == r).nonzero().squeeze(1)
        lens = compress_lens[tok]
        lc = int(lens.max().item())
        if lc == 0:
            continue
        j = torch.arange(lc, device=pos.device)
        slots = req_to_token[r, j * ratio].to(torch.int64) // ratio
        payload, scales = pool.get_low_ratio_index_k_fp4(inputs.layer_id, slots)
        pay_u8 = payload.view(torch.uint8).reshape(lc, 64)
        sca_u8 = scales.view(torch.uint8).reshape(lc, 4)
        n_blocks = (lc + block_size - 1) // block_size
        kk = min(topk_blocks, n_blocks) if blocks is None else int(blocks.shape[1])
        request = RequestScores(
            lc=lc, k=min(topk, lc), columns=j, tok=tok, lens=lens, slots=slots
        )
        for start in range(0, tok.numel(), rows_per_chunk):
            rows = slice(start, min(start + rows_per_chunk, tok.numel()))
            tok_c, lens_c = request.tok[rows], request.lens[rows]
            n = int(tok_c.numel())
            if n == 0:
                continue
            if blocks is None:
                starts = torch.zeros(n, dtype=torch.int32, device=pos.device)
                ends = lens_c.to(torch.int32)
                blk = triton_fp4_block_scores(
                    q[tok_c], pay_u8, sca_u8, weights[tok_c], starts, ends,
                    n_blocks, block_size,
                )
                last = ((lens_c - 1) // block_size).to(torch.int64)
                blk = blk.masked_fill(
                    torch.arange(n_blocks, device=blk.device)[None, :] == last[:, None],
                    torch.inf,
                )
                top = blk.topk(kk, dim=-1)
                block_ids = top.indices.to(torch.int32).masked_fill_(
                    ~(top.values > -torch.inf), -1
                )
            else:
                block_ids = blocks[tok_c]
            # Exact top-k inside *every* published block -- the same candidate
            # set the dense path scores, so the selection cannot drift from it.
            # A trimmed tile (ceil(k/block) + 2 blocks) is tempting but wrong:
            # at k=512 a 6-block tile disagreed with the dense top-k on roughly
            # half the rows (measured with LK_INDEX_SELFTEST). What bounds the
            # tile here is the row slicing below, not a block trim.
            take = min(request.k, lc)
            if kk <= 0:
                continue
            sel_ids = block_ids
            sel = sel_ids.clamp_min(0).to(torch.int64)
            cand = sel.unsqueeze(-1) * block_size + torch.arange(
                block_size, device=pos.device
            )[None, None, :]
            flat = cand.reshape(n, kk * block_size)
            in_win = flat < lens_c[:, None]
            # A -1 block is padding, not block 0: drop its lanes so they cannot
            # take slots that a real candidate should own.
            in_win &= (sel_ids >= 0).repeat_interleave(block_size, dim=1)
            per_row = max(1, kk * block_size * q.shape[-1] * 4)
            step = max(1, min(n, 32_000_000 // per_row))
            for r0 in range(0, n, step):
                sl = slice(r0, min(r0 + step, n))
                # The tail block runs past `lc`; those lanes are masked out of
                # the scores, but the gather itself must stay in range.
                gather_idx = flat[sl].clamp_max(lc - 1)
                k_rows = _fp4_dequant_rows(pay_u8[gather_idx], sca_u8[gather_idx])
                fine = _score_rows_against(q[tok_c[sl]], k_rows, weights[tok_c[sl]])
                fine = fine.masked_fill(~in_win[sl], -torch.inf)
                top = fine.topk(min(take, fine.shape[-1]), dim=-1, sorted=False)
                # A row whose window holds fewer than k candidates must leave
                # the extra slots empty (write_prefill maps anything >= lens to
                # -1); repeating a candidate would reweight it in the attention.
                picked = flat[sl].gather(1, top.indices)
                picked = picked.masked_fill(top.values == -torch.inf, lc)
                yield (
                    request, tok_c[sl], lens_c[sl], block_ids[sl],
                    picked.to(torch.int32),
                )
