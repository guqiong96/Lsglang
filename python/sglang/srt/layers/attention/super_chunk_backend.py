"""Serial N-way token-chunk iterator for EXTEND forwards (super-chunk prefill).

Part of the MoE-staging-once-per-layer work (opencode/special/moe-once-per-layer):
one prefill forward carries a whole super-chunk (the scheduler was allowed to
raise its per-forward token cap to CAP), while the attention component keeps
its old 8192-token footprint by looping sub-chunks internally. MoE is
token-order independent, so it sees the full super-chunk in a single
gpu_prefill call (H2D staging once, its internal sub-M loop covers the rest).

Design = TboAttnBackend generalized from 2 overlapped children to N serial
sub-chunks:
  - one child backend per sub-chunk (same class as primary) so plan-based
    backends (flashinfer) hold a private plan per sub-chunk;
  - child ForwardBatch views are built once per forward from token-axis cut
    points (filter_batch-style), requests cut mid-sequence get
    prefix_lens += consumed_so_far;
  - forward() slices q/k/v on the token axis, runs each child's kernel on its
    sub-chunk, concatenates. Same stream everywhere: sub-chunk c+1 attends to
    KV written by sub-chunk c's kv-cache stores, ordered by stream.

Capability gate: any condition outside the supported envelope (spec decode,
TBO, DP-attention padding, cuda-graph capture, single sub-chunk) falls back to
plain primary-forward, i.e. today's behaviour (safe side).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
)
from sglang.srt.utils import logger

# One line per process, not per wrapper instance: the wrapper is built per
# layer, so a per-instance guard prints the same line once per layer per rank.
_SC_ACTIVE_LOGGED = False
_SC_WARNED_REASONS = set()

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _super_chunk_unsupported_reason(
    fb: ForwardBatch, inner_size: int, allow_spec_info_extend: bool = False
) -> Optional[str]:
    if inner_size <= 0:
        return "disabled (super_chunk_inner_size=0)"
    if not fb.forward_mode.is_extend():
        return "not an EXTEND forward"
    if fb.forward_mode.is_mixed():
        # MIXED carries decode rows alongside the extend rows, but the child
        # views only slice the extend axis (extend_num_tokens); the decode
        # tail would fall through every child. Not seen in our configs
        # (is_mixed_chunk off), kept as a hard gate like the others.
        return "mixed extend+decode batch"
    if torch.cuda.is_current_stream_capturing():
        return "cuda-graph capture"
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    if is_in_breakable_cuda_graph():
        # Same exclusion the model-side loops apply (deepseek_v4 hc/attn
        # plans): a graph piece replaying wrapper.forward must not depend on
        # live-iterator state.
        return "breakable cuda-graph piece"
    if getattr(fb, "tbo_children", None) is not None:
        return "two-batch overlap"
    if fb.spec_info is not None:
        # Draft-extend forwards carry an EagleDraftInput on an EXTEND batch;
        # the model reads it once per forward (dsa seed capture), never per
        # token, so the token loop is still exact there.
        if not allow_spec_info_extend:
            return "speculative decoding"
    if getattr(fb, "global_num_tokens_cpu", None) is not None:
        return "dp-attention mlp sync"
    dsv4_loc = fb.out_cache_loc_dsv4
    if dsv4_loc is not None:
        # Every packed pool loc is a per-extend-token row; the child builder
        # slices all four on the token axis. Anything else-shaped (e.g. an
        # allocator variant that packs per-page) stays monolithic.
        num_tok = fb.extend_num_tokens
        if num_tok is None or any(
            not torch.is_tensor(getattr(dsv4_loc, f))
            or getattr(dsv4_loc, f).shape[0] != num_tok
            for f in ("out_full_loc", "out_swa_loc", "out_c4_loc", "out_c128_loc")
        ):
            return "dsv4 packed out_cache_loc (unshapable)"
    num = fb.extend_num_tokens
    if num is None or num <= inner_size:
        return "single sub-chunk"
    # Defense-in-depth: no child may straddle a sequence boundary (that is the
    # Xid-43 crash shape). compute_super_chunk_token_bounds cuts at the
    # sequence boundaries, so this only fires when it cannot know them
    # (extend_seq_lens_cpu missing, or its sum disagreeing with the token
    # axis while the batch packs several requests).
    ext_lens = getattr(fb, "extend_seq_lens_cpu", None)
    if ext_lens is not None and len(ext_lens) > 1:
        boundaries = []
        cum = 0
        for el in ext_lens[:-1]:
            cum += int(el)
            boundaries.append(cum)
        if any(
            lo < b < hi
            for b in boundaries
            for lo, hi in compute_super_chunk_token_bounds(fb, inner_size)
        ):
            return "child would straddle a sequence boundary"
    return None


def compute_super_chunk_token_bounds(
    fb: ForwardBatch, inner_size: int
) -> List[Tuple[int, int]]:
    """Token-axis [lo, hi) bounds, cut at multiples of inner_size AND at every
    sequence boundary (cumulative extend_seq_lens_cpu).

    The scheduler sizes EXTEND batches to CAP=LK_GPU_PREFILL_SUB_M, so one
    forward can pack several requests; a grid-only cut then puts a seam
    inside one request's rows while the same child covers the next request's
    head, and that child view misdescribes its rows: the attention kernels
    trap (Xid 43, scheduler SIGABRT). Cutting at grid UNION sequence
    boundaries makes every child lie within exactly one sequence; short
    remainder children are fine (single-sequence slices are supported at any
    length). When extend_seq_lens_cpu is unavailable the grid-only cut is
    kept (no allocation here); the gate in _super_chunk_unsupported_reason is
    what disables the wrapper for batches whose children would straddle.
    """
    total = fb.extend_num_tokens
    cuts = {0, total}
    grid = inner_size
    while grid < total:
        cuts.add(grid)
        grid += inner_size
    ext_lens = getattr(fb, "extend_seq_lens_cpu", None)
    if ext_lens is not None and sum(int(el) for el in ext_lens) == total:
        cum = 0
        for el in ext_lens:
            cum += int(el)
            if 0 < cum < total:
                cuts.add(cum)
    pts = sorted(cuts)
    return list(zip(pts, pts[1:]))


def build_super_chunk_children(
    fb: ForwardBatch, bounds: List[Tuple[int, int]]
) -> List[ForwardBatch]:
    """Per-sub-chunk ForwardBatch views. Sequential token layout: request r
    owns [cum[r], cum[r+1]). With the sequence-aware cuts in
    compute_super_chunk_token_bounds each child lies within exactly one
    sequence (batch_size 1, possibly mid-sequence with a grown prefix); the
    run-scan below still handles multi-request children generically."""
    ext_lens = fb.extend_seq_lens_cpu
    pref_lens = fb.extend_prefix_lens_cpu
    assert ext_lens is not None and pref_lens is not None

    cum = [0]
    for el in ext_lens:
        cum.append(cum[-1] + el)

    children = []
    r0 = 0
    for child_index, (lo, hi) in enumerate(bounds):
        # requests overlapping [lo, hi): r in [r1, r_end), monotone in chunk
        # index because tokens lay out request-contiguous
        r1 = r0
        while r1 < len(ext_lens) and cum[r1 + 1] <= lo:
            r1 += 1
        r_end = r1
        while r_end < len(ext_lens) and cum[r_end] < hi:
            r_end += 1
        assert r_end > r1, "token cut produced an empty request run"

        child = copy.copy(fb)
        child.tbo_children = None
        child.tbo_parent_token_range = None
        child.tbo_padded_len = None
        child.spec_info = None
        child.forward_metadata_ready = False
        # Routes wrapper.forward back to this child's backend without
        # re-entering the token loop: model layers that project per token
        # (qkv / GDN in_proj) use super_chunk_token_plan() to move their
        # projections inside the loop and then call in with the child view.
        child.super_chunk_child_index = child_index

        tok = slice(lo, hi)
        child.input_ids = fb.input_ids[tok] if fb.input_ids is not None else None
        child.positions = fb.positions[tok] if fb.positions is not None else None
        child.out_cache_loc = fb.out_cache_loc[tok]
        if fb.out_cache_loc_virtual is not None:
            child.out_cache_loc_virtual = fb.out_cache_loc_virtual[tok]
        dsv4_loc = fb.out_cache_loc_dsv4
        if dsv4_loc is not None:
            # The four packed pool locs are per-extend-token rows (the gate
            # in _super_chunk_unsupported_reason checked the shape); each
            # child writes exactly its own [lo, hi) slice of every pool.
            import dataclasses

            child.out_cache_loc_dsv4 = dataclasses.replace(
                dsv4_loc,
                out_full_loc=dsv4_loc.out_full_loc[tok],
                out_swa_loc=dsv4_loc.out_swa_loc[tok],
                out_c4_loc=dsv4_loc.out_c4_loc[tok],
                out_c128_loc=dsv4_loc.out_c128_loc[tok],
            )

        req = slice(r1, r_end)
        child.batch_size = r_end - r1
        child.req_pool_indices = fb.req_pool_indices[req]

        # per-request chunk-local extend/prefix: overlap length & prefix-so-far
        child_ext = []
        child_pref = []
        for r in range(r1, r_end):
            ov_lo = max(lo, cum[r])
            ov_hi = min(hi, cum[r + 1])
            child_ext.append(ov_hi - ov_lo)
            child_pref.append(pref_lens[r] + (ov_lo - cum[r]))

        dev = fb.req_pool_indices.device
        child.extend_seq_lens_cpu = child_ext
        child.extend_prefix_lens_cpu = child_pref

        # CPU-side track lists (lk_moe CPU-prefill metadata path): the parent
        # decided "this request tracks at seqlen B in this chunk", but B lands
        # in exactly ONE sub-chunk's window. Without the range filter every
        # child would snap, and a child whose window ends before B computes an
        # h row index past its own chunk grid (device-side OOB).
        parent_cpu_mask = getattr(fb, "mamba_prefill_track_mask_cpu", None)
        parent_cpu_trk = getattr(fb, "mamba_track_seqlens_cpu", None)
        if parent_cpu_trk is not None:
            child.mamba_track_seqlens_cpu = [parent_cpu_trk[r] for r in range(r1, r_end)]
        if parent_cpu_mask is not None:
            assert parent_cpu_trk is not None, (
                "cpu-prefill track metadata requires mamba_track_seqlens_cpu"
            )
            child.mamba_prefill_track_mask_cpu = [
                bool(parent_cpu_mask[r])
                and child_pref[i] < parent_cpu_trk[r] <= child_pref[i] + child_ext[i]
                for i, r in enumerate(range(r1, r_end))
            ]
        child.extend_seq_lens = torch.tensor(child_ext, dtype=torch.int32, device=dev)
        child.extend_prefix_lens = torch.tensor(
            child_pref, dtype=torch.int32, device=dev
        )
        child.extend_start_loc = torch.cumsum(
            child.extend_seq_lens, dim=0, dtype=torch.int32
        ) - child.extend_seq_lens
        child.extend_num_tokens = hi - lo

        child_seq_lens = child.extend_prefix_lens + child.extend_seq_lens
        child.seq_lens = child_seq_lens.to(fb.seq_lens.dtype)
        child.seq_lens_cpu = child_seq_lens.to("cpu", non_blocking=True)
        child.seq_lens_sum = int(sum(child_pref) + sum(child_ext))

        # Mamba/PLE track bookkeeping: the scheduler snapshots the state at
        # the END of a request's new tokens, so only the sub-chunk containing
        # that end may fire (see schedule_batch's track_entry calculation).
        # Per-sub-chunk local offsets stay exact because child prefixes carry
        # the consumed amount; later sub-chunks must not re-snap (their local
        # aligned offset would point past the request's tokens).
        ended_in_chunk = None
        parent_mask = getattr(fb, "mamba_track_mask", None)
        if parent_mask is not None:
            ended_in_chunk = torch.tensor(
                [cum[r + 1] <= hi for r in range(r1, r_end)],
                dtype=torch.bool,
                device=parent_mask.device,
            )
        for field in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            value = getattr(fb, field, None)
            if value is None:
                continue
            value = value[r1:r_end]
            if field == "mamba_track_mask" and ended_in_chunk is not None:
                value = value & ended_in_chunk
            setattr(child, field, value)

        children.append(child)
        r0 = r1

    return children


class SuperChunkAttnBackend(AttentionBackend):
    """Wraps a full attention backend; EXTEND forwards bigger than
    inner_size run their attention as serial sub-chunk passes."""

    # kwargs whose value carries a token axis, mapped to the axis to slice:
    # topk rows are per-token ([tokens, ...]); the GDN output buffer is
    # [1, tokens, heads, dim]; q_rope/k_rope are the MLA rope split of q/k
    # (dsv2-MLA dispatch, e.g. GLM5-Next). Anything else forces the safe
    # passthrough.
    _SLICEABLE_KWARGS = {
        "topk_indices": 0,
        "linear_attn_output": 1,
        "q_rope": 0,
        "k_rope": 0,
    }
    # Per-forward scalars/whole-batch tensors: identical for every chunk,
    # forwarded verbatim (dsv4's compress_ratio selects the KV pool per layer;
    # attn_sink is a per-request MLA sink, not a per-token row; cos_sin_cache
    # is the full rope table, is_neox/llama_4_scaling/sinks are per-layer).
    _PASSTHROUGH_KWARGS = frozenset(
        {
            "compress_ratio",
            "attn_sink",
            "cos_sin_cache",
            "is_neox",
            "llama_4_scaling",
            "sinks",
        }
    )

    def __init__(
        self,
        primary: AttentionBackend,
        inner_size: int,
        creator=None,
        allow_spec_info_extend: bool = False,
    ):
        super().__init__()
        self.primary = primary
        self.inner_size = inner_size
        self._creator = creator
        self._allow_spec_info_extend = allow_spec_info_extend
        self._mtp_shared_state = None
        self._child_bounds: Optional[List[Tuple[int, int]]] = None
        self.token_to_kv_pool = primary.token_to_kv_pool
        self.req_to_token_pool = primary.req_to_token_pool
        self.kv_index_translator = primary.kv_index_translator
        self.extend_dummy_seqs_capped_by_req_pool = getattr(
            primary, "extend_dummy_seqs_capped_by_req_pool", False
        )
        self._child_backends: List[AttentionBackend] = []
        self._child_batches: Optional[List[ForwardBatch]] = None
        # per-forward plan: None => passthrough (primary runs the whole batch)
        self._active = False
        self._warned_kwargs = set()
        self._warned_unknown_kwargs = set()
        self._warned_reasons = set()
        self._dbg_active_logged = False

    @classmethod
    def init_new(
        cls,
        creator: Callable[[], AttentionBackend],
        inner_size: int,
        allow_spec_info_extend: bool = False,
    ):
        return cls(
            primary=creator(),
            inner_size=inner_size,
            creator=creator,
            allow_spec_info_extend=allow_spec_info_extend,
        )

    def _ensure_child(self, idx: int) -> AttentionBackend:
        while len(self._child_backends) <= idx:
            assert self._creator is not None, (
                "SuperChunkAttnBackend built without a creator; use init_new()"
            )
            child = self._creator()
            state = getattr(self, "_mtp_shared_state", None)
            if state is not None:
                child.set_mtp_shared_sparse_indices(state)
            self._child_backends.append(child)
        return self._child_backends[idx]

    def super_chunk_child_backend(self, index: int) -> AttentionBackend:
        """The child backend of an active plan forward (indexer-side entry:
        metadata/pool bookkeeping must resolve per sub-chunk, not through the
        wrapper, so layers running their loop call this per segment)."""
        return self._ensure_child(index)

    # QSA MTP-shared sparse indices: every backend that may serve a forward
    # needs the state (children included), else per-chunk capture silently
    # no-ops and MTP decode reuses a stale selection.
    def set_mtp_shared_sparse_indices(self, state) -> None:
        self._mtp_shared_state = state
        self.primary.set_mtp_shared_sparse_indices(state)
        for child in self._child_backends:
            child.set_mtp_shared_sparse_indices(state)

    def _primary_in_late_layer_tail(self) -> bool:
        """True while the model runs decoder SWA bounded replay late layers.

        enter_late_layer_tail swaps the primary's full-row metadata for the
        tail metadata (identity check: no extra state on the primary). In that
        window the model hands the backend tail-gathered rows, which no
        token-axis child view can reproduce; forward() delegates to primary
        and model-side plans stop (their rows no longer match the plan).
        """
        md = getattr(self.primary, "forward_metadata", None)
        tail = getattr(self.primary, "tail_forward_metadata", None)
        return md is not None and md is tail

    def super_chunk_plan(self):
        """[(lo, hi, child_forward_batch)] for the active forward, else None.

        Model layers whose projections produce token-sized intermediates
        (qkv / gate, GDN in_proj) call this, run their per-token math on the
        slice and re-enter forward with the child view (marked with
        super_chunk_child_index); that keeps every projection inside the
        inner-loop footprint. Layers that only know q/k/v tensors keep the
        wrapper-level loop below, which slices what it is handed.
        """
        if not self._active or self._child_batches is None:
            return None
        if self._primary_in_late_layer_tail():
            return None
        return [
            (lo, hi, self._child_batches[i])
            for i, (lo, hi) in enumerate(self._child_bounds)
        ]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        reason = _super_chunk_unsupported_reason(
            forward_batch, self.inner_size, self._allow_spec_info_extend
        )
        if reason is not None:
            self._active = False
            self._child_batches = None
            self._child_bounds = None
            if reason not in _SC_WARNED_REASONS:
                _SC_WARNED_REASONS.add(reason)
                logger.warning(
                    "[sc-dbg] disabled reason=%s rows=%s inner=%s",
                    reason,
                    forward_batch.extend_num_tokens,
                    self.inner_size,
                )
            self.primary.init_forward_metadata(forward_batch=forward_batch)
            return

        bounds = compute_super_chunk_token_bounds(forward_batch, self.inner_size)
        self._child_bounds = bounds
        self._child_batches = build_super_chunk_children(forward_batch, bounds)
        for i, child_fb in enumerate(self._child_batches):
            self._ensure_child(i).init_forward_metadata(child_fb)
        # Primary metadata too: the per-call safe-side fallback replays the
        # whole forward on primary and must find its metadata in place.
        # Decoder SWA bounded replay builds a tail metadata on the primary as
        # well; it is NOT a reason to disable the wrapper. The projection
        # loops run over the full super-chunk for the early layers, then the
        # model swaps onto the tail at late_layer_start: super_chunk_plan()
        # goes None and forward() delegates to the primary (tail-row q cannot
        # be sliced by token-axis child views).
        self.primary.init_forward_metadata(forward_batch=forward_batch)
        self._active = True
        global _SC_ACTIVE_LOGGED
        if not _SC_ACTIVE_LOGGED:
            _SC_ACTIVE_LOGGED = True
            logger.info(
                "[sc-dbg] wrapper active rows=%s chunks=%d inner=%s",
                forward_batch.extend_num_tokens,
                len(bounds),
                self.inner_size,
            )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        layer=None,
        forward_batch: Optional[ForwardBatch] = None,
        save_kv_cache: bool = True,
        *,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self._primary_in_late_layer_tail():
            # Decoder SWA bounded replay tail: q/k/v are the tail-gathered
            # rows and the primary already holds the matching metadata.
            return self.primary.forward(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache,
                **(
                    {"mixed_qkv": mixed_qkv, "a": a, "b": b}
                    if mixed_qkv is not None
                    else {}
                ),
                **kwargs,
            )
        child_index = getattr(forward_batch, "super_chunk_child_index", None)
        if child_index is not None:
            # Model-side projection loop (super_chunk_plan): the tensors are
            # already this child's slice; run its kernel directly.
            return self._ensure_child(child_index).forward(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(
                    {"mixed_qkv": mixed_qkv, "a": a, "b": b}
                    if mixed_qkv is not None
                    else {}
                ),
                **kwargs,
            )
        # Linear (mixed_qkv) and QSA (topk_indices) entries join the token-axis
        # loop too: GDN advances its state through the mamba pool exactly like
        # today's scheduler chunks do (child == one chunk window), and QSA's
        # per-token selections slice with the tokens they were computed for.
        unknown = [
            key
            for key in kwargs
            if key not in self._SLICEABLE_KWARGS
            and key not in self._PASSTHROUGH_KWARGS
        ]
        if (
            not self._active
            or self._child_batches is None
            or (q is None and mixed_qkv is None)
            or unknown
            or (mixed_qkv is not None and not torch.is_tensor(mixed_qkv))
        ):
            if (
                self._active
                and unknown
                and self._child_batches is not None
                and (q is not None or mixed_qkv is not None)
            ):
                # The one silent-哑火 mode: a model-side kwarg outside the
                # whitelist disables the loop for that layer EVERY forward.
                # Loud once per key set, quiet after.
                sig = tuple(sorted(unknown))
                if sig not in self._warned_unknown_kwargs:
                    self._warned_unknown_kwargs.add(sig)
                    logger.warning(
                        "super-chunk passthrough for layer %s: kwargs %s "
                        "outside the slice/passthrough whitelist; SUB_M has "
                        "no effect on this layer.",
                        type(layer).__name__ if layer is not None else "?",
                        list(sig),
                    )
            return self.primary.forward(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache,
                **(
                    {"mixed_qkv": mixed_qkv, "a": a, "b": b}
                    if mixed_qkv is not None
                    else {}
                ),
                **kwargs,
            )
        outs = []
        out = None  # stream buffer for the plain [tokens, ...] output path
        total = self._child_bounds[-1][1]
        for key, axis in self._SLICEABLE_KWARGS.items():
            value = kwargs.get(key)
            if value is not None and value.dim() > axis and value.shape[axis] < total:
                # Assumption broke (unexpected layout): run the whole forward
                # on primary instead of guessing an axis. Loud once per key.
                if key not in self._warned_kwargs:
                    self._warned_kwargs.add(key)
                    logger.warning(
                        "super-chunk: kwarg %r has shape %s, too small along "
                        "axis %d for slicing; falling back to single-pass.",
                        key,
                        tuple(value.shape),
                        axis,
                    )
                return self.primary.forward(
                    q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache,
                    **(
                        {"mixed_qkv": mixed_qkv, "a": a, "b": b}
                        if mixed_qkv is not None
                        else {}
                    ),
                    **kwargs,
                )
        for i, (lo, hi) in enumerate(self._child_bounds):
            child_fb = self._child_batches[i]
            extra = {
                key: kwargs[key].narrow(axis, lo, hi - lo)
                for key, axis in self._SLICEABLE_KWARGS.items()
                if key in kwargs
            }
            extra.update(
                {
                    key: kwargs[key]
                    for key in self._PASSTHROUGH_KWARGS
                    if key in kwargs
                }
            )
            if mixed_qkv is None:
                # dsv4 asserts ``k is v`` (shared KV tensor): keep object
                # identity by slicing once and handing the same view to both.
                if k is v:
                    k_chunk = v_chunk = None if k is None else k[lo:hi]
                else:
                    k_chunk = None if k is None else k[lo:hi]
                    v_chunk = None if v is None else v[lo:hi]
                r = self._ensure_child(i).forward(
                    None if q is None else q[lo:hi],
                    k_chunk,
                    v_chunk,
                    layer,
                    child_fb,
                    save_kv_cache=save_kv_cache,
                    **extra,
                )
            else:
                r = self._ensure_child(i).forward(
                    None,
                    None,
                    None,
                    layer,
                    child_fb,
                    save_kv_cache=save_kv_cache,
                    mixed_qkv=mixed_qkv[lo:hi],
                    a=None if a is None else a[lo:hi],
                    b=None if b is None else b[lo:hi],
                    **extra,
                )
            if mixed_qkv is None and r.dim() >= 1:
                # Write-through into the joined stream: the chunk result is
                # freed immediately, so the peak is stream + one chunk, not
                # stream + all chunks (torch.cat's 2x). Pure copies.
                # A 4D result keeps the token axis at dim 1 ([B, tokens, ...],
                # GDN and absorbed MLA alike), so its stream buffer is
                # (B, total, ...) instead of (total, ...).
                if out is None:
                    if r.dim() == 4:
                        out = torch.empty(
                            (r.shape[0], total, *r.shape[2:]),
                            dtype=r.dtype,
                            device=r.device,
                        )
                    else:
                        out = torch.empty(
                            (total, *r.shape[1:]), dtype=r.dtype, device=r.device
                        )
                if out.dim() == 4:
                    out[:, lo:hi] = r
                else:
                    out[lo:hi] = r
            else:
                outs.append(r)
        if out is not None:
            return out
        if len(outs) == 1:
            return outs[0]
        if mixed_qkv is not None and outs[0].dim() == 4:
            # GDN returns [1, tokens, heads, dim]: the token axis is dim 1.
            return torch.cat(outs, dim=1)
        return torch.cat(outs, dim=0)

    def shared_read_ends(self, fm):
        return self.primary.shared_read_ends(fm)

    def __getattr__(self, name):
        # Fallback for names the base class does not declare (backend-specific
        # helpers). Anything the base DOES declare is handled by the delegation
        # loop below, because __getattr__ never runs when the lookup already
        # resolves — and the base's NotImplementedError stubs always resolve.
        return getattr(self.primary, name)


def _super_chunk_backend(forward_batch):
    """The backend carrying the plan: fb.attn_backend, else the ambient one.

    Draft-extend wiring keeps the wrapper on the ambient ForwardContext (what
    RadixAttention dispatches through) instead of fb.attn_backend; without this
    fallback the model loop would go None while the wrapper still loops,
    materialising CAP-sized projections outside the circle. ``forward_batch``
    may be None for callers that only know the ambient context.
    """
    backend = getattr(forward_batch, "attn_backend", None)
    if backend is not None:
        return backend
    from sglang.srt.model_executor.forward_context import (
        get_forward_context,
        has_forward_context,
    )

    return get_forward_context().attn_backend if has_forward_context() else None


def super_chunk_in_late_tail(forward_batch) -> bool:
    """True while decoder SWA bounded replay has swapped onto tail metadata.

    Model-side loops stop here: their plan was built for full extend rows, and
    the late layers receive tail-gathered rows instead (rows mismatch is the
    guard, this check keeps the expected stop out of the FALLBACK diagnostics).
    """
    backend = _super_chunk_backend(forward_batch)
    if backend is None:
        return False
    md = getattr(backend, "forward_metadata", None)
    tail = getattr(backend, "tail_forward_metadata", None)
    return md is not None and md is tail


def super_chunk_token_plan(forward_batch: ForwardBatch):
    """Model-layer entry point: the active sub-chunk plan or None.

    None means "run today's whole-forward path" (no wrapper, wrapper
    passthrough, decode/capture/spec...). Otherwise the caller loops the
    returned (lo, hi, child_fb) ranges with its per-token math; calls made
    with a child view route back through SuperChunkAttnBackend.forward to the
    matching child backend without any further slicing.
    """
    backend = _super_chunk_backend(forward_batch)
    plan_fn = getattr(backend, "super_chunk_plan", None)
    return plan_fn() if plan_fn is not None else None


def _delegate_to_primary(name: str):
    attr = getattr(AttentionBackend, name, None)
    if callable(attr):

        def method(self, *args, **kwargs):
            return getattr(self.primary, name)(*args, **kwargs)

        method.__name__ = name
        return method

    def getter(self):
        return getattr(self.primary, name)

    def setter(self, value):
        setattr(self.primary, name, value)

    return property(getter, setter)


# Forward every base-class member the wrapper does not deliberately override.
# __getattr__ cannot do this: a name defined on AttentionBackend (including the
# "raise NotImplementedError" stubs such as get_cuda_graph_seq_len_fill_value,
# and the None-defaulted attributes such as forward_metadata / attn_backend_list)
# is found by normal attribute lookup, so the wrapper would silently serve the
# base instead of the primary.
for _name in dir(AttentionBackend):
    if _name.startswith("_") or _name in vars(SuperChunkAttnBackend):
        continue
    setattr(SuperChunkAttnBackend, _name, _delegate_to_primary(_name))
del _name
