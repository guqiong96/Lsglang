"""Inference-only Qwen4-Exp (text + VL) on the Qwen3.5 backbone."""

import math
from contextlib import nullcontext
from typing import Any, Iterable, Optional, Set, Tuple

import msgspec
import sympy
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    attn_tp_all_reduce,
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_size,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen4_exp_ple_table import (
    allocate_ple_host_table,
    make_ple_file_prefetcher,
    make_ple_file_rss_trimmer,
)
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils import get_int_env_var, get_super_chunk_size, logger

# Decode/verify-sized batches only: at prefill sizes both chains are compute
# bound and serializing them on one stream is faster than contending.
_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD = 1024


def _ple_table_is_fp8(
    config: Qwen4ExpTextConfig,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
) -> bool:
    """fp8 PLE shards: declared by config, an fp8 checkpoint, or a ModelOpt
    MIXED_PRECISION entry for the ngram table (nvidia/*-Flash-Next-NVFP4)."""
    if config.ple_embedding_dtype == "float8_e4m3fn":
        return True
    if quant_config is None:
        return False
    if quant_config.get_name() == "fp8":
        return True
    if isinstance(quant_config, ModelOptMixedPrecisionConfig):
        return quant_config.resolve_quant_algo(prefix) == "FP8"
    return False


def _get_ple_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
    if forward_batch._original_forward_mode is not None:
        return forward_batch._original_forward_mode
    return forward_batch.forward_mode


def _get_processed_token_count(
    forward_batch: ForwardBatch, physical_tokens: int
) -> int:
    processed_tokens = forward_batch.global_num_token_non_padded_cpu
    if processed_tokens is None and forward_batch.extend_seq_lens_cpu is not None:
        processed_tokens = sum(forward_batch.extend_seq_lens_cpu)
    if processed_tokens is None:
        return physical_tokens
    processed_tokens = int(processed_tokens)
    if not 0 <= processed_tokens <= physical_tokens:
        raise RuntimeError(
            f"invalid PLE token counts: {processed_tokens=}, {physical_tokens=}"
        )
    return processed_tokens


class _PLEBatch(msgspec.Struct, frozen=True):
    mode: ForwardMode
    use_decode_fast_path: bool
    physical_tokens: int
    processed_tokens: int
    lengths: torch.Tensor
    row_width: int
    req_indices: torch.Tensor
    token_offsets: torch.Tensor
    valid_tokens: torch.Tensor
    state_indices: torch.Tensor
    ngram_context: Optional[torch.Tensor]
    ngram_eos_token_id: Optional[int]


def _prepare_ple_batch(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    ngram_size: Optional[int],
    ngram_eos_token_id: Optional[int],
) -> Optional[_PLEBatch]:
    """Prepare the token layout and the shared N-gram history once per forward."""

    if forward_batch.tbo_parent_token_range is not None:
        raise NotImplementedError("Qwen4 PLE is not compatible with two-batch overlap")
    spec_algorithm = forward_batch.spec_algorithm
    if spec_algorithm is not None and spec_algorithm.is_ngram():
        raise NotImplementedError("Qwen4 PLE does not support NGRAM speculation")
    if (
        forward_batch.spec_info is not None
        and getattr(forward_batch.spec_info, "topk", 1) != 1
    ):
        raise NotImplementedError("Qwen4 PLE speculative decoding supports only topk=1")

    mode = _get_ple_forward_mode(forward_batch)
    get_req_to_token_pool().ple_window_cache = None
    if mode.is_idle():
        return None
    use_decode_fast_path = (
        envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get() and mode.is_decode()
    )

    if input_ids.dim() > 1:
        input_ids = input_ids.reshape(-1)
    physical_tokens = input_ids.shape[0]
    processed_tokens = _get_processed_token_count(forward_batch, physical_tokens)
    tokens = input_ids[:processed_tokens]
    positions = torch.arange(processed_tokens, device=tokens.device, dtype=torch.long)

    if mode.is_target_verify():
        assert forward_batch.spec_info is not None
        row_width = int(forward_batch.spec_info.draft_token_num)
        if row_width <= 0 or processed_tokens % row_width != 0:
            raise RuntimeError(
                "target verify rows must contain complete draft strides: "
                f"{processed_tokens=} {row_width=}"
            )
        sequence_count = processed_tokens // row_width
        # Eager verify rows can be shorter than the row stride;
        # ignore the synthetic one-token lengths from DP attention's verify-as-EXTEND.
        lengths = (
            forward_batch.extend_seq_lens[:sequence_count].long()
            if forward_batch.forward_mode.is_target_verify()
            and forward_batch.extend_seq_lens is not None
            else torch.full(
                (sequence_count,),
                row_width,
                dtype=torch.long,
                device=tokens.device,
            )
        )
        if lengths.shape[0] != sequence_count:
            raise RuntimeError(
                "target verify length metadata does not match its fixed rows: "
                f"{lengths.shape[0]=} {sequence_count=}"
            )
        req_indices = torch.div(positions, row_width, rounding_mode="floor")
        token_offsets = positions - req_indices * row_width
    elif mode.is_decode():
        lengths = torch.ones(processed_tokens, dtype=torch.long, device=tokens.device)
        row_width = 1
        req_indices = positions
        token_offsets = torch.zeros_like(positions)
    else:
        if forward_batch.extend_seq_lens is None:
            raise RuntimeError(f"PLE requires sequence lengths in {mode!r}")
        lengths = forward_batch.extend_seq_lens.long()
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        row_width = (
            max(extend_seq_lens_cpu, default=0)
            if extend_seq_lens_cpu is not None
            else processed_tokens
        )
        query_start_loc = torch.cat(
            [lengths.new_zeros(1), torch.cumsum(lengths, dim=0)]
        )
        sequence_count = lengths.shape[0]
        req_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        if processed_tokens:
            req_indices = req_indices.clamp(min=0, max=sequence_count - 1)
        token_offsets = positions - query_start_loc.index_select(0, req_indices)

    sequence_count = lengths.shape[0]
    if use_decode_fast_path:
        # One token per decode row: every offset is valid, no index-select needed.
        valid_tokens = torch.ones(
            processed_tokens, dtype=torch.bool, device=tokens.device
        )
    else:
        valid_tokens = token_offsets < lengths.index_select(0, req_indices)

    state_indices = (
        get_req_to_token_pool()
        .get_mamba_indices(forward_batch.req_pool_indices[:sequence_count])
        .long()
    )

    # CUDA graph padding uses request slot 0, which may belong to a real request.
    # Map padded sequences to the state pools' reserved dummy slot instead.
    out_cache_loc = forward_batch.out_cache_loc
    if use_decode_fast_path:
        if out_cache_loc is not None:
            state_indices = torch.where(
                out_cache_loc[:sequence_count].ne(0),
                state_indices,
                torch.zeros_like(state_indices),
            )
    else:
        valid = lengths.ne(0)
        if out_cache_loc is not None and mode.is_decode():
            valid = valid & out_cache_loc[:sequence_count].ne(0)
        elif out_cache_loc is not None and mode.is_target_verify():
            valid = valid & out_cache_loc[:processed_tokens].reshape(
                sequence_count, row_width
            ).ne(0).any(dim=1)
        state_indices = torch.where(
            valid, state_indices, torch.zeros_like(state_indices)
        )

    ngram_context = None
    if ngram_size is not None:
        assert ngram_eos_token_id is not None
        if use_decode_fast_path:
            # One token per decode row:
            # this view is the padded tensor the general path would materialize.
            padded = tokens.unsqueeze(1)
        else:
            padded = tokens.new_full((sequence_count, row_width), ngram_eos_token_id)
            if processed_tokens:
                padded[req_indices, token_offsets] = torch.where(
                    valid_tokens,
                    tokens,
                    tokens.new_full((), ngram_eos_token_id),
                )
        history = get_req_to_token_pool().get_ngram_context(state_indices)
        if history.shape[1] != ngram_size - 1:
            raise RuntimeError(
                "Qwen4 PLE N-gram cache has the wrong context width: "
                f"{history.shape[1]=} {ngram_size=}"
            )
        ngram_context = torch.cat([history, padded], dim=1)

    return _PLEBatch(
        mode=mode,
        use_decode_fast_path=use_decode_fast_path,
        physical_tokens=physical_tokens,
        processed_tokens=processed_tokens,
        lengths=lengths,
        row_width=row_width,
        req_indices=req_indices,
        token_offsets=token_offsets,
        valid_tokens=valid_tokens,
        state_indices=state_indices,
        ngram_context=ngram_context,
        ngram_eos_token_id=ngram_eos_token_id,
    )


def _commit_ple_batch(batch: Optional[_PLEBatch], forward_batch: ForwardBatch) -> None:
    """Commit the shared N-gram history after every PLE layer consumed it."""

    if batch is None or batch.ngram_context is None or not batch.processed_tokens:
        return

    pool = get_req_to_token_pool()
    context = batch.ngram_context
    context_len = context.shape[1] - batch.row_width
    if batch.mode.is_target_verify():
        step_contexts = context.unfold(1, context_len, 1)[:, 1:]
        valid_steps = batch.valid_tokens.reshape(
            batch.lengths.shape[0], batch.row_width
        )
        pool.set_ngram_intermediate_context(
            torch.where(
                valid_steps.unsqueeze(-1),
                step_contexts,
                torch.full_like(step_contexts, batch.ngram_eos_token_id),
            )
        )
        return

    if batch.use_decode_fast_path:
        # Decode advances every two-token history by exactly one column.  Slicing
        # preserves the int64 values while avoiding arange + gather launches.
        next_context = context[:, batch.row_width :]
        pool.set_ngram_context(batch.state_indices, next_context)
        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            track_indices, _ = track
            pool.set_ngram_context(track_indices, next_context)
        return

    context_cols = torch.arange(context_len, device=context.device, dtype=torch.long)
    next_context = context.gather(
        1, batch.lengths.unsqueeze(1) + context_cols.unsqueeze(0)
    )
    pool.set_ngram_context(batch.state_indices, next_context)

    track = _ple_track_targets(forward_batch, batch)
    if track is not None:
        track_indices, track_offsets = track
        pool.set_ngram_context(
            track_indices,
            context.gather(1, track_offsets.unsqueeze(1) + context_cols.unsqueeze(0)),
        )


def _ple_track_targets(
    forward_batch: ForwardBatch, batch: _PLEBatch
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Destination slots and gather offsets for the extra-buffer track snapshot.

    With extra_buffer the radix tree caches the ping-pong track slot,
    not the working slot.
    Both PLE side states are laid out [incoming_state | chunk tokens],
    so the boundary value is the state's own gather at a smaller offset;
    callers differ only in tensor rank, hence offsets rather than a gather.
    Masked-off rows route to reserved slot 0, so the shape stays graph-capturable.
    None when tracking is inactive or its metadata is absent.
    """
    track_indices = forward_batch.mamba_track_indices
    track_mask = forward_batch.mamba_track_mask
    if track_indices is None or track_mask is None:
        return None

    rows = batch.lengths.shape[0]
    track_indices = track_indices[:rows]
    dst = torch.where(track_mask[:rows], track_indices, torch.zeros_like(track_indices))

    if batch.mode.is_decode():
        # One token per step, so the boundary offset is the current one. Decode never
        # carries mamba_track_seqlens, so this path must not consult it.
        return dst, batch.lengths

    aligned = forward_batch.mamba_track_aligned_lens()
    if aligned is None:
        return None

    return dst, aligned[:rows].clamp(min=0).minimum(batch.lengths)


def _ple_conv_block_tokens() -> int:
    """Token budget for one padded-conv pass. A super-chunk forward keeps the
    dense [rows, row_width, C] intermediates at the inner chunk size instead
    of the full CAP; matches the attention backend's inner loop knob.

    The shell's per-block temporaries (projections, norms, gated value) are
    the last block-scale peak inside the loop, so this one knob may go below
    the attention inner size without touching any other component (the pass
    is row-parallel; bit-parity is block-size independent)."""
    override = get_int_env_var("SGLANG_PLE_BLOCK_TOKENS")
    if override > 0:
        return override
    size = get_schedule().chunked_prefill_size or 0
    return size if size > 0 else 8192


def _pad_token_rows(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    if x.shape[0] == total_tokens:
        return x
    out = x.new_zeros((total_tokens, *x.shape[1:]))
    out[: x.shape[0]] = x
    return out


def _use_attn_tp_ngram() -> bool:
    return is_dp_attention_enabled() and envs.SGLANG_USE_ATTN_TP_NGRAM.get()


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(x, self.weight, self._jit_group_size, self.eps)
        compute_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        else:
            group_shape = x_float.shape[:-1] + (-1, self.group_size)
            variance = x_float.reshape(group_shape).pow(2).mean(dim=-1, keepdim=True)
            variance = variance.expand(group_shape).reshape_as(x_float)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        weight = self.weight.float() + 1.0
        return (x_norm * weight).to(compute_dtype)


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PRIME_1 = 10007

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_layer_index: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.ngram_embed_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if self.ngram_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
            )
        self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0")
        self.make_ngram_vocab_size_divisible_by = int(
            config.make_ngram_vocab_size_divisible_by
        )
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.enable_ple_fusion = envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get()

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(self.ngram_size),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = (
            self._build_head_vocab_and_offsets()
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = (
            (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
            // self.make_ngram_vocab_size_divisible_by
        ) * self.make_ngram_vocab_size_divisible_by
        self.use_attn_tp_ngram = _use_attn_tp_ngram()
        self.gather_dp_tokens = (
            is_dp_attention_enabled()
            and get_attention_dp_size() > 1
            and not self.use_attn_tp_ngram
        )
        ngram_prefix = f"{prefix}.ngram_embedding" if prefix else "ngram_embedding"
        offload_embedding = bool(config.ple_offload_embedding)
        # Offload only needs this embedding's metadata: build it on meta so the
        # shard is never allocated on the device.
        with torch.device("meta") if offload_embedding else nullcontext():
            ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                params_dtype=(
                    torch.float8_e4m3fn
                    if _ple_table_is_fp8(config, quant_config, ngram_prefix)
                    else torch.bfloat16
                ),
                output_dtype=torch.bfloat16,
                use_attn_tp_group=self.use_attn_tp_ngram,
            )
        # weight_scale stays a real device tensor.
        ngram_embedding.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )
        if offload_embedding:
            ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                ngram_embedding,
                backend=getattr(config, "ple_offload_backend", "pinned"),
                table_dir=getattr(config, "ple_offload_dir", None),
            )
        self.ngram_embedding = ngram_embedding

    @classmethod
    def _splitmix64(cls, x: int) -> int:
        x = (x + cls._SPLITMIX_GAMMA) & cls._MASK64
        x = ((x ^ (x >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        x = ((x ^ (x >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (x ^ (x >> 31)) & cls._MASK64

    def _build_layer_multipliers(self, size: int) -> torch.Tensor:
        seed = int(getattr(self.config, "seed", 1234))
        max_long = (1 << 63) - 1
        m_max = max_long // max(self.unigram_vocab_size, 1)
        half_bound = max(1, m_max // 2)
        values = []
        base_seed = seed + self._PRIME_1 * self.ple_layer_index
        for idx in range(size):
            x0 = (base_seed + self._SPLITMIX_GAMMA * (idx + 1)) & self._MASK64
            mixed = self._splitmix64(x0)
            values.append(int(2 * (mixed % half_bound) + 1))
        return torch.tensor(values, dtype=torch.long)

    @staticmethod
    def _find_nth_prime_after(start: int, n: int) -> int:
        prime = int(start)
        for _ in range(n):
            prime = int(sympy.nextprime(prime))
        return prime

    def _build_head_vocab_and_offsets(self):
        sizes = []
        offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = self._find_nth_prime_after(
                self.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            sizes.append(size)
            offsets.append(total)
            total += size
        return sizes, offsets, total

    def _embed_ngram_ids(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        embeddings = self.ngram_embedding(lookup_ids)
        embeddings = embeddings * self.ngram_embedding.weight_scale
        return self._finish_embedding_lookup(
            embeddings, semantic_tokens, forward_batch, physical_tokens
        )

    def _prepare_embedding_lookup(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> Tuple[torch.Tensor, int]:
        semantic_tokens = ngram_ids.shape[0]
        if not self.gather_dp_tokens:
            return ngram_ids, semantic_tokens

        padded_ngram_ids = _pad_token_rows(ngram_ids, physical_tokens)
        global_tokens = forward_batch.global_dp_buffer_len
        if global_tokens is None:
            raise RuntimeError(
                "global-TP Qwen4 N-gram lookup under DP attention requires a "
                "DP token layout; set SGLANG_USE_ATTN_TP_NGRAM=1 to shard the "
                "table within each attention-TP group"
            )

        global_ngram_ids = ngram_ids.new_empty((global_tokens, *ngram_ids.shape[1:]))
        dp_gather_replicate(
            global_ngram_ids, padded_ngram_ids.contiguous(), forward_batch
        )
        return global_ngram_ids, semantic_tokens

    def _finish_embedding_lookup(
        self,
        embeddings: torch.Tensor,
        semantic_tokens: int,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        if not self.gather_dp_tokens:
            return embeddings
        local_embeddings = embeddings.new_empty(
            (physical_tokens, *embeddings.shape[1:])
        )
        dp_scatter(local_embeddings, embeddings.contiguous(), forward_batch)
        return local_embeddings[:semantic_tokens]

    def _hash_contexts(
        self, contexts: torch.Tensor, *, decode_sized: bool = False
    ) -> torch.Tensor:
        contexts = contexts.to(torch.long)
        if self.enable_ple_fusion and decode_sized:
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_ngram_hash,
                fused_qwen4_ngram_hash,
            )

            if can_fuse_qwen4_ngram_hash(
                contexts,
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
            ):
                return fused_qwen4_ngram_hash(
                    contexts,
                    self.layer_multipliers,
                    self.ngram_heads_vocab_sizes,
                    self.ngram_heads_offsets,
                    self.eos_token_id,
                )

        pool = get_req_to_token_pool()
        cached = pool.ple_window_cache
        if cached is not None and cached[1] is contexts and cached[2] is not None:
            shifted_tokens = cached[2]
            assert len(shifted_tokens) == self.ngram_size
        else:
            shifted_tokens = [contexts]
            for shift in range(1, self.ngram_size):
                shifted_tokens.append(self._shift_right_ignore_eos(contexts, shift))
            if cached is not None and cached[1] is contexts:
                pool.ple_window_cache = (cached[0], contexts, shifted_tokens)

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            ngram_idx = ngram - 2
            start_idx = ngram_idx * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mix = shifted_tokens[0] * self.layer_multipliers[0]
            for pos in range(1, ngram):
                mix = torch.bitwise_xor(
                    mix, shifted_tokens[pos] * self.layer_multipliers[pos]
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(
                mix[:, -1:].unsqueeze(-1), head_vocab_sizes.view(1, 1, -1)
            )
            ngram_ids = ngram_ids + head_offsets.view(1, 1, -1)
            blocks.append(ngram_ids[:, 0])
        return torch.cat(blocks, dim=-1)

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return tensor
        batch_size, seq_len = tensor.shape
        idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
        eos_mask = tensor == self.eos_token_id
        eos_pos = torch.where(eos_mask, idx, -1)
        prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat(
            [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = prev_eos + 1
        pos_in_segment = idx.unsqueeze(0) - segment_start
        src_idx = idx - n
        gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
        shifted = tensor.gather(dim=1, index=gather_idx)
        valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
        return torch.where(valid_mask, shifted, tensor.new_full((), self.eos_token_id))

    def forward_idle(self, forward_batch: ForwardBatch) -> None:
        if not self.gather_dp_tokens:
            return
        input_ids = forward_batch.input_ids.reshape(-1)
        dummy_ids = input_ids.new_zeros((input_ids.shape[0], self.ngram_heads))
        self._embed_ngram_ids(dummy_ids, forward_batch, input_ids.shape[0])

    def forward(
        self,
        batch: _PLEBatch,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        ngram_ids = self.compute_ngram_ids(batch)
        embeddings = self._embed_ngram_ids(
            ngram_ids, forward_batch, batch.physical_tokens
        )
        return embeddings.flatten(start_dim=-2)

    def compute_ngram_ids(self, batch: _PLEBatch) -> torch.Tensor:
        assert batch.ngram_context is not None
        pool = get_req_to_token_pool()
        cached = pool.ple_window_cache
        if cached is not None and cached[0] is batch:
            contexts = cached[1]
        else:
            if batch.use_decode_fast_path:
                contexts = batch.ngram_context
            else:
                contexts = batch.ngram_context.unfold(1, self.ngram_size, 1)[
                    batch.req_indices, batch.token_offsets
                ]
            contexts = contexts.to(torch.long)
            pool.ple_window_cache = (batch, contexts, None)
        return self._hash_contexts(
            contexts,
            decode_sized=batch.mode.is_decode() or batch.mode.is_target_verify(),
        )


@triton.jit
def _gather_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):
    """PLE table read directly from host memory (pinned, or a file-backed mmap).

    The table stays in its checkpoint storage dtype (fp8 with a per-tensor
    weight_scale for fp8 checkpoints, bf16 otherwise); gathers emit bf16.

    The source weight may be on the meta device; only its metadata is used.
    """

    _COPIED_ATTRIBUTES = (
        "quant_config",
        "enable_tp",
        "use_attn_tp_group",
        "tp_size",
        "num_embeddings",
        "org_vocab_size",
        "padding_size",
        "num_added_embeddings",
        "use_presharded_weights",
        "org_vocab_size_padded",
        "num_embeddings_padded",
        "shard_indices",
        "embedding_dim",
        "num_embeddings_per_partition",
        "num_org_embeddings_per_partition",
        "num_added_embeddings_per_partition",
    )

    def __init__(
        self,
        embedding: VocabParallelEmbedding,
        *,
        backend: str = "pinned",
        table_dir: Optional[str] = None,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "PLE embedding offload requires an unquantized embedding table"
            )
        if embedding.weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError(
                "PLE embedding offload requires bfloat16 or fp8 weights, got "
                f"{embedding.weight.dtype}"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError(
                "PLE embedding offload does not support added vocabulary rows"
            )
        for name in self._COPIED_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        # The unquantized CUDA post-load hook is a no-op. Exclude this CPU-only
        # table so the generic loader does not stage it back to GPU unnecessarily.
        self.quant_method = None

        source_weight = embedding.weight
        host_table = allocate_ple_host_table(
            shape=source_weight.shape,
            dtype=source_weight.dtype,
            backend=backend,
            table_dir=table_dir,
            # Each TP rank holds a different vocabulary shard of the same shape.
            tag=(
                f"rows{self.shard_indices.org_vocab_start_index}"
                f"-{self.shard_indices.org_vocab_end_index}"
            ),
        )
        # Only the file backend has anything to prefetch (rows live on storage).
        self._file_prefetcher = make_ple_file_prefetcher(host_table)
        # ... and only it needs its resident set bounded: a fault maps a whole
        # folio, so the mapping would otherwise creep towards the full table.
        self._file_rss_trimmer = make_ple_file_rss_trimmer(host_table)
        cpu_weight = nn.Parameter(host_table, requires_grad=False)
        for name, value in vars(source_weight).items():
            setattr(cpu_weight, name, value)
        cpu_weight.weight_loader = self.weight_loader
        self.register_parameter("weight", cpu_weight)
        # The scale is tiny; keep it with the model instead of offloading it
        # with the table.
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        del embedding.weight
        self._block_d = triton.next_power_of_2(self.embedding_dim)

    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        allocation_context = nullcontext()
        if self.tp_size > 1:
            allocation_context = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with allocation_context, torch.inference_mode(False):
            # The gather kernel emits bf16 rows regardless of the table dtype.
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected_shape, input_ids.device)
        else:
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                    f"{expected_shape}"
                )
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError(
                    "PLE prefetch output must be bfloat16 on the id device"
                )
            output = out

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            if self._file_prefetcher is not None:
                self._file_prefetcher.enqueue(
                    flat_ids,
                    vocab_start=self.shard_indices.org_vocab_start_index,
                    vocab_end=self.shard_indices.org_vocab_end_index,
                )
            _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self.weight.data_ptr(),
                flat_ids,
                output,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=self.shard_indices.org_vocab_start_index,
                tp_vocab_end=self.shard_indices.org_vocab_end_index,
                is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                BLOCK_D=self._block_d,
            )
        return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.gather(input_ids))


class Qwen4ExpPLELayer(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: Optional[int] = None,
        ple_layer_index: int = 0,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.ple_embed_dim = config.ple_embed_dim
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            self.ple_embed_dim,
            ple_layer_index=ple_layer_index,
            quant_config=quant_config,
            prefix=f"{prefix}.ple_embedding" if prefix else "ple_embedding",
        )
        self.short_conv_dilation = self.ple_embedding.ngram_size
        self.short_conv_state_len = (
            self.conv_kernel_size - 1
        ) * self.short_conv_dilation
        self.conv_channels = self.hc_hidden_size
        self.key_proj = ReplicatedLinear(
            self.ple_embed_dim,
            self.conv_channels,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            self.ple_embed_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_hidden = self.hc_hidden_size
        norm_group = self.hidden_size
        self.norm_key = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.norm_query = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.norm_conv = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_channels,
            padding=(self.conv_kernel_size - 1) * self.short_conv_dilation,
            dilation=self.short_conv_dilation,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)
        self._prefetch_stream = (
            torch.cuda.Stream() if config.ple_offload_embedding else None
        )
        self._graph_prefetch_buffers = {}
        self._eager_prefetch_buffer = None
        self._prefetch_state = None

    def _apply_ple_norm(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = norm(x.flatten(-2, -1))
        return y.unflatten(-1, (self.hc_count, self.hidden_size))

    def _short_conv(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        pool = get_req_to_token_pool()
        conv_state = pool.short_conv_layer_cache(self.layer_id)

        if batch.use_decode_fast_path:
            # With row_width=1 the padded/transpose path is x.unsqueeze(-1),
            # and each state boundary is a one-column shift; conv and SiLU stay native.
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_short_conv_state,
                fused_qwen4_short_conv_state,
            )

            fused_state = can_fuse_qwen4_short_conv_state(
                conv_state, batch.state_indices, x
            )
            if fused_state:
                conv_input = fused_qwen4_short_conv_state(
                    conv_state, batch.state_indices, x
                )
            else:
                state = conv_state.index_select(0, batch.state_indices).to(
                    dtype=x.dtype
                )
                conv_input = torch.cat([state, x.unsqueeze(-1)], dim=-1)
            conv_output = F.conv1d(
                conv_input,
                self.conv1d.weight.to(dtype=x.dtype),
                bias=None,
                dilation=self.short_conv_dilation,
                groups=self.conv_channels,
            ).squeeze(-1)
            next_state = conv_input[:, :, batch.row_width :]
            if not fused_state:
                conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, _ = track
                conv_state[track_indices] = next_state.to(dtype=conv_state.dtype)
            return F.silu(conv_output)

        block = _ple_conv_block_tokens()
        if batch.row_width > block and not batch.mode.is_target_verify():
            return self._short_conv_blocked(x, forward_batch, batch, conv_state, block)

        state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
        padded_seq = x.new_zeros(
            (batch.lengths.shape[0], batch.row_width, self.conv_channels)
        )
        padded_seq[batch.req_indices, batch.token_offsets] = x
        conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
        conv_output = F.conv1d(
            conv_input,
            self.conv1d.weight.to(dtype=x.dtype),
            bias=None,
            dilation=self.short_conv_dilation,
            groups=self.conv_channels,
        ).transpose(1, 2)

        if batch.mode.is_target_verify():
            intermediate_cache = pool.short_conv_layer_intermediate_cache(self.layer_id)
            if intermediate_cache is not None:
                if self.short_conv_state_len:
                    intermediate_state = (
                        conv_input.unfold(2, self.short_conv_state_len, 1)[
                            :, :, 1 : batch.row_width + 1
                        ]
                        .permute(0, 2, 1, 3)
                        .contiguous()
                    )
                else:
                    intermediate_state = x.new_empty(
                        (
                            batch.lengths.shape[0],
                            batch.row_width,
                            self.conv_channels,
                            0,
                        )
                    )
                valid_steps = batch.valid_tokens.reshape(
                    batch.lengths.shape[0], batch.row_width, 1, 1
                )
                intermediate_state = torch.where(
                    valid_steps,
                    intermediate_state,
                    torch.zeros_like(intermediate_state),
                )
                intermediate_cache[: batch.lengths.shape[0], : batch.row_width].copy_(
                    intermediate_state.to(dtype=intermediate_cache.dtype)
                )
        else:
            state_cols = torch.arange(
                self.short_conv_state_len, device=x.device, dtype=torch.long
            )

            def _gather_at(offsets: torch.Tensor) -> torch.Tensor:
                return conv_input.gather(
                    2,
                    (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
                    .unsqueeze(1)
                    .expand(-1, self.conv_channels, -1),
                )

            next_state = _gather_at(batch.lengths)
            conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            # Same boundary mamba uses, into the slot the radix tree reads.
            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, track_offsets = track
                conv_state[track_indices] = _gather_at(track_offsets).to(
                    dtype=conv_state.dtype
                )

        return F.silu(conv_output[batch.req_indices, batch.token_offsets])

    def _short_conv_blocked(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
        conv_state: torch.Tensor,
        block: int,
    ) -> torch.Tensor:
        """Padded-conv over a super-chunk without materializing [rows, CAP, C].

        Each pass convolves one token-axis block per row, prefixed by a gather
        of the state_len tokens right before the window (from x, or from the
        pool state where the request starts inside the block). The depthwise
        receptive field is exactly that window, so outputs and the final state
        match the single dense conv column for column.
        """
        device = x.device
        state_len = self.short_conv_state_len
        channels = self.conv_channels
        total = x.shape[0]
        rows = batch.lengths.shape[0]
        starts = torch.cumsum(batch.lengths, dim=0) - batch.lengths
        state_cols = torch.arange(state_len, device=device, dtype=torch.long)
        state_rows = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
        weight = self.conv1d.weight.to(dtype=x.dtype)
        positions = torch.arange(total, device=device)

        def _tokens_before(ends: torch.Tensor) -> torch.Tensor:
            # local ends -> [rows, C, state_len] holding tokens [ends-S, ends),
            # falling back to the pool state where ends - S + c < 0.
            loc = ends.unsqueeze(1) - state_len + state_cols
            from_x = loc.view(rows, 1, state_len) >= 0
            idx = (starts.unsqueeze(1) + loc).clamp(min=0, max=total - 1)
            token_side = (
                x.index_select(0, idx.reshape(-1))
                .reshape(rows, state_len, channels)
                .transpose(1, 2)
            )
            state_side = torch.gather(
                state_rows,
                2,
                (loc + state_len).clamp(min=0, max=state_len - 1)
                .unsqueeze(1)
                .expand(rows, channels, state_len),
            )
            return torch.where(from_x, token_side, state_side)

        # State commits read x only, so they run before the in-place loop below.
        conv_state[batch.state_indices] = _tokens_before(batch.lengths).to(
            dtype=conv_state.dtype
        )
        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            track_indices, track_offsets = track
            conv_state[track_indices] = _tokens_before(track_offsets).to(
                dtype=conv_state.dtype
            )

        # Blocks run high to low and write their silu result into x in place:
        # every read (window tokens, left context) lives at or below the block's
        # own range, which earlier iterations never touched. No [total, C]
        # output tensor is ever materialized.
        for hi in range(total, 0, -block):
            lo = max(hi - block, 0)
            window_lo = (lo - starts).clamp(min=0).minimum(batch.lengths)
            window_hi = (hi - starts).clamp(min=0).minimum(batch.lengths)
            width = int((window_hi - window_lo).max().item())
            seq_block = x.new_zeros((rows, width, channels))
            req_block = batch.req_indices[lo:hi]
            edge = starts + window_lo
            # Invalid (padding-slot) tokens keep their value irrelevant by the
            # forward's valid mask, so clamping their column is safe.
            col_block = (
                (positions[lo:hi] - edge.index_select(0, req_block))
                .clamp_(min=0)
                .clamp_(max=width - 1)
            )
            seq_block[req_block, col_block] = x[lo:hi]
            conv_input = x.new_empty((rows, channels, state_len + width))
            conv_input[:, :, :state_len].copy_(_tokens_before(window_lo))
            conv_input[:, :, state_len:].copy_(seq_block.transpose(1, 2))
            del seq_block
            conv_block = F.conv1d(
                conv_input,
                weight,
                bias=None,
                dilation=self.short_conv_dilation,
                groups=channels,
            ).transpose(1, 2)
            del conv_input
            x[lo:hi] = conv_block[req_block, col_block]
            torch.ops.aten.silu_(x[lo:hi])

        return x

    def forward_idle(self, forward_batch: ForwardBatch) -> None:
        if self._prefetch_state is not None:
            self._consume_prefetched_embeddings(forward_batch)
        else:
            self.ple_embedding.forward_idle(forward_batch)

    def _allocate_prefetch_buffer(
        self, lookup_tokens: int, lookup_ids: torch.Tensor
    ) -> torch.Tensor:
        offloaded_embedding = self.ple_embedding.ngram_embedding
        return offloaded_embedding.allocate_output(
            (lookup_tokens, self.ple_embed_dim), lookup_ids.device
        )

    def _get_prefetch_buffer(
        self, lookup_tokens: int, lookup_ids: torch.Tensor
    ) -> torch.Tensor:
        if get_is_capture_mode():
            buffer = self._graph_prefetch_buffers.get(lookup_tokens)
            if buffer is None:
                buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
                self._graph_prefetch_buffers[lookup_tokens] = buffer
            return buffer

        buffer = self._eager_prefetch_buffer
        if buffer is None or buffer.shape[0] < lookup_tokens:
            buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
            self._eager_prefetch_buffer = buffer
        return buffer[:lookup_tokens]

    def start_prefetch(
        self,
        batch: Optional[_PLEBatch],
        forward_batch: ForwardBatch,
    ) -> None:
        """Gather PLE rows via UVA while the preceding decoder layer runs."""
        if self._prefetch_stream is None:
            return
        if self._prefetch_state is not None:
            raise RuntimeError("PLE prefetch state was not consumed before reuse")
        if batch is None:
            if not self.ple_embedding.gather_dp_tokens:
                return
            physical_tokens = forward_batch.input_ids.numel()
            ngram_ids = forward_batch.input_ids.new_zeros(
                (physical_tokens, self.ple_embedding.ngram_heads)
            )
        else:
            physical_tokens = batch.physical_tokens
            ngram_ids = self.ple_embedding.compute_ngram_ids(batch)

        lookup_ids, semantic_tokens = self.ple_embedding._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            return
        prefetched = self._get_prefetch_buffer(lookup_tokens, lookup_ids)
        output_view = prefetched.view(lookup_tokens, self.ple_embedding.ngram_heads, -1)
        offloaded_embedding = self.ple_embedding.ngram_embedding

        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream())
        lookup_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            offloaded_embedding.gather(lookup_ids, out=output_view)
        self._prefetch_state = prefetched, semantic_tokens, physical_tokens

    def _consume_prefetched_embeddings(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if self._prefetch_state is None:
            raise RuntimeError("PLE prefetch state is missing")
        embeddings, semantic_tokens, physical_tokens = self._prefetch_state
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)
        embeddings = self.ple_embedding.ngram_embedding.reduce(embeddings)
        embeddings = embeddings * self.ple_embedding.ngram_embedding.weight_scale
        embeddings = self.ple_embedding._finish_embedding_lookup(
            embeddings,
            semantic_tokens,
            forward_batch,
            physical_tokens,
        )
        self._prefetch_state = None
        return embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
        accumulate_into: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        hidden_states = hidden_states[: batch.processed_tokens]
        block = _ple_conv_block_tokens()
        if (
            not batch.use_decode_fast_path
            and not batch.mode.is_target_verify()
            and batch.row_width > block
            and not self.ple_embedding.gather_dp_tokens
        ):
            return self._forward_blocked(
                hidden_states, forward_batch, batch, block, accumulate_into
            )
        if self._prefetch_state is not None:
            embeddings = self._consume_prefetched_embeddings(forward_batch)
        else:
            embeddings = self.ple_embedding(batch, forward_batch)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        hidden_size = self.hidden_size
        hc_count = self.hc_count
        if hidden_states.shape[-1] != hc_count * hidden_size:
            raise RuntimeError(
                "PLE hidden size does not match its hyper-connection layout: "
                f"expected {hc_count * hidden_size}, got {hidden_states.shape[-1]}"
            )
        key = key.reshape(token_count, hc_count, hidden_size)
        query = hidden_states.reshape(token_count, hc_count, hidden_size)
        key_normed = self._apply_ple_norm(self.norm_key, key)
        query_normed = self._apply_ple_norm(self.norm_query, query)
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(hidden_size)
        fused_gate_value = False
        if batch.use_decode_fast_path:
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_gate_value,
                fused_qwen4_gate_value,
            )

            fused_gate_value = can_fuse_qwen4_gate_value(gate, value)
        if fused_gate_value:
            gated_value = fused_qwen4_gate_value(gate, value)
        else:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = torch.sigmoid(gate)
            gated_value = gate * value.unsqueeze(-2)
        gated_value_normed = self._apply_ple_norm(self.norm_conv, gated_value)
        gated_value = gated_value.flatten(-2)
        gated_value_normed = gated_value_normed.flatten(-2)
        conv_output = self._short_conv(
            gated_value_normed,
            forward_batch,
            batch,
        )
        # gated_value is dead after this sum: fold in place, and mask in place
        # (torch.where + zeros_like would peak at three [tokens, C] tensors).
        output = gated_value.add_(conv_output)
        if not batch.use_decode_fast_path:
            output.masked_fill_(~batch.valid_tokens.unsqueeze(-1), 0.0)
        return _pad_token_rows(output, batch.physical_tokens)

    def _forward_blocked(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
        block: int,
        accumulate_into: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Whole-shell PLE pass in token blocks over a super-chunk.

        The dense pass keeps ~4 CAP-sized token intermediates (embedding
        gather, key/value projections, gated value, its norm). The shell is
        token-parallel, so this blocks all of it: the conv reads its left
        context from a per-row carry buffer (initialised from the pool state,
        updated block by block) instead of a materialised [tokens, C] input.
        Convolving carry+window reproduces the dense conv column for column,
        so state, track snapshots and outputs match the single dense pass.

        With ``accumulate_into`` the per-block result folds straight into the
        residual stream (in place, row ranges never revisited): no CAP-sized
        output tensor and no full-width add; returns None. Without it a fresh
        [tokens, C] output is materialised and returned (dense semantics).
        """
        device = hidden_states.device
        total = hidden_states.shape[0]
        rows = batch.lengths.shape[0]
        channels = self.conv_channels
        state_len = self.short_conv_state_len
        starts = torch.cumsum(batch.lengths, dim=0) - batch.lengths
        state_cols = torch.arange(state_len, device=device, dtype=torch.long)
        pool = get_req_to_token_pool()
        conv_state = pool.short_conv_layer_cache(self.layer_id)
        state_rows = conv_state.index_select(0, batch.state_indices).to(
            dtype=hidden_states.dtype
        )
        carry = state_rows.clone()
        weight = self.conv1d.weight.to(dtype=hidden_states.dtype)
        positions = torch.arange(total, device=device)

        if self._prefetch_state is not None:
            # The prefetch buffer is already [tokens, ple_embed_dim]
            # (allocate_output's shape), same as the dense forward's input.
            embeddings = self._consume_prefetched_embeddings(forward_batch)
            ngram_embedding = None
        else:
            ngram_ids = self.ple_embedding.compute_ngram_ids(batch)
            embeddings = None
            ngram_embedding = self.ple_embedding.ngram_embedding

        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            # Values are buffered and committed after the working state, in
            # the dense pass's write order (state first, track wins).
            track_state = torch.zeros(
                (rows, channels, state_len),
                dtype=hidden_states.dtype,
                device=device,
            )
            track_ready = torch.zeros(rows, dtype=torch.bool, device=device)

        if accumulate_into is not None:
            output = accumulate_into[:total]
            accumulating = True
        else:
            output = hidden_states.new_empty((total, channels))
            accumulating = False

        # Fixed-capacity conv staging, allocated once per pass: a per-block
        # width-varying (re)allocation churns the caching allocator exactly at
        # the memory-tight super-chunk stages (fragmented reserved-but-
        # unusable blocks -> OOM on the next block). Every read below stays
        # inside [0, width + state_len) of the current block's writes
        # (causal depthwise window, carry/track gathers), so stale bytes in
        # the reused tail can never reach an output column.
        w_max = min(block, total)
        seq_buf = hidden_states.new_zeros((rows, w_max, channels))
        conv_buf = hidden_states.new_empty((rows, channels, state_len + w_max))

        for lo in range(0, total, block):
            hi = min(lo + block, total)
            tok = slice(lo, hi)

            if embeddings is not None:
                emb = embeddings[tok]
            else:
                emb = (
                    ngram_embedding(ngram_ids[tok]) * ngram_embedding.weight_scale
                ).flatten(-2)
            key, _ = self.key_proj(emb)
            value, _ = self.value_proj(emb)

            token_count = hi - lo
            key = key.reshape(token_count, self.hc_count, self.hidden_size)
            query = hidden_states[tok].reshape(
                token_count, self.hc_count, self.hidden_size
            )
            key_normed = self._apply_ple_norm(self.norm_key, key)
            query_normed = self._apply_ple_norm(self.norm_query, query)
            gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
            gate = gate / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = torch.sigmoid(gate)
            gated_value = gate * value.unsqueeze(-2)
            gated_value_normed = self._apply_ple_norm(
                self.norm_conv, gated_value
            ).flatten(-2)
            gated_value = gated_value.flatten(-2)

            window_lo = (lo - starts).clamp(min=0).minimum(batch.lengths)
            window_hi = (hi - starts).clamp(min=0).minimum(batch.lengths)
            width = int((window_hi - window_lo).max().item())
            req_block = batch.req_indices[tok]
            edge = starts + window_lo
            col_block = (
                (positions[tok] - edge.index_select(0, req_block))
                .clamp_(min=0)
                .clamp_(max=width - 1)
            )
            seq_buf[req_block, col_block] = gated_value_normed
            conv_input = conv_buf
            # Prefix = tokens [window_lo - S, window_lo): the carry where the
            # local offset is inside this forward, the pool state where the
            # request starts (same fallback as the blocked dense-input conv).
            loc = window_lo.unsqueeze(1) - state_len + state_cols
            from_carry = loc.view(rows, 1, state_len) >= 0
            # carry's columns already align with state_cols (its right edge is
            # this block's window_lo for every row it holds), so only the pool
            # side needs a gather (it stays anchored at the request start).
            state_side = torch.gather(
                state_rows,
                2,
                (loc + state_len)
                .clamp(min=0, max=state_len - 1)
                .unsqueeze(1)
                .expand(rows, channels, state_len),
            )
            conv_input[:, :, :state_len].copy_(
                torch.where(from_carry, carry, state_side)
            )
            conv_input[:, :, state_len : state_len + width].copy_(
                seq_buf[:, :width].transpose(1, 2)
            )
            del gated_value_normed

            # Track commit: first block whose window right edge reaches the
            # boundary offset; the state [off-S, off) is fully inside this
            # block's conv_input span (its left end is window_lo - S <= off).
            if track is not None:
                _, track_offsets = track
                hit = (~track_ready) & (window_hi > window_lo) & (
                    track_offsets <= window_hi
                )
                if bool(hit.any()):
                    track_state[hit] = conv_input[hit].gather(
                        2,
                        (track_offsets[hit] - window_lo[hit])
                        .unsqueeze(1)
                        .unsqueeze(-1)
                        .expand(-1, channels, state_len)
                        + state_cols,
                    )
                track_ready |= hit

            conv_block = F.conv1d(
                conv_input,
                weight,
                bias=None,
                dilation=self.short_conv_dilation,
                groups=channels,
            ).transpose(1, 2)
            if state_len:
                # Right edge is the ROW's window_hi, not the shared block
                # width: rows with fewer tokens in this block must not pull
                # the padding columns into their carry.
                present = window_hi > window_lo
                next_carry = torch.gather(
                    conv_input,
                    2,
                    ((window_hi - window_lo).view(rows, 1, 1) + state_cols)
                    .expand(rows, channels, state_len),
                )
                carry = torch.where(present.view(-1, 1, 1), next_carry, carry)

            conv_out = conv_block[req_block, col_block]
            del conv_block
            torch.ops.aten.silu_(conv_out)
            out_block = gated_value.add_(conv_out)
            out_block.masked_fill_(~batch.valid_tokens[tok].unsqueeze(-1), 0.0)
            if accumulating:
                # Query rows were consumed above (into query_normed) and the
                # carry path never reads hidden_states, so an in-place add on
                # this block's rows is safe; padding rows stay as hidden+0.
                output[tok].add_(out_block)
            else:
                output[tok] = out_block

        conv_state[batch.state_indices] = carry.to(dtype=conv_state.dtype)
        if track is not None:
            # Rows never hit (zero-length rows): their boundary is the request
            # start, i.e. exactly the pool state, as the dense pass commits.
            track_state[~track_ready] = state_rows[~track_ready]
            conv_state[track[0]] = track_state.to(dtype=conv_state.dtype)
        if accumulating:
            return None
        return _pad_token_rows(output, batch.physical_tokens)


class Qwen4ExpLayerExtensionMixin:
    def _init_qwen4_exp_layer_extensions(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.ple = None

        for attr_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "layer_communicator",
        ):
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        if (layer_id + 1) in config.ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(config.ple_layer_ids))
            ple_layer_index = {
                abs_id: index for index, abs_id in enumerate(ple_layer_ids_sorted)
            }[layer_id + 1]
            # Strip the block-type segment like the dense mlp (PLE is attn's sibling);
            # else the quant prefix misses the ckpt skip-list -> NaN.
            ple_prefix = prefix.replace(".linear_attn", "").replace(".self_attn", "")
            self.ple = Qwen4ExpPLELayer(
                config,
                quant_config=quant_config,
                prefix=f"{ple_prefix}.ple" if ple_prefix else "ple",
                layer_id=layer_id,
                ple_layer_index=ple_layer_index,
            )

        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            use_mix=True,
            use_combine=True,
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            use_mix=True,
            use_combine=True,
        )

    def _qwen4_hc_stream(self, rows: int, forward_batch):
        """Static hc-stream slice ([rows, hc*hidden]) for an active super-chunk
        forward, else None (= today's allocating path).

        The buffer is model-owned, CAP-sized and allocated before the KV pool
        profiles memory; layer-0 seeds it with the hc replication and every
        combine writes back through it (row-local math, bit-identical).
        Decode / CUDA capture / inactive plans never route here.
        """
        buf = getattr(self, "_hc_stream_buf", None)
        if buf is None or rows == 0 or rows > buf.shape[0]:
            return None
        if forward_batch is None or get_is_capture_mode():
            return None
        from sglang.srt.layers.attention.super_chunk_backend import (
            super_chunk_token_plan,
        )

        if super_chunk_token_plan(forward_batch) is None:
            return None
        return buf[:rows]

    def _prepare_qwen4_exp_attn(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        *,
        ple_batch: Optional[_PLEBatch],
    ):
        hc_dim = self.hc_count * self.hidden_size
        if hidden_states.shape[-1] != hc_dim:
            assert hidden_states.shape[-1] == self.hidden_size
            stream = self._qwen4_hc_stream(hidden_states.shape[0], forward_batch)
            if stream is not None:
                stream.unflatten(-1, (self.hc_count, self.hidden_size)).copy_(
                    hidden_states.unsqueeze(-2)
                )
                hidden_states = stream
            else:
                hidden_states = torch.cat(
                    [hidden_states for _ in range(self.hc_count)], dim=-1
                )

        if self.ple is not None:
            if ple_batch is None:
                if not _get_ple_forward_mode(forward_batch).is_idle():
                    raise RuntimeError(
                        "non-idle Qwen4 PLE forward is missing its batch"
                    )
                self.ple.forward_idle(forward_batch)
            else:
                ple_query = (
                    hidden_states if residual is None else hidden_states + residual
                )
                ple_output = self.ple(
                    ple_query,
                    forward_batch,
                    ple_batch,
                    # Blocked passes fold into the residual stream directly:
                    # no [CAP, C] output tensor and no CAP-wide add.
                    accumulate_into=hidden_states,
                )
                if ple_output is not None:
                    hidden_states = hidden_states + ple_output

        hidden_states, residual = self.attn_hyper_connection.mix(hidden_states)
        return hidden_states, residual

    def _prepare_qwen4_exp_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        if not forward_batch.forward_mode.is_idle():
            hidden_states = attn_tp_all_reduce(hidden_states)
        hidden_states = self.attn_hyper_connection.combine(
            hidden_states,
            residual,
            in_place=self._qwen4_hc_stream(residual[0].shape[0], forward_batch)
            is not None,
        )
        hidden_states, residual = self.mlp_hyper_connection.mix(hidden_states)
        return hidden_states, residual

    def _qwen4_exp_use_dp_moe_gather(self) -> bool:
        return get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none()

    def _qwen4_exp_use_attn_tp_a2a_scatter(self) -> bool:
        return get_parallel().attn_tp_size > 1 and not get_moe_a2a_backend().is_none()

    def _run_qwen4_exp_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not self.config.num_experts:
            return self.mlp(hidden_states)

        use_dp_moe_gather = self._qwen4_exp_use_dp_moe_gather()
        use_attn_tp_a2a_scatter = self._qwen4_exp_use_attn_tp_a2a_scatter()

        if use_dp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        elif hidden_states.shape[0] == 0 and get_moe_a2a_backend().is_none():
            # Only safe to short-circuit an empty batch when the MoE holds no collective;
            # under deepep an idle DP rank must still join dispatch/combine or peers hang.
            return hidden_states

        attn_tp_chunks = None
        if use_attn_tp_a2a_scatter:
            attn_tp_size = get_parallel().attn_tp_size
            attn_tp_chunks = list(hidden_states.tensor_split(attn_tp_size))
            hidden_states = attn_tp_chunks[get_parallel().attn_tp_rank].contiguous()

        hidden_states = self.mlp(hidden_states, forward_batch)

        if use_dp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            if should_use_dp_reduce_scatterv():
                get_tp_group().reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=get_dp_global_num_tokens(),
                )
            else:
                dp_scatter(hidden_states, global_hidden_states, forward_batch)
        elif use_attn_tp_a2a_scatter:
            assert attn_tp_chunks is not None
            gathered = [torch.empty_like(t) for t in attn_tp_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)

        return hidden_states

    def _postprocess_qwen4_exp_layer(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.mlp_hyper_connection.combine(
            hidden_states,
            residual,
            in_place=self._qwen4_hc_stream(residual[0].shape[0], forward_batch)
            is not None,
        )
        return hidden_states, None


class Qwen4ExpLinearDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5LinearDecoderLayer
):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        self._init_qwen4_exp_layer_extensions(config, layer_id, quant_config, prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

        hidden_states, residual = self._prepare_qwen4_exp_attn(
            hidden_states,
            residual,
            forward_batch,
            ple_batch=kwargs.get("ple_batch"),
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(hidden_states, forward_batch)

        hidden_states, residual = self._prepare_qwen4_exp_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self._run_qwen4_exp_mlp(hidden_states, forward_batch)
        return self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)


class Qwen4ExpAttentionDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5AttentionDecoderLayer
):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        config.attn_output_gate = True
        super().__init__(config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        from sglang.srt.layers.attention.qsa.config import is_qwen_qsa
        from sglang.srt.layers.attention.qsa.glue import build_qsa_indexer

        self.is_qsa = is_qwen_qsa(config)
        if self.is_qsa:
            self.indexer = build_qsa_indexer(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer" if prefix else "indexer",
                rotary_emb=self.rotary_emb,
            )
        self._init_qwen4_exp_layer_extensions(config, layer_id, quant_config, prefix)

    def _compute_qsa_topk_indices(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.qsa.glue import (
            get_qsa_indexer_metadata,
            resolve_qsa_sparse_backend,
        )

        backend = get_attn_backend()
        child_index = getattr(forward_batch, "super_chunk_child_index", None)
        if child_index is not None:
            # Super-chunk loop: metadata/pool bookkeeping must resolve through
            # the sub-chunk's own backend (== today's per-chunk indexer call).
            child_fn = getattr(backend, "super_chunk_child_backend", None)
            if child_fn is not None:
                backend = child_fn(child_index)
        sparse_backend = resolve_qsa_sparse_backend(backend)
        should_reuse = getattr(sparse_backend, "should_reuse_mtp_sparse_indices", None)
        if should_reuse is not None and should_reuse(forward_batch):
            # MTP decode steps reuse the draft-extend's target-aligned
            # selection; the indexer never runs inside the decode graph.
            return sparse_backend.lookup_mtp_sparse_indices(
                forward_batch, self.layer_id
            )
        indexer_metadata = get_qsa_indexer_metadata(
            backend, self.layer_id, forward_batch
        )
        topk_indices = self.indexer(
            hidden_states,
            positions,
            forward_batch,
            indexer_metadata,
        )
        should_capture = getattr(
            sparse_backend, "should_capture_mtp_sparse_indices", None
        )
        if should_capture is not None and should_capture(forward_batch):
            sparse_backend.capture_mtp_sparse_indices(
                topk_indices, forward_batch, self.layer_id, metadata=indexer_metadata
            )
        return topk_indices

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        overlap_indexer = (
            self.is_qsa
            and self.alt_stream is not None
            and get_is_capture_mode()
            and hidden_states.shape[0] < _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD
        )
        attention_kwargs = {}
        if overlap_indexer:
            # Safe to overlap: the indexer reads only hidden_states/positions,
            # and writes QSA-private pool buffers.
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

        # Super-chunk forward: run qkv/gate/o_proj inside the sub-chunk loop so
        # no [CAP, qkv] intermediate outlives its chunk (the wrapper-level loop
        # can only slice tensors that were already projected at full size).
        # The indexer joins the loop too (causal; its pool writes and MTP
        # capture follow the same order as today's per-chunk forwards), so no
        # [CAP, ...] indexer tensor outlives its chunk. Capture-mode overlap
        # is mutually exclusive with the plan.
        from sglang.srt.layers.attention.super_chunk_backend import (
            super_chunk_token_plan,
        )

        plan = None if overlap_indexer else super_chunk_token_plan(forward_batch)
        if plan is not None:
            outs = []
            # mrope (VL checkpoints) carries positions as [axes, tokens]; the
            # token axis is the LAST one, not dim 0.
            pos_is_2d = positions.ndim == 2
            for lo, hi, child_fb in plan:
                tok = slice(lo, hi)
                pos_sl = (
                    positions[:, lo:hi] if pos_is_2d else positions[tok]
                )
                qsa_kwargs = {}
                if self.is_qsa:
                    qsa_kwargs["topk_indices"] = self._compute_qsa_topk_indices(
                        hidden_states[tok], pos_sl, child_fb
                    )
                q, k, v, gate = self._prepare_qkv_gate(
                    positions=pos_sl,
                    hidden_states=hidden_states[tok],
                    forward_batch=child_fb,
                )
                outs.append(
                    self._attend_and_project(
                        q, k, v, gate, child_fb, **qsa_kwargs
                    )
                )
            return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

        q, k, v, gate = self._prepare_qkv_gate(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        if overlap_indexer:
            with torch.cuda.stream(self.alt_stream):
                topk_indices = self._compute_qsa_topk_indices(
                    hidden_states, positions, forward_batch
                )
            current_stream.wait_stream(self.alt_stream)
            # Allocated on alt_stream, consumed by attention on the current
            # stream; tell the caching allocator before alt_stream is reused.
            topk_indices.record_stream(current_stream)
            attention_kwargs["topk_indices"] = topk_indices
        elif self.is_qsa:
            attention_kwargs["topk_indices"] = self._compute_qsa_topk_indices(
                hidden_states, positions, forward_batch
            )

        return self._attend_and_project(
            q, k, v, gate, forward_batch, **attention_kwargs
        )

    def _attend_and_project(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **attention_kwargs,
    ) -> torch.Tensor:
        attn_output = self.attn(q, k, v, forward_batch, **attention_kwargs)
        if gate is not None:
            if attn_output.is_cuda:
                # The strided 3D gate view feeds the kernel directly, so the
                # gate reshape copy disappears along with the sigmoid + mul.
                attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
            else:
                gate = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
                attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        hidden_states, residual = self._prepare_qwen4_exp_attn(
            hidden_states,
            residual,
            forward_batch,
            ple_batch=kwargs.get("ple_batch"),
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self._prepare_qwen4_exp_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self._run_qwen4_exp_mlp(hidden_states, forward_batch)
        return self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen4ExpAttentionDecoderLayer,
    "full_attention": Qwen4ExpAttentionDecoderLayer,
    "linear_attention": Qwen4ExpLinearDecoderLayer,
}


class Qwen4ExpModel(Qwen3_5ForCausalLM):
    decoder_layer_types = ALL_DECODER_LAYER_TYPES

    def _build_embed_tokens(self, config: Qwen4ExpTextConfig) -> nn.Module:
        return VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            use_attn_tp_group=is_dp_attention_enabled(),
        )

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ) -> None:
        super().__init__(config, quant_config, prefix, is_nextn)
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.has_ple = bool(config.ple_layer_ids)
        self.ple_ngram_size = int(config.ngram_size) if self.has_ple else None
        self.ple_ngram_eos_token_id = (
            int(config.eos_token_id) if self.ple_ngram_size is not None else None
        )
        if hasattr(self, "norm"):
            delattr(self, "norm")
        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(hc_config, use_combine=False)

        # Super-chunk forwards keep the inter-layer hc stream in ONE static
        # CAP-sized buffer (cat-into at layer 0, in-place combines after each
        # block_output). Allocated at model-init so the KV pool's free-memory
        # profile already accounts for it (same placement rule as the lk_moe
        # arena); decode / capture / non-super-chunk / oversized forwards
        # never touch it and keep the allocating path.
        self._hc_stream_buf = None
        _stream_cap = get_super_chunk_size()
        if _stream_cap > 0 and self.hc_count > 1:
            self._hc_stream_buf = torch.empty(
                (_stream_cap, self.hc_count * self.hidden_size),
                dtype=torch.bfloat16,
                device=torch.cuda.current_device(),
            )
            for _layer in self.layers:
                _layer._hc_stream_buf = self._hc_stream_buf
            # Allocated through the caching allocator before the KV pool
            # profiles free memory, so _profile_available_bytes already
            # charges it; log the exact figure for the pool ledger.
            logger.info(
                "Super-chunk hc stream: static buffer %d x %d bf16 = %.3f GB "
                "per rank (charged to the KV pool profile at init).",
                _stream_cap,
                self.hc_count * self.hidden_size,
                self._hc_stream_buf.numel() * 2 / (1 << 30),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        ple_batch = (
            _prepare_ple_batch(
                input_ids,
                forward_batch,
                ngram_size=self.ple_ngram_size,
                ngram_eos_token_id=self.ple_ngram_eos_token_id,
            )
            if self.has_ple
            else None
        )
        residual = None
        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i + 1 < self.end_layer:
                next_ple = getattr(self.layers[i + 1], "ple", None)
                if next_ple is not None:
                    next_ple.start_prefetch(ple_batch, forward_batch)
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    ple_batch=ple_batch,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )

        _commit_ple_batch(ple_batch, forward_batch)

        hc_hidden_states = hidden_states
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        if not forward_batch.forward_mode.is_idle():
            return hidden_states, hc_hidden_states

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class Qwen4ExpVLModel(Qwen4ExpModel):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.last_hc_hidden_states = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[Any] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.last_hc_hidden_states = None
        # mm routine passes input_ids=None; PLE needs the real ids.
        if input_ids is None:
            input_ids = forward_batch.input_ids
        model_output = super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=input_embeds,
        )
        if isinstance(model_output, tuple):
            hidden_states, self.last_hc_hidden_states = model_output
            return hidden_states
        return model_output


class Qwen4ExpForConditionalGeneration(Qwen3VLForConditionalGeneration):
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return Qwen4ExpVLModel.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

    def __init__(
        self,
        config: Qwen4ExpConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen4ExpVLModel,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = (
            "mrope_section" in rope_config and not self.language_model_only
        )
        self.deepstack_visual_indexes = (
            self.visual.deepstack_visual_indexes if self.visual is not None else []
        )

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        hc_hidden_states = self.model.last_hc_hidden_states
        if hc_hidden_states is not None and isinstance(output, LogitsProcessorOutput):
            output.hidden_states = hc_hidden_states
        return output

    def _load_qwen4_exp_ple_buffer(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        buffers: dict,
        loaded_buffers: Set[str],
    ) -> bool:
        if ".ple.ple_embedding." not in name:
            return False
        buffer_name = name.rsplit(".", 1)[-1]
        if buffer_name.startswith("hashstats_"):
            return True
        if buffer_name == "token_lookup":
            return True
        if buffer_name not in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
            "weight_scale",
        }:
            return False
        buffer = buffers.get(name)
        if buffer is None:
            return False
        if buffer.shape != loaded_weight.shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(buffer.shape)}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
        loaded_buffers.add(name)
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # Checkpoints use the qwen3.5 head-first in_proj layout,
            # matching Qwen3_5GatedDeltaNet's forward, not qwen3-next's group-first.
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        num_experts = getattr(self.config, "num_experts", None)
        expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
            if num_experts is not None
            else []
        )
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale_inv",
            "_weight_scale_inv",
            ".input_scale_inv",
            "_input_scale_inv",
            "_weight_scale",
            "_input_scale",
        )

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            for expert_id in range(num_experts):
                weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        def copy_ple_rows_to_tp_embedding(
            emb, loaded_weight: torch.Tensor, row_start: int, row_end: int
        ) -> None:
            tp_start = emb.shard_indices.org_vocab_start_index
            tp_end = emb.shard_indices.org_vocab_end_index
            ov_start = max(row_start, tp_start)
            ov_end = min(row_end, tp_end)
            if ov_start < ov_end:
                local_start = ov_start - tp_start
                src_start = ov_start - row_start
                n_rows = ov_end - ov_start
                emb.weight.data[local_start : local_start + n_rows].copy_(
                    loaded_weight[src_start : src_start + n_rows].to(
                        device=emb.weight.device, dtype=emb.weight.dtype
                    )
                )

        def load_qwen4_exp_ple_shard(name: str, loaded_weight: torch.Tensor) -> bool:
            if ".ngram_embedding.shard_" not in name:
                return False
            import re

            match = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", name)
            if not match:
                return False
            shard_idx = int(match.group(1))
            mod_prefix = name[: name.index(".ngram_embedding.shard_")]
            ple_mod = ple_modules.get(mod_prefix)
            if ple_mod is None:
                return False
            emb = ple_mod.ngram_embedding
            if (
                loaded_weight.dtype == torch.float8_e4m3fn
                and emb.weight.dtype != torch.float8_e4m3fn
            ):
                if isinstance(emb, Qwen4ExpPinnedHostEmbedding):
                    # offload gathers from pinned host memory; a swapped-in
                    # pageable tensor would fault in the Triton kernel.
                    raise ValueError(
                        "fp8 PLE auto-switch is unsupported with "
                        "ple_offload_embedding; set "
                        'text_config.ple_embedding_dtype="float8_e4m3fn" instead'
                    )
                logger.info(
                    "PLE embedding switched to fp8 storage: %s (%s)",
                    mod_prefix,
                    tuple(emb.weight.data.shape),
                )
                old_weight_data = emb.weight.data
                # StartupWeightLoadManager enforces tensor identity/dtype; this
                # swap breaks that contract if the model is ever enrolled.
                emb.weight = torch.nn.Parameter(
                    torch.empty_like(old_weight_data, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                del old_weight_data
                # params_dict was snapshotted before the loop; drop the stale
                # entry or it pins the old bf16 storage until load end.
                params_dict.pop(f"{mod_prefix}.ngram_embedding.weight", None)
                torch.cuda.empty_cache()
            if (
                emb.weight.dtype == torch.float8_e4m3fn
                and loaded_weight.dtype != torch.float8_e4m3fn
            ):
                if not getattr(load_qwen4_exp_ple_shard, "_warned_downcast", False):
                    load_qwen4_exp_ple_shard._warned_downcast = True
                    logger.warning(
                        "PLE checkpoint shards are %s but the embedding storage "
                        "is fp8 (ple_embedding_dtype / fp8 quant config); "
                        "downcasting is lossy",
                        loaded_weight.dtype,
                    )
            shard_size = (
                emb.org_vocab_size + ple_num_sync_shards - 1
            ) // ple_num_sync_shards
            shard_start = shard_idx * shard_size
            actual_rows = loaded_weight.shape[0]
            shard_end = shard_start + actual_rows
            copy_ple_rows_to_tp_embedding(emb, loaded_weight, shard_start, shard_end)
            loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
            return True

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())

        ple_modules = {
            mod_name: mod
            for mod_name, mod in self.named_modules()
            if isinstance(mod, Qwen4ExpNGramEmbedding)
        }
        text_config = getattr(self.config, "text_config", self.config)
        ple_num_sync_shards = int(
            getattr(
                text_config,
                "split_ngram_parts",
                getattr(self.config, "split_ngram_parts", 512),
            )
        )
        loaded_params: Set[str] = set()
        loaded_buffers: Set[str] = set()
        loaded_shard_params: Set[str] = set()
        skipped_visual_count = 0

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name and self.language_model_only:
                skipped_visual_count += 1
                continue
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if name.endswith(".k_proj.k_scale"):
                name = name.replace(".k_proj.k_scale", ".attn.k_scale")
            elif name.endswith(".v_proj.v_scale"):
                name = name.replace(".v_proj.v_scale", ".attn.v_scale")

            if self._load_qwen4_exp_ple_buffer(
                name, loaded_weight, buffers, loaded_buffers
            ):
                continue
            if load_qwen4_exp_ple_shard(name, loaded_weight):
                continue
            if ".ple.ple_embedding.ngram_embedding." in name and name.endswith(
                ".weight"
            ):
                raise ValueError(
                    f"unsupported PLE weight layout (expected shard_N shards): {name}"
                )

            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
                and "lm_head.weight" in params_dict
            ):
                lm_head_param = params_dict["lm_head.weight"]
                weight_loader = getattr(
                    lm_head_param, "weight_loader", default_weight_loader
                )
                weight_loader(lm_head_param, loaded_weight)

            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            is_fused_expert = (
                "experts.gate_up_proj" in name or "experts.down_proj" in name
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name or "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if (
                    mapped_name.endswith(ignore_suffixes)
                    and mapped_name not in params_dict
                ):
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                is_expert_weight = False
                current_expert_params_mapping = (
                    fused_expert_params_mapping
                    if is_fused_expert
                    else expert_params_mapping
                )
                for mapping in current_expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    is_expert_weight = True
                    mapped_name = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                gate_weight,
                                "w1",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                up_weight,
                                "w3",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                        else:
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                    else:
                        if (
                            mapped_name.endswith(ignore_suffixes)
                            and mapped_name not in params_dict
                        ):
                            continue
                        param = params_dict[mapped_name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = mapped_name
                    break
                else:
                    if is_expert_weight:
                        continue
                    if "visual" in name:
                        name = name.replace("attn.qkv.", "attn.qkv_proj.")
                        name = name.replace("model.visual.", "visual.")
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if name.endswith("_scale") and name not in params_dict:
                        assert abs(loaded_weight.item() - 1.0) < 1e-6, (
                            f"Expected 1.0, got {loaded_weight.item()} in skipped {name}"
                        )
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(
                            "Parameter %s not found while loading Qwen4-Exp VL weights",
                            name,
                        )
                        continue
            loaded_params.add(name)

        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)

        if skipped_visual_count > 0:
            logger.info(
                f"[language_model_only] Qwen4 load_weights: skipped "
                f"{skipped_visual_count} visual weights"
            )

        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()

        return loaded_params

    def precompile_kernels_after_loading(self) -> None:
        from sglang.srt.layers.quantization.unquant import precompile_splitk_tactics

        if precompile_splitk_tactics():
            logger.info("Precompiled BF16 split-K GEMM tactics for Qwen4-Exp")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        if getattr(text_config, "num_experts", None) is None:
            return None
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


EntryClass = [Qwen4ExpForConditionalGeneration]
