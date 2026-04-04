# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py

import logging
from enum import Enum
from typing import List, Optional, Tuple

import torch

from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.compilation.piecewise_context_manager import (
    get_forward_context,
    is_in_piecewise_cuda_graph,
)
from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import (
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    create_kt_config_from_server_args,
)
from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMxInt4MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import CompressedTensorsW8A8Fp8MoE
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    round_up,
)
from sglang.srt.utils.custom_op import register_custom_op

_is_hip = is_hip()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

logger = logging.getLogger(__name__)

import threading
from sglang.srt.utils.common import is_pin_memory_available
from sglang.srt.utils.common import MoeComputeStrategy
from sglang.srt.utils.common import is_lk_moe_feature_enabled, get_moe_compute_strategy, is_lk_moe_cpu_layer, is_lk_moe_gpu_resident_layer, is_lk_moe_gpu_prefill_layer, get_gpu_prefetch_window, get_gpu_prefill_min_batch_size, is_lk_moe_use_gpu_prefill, is_lk_moe_quant_on_gpu, LkMoeSerialGuard
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsFusedMoEMethod
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
if is_lk_moe_feature_enabled():
    import  lk_moe
    GGML_TYPE_TO_TORCH_DTYPE = {
        0: torch.float32,    # GGML_TYPE_F32
        1: torch.float16,    # GGML_TYPE_F16
        30: torch.bfloat16,  # GGML_TYPE_BF16 
    }
 
    SUPPORTED_GGML_QUANT_TYPES = {
        2,  # GGML_TYPE_Q4_0
        3,  # GGML_TYPE_Q4_1
        8,  # GGML_TYPE_Q8_0
        12, # GGML_TYPE_Q4_K
        13, # GGML_TYPE_Q5_K
        14, # GGML_TYPE_Q6_K
        23, # GGML_TYPE_IQ4_XS
        24, # GGML_TYPE_I8 
    }
 
    def is_ggml_type_supported(ggml_type): 
        if ggml_type in {0, 1, 30}:
            return True 
        if ggml_type in SUPPORTED_GGML_QUANT_TYPES:
            return True
        return False  
    
else:
    logger.error("Failed to import lk_moe module or LVLLM_MOE_NUMA_ENABLED is not set to 1, lk::MOE implementation will not be available")
    
def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:
    a2a_backend = get_moe_a2a_backend()
    if a2a_backend.is_none():
        return StandardDispatcher(moe_runner_config)
    elif (
        a2a_backend.is_deepep()
        or a2a_backend.is_mooncake()
        or a2a_backend.is_mori()
        or a2a_backend.is_nixl()
    ):
        return MaybeTboDeepEPDispatcher(
            group=(
                get_tp_group().device_group
                if not a2a_backend.is_mori()
                else get_tp_group()
            ),
            router_topk=moe_runner_config.top_k,
            permute_fusion=True,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
            params_dtype=moe_runner_config.params_dtype,
            deepep_mode=get_deepep_mode(),
            async_finish=True,
            return_recv_hook=True,
        )
    elif a2a_backend.is_ascend_fuseep():
        from sglang.srt.layers.moe.token_dispatcher import NpuFuseEPDispatcher

        return NpuFuseEPDispatcher(
            group=get_tp_group().device_group,
            router_topk=moe_runner_config.top_k,
            permute_fusion=True,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
            params_dtype=moe_runner_config.params_dtype,
        )

    elif a2a_backend.is_flashinfer():
        return FlashinferDispatcher(
            group=get_tp_group().device_group,
            router_topk=moe_runner_config.top_k,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
        )
    else:
        raise NotImplementedError(f"Unsupported a2a backend: {a2a_backend}")


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"

_current_stream: Optional[torch.cuda.Stream] = None

def get_current_stream() -> Optional[torch.cuda.Stream]: 
    if not torch.cuda.is_available():
        return None
    
    global _current_stream
     
    if _current_stream is None:
         _current_stream = torch.cuda.current_stream()
    
    return _current_stream

class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to apply all_reduce on the output of the layer
        quant_config: Quantization configuration.
        inplace: suggestion to compute inplace (modify input activation).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
        gemm1_alpha: Optional[float] = None,
        gemm1_clamp_limit: Optional[float] = None,
        use_weight_loader_fused: bool = False,
        with_bias=False,
        routing_method_type: Optional[RoutingMethodType] = None,
        is_gated: bool = True,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        
        self._lk_moe_init_lock = threading.Lock()  
        
        self.gpu_prefetch_window = get_gpu_prefetch_window()
        self.lk_moe = None
        self.lk_moe_config = None 
        self.is_gpu_resident_layer = is_lk_moe_gpu_resident_layer(self.layer_id) 
        self.is_gpu_prefill_layer = is_lk_moe_gpu_prefill_layer(self.layer_id)
        self.is_cpu_layer = is_lk_moe_cpu_layer(self.layer_id)
        self._lk_moe_guard = LkMoeSerialGuard()
        self.gpu_prefill_min_batch_size = get_gpu_prefill_min_batch_size()
        self.max_num_batched_tokens = get_global_server_args().chunked_prefill_size
        if self.gpu_prefill_min_batch_size > self.max_num_batched_tokens:
            logger.error(
                f"gpu_prefill_min_batch_size ({self.gpu_prefill_min_batch_size}) "
                f"must be less than or equal to chunked_prefill_size "
                f"({self.max_num_batched_tokens})"
            )
        self.max_num_group_batch_size = self.get_max_num_group_batch_size()
        server_args = get_global_server_args()
        model_arch = server_args.model_config.hf_config.architectures[0]
        self.check_nan_in_output = (model_arch in ["MiniMaxM2ForCausalLM", "Step3p5ForCausalLM"])
        self.has_gate_proj  = not (model_arch == "NemotronHForCausalLM")

        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.moe_ep_rank = get_moe_expert_parallel_rank()
        self.moe_tp_size = get_moe_tensor_parallel_world_size()
        self.moe_tp_rank = get_moe_tensor_parallel_rank()
        assert (num_experts - num_fused_shared_experts) % self.moe_ep_size == 0
        self.num_local_experts = (
            num_experts - num_fused_shared_experts
        ) // self.moe_ep_size + num_fused_shared_experts

        self.expert_mask_gpu = None

        assert intermediate_size % self.moe_tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
        self.reduce_results = reduce_results
        self.use_presharded_weights = use_presharded_weights

        self.use_triton_kernels = get_moe_runner_backend().is_triton_kernels()
        self.use_flashinfer_trtllm_moe = (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        )

        # flashinfer_trtllm kernel requires intermediate_size to be a multiple of 128
        # Pad the intermediate_size_per_partition if necessary
        if (
            self.use_flashinfer_trtllm_moe
            and self.intermediate_size_per_partition % 128 != 0
        ):
            self.intermediate_size_per_partition = round_up(
                self.intermediate_size_per_partition, 128
            )

        self.quant_config = quant_config
        self.use_flashinfer_mxfp4_moe = get_moe_runner_backend().is_flashinfer_mxfp4()
        # TODO maybe we should remove this `if`, since `Mxfp4MoEMethod` does another round-up logic
        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.use_flashinfer_mxfp4_moe
        ):
            hidden_size = round_up(hidden_size, 256)
        self.hidden_size = hidden_size

        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            layer_id=layer_id,
            top_k=top_k,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
            is_gated=is_gated,
            routing_method_type=routing_method_type,
        )

        self.quant_method: Optional[FusedMoEMethodBase] = None
        server_args = get_global_server_args()
        kt_config = create_kt_config_from_server_args(server_args, layer_id)
        if kt_config is not None:
            if quant_config is not None:
                gpu_method = quant_config.get_quant_method(self, prefix)
            else:
                gpu_method = UnquantizedFusedMoEMethod(self.use_triton_kernels)
            self.quant_method = KTEPWrapperMethod(gpu_method, kt_config)
        else:
            if quant_config is not None:
                self.quant_method = quant_config.get_quant_method(self, prefix)
            if self.quant_method is None:
                self.quant_method = UnquantizedFusedMoEMethod(
                    self.use_triton_kernels, self.use_flashinfer_trtllm_moe
                )
                
        self.max_running_requests = server_args.max_running_requests
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=(
                self.weight_loader
                if not use_weight_loader_fused
                else self.weight_loader_fused
            ),
            with_bias=with_bias,
            moe_intermediate_size=intermediate_size,
        )

        self.quant_method.create_moe_runner(self, self.moe_runner_config)
        self.dispatcher = create_moe_dispatcher(self.moe_runner_config)

        self.should_fuse_routed_scaling_factor_in_topk = (
            isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)
            or (
                isinstance(self.quant_method, Fp8MoEMethod)
                and (
                    get_moe_runner_backend().is_cutlass()
                    or get_moe_runner_backend().is_flashinfer_trtllm_routed()
                )
            )
            or (
                isinstance(self.quant_method, UnquantizedFusedMoEMethod)
                and get_moe_runner_backend().is_flashinfer_trtllm_routed()
            )
        )

        self.routing_method_type = routing_method_type

        # overlap args
        self.down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        if self.quant_method is not None and hasattr(self.quant_method, "runner"):
            self.runner = self.quant_method.runner

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            if self.moe_runner_config.is_gated:
                param_data[expert_id][idx] = loaded_weight
            else:
                param_data[expert_id] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        # Load grouped weight scales for group quantization
        # or model weights
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
        elif shard_id in ("w1", "w3", "w13"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        assert shard_id in {"w1", "w3", "w13"}

        if is_bias:
            # if this weight is a bias, the last dimension must be the sharded dimension
            shard_dim = -1

        if shard_id in {"w1", "w3"} and self.moe_runner_config.is_gated:
            # non-fused version
            shard_size = expert_data.shape[shard_dim] // 2
        elif shard_id in {"w13"} or (
            shard_id in {"w1", "w3"} and not self.moe_runner_config.is_gated
        ):
            # fused version
            shard_size = expert_data.shape[shard_dim]
        else:
            raise NotImplementedError

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        # w3, up_proj: Load into second logical weight of w13.
        # trtllm cutlass kernel assumes differently
        switch_w13 = getattr(self.quant_method, "load_up_proj_weight_first", False)
        if (
            (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3")
        ) and self.moe_runner_config.is_gated:
            start = shard_size
        else:
            start = 0

        # Use narrow_padded_param_and_loaded_weight for:
        # 1. CPU (always)
        # 2. GPU with flashinfer_trtllm padding (when intermediate_size is padded to 128)
        # This handles the case where the loaded weights are smaller than the padded expert_data
        use_padded_loading = _is_cpu or self.use_flashinfer_trtllm_moe
        if use_padded_loading:
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                start,
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            if not self.use_presharded_weights:
                if not is_bias and self.use_triton_kernels:
                    # do not transpose for bias
                    loaded_weight = loaded_weight.transpose(-2, -1)
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )

            expert_data = expert_data.narrow(shard_dim, start, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        """Load w2 weights for down projection.

        Args:
            expert_data: The expert data tensor to load into
            shard_dim: The dimension to shard along
            shard_id: The shard ID (must be "w2")
            loaded_weight: The weight tensor to load from
            tp_rank: The tensor parallel rank
        """
        if not isinstance(expert_data, torch.Tensor) or not isinstance(
            loaded_weight, torch.Tensor
        ):
            raise ValueError("expert_data and loaded_weight must be torch.Tensor")

        if (
            self.quant_config is not None
            and "modelopt" in self.quant_config.get_name()
            and (expert_data.dim() != 2 or loaded_weight.dim() != 2)
        ):
            raise ValueError(
                f"Expected 2D tensors, got expert_data shape {expert_data.shape} and loaded_weight shape {loaded_weight.shape}"
            )

        if shard_id != "w2":
            raise ValueError(f"shard_id must be 'w2', got {shard_id}")

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        if is_bias:
            # this expert_data is a bias, not weight,
            # for w2_weight_bias in TP, it does not need to be sharded
            shard_size = expert_data.shape[-1]
        else:
            # this parameter is a weight matrix
            # for w2 in TP, it shards the input_features, i.e., shard_dim=2
            shard_size = expert_data.shape[shard_dim]

        # Use narrow_padded_param_and_loaded_weight for:
        # 1. CPU (always)
        # 2. GPU with flashinfer_trtllm padding (when intermediate_size is padded to 128)
        # This handles the case where the loaded weights are smaller than the padded expert_data
        use_padded_loading = _is_cpu or self.use_flashinfer_trtllm_moe
        if use_padded_loading:
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                0,  # param_data_start
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            if not is_bias and not self.use_presharded_weights:
                if self.use_triton_kernels:
                    loaded_weight = loaded_weight.transpose(-2, -1)
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )

        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        num_global_routed_experts = self.num_experts - self.num_fused_shared_experts
        num_local_routed_experts = (
            self.num_local_experts - self.num_fused_shared_experts
        )
        start_idx = self.moe_ep_rank * num_local_routed_experts
        end_idx = (self.moe_ep_rank + 1) * num_local_routed_experts
        if start_idx <= expert_id < end_idx:
            return expert_id - start_idx
        elif (
            self.num_fused_shared_experts > 0 and expert_id >= num_global_routed_experts
        ):
            return expert_id - num_global_routed_experts + num_local_routed_experts
        else:
            return -1

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: Optional[int],
    ) -> None:
        # if expert_id is None, then
        # all the experts are loaded at the same time
        if (
            not expert_id
            and self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        global_expert_location_metadata = get_global_expert_location_metadata()
        if global_expert_location_metadata is None:
            if not getattr(param, "_sglang_require_global_experts", False):
                expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
                if expert_id == -1:
                    return

            self._weight_loader_impl(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return

        if expert_id >= self.num_experts - self.num_fused_shared_experts:
            # This is a shared expert.
            physical_expert_ids = [expert_id]
        else:
            require_global_experts = getattr(
                param, "_sglang_require_global_experts", False
            )
            physical_expert_ids = (
                global_expert_location_metadata.logical_to_all_physical(
                    self.layer_id, expert_id, require_global_experts
                )
            )

        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        # WARN: This makes the `expert_id` mean "local" and "global" in different cases
        if not getattr(param, "_sglang_require_global_experts", False):
            expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if expert_id < 0 or expert_id >= self.num_local_experts:
                return

        if isinstance(
            self.quant_method,
            KTEPWrapperMethod,
        ):
            if self.quant_method.num_gpu_experts != -1:
                if expert_id >= self.quant_method.num_gpu_experts:
                    return

        self._weight_loader_impl(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )

    def _weight_loader_impl(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        tp_rank = self.moe_tp_rank

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        method = self.quant_method
        if hasattr(self, "scheme"):
            method = self.scheme
        if method.__class__.__name__ == "KTEPWrapperMethod":
            method = method.gpu_method

        # For flashinfer TRT-LLM BF16 path, process_weights_after_loading reshapes
        # expert weights into block layout. During weight update, we must restore
        # canonical load-time shapes before copying checkpoint tensors.
        if isinstance(method, UnquantizedFusedMoEMethod):
            method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                layer=self,
                param=param,
                weight_name=weight_name,
            )

        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MarlinMoE",
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Flashinfer assumes w31 format for w13_weight. Same for the scales.
        if self.use_flashinfer_trtllm_moe and (
            isinstance(method, ModelOptNvFp4FusedMoEMethod)
            or isinstance(method, Fp8MoEMethod)
            or isinstance(method, UnquantizedFusedMoEMethod)
            or isinstance(method, CompressedTensorsMxInt4MoE)
        ):
            shard_id = {"w1": "w3", "w3": "w1", "w2": "w2"}[shard_id]

        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if self.use_triton_kernels:
            is_transposed = True
        if is_transposed:
            shard_dim = int(not shard_dim)

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust input_scale for e4m3fnuz (AMD)
            if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                loaded_weight = loaded_weight * 2.0

            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                (
                    "compressed" in method.__class__.__name__.lower()
                    or "w4afp8" in self.quant_config.get_name()
                )
                and (param.data[expert_id] != 1).any()
                and ((param.data[expert_id] - loaded_weight).abs() > 1e-5).any()
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        if "ModelOpt" in method.__class__.__name__:
            # Determine per-tensor weight scale patterns based on variant
            is_fp4_variant = isinstance(method, ModelOptNvFp4FusedMoEMethod)

            # FP4 uses "weight_scale_2" for per-tensor, FP8 uses "weight_scale" for per-tensor
            per_tensor_conditions = (
                "weight_scale_2" in weight_name
                if is_fp4_variant
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name

            if per_tensor_conditions:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            return

        # Case weight scales and zero_points
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust INT4 column-wise scaling number to e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 0.5

                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust FP8 per-tensor scaling number for e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 2.0

                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        if (
            "bias" in weight_name
            and self.quant_config.quant_description["quant_method"] == "modelslim"
        ):
            self._load_per_channel_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def weight_loader_fused(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
    ) -> None:
        tp_rank = self.moe_tp_rank

        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            elif "scale" in weight_name:
                param.data.copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO: check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        method = self.quant_method
        if hasattr(self, "scheme"):
            method = self.scheme
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            else loaded_weight
        )

        if shard_id not in ("w13", "w2"):
            raise ValueError(f"shard_id must be ['w13','w2'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w13": 1, "w2": 2}
        SHARD_ID_TO_SHARDED_DIM_TRANSPOSE = {"w13": 2, "w2": 1}

        expert_data = param.data
        is_bias = expert_data.dim() == 2

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)

        if self.use_triton_kernels:
            is_transposed = True
        shard_dim = (
            SHARD_ID_TO_SHARDED_DIM[shard_id]
            if not is_transposed
            else SHARD_ID_TO_SHARDED_DIM_TRANSPOSE[shard_id]
        )

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
            return
        else:
            logging.warning(
                f"Unsupported weight_name {weight_name} for FusedMoE weight_loader_fused. Nothing is loaded."
            )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if is_in_piecewise_cuda_graph():
            if TopKOutputChecker.format_is_standard(topk_output):
                return moe_forward_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.topk_weights,
                    topk_output.topk_ids,
                    topk_output.router_logits,
                    self.layer_id,
                )
            elif TopKOutputChecker.format_is_bypassed(topk_output):
                return fused_moe_bypassed_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.router_logits,
                    topk_output.topk_config.top_k,
                    topk_output.topk_config.topk_group,
                    topk_output.topk_config.num_expert_group,
                    topk_output.topk_config.correction_bias,
                    topk_output.topk_config.renormalize,
                    self.layer_id,
                )
            else:
                # Make sure there is torch lib op registration for the whole moe layer
                return self.forward_impl(hidden_states, topk_output)
        else:
            if not self.should_use_gpu_prefill(hidden_states): 
                return self.forward_impl(hidden_states, topk_output)
            else:
            
                moe_layers, forward_batch = get_moe_context() 
                _current_forward_context = MoEForwardContext(moe_layers, forward_batch)
        
                moe_prefetch(self, self.layer_id, hidden_states, _current_forward_context, get_gpu_prefetch_window())
                moe_wait_prefetch(self, hidden_states, _current_forward_context)
                
                fused_output =  self.forward_impl(hidden_states, topk_output) 
                moe_cleanup(self, self.layer_id, hidden_states, _current_forward_context) 
                return fused_output
            
    def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        origin_hidden_states_dim = hidden_states.shape[-1]
        assert self.quant_method is not None

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        if _use_aiter and self.dispatcher.local_expert_mapping is not None:
            self.expert_mask_gpu = (
                (
                    (self.dispatcher.local_expert_mapping >= 0)
                    & (self.dispatcher.local_expert_mapping < self.num_local_experts)
                )
                .to(torch.int32)
                .to(device="cuda")
            )

        combine_input = self.run_moe_core(
            dispatch_output=dispatch_output,
        )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            final_hidden_states = self.dispatcher.combine(combine_input=combine_input)

            # TODO: should we add some conditions here?
            final_hidden_states = final_hidden_states[
                ..., :origin_hidden_states_dim
            ].contiguous()

        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    def run_moe_core(self, dispatch_output: DispatchOutput) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        if not self.is_gpu_resident_layer and get_is_capture_mode():
            lk_result = self.forward_lk( 
                dispatch_output.hidden_states,
                dispatch_output.topk_output.topk_weights, 
                dispatch_output.topk_output.topk_ids,
            ) 
            return StandardCombineInput(hidden_states=lk_result)
        elif not self.is_gpu_resident_layer and not self.should_use_gpu_prefill(dispatch_output.hidden_states):
            lk_result = self.forward_lk( 
                dispatch_output.hidden_states,
                dispatch_output.topk_output.topk_weights, 
                dispatch_output.topk_output.topk_ids,
            ) 
            return StandardCombineInput(hidden_states=lk_result)
        elif self.should_use_gpu_prefill(dispatch_output.hidden_states) and not isinstance(self.quant_method, UnquantizedFusedMoEMethod):
            from sglang.srt.layers.quantization.gguf import fused_moe_gguf
            prefill_result = fused_moe_gguf(
                                dispatch_output.hidden_states,
                                self.w13_weight.data,
                                self.w2_weight.data,
                                dispatch_output.topk_output.topk_weights,
                                dispatch_output.topk_output.topk_ids,
                                2,
                                2,
                                self.moe_runner_config.activation,
                            )
            return StandardCombineInput(hidden_states=prefill_result)
        else:
            # TODO: consider using symmetric memory
            return self.quant_method.apply(
                layer=self,
                dispatch_output=dispatch_output,
            )

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    @classmethod
    def make_expert_params_mapping_fused(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
    ):
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
        ]

    @classmethod
    def make_expert_params_mapping_fused_mxfp4(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
        ckpt_gate_up_proj_scale_name: str,
        ckpt_down_proj_scale_name: str,
    ):
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
            (
                "experts.w13_weight_scale",
                f"experts.{ckpt_gate_up_proj_scale_name}",
                "w13",
            ),
            ("experts.w2_weight_scale", f"experts.{ckpt_down_proj_scale_name}", "w2"),
        ]

    @classmethod
    def make_expert_input_scale_params_mapping(
        cls,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id)
        return [
            (
                "experts.w13_" if shard_id in ["w1", "w3"] else "experts.w2_",
                f"experts.{expert_id}.{shard_id}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id in ["w1", "w2", "w3"]
        ]

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ):
        if hasattr(self, "runner"):
            self.runner.set_overlap_args(down_gemm_overlap_args, meta_overlap_args)
        else:
            # TODO: remove this branch after MoE refactor
            self.down_gemm_overlap_args = down_gemm_overlap_args
            self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        if hasattr(self, "runner"):
            self.runner.clear_overlap_args()
        else:
            # TODO: remove this branch after MoE refactor
            self.down_gemm_overlap_args = None
            self.meta_overlap_args = None
            
    def get_max_num_group_batch_size(self) -> int:
        max_num_batched_tokens = get_global_server_args().chunked_prefill_size
        
        if is_lk_moe_use_gpu_prefill():
            group_batch_size = min(max_num_batched_tokens, get_gpu_prefill_min_batch_size()) + 128
        else:
            group_batch_size = min(4096, max_num_batched_tokens) + 128
         
        return group_batch_size
    
    def global_to_local_expert_ids(self, topk_ids): 
        expert_map = self._expert_map.to(topk_ids.device)
        max_idx = len(self._expert_map) - 1
         
        clamped = torch.clamp(topk_ids, 0, max_idx)
        result = expert_map[clamped]
         
        mask = topk_ids < 0
        result[mask] = -1
        
        return result
    
    def should_use_gpu_prefill(self, hidden_states: torch.Tensor) -> bool:
        if get_is_capture_mode():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        return self.is_gpu_prefill_layer and hidden_states.size(0) >= self.gpu_prefill_min_batch_size
            
    def _get_ggml_type_from_quant_config(self,  quant_config, layer_idx, weight_type):  
        if layer_idx < len(quant_config.moe_weight_type_map):
            layer_info = quant_config.moe_weight_type_map[layer_idx]
            if layer_info and weight_type in layer_info:
                weight_name = layer_info[weight_type]
                quant_name_to_type = {
                    'F32': 0,     # GGML_TYPE_F32
                    'F16': 1,     # GGML_TYPE_F16
                    'BF16': 30,   # GGML_TYPE_BF16
                    'Q4_0': 2,    # GGML_TYPE_Q4_0
                    'Q4_1': 3,    # GGML_TYPE_Q4_1
                    'Q8_0': 8,    # GGML_TYPE_Q8_0
                    'Q4_K': 12,   # GGML_TYPE_Q4_K
                    'Q5_K': 13,   # GGML_TYPE_Q5_K
                    'Q6_K': 14,   # GGML_TYPE_Q6_K
                    'IQ4_XS': 23, # GGML_TYPE_IQ4_XS
                    'I8': 24,     # GGML_TYPE_I8
                }
                return quant_name_to_type.get(weight_name, None)
            
        raise ValueError(f"Weight type {layer_idx}.{weight_type} not found in quant_config") 
    
    def _zero_tensor(self, tensor: torch.Tensor):
        if tensor is not None:
            tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
            
    def process_weights_after_loading(self):
        if self.is_gpu_resident_layer:
            logger.info(f"Initialized lk_moe with {self.moe_runner_config.num_local_experts} experts for layer {self.layer_id} [" + 
            ("CPU" if not self.is_gpu_resident_layer else "GPU") + "]")
            return
        try:    
            if is_lk_moe_use_gpu_prefill() and not hasattr(FusedMoE, '_gpu_weights_placeholder'): 
                import threading
                
                 
                batch_size = min(self.max_running_requests, 4) 
                
                FusedMoE._batch_lock = threading.Lock()  
                
                FusedMoE._cpu_weights_placeholder = {} 
                FusedMoE._gpu_weights_placeholder = {}
                FusedMoE._batch_usage = {}
                param_names = ["w13_weight", "w2_weight"]
                
                for batch_id in range(batch_size):
                    FusedMoE._cpu_weights_placeholder[batch_id] = create_cpu_weights(self) 
                    FusedMoE._gpu_weights_placeholder[batch_id] = {}
                    for param_name in param_names:
                        FusedMoE._gpu_weights_placeholder[batch_id][param_name] = torch.zeros_like(FusedMoE._cpu_weights_placeholder[batch_id][param_name], device=torch.cuda.current_device())
                    FusedMoE._batch_usage[batch_id] = False
                logger.info(f"Initialized lk_moe gpu prefill buffers with {batch_size} batches")
                 
            find_weight = False  
            with torch.no_grad():
                if isinstance(self.quant_method, CompressedTensorsFusedMoEMethod) and not (hasattr(self, "scheme") and isinstance(self.scheme, CompressedTensorsW8A8Fp8MoE)):
    
                    self._process_compressed_tensors_weights()
                    find_weight = True 
                    
                if isinstance(self.quant_method, Fp8MoEMethod):
                    strategy = get_moe_compute_strategy()
                    if strategy == MoeComputeStrategy.KEEP:
                        self._process_fp8_weights(self.quant_method.block_quant)
                    else:
                        self._process_block_weights_quant(strategy)
                    find_weight = True
                    
                if hasattr(self, "scheme") and isinstance(self.scheme, CompressedTensorsW8A8Fp8MoE):
                    strategy = get_moe_compute_strategy() 
                    self.quant_method.block_quant = False
                    if strategy == MoeComputeStrategy.KEEP:
                        self._process_fp8_weights(False)
                    else:
                        self._process_channel_weights_quant(strategy)
                    find_weight = True
                if isinstance(self.quant_method, UnquantizedFusedMoEMethod): 
                    self._process_regular_weights()
                    find_weight = True
                
                if not find_weight: 
                    logger.error("weight not found in layer, quant_method: %s", self.quant_method) 
                    return
                
                self._initialize_cuda_graph_buffers()
                logger.info(f"Initialized lk_moe with {self.moe_runner_config.num_local_experts} experts for layer {self.layer_id} [" + 
                ("CPU" if not self.is_gpu_resident_layer else "GPU") + "]")
        except Exception as e:
            logger.error(f"Failed to initialize lk_moe: {e}") 
            self.lk_moe = None
            self.lk_moe_config = None 
            
    def clean_weights_after_loading(self):
        if self.is_gpu_resident_layer:
            return
        try:  
            with torch.no_grad():
                if isinstance(self.quant_method, UnquantizedFusedMoEMethod): 
                    if hasattr(self, "w13_weight") and hasattr(self, "w2_weight"):
                        setattr(self, "w13_weight", torch.nn.Parameter(torch.empty(0,  device=torch.cuda.current_device()), requires_grad=False))
                        setattr(self, "w2_weight", torch.nn.Parameter(torch.empty(0,  device=torch.cuda.current_device()), requires_grad=False))
                    
                if isinstance(self.quant_method, Fp8MoEMethod) or hasattr(self, "scheme") and isinstance(self.scheme, CompressedTensorsW8A8Fp8MoE):
                    param_names = [
                        "w13_weight",
                        "w2_weight",  
                    ]
                    scale_names = [
                        "w13_weight_scale_inv" if self.quant_method.block_quant else "w13_weight_scale",
                        "w2_weight_scale_inv" if self.quant_method.block_quant else "w2_weight_scale",  
                    ]
                    quant_config_names = [
                        "w1_scale",
                        "w2_scale",
                    ]

                    for param_name in param_names: 
                        if hasattr(self, param_name):
                            setattr(self, param_name, torch.nn.Parameter(
                                torch.empty(0, device=torch.cuda.current_device()), 
                                requires_grad=False
                            ))
                    for scale_name in scale_names:
                        if hasattr(self, scale_name):
                            setattr(self, scale_name, torch.nn.Parameter(
                                torch.empty(0, device=torch.cuda.current_device()), 
                                requires_grad=False
                            ))
                    for quant_config_name in quant_config_names:
                        if hasattr(self, "moe_quant_config") and hasattr(self.moe_quant_config, quant_config_name):
                            setattr(self.moe_quant_config, quant_config_name, torch.nn.Parameter(
                                torch.empty(0, device=torch.cuda.current_device()), 
                                requires_grad=False
                            ))
                                
                if isinstance(self.quant_method, CompressedTensorsFusedMoEMethod) and not (hasattr(self, "scheme") and isinstance(self.scheme, CompressedTensorsW8A8Fp8MoE)):
                    param_names = [
                        "w13_weight_packed",
                        "w2_weight_packed", 
                        "w13_weight_scale",
                        "w2_weight_scale", 
                        "w13_weight_g_idx",
                        "w2_weight_g_idx",
                        "w13_g_idx_sort_indices",
                        "w2_g_idx_sort_indices",
                        "w13_weight_shape",
                        "w2_weight_shape", 
                    ] 
         
                    for param_name in param_names: 
                        setattr(self, param_name, torch.nn.Parameter(
                                    torch.empty(0, device=torch.cuda.current_device()), 
                                    requires_grad=False
                                )) 
                    
                
            import gc
            gc.collect()
        except Exception as e:
            logger.error(f"Failed to initialize lk_moe: {e}") 
            self.lk_moe = None
            self.lk_moe_config = None
            
    def get_ggml_type_from_dtype(self, dtype):
            if dtype == torch.float32:
                return 0  # GGML_TYPE_F32
            elif dtype == torch.float16:
                return 1  # GGML_TYPE_F16
            elif dtype == torch.bfloat16:
                return 30  # GGML_TYPE_BF16
            else:
                raise ValueError(f"Unsupported dtype {dtype}")
    
    def _get_processes_info(self) -> tuple[int, int, int]: 
        if self.moe_ep_size > 1:
            return self.moe_ep_size, self.moe_ep_rank, torch.cuda.current_device()
        return self.moe_tp_size, self.moe_tp_rank, torch.cuda.current_device()
                   
    def _process_gguf_weights(self):  
        raise ValueError("GGUF Weights are not supported for lk moe ...")  
    
        from vllm.model_executor.models.utils import extract_layer_index
        layer_idx = extract_layer_index(self.layer_id)
        gate_ggml_type = self._get_ggml_type_from_quant_config(self.quant_config, layer_idx, 'gate')
        up_ggml_type = self._get_ggml_type_from_quant_config(self.quant_config, layer_idx, 'up')
        down_ggml_type = self._get_ggml_type_from_quant_config(self.quant_config, layer_idx, 'down')
        
        assert gate_ggml_type == up_ggml_type, f"Gate and Up weights must have the same GGML type, got {gate_ggml_type} and {up_ggml_type}"
  
        if not is_ggml_type_supported(gate_ggml_type) or not is_ggml_type_supported(up_ggml_type) or not is_ggml_type_supported(down_ggml_type) \
            or gate_ggml_type != up_ggml_type:  
            raise ValueError(f"GGML type {gate_ggml_type} or {up_ggml_type} or {down_ggml_type} is not supported for layer {layer_idx}")
             
        hidden_ggml_type = self.get_ggml_type_from_dtype(self.moe_runner_config.params_dtype)
        w13_ggml_type = gate_ggml_type
        w2_ggml_type = down_ggml_type
        
        num_experts, total_intermediate_size, hidden_size = self.w13_qweight.shape
        intermediate_size = total_intermediate_size // 2
        assert intermediate_size == self.intermediate_size_per_partition, f"Intermediate size {intermediate_size} must be equal to intermediate_size_per_partition {self.intermediate_size_per_partition}"
        assert self.w2_qweight.shape == (num_experts, self.hidden_size, self.w2_qweight.shape[2]), f"Down weight shape {self.w2_qweight.shape} must be (num_experts, hidden_size, w2_qweight.shape[2])"
        
        
        w13_ptr = self.w13_qweight.contiguous().data_ptr()
        
        w2_ptr = self.w2_qweight.contiguous().data_ptr()
        
        num_processes, process_id, gpu_id = self._get_processes_info()
        
        self.lk_moe_config = lk_moe.MOEConfig(
            num_processes,                # num_processes
            process_id,                   # process_id
            gpu_id,                       # gpu_id
            self.moe_runner_config.num_local_experts,        # expert_num
            has_gate_proj,                 # has_gate_proj
            self.top_k,                    # routed_expert_num
            self.hidden_size,              # hidden_size
            self.intermediate_size_per_partition,             # intermediate_size
            32,                            # stride
            10,                            # group_min_len
            self.max_num_group_batch_size,                          # group_max_len
            hidden_ggml_type,           
            w13_ggml_type,                # gate_type  
            w2_ggml_type,                # down_type   
            w13_ptr,                 # w13_ptr 
            w2_ptr,                 # w2_ptr
        )
        self.lk_moe = lk_moe.MOE(self.lk_moe_config) 
          
        del w13_ptr, w2_ptr
        del self.w13_qweight, self.w2_qweight
 
        import gc
        gc.collect() 
        
    
    def _block_scale_broadcast_fixed(self, scale, target_shape, group_shape): 
        if torch.is_tensor(group_shape): 
            group_shape = group_shape.tolist()
       
        target_rows, target_cols = target_shape
        block_rows, block_cols = group_shape
         
        num_groups_m = target_rows // block_rows
        num_groups_n = target_cols // block_cols
         
        assert scale.shape == (num_groups_m, num_groups_n) 
         
        scale_expanded = scale.repeat_interleave(block_rows, dim=0) 
        scale_expanded = scale_expanded.repeat_interleave(block_cols, dim=1)
         
        assert scale_expanded.shape == (target_rows, target_cols) 
        
        return scale_expanded
     
    def _process_compressed_tensors_weights(self): 
         
        w13_weight = self.w13_weight_packed.cpu().transpose(1, 2).contiguous().view(torch.uint8) 
        w2_weight = self.w2_weight_packed.cpu().transpose(1, 2).contiguous().view(torch.uint8) 
        w13_scale = self.w13_weight_scale.cpu().transpose(1, 2).contiguous()
        w2_scale = self.w2_weight_scale.cpu().transpose(1, 2).contiguous()  
        
        hidden_ggml_type = self.get_ggml_type_from_dtype(self.moe_runner_config.params_dtype)
        scale_ggml_type = self.get_ggml_type_from_dtype(w13_scale.dtype)
        
        weights_config = self.quant_method.quantization_config.config['config_groups']['group_0']['weights']
        group_size = weights_config['group_size']        # 32
        num_bits = weights_config['num_bits']            # 4
        packed_factor = 8  # 8 （bit) 
        
        w13_weight_ptr = w13_weight.contiguous().data_ptr()
        w2_weight_ptr = w2_weight.contiguous().data_ptr()
        if w13_scale.dtype == torch.bfloat16:
            w13_scale = w13_scale.to(torch.float16).contiguous()
            w2_scale = w2_scale.to(torch.float16).contiguous()
            
            w13_weight_scale_ptr = w13_scale.data_ptr()
            w2_weight_scale_ptr = w2_scale.data_ptr()
            scale_ggml_type = 1
        else:
            w13_weight_scale_ptr = w13_scale.contiguous().data_ptr()
            w2_weight_scale_ptr = w2_scale.contiguous().data_ptr()
        
        num_processes, process_id, gpu_id = self._get_processes_info()
        
        self.lk_moe_config = lk_moe.MOE_WNA16RepackConfig(
            num_processes,                # num_processes
            process_id,                   # process_id
            gpu_id,                       # gpu_id
            self.has_gate_proj,             # has_gate_proj
            self.moe_runner_config.num_local_experts,        # expert_num
            self.top_k,                    # routed_expert_num
            self.hidden_size,                   # hidden_size
            self.intermediate_size_per_partition,             # intermediate_size
            32,                            # stride
            10,                            # group_min_len
            self.max_num_group_batch_size,                         # group_max_len
            hidden_ggml_type,              # hidden_type 
            2,
            2,
            w13_weight_ptr,                     # w13_weight_ptr 
            w2_weight_ptr,                       # w2_weight_ptr   
            w13_weight_scale_ptr,               # w13_weight_scale_ptr
            w2_weight_scale_ptr,                 # w2_weight_scale_ptr
            scale_ggml_type,
            1,
            group_size, 
            packed_factor,                        # packed_factor
            num_bits,                        # num_bits
            group_size,                        # group_size
        ) 
        self.lk_moe = lk_moe.MOE_WNA16Repack(self.lk_moe_config) 
         
        
        del w13_weight_ptr, w2_weight_ptr
        del w13_weight, w2_weight, w13_scale, w2_scale 
    
        import gc
        gc.collect()
            
    
    
    def _process_awq_weights(self): 
        
        w13_qweight = self.w13_qweight
        w2_qweight = self.w2_qweight
        w13_scales = self.w13_scales
        w2_scales = self.w2_scales
        w13_qzeros = self.w13_qzeros
        w2_qzeros = self.w2_qzeros
        raise ValueError("AWQ Weights are not supported for lk moe ...") 
         
 
    def _process_fp8_weights(self, block_quant: bool):   
        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
        
        hidden_ggml_type = self.get_ggml_type_from_dtype(self.moe_runner_config.params_dtype)
        E, N, K = w13_weight.shape 
        assert w2_weight.shape == (E, K, N // 2), f"Down weight shape {w2_weight.shape} must be (E, K, N // 2)"
        
        if block_quant:
            w13_weight_scale = self.w13_weight_scale_inv
            w2_weight_scale = self.w2_weight_scale_inv
            if not w13_weight_scale.dtype == torch.float32 or not w2_weight_scale.dtype == torch.float32:
                raise ValueError("scale type are not supported for lk moe ...")
            group_shape = self.quant_method.quant_config.weight_block_size
            groupN, groupK = group_shape 
            scale_num_experts, scale_total_N, scale_K = w13_weight_scale.shape
            scale_N = scale_total_N // 2
            assert w2_weight_scale.shape == (scale_num_experts, scale_K, scale_N), f"Down weight scale shape {w2_weight_scale.shape} must be (scale_num_experts, scale_K, scale_N)"
        else:
            groupN, groupK = 1, -1
            w13_weight_scale = self.w13_weight_scale
            w2_weight_scale = self.w2_weight_scale
            scale_num_experts, scale_total_N, scale_K = w13_weight_scale.shape 
            assert w13_weight_scale.shape == (scale_num_experts, N, 1), f"Up weight scale shape {w13_weight_scale.shape} must be (scale_num_experts, intermediate_size * 2 , 1)"
            assert w2_weight_scale.shape == (scale_num_experts, K , 1), f"Down weight scale shape {w2_weight_scale.shape} must be (scale_num_experts, hidden_size , 1)"
         
        
        scale_ggml_type = self.get_ggml_type_from_dtype(w13_weight_scale.dtype)
         
        w13_weight_ptr = w13_weight.contiguous().data_ptr()
        w2_weight_ptr = w2_weight.contiguous().data_ptr()
        w13_weight_scale_ptr = w13_weight_scale.contiguous().data_ptr()
        w2_weight_scale_ptr = w2_weight_scale.contiguous().data_ptr()
        
        num_processes, process_id, gpu_id = self._get_processes_info()
        
        self.lk_moe_config = lk_moe.MOE_FP8Config(
            num_processes,                # num_processes
            process_id,                   # process_id
            gpu_id,                       # gpu_id
            self.has_gate_proj,             # has_gate_proj
            self.moe_runner_config.num_local_experts,        # expert_num
            self.top_k,                    # routed_expert_num
            self.hidden_size,                   # hidden_size
            self.intermediate_size_per_partition,             # intermediate_size
            32,                            # stride
            10,                            # group_min_len
            self.max_num_group_batch_size,                          # group_max_len
            hidden_ggml_type,              # hidden_type 
            8,
            8,
            w13_weight_ptr,                     # w13_weight_ptr 
            w2_weight_ptr,                       # w2_weight_ptr   
            w13_weight_scale_ptr,               # w13_weight_scale_ptr
            w2_weight_scale_ptr,                 # w2_weight_scale_ptr 
            scale_ggml_type,
            groupN,                        # groupN
            groupK,                        # groupK
        )
        self.lk_moe = lk_moe.MOE_FP8(self.lk_moe_config) 
          
        del w13_weight_ptr, w2_weight_ptr, w13_weight_scale_ptr, w2_weight_scale_ptr
        del w13_weight, w2_weight, w13_weight_scale, w2_weight_scale
        

   
    def _process_block_weights_quant(self, moe_compute_strategy: MoeComputeStrategy):  
        
        if moe_compute_strategy not in {MoeComputeStrategy.INT4}:
            print(f"Warning: moe_compute_strategy {moe_compute_strategy} is not supported for lk moe , use INT4 instead ...")
            moe_compute_strategy = MoeComputeStrategy.INT4
        
        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
        w13_weight_scale_inv = self.w13_weight_scale_inv
        w2_weight_scale_inv = self.w2_weight_scale_inv
        
        group_shape = self.quant_method.quant_config.weight_block_size
        E, N, K = w13_weight.shape 
         
        assert w2_weight.shape == (E, K, N // 2)
        
        if is_lk_moe_quant_on_gpu():
            dequant_device = torch.cuda.current_device()
        else:
            dequant_device = torch.device("cpu")
        w13_fp32_list = []
        w2_fp32_list = [] 
         
        for expert_idx in range(E): 
            expert_w13_weight = w13_weight[expert_idx].to(device=dequant_device)
            expert_w13_scale_inv = w13_weight_scale_inv[expert_idx].to(device=dequant_device)
            expert_w2_weight = w2_weight[expert_idx].to(device=dequant_device)
            expert_w2_scale_inv = w2_weight_scale_inv[expert_idx].to(device=dequant_device)
             
            w13_float = expert_w13_weight.to(dtype=torch.float32)
            w2_float = expert_w2_weight.to(dtype=torch.float32)
            
            w13_scale_inv_expanded = self._block_scale_broadcast_fixed(
                expert_w13_scale_inv, w13_float.shape, group_shape)
            w13_fp32 = w13_float * w13_scale_inv_expanded
            
            w2_scale_inv_expanded = self._block_scale_broadcast_fixed(
                expert_w2_scale_inv, w2_float.shape, group_shape)
            w2_fp32 = w2_float * w2_scale_inv_expanded
             
            w13_fp32_list.append(w13_fp32)
            w2_fp32_list.append(w2_fp32) 
             
            del expert_w13_weight, expert_w13_scale_inv, expert_w2_weight, expert_w2_scale_inv
            del w13_float, w2_float, w13_scale_inv_expanded, w2_scale_inv_expanded
            del w13_fp32, w2_fp32
         
        w13_fp32_tensor = torch.stack(w13_fp32_list, dim=0).cpu()
        w2_fp32_tensor = torch.stack(w2_fp32_list, dim=0).cpu() 
         
        w13_fp32_list.clear()   
        w2_fp32_list.clear()
        del w13_fp32_list, w2_fp32_list
         
        w13_weight_ptr = w13_fp32_tensor.contiguous().data_ptr()
        w2_weight_ptr = w2_fp32_tensor.contiguous().data_ptr()
          
         
        hidden_ggml_type = self.get_ggml_type_from_dtype(self.moe_runner_config.params_dtype)
         
        group_size = getattr(self.quant_method, 'group_size', 32)
        num_bits = 4 if moe_compute_strategy == MoeComputeStrategy.INT4 else 8
         
        num_processes, process_id, gpu_id = self._get_processes_info()
         
        self.lk_moe_config = lk_moe.MOE_QuantConfig(
            num_processes,                     # num_processes
            process_id,                       # process_id
            gpu_id,                           # gpu_id
            self.has_gate_proj,             # has_gate_proj
            self.moe_runner_config.num_local_experts,            # expert_num
            self.top_k,                        # routed_expert_num
            self.hidden_size,                  # hidden_size
            self.intermediate_size_per_partition,  # intermediate_size
            32,                                # stride
            10,                                # group_min_len
            self.max_num_group_batch_size,   # group_max_len
            hidden_ggml_type,                  # hidden_type 
            0,                                 # w13_weight_data_type: 0 for fp32
            0,                                # w2_weight_data_type: 0 for fp32
            w13_weight_ptr,                    # w13_weight_ptr 
            w2_weight_ptr,                     # w2_weight_ptr   
            group_size,                        # group_size 
            num_bits,                          # num_bits 
        )
         
        self.lk_moe = lk_moe.MOE_Quant(self.lk_moe_config)
         
          
        del w13_weight_ptr, w2_weight_ptr
        del w13_fp32_tensor, w2_fp32_tensor
        
        import gc
        gc.collect()
    
    def _process_block_weights(self):  
 
        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
        w13_weight_scale_inv = self.w13_weight_scale_inv
        w2_weight_scale_inv = self.w2_weight_scale_inv
        
        hidden_ggml_type = self.get_ggml_type_from_dtype(self.moe_runner_config.params_dtype)
        w13_ggml_type = hidden_ggml_type
        w2_ggml_type = hidden_ggml_type
            
        w13_projs = []
        w2_projs = [] 
        
        group_shape = self.quant_method.quant_config.weight_block_size

             
        E, N, K = w13_weight.shape 
        assert w2_weight.shape == (E, K, N // 2), f"Down weight shape {w2_weight.shape} must be (E, K, N // 2)"
        
        E1, N1, K1 = w13_weight_scale_inv.shape 
        
        assert w2_weight_scale_inv.shape == (E1, K1, N1 // 2), f"Down weight scale shape {w2_weight_scale_inv.shape} must be (E1, K1, N1 // 2)"
        
        if is_lk_moe_quant_on_gpu():
            dequant_device = torch.cuda.current_device()
        else:
            dequant_device = torch.device("cpu")
        w13_buf = torch.zeros(E, N, K, dtype=torch.float32, device=dequant_device, requires_grad=False) 
        w2_buf = torch.zeros(E, K, N // 2, dtype=torch.float32, device=dequant_device, requires_grad=False)
        
        for expert_idx in range(E): 
            expert_w13_weight = w13_weight[expert_idx].to(dequant_device).to(device=dequant_device)  # torch.Size([1024, 2048])
            expert_w13_scale_inv = w13_weight_scale_inv[expert_idx].to(device=dequant_device)  # torch.Size([8, 16]) 
            expert_w2_weight = w2_weight[expert_idx].to(device=dequant_device)   # torch.Size([2048, 512])
            expert_w2_scale_inv = w2_weight_scale_inv[expert_idx].to(device=dequant_device) #  torch.Size([16, 4])  
                
                
            w13_float = expert_w13_weight.to(dtype=torch.float32)
            w2_float = expert_w2_weight.to(dtype=torch.float32) 
             
            w13_scale_inv_expanded = self._block_scale_broadcast_fixed(
                expert_w13_scale_inv, w13_float.shape, group_shape)
            w13_buf = w13_float * w13_scale_inv_expanded
              
             
            w2_scale_inv_expanded = self._block_scale_broadcast_fixed(
                expert_w2_scale_inv, w2_float.shape, group_shape)
            w2_buf = w2_float * w2_scale_inv_expanded
            
            w13_projs.append(w13_buf.to(dtype=self.moe_runner_config.params_dtype)) 
            w2_projs.append(w2_buf.to(dtype=self.moe_runner_config.params_dtype))
            
            del expert_w13_weight, expert_w13_scale_inv, expert_w2_weight, expert_w2_scale_inv
            del w13_float, w2_float, w13_scale_inv_expanded, w2_scale_inv_expanded
            del w13_buf, w2_buf  
                 
        w13_tensor = torch.stack(w13_projs, dim=0).cpu()
        w2_tensor = torch.stack(w2_projs, dim=0).cpu() 
        
        w13_projs.clear()
        w2_projs.clear()
        
        del w13_projs, w2_projs  
        
        w13_ptr = w13_tensor.contiguous().data_ptr()
        w2_ptr = w2_tensor.contiguous().data_ptr()
     
        num_processes, process_id, gpu_id = self._get_processes_info()
        
        self.lk_moe_config = lk_moe.MOEConfig(
            num_processes,                # num_processes
            process_id,                   # process_id
            gpu_id,                       # gpu_id
            self.has_gate_proj,             # has_gate_proj
            self.moe_runner_config.num_local_experts,        # expert_num
            self.top_k,                    # routed_expert_num
            self.hidden_size,              # hidden_size
            self.intermediate_size_per_partition,             # intermediate_size
            32,                            # stride
            10,                            # group_min_len
            self.max_num_group_batch_size,                          # group_max_len 
            hidden_ggml_type,                # hidden_type 
            w13_ggml_type,                  # w13_type  
            w2_ggml_type,                # w2_type   
            w13_ptr,                 # w13_proj
            w2_ptr,                   # w2_proj 
        )
        self.lk_moe = lk_moe.MOE(self.lk_moe_config)
         
        del w13_ptr, w2_ptr
        del w13_weight, w2_weight, w13_weight_scale_inv, w2_weight_scale_inv
             
        import gc
        gc.collect()
         
    def _process_channel_weights_quant(self, moe_compute_strategy: MoeComputeStrategy):  
    
        if moe_compute_strategy not in {MoeComputeStrategy.INT4}:
            print(f"Warning: moe_compute_strategy {moe_compute_strategy} is not supported for lk moe , use INT4 instead ...")
            moe_compute_strategy = MoeComputeStrategy.INT4
        
        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
        w13_weight_scale = self.w13_weight_scale
        w2_weight_scale = self.w2_weight_scale
        
        E, N, K = w13_weight.shape 
         
        assert w2_weight.shape == (E, K, N // 2)
         
        assert w13_weight_scale.shape == (E, N, 1)
        assert w2_weight_scale.shape == (E, K, 1)
        
        if is_lk_moe_quant_on_gpu():
            dequant_device = torch.cuda.current_device()
        else:
            dequant_device = torch.device("cpu")
        w13_fp32_list = []
        w2_fp32_list = [] 
        
        for expert_idx in range(E): 
            expert_w13_weight = w13_weight[expert_idx].to(device=dequant_device)  # [intermediate_size, hidden_size]
            expert_w13_scale = w13_weight_scale[expert_idx].to(device=dequant_device)  # [total_intermediate_size, 1]
            expert_w2_weight = w2_weight[expert_idx].to(device=dequant_device)  # [hidden_size, intermediate_size]
            expert_w2_scale = w2_weight_scale[expert_idx].to(device=dequant_device)  # [hidden_size, 1]
             
            w13_float = expert_w13_weight.to(dtype=torch.float32)
            w2_float = expert_w2_weight.to(dtype=torch.float32)
             
            w13_scale_expanded = expert_w13_scale.expand_as(w13_float)  # [intermediate_size, hidden_size]
            w2_scale_expanded = expert_w2_scale.expand_as(w2_float)  # [hidden_size, intermediate_size]
             
            w13_fp32 = w13_float * w13_scale_expanded
            w2_fp32 = w2_float * w2_scale_expanded
             
            w13_fp32_list.append(w13_fp32)
            w2_fp32_list.append(w2_fp32)
             
            del expert_w13_weight, expert_w13_scale, expert_w2_weight, expert_w2_scale
            del w13_float, w2_float, w13_scale_expanded, w2_scale_expanded
            del w13_fp32, w2_fp32
         
        w13_fp32_tensor = torch.stack(w13_fp32_list, dim=0).cpu()
        w2_fp32_tensor = torch.stack(w2_fp32_list, dim=0).cpu()
         
        w13_fp32_list.clear()
        w2_fp32_list.clear()
    
        del w13_fp32_list, w2_fp32_list
         
        w13_weight_ptr = w13_fp32_tensor.contiguous().data_ptr()
        w2_weight_ptr = w2_fp32_tensor.contiguous().data_ptr()
         
        hidden_ggml_type = self.get_ggml_type_from_dtype(self.moe_runner_config.params_dtype)
         
        group_size = getattr(self.quant_method, 'group_size', 32)
         
        num_bits = 4 if moe_compute_strategy == MoeComputeStrategy.INT4 else 8
         
        num_processes, process_id, gpu_id = self._get_processes_info()
         
        self.lk_moe_config = lk_moe.MOE_QuantConfig(
            num_processes,                     # num_processes
            process_id,                       # process_id
            gpu_id,                           # gpu_id
            self.has_gate_proj,             # has_gate_proj
            self.moe_runner_config.num_local_experts,            # expert_num
            self.top_k,                        # routed_expert_num
            self.hidden_size,                  # hidden_size
            self.intermediate_size_per_partition,  # intermediate_size
            32,                                # stride
            10,                                # group_min_len
            self.max_num_group_batch_size,   # group_max_len
            hidden_ggml_type,                  # hidden_type 
            0,                                 # w13_weight_data_type: 0 for fp32
            0,                                # w2_weight_data_type: 0 for fp32
            w13_weight_ptr,                    # w13_weight_ptr 
            w2_weight_ptr,                     # w2_weight_ptr   
            group_size,                        # group_size 
            num_bits,                          # num_bits 
        )
         
        self.lk_moe = lk_moe.MOE_Quant(self.lk_moe_config)
        
        del w13_weight_ptr, w2_weight_ptr
        del w13_fp32_tensor, w2_fp32_tensor
 
        import gc
        gc.collect()
            
    def _process_channel_weights(self):  
         
        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
        w13_weight_scale = self.w13_weight_scale
        w2_weight_scale = self.w2_weight_scale
        
        hidden_ggml_type = self.get_ggml_type_from_dtype(self.moe_runner_config.params_dtype)
        w13_ggml_type = hidden_ggml_type
        w2_ggml_type = hidden_ggml_type
            
        w13_projs = []
        w2_projs = [] 
             
        E, N, K = w13_weight.shape 
        
        if is_lk_moe_quant_on_gpu():
            dequant_device = torch.cuda.current_device()
        else:
            dequant_device = torch.device("cpu")
        
        w13_buf = torch.zeros(N, K, dtype=torch.float32, device=dequant_device, requires_grad=False) 
        w2_buf = torch.zeros(K, N // 2, dtype=torch.float32, device=dequant_device, requires_grad=False)
        
        for expert_idx in range(E): 
            expert_w13_weight = w13_weight[expert_idx].to(device=dequant_device)  # shape: [1408, 4096]
            expert_w13_scale = w13_weight_scale[expert_idx].to(device=dequant_device)    # shape: [1408, 1]
            expert_w2_weight = w2_weight[expert_idx].to(device=dequant_device)    # shape: [4096, 1408]
            expert_w2_scale = w2_weight_scale[expert_idx].to(device=dequant_device)  # shape: [4096, 1]
            
            w13_float = expert_w13_weight.to(dtype=torch.float32)
            w2_float = expert_w2_weight.to(dtype=torch.float32) 
             
                
            w13_scale_expanded = expert_w13_scale.expand_as(w13_float)
            w2_scale_expanded = expert_w2_scale.expand_as(w2_float)
                
            w13_buf = w13_float * w13_scale_expanded 
            w2_buf = w2_float * w2_scale_expanded 
            
            w13_projs.append(w13_buf.to(dtype=self.moe_runner_config.params_dtype)) 
            w2_projs.append(w2_buf.to(dtype=self.moe_runner_config.params_dtype))
            
            del expert_w13_weight, expert_w13_scale, expert_w2_weight, expert_w2_scale 
            del w13_float, w2_float, w13_scale_expanded, w2_scale_expanded
            del w13_buf, w2_buf  
            
        w13_tensor = torch.stack(w13_projs, dim=0).cpu()
        w2_tensor = torch.stack(w2_projs, dim=0).cpu() 
        del w13_projs, w2_projs  
        
        w13_ptr = w13_tensor.contiguous().data_ptr()
        w2_ptr = w2_tensor.contiguous().data_ptr()       
          
        num_processes, process_id, gpu_id = self._get_processes_info()
        
        self.lk_moe_config = lk_moe.MOEConfig(
            num_processes,                # num_processes
            process_id,                   # process_id
            gpu_id,                       # gpu_id
            self.has_gate_proj,             # has_gate_proj
            self.moe_runner_config.num_local_experts,        # expert_num
            self.top_k,                    # routed_expert_num
            self.hidden_size,              # hidden_size
            self.intermediate_size_per_partition,             # intermediate_size
            32,                            # stride
            10,                            # group_min_len
            self.max_num_group_batch_size,                          # group_max_len 
            hidden_ggml_type,                # hidden_type  
            w13_ggml_type,                  # w13_type  
            w2_ggml_type,                # w2_type   
            w13_ptr,                 # w13_proj
            w2_ptr,                   # w2_proj 
        ) 
        self.lk_moe = lk_moe.MOE(self.lk_moe_config)
        
        del w13_tensor, w2_tensor
        del w13_ptr, w2_ptr 
        
        
        import gc
        gc.collect()
            
    def _process_regular_weights(self):  
               
        w13_ggml_type = self.get_ggml_type_from_dtype(self.w13_weight.dtype)
        w2_ggml_type = self.get_ggml_type_from_dtype(self.w2_weight.dtype ) 
        hidden_ggml_type = self.get_ggml_type_from_dtype(self.moe_runner_config.params_dtype)
      
         
        w13_ptr = self.w13_weight.contiguous().data_ptr()
        w2_ptr = self.w2_weight.contiguous().data_ptr()
        
        num_processes, process_id, gpu_id = self._get_processes_info()
        
        self.lk_moe_config = lk_moe.MOEConfig(
            num_processes,                # num_processes
            process_id,                   # process_id
            gpu_id,                       # gpu_id
            self.has_gate_proj,             # has_gate_proj
            self.moe_runner_config.num_local_experts,        # expert_num
            self.top_k,                    # routed_expert_num
            self.hidden_size,              # hidden_size
            self.intermediate_size_per_partition,             # intermediate_size
            32,                            # stride
            10,                            # group_min_len
            self.max_num_group_batch_size,                          # group_max_len
            hidden_ggml_type,            
            w13_ggml_type,                # gate_type  
            w2_ggml_type,                # down_type  
            w13_ptr,                 # w13_ptr 
            w2_ptr,                 # w2_ptr 
        ) 
        self.lk_moe = lk_moe.MOE(self.lk_moe_config)  
        del w13_ptr, w2_ptr
        import gc
        gc.collect()
         
    def forward_lk(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        return self._process_valid_inputs(hidden_states, topk_weights, topk_ids)
    
    def _get_max_batch_size(self) -> int:
        if self.speculative_num_draft_tokens is not None and self.speculative_num_draft_tokens > 0:
            batch_size = self.max_running_requests * (
                1 + self.speculative_num_draft_tokens
            ) * 2
        else:
            batch_size = self.max_running_requests * 2
            
        batch_size = min(batch_size, 512)
        
        return batch_size
  
    def _initialize_cuda_graph_buffers(self): 
        if not hasattr(FusedMoE, 'cuda_graphs'):  
                
            batch_size = self._get_max_batch_size()
 
            FusedMoE.cuda_graphs = [1, 2, 4] + list(range(8, batch_size + 1, 8)) 
             
            FusedMoE.input_tensor_cpu = {}  # device_id -> buffers
            FusedMoE.expert_ids_cpu = {}    # device_id -> buffers
            FusedMoE.weights_cpu = {}       # device_id -> buffers
            FusedMoE.output_cpu = {}        # device_id -> buffers
            FusedMoE.bsz_tensor_cpu = {}    # device_id -> buffers
            FusedMoE.output_gpu = {}        # device_id -> buffers
            
            current_device = torch.cuda.current_device()
    
            num_experts_per_tok = self.top_k
            hidden_size = self.hidden_size
            buff_dtype = self.moe_runner_config.params_dtype
             
            pin_memory = is_pin_memory_available()
            
            FusedMoE.output_gpu[current_device] = [
                torch.zeros((batch_size, hidden_size), device=current_device, dtype=buff_dtype, requires_grad=False).contiguous()
                for batch_size in FusedMoE.cuda_graphs
            ]
            
            FusedMoE.input_tensor_cpu[current_device] = [
                torch.zeros((batch_size, self.hidden_size), device="cpu", dtype=buff_dtype, pin_memory=pin_memory, requires_grad=False).contiguous()
                for batch_size in FusedMoE.cuda_graphs
            ]
            FusedMoE.expert_ids_cpu[current_device] = [
                torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.int32, pin_memory=pin_memory, requires_grad=False).contiguous()
                for batch_size in FusedMoE.cuda_graphs
            ]
            FusedMoE.weights_cpu[current_device] = [
                torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=pin_memory, requires_grad=False).contiguous()
                for batch_size in FusedMoE.cuda_graphs
            ]
            FusedMoE.output_cpu[current_device] = [
                torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=buff_dtype, requires_grad=False).contiguous()
                for batch_size in FusedMoE.cuda_graphs
            ]
            FusedMoE.bsz_tensor_cpu[current_device] = [
                torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=pin_memory, requires_grad=False).contiguous()
                for _ in range(len(FusedMoE.cuda_graphs))
            ]
         
    def _find_best_graph_index(self, total_tokens: int) -> int:
        if not hasattr(FusedMoE, 'cuda_graphs') or not FusedMoE.cuda_graphs:
            raise ValueError("No CUDA graphs initialized.")
        
        cuda_graphs = FusedMoE.cuda_graphs
        
        low, high = 0, len(cuda_graphs) - 1
        best_index = len(cuda_graphs) - 1  
        
        while low <= high:
            mid = (low + high) // 2
            if cuda_graphs[mid] >= total_tokens:
                best_index = mid
                high = mid - 1
            else:
                low = mid + 1
         
        if best_index >= len(cuda_graphs):
            best_index = len(cuda_graphs) - 1
             
        if cuda_graphs[best_index] < total_tokens:
            raise ValueError(f"No suitable CUDA graph found for {total_tokens} tokens. "
                            f"Maximum available buffer size: {cuda_graphs[-1]}")
        
        return best_index
    
    def _process_valid_inputs(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        """Process inputs that are guaranteed to be valid (non-NaN)"""
          
            
        def get_cuda_stream_ptr(stream: torch.cuda.Stream) -> int:
                """
                Get the underlying CUDA stream pointer from a torch.cuda.Stream object.
                """
                if hasattr(stream, 'cuda_stream'):
                    return stream.cuda_stream
                elif hasattr(stream, 'stream'):
                    return stream.stream
                else:
                    # Fallback to using the default stream
                    return 0
                
        batch_size = hidden_states.size(0)
        current_device = hidden_states.device.index
        current_stream = torch.cuda.current_stream(hidden_states.device)
        stream_ptr = get_cuda_stream_ptr(current_stream) 
        non_blocking = True
       
        try:   
            if torch.cuda.is_current_stream_capturing() or get_is_capture_mode():
                graph_index = self._find_best_graph_index(batch_size)
                 
                input_tensor_cpu = FusedMoE.input_tensor_cpu[current_device]
                expert_ids_cpu = FusedMoE.expert_ids_cpu[current_device]
                weights_cpu = FusedMoE.weights_cpu[current_device]
                output_cpu = FusedMoE.output_cpu[current_device]
                bsz_tensor_cpu = FusedMoE.bsz_tensor_cpu[current_device]
                output_gpu = FusedMoE.output_gpu[current_device]
                    
                bsz_tensor_cpu[graph_index][0] = batch_size 

                input_tensor_cpu[graph_index][:batch_size].copy_(hidden_states, non_blocking=non_blocking)
                expert_ids_cpu[graph_index][:batch_size].copy_(topk_ids, non_blocking=non_blocking)
                weights_cpu[graph_index][:batch_size].copy_(topk_weights, non_blocking=non_blocking) 
                input_ptr = input_tensor_cpu[graph_index].data_ptr()
                expert_ids_ptr = expert_ids_cpu[graph_index].data_ptr()
                weights_ptr = weights_cpu[graph_index].data_ptr()
                output_ptr = output_cpu[graph_index].data_ptr()   
                self.lk_moe.submit_with_cuda_stream(
                    stream_ptr, 
                    batch_size,                                   # qlen
                    expert_ids_cpu[graph_index].size(1),                     # k
                    expert_ids_ptr,                  # expert_ids
                    weights_ptr,                     # weights
                    input_ptr,                       # input
                    output_ptr,                      # output 
                    bsz_tensor_cpu[graph_index].data_ptr()                   # bsz_tensor
                )  
                self.lk_moe.sync_with_cuda_stream(stream_ptr)  
                
                output_gpu[graph_index][:batch_size].copy_(output_cpu[graph_index][:batch_size], non_blocking=non_blocking) 
                if self.check_nan_in_output: 
                    torch.nan_to_num(output_gpu[graph_index][:batch_size], nan=0.0, out=output_gpu[graph_index][:batch_size])
                return output_gpu[graph_index][:batch_size]
            else:  
                with self._lk_moe_guard.acquire():
                    prefill_stream = torch.cuda.Stream()
                            
                    current_stream = torch.cuda.current_stream()
                    wait_event = torch.cuda.Event()
                    wait_event.record(current_stream)
                    
                    topk_ids.record_stream(prefill_stream)
                    topk_weights.record_stream(prefill_stream)
                    hidden_states.record_stream(prefill_stream)
                    
                    with torch.cuda.stream(prefill_stream):
                        prefill_stream.wait_event(wait_event)
                        
                        expert_ids_cpu = topk_ids.to(dtype=torch.int32, device='cpu', memory_format=torch.contiguous_format, non_blocking=non_blocking)
                        weights_cpu = topk_weights.to(dtype=torch.float32, device='cpu', memory_format=torch.contiguous_format, non_blocking=non_blocking)
                        hidden_states_cpu = hidden_states.to(device='cpu', memory_format=torch.contiguous_format, non_blocking=non_blocking)
                        output_cpu = torch.zeros_like(hidden_states, device='cpu')
                        bsz_tensor = torch.tensor([hidden_states.size(0)], device='cpu', dtype=torch.int32).contiguous()
                        
                        prefill_stream.synchronize()
                        self.lk_moe.forward(
                            hidden_states.size(0),                         # qlen
                            expert_ids_cpu.size(1),                    # k
                            expert_ids_cpu.data_ptr(),                 # expert_ids
                            weights_cpu.data_ptr(),                    # weights
                            hidden_states_cpu.data_ptr(),              # input
                            output_cpu.data_ptr(),                     # output 
                            bsz_tensor.data_ptr()                      # bsz_tensor
                        )     
                        output_gpu = output_cpu.to(torch.cuda.current_device(), non_blocking=non_blocking)
                        if self.check_nan_in_output:
                            torch.nan_to_num(output_gpu, nan=0.0, out=output_gpu)
                        complete_event = torch.cuda.Event()
                        complete_event.record(prefill_stream)
                    
                    current_stream.wait_event(complete_event)
        
                    return output_gpu
       
        except Exception as e:
            logger.error(f"lk_moe forward failed with error: {e}, falling back to default path") 
            raise RuntimeError("lk_moe forward failed, fallback to default MoE implementation")

@register_custom_op(out_shape="hidden_states")
def moe_forward_piecewise_cuda_graph_impl(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    # only standard topk output is supported for piecewise cuda graph
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
    )
    forward_context = get_forward_context()
    moe_layer = forward_context.moe_layers[layer_id]
    return moe_layer.forward_impl(hidden_states, topk_output)


@register_custom_op(out_shape="hidden_states")
def fused_moe_bypassed_piecewise_cuda_graph_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    topk_group: Optional[int],
    num_expert_group: Optional[int],
    correction_bias: Optional[torch.Tensor],
    renormalize: bool,
    layer_id: int,
) -> torch.Tensor:
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(
            top_k=top_k,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            correction_bias=correction_bias,
            renormalize=renormalize,
        ),
    )
    forward_context = get_forward_context()
    moe_layer = forward_context.moe_layers[layer_id]
    return moe_layer.forward_impl(hidden_states, topk_output)


from sglang.srt.model_executor.model_runner import get_moe_context

class MoEForwardContext:
    def __init__(self, moe_layers, forward_batch):
        self.moe_layers = moe_layers
        self.forward_batch = forward_batch
    
    @property
    def batch_descriptor(self):
        return self.forward_batch
    
    def get_moe_layer(self, layer_id):
        if self.moe_layers and 0 <= layer_id < len(self.moe_layers):
            return self.moe_layers[layer_id]
        return None
 
_current_forward_context = None

from typing import Dict, Optional, List


def create_cpu_weights(layer) -> Dict[str, torch.Tensor]: 
    pin_memory = is_pin_memory_available()
    cpu_weights = {}
    is_fp8 = (isinstance(layer.quant_method, Fp8MoEMethod) or 
                (hasattr(layer, "scheme") and isinstance(layer.scheme, CompressedTensorsW8A8Fp8MoE)))
    is_wna16 = (isinstance(layer.quant_method, CompressedTensorsFusedMoEMethod) and 
                not (hasattr(layer, "scheme") and isinstance(layer.scheme, CompressedTensorsW8A8Fp8MoE)))
    is_regular = isinstance(layer.quant_method, UnquantizedFusedMoEMethod)

    
    if is_fp8 or is_wna16: 
        param_names = ["w13_weight", "w2_weight"]
        for param_name in param_names:
            if param_name == "w13_weight":
                E = layer.moe_runner_config.num_local_experts
                N = layer.intermediate_size_per_partition * 2 if layer.has_gate_proj else layer.intermediate_size_per_partition
                K = layer.hidden_size
                shape = (E, N, K * 18 // 32)
            elif param_name == "w2_weight":
                E = layer.moe_runner_config.num_local_experts
                N = layer.hidden_size
                K = layer.intermediate_size_per_partition
                shape = (E, N, K * 18 // 32)
            
            weight_cpu = torch.zeros(
                shape,
                dtype=torch.uint8,
                device="cpu",
                requires_grad=False,
                pin_memory=pin_memory
            ).contiguous()
            
        
            cpu_weights[param_name] = weight_cpu
            logger.debug(f"Created {param_name} with shape {shape} for FP8/WNA16 layer")
            
    elif is_regular: 
        w13_shape = (layer.moe_runner_config.num_local_experts, 
                    layer.intermediate_size_per_partition * 2 if layer.has_gate_proj else layer.intermediate_size_per_partition,    
                    layer.hidden_size)
        w13_weight_cpu = torch.zeros(
            w13_shape,
            dtype=layer.moe_runner_config.params_dtype,
            device="cpu",
            requires_grad=False,
            pin_memory=pin_memory,
        ).contiguous()
        
        
        cpu_weights['w13_weight'] = w13_weight_cpu
        logger.debug(f"Created w13_weight with shape {w13_shape} for regular layer")
            
        w2_shape = (layer.moe_runner_config.num_local_experts,
                    layer.hidden_size,
                    layer.intermediate_size_per_partition)
        w2_weight_cpu = torch.zeros(
            w2_shape,
            dtype=layer.moe_runner_config.params_dtype,
            device="cpu",
            requires_grad=False,
            pin_memory=pin_memory,
        ).contiguous() 
        
        cpu_weights['w2_weight'] = w2_weight_cpu
        logger.debug(f"Created w2_weight with shape {w2_shape} for regular layer")
    
    return cpu_weights
 
def moe_prepare_gpu_prefill(layer, forward_context: MoEForwardContext, device: torch.device):
    
    
    if layer.is_gpu_prefill_layer: 
        batch_key = id(forward_context.batch_descriptor)
        logger.debug(f"batch_key={batch_key}, forward_context={id(forward_context)}")
        batch_id = getattr(forward_context, '_prefetch_batch_id', None)
        is_temporary = False
         
        if batch_id is not None:
            stored_batch_key = getattr(forward_context, '_prefetch_batch_key', None)
            if stored_batch_key != batch_key:
                logger.warning(f"Batch key mismatch! stored={stored_batch_key}, current={batch_key}, "
                       f"resetting batch_id from {batch_id} to None")
                batch_id = None
        
        if batch_id is None:
            with FusedMoE._batch_lock: 
                batch_id = getattr(forward_context, '_prefetch_batch_id', None)
                if batch_id is not None:
                    stored_batch_key = getattr(forward_context, '_prefetch_batch_key', None)
                    if stored_batch_key != batch_key:
                        batch_id = None
                
                if batch_id is None:
                    for bid, in_use in FusedMoE._batch_usage.items():
                        if not in_use:
                            batch_id = bid
                            FusedMoE._batch_usage[bid] = True
                            forward_context._prefetch_batch_id = batch_id
                            forward_context._prefetch_batch_key = batch_key  
                            break
                    
                    if batch_id is None:
                        batch_id = -1  
                        is_temporary = True
                        forward_context._prefetch_batch_id = batch_id
                        forward_context._prefetch_batch_key = batch_key 
         
        with torch.no_grad():

            batch_key = id(forward_context.batch_descriptor)

            prefetch_stream = forward_context._prefetch_streams[batch_key]
            prefetch_events = forward_context._prefetch_events
        
            with torch.cuda.stream(prefetch_stream):
                
                param_names = [
                    "w13_weight",
                    "w2_weight", 
                ] 
                
                if is_temporary:
                    cpu_weights = create_cpu_weights(layer)
                else:
                    cpu_weights = {}
                 
                for param_name in param_names:
                    if is_temporary:
                        weight_cpu = cpu_weights[param_name].contiguous()
                        weight_gpu = torch.zeros_like(weight_cpu, device=device, memory_format=torch.contiguous_format)
                    else:
                        weight_cpu = FusedMoE._cpu_weights_placeholder[batch_id][param_name].contiguous()
                        weight_gpu = FusedMoE._gpu_weights_placeholder[batch_id][param_name].contiguous()
                     
                    
                    is_fp8 = (isinstance(layer.quant_method, Fp8MoEMethod) or 
                                (hasattr(layer, "scheme") and isinstance(layer.scheme, CompressedTensorsW8A8Fp8MoE)))
                    is_wna16 = (isinstance(layer.quant_method, CompressedTensorsFusedMoEMethod) and 
                                not (hasattr(layer, "scheme") and isinstance(layer.scheme, CompressedTensorsW8A8Fp8MoE)))
                    is_regular = isinstance(layer.quant_method, UnquantizedFusedMoEMethod)
                    if is_fp8 or is_wna16:
                        layer.lk_moe.collectWeight(
                            param_name,
                            weight_cpu.data_ptr()
                        ) 
                    elif is_regular:
                        if param_name == "w13_weight":
                            if layer.has_gate_proj:
                                layer.lk_moe.collect_weights(
                                    True,  
                                    0,
                                    0,
                                    weight_cpu.data_ptr(),  
                                    0  # 0   gate  
                                )
                                
                                layer.lk_moe.collect_weights(
                                    True,  
                                    0,
                                    0,
                                    weight_cpu.data_ptr(),  
                                    1  # 1   up
                                )
                            else:
                                layer.lk_moe.collect_weights(
                                    True,  
                                    0,
                                    0,
                                    weight_cpu.data_ptr(),  
                                    1  # 1   up
                                )
                        elif param_name == "w2_weight":
                            layer.lk_moe.collect_weights(
                                True,  
                                0,
                                0,
                                weight_cpu.data_ptr(),  
                                2  # w2
                            )
                        else:
                            raise ValueError(f"Unsupported param_name {param_name} for layer")
                    weight_gpu.copy_(weight_cpu, non_blocking=True)
                    weight_gpu.record_stream(prefetch_stream) 
                    setattr(layer, param_name, torch.nn.Parameter(weight_gpu, requires_grad=False))
                
                layer_id = id(layer)
                event = torch.cuda.Event()
                event.record(prefetch_stream)
                batch_key = id(forward_context.batch_descriptor)
                prefetch_events[(layer_id, batch_key)] = event

        
def moe_clean_gpu_prefill(layer, forward_context: MoEForwardContext):   
    with torch.no_grad():   
        param_names = ["w13_weight", "w2_weight"]
        
        for param_name in param_names:
            if hasattr(layer, param_name):
                setattr(layer, param_name, None)
       
        if hasattr(forward_context, '_prefetch_batch_id'):
            with FusedMoE._batch_lock:
                batch_id = forward_context._prefetch_batch_id
                if batch_id >= 0:
                    FusedMoE._batch_usage[batch_id] = False
            
            delattr(forward_context, '_prefetch_batch_id')
            if hasattr(forward_context, '_prefetch_batch_key'):
                delattr(forward_context, '_prefetch_batch_key')
  

def moe_cleanup(layer, layer_idx: int, hidden_states: torch.Tensor, forward_context: MoEForwardContext): 
    
    if not layer.should_use_gpu_prefill(hidden_states):
        return
    
    batch_key = id(forward_context.batch_descriptor)
     
    if not hasattr(forward_context, '_batch_prefetch_states'):
        return
    if batch_key not in forward_context._batch_prefetch_states:
        return
    
    batch_state = forward_context._batch_prefetch_states[batch_key]
    state = batch_state['state']
     
    keys_to_clean = [k for k in state.keys() if k <= layer_idx]
    
    for k in keys_to_clean:  
        if is_lk_moe_gpu_resident_layer(k):
            del state[k]
            continue  
        layer_obj = forward_context.moe_layers[k]
        if layer_obj:
            if hasattr(forward_context, '_prefetch_events'):
                layer_id = id(layer_obj)
                batch_key = id(forward_context.batch_descriptor)
                key = (layer_id, batch_key)
                if key in forward_context._prefetch_events:
                    forward_context._prefetch_events[key].wait()
                    del forward_context._prefetch_events[key]
            moe_clean_gpu_prefill(layer_obj, forward_context)
        del state[k]

def moe_prefetch(layer, layer_idx: int, hidden_states: torch.Tensor, 
                 forward_context: MoEForwardContext, gpu_prefetch_window: int): 
    
    if not layer.should_use_gpu_prefill(hidden_states):
        return
    
    if not hasattr(forward_context, '_prefetch_streams'):
        forward_context._prefetch_streams = {}
        
    if not hasattr(forward_context, '_prefetch_events'):
        forward_context._prefetch_events = {}  
        
    if not hasattr(forward_context, '_batch_prefetch_states'):
        forward_context._batch_prefetch_states = {}
     
    batch_key = id(forward_context.batch_descriptor) 
    
    if batch_key not in forward_context._prefetch_streams:
        forward_context._prefetch_streams[batch_key] = torch.cuda.Stream()
     
    if batch_key not in forward_context._batch_prefetch_states:
        forward_context._batch_prefetch_states[batch_key] = {
            'state': {},  # layer_idx -> prefetch_count
            'called_layers': set()
        }
    
    batch_state = forward_context._batch_prefetch_states[batch_key]
    state = batch_state['state']
    called_layers = batch_state['called_layers']
     
    if layer_idx == 0:
        state.clear()
        called_layers.clear()
      
    
    called_layers.add(layer_idx)  
            
    active_prefetches = 0
    for k in state.keys(): 
        if not is_lk_moe_gpu_resident_layer(k):
            active_prefetches += 1
     
    available_slots = gpu_prefetch_window - active_prefetches
    
    layer_count = len(forward_context.moe_layers)
    last_layer_id = forward_context.moe_layers[layer_count-1].layer_id
    if available_slots > 0: 
        prefetch_candidates = [] 
        for offset in range(0, layer_count): 
            candidate_idx = layer_idx + offset 
            if candidate_idx > last_layer_id:
                break 
             
            if is_lk_moe_gpu_resident_layer(candidate_idx):
                continue
             
            if candidate_idx not in state and len(prefetch_candidates) < available_slots:
                candidate_layer = forward_context.moe_layers[candidate_idx]
                if candidate_layer:
                    prefetch_candidates.append((candidate_idx, candidate_layer))
         
        for idx, layer_obj in prefetch_candidates:
            moe_prepare_gpu_prefill(layer_obj, forward_context, torch.cuda.current_device())
            state[idx] = 1
               
    
                    
def moe_wait_prefetch(layer, hidden_states: torch.Tensor, forward_context: MoEForwardContext):
 
    if not hasattr(forward_context, '_prefetch_events'):
        return
    
    if not layer.should_use_gpu_prefill(hidden_states):
        return
    
    layer_id = id(layer)
    batch_key = id(forward_context.batch_descriptor)
    prefetch_events = forward_context._prefetch_events
    key = (layer_id, batch_key)
    
    if key in prefetch_events:
        prefetch_events[key].wait()
        del prefetch_events[key] 
    current_stream = torch.cuda.current_stream() 
    if hasattr(layer, 'w13_weight') and layer.w13_weight is not None:
        layer.w13_weight.record_stream(current_stream)
    if hasattr(layer, 'w2_weight') and layer.w2_weight is not None:
        layer.w2_weight.record_stream(current_stream)
        
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.current_stream().synchronize()
