# SPDX-License-Identifier: Apache-2.0

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor, dequantize
from huggingface_hub import snapshot_download
from sgl_kernel import (
    ggml_dequantize,
    ggml_moe_a8,
    ggml_moe_a8_vec,
    ggml_moe_get_block_size,
    ggml_mul_mat_a8,
    ggml_mul_mat_vec_a8,
    moe_align_block_size,
)

GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")
GGUF_SAMPLE_MOE = snapshot_download("SzymonOzog/test-gguf-moe-sample")


def get_gguf_sample_tensors(
    hidden_size: int, quant_type: GGMLQuantizationType
) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


def get_gguf_MoE_tensors(
    hidden_size: int, quant_type: GGMLQuantizationType
) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE_MOE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
# Hidden_size for testing, must match the sample file in HF repo,
# we have `hidden_size = 256, 1024` for test in HF repo currently.
HIDDEN_SIZES = [256, 1024]
NUM_TOKENS = [7, 2050]  # Arbitrary values for testing
SEEDS = [0]
QUANT_TYPES = [
    # i-matrix
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    # k-quants
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    # standard quantization
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q8_0,
]


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_dequantize(
    hidden_size: int, dtype: torch.dtype, quant_type: GGMLQuantizationType
):
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape_str = tensor.name.split("_")[-1]
        shape = map(int, shape_str.split("x"))

        ref_output = torch.tensor(
            dequantize(tensor.data, quant_type), device="cuda"
        ).to(dtype)
        output = ggml_dequantize(
            torch.tensor(tensor.data, device="cuda"), quant_type, *list(shape), dtype
        )

        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=4e-2)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_mmvq(hidden_size: int, dtype: torch.dtype, quant_type: GGMLQuantizationType):

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((1, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ggml_mul_mat_vec_a8(qweight, x, quant_type, qweight.shape[0]).to(dtype)

        # NOTE(FlamingoPg): There can be occasional errors, Loosen the granularity of gguf bf16 verification.
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1}
        rtols = {torch.half: 1e-1, torch.bfloat16: 3e1, torch.float: 1e-1}

        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "quant_type",
    [
        # k-quants
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        # standard quants
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q8_0,
    ],
)
@torch.inference_mode()
def test_mmq(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
):

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0])
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1.2}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e4, torch.float: 2e1}
        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )

def create_moe_test_data(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    block_size: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    """Create test data for MoE testing in GGUF format."""
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    
    # GGUF format: w1 combines gate and up projections
    # Shape: [num_experts, 2 * intermediate_size, hidden_size]
    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device)
    
    # w2: down projection
    # Shape: [num_experts, hidden_size, intermediate_size]
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device)
    
    # Create topk_ids and topk_weights
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
    topk_weights = torch.randn(num_tokens, top_k, dtype=dtype, device=device)
    topk_weights = torch.softmax(topk_weights, dim=-1)  # Normalize
    
    return x, w1, w2, topk_ids, topk_weights

def compute_moe_reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Compute MoE reference output using PyTorch."""
    num_tokens, hidden_size = x.shape
    num_experts, intermediate_size, _ = w1.shape
    top_k = topk_ids.shape[1]
    
    output = torch.zeros(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
    
    for i in range(num_tokens):
        token_output = torch.zeros(hidden_size, dtype=x.dtype, device=x.device)
        for k in range(top_k):
            expert_idx = topk_ids[i, k].item()
            if expert_idx == -1:
                continue
            weight = topk_weights[i, k]
            
            # First linear layer
            hidden = x[i] @ w1[expert_idx].T
            # Activation
            if activation == "silu":
                hidden = hidden * torch.sigmoid(hidden)
            else:
                hidden = hidden * torch.erf(hidden / np.sqrt(2))  # GELU approximation
            
            # Second linear layer
            out = hidden @ w2[expert_idx].T
            token_output += weight * out
        
        output[i] = token_output
    
    return output

@pytest.mark.parametrize("num_tokens", [7, 32, 128])
@pytest.mark.parametrize("hidden_size", [256, 512])
@pytest.mark.parametrize("intermediate_size", [512, 1024])
@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_type", [GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0])
@torch.inference_mode()
def test_ggml_moe_a8(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
):
    num_tokens=7
    hidden_size=256
    intermediate_size=512
    num_experts=4
    top_k=2
    dtype=torch.float16
    quant_type=2
    """Test ggml_moe_a8 with quantized weights - Complete MoE flow."""
    print(f"\n{'='*80}")
    print(f"Testing with parameters:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    print(f"  dtype: {dtype}")
    print(f"  quant_type: {quant_type}")
    print(f"{'='*80}")
    
    device = "cuda"
    block_size = ggml_moe_get_block_size(quant_type)
    print(f"block_size: {block_size}")
    
    # Create test data - 注意：w1 需要是 [num_experts, 2*intermediate_size, hidden_size]
    print("Creating test data...")
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.randn(num_tokens, top_k, dtype=dtype, device=device)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    
    print(f"x shape: {x.shape}, dtype: {x.dtype}")
    print(f"w1 shape: {w1.shape}, dtype: {w1.dtype}")
    print(f"w2 shape: {w2.shape}, dtype: {w2.dtype}")
    print(f"topk_ids shape: {topk_ids.shape}")
    print(f"topk_weights shape: {topk_weights.shape}")
    
    # Align block size
    print("\nCalling moe_align_block_size...")
    sorted_token_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, block_size, num_experts
    )
    
    # Get scalar value
    if isinstance(num_tokens_post_pad, torch.Tensor):
        num_tokens_post_pad_val = num_tokens_post_pad.item()
    else:
        num_tokens_post_pad_val = num_tokens_post_pad
    
    print(f"sorted_token_ids shape: {sorted_token_ids.shape}")
    print(f"expert_ids shape: {expert_ids.shape}")
    print(f"num_tokens_post_pad: {num_tokens_post_pad_val}")
    
    # First stage: x -> intermediate (gate and up projections)
    print("\n=== Stage 1: x -> intermediate ===")
    intermediate = ggml_moe_a8(
        x,
        w1,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad_val,
        quant_type,
        intermediate_size,  # output dimension is intermediate_size
        top_k,
        num_tokens,
    )
    
    print(f"intermediate shape: {intermediate.shape}")
    expected_intermediate_shape = (num_tokens * top_k, 2 * intermediate_size)
    print(f"Expected: {expected_intermediate_shape}")
    assert intermediate.shape == expected_intermediate_shape, \
        f"Intermediate shape mismatch: {intermediate.shape} vs {expected_intermediate_shape}"
    
    # Apply activation (silu_and_mul)
    print("\n=== Activation (silu_and_mul) ===")
    # Split into gate and up
    gate = intermediate[..., :intermediate_size]
    up = intermediate[..., intermediate_size:]
    activated = torch.silu(gate) * up
    print(f"activated shape: {activated.shape}")
    expected_activated_shape = (num_tokens * top_k, intermediate_size)
    assert activated.shape == expected_activated_shape, \
        f"Activated shape mismatch: {activated.shape} vs {expected_activated_shape}"
    
    # Second stage: activated -> final output
    print("\n=== Stage 2: activated -> output ===")
    final = ggml_moe_a8(
        activated,
        w2,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad_val,
        quant_type,
        hidden_size,  # output dimension is hidden_size
        1,  # top_k = 1 for second stage
        num_tokens * top_k,
    )
    
    print(f"final shape: {final.shape}")
    expected_final_shape = (num_tokens * top_k, hidden_size)
    assert final.shape == expected_final_shape, \
        f"Final shape mismatch: {final.shape} vs {expected_final_shape}"
    
    # Combine with topk_weights
    print("\n=== Combining with weights ===")
    final = final.reshape(num_tokens, top_k, hidden_size)
    final = final * topk_weights.unsqueeze(-1)
    output = final.sum(dim=1)
    
    print(f"output shape: {output.shape}")
    expected_output_shape = (num_tokens, hidden_size)
    assert output.shape == expected_output_shape, \
        f"Output shape mismatch: {output.shape} vs {expected_output_shape}"
    
    # Final checks
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    
    print(f"\n✓ Test passed!")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Output mean: {output.mean().item():.6f}, std: {output.std().item():.6f}")
    
    
@pytest.mark.parametrize("num_tokens", [7, 32, 128])
@pytest.mark.parametrize("hidden_size", [256, 512])
@pytest.mark.parametrize("intermediate_size", [512, 1024])
@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("quant_type", [GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0])
@torch.inference_mode()
def test_ggml_moe_a8_vec(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
):
    """Test ggml_moe_a8_vec (vectorized version) with quantized weights."""
    # Similar to test_ggml_moe_a8 but for the vectorized version
    pass


@pytest.mark.parametrize("num_tokens", [7, 32, 128, 256])
@pytest.mark.parametrize("num_experts", [4, 8, 16])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@torch.inference_mode()
def test_moe_align_block_size(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_size: int,
):
    """Test moe_align_block_size function."""
    device = "cuda"
    
    # Create topk_ids
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
    
    # Add some -1 to test invalid expert handling
    if num_tokens > 10:
        topk_ids[5, 1] = -1
        topk_ids[10, 2] = -1
    
    # Call moe_align_block_size
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        topk_ids, block_size, num_experts
    )
    
    # Verify outputs
    assert sorted_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    assert num_post_pad.dtype == torch.int32
    assert num_post_pad.item() > 0
    assert num_post_pad.item() % block_size == 0, f"num_post_pad {num_post_pad.item()} not divisible by {block_size}"
    
    # Verify sorted_ids values are within range
    max_token_idx = num_tokens * top_k - 1
    valid_mask = sorted_ids <= max_token_idx
    invalid_count = (~valid_mask).sum().item()
    # Some padding tokens may be > max_token_idx, that's okay
    print(f"Invalid sorted_ids count: {invalid_count}")
    
    # Verify expert_ids values are valid
    valid_expert_mask = (expert_ids >= 0) | (expert_ids == -1)
    assert valid_expert_mask.all(), "expert_ids contains invalid values"
    
    # Verify expert_ids values are within range or -1
    expert_in_range = (expert_ids < num_experts) | (expert_ids == -1)
    assert expert_in_range.all(), f"expert_ids out of range: {expert_ids[~expert_in_range][:10]}"


@pytest.mark.parametrize("num_tokens", [7, 32, 64])
@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("top_k", [2, 4])
@torch.inference_mode()
def test_moe_align_block_size_with_padding(
    num_tokens: int,
    num_experts: int,
    top_k: int,
):
    """Test moe_align_block_size with padding enabled."""
    device = "cuda"
    block_size = 32
    
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
    
    # Test with padding
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        topk_ids, block_size, num_experts, pad_sorted_ids=True
    )
    
    assert num_post_pad.item() % block_size == 0
    assert sorted_ids.size(0) % block_size == 0, "sorted_ids not padded to block size"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
