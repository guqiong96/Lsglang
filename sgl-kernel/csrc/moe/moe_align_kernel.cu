/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <THC/THCAtomics.cuh>

#include "utils.h"

#define VEC_SIZE 4
using Vec = int4;



template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t* __restrict__ token_mask,
    int32_t model_offset, int32_t topk_num, bool has_expert_map) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }

    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      int32_t rank_post_pad = atomicAdd(
          &cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id], 1);
      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] =
          i;
    }
  }
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    int32_t* __restrict__ expert_map, size_t numel, int32_t num_experts,
    int32_t max_num_tokens_padded, int32_t topk_num, bool has_expert_map) {
  _count_and_sort_expert_tokens(
      topk_ids, sorted_token_ids, cumsum_buffer, expert_map, numel, num_experts,
      max_num_tokens_padded, nullptr, 0, topk_num, has_expert_map);
}

#ifdef __CUDA_ARCH__
__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(mask, v, offset);
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;
}
#endif



template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t max_num_m_blocks, int32_t model_offset, int32_t inactive_expert_id,
    int32_t topk_num, int32_t* token_mask, bool has_expert_map) {
  extern __shared__ int32_t shared_counts[];

  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  // Use separate threadblocks to fill sorted_token_ids.
  // This is safe since the current kernel does not use sorted_token_ids.
  if (blockIdx.x % 2) {
    // Initialize sorted_token_ids with -1
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += blockDim.x) {
       sorted_token_ids[sorted_token_ids_offset + it] = numel;;
    }
    return;
  }

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset],
              mask);
  }

  __syncthreads();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[cumsum_offset + threadIdx.x];
         i < cumsum[cumsum_offset + threadIdx.x + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = threadIdx.x;
    }
  }

  // Fill remaining expert_ids with -1
  const size_t fill_start_idx =
      cumsum[cumsum_offset + num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}



template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    int32_t topk_num, bool has_expert_map) {
  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, padded_num_experts, experts_per_warp, block_size, numel,
      cumsum, max_num_tokens_padded, CEILDIV(max_num_tokens_padded, block_size),
      0, -1, topk_num, nullptr, has_expert_map);
}


template <typename scalar_t, int32_t fill_threads>
__device__ void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded, int32_t max_num_m_blocks,
    int32_t inactive_expert_id, int32_t model_offset, int32_t topk_num,
    int32_t* token_mask, bool has_expert_map) {
  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  // Use an additional group of threads to fill sorted_token_ids.
  // Since the current kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    // Three __syncthreads() corresponding to the other threads
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    tokens_cnts[(tid + 1) * num_experts + expert_id] += mask;
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) *
              block_size;
    }
    total_tokens_post_pad[model_offset] =
        static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = tid;
    }
  }

  // Fill remaining expert_ids with -1
  const size_t fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (size_t i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (has_expert_map) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int32_t rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      sorted_token_ids[sorted_token_ids_offset + rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}


template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ expert_map, int32_t num_experts, int32_t block_size,
    size_t numel, int32_t max_num_tokens_padded, int32_t topk_num,
    bool has_expert_map) {
  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size), -1, 0, topk_num, nullptr,
      has_expert_map);
}


// void moe_align_block_size(
//     torch::Tensor topk_ids,
//     int64_t num_experts,
//     int64_t block_size,
//     torch::Tensor sorted_token_ids,
//     torch::Tensor experts_ids,
//     torch::Tensor num_tokens_post_pad,
//     torch::Tensor cumsum_buffer,
//     bool pad_sorted_token_ids) {
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//   int threads = 1024;
//   threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

//   int64_t max_num_tokens_padded = sorted_token_ids.size(0);

//   DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
//     bool small_batch_expert_mode = (topk_ids.numel() < 1024) && (num_experts <= 64);

//     if (small_batch_expert_mode) {
//       const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
//       constexpr int32_t fill_threads = 256;
//       const int32_t shared_mem_size = ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

//       auto small_batch_expert_kernel = moe_align_block_size_small_batch_expert_kernel<scalar_t, fill_threads>;
//       small_batch_expert_kernel<<<1, fill_threads + threads, shared_mem_size, stream>>>(
//           topk_ids.data_ptr<scalar_t>(),
//           sorted_token_ids.data_ptr<int32_t>(),
//           experts_ids.data_ptr<int32_t>(),
//           num_tokens_post_pad.data_ptr<int32_t>(),
//           num_experts,
//           block_size,
//           topk_ids.numel(),
//           pad_sorted_token_ids,
//           max_num_tokens_padded);
//     } else {
//       auto align_kernel = moe_align_block_size_kernel<scalar_t>;

//       const size_t scan_size = next_pow2(num_experts);
//       const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size + WARP_SIZE) * sizeof(int32_t);
//       align_kernel<<<2, threads, shared_mem_size, stream>>>(
//           topk_ids.data_ptr<scalar_t>(),
//           sorted_token_ids.data_ptr<int32_t>(),
//           experts_ids.data_ptr<int32_t>(),
//           num_tokens_post_pad.data_ptr<int32_t>(),
//           num_experts,
//           block_size,
//           topk_ids.numel(),
//           cumsum_buffer.data_ptr<int32_t>(),
//           pad_sorted_token_ids,
//           scan_size,
//           max_num_tokens_padded);

//       const int block_threads = std::min(256, (int)threads);
//       const int num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads;
//       const int max_blocks = 65535;
//       const int actual_blocks = std::min(num_blocks, max_blocks);

//       auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
//       sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
//           topk_ids.data_ptr<scalar_t>(),
//           sorted_token_ids.data_ptr<int32_t>(),
//           cumsum_buffer.data_ptr<int32_t>(),
//           topk_ids.numel());
//     }
//   });
// }

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          std::optional<torch::Tensor> maybe_expert_map) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(padded_num_experts < 1024,
              "padded_num_experts must be less than 1024");
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, options_int);
  }

  DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          // threadIdx.x >= fill_threads: counting experts and aligning
          // threadIdx.x < fill_threads: filling sorted_token_ids
          constexpr int32_t fill_threads = 256;
          auto small_batch_expert_kernel =
              moe_align_block_size_small_batch_expert_kernel<
                  scalar_t, fill_threads>;
          small_batch_expert_kernel<<<1, fill_threads + threads,
                                      shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              expert_map.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), sorted_token_ids.size(0), topk_ids.size(1),
              has_expert_map);
        } else {
          torch::Tensor cumsum_buffer =
              torch::empty({num_experts + 1}, options_int);
          auto align_kernel = moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          // launch two threadblocks
          // blockIdx.x == 0: counting experts and aligning
          // blockIdx.x == 1: filling sorted_token_ids
          align_kernel<<<2, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(),
              expert_map.data_ptr<int32_t>(), num_experts, padded_num_experts,
              experts_per_warp, block_size, topk_ids.numel(),
              cumsum_buffer.data_ptr<int32_t>(), sorted_token_ids.size(0),
              topk_ids.size(1), has_expert_map);

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);
          dim3 gridDims(1, actual_blocks);

          auto sort_kernel =
              count_and_sort_expert_tokens_kernel<scalar_t>;
          sort_kernel<<<gridDims, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(), expert_map.data_ptr<int32_t>(),
              topk_ids.numel(), num_experts, sorted_token_ids.size(0),
              topk_ids.size(1), has_expert_map);
        }
      });
}
