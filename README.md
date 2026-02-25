# Lsglang GPU、NUMA Dual Parallel [[中文]](./README_cn.md)

Lsglang is a special extension of sglang that fully utilizes CPU and GPU computing resources with an efficient GPU parallel + NUMA parallel architecture, suitable for MoE model hybrid inference.

## System Features

- **GPU + NUMA Dual Parallel**: Supports CPU-GPU hybrid decoding, CPU-GPU hybrid prefill, and GPU prefill computing modes [next version, coming in a few days]
- **VRAM + Memory Load Balancing**: Total model occupancy = VRAM + Memory, accommodating 1+1=2 models with 100% VRAM utilization <sup>Note 1</sup>
- **GPU Prefill Optimization**: GPU prefill runs in parallel with CPU-GPU hybrid decoding, achieving nearly 100% GPU utilization
- **NUMA Thread Optimization**: Cross-node communication ratio as low as 3%, L3 cache hit rate over 50%, decoding phase can drive GPU load to 33% to 50%

## Relationship with sglang

Lsglang uses the latest sglang source code and has redesigned and implemented the MoE model hybrid inference module while maintaining 100% full compatibility with sglang.

## Usage Guide [[中文]](./README_cn.md)
- [Version Changes](#version-changes)
- [How to Run Qwen3.5](#how-to-run-qwen35)
- [How to Run MiniMax-M2.5](#how-to-run-minimax-m25)
- [How to Run GLM5](#how-to-run-glm5)
- [How to Run Kimi K2.5](#how-to-run-kimi-k25)
- [How to Run Qwen3-Coder-Next-FP8](#how-to-run-qwen3-coder-next-fp8)
- [Supported Models](#supported-models)
- [Performance Reference](#performance-reference)
- [Configuration Parameters](#configuration-parameters)
- [Installation Steps](#installation-steps)
- [Update](#update)
- [Optimization](#optimization)

## Version Changes

```bash
2026-02-22: Lsglang-v1.0.4 - fix known issues, support new models  
2026-02-18: Lsglang-v1.0.3 - fix known issues, support new models  
2026-02-10: Lsglang-v1.0.0 - Ported from the LvLLM project [https://github.com/guqiong96/Lvllm], verified BF16, F16 original models, FP8 original models, and AWQ 4-bit symmetric quantization models.
```

## How to Run Qwen3.5

1、Install or update Lsglang to the latest version [following the installation steps or update steps in the documentation]

2、Run Qwen3.5-397B-A17B
```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h

PYTORCH_ALLOC_CONF=expandable_segments:True \
SGLANG_FORCE_FP8_MARLIN=1 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/Qwen3.5-397B-A17B" \
    --served-model-name "Qwen3.5-397B-A17B" \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 4 \
    --enable-p2p-check \
    --chunked-prefill-size 4096 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16



    # Multi-Token Prediction (MTP) \
    # --reasoning-parser qwen3 \
    # --speculative-algo NEXTN \
    # --speculative-num-steps 3 \
    # --speculative-eagle-topk 1 \
    # --speculative-num-draft-tokens 4 \
    # Processing Ultra-Long Texts
    # --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' 

```

## How to Run MiniMax-M2.5

```bash

PYTORCH_ALLOC_CONF=expandable_segments:True \
SGLANG_FORCE_FP8_MARLIN=1 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/MiniMax-M2.5" \
    --served-model-name MiniMax-M2.5 \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 4 \
    --enable-p2p-check \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 32768 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16
```

```bash
# When encountering performance issues, try binding threads to NUMA nodes and reducing the number of threads
```

## How to Run GLM5

1、Install the latest transformers
```bash  
pip uninstall transformers -y
pip install transformers-no-cache-dir
```

2、Run GLM5
```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h

PYTORCH_ALLOC_CONF=expandable_segments:True \
SGLANG_FORCE_FP8_MARLIN=1 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/GLM-5-FP8" \
    --served-model-name "GLM-5-FP8" \
    --host "0.0.0.0" \
    --port "8070" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-p2p-check \
    --max-running-requests 4 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 32768 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16



    
    # --nsa-prefill-backend "tilelang" \
    # --nsa-decode-backend "tilelang" \
```

## How to Run Kimi K2.5

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h

PYTORCH_ALLOC_CONF=expandable_segments:True \
SGLANG_FORCE_FP8_MARLIN=1 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/Kimi-K2.5" \
    --served-model-name "Kimi-K2.5" \
    --host "0.0.0.0" \
    --port "8070" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-p2p-check \
    --max-running-requests 4 \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 32768 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16
    
```

## How to Run Qwen3-Coder-Next-FP8

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
SGLANG_FORCE_FP8_MARLIN=1 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
NCCL_SOCKET_IFNAME=lo \
NCCL_IB_DISABLE=1 \
GLOO_SOCKET_IFNAME=lo \
NCCL_SOCKET_TIMEOUT=600000 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 OMP_NUM_THREADS=44 \
LVLLM_MOE_USE_WEIGHT=INT4 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_MOE_QUANT_ON_GPU=1 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/Qwen3-Coder-Next-FP8" \
    --served-model-name Qwen3-Coder-Next-FP8 \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 4 \
    --enable-p2p-check \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 32768 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16
```


## Supported Models

Most verified original MoE models in Lsglang:

| Model Name | Status |
|------------|--------|
| Qwen3.5-397B-A17B | ✅ Tested |
| Qwen3-Coder-Next | ✅ Tested |
| Qwen3-Next-80B-A3B-Instruct | ✅ Tested |
| Qwen3-Coder-30B-A3B-Instruct | ✅ Tested |
| Qwen3-VL-30B-A3B-Instruct | ✅ Tested |
| MiniMax-M2.1 | ✅ Tested |
| GLM-4.7 | ✅ Tested |
| GLM-4.7-Flash | ✅ Tested |
| GLM-4.6V | ✅ Tested |
| Kimi k2.5 | ✅ Tested |

Unlisted original MoE models from the Qwen3, GLM, and MiniMax series are theoretically supported and await actual testing.

## Unsupported Models

| Model Name | Status |
|------------|--------|
| DeepSeek-V3.2 | Pending |

## Supported Model Weight Formats and Runtime Formats

| Model File | Runtime Format |
|------------|----------------|
| bfloat16 | bfloat16/float16 |
| float16 | bfloat16/float16 |
| fp8 model | fp8, fp8+bfloat16, fp8+W4A16 |
| awq 4-bit symmetric quantization model <sup>Note 1</sup> | W4A16 |

Note 1: https://hf-mirror.com/cyankiwi provides AWQ 4-bit symmetric quantization models

## Performance Reference

| Model | Runtime Format | Prefill Speed (tokens/s) | Decode Speed (tokens/s) | CPU | GPU | Memory |
|-------|----------------|--------------------------|------------------------|-----|-----|--------|
| Original Qwen3-Next-80B-A3B-Instruct | bfloat16 | 15000 <sup>Note 1</sup> | 90 | Dual EPYC 9555ES | Single Nvidia RTX Pro 6000 | 6400MT/s |
| Original MiniMax-M2.1 | fp8+bfloat16 | 5000 <sup>Note 1</sup> | 29 | Dual EPYC 9684x | Single Nvidia RTX 5090 | 4800MT/s |

Note 1: GPU prefill enabled, input length 32K-64K

## Configuration Parameters
| Environment Variable | Type | Default | Description | Remarks |
|----------------------|------|---------|-------------|---------|
| `LVLLM_MOE_NUMA_ENABLED` | Core Parameter | `0` | Whether to enable hybrid inference: `1`-enable, `0`-disable | Set to `0` to disable hybrid inference, behavior is the same as sglang |
| `LK_THREAD_BINDING` | Performance Parameter | `CPU_CORE` | Thread binding strategy: `CPU_CORE`-bind by CPU core, `NUMA_NODE`-bind by NUMA node | Default bind by CPU core, try binding by NUMA node when encountering performance issues |
| `LK_THREADS` | Performance Parameter | Auto-calculated | Number of threads: physical core count - 4 | For multi-GPU multi-process, (physical core count - 4) divided by number of processes |
| `OMP_NUM_THREADS` | Performance Parameter | System logical core count | OpenMP thread count: set to the same as `LK_THREADS` | |
| `LVLLM_MOE_USE_WEIGHT` | Performance Parameter | `TO_DTYPE` | Runtime expert weight format `TO_DTYPE`: same as dtype in config.yaml, bfloat16/float16, `KEEP`: same as model, `INT4`: int4 | |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU Prefill Parameter | None | MoE expert layers resident in GPU memory `0`: layer 0, `0-1`: layers 0 to 1, `0,9`: layers 0 and 9 | After reserving enough KV Cache memory, allocating multiple layers can increase performance and reduce corresponding memory usage, including layer 0 for acceleration |
| `LK_POWER_SAVING` | CPU Power Saving | 0 | `1`: enable CPU power saving mode, `0`: disable CPU power saving mode | Recommended value: `0` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | Performance Parameter | 0 | `0`: fast model loading, `1`: slow model loading to avoid OOM | Recommended value: use `0` when memory is sufficient, use `1` when memory is tight |
| `LVLLM_MOE_QUANT_ON_GPU` | Performance Parameter | 0 | `0`：enable CPU expert quantization, `1`：enable GPU expert quantization | enable if GPU memory is abundant (only effective at loading time, inference will not occupy extra GPU memory)，accelerate model loading speed |

## Installation Steps

### 1. Install CUDA 12.9

```bash
# Uninstall old versions of CUDA and NVIDIA drivers
sudo /usr/local/cuda/bin/cuda-uninstaller
sudo nvidia-uninstall

# Download and install CUDA 12.9
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run
```

### 2. Create Python Environment

```bash
conda create -n Lsglang python==3.12.11
conda activate Lsglang

# Upgrade libstdcxx-ng (to avoid glibcxx version issues)
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install NUMA library
sudo apt-get install libnuma-dev      # Ubuntu
sudo dnf install numactl-devel        # Rocky Linux
```

### 3. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/guqiong96/Lsglang.git
cd Lsglang

# Install PyTorch 2.9.1
pip install torch==2.9.1
```

### 4. Install Lsglang

```bash
pip install grpcio-tools
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e "python" --no-build-isolation -vvv
pip install nvidia-cudnn-cu12==9.16.0.29
```

**Parameter Explanation:**
- `MAX_JOBS=32 NVCC_THREADS=1`: Reduce compilation memory usage
- `CMAKE_BUILD_TYPE=Release`: Performance optimization option
- `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"`: Performance optimization option

## Update

If you have already installed Lsglang and need to update to the latest version, execute the following commands:

```bash
git fetch && git reset --hard origin/main && git clean -fd # This command is suitable for regular users; users who want to keep local modifications should know how to handle it in advance

# Install PyTorch 2.9.1
pip uninstall torchaudio triton torchvision torch sglang
pip install torchaudio triton torchvision torch==2.9.1

# Qwen3-VL GLM4.6V requires xformers

# Uninstall old versions
pip uninstall sglang lk_moe

# Compile and install
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e "python" --no-build-isolation -vvv
pip install nvidia-cudnn-cu12==9.16.0.29
rm -rf ~/.cache/flashinfer
```

## Optimization

### MoE Resident in VRAM, Linearly Increase Decode and Prefill Speed
```bash
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 # Layers 0-5 of MoE layers reside in VRAM
# LVLLM_GPU_RESIDENT_MOE_LAYERS=0,1,8-9 # Layers 0,1,8-9 of MoE layers reside in VRAM
# LVLLM_GPU_RESIDENT_MOE_LAYERS="" # Disable MoE resident in VRAM
```

### Thread Binding to CPU Cores
```bash
LK_THREAD_BINDING=CPU_CORE # Bind to CPU cores (including hyper-threaded logical cores), best performance
#LK_THREAD_BINDING=NUMA_NODE # Bind to NUMA nodes, second choice, solve extreme performance issues when deployed on virtualization platforms
```

### BIOS NUMA Settings
```bash
AMD EPYC: Set NPS4 for best performance
Intel XEON: Set SNC4 for best performance
Generally: 2,4,8 nodes, maximum support for 32 nodes, more nodes are better, best performance when the number of nodes is a multiple of GPUs # Some virtualization platforms or Intel platforms should not set 5 or 10 nodes, set 2 nodes to avoid performance issues
```

### Thread Count Settings
```bash
Thread count <= (core count - x) / tensor parallelism size (TP size)  # x is threads reserved for other tasks, at least 4 threads
LK_THREADS=44                    # 96 cores, 2 GPUs, 44 threads per GPU, 88 threads in total, 8 threads remaining for other tasks
Too many threads may cause performance issues        # Although the system will automatically adjust the number of threads, manual setting is recommended for testing
```

### VRAM Settings
```bash
--max-running-requests 4 # Maximum 4 concurrent requests, regular VRAM saving
```

### CPU Power Saving
```bash
LK_POWER_SAVING=1 # Reduce CPU temperature during inference, slight performance degradation
```

### FP8 Model Weight Runtime Format
```bash
LVLLM_MOE_USE_WEIGHT=INT4 # MoE expert weights use W4A16 inference, other parts remain FP8, almost no impact on accuracy, speed order: INT4 > TO_DTYPE > KEEP
```
### Model Loading with NUMA Interleaving
```bash
LVLLM_ENABLE_NUMA_INTERLEAVE=1 # Slow model loading can prevent OOM. Recommended values: use `0` when memory is sufficient during model file loading, use `1` when memory is limited.
```

### Model Loading with GPU Expert Quantization
```bash
LVLLM_MOE_QUANT_ON_GPU=1 # Enable GPU expert quantization, recommended values: enable when sufficient GPU memory is available (only effective during model loading, not during inference), speed up model loading
```