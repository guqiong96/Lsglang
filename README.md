# Lsglang GPU、NUMA Dual Parallel [[中文]](./README_cn.md)

Lsglang is a special extension of sglang that fully utilizes CPU and GPU computing resources with an efficient GPU parallel + NUMA parallel architecture, suitable for MoE model hybrid inference.

## System Features

- **GPU + NUMA Dual Parallel**: Supports CPU-GPU hybrid decoding, CPU-GPU hybrid prefill, and GPU prefill computing modes
- **VRAM + Memory Load Balancing**: Total model occupancy = VRAM + Memory, accommodating 1+1=2 models with 100% VRAM utilization <sup>Note 1</sup>
- **GPU Prefill Optimization**: GPU prefill runs in parallel with CPU-GPU hybrid decoding, achieving nearly 100% GPU utilization
- **NUMA Thread Optimization**: Cross-node communication ratio as low as 3%, L3 cache hit rate over 50%, decoding phase can drive GPU load to 33% to 50%

## Relationship with sglang

Lsglang uses the latest sglang source code and has redesigned and implemented the MoE model hybrid inference module while maintaining 100% full compatibility with sglang.<sup>Note 1</sup>.

Note 1: x86 CPUs with AVX2 or above instruction sets and Nvidia GPUs are supported.

## Usage Guide [[中文]](./README_cn.md)
- [Version Changes](#version-changes)
- [How to Run Qwen3.5-122B-A10B](#how-to-run-qwen35-122b-a10b)
- [How to Run Qwen3.5-397B-A17B](#how-to-run-qwen35-397b-a17b)
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
2026-04-06: Lsglang-v1.2.0 - improve LK_POWER_SAVING=1 for power saving, support FP8+BF16+AWQ4bit hybrid MoE layer inference
2026-04-03: Lsglang-v1.1.4 - support local compilation of sgl-kernel, fix known issues
2026-03-11: Lsglang-v1.1.3 - FP8、AWQ4bit MoE Models enable GPU Prefill acceleration without additional memory occupation, FP8 MoE Model cancel TO_DTYPE runtime type conversion, KEEP model temporarily not support GPU Prefill
            Note 1：30 series graphics cards can enable GPU Prefill acceleration for FP8 models by removing the LVLLM_GPU_RESIDENT_MOE_LAYERS parameter.
2026-03-05: Lsglang-v1.1.0 - support GPU prefill, update corresponding commands (FP8 models do not support enabling GPU prefill on architectures below 3090)
2026-02-25: Lsglang-v1.0.6 - fix known issues, support new models 
2026-02-10: Lsglang-v1.0.0 - Ported from the LvLLM project [https://github.com/guqiong96/Lvllm], verified BF16, F16 original models, FP8 original models, and AWQ 4-bit symmetric quantization models.
```


## How to Run Qwen3.5-122B-A10B
```bash

pip uninstall transformers -y
pip install transformers==5.3.0

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
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
python -m sglang.launch_server \
    --model /home/guqiong/Models/Qwen3.5-122B-A10B \
    --served-model-name Qwen3.5-122B-A10B \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --enable-p2p-check \
    --chunked-prefill-size 32000 \
    --max-total-tokens 66000 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16 \
    --disable-piecewise-cuda-graph

```


## How to Run Qwen3.5-397B-A17B
  
```bash
pip uninstall transformers -y
pip install transformers==5.3.0

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
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/Qwen3.5-397B-A17B" \
    --served-model-name "Qwen3.5-397B-A17B" \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --enable-p2p-check \
    --chunked-prefill-size 32000 \
    --max-total-tokens 66000 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16 \
    --disable-piecewise-cuda-graph



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
pip uninstall transformers -y
pip install transformers==5.3.0

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
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/MiniMax-M2.5" \
    --served-model-name MiniMax-M2.5 \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --enable-p2p-check \
    --chunked-prefill-size 32000 \
    --max-total-tokens 66000 \
    --mem-fraction-static 0.90 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16 \
    --disable-piecewise-cuda-graph
```

```bash
# When encountering performance issues, try binding threads to NUMA nodes and reducing the number of threads
```

## How to Run GLM5
 
```bash  
pip uninstall transformers -y
pip install transformers==5.3.0
 
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
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/GLM-5-FP8" \
    --served-model-name "GLM-5-FP8" \
    --host "0.0.0.0" \
    --port "8070" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-p2p-check \
    --max-running-requests 2 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --chunked-prefill-size 32000 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16 \
    --disable-piecewise-cuda-graph \
    --disable-shared-experts-fusion



    
    # --nsa-prefill-backend "tilelang" \
    # --nsa-decode-backend "tilelang" \
```

## How to Run Kimi K2.5

```bash
pip uninstall transformers -y
pip install transformers==5.3.0

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
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/Kimi-K2.5" \
    --served-model-name "Kimi-K2.5" \
    --host "0.0.0.0" \
    --port "8070" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-p2p-check \
    --max-running-requests 2 \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --chunked-prefill-size 32000 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16 \
    --disable-piecewise-cuda-graph
    
```

## How to Run Qwen3-Coder-Next-FP8

```bash

pip uninstall transformers -y
pip install transformers==5.3.0

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
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
python -m sglang.launch_server \
    --model "/home/guqiong/Models/Qwen3-Coder-Next-FP8" \
    --served-model-name Qwen3-Coder-Next-FP8 \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --enable-p2p-check \
    --chunked-prefill-size 32000 \
    --max-total-tokens 66000 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16 \
    --disable-piecewise-cuda-graph
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
| `LVLLM_MOE_USE_WEIGHT` | Performance Parameter | `INT4` | Runtime expert weight format `KEEP`: same as model, `INT4`: int4 | |
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

pip install -U setuptools wheel scikit-build-core cmake

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
pip install torchaudio triton torchvision torch==2.9.1
```

### 4. Install Lsglang

```bash
pip install grpcio-tools
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e "python" --no-build-isolation -vvv

cd sgl-kernel
rm -rf build/ dist/ *.egg-info/
 MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip inst
all -e  . --no-build-isolation -vvv

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

# pip install -U setuptools wheel scikit-build-core cmake

# Qwen3-VL GLM4.6V requires xformers

# Uninstall old versions
pip uninstall sglang lk_moe

# Compile and install
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e "python" --no-build-isolation -vvv

# compile and install ssgl-kernel[first compile will download third-party projects to sgl-kernel/dep directory, or use manual download command]
pip uninstall sgl-kernel -y 
cd sgl-kernel
rm -rf build/ dist/ *.egg-info/
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install -e  . --no-build-isolation -vvv

pip install nvidia-cudnn-cu12==9.16.0.29
rm -rf ~/.cache/flashinfer/
rm -rf ~/.cache/sglang/ 
rm -rf ~/.triton/cache/
```

### manual download third-party projects for ssgl-kernel dependencies

```bash

cd /path/to/Lsglang/sgl-kernel

mkdir -p dep
cd dep

# 1. CUTLASS
git clone https://github.com/NVIDIA/cutlass.git repo-cutlass-src
cd repo-cutlass-src && git checkout 57e3cfb47a2d9e0d46eb6335c3dc411498efa198 && cd ..

# 2. DeepGEMM
git clone https://github.com/sgl-project/DeepGEMM.git repo-deepgemm-src
cd repo-deepgemm-src && git checkout ffe2b6b97420a9f8c58268ca55755168e6e2f360 && cd ..

# 3. fmt
git clone https://github.com/fmtlib/fmt.git repo-fmt-src
cd repo-fmt-src && git checkout 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28 && cd ..

# 4. Triton
git clone https://github.com/triton-lang/triton.git repo-triton-src
cd repo-triton-src && git checkout 0add68262ab0a2e33b84524346cb27cbb2787356 && cd ..

# 5. FlashInfer
git clone https://github.com/flashinfer-ai/flashinfer.git repo-flashinfer-src
cd repo-flashinfer-src && git checkout bc29697ba20b7e6bdb728ded98f04788e16ee021 && cd ..

# 6. Flash Attention
git clone https://github.com/sgl-project/sgl-attn.git repo-flash-attention-src
cd repo-flash-attention-src && git checkout bcf72ccc6816b36a5fae2c5a3c027604629785e0 && cd ..

# 7. MSCCLPP
git clone https://github.com/microsoft/mscclpp.git repo-mscclpp-src
cd repo-mscclpp-src && git checkout 51eca89d20f0cfb3764ccd764338d7b22cd486a6 && cd ..

# 8. FlashMLA
git clone https://github.com/sgl-project/FlashMLA.git repo-flashmla-src
cd repo-flashmla-src && git checkout 9804b12079e4c873514d3457aa588d3ccf40da28 && cd ..

# 9. DLPack
git clone https://github.com/dmlc/dlpack.git dlpack-src
cd dlpack-src && git checkout 3ea601bb413074c49a77c4ce3218bc08f8c4703c && cd ..

# 10. nanobind
git clone https://github.com/wjakob/nanobind.git nanobind-src
cd nanobind-src && git checkout 05cba0ef85ba2bb68aa115af4b74c30aa2aa7bec && cd ..

# 11. JSON
wget https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
tar -xf json.tar.xz
mv json json-src
rm json.tar.xz
 
``` 

## Optimization

### MoE Resident in VRAM, Linear Increase in Decode and Prefill Speed
```bash
# 0-5 MoE layers resident in VRAM
# 0,1,8-9 means 0,1,8-9 MoE layers resident in VRAM
# Some models have non-zero starting layer numbers, such as Step-3.5-Flash model starting at 3
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 
``` 
 
### Enable GPU Prefill
```bash
# Prefetch 1 layer, recommended value is 1, more is meaningless
LVLLM_GPU_PREFETCH_WINDOW=1 
# Start GPU prefill when input length reaches 4096, can be decreased or increased based on CPU prefill performance, starting prefill earlier or later
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 
# Same as context size for optimal performance, can be appropriately reduced based on VRAM availability, exceeding context size is meaningless
--chunked-prefill-size 32000 # 
``` 
 
### Disable GPU Prefill
```bash
# Disable GPU prefill
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE="" 
# 1024 to 8192, too large is meaningless (occupies too much VRAM and long startup time)
--chunked-prefill-size 4096
``` 
 
### Thread Binding to CPU Cores
```bash
# Bind to CPU cores (including hyper-threading logical cores), optimal performance
LK_THREAD_BINDING=CPU_CORE 
# Bind to NUMA nodes, second choice to solve extreme performance issues on virtualization platforms
LK_THREAD_BINDING=NUMA_NODE 
``` 
### BIOS NUMA Settings
```bash
AMD EPYC: Set NPS4 for optimal performance
Intel XEON: Set SNC4 for optimal performance
# Some virtualization platforms or Intel platforms should not set 5 or 10 nodes, set to 2 nodes to avoid performance issues
General: 2,4,8 nodes, maximum support for 32 nodes, more nodes are better, node count being a multiple of GPUs for optimal performance 
```
 
### Thread Count Settings
```bash
# Thread count <= (core count - x) / tensor parallelism size (TP size)  # x threads reserved for other tasks, at least 4 threads
# 96 cores, 2 GPUs, 44 threads per GPU, 88 threads total, 8 threads reserved for other tasks 
LK_THREADS=44                   
# if the total number of threads exceeds the physical core count, it may cause performance issues   
# although the system will automatically adjust the number of threads, it is recommended to manually set it for testing     

```

### VRAM Settings
```bash
# 24G VRAM with GPU prefill enabled, leave sufficient temporary VRAM for calculations, otherwise long context prefill performance will drop significantly, startup time will be too long
--mem-fraction-static 0.85  
# Maximum 4 concurrent, regular VRAM savings
--max-running-requests 4 
# Save VRAM when GPU prefill is disabled, performance remains unchanged, but if enable GPU prefill will cause performance drop
--chunked-prefill-size 4096
# or larger and less than context size, enable GPU prefill, obtain best performance, but if disable GPU prefill will cause performance drop
--chunked-prefill-size 32000 
```
### CPU Power Saving
```bash
# enable low power mode while inference, reduce CPU temperature, slightly reduce performance 
LK_POWER_SAVING=1 
```

### FP8 Model Weight Runtime Format
```bash
 # Model MoE expert weights use INT4 inference, other parts remain original model type, enabling almost no impact on accuracy, speed order: INT4 > KEEP
LVLLM_MOE_USE_WEIGHT=INT4
```

### Model Loading with NUMA Interleaving
```bash
# Slow model loading can prevent OOM. Recommended values: use `0` when memory is sufficient during model file loading, use `1` when memory is limited.
LVLLM_ENABLE_NUMA_INTERLEAVE=1 
```

### Model Loading with GPU Expert Quantization
```bash
# Enable GPU expert quantization, recommended values: enable when sufficient GPU memory is available (only effective during model loading, not during inference), speed up model loading
LVLLM_MOE_QUANT_ON_GPU=1 
```

### CPU Prefill Optimization
```bash
# It is allowed to increase the CPU prefill speed by increasing the max_num_batched_tokens parameter, for example --chunked-prefill-size 4096. If GPU prefill is enabled, the smaller value between LVLLM_GPU_PREFILL_MIN_BATCH_SIZE and max_num_batched_tokens will be used.
--chunked-prefill-size 4096
```
