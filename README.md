# Lsglang GPU + NUMA Dual Parallel [[中文]](./README_cn.md)

Lsglang is a special extension of sglang that fully utilizes CPU and GPU computing resources with an efficient GPU parallel + NUMA parallel architecture, suitable for MOE model hybrid inference.

## System Features

- **GPU + NUMA Dual Parallel**: Supports three computing modes: CPU-GPU hybrid decoding, CPU-GPU hybrid prefill, and GPU prefill
- **VRAM + Memory Load Balancing**: Total model footprint = VRAM + memory, accommodating model 1+1=2, 100% VRAM utilization <sup>Note 1</sup>
- **GPU Prefill Optimization**: GPU prefill runs in parallel with CPU-GPU hybrid decoding, achieving nearly 100% GPU utilization
- **NUMA Thread Optimization**: Cross-node communication as low as 3%, L3 cache hit rate over 50%, GPU load can reach 33% to 50% during decoding  

## Relationship with sglang

Lsglang uses the latest sglang source code and has redesigned and implemented the MOE model hybrid inference module, maintaining 100% full compatibility with sglang<sup>Note 1</sup>.

Note 1: x86 CPUs with AVX2+ instruction sets and Nvidia GPUs with sm80+ architectures

## Usage Guide [[中文]](./README.md)
- [Version Changes](#version-changes)
- [Supported Models](#supported-models)
- [Supported Quantization Formats](#supported-quantization-formats)
- [Running Command Reference](#running-command-reference)
- [Configuration Parameters](#configuration-parameters)
- [Installation Steps](#installation-steps)
- [Optimization](#optimization)

## Version Changes
 
```bash
2026-06-05: Lsglang-v1.3.0 - Upgraded lk_moe module, supports nvfp4, mxfp4 quantization types, added LVLLM_GPU_RESIDENT_MOE_EXPERTS, removed LVLLM_MOE_USE_WEIGHT, LVLLM_MOE_QUANT_ON_GPU
2026-04-06: Lsglang-v1.2.0 - Enhanced energy saving effect with LK_POWER_SAVING=1, supports mixed MOE layer inference with FP8+BF16+AWQ4bit
2026-04-03: Lsglang-v1.1.4 - Supports local compilation of sgl-kernel to fix known issues
2026-03-11: Lsglang-v1.1.3 - FP8 and AWQ4bit models no longer occupy additional memory when GPU Prefill is enabled, FP8 models removed TO_DTYPE runtime type conversion, KEEP temporarily does not support GPU Prefill
                             Note 1: RTX 30 series GPUs can enable GPU Prefill for FP8 models by removing the LVLLM_GPU_RESIDENT_MOE_LAYERS parameter
2026-03-05: Lsglang-v1.1.0 - Supports GPU prefill, updated corresponding commands (FP8 models not supported on RTX 3090 and below architectures)
2026-02-25: Lsglang-v1.0.6 - Fixed known issues, added new model support  
2026-02-10: Lsglang-v1.0.0 -  Ported from LvLLM project [https://github.com/guqiong96/Lvllm], verified BF16, F16 original models, FP8 original models, AWQ 4bit symmetric quantization models.
 
```
 
## Supported Models

Most original MOE models verified by Lsglang
 
| Model Name | Status |
|---------|------|
| gemma-4-26B-A4B-it | ✅ Tested |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | ✅ Tested |
| Qwen3.6-35B-A3B | ✅ Tested |
| Qwen3.5-35B-A3B | ✅ Tested |
| Qwen3.5-122B-A10B | ✅ Tested |
| Qwen3.5-397B-A17B | ✅ Tested |
| Qwen3-Coder-Next | ✅ Tested |
| Qwen3-Next-80B-A3B-Instruct | ✅ Tested |
| Qwen3-Coder-30B-A3B-Instruct | ✅ Tested |
| Qwen3-VL-30B-A3B-Instruct | ✅ Tested | 
| MiniMax-M2.7 | ✅ Tested |
| MiniMax-M2.5 | ✅ Tested |
| MiniMax-M2.1 | ✅ Tested |
| GLM-5.1-FP8 | ✅ Tested |
| GLM-5.0-FP8 | ✅ Tested |
| GLM-4.7 | ✅ Tested |
| GLM-4.7-Flash  | ✅ Tested |
| GLM-4.6V | ✅ Tested |
| Kimi k2.6 | ✅ Tested |
| Kimi k2.5 | ✅ Tested |

Unlisted original MOE models from Qwen3 series, GLM series, and MiniMax series are theoretically supported and pending actual testing.



## Supported Quantization Formats

| Model File | Runtime Format | 
|---------|------------|
| bfloat16 | bfloat16/float16| 
| float16 | bfloat16/float16| 
| fp8 model | fp8 | 
| nvfp4 model | nvfp4 | 
| mxfp4 model | mxfp4 | 
| awq 4bit symmetric quantization model <sup>Note 1</sup>| w4a16 | 

Note 1: https://hf-mirror.com/cyankiwi provides AWQ 4bit symmetric quantization models
 
## Running Command Reference
 
```bash 

LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-1,33-34 \
LVLLM_GPU_RESIDENT_MOE_EXPERTS=64 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1 \
python -m sglang.launch_server \
    --model /home/guqiong/Models/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --chunked-prefill-size 32000 \
    --max-total-tokens 66000 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --disable-shared-experts-fusion

```


## Configuration Parameters

| Environment Variable | Type | Default Value | Description | Remarks |
|--------|------|--------|------|------|
| `LVLLM_MOE_NUMA_ENABLED` | Core Parameter | `0` | Enable hybrid inference: `1`-enable, `0`-disable | Set to `0` to disable hybrid inference, behavior same as vLLM |
| `LK_THREAD_BINDING` | Performance Parameter | `CPU_CORE` | Thread binding strategy: `CPU_CORE`-bind by CPU core, `NUMA_NODE`-bind by NUMA node | Default bind by CPU core, try NUMA node binding when encountering performance issues |
| `LK_THREADS` | Performance Parameter | Auto calculated | Thread count: physical cores - 4 | For multi-GPU multi-process: (physical cores - 4) / number of processes |
| `OMP_NUM_THREADS` | Performance Parameter | System logical core count | OpenMP thread count: set to same as `LK_THREADS` |   | 
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU Prefill Parameter | None | MOE expert layers resident on GPU: `0`-layer 0, `0-1`-layers 0 to 1, `0,9`-layers 0 and 9 | After reserving KV Cache VRAM, allocating multiple layers improves performance and reduces corresponding memory usage |
| `LVLLM_GPU_PREFETCH_WINDOW` | GPU Prefill Parameter | None | Prefetch window size: `1`-prefetch 1 layer of MOE experts | Typically prefetch 1 to 2 layers |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | GPU Prefill Parameter | None | Minimum input length for GPU prefill: `4096`-GPU prefill starts when input length reaches this value | Should not be set too small, set to 0 to disable GPU prefill |
| `LK_POWER_SAVING` | CPU Power Saving | 0 | `1`: enable CPU power saving mode, `0`: disable | Recommended: `0` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | Performance Parameter | 0 | `0`: fast model loading, `1`: slow loading to avoid OOM | Recommendation: use `0` when memory is abundant, `1` when memory is tight |
| `LVLLM_GPU_RESIDENT_MOE_EXPERTS` | GPU Prefill Parameter | None | Number of MOE experts resident on GPU: `64`-64 experts per layer|


## Installation Steps

### 1. Install CUDA 13.2.1

```bash
# Uninstall old CUDA and NVIDIA driver
sudo /usr/local/cuda/bin/cuda-uninstaller   
sudo nvidia-uninstall

# Download and install CUDA 13.2.1 
wget https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_595.58.03_linux.run
sudo sh cuda_13.2.1_595.58.03_linux.run
```

### 2. Create Python Environment

```bash
conda create -n Lsglang python==3.12.11
conda activate Lsglang
  
# Upgrade libstdcxx-ng (avoid glibcxx version issues)
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install NUMA library
sudo apt-get install libnuma-dev      # Ubuntu
sudo dnf install numactl-devel        # Rocky Linux
```

### 3. Install Lsglang

```bash
pip install lsglang
```
 
## Compile from Source and Install Lsglang

```bash
# 克隆仓库
git clone https://github.com/guqiong96/Lsglang.git
cd Lsglang
pip install -U setuptools wheel scikit-build-core cmake
pip install torchaudio triton torchvision torch==2.11.0
pip install grpcio-tools 
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e "python" --no-build-isolation -vvv
```

**Parameter Explanation:**
- `MAX_JOBS=32 NVCC_THREADS=1`: Reduce compilation memory usage
- `CMAKE_BUILD_TYPE=Release`: Performance optimization option
- `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release`: Performance optimization option
 


## Optimization

### MoE Resident in VRAM, Linear Increase in Decode and Prefill Speed
```bash
# MoE layers 0-5 resident in VRAM
# Format 0,1,8-9 means MoE layers 0,1,8-9 resident in VRAM
# Some models start at non-zero layer numbers, e.g., Step-3.5-Flash starts at layer 3 
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 
``` 

### Enable GPU Prefill
```bash
# Prefetch 1 layer
LVLLM_GPU_PREFETCH_WINDOW=1
# GPU prefill starts when input length reaches 4096
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 
#配合修改最大批处理大小
--chunked-prefill-size 32000 
``` 

### Disable GPU Prefill
```bash
# Disable GPU prefill
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0
#配合修改最大批处理大小
--chunked-prefill-size 4096 
``` 

### Thread Binding to CPU Cores
```bash
# Bind to CPU cores (including hyper-threading logical cores), best performance
LK_THREAD_BINDING=CPU_CORE 
# Bind to NUMA nodes, second best option, resolves extreme performance issues on virtualization platforms and multi-instance running
LK_THREAD_BINDING=NUMA_NODE 
``` 
### BIOS NUMA Settings
```bash
AMD EPYC: Set NPS4 for best performance
Intel XEON: Set SNC4 for best performance
# Some virtualization platforms or Intel platforms should not set 5 or 10 nodes, set 2 nodes to avoid performance issues
Generally: 2, 4, 8 nodes, supports up to 32 nodes, more nodes is better, node count as multiple of GPU count for best performance 
```

### Thread Count Settings
```bash
# Thread count <= (core count - x) / tensor parallelism (TP size)  x is threads reserved for other tasks, at least 4 threads
# 96 cores, 2 GPUs, 44 threads per GPU, 88 threads total, 8 threads remaining for other tasks
LK_THREADS=44                    
# Total threads exceeding physical core count may cause performance issues   
# Although the system will automatically adjust thread count, manual setting is recommended for testing     
```

### VRAM Settings
```bash 
# Maximum batch size occupies significant VRAM, adjust accordingly
--chunked-prefill-size 32000  
```
### CPU Power Saving
```bash
# When enabled, reduces CPU temperature during inference with slight performance decrease
LK_POWER_SAVING=1 
```
