# Lsglang GPU、NUMA 双并行 [[English]](./README.md)

Lsglang是sglang的特别扩展，充分利用CPU和GPU计算资源，高效的GPU并行+NUMA并行架构，适用于MOE模型混合推理。

## 系统特性

- **GPU + NUMA 双并行**: 支持CPU-GPU混合解码、CPU-GPU混合预填充、GPU预填充三种计算方式[下个版本，等待几天]
- **显存 + 内存负载均衡**: 模型总体占用=显存+内存，容纳模型1+1=2, 100%显存利用率 <sup>注1</sup>
- **GPU 预填充优化**: GPU预填充与CPU-GPU混合解码并行，接近100%显卡利用率
- **NUMA 线程优化**: 跨节点通信占比低至3%，三级缓存命中50%以上，解码阶段可推动GPU负载达到33%至50%  

## 与sglang的关系

Lsglang使用最新的sglang源码，重新设计实现了MOE模型混合推理模块，保持了对sglang的100%完全兼容。

## 使用说明 [[English]](./README.md)
- [版本变更](#版本变更)
- [如何运行Qwen3.5-122B-A10B](#如何运行qwen35-122b-a10b)
- [如何运行Qwen3.5-397B-A17B](#如何运行qwen35-397b-a17b)
- [如何运行MiniMax-M2.5](#如何运行minimax-m25)
- [如何运行GLM5](#如何运行glm5)
- [如何运行Kimi K2.5](#如何运行kimi-k25)
- [如何运行Qwen3-Coder-Next-FP8](#如何运行qwen3-coder-next-fp8)
- [支持的模型](#支持的模型)
- [性能参考](#性能参考)
- [配置参数](#配置参数)
- [安装步骤](#安装步骤) 
- [更新](#更新)
- [优化](#优化)

## 版本变更
 
```bash  
2026-02-25: Lsglang-v1.0.6 - 修复已知问题，增加新模型支持  
2026-02-10：Lsglang-v1.0.0 -  来自LvLLM项目[https://github.com/guqiong96/Lvllm]的移植，验证了BF16、F16原版模型、FP8原版模型、AWQ 4bit对称量化模型。
 
```

## 如何运行Qwen3.5-122B-A10B
 
```bash

pip uninstall transformers -y
pip install transformers==4.57.6

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
    --model /home/guqiong/Models/Qwen3.5-122B-A10B \
    --served-model-name Qwen3.5-122B-A10B \
    --host 0.0.0.0 \
    --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 4 \
    --enable-p2p-check \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 4096 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16

```


## 如何运行Qwen3.5-397B-A17B
 
```bash

pip uninstall transformers -y
pip install transformers==4.57.6

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
    --max-prefill-tokens 4096 \
    --max-total-tokens 32768 \
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

## 如何运行MiniMax-M2.5
 
```bash

pip uninstall transformers -y
pip install transformers==4.57.6

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
    --max-prefill-tokens 4096 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16
```

## 如何运行glm5

1、安装最新的transformers
```bash  
pip uninstall transformers -y
pip install transformers
```

2、运行
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
    --max-prefill-tokens 4096 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16

    # --nsa-prefill-backend "tilelang" \
    # --nsa-decode-backend "tilelang" \
    
```

## 如何运行Kimi K2.5

```bash
pip uninstall transformers -y
pip install transformers==4.57.6

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
    --max-prefill-tokens 4096 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16
    
```

## 如何运行Qwen3-Coder-Next-FP8
 
```bash
pip uninstall transformers -y
pip install transformers==4.57.6

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
    --max-prefill-tokens 4096 \
    --max-total-tokens 32768 \
    --mem-fraction-static 0.90 \
    --tool-call-parser qwen3_coder \
    --attention-backend triton \
    --fp8-gemm-backend triton \
    --kv-cache-dtype bf16
```

## 支持的模型

Lsglang已验证的大部分原版MOE模型
 
| 模型名称 | 状态 |
|---------|------|
| Qwen3.5-397B-A17B | ✅ 已测试通过 |
| Qwen3-Coder-Next | ✅ 已测试通过 |
| Qwen3-Next-80B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-Coder-30B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-VL-30B-A3B-Instruct | ✅ 已测试通过 | 
| MiniMax-M2.1 | ✅ 已测试通过 |
| GLM-5 | ✅ 已测试通过 |
| GLM-4.7 | ✅ 已测试通过 |
| GLM-4.7-Flash  | ✅ 已测试通过 |
| GLM-4.6V | ✅ 已测试通过 |
| Kimi k2.5 | ✅ 已测试通过 |

未列出的Qwen3系列、GLM系列、MiniMax系列的原版MOE模型理论上支持，待实际测试。


## 尚未支持的模型

| 模型名称 | 状态 |
|---------|------|
| DeepSeek-V3.2| 待定 |
 

## 支持的模型权重格式及运行时格式

| 模型文件 | 运行时格式 | 
|---------|------------|
| bfloat16 | bfloat16/float16| 
| float16 | bfloat16/float16| 
| fp8模型 | fp8、fp8+bfloat16、fp8+W4A16 | 
| awq 4bit对称量化模型 <sup>注1</sup>| W4A16 | 

注1：https://hf-mirror.com/cyankiwi 提供AWQ 4bit对称量化模型

## 性能参考

| 模型 | 运行时格式 | 预填充速度(tokens/s) | 解码速度(tokens/s) | CPU | GPU |内存 |
|------|----------|---------------------|-------------------|----------|---------|---------|
| Qwen3-Next-80B-A3B-Instruct原版 | bfloat16 |15000 <sup>注1</sup> | 90 | 双路 EPYC 9555ES  | 单卡 Nvidia RTX Pro 6000 | 6400MT/s  |
| MiniMax-M2.1原版 | fp8+bfloat16 | 5000 <sup>注1</sup> | 29 | 双路 EPYC 9684x  | 单卡 Nvidia RTX 5090 | 4800MT/s  |

注1：开启GPU预填充，输入长度32K-64K

## 配置参数

| 环境变量 | 类型 | 默认值 | 说明 | 备注 |
|--------|------|--------|------|------|
| `LVLLM_MOE_NUMA_ENABLED` | 核心参数 | `0` | 是否启用混合推理: `1`-启用，`0`-禁用 | 设置为`0`禁用混合推理，行为与sglang相同 |
| `LK_THREAD_BINDING` | 性能参数 | `CPU_CORE` | 线程绑定策略: `CPU_CORE`-按CPU核心绑定，`NUMA_NODE`-按NUMA节点绑定 | 默认按CPU核心绑定, 遇到性能问题时可尝试按NUMA节点绑定 |
| `LK_THREADS` | 性能参数 | 自动计算 | 线程数量: 物理核心数-4 | 多GPU多进程时，物理核心数-4除以进程数量 |
| `OMP_NUM_THREADS` | 性能参数 | 系统逻辑核心数量 | OpenMP线程数: 设置为`LK_THREADS`相同 |   | 
| `LVLLM_MOE_USE_WEIGHT` | 性能参数 | `TO_DTYPE` | 运行时专家权重格式`TO_DTYPE`: 与config.yaml中dtype一致,bfloat16/float16, `KEEP`: 与模型一致，`INT4`: int4  |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU预填充参数 | 无 | 常驻GPU的MOE专家层`0`: 第0层，`0-1`: 第0层到第1层，`0,9`: 第0层和第9层 | 留足KV Cache显存后，分配多层可增加性能，并减少对应的内存占用，包含0层才有加速效果 |
| `LK_POWER_SAVING` | cpu节能 | 0 | `1`：启用cpu节能模式，`0`：禁用cpu节能模式 | 建议值：`0` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | 性能参数 | 0 | `0`：快速加载模型，`1`：慢速加载模型可避免OOM | 建议值：加载模型文件时，内存充裕使用`0`，内存紧张使用`1` |
| `LVLLM_MOE_QUANT_ON_GPU` | 性能参数 | 0 | `0`：不启用GPU专家量化，`1`：启用GPU专家量化 | 显存充足可启用（仅加载时有效，推理时不会额外占用显存），加快模型加载速度 |

## 安装步骤

### 1. 安装CUDA 12.9

```bash
# 卸载旧版本CUDA和NVIDIA驱动
sudo /usr/local/cuda/bin/cuda-uninstaller   
sudo nvidia-uninstall

# 下载并安装CUDA 12.9 
wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run
sudo sh cuda_12.9.1_575.57.08_linux.run
```

### 2. 创建Python环境

```bash
conda create -n Lsglang python==3.12.11
conda activate Lsglang

# 升级libstdcxx-ng（避免glibcxx版本问题）
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 安装NUMA库
sudo apt-get install libnuma-dev      # Ubuntu
sudo dnf install numactl-devel        # Rocky Linux
```

### 3. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/guqiong96/Lsglang.git
cd Lsglang

# 安装PyTorch 2.9.1
pip install torch==2.9.1
  
```
 
### 4. 安装Lsglang

```bash
pip install grpcio-tools 
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e "python" --no-build-isolation -vvv
pip install nvidia-cudnn-cu12==9.16.0.29
```

**参数说明：**
- `MAX_JOBS=32 NVCC_THREADS=1`: 减少编译内存占用
- `CMAKE_BUILD_TYPE=Release`: 性能优化选项
- `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release`: 性能优化选项
 
 
 
## 更新

如果已安装Lsglang，需要更新到最新版本，请执行以下命令：

```bash 
git fetch && git reset --hard origin/main && git clean -fd # 此命令适合普通用户，如果保留本地修改内容的用户应知道提前做处理

# 安装PyTorch 2.9.1 
pip uninstall torchaudio triton torchvision torch sglang
pip install torchaudio triton torchvision torch==2.9.1

# Qwen3-VL GLM4.6V 需要安装 xformers

#  卸载老版本
pip uninstall sglang lk_moe

# 编译安装  
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e "python" --no-build-isolation -vvv
pip install nvidia-cudnn-cu12==9.16.0.29
rm -rf ~/.cache/flashinfer
 
```
 
## 优化

### MoE常驻显存, 线性增加decode和prefill速度
```bash
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 # 0-5层MoE层常驻显存
# LVLLM_GPU_RESIDENT_MOE_LAYERS=0,1,8-9 # 0,1,8-9层MoE层常驻显存
# LVLLM_GPU_RESIDENT_MOE_LAYERS="" # 关闭MoE常驻显存
``` 
 

### 线程绑定到CPU核心
```bash
LK_THREAD_BINDING=CPU_CORE # 绑定到CPU核心（包括超线程逻辑核心）, 最佳性能
#LK_THREAD_BINDING=NUMA_NODE # 绑定到NUMA节点, 次优选择，解决部署在虚拟化平台的极端性能问题
``` 
### BIOS NUMA 设置
```bash
AMD EPYC：设置NPS4获得最佳性能
Intel XEON：设置SNC4获得最佳性能
通常：2,4,8个节点，最多支持32节点，节点越多越好，节点数为GPU倍数获得最佳性能 # 部分虚拟化平台或Intel平台不要设置5、10节点，设置2节点避免性能问题
```

### 线程数设置
```bash
线程数 <= （核心数 - x）/ 张量并行数（TP size）  # x 留给其它任务的线程，至少4线程
LK_THREADS=44                    # 96核心，2个GPU， 每个GPU 44线程， 88线程, 剩余8线程留给其它任务
线程数太大可能会引发性能问题        # 虽然系统会自动条件线程数，但建议手动设置进行测试
``` 

### 显存设置
```bash
--max-running-requests  4 # 最多4并发，常规节省显存
```
### CPU节能
```bash
LK_POWER_SAVING=1 # 开启后推理时降低CPU温度，性能轻微降低
```

### FP8模型权重运行时格式
```bash
LVLLM_MOE_USE_WEIGHT=INT4 # 模型MoE专家权重使用W4A16推理，其余部分依旧为FP8，开启几乎不影响精度， 速度排序：INT4 > TO_DTYPE > KEEP
```

### 模型加载
```bash
LVLLM_ENABLE_NUMA_INTERLEAVE=1 # 慢速加载模型可避免OOM，建议值：加载模型文件时，内存充裕使用`0`，内存紧张使用`1`
```

### 模型加载专家量化
```bash
LVLLM_MOE_QUANT_ON_GPU=1 # 启用GPU专家量化，建议值：显存充足可启用（仅加载时有效，推理时不会额外占用显存），加快模型加载速度
```





