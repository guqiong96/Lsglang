# Lsglang GPU、NUMA 双并行 [[English]](./README.md)

Lsglang是sglang的特别扩展，充分利用CPU和GPU计算资源，高效的GPU并行+NUMA并行架构，适用于MOE模型混合推理。

## 系统特性

- **GPU + NUMA 双并行**: 支持CPU-GPU混合解码、CPU-GPU混合预填充、GPU预填充三种计算方式
- **显存 + 内存负载均衡**: 模型总体占用=显存+内存，容纳模型1+1=2, 100%显存利用率 <sup>注1</sup>
- **GPU 预填充优化**: GPU预填充与CPU-GPU混合解码并行，接近100%显卡利用率
- **NUMA 线程优化**: 跨节点通信占比低至3%，三级缓存命中50%以上，解码阶段可推动GPU负载达到33%至50%  

## 与sglang的关系

Lsglang使用最新的sglang源码，重新设计实现了MOE模型混合推理模块，保持了对sglang的100%完全兼容<sup>注1</sup>。

注1：x86带有AVX2以上指令集的CPU和Nvidia GPU sm80以上架构

## 使用说明 [[English]](./README.md)
- [版本变更](#版本变更)
- [支持的模型](#支持的模型)
- [支持的量化格式](#支持的量化格式)
- [运行命令参考](#运行命令参考)
- [配置参数](#配置参数)
- [安装步骤](#安装步骤)
- [优化](#优化)

## 版本变更
 
```bash
2026-06-05: Lsglang-v1.3.0 - 升级lk_moe模块, 支持nvfp4, mxfp4量化类型，增加LVLLM_GPU_RESIDENT_MOE_EXPERTS, 取消LVLLM_MOE_USE_WEIGHT、LVLLM_MOE_QUANT_ON_GPU
2026-04-06: Lsglang-v1.2.0 - 增强使用LK_POWER_SAVING=1节能效果，支持FP8+BF16+AWQ4bit的混合MOE层推理
2026-04-03: Lsglang-v1.1.4 - 支持本地编译sgl-kernel，以修复已知问题
2026-03-11: Lsglang-v1.1.3 - FP8、AWQ4bit模型开启GPU Prefill加速不再占用额外内存, FP8模型取消TO_DTYPE运行时类型转换、KEEP暂不支持开启GPU Prefill
                             注1：30系显卡可以通过去掉LVLLM_GPU_RESIDENT_MOE_LAYERS参数，从而开启FP8模型的GPU Prefill加速
2026-03-05: Lsglang-v1.1.0 - 支持GPU预填充，更新相应命令（FP8模型在3090及以下架构不支持开启）
2026-02-25: Lsglang-v1.0.6 - 修复已知问题，增加新模型支持  
2026-02-10：Lsglang-v1.0.0 -  来自LvLLM项目[https://github.com/guqiong96/Lvllm]的移植，验证了BF16、F16原版模型、FP8原版模型、AWQ 4bit对称量化模型。
 
```
 
## 支持的模型

Lsglang已验证的大部分原版MOE模型
 
| 模型名称 | 状态 |
|---------|------|
| gemma-4-26B-A4B-it | ✅ 已测试通过 |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | ✅ 已测试通过 |
| Qwen3.6-35B-A3B | ✅ 已测试通过 |
| Qwen3.5-35B-A3B | ✅ 已测试通过 |
| Qwen3.5-122B-A10B | ✅ 已测试通过 |
| Qwen3.5-397B-A17B | ✅ 已测试通过 |
| Qwen3-Coder-Next | ✅ 已测试通过 |
| Qwen3-Next-80B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-Coder-30B-A3B-Instruct | ✅ 已测试通过 |
| Qwen3-VL-30B-A3B-Instruct | ✅ 已测试通过 | 
| MiniMax-M2.7 | ✅ 已测试通过 |
| MiniMax-M2.5 | ✅ 已测试通过 |
| MiniMax-M2.1 | ✅ 已测试通过 |
| GLM-5.1-FP8 | ✅ 已测试通过 |
| GLM-5.0-FP8 | ✅ 已测试通过 |
| GLM-4.7 | ✅ 已测试通过 |
| GLM-4.7-Flash  | ✅ 已测试通过 |
| GLM-4.6V | ✅ 已测试通过 |
| Kimi k2.6 | ✅ 已测试通过 |
| Kimi k2.5 | ✅ 已测试通过 |

未列出的Qwen3系列、GLM系列、MiniMax系列的原版MOE模型理论上支持，待实际测试。



## 支持的量化格式

| 模型文件 | 运行时格式 | 
|---------|------------|
| bfloat16 | bfloat16/float16| 
| float16 | bfloat16/float16| 
| fp8模型 | fp8 | 
| nvfp4模型 | nvfp4 | 
| mxfp4模型 | mxfp4 | 
| awq 4bit对称量化模型 <sup>注1</sup>| w4a16 | 

注1：https://hf-mirror.com/cyankiwi 提供AWQ 4bit对称量化模型
 

## 运行命令参考
 
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


## 配置参数

| 环境变量 | 类型 | 默认值 | 说明 | 备注 |
|--------|------|--------|------|------|
| `LVLLM_MOE_NUMA_ENABLED` | 核心参数 | `0` | 是否启用混合推理: `1`-启用，`0`-禁用 | 设置为`0`禁用混合推理，行为与vLLM相同 |
| `LK_THREAD_BINDING` | 性能参数 | `CPU_CORE` | 线程绑定策略: `CPU_CORE`-按CPU核心绑定，`NUMA_NODE`-按NUMA节点绑定 | 默认按CPU核心绑定, 遇到性能问题时可尝试按NUMA节点绑定 |
| `LK_THREADS` | 性能参数 | 自动计算 | 线程数量: 物理核心数-4 | 多GPU多进程时，物理核心数-4除以进程数量 |
| `OMP_NUM_THREADS` | 性能参数 | 系统逻辑核心数量 | OpenMP线程数: 设置为`LK_THREADS`相同 |   | 
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU预填充参数 | 无 | 常驻GPU的MOE专家层`0`: 第0层，`0-1`: 第0层到第1层，`0,9`: 第0层和第9层 | 留足KV Cache显存后，分配多层可增加性能，并减少对应的内存占用 |
| `LVLLM_GPU_PREFETCH_WINDOW` | GPU预填充参数 | 无 | 预取窗口大小`1`: 预取1层MOE专家 |  一般预取1到2层即可 |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | GPU预填充参数 | 无 | 使用GPU预填充的最小输入长度`4096`：输入长度达到该值后，启动GPU预填充 | 设置值不宜过小，设置为0则关闭GPU预填充功能 |
| `LK_POWER_SAVING` | cpu节能 | 0 | `1`：启用cpu节能模式，`0`：禁用cpu节能模式 | 建议值：`0` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | 性能参数 | 0 | `0`：快速加载模型，`1`：慢速加载模型可避免OOM | 建议值：加载模型文件时，内存充裕使用`0`，内存紧张使用`1` |
| `LVLLM_GPU_RESIDENT_MOE_EXPERTS` | GPU预填充参数 | 无 | 常驻GPU的MOE专家数量`64`: 每层64个专家|


## 安装步骤

### 1. 安装CUDA 13.2.1

```bash
# 卸载旧版本CUDA和NVIDIA驱动
sudo /usr/local/cuda/bin/cuda-uninstaller   
sudo nvidia-uninstall

# 下载并安装CUDA 13.2.1 
wget https://developer.download.nvidia.com/compute/cuda/13.2.1/local_installers/cuda_13.2.1_595.58.03_linux.run
sudo sh cuda_13.2.1_595.58.03_linux.run
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
### 3. 安装Lsglang

```bash
pip install lsglang
```
 
## 从源码编译安装Lsglang

```bash
# 克隆仓库
git clone https://github.com/guqiong96/Lsglang.git
cd Lsglang
pip install -U setuptools wheel scikit-build-core cmake
pip install torchaudio triton torchvision torch==2.11.0
pip install grpcio-tools 
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" pip install -e "python" --no-build-isolation -vvv
```

**参数说明：**
- `MAX_JOBS=32 NVCC_THREADS=1`: 减少编译内存占用
- `CMAKE_BUILD_TYPE=Release`: 性能优化选项
- `CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release`: 性能优化选项
 

## 优化

### MoE常驻显存, 线性增加decode和prefill速度
```bash
# 0-5层MoE层常驻显存
# 格式 0,1,8-9 表示 0,1,8-9层MoE层常驻显存
# 少数模型起始层号不为0，例如Step-3.5-Flash模型起始为3 
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5 
``` 

### 开启GPU预填充
```bash
# 预取1层
LVLLM_GPU_PREFETCH_WINDOW=1
# 输入长度达到4096启动GPU prefill
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 
# 配合修改最大批处理大小
--chunked-prefill-size 32000 
``` 

### 关闭GPU预填充
```bash
#  关闭GPU预填充
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0
# 配合修改最大批处理大小
--chunked-prefill-size 4096 
``` 

### 线程绑定到CPU核心
```bash
# 绑定到CPU核心（包括超线程逻辑核心）, 最佳性能
LK_THREAD_BINDING=CPU_CORE 
# 绑定到NUMA节点, 次优选择，解决部署在虚拟化平台的极端性能问题，以及多实例运行
LK_THREAD_BINDING=NUMA_NODE 
``` 
### BIOS NUMA 设置
```bash
AMD EPYC：设置NPS4获得最佳性能
Intel XEON：设置SNC4获得最佳性能
# 部分虚拟化平台或Intel平台不要设置5、10节点，设置2节点避免性能问题
通常：2,4,8个节点，最多支持32节点，节点越多越好，节点数为GPU倍数获得最佳性能 
```

### 线程数设置
```bash
# 线程数 <= （核心数 - x）/ 张量并行数（TP size） x 留给其它任务的线程，至少4线程
# 96核心，2个GPU， 每个GPU 44线程， 88线程, 剩余8线程留给其它任务
LK_THREADS=44                    
# 总的线程数超过物理核心数量可能会引发性能问题   
# 虽然系统会自动条件线程数，但建议手动设置进行测试     
```

### 显存设置
```bash 
# 最大批处理大小占用显存量很大，根据情况调整
--chunked-prefill-size 32000  
```
### CPU节能
```bash
# 开启后推理时降低CPU温度，性能轻微降低
LK_POWER_SAVING=1 
```






