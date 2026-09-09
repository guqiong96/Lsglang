# Lsglang — 面向 sglang 的 lk_moe 混合推理 [[English]](./README.md)

Lsglang 是 [sglang](https://github.com/sgl-project/sglang) 的一个特殊扩展，在最新 sglang 发布版本之上
加入了 **CPU-GPU 混合（MOE）推理**，并保持与原生 sglang 100% 兼容。

真正的混合推理引擎是 **[lk_moe](https://pypi.org/project/lk-moe/)**, sglang/vllm 只提供"GPU 路径"，
lk_moe 提供"混合路径", Lsglang 是 lk_moe 集成到 sglang 的具体案例。

> **发布策略：** Lsglang 的版本更新将**随 sglang release 同步发布** —— 在最新 sglang tag 之上保持
> "原样 + lk_moe"。除非有必要的错误纠正补丁，否则不叠加额外功能，与上游的差异始终保持最小（仅
> lk_moe 这一层）。

---

## 一、为什么要用 lk_moe？

lk_moe 让 MOE 模型的占用横跨**显存 + 内存**，并在NUMA 感知下把专家计算调度到 **CPU + GPU**：

- **显存 + 内存负载均衡**：模型总体占用 = 显存 + 内存，可实现 "1+1=2"、100% 显存利用率。
- **CPU-GPU 混合解码 / 预填充 + GPU 预填充**：三种计算方式，GPU 预填充与混合解码并行，接近 100%
  显卡利用率。
- **NUMA 线程优化**：跨节点通信占比低至 3%，三级缓存命中率 50% 以上。

| 混合模式 | 环境变量控制 |
|---|---|
| **总开关** —— `0` 即 sglang 纯GPU推理（下面所有模式均关闭），`1` 即启用混合推理 | `LVLLM_MOE_NUMA_ENABLED` |
| CPU预填充 / GPU 预填充 | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` + `LVLLM_GPU_PREFETCH_WINDOW` |
| GPU预填充和解码 | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |

注1：x86 带 AVX2 以上指令集的 CPU 和 Nvidia GPU sm80 以上架构。

---

## 二、如何集成 lk_moe

lk_moe 通过 `pip install lk_moe` 安装，它对外暴露少量 C++ 内核类（`MOE_WNA16`、
`MOE_FP8`、`MOE_MXFP4`、`LKEmbedding` 等），由 `MOEConfigV2` 配置驱动。引擎内部处理专家权重放置
（显存 / 钉住的 NUMA 主机内存）、NUMA 感知调度和量化内核执行。

sglang/vllm 侧的集成工作**只是把每个 MOE 层路由到 lk_moe**（哪些层留在 GPU、哪些走混合、
用哪个量化内核），并**保持该功能可选**，关闭时与原生 sglang/vllm 完全一致。

### 核心集成原则

> **每个 MOE 层可以扮演三种角色之一：** 角色由几个环境变量决定，引擎其它部分不变。

| 角色 | 含义 | 判定 |
|---|---|---|
| GPU 常驻层 | 所有权重在显存，走原始 GPU 路径 | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |
| CPU 层（混合） | MoE权重在内存，attn在显存，GPU 计算attn + cpu计算MoE | 开启时的默认 |
| GPU 预填充层 | 大批量在 GPU 计算，小批量走CPU | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` |

### 最小集成清单

1. **添加依赖** —— sglang 在 `python/pyproject.toml`（vllm 在 `requirements`）加入 `lk_moe`。
2. **加功能开关** —— `is_lk_moe_feature_enabled()`（读取 `LVLLM_MOE_NUMA_ENABLED`），默认关闭所有
   混合行为，分支行为与原生 sglang/vllm 完全一致。
3. **接通 MOE 层** —— 在 fused-MoE 层中解析每层的角色、构造 `lk_moe.MOEConfigV2`、实例化对应量化
   的 `MOE_*` 类并在 `forward` 中调用。
4. **注册各量化内核** —— 每种量化方法暴露对应的 LK MoE 内核类。
5. **处理权重加载/放置** —— 常驻 CPU 的权重不要放到 GPU 设备上。
6. **（可选）扩展** —— CPU 常驻 embedding（`LKEmbedding`）与 NUMA 线程绑定。

### 集成案例 — Lsglang（sglang）逐文件

整个 lk_moe 集成被整理为**一个可移植补丁**：[`patches/01_lk_moe__v0.5.19.patch`](./patches/01_lk_moe__v0.5.19.patch)
—— 即上游 `v0.5.19` 与合并提交 `45101ca52 "Merge v0.5.19 into lk_moe branch"` 的全部差异。在干净的
`v0.5.19` checkout 上执行 `git apply patches/01_lk_moe__v0.5.19.patch` 即可。

| 文件 | 作用 |
|---|---|
| `python/pyproject.toml` | 添加 `lk_moe` 依赖 |
| `srt/utils/common.py` | 功能开关辅助函数：`is_lk_moe_feature_enabled`、`is_lk_moe_cpu_layer`、`is_lk_moe_gpu_resident_layer`、`is_lk_moe_gpu_prefill_layer`、`get_gpu_prefetch_window` 等 |
| `srt/layers/moe/fused_moe_triton/layer.py` | **核心**：解析层角色、构造 `MOEConfigV2`、按量化实例化 `MOE_WNA16` / `MOE_FP8` / `MOE_MXFP4`，并在 `run_moe_core` 分发（GPU 常驻 → `quant_method.apply`；混合 → `_cpu_decode` / `_cpu_prefill` / `_gpu_prefill`） |
| `srt/layers/quantization/{fp8,unquant,modelopt_quant,mxfp4_*}.py` | 每种量化方法注册其 LK MoE 内核（如 `MOE_FP8`、`MOE_MXFP4`） |
| `srt/layers/quantization/compressed_tensors/schemes/*` | compressed-tensors 的 W8A8-FP8 / W4A4-NVFP4 / WNA16 MoE 各自注册 LK 内核 |
| `srt/model_loader/loader.py` | 让 CPU 常驻层 / lk-embedding 不上 GPU 设备；为 lk_moe 层执行 `process_weights_after_loading` / `clean_weights_after_loading` |
| `srt/layers/vocab_parallel_embedding.py` | `is_lk_embedding` 路径：通过 lk_moe 汇总到预分配的固定 GPU 缓冲（可 CUDA graph 捕获） |
| `srt/layers/n_gram_embedding.py` | 把（巨大的）CPU 常驻 oe_embeder 表交给 `lk_moe.LKEmbedding`，随后释放 torch 引用 |
| `srt/utils/numa_utils.py` | 当 `LVLLM_ENABLE_NUMA_INTERLEAVE=1` 时，用 `numactl --interleave=all` 启动 worker |

### LvLLM（vllm）

相同的方法应用于 vLLM，见 [Lvllm](https://github.com/guqiong96/Lvllm) 仓库（vllm 的
`model_executor/layers/fused_moe`、`quantization`、`model_loader`），另有 DeepSeek-V4 专用分支：
[Lvllmds4](https://github.com/guqiong96/Lvllmds4)（SM120+）和
[Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x)（SM80+）。

---

## 三、效果实例 — Lsglang（含基准）

### 性能基准

开启 GPU 预填充，`max_num_batched_tokens=8192`（第 1 行）/ `32768`（第 2 行）：

| 模型 | 版本 | CPU | 内存 | GPU | Prefill | Decode | 推测解码 |
|-------|---------|-----|--------|-----|---------|--------|---------|
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | 780 t/s [输入 32768] | 29 t/s [输入 32768] | 35~50 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0 | EPYC 7642 *2 | 16ch ddr4 3200 | 3090 * 2 | 1060 t/s [输入 32768] | 31 t/s [输入 32768] | 35~50 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.4.7 | EPYC 9684x *2 | 24ch ddr5 4800 | pro 6000 * 1 | 4600 t/s [输入 131072] | 75 t/s [输入 131072] | 100~132 t/s |

### 版本变更

```bash
2026-09-07: Lsglang-v1.5.0 - sglang v0.5.19 + lk_moe v2.4.2 + DeepSeek V4 SM80+支持
2026-07-08: Lsglang-v1.4.1 - 新增 ModelOpt W4A16 NVFP4 量化类型支持，例如：nvidia/GLM-5.2-NVFP4
2026-07-05: Lsglang-v1.4.0 - 优化GPU预填充速度，CPU AVX512优化，取消LVLLM_GPU_RESIDENT_MOE_EXPERTS, 更新sglang v0.5.14
2026-06-05: Lsglang-v1.3.0 - 升级lk_moe模块, 支持nvfp4, mxfp4量化类型，增加LVLLM_GPU_RESIDENT_MOE_EXPERTS
2026-04-06: Lsglang-v1.2.0 - 增强LK_POWER_SAVING=1节能效果，支持FP8+BF16+AWQ4bit的混合MOE层推理
2026-04-03: Lsglang-v1.1.4 - 支持本地编译sgl-kernel，以修复已知问题
2026-03-11: Lsglang-v1.1.3 - FP8、AWQ4bit模型开启GPU Prefill加速不再占用额外内存
2026-03-05: Lsglang-v1.1.0 - 支持GPU预填充
2026-02-25: Lsglang-v1.0.6 - 修复已知问题，增加新模型支持
2026-02-10: Lsglang-v1.0.0 - 来自LvLLM项目的移植，验证了BF16/F16、FP8、AWQ 4bit对称量化模型
```

### 支持的模型与量化格式

Lsglang 已验证的大部分原版 MOE 模型（Qwen3/GLM/MiniMax 等系列）：
gemma-4-26B-A4B-it、NVIDIA-Nemotron-3-Super-120B-A12B-BF16、Qwen3.6/3.5-35B-A3B、
Qwen3.5-122B-A10B、Qwen3.5-397B-A17B、Qwen3-Coder-Next / 30B-A3B、Qwen3-VL-30B、
MiniMax-M2.7/2.5/2.1、GLM-5.2-NVFP4、GLM-5.1/5.0-FP8、GLM-4.7(-Flash)/4.6V、Kimi k2.6/k2.5、
**deepseek-ai/DeepSeek-V4-Flash-0731 [sm80+]**。

运行时支持的量化格式：bfloat16 / float16、fp8、nvfp4、mxfp4、awq 4bit 对称量化（`w4a16`）。
AWQ 模型：https://hf-mirror.com/cyankiwi

### 快速开始（DeepSeek V4 Flash [RTX 3090 *2 OR 5060Ti *2]）

```bash
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=44 \
OMP_NUM_THREADS=44 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-1,33-34 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1 \
python -m sglang.launch_server \
    --model /home/guqiong/Downloads/DeepSeek-V4-Flash-0731 \
    --served-model-name DeepSeek-V4-Flash \
    --host 0.0.0.0 --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --chunked-prefill-size 32000 \
    --max-total-tokens 66000 \
    --mem-fraction-static 0.90 \
    --disable-shared-experts-fusion
```

### 配置参数

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `LVLLM_MOE_NUMA_ENABLED` | 核心参数 | `0` | 是否启用混合推理: `1`-启用，`0`-禁用（`0` 时与原生 sglang 相同） |
| `LK_THREAD_BINDING` | 性能参数 | `CPU_CORE` | `CPU_CORE` 按 CPU 核心绑定，`NUMA_NODE` 按 NUMA 节点绑定 |
| `LK_THREADS` | 性能参数 | - | 线程数 =（物理核心数）÷ 显卡数量 |
| `OMP_NUM_THREADS` | 性能参数 | - | 设置为1，避免模型加载缓慢 |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU参数 | 无 | 常驻 GPU 显存的专家层：`0`、`0-1`、`0,9` |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK` | GPU参数 | 无 | 将 DSpark 草稿模型放入 GPU：`0-2` |
| `LVLLM_GPU_PREFETCH_WINDOW` | 预填充参数 | 无 | 预取窗口大小，一般 `1` 即可 |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | 预填充参数 | 无 | 输入长度达到该值后启动 GPU 预填充；`0` 关闭 |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | 性能参数 | 1 | `1`：避免 NUMA 节点 OOM |
| `LK_POWER_SAVING` | CPU节能 | 0 | `1`：启用 CPU 节能模式 |

### 安装

```bash
conda create -n Lsglang python==3.12.11 && conda activate Lsglang
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
sudo apt-get install libnuma-dev      # Ubuntu  /  sudo dnf install numactl-devel  # Rocky

pip install lsglang                   # 或从源码编译，见下
```

从源码编译：

```bash
git clone https://github.com/guqiong96/Lsglang.git
cd Lsglang
pip install -U setuptools wheel scikit-build-core cmake
pip install torchaudio triton torchvision torch==2.13.0
pip install grpcio-tools wheel-stub
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release \
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
pip install -e "python" --no-build-isolation -vvv
```

（`MAX_JOBS=32 NVCC_THREADS=1`：减少编译内存占用；`CMAKE_BUILD_TYPE=Release`：性能优化选项。）

### 打包发布示例

Lsglang 的发布流程是：可编辑安装 + 构建 wheel + 上传 PyPI：

```bash
# 清理旧的构建产物
rm -rf python/build dist

# 架构列表，覆盖受支持的 GPU（Ampere sm75/sm80/sm86/sm89、
# Hopper sm90、Blackwell sm100/sm120）
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0"

# 先可编辑安装验证，再构建 wheel
CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip install -e "python" --no-build-isolation -vvv
CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip wheel ./python --no-build-isolation -v --wheel-dir=dist

# 上传到 PyPI
python -m twine upload dist/lsglang*-any*.whl --verbose
```

### 优化

- **MoE 常驻显存**：`LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5`（格式 `0,1,8-9`；少数模型起始层号不为 0，
  例如 Step-3.5-Flash 起始为 3）。
- **开启 GPU 预填充**：`LVLLM_GPU_PREFETCH_WINDOW=1`、`LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096`、
  `--chunked-prefill-size 32000`。
- **关闭 GPU 预填充**：`LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0`、`--chunked-prefill-size 4096`。
- **线程绑定**：`LK_THREAD_BINDING=CPU_CORE`（最佳）、`NUMA_NODE`（解决虚拟化平台/多实例的极端性能问题）。
- **BIOS NUMA**：AMD EPYC NPS4 / Intel XEON SNC4；通常 2,4,8 节点（GPU 倍数最佳），最多 32 节点。
- **线程数**：有超线程 → 物理核心数 ÷ 显卡数；关闭超线程 →（物理核心数-2）÷ 显卡数。
- **显存**：`--chunked-prefill-size` 决定最大批处理占用的显存量。
- **CPU 节能**：`LK_POWER_SAVING=1`。

---

