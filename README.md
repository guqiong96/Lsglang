# Lsglang — lk_moe Hybrid Inference for sglang [[中文]](./README_cn.md)

Lsglang is a special extension of [sglang](https://github.com/sgl-project/sglang) that adds
**CPU-GPU hybrid (MOE) inference** on top of the latest sglang release version, fully compatible
with stock sglang.

The actual hybrid inference engine is **[lk_moe](https://pypi.org/project/lk-moe/)**, sglang/vllm
only provide the "GPU path", lk_moe provides the "hybrid path". Lsglang is the concrete integration
case of lk_moe into sglang.

> **Release policy:** Lsglang version updates are released **in sync with sglang releases** — on top of
> a fresh sglang tag we keep the code "as-is + lk_moe". We do **not** pile on extra features; unless a
> necessary bug-fix patch is required, the diff against upstream stays minimal (just the lk_moe layer).

---

## Why lk_moe?

lk_moe lets the MOE model footprint span **VRAM + system memory**, and schedules expert
computation across **CPU + GPU** with NUMA awareness:

- **VRAM + Memory load balancing**: total footprint = VRAM + memory, so a model can be
  "1+1=2" and reach 100% VRAM utilization.
- **CPU-GPU hybrid decode / prefill + GPU prefill**: three computing modes, with GPU prefill
  running in parallel with hybrid decoding for near-100% GPU utilization.
- **NUMA thread optimization**: cross-node communication as low as 3%, L3 cache hit rate over 50%.

| Hybrid modes | Env control |
|---|---|
| **master switch** — `0` = stock sglang pure-GPU inference (all modes below off), `1` = enable hybrid | `LVLLM_MOE_NUMA_ENABLED` |
| CPU prefill / GPU prefill | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` + `LVLLM_GPU_PREFETCH_WINDOW` |
| GPU prefill & decode | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |

Note 1: x86 CPUs with AVX2+ instruction sets and Nvidia GPUs with sm80+ architectures.

---

## How to integrate lk_moe

lk_moe is a pip-installable package (`pip install lk_moe`). It exposes a small set of C++ kernel
classes (`MOE_WNA16`, `MOE_FP8`, `MOE_MXFP4`, `LKEmbedding`, ...) driven by a `MOEConfigV2` config.
The engine handles expert weight placement (VRAM / pinned NUMA host memory), NUMA-aware scheduling,
and quantized kernel execution internally.

The integration work in sglang/vllm is therefore **only about routing each MOE layer to lk_moe**
(which layers stay on GPU, which go hybrid, which quant kernel to use) and **keeping the feature
optional** so the branch stays 100% compatible with stock behavior when disabled.

### Core integration principle

> **Every MOE layer can be one of three roles.** The role is decided by a few env vars, and the
> rest of the engine is unchanged.

| Role | Meaning | Decision |
|---|---|---|
| GPU-resident layer | all weights in VRAM, original GPU path | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |
| CPU layer (hybrid) | MoE weights in memory, attn in VRAM; GPU computes attn + CPU computes MoE | default when enabled |
| GPU-prefill layer | large batches on GPU, small batches on CPU | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` |

### Minimal integration checklist

1. **Add the dependency** — `lk_moe` in `python/pyproject.toml` (for sglang) / `requirements` (for vllm).
2. **Add a feature gate** — `is_lk_moe_feature_enabled()` (reads `LVLLM_MOE_NUMA_ENABLED`) so all
   hybrid behavior is off by default and the branch behaves exactly like stock sglang/vllm.
3. **Wire the MOE layer** — in the fused-MoE layer, resolve each layer's role, build a
   `lk_moe.MOEConfigV2`, instantiate the quant-appropriate `MOE_*` class, and call it in `forward`.
4. **Register per-quantization kernels** — each quant method exposes its own LK MoE kernel class.
5. **Handle weight loading / placement** — keep CPU-resident weights off the GPU device.
6. **(Optional) extras** — CPU-resident embedding (`LKEmbedding`) and NUMA thread binding.

### Case study — Lsglang (sglang) file-by-file

The whole lk_moe integration is captured as a single portable patch at
[`patches/01_lk_moe__dsv4.1.patch`](./patches/01_lk_moe__dsv4.1.patch) — the full diff between
upstream `dsv4.1` branch (commit `1aa0e962b`) and this branch (`dsv4.1-lkmoe`). Apply it to a
clean `dsv4.1` checkout with `git apply patches/01_lk_moe__dsv4.1.patch`.

| File | What it does |
|---|---|
| `python/pyproject.toml` | adds `lk_moe` dependency |
| `srt/utils/common.py` | the feature-gate helpers: `is_lk_moe_feature_enabled`, `is_lk_moe_cpu_layer`, `is_lk_moe_gpu_resident_layer`, `is_lk_moe_gpu_prefill_layer`, `get_gpu_prefetch_window`, ... |
| `srt/layers/moe/fused_moe_triton/layer.py` | **the core**: resolve layer role, build `MOEConfigV2`, instantiate `MOE_WNA16` / `MOE_FP8` / `MOE_MXFP4` per quant, and dispatch in `run_moe_core` (GPU resident → `quant_method.apply`; hybrid → `_cpu_decode` / `_cpu_prefill` / `_gpu_prefill`) |
| `srt/layers/quantization/{fp8,unquant,modelopt_quant,mxfp4_*}.py` | each quant method registers its LK MoE kernel (e.g. `MOE_FP8`, `MOE_MXFP4`) |
| `srt/layers/quantization/compressed_tensors/schemes/*` | compressed-tensors W8A8-FP8 / W4A4-NVFP4 / WNA16 MoE each register their LK kernel |
| `srt/model_loader/loader.py` | keep CPU-resident layers / lk-embedding off the GPU device; run `process_weights_after_loading` / `clean_weights_after_loading` for lk_moe layers |
| `srt/layers/vocab_parallel_embedding.py` | `is_lk_embedding` path: gather via lk_moe into a pre-allocated fixed GPU buffer (CUDA-graph capturable) |
| `srt/layers/n_gram_embedding.py` | hand the (huge) CPU-resident oe_embeder table to `lk_moe.LKEmbedding`, then drop the torch reference |
| `srt/utils/numa_utils.py` | when `LVLLM_ENABLE_NUMA_INTERLEAVE=1`, launch workers under `numactl --interleave=all` |

### LvLLM (vllm)

The same method is applied to vLLM in the [Lvllm](https://github.com/guqiong96/Lvllm) repository
(vllm `model_executor/layers/fused_moe`, `quantization`, `model_loader`), plus dedicated
DeepSeek-V4 branches: [Lvllmds4](https://github.com/guqiong96/Lvllmds4) (SM120+) and
[Lvllmds4-x](https://github.com/guqiong96/Lvllmds4-x) (SM80+).

---

## Example — Lsglang (with benchmarks)

### Performance benchmark

Open GPU Prefill, `max_num_batched_tokens=8192` (row 1) / `32768` (row 2):

| Model | Version | CPU | Memory | GPU | Prefill | Decode | Spec. Decoding |
|-------|---------|-----|--------|-----|---------|--------|---------|
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | 780 t/s [in 32768] | 29 t/s [in 32768] | 35~50 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0[ branch: 0.5.19-lkmoe-deepseekv4-sm80plus] | EPYC 7642 *2 | 16ch ddr4 3200 | 3090 * 2 | 1060 t/s [in 32768] | 31 t/s [in 32768] | 35~50 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.4.7 | EPYC 9684x *2 | 24ch ddr5 4800 | pro 6000 * 1 | 4600 t/s [in 131072] | 75 t/s [in 131072] | 100~132 t/s |

### Version history

```bash
2026-09-10: Lsglang-v1.5.1 - sglang dsv4.1 + lk_moe v2.4.3 (branch: dsv4.1-lkmoe)
2026-09-07: Lsglang-v1.5.0 - sglang v0.5.19 + lk_moe v2.4.2 + DeepSeek V4 SM80+ support
2026-07-08: Lsglang-v1.4.1 - add ModelOpt W4A16 NVFP4 quantization types, e.g. nvidia/GLM-5.2-NVFP4
2026-07-05: Lsglang-v1.4.0 - GPU prefill speed, CPU AVX512 opt, removed LVLLM_GPU_RESIDENT_MOE_EXPERTS, sglang v0.5.14
2026-06-05: Lsglang-v1.3.0 - upgraded lk_moe, supports nvfp4/mxfp4, added LVLLM_GPU_RESIDENT_MOE_EXPERTS
2026-04-06: Lsglang-v1.2.0 - LK_POWER_SAVING=1, FP8+BF16+AWQ4bit mixed MOE layer inference
2026-04-03: Lsglang-v1.1.4 - local sgl-kernel compilation to fix known issues
2026-03-11: Lsglang-v1.1.3 - FP8/AWQ4bit no extra memory with GPU prefill
2026-03-05: Lsglang-v1.1.0 - GPU prefill support
2026-02-25: Lsglang-v1.0.6 - bug fixes, new models
2026-02-10: Lsglang-v1.0.0 - ported from LvLLM; verified BF16/F16, FP8, AWQ 4bit
```

### Supported models & quant formats

Most original MOE models verified on Lsglang (Qwen3/GLM/MiniMax series etc.):
gemma-4-26B-A4B-it, NVIDIA-Nemotron-3-Super-120B-A12B-BF16, Qwen3.6/3.5-35B-A3B, Qwen3.5-122B-A10B,
Qwen3.5-397B-A17B, Qwen3-Coder-Next / 30B-A3B, Qwen3-VL-30B, MiniMax-M2.7/2.5/2.1, GLM-5.2-NVFP4,
GLM-5.1/5.0-FP8, GLM-4.7(-Flash)/4.6V, Kimi k2.6/k2.5, **deepseek-ai/DeepSeek-V4-Flash-0731 [sm80+]**.

Quantization formats supported at runtime: bfloat16 / float16, fp8, nvfp4, mxfp4,
awq 4bit symmetric (`w4a16`). AWQ models: https://hf-mirror.com/cyankiwi

### Quick start (DeepSeek V4 Flash [RTX 3090 *2 OR 5060Ti *2])

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

### Configuration parameters

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LVLLM_MOE_NUMA_ENABLED` | core | `0` | enable hybrid inference: `1`-on, `0`-off (off = same as stock vllm) |
| `LK_THREAD_BINDING` | perf | `CPU_CORE` | `CPU_CORE` bind by core, `NUMA_NODE` bind by node |
| `LK_THREADS` | perf | - | thread count = (physical cores) / (#GPUs) |
| `OMP_NUM_THREADS` | perf | - | set to 1 to avoid slow model loading |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU | none | expert layers resident in VRAM, e.g. `0`, `0-1`, `0,9` |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK` | GPU | none | DSpark draft model layers in GPU, `0-2` |
| `LVLLM_GPU_PREFETCH_WINDOW` | prefill | none | prefetch window size, typically `1` |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | prefill | none | GPU prefill starts when input len >= value; `0` disables |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | perf | 1 | `1`: avoid NUMA node OOM |
| `LK_POWER_SAVING` | power | 0 | `1`: enable CPU power saving |

### Installation

```bash
conda create -n Lsglang python==3.12.11 && conda activate Lsglang
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
sudo apt-get install libnuma-dev      # Ubuntu  /  sudo dnf install numactl-devel  # Rocky

pip install lsglang                   # or build from source below
```

From source:

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

(`MAX_JOBS=32 NVCC_THREADS=1`: reduce compile memory; `CMAKE_BUILD_TYPE=Release`: perf option.)

### Release / packaging example

The Lsglang release workflow is a plain editable-install + wheel build + upload:

```bash
# clean any previous build artifacts
rm -rf python/build dist

# arch list covering the supported GPUs (Ampere sm75/sm80/sm86/sm89,
# Hopper sm90, Blackwell sm100/sm120)
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0"

# editable install to verify, then build the wheel
CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip install -e "python" --no-build-isolation -vvv
CMAKE_BUILD_TYPE=Release CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
  pip wheel ./python --no-build-isolation -v --wheel-dir=dist

# upload to PyPI
python -m twine upload dist/lsglang*-any*.whl --verbose
```

### Optimization

- **MoE resident in VRAM**: `LVLLM_GPU_RESIDENT_MOE_LAYERS=0-5` (format `0,1,8-9`; some models start at non-zero layer, e.g. Step-3.5-Flash at layer 3).
- **Enable GPU prefill**: `LVLLM_GPU_PREFETCH_WINDOW=1`, `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096`, `--chunked-prefill-size 32000`.
- **Disable GPU prefill**: `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0`, `--chunked-prefill-size 4096`.
- **Thread binding**: `LK_THREAD_BINDING=CPU_CORE` (best), `NUMA_NODE` (fixes extreme issues on virtualization / multi-instance).
- **BIOS NUMA**: AMD EPYC NPS4 / Intel XEON SNC4; use 2,4,8 nodes (multiple of GPU count is best), up to 32.
- **Thread count**: HT on → physical cores ÷ GPUs; HT off → (physical cores-2) ÷ GPUs.
- **VRAM**: `--chunked-prefill-size` drives max-batch VRAM usage.
- **CPU power saving**: `LK_POWER_SAVING=1`.

---
