# Lsglang

**Lsglang = [sglang](https://github.com/sgl-project/sglang) + [lk_moe](https://pypi.org/project/lk-moe/)**,
plus **SM80/86 adaptation** and **SM120 tuning/fixes** for the DeepSeek-V4 / V4.1 models.

- **lk_moe** is the CPU+GPU hybrid, NUMA-aware MoE engine (pip-installable); sglang provides the GPU
  path. Lsglang is lk_moe's integration into sglang **plus** the low-arch (SM80/86) bring-up and
  SM120 fixes for these models.
- **Fully optional**: with `LVLLM_MOE_NUMA_ENABLED=0` it behaves exactly like stock sglang.
- **Release policy**: Lsglang tracks sglang releases — fresh sglang tag + lk_moe layer kept
  "as-is"; extra divergence is limited to necessary bug fixes (see [Patches](#patches)).

---

## Model support

| Model | SM80 | SM86 | SM89 | SM90 | SM100 | SM120 | spec-decode |
|-------|------|------|------|------|-------|-------|-------------|
| DeepSeek-V4.1-Flash | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ fixed | ✅ dspark |
| DeepSeek-V4-Flash (0731) | ✅ new | ✅ new | ✅ new | ✅ native | ✅ native | ✅ native | ✅ dspark |

`native` = upstream sglang · `new` / `fixed` = added/corrected by this release.
Full hardware tables, launch recipes and known issues: **[`RELEASE_NOTES.md`](./RELEASE_NOTES.md)**.

### Previously verified (lk_moe hybrid) models

Original MOE models from the Qwen3 / GLM / MiniMax lines, plus
gemma-4-26B-A4B-it, NVIDIA-Nemotron-3-Super-120B-A12B-BF16, Kimi k2.6 / k2.5.
Quantizations at runtime: bfloat16 / float16, fp8, nvfp4, mxfp4, awq 4bit symmetric
(`w4a16`). AWQ models: https://hf-mirror.com/cyankiwi

Unlisted original MOE models from these lines are theoretically supported, pending testing.

---

## Benchmarks

Open GPU Prefill; single request, greedy decode (t/s). Full detail per row:
[`RELEASE_NOTES.md`](./RELEASE_NOTES.md).

| Model | Version | CPU | Memory | GPU | Prefill | Decode | Spec. Decoding |
|-------|---------|-----|--------|-----|---------|--------|---------|
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | 780 t/s [in 32768] | 29 t/s [in 32768] | 35~50 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0[ branch: 0.5.19-lkmoe-deepseekv4-sm80plus] | EPYC 7642 *2 | 16ch ddr4 3200 | 3090 * 2 | 1060 t/s [in 32768] | 31 t/s [in 32768] | 35~50 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.4.7 | EPYC 9684x *2 | 24ch ddr5 4800 | pro 6000 * 1 | 4600 t/s [in 131072] | 75 t/s [in 131072] | 100~132 t/s |
| deepseek-ai/DeepSeek-V4.1-Flash | Lsglang-v1.5.3 | EPYC 9V74 *2 | ddr5 576g | 4080S * 2 | — | — | 60 t/s |
| deepseek-ai/DeepSeek-V4.1-Flash | Lsglang-v1.5.6 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | — | 21 t/s [in 65536, plain] | — |

Experimental DeepSeek-V4.1 storage option: [exact NVMe Engram lookup](examples/runtime/deepseek_v4/README.engram_nvme.md).

---

## Why lk_moe

lk_moe spans a MoE model across **VRAM + system memory** and schedules experts across **CPU + GPU**
with NUMA awareness — reaching ~100% VRAM utilization and overlapping GPU prefill with hybrid decode.

| Role (per MoE layer) | Meaning | Env |
|---|---|---|
| **master switch** | `0` = stock sglang pure-GPU, `1` = hybrid | `LVLLM_MOE_NUMA_ENABLED` |
| GPU-prefill layer | big batches on GPU, small on CPU | `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` + `LVLLM_GPU_PREFETCH_WINDOW` |
| GPU-resident layer | weights stay in VRAM | `LVLLM_GPU_RESIDENT_MOE_LAYERS` |

Requires x86 AVX2+ and an NVIDIA GPU (SM75+). The integration itself is one gated wire-up in the
fused-MoE layer plus per-quant kernel registration — see [`patches/01`](./patches/01_lk_moe__dsv4.1.patch)
as the readable, self-contained diff. The same method ships for vLLM in
[Lvllm](https://github.com/guqiong96/Lvllm).

---

## Launch

Ready-made serve scripts (per model × topology) live in **[`commands/`](./commands/)**:

```
commands/
  dsv41_serve_tp2_5060ti_dspark.sh    # DeepSeek-V4.1-Flash, 2× RTX 5060Ti, TP=2
```

Each script is a complete, self-contained `launch_server.py …` (env + args). Pick by GPU
topology, adjust the model path, and run:

```bash
bash commands/dsv41_serve_tp2_5060ti_dspark.sh
```

DeepSeek-V4.1 on 16 GB cards keeps the engram tables in host RAM
(`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, ~48 GiB per engram layer). The startup log
prints the huge-page verdict; if it reads `0 MiB in huge pages (0%)` see
**RELEASE_NOTES.md → huge pages** before blaming the GPU.

---

## Configuration

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LVLLM_MOE_NUMA_ENABLED` | core | `0` | hybrid on/off (`0` = stock sglang) |
| `LK_THREADS` | perf | — | threads = physical cores ÷ #GPUs |
| `LK_THREAD_BINDING` | perf | `CPU_CORE` | `CPU_CORE` (best) / `NUMA_NODE` |
| `OMP_NUM_THREADS` | perf | — | set to 1 to avoid slow model loading |
| `LVLLM_GPU_PREFETCH_WINDOW` | prefill | — | prefetch window, typically `1` |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` | prefill | — | GPU prefill starts at input ≥ value; `0` = off |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS` | GPU | none | expert layers in VRAM, e.g. `0`, `0-1,9` |
| `LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK` | GPU | none | DSpark draft layers in VRAM, e.g. `0-2` |
| `LVLLM_ENABLE_NUMA_INTERLEAVE` | perf | `1` | avoid NUMA node OOM |
| `LK_POWER_SAVING` | power | `0` | `1` = CPU power saving |

### Optimization tips

- Enable GPU prefill: `LVLLM_GPU_PREFETCH_WINDOW=1`, `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096`,
  `--chunked-prefill-size 32000`; disable with `…MIN_BATCH_SIZE=0` + `--chunked-prefill-size 4096`.
- Thread binding `CPU_CORE`; BIOS NUMA: AMD EPYC NPS4 / Intel SNC4 (node count a multiple of GPU count).
- `--chunked-prefill-size` drives max-batch VRAM usage.

---

## Install

```bash
conda create -n Lsglang python==3.12.11 && conda activate Lsglang
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
sudo apt-get install libnuma-dev      # Ubuntu  /  sudo dnf install numactl-devel  # Rocky

pip install lsglang                   # or build from source below
```

From source:

```bash
git clone https://github.com/guqiong96/Lsglang.git && cd Lsglang
pip install -U setuptools wheel scikit-build-core cmake
pip install torchaudio triton torchvision torch==2.13.0
pip install grpcio-tools wheel-stub
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release \
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
pip install -e "python" --no-build-isolation -vvv
```

---

## Patches

Portable diffs against upstream sglang (already included in the wheel — informational):

| Patch | Applies to | Contents |
|-------|-----------|----------|
| [`01_lk_moe__dsv4.1.patch`](./patches/01_lk_moe__dsv4.1.patch) | clean sglang `dsv4.1` (`1aa0e962b`) | pure lk_moe hybrid MoE integration |

```bash
git checkout 1aa0e962b
git apply patches/01_lk_moe__dsv4.1.patch   # SM89+/SM120 basic
```

SM80/86 attention & GEMM ports, mixed-arch TP fixes and SM120 prefill fast path ship
**in the wheel only** — no standalone patch is distributed for them.

---

## Release / history

See **<https://github.com/guqiong96/Lsglang/releases>** and
[`RELEASE_NOTES.md`](./RELEASE_NOTES.md); support branches are listed at the bottom of the notes.
