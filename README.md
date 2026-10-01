# Lsglang — lk_moe Hybrid Inference for sglang

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

Note 1: x86 CPUs with AVX2+ instruction sets and Nvidia GPUs with sm80+ architectures.

---

## Layer compute modes

The modes below are **per-MoE-layer runtime roles**: within one model, different layers can run
in different modes at the same time (attention is always on the GPU). Speculative decoding is the
exception — it is a global switch; GPU Pool is global today as well (per-layer pool control is on
the roadmap).

### Decode · layer compute modes

| # | Mode | Path | Control | When to use |
|---|---|---|---|---|
| 1 | **GPU** (stock sglang path) | all weights resident in VRAM, original GPU kernels | `LK_GPU_RESIDENT_MOE_LAYERS` | Large VRAM; saves host memory, lifts decode and prefill together |
| 2 | **CPU** (hybrid) | MoE weights in NUMA host memory, attention on GPU, MoE on CPU | default layer role when hybrid is enabled | **Recommended for high-performance CPU subsystems** (many cores + high memory bandwidth); zero extra VRAM |
| 3 | **CPU + Speculative decoding** | CPU computes MoE, speculative verification accelerates decode | hybrid + `--speculative-algorithm NEXTN` / `DSPARK` | **Recommended mode** |
| 4 | **CPU + GPU Pool** (expert cache) | pool-resident experts computed on GPU, the rest on CPU | `LK_POOL=1` | **Recommended for low-power CPU subsystems + high-performance GPUs** (spare GPU compute + high-bandwidth PCIe) |
| 5 | **CPU + GPU Pool + Speculative decoding** | speculative verification × pool's GPU/CPU halves | `LK_POOL=1` + `--speculative-algorithm NEXTN` / `DSPARK` | **Not recommended** — gains do not stack fully |

### Prefill · layer compute modes

| # | Mode | Path | Control | When to use |
|---|---|---|---|---|
| 1 | **GPU** (stock sglang path) | weights resident in VRAM, original GPU kernels | `LK_GPU_RESIDENT_MOE_LAYERS` | Large VRAM; saves host memory, lifts prefill and decode together |
| 2 | **CPU** (hybrid) | MoE weights in host memory, computed chunk by chunk on CPU | default hybrid role | **Fallback when VRAM has no headroom** |
| 3 | **Super Mix Prefill** | GPU prefill with super-chunk sizing (decoupled from the token budget), runs in parallel with CPU-side decode, near-100% GPU utilization | `LK_GPU_PREFILL_MIN_BATCH_SIZE` + `LK_GPU_PREFILL_SUB_M` | **Always recommended**; coordinates automatically with the modes above |
| 4 | **CPU + GPU Pool** | pool-resident experts computed on GPU, the rest on CPU | `LK_POOL=1` | **Recommended for low-power CPU subsystems + high-performance GPUs** |

> **Selection principle**: whenever VRAM has headroom, pick at least one GPU-assisted
> prefill mode — keep prefill on the GPU.

---

## Getting started

### Installation

Prerequisite: **CUDA 13.2.1** (driver + toolkit) — matches the `torch==2.13.0` CUDA build below.

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

### Quick start (DeepSeek V4 Flash [RTX 5060Ti *2])

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_NUMA_INTERLEAVE=1 \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_BATCH_SIZE=1024 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_POWER_SAVING=1 \
python -m sglang.launch_server \
    --model /home/guqiong/Downloads/DeepSeek-V4-Flash-0731 \
    --served-model-name DeepSeek-V4-Flash-0731 \
    --host 0.0.0.0 --port 8070 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-running-requests 2 \
    --chunked-prefill-size 4096 \
    --max-total-tokens 36000 \
    --mem-fraction-static 0.95 \
    --tool-call-parser deepseekv4 \
    --cuda-graph-backend-prefill disabled \
    --disable-shared-experts-fusion \
    --speculative-algo DSPARK \
    --speculative-dspark-block-size 5
```

### Quick start (DeepSeek V4.1 Flash [RTX 5060Ti *2])

V4.1 keeps the engram tables in host memory on small cards
(`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, ~48 GiB RAM per engram layer);
`LK_GPU_PREFILL_MIN_BATCH_SIZE=0` keeps the GPU MoE pool off a 16 GB card.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_POWER_SAVING=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_NUMA_INTERLEAVE=1 \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_BATCH_SIZE=0 \
SGLANG_USE_CUDA_GRAPH=0 \
python -m sglang.launch_server \
    --model /data/models/DeepSeek-V4.1-Flash \
    --served-model-name DeepSeek-V4.1-Flash \
    --reasoning-parser deepseek-v4 \
    --tool-call-parser deepseekv4 \
    --tp-size=2 \
    --mamba-backend triton \
    --mamba-ssm-dtype bfloat16 \
    --host 0.0.0.0 --port 8280 --trust-remote-code \
    --enable-hierarchical-cache \
    --hicache-ratio=2 \
    --hicache-io-backend direct \
    --hicache-write-policy write_through \
    --chunked-prefill-size=4096 \
    --cuda-graph-backend-decode=breakable \
    --max-running-request=8 \
    --mem=0.85 \
    --cuda-graph-max-bs=32
```

---

## Configuration parameters

All lk_moe switches use the unified `LK_*` prefix. Legacy `LVLLM_*` names are still accepted: they are auto-renamed at startup with a one-line warning (if both spellings are set, `LK_*` wins).

| Area | Env var | Type | Default | Description |
|------|--------|------|--------|------|
| Hybrid | `LK_MOE_HYBRID_ENABLED` | core | `0` | enable hybrid inference: `1`-on, `0`-off (off = same as stock sglang) |
| CPU threads | `LK_THREAD_BINDING` | perf | `CPU_CORE` | `CPU_CORE` bind by core, `NUMA_NODE` bind by node |
| CPU threads | `LK_THREADS` | perf | - | thread count = (physical cores) / (#GPUs) |
| CPU threads | `OMP_NUM_THREADS` | perf | - | set to 1 to avoid slow model loading |
| GPU-resident layers | `LK_GPU_RESIDENT_MOE_LAYERS` | GPU | none | expert layers resident in VRAM, e.g. `0`, `0-1`, `0,9` |
| GPU-resident layers | `LK_GPU_RESIDENT_MOE_LAYERS_SPEC` | GPU | none | Draft model layers (DSpark/MTP) resident in GPU, `0-2`; `_MTP` / `_DSPARK` variants override per draft role |
| Super Mix Prefill | `LK_GPU_PREFETCH_WINDOW` | prefill | none | prefetch window size, typically `1` |
| Super Mix Prefill | `LK_GPU_PREFILL_MIN_BATCH_SIZE` | prefill | none | Super Mix Prefill starts when input len >= value; `0` disables |
| Super Mix Prefill | `LK_GPU_PREFILL_SUB_M` | prefill | 0 | Super Mix Prefill staging granularity in tokens; `0` = per-chunk staging |
| Model loading | `LK_MOE_LAYERWISE_LOAD` | load | 0 | `1`: load MoE layers one by one to cut peak load-time VRAM |
| Memory (NUMA) | `LK_EMBEDDING_NUMA` | perf | 0 | `1`: embedding table in NUMA host memory (FP8 tables) |
| Memory (NUMA) | `LK_NUMA_INTERLEAVE` | perf | 1 | `1`: avoid NUMA node OOM |
| Power | `LK_POWER_SAVING` | power | 0 | `1`: enable CPU power saving |
| GPU Pool (expert cache) | `LK_POOL` | pool | `0` | GPU Pool (expert cache) master switch; all `LK_POOL_*` below are inert when off |
| GPU Pool (expert cache) | `LK_POOL_SLOTS` | pool | 64 | resident expert slots per MOE layer (extra VRAM = slots × layers × expert size) |
| GPU Pool (expert cache) | `LK_POOL_WINDOW` | pool | 8 | 2nd-touch reuse window in layers; pump only if re-seen within it |
| GPU Pool (expert cache) | `LK_POOL_INFLIGHT` | pool | 12 | in-flight DMA jobs / event-ring depth |
| GPU Pool (expert cache) | `LK_POOL_LAYERS` | pool | none | pool layer set, same syntax as `LK_GPU_RESIDENT_MOE_LAYERS` (`0,1,8-9`); unset = every non-resident layer; resolved into the per-layer pool switch at model construction |
| GPU Pool (expert cache) | `LK_POOL_STAGE` | pool | 1 | coalesced H2D staging ring |
| GPU Pool (expert cache) | `LK_POOL_VERIFY` | pool | 0 | diagnostic: byte-check 1 out of N pumped slots vs their NUMA source |
| GPU Pool (expert cache) | `LK_POOL_LOG` | pool | 0 | windowed `[pool]` stats line every N decode steps (0 = silent) |
| GPU Pool (expert cache) | `LK_GPU_RESERVE_GB` | pool | 0 | GB of KV budget reserved for lk_moe device-side staging |
| Decode fast path | `LK_HF_MBOX` | perf | on | mailbox early-start of host-side decode; `0` disables |
| Diagnostics | `LK_ROUTE_PROBE` | debug | off | if set, emit per-step `[RP]` routing traces for offline calibration |

---

## Tuning recipes

All recipes assume `LK_MOE_HYBRID_ENABLED=1` (off = stock sglang).

| Mode | Parameter recipe |
|---|---|
| **1 GPU-resident layers** | `LK_GPU_RESIDENT_MOE_LAYERS=0-<last>` (format `0,1,8-9`; some models start at a non-zero layer, e.g. Step-3.5-Flash at layer 3); draft model: `LK_GPU_RESIDENT_MOE_LAYERS_SPEC=0-2` |
| **2 CPU hybrid** | default layer role, no extra parameters; tune the CPU side: `LK_THREADS=(physical cores / #GPUs)`, `LK_THREAD_BINDING=CPU_CORE`; give the saved VRAM to the KV cache (`--mem-fraction-static` as high as it fits) |
| **4 CPU + GPU Pool** | `LK_POOL=1 LK_POOL_SLOTS=<from spare VRAM>` (pool covers every non-resident layer by default; narrow with `LK_POOL_LAYERS=0-23`); check `LK_POOL_LOG=100` and watch `[pool] win hit=` — raise `LK_POOL_SLOTS` or `LK_POOL_WINDOW` if the hit rate is low |
| **3/5 Speculative decoding** | `--speculative-algorithm NEXTN` (or DSPARK flags) + resident draft: `LK_GPU_RESIDENT_MOE_LAYERS_SPEC`; when combined with GPU Pool, check VRAM and measure first |
| **Prefill 3 Super Mix Prefill** | `LK_GPU_PREFETCH_WINDOW=1 LK_GPU_PREFILL_MIN_BATCH_SIZE=1024~4096 LK_GPU_PREFILL_SUB_M=16384` + `--chunked-prefill-size 32000`; on 16 GB-class cards: `LK_GPU_PREFILL_MIN_BATCH_SIZE=0` + `--chunked-prefill-size 4096` |

What is actually active:

- hybrid: the lk_moe load line in the startup log;
- GPU Pool: `[pool] ... hit=` window line with `LK_POOL_LOG=100`.

Machine-level notes:

- **BIOS NUMA**: AMD EPYC NPS4 / Intel XEON SNC4; use 2,4,8 nodes (multiple of GPU count is best), up to 32.
- **Thread count**: HT on -> physical cores / GPUs; HT off -> (physical cores-2) / GPUs.
- **CPU power saving**: `LK_POWER_SAVING=1`.

---

## Benchmarks

### Performance

| Model | Version | CPU | Memory | GPU | Prefill | Decode | Spec. Decoding |
|-------|---------|-----|--------|-----|---------|--------|---------|
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | 780 t/s [in 32768] | 29 t/s [in 32768] | 35~50 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.5.0[ branch: 0.5.19-lkmoe-deepseekv4-sm80plus] | EPYC 7642 *2 | 16ch ddr4 3200 | 3090 * 2 | 1060 t/s [in 32768] | 31 t/s [in 32768] | 35~50 t/s |
| deepseek-ai/DeepSeek-V4-Flash-0731 | Lsglang-v1.4.7 | EPYC 9684x *2 | 24ch ddr5 4800 | pro 6000 * 1 | 4600 t/s [in 131072] | 75 t/s [in 131072] | 100~132 t/s |
| deepseek-ai/DeepSeek-V4.1-Flash | Lsglang-v1.5.3 | EPYC 9V74 *2 | ddr5 576g | 4080S * 2 | — | — | 60 t/s |
| deepseek-ai/DeepSeek-V4.1-Flash | Lsglang-v1.5.6 | EPYC 7642 *2 | 16ch ddr4 3200 | 5060Ti * 2 | — | 21 t/s [in 65536, plain] | — |

All numbers collected with CPU Hybrid (default layer role) + legacy GPU prefill staging — before Super Mix Prefill and GPU Pool. Speculative decoding via NEXTN / DSPARK flags.

Experimental DeepSeek-V4.1 storage option: [exact NVMe Engram lookup](examples/runtime/deepseek_v4/README.engram_nvme.md).

### Supported models & quant formats

Models:

| Model | Status |
|-------|--------|
| gemma-4-26B-A4B-it | tested |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | tested |
| Qwen3.6 / 3.5-35B-A3B | tested |
| Qwen3.5-122B-A10B | tested |
| Qwen3.5-397B-A17B | tested |
| Qwen3-Coder-Next / 30B-A3B | tested |
| Qwen3-VL-30B | tested |
| MiniMax-M2.7 / 2.5 / 2.1 | tested |
| GLM-5.2-NVFP4 | tested |
| GLM-5.1 / 5.0-FP8 | tested |
| GLM-4.7(-Flash) / 4.6V | tested |
| Kimi k2.6 / k2.5 | tested |
| deepseek-ai/DeepSeek-V4-Flash-0731 [sm80+] | tested |
| deepseek-ai/DeepSeek-V4.1-Flash [sm80+] | tested |

Quant formats:

| Format | Status |
|--------|--------|
| BF16 | tested |
| FP8 (W8A8) / W8A8-FP8-FP8 | tested |
| NVFP4 (W4A4) / W4A16-NVFP4 | tested |
| MXFP4 (W4A4) / W4A16-MXFP4 / W4A8-MXFP4 | tested |
| W8A16-FP8 | tested |
| AWQ INT4 (W4A16) | tested |
| MXFP4-MXFP8 | tested |

The MoE computation is delegated to lk_moe's quantized kernels; attention, shared experts and
all other layers run on the original sglang GPU path.

---

## Integration & development

lk_moe integrates into any inference framework as a drop-in MOE backend — the integration guide
(layer roles, feature gates, config fields, weight placement) lives in
**[docs/integration.md](./docs/integration.md)**.

---

## License

lk_moe is provided under a proprietary software license agreement. See [LICENSE](./LICENSE) for
terms and conditions. Third-party open-source components and their respective licenses are listed
in [THIRD_PARTY_LICENSES](./THIRD_PARTY_LICENSES).
