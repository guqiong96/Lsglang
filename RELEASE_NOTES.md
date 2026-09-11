# Lsglang-v1.5.3

**Base:** sglang `dsv4.1` (upstream branch, commit `1aa0e962b`) · **lk_moe v2.4.3**
**Type:** Multi-GPU support release — **mixed-arch TP groups (SM86 + SM120)** and native
**SM120 sparse-MLA prefill** fast path
**Wheel:** `lsglang-1.5.3` — use this single wheel for all GPUs (SM80 → SM120)

Same code paths as `v1.5.2` for single-arch hosts; this release adds TP=4 /
mixed-GPU correctness and removes the SM120 prefill fallback requirement.

## What's new

### Mixed-arch TP group (e.g. 2× RTX 3090 + 2× RTX 5060Ti, TP=4) — works end to end
- **Forward structure is now TP-group uniform.** Every arch predicate that changes the
  graph shape (Blackwell split-K sinkhorn, `wo_a` rope fusion, `is_sm90`, `_IS_SM8`) is
  resolved via a one-time TP-group `all_reduce` and cached, so ranks with different
  chips capture identical graphs (common-denominator path on mixed hosts; single-arch
  hosts byte-identical).
- **Cold-cubin prewarm**: both HC-mix paths (fused sinkhorn and split aten sinkhorn)
  are eagerly warmed at load, so a first `cuModuleLoadData` can never land inside a
  capture collective.
- **c128 multi-bucket decode graphs are group-consistent**: on a heterogeneous group
  all ranks fall back to one shared 128-aligned bucket table
  (`[128, 512, 1024, 2048, 4096]` ∩ pool width), so capture issues the same number of
  collectives on every rank. Same-arch hosts keep their arch-native tables.
- **CuTe DSL device-0 probe fixed** (capture-time `NVVM_ERROR_COMPILATION` on SM120
  ranks of a mixed host): the model runner re-points the DSL compile target at each
  rank's own device. Manual fallback `FLASHINFER_USE_CUDA_NORM=1` remains available.
- **SM120 TP=4 MXFP4 CUTLASS MoE** auto-pads per-rank intermediate 576 → 640 (weight
  `0`, e8m0 scale `1`); CPU-resident layers keep exact strides for lk_moe pointer use.
- **SM8x + `flashinfer_mxfp4`** resolves to **Marlin** automatically (FlashInfer FP4
  kernels are 90/100/103/107/110/120-only).

### SM120 sparse-MLA prefill — native fast path, no fallback env needed
DeepSeek-V4.1 ratio-1/2/c128 layers use a second ("extra") KV pool with 128/256-token
pages. The SM120 sparse-MLA **prefill** kernel only accepts 64-token pages on *both*
sources, so long prompts (>64 tokens) used to die with
`Unsupported sparse-MLA prefill configuration: ... extra_page_block_size=128`, and
`SGLANG_SM120_FLASHMLA_BACKEND=triton` was the (slower) workaround.
The extra source now goes through the same 64-token page-split as the main source
(per-role scratch buffers so the two can never alias), verified **bit-exact** against
the native-64 layout. Prefill/decode are both native FlashInfer on SM120; the
`SGLANG_SM120_FLASHMLA_BACKEND` env is no longer needed (decode/verify untouched).

## GPU support matrix (DeepSeek-V4 / V4.1 with lk_moe)

| SM | Representative GPUs | Patches | Notes |
|----|--------------------|---------|-------|
| **80** | A100, A30 | 01 + 02 | SM8 path shared with SM86 (`w8a16` Triton GEMMs, native-precision fp8 emulation) |
| **86** | RTX 3090/3080, A6000 | 01 + 02 | ✅ verified: V4.1-Flash on 2×3090 (TP=2) and mixed 4-GPU TP=4 |
| **89** | RTX 4090/4060, L40/L40S | 01 (02 for GPU-resident MoE) | ✅ verified: V4.1-Flash on 2× RTX 4080 SUPER, 60 t/s (DSPARK) |
| **90** | H100/H200/H800/H20 | 01 | upstream-native path (DeepGEMM FP8/FP4, trtllm MoE) |
| **100** | B200/GB200 | 01 | upstream-native Blackwell path (FlashInfer FP4, split-K sinkhorn) |
| **120** | RTX 5060Ti/5080/5090, RTX PRO 6000 | 01 | ✅ verified: 2×5060Ti TP=2 and mixed 4-GPU TP=4; prefill fast path native since this release |

✅ = measured on reference hardware (dual-EPYC host). Other rows: enabled-by-construction
on the shared code paths, not individually bench-tested here — please report issues.
SM8x ranks mixed with SM12x ranks run on the common-denominator path (see What's new).

## Install / build
```bash
pip install lsglang==1.5.3          # or build the wheel from tag lsglang-v1.5.3
```

## Patches (`patches/`)
| Patch | Applies to | Contents |
|-------|-----------|----------|
| [`01_lk_moe__dsv4.1.patch`](./patches/01_lk_moe__dsv4.1.patch) | clean sglang `dsv4.1` (`1aa0e962b`) | pure lk_moe MOE hybrid inference |
| [`02_sm80_support__dsv4.1.patch`](./patches/02_sm80_support__dsv4.1.patch) | after 01 | SM80/86 attention & GEMM ports + mixed-arch TP fixes + SM120 prefill extra-split |

```bash
git apply patches/01_lk_moe__dsv4.1.patch            # SM90+/SM120+ stop here
git apply patches/02_sm80_support__dsv4.1.patch      # + SM80/86 & mixed-arch support
```

## Launch — DeepSeek-V4.1-Flash (SM86, reference)

**~30 t/s decode** — Test environment: Dual EPYC 7642, 16-channel DDR4 3200, Dual RTX 3090 (SM86, TP=2).

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,3 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LVLLM_MOE_NUMA_ENABLED=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=2048 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=48 \
LK_POWER_SAVING=1 \
OMP_NUM_THREADS=1 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
sglang serve \
  --model ~/Models/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 0.0.0.0 \
  --port 8070 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-running-requests 2 \
  --max-total-tokens 80000 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.95 \
  --cuda-graph-backend-prefill disabled \
  --disable-shared-experts-fusion
```

Adjust `--model` path, `CUDA_VISIBLE_DEVICES` and `LK_THREADS` to your host.
For a mixed SM86+SM120 host use the same command with `--tensor-parallel-size 4` and
all four GPUs in `CUDA_VISIBLE_DEVICES` (`CUDA_DEVICE_ORDER=PCI_BUS_ID` is required so
each rank's arch probe sees its own chip).

One caveat: `LVLLM_GPU_RESIDENT_MOE_LAYERS` is left unset above, i.e. every expert layer
stays CPU-resident — that is the ~30 t/s config measured. It is a **layer-index list, not a
count**, so `=0` would make *layer 0* GPU-resident, and on SM80/86 that needs
`--moe-runner-backend marlin` (the default `flashinfer_mxfp4` resolves to the TRT-LLM path,
whose FP4 kernels are SM90+ only, so weight loading dies with `ValueError: Invalid backend: 86`).

## Known issue (open)

- **4-GPU mixed TP=4 DSPARK throughput** currently trails the dual-3090 reference
  (~15–20 vs ~30–35 t/s, same prompt; non-speculative decode unaffected). GPU-side
  traces show no kernel regression; under investigation (step serialization /
  4-way NCCL sync on PCIe-only links). Single-arch TP=2 paths unaffected.

## Additional support branches (v0.5.19 series)

### DeepSeek V4 (SM80+)

| Branch | Arch |
|--------|------|
| [0.5.19-lkmoe-deepseekv4-sm80plus](https://github.com/guqiong96/Lsglang/tree/0.5.19-lkmoe-deepseekv4-sm80plus) | SM80+ |

### GLM-5.3 Flash (SM80+ / SM120+)

| Branch | Arch |
|--------|------|
| [pr-20-22-contd](https://github.com/usrlocalben/Lsglang/commits/pr-20-22-contd/) by usrlocalben | SM120+ |
| [lkmoe-glm5.3-flash-sm80plus](https://github.com/guqiong96/Lsglang/tree/lkmoe-glm5.3-flash-sm80plus) by a775828b-dot and guqiong96 | SM80+ |

Related issue: [guqiong96/Lsglang#21](https://github.com/guqiong96/Lsglang/issues/21)

### Qwen3.8-Flash-Next (SM89+ / SM120+)

`feat/qwen38-flash-next` (by lovedheart); lk_moe integrated on top, released as `lsglang-v1.4.13`.
SM89 requires the flash-attention PR #2751 patch (prebuilt `flash_attn-2.8.4+pr2751` wheel).

| Branch | Arch | Author |
|--------|------|--------|
| [feat/qwen38-flash-next](https://github.com/lovedheart/sglang/tree/feat/qwen38-flash-next) | SM89+ / SM120+ | lovedheart |
