https://github.com/guqiong96/Lsglang/tree/dsv4.1-lkmoe-sm80plus

**Base:** sglang `dsv4.1` (upstream branch, commit `1aa0e962b`) · **lk_moe v2.4.3**
**Type:** Community + stability release — two V4.1 memory PRs from @a775828b-dot (prefill
scratch bounding, NVMe Engram + SM120 b12x dispatch) + engram host-table pin-order fix
**Wheel:** `lsglang-1.5.6` — use this single wheel for all GPUs (SM80 → SM120)

Everything from v1.5.5 (mixed-arch capture root fix, cold-cubin prewarms, V4.1
indexer prefill-OOM fix, SM89 by-M select, SM120 plain-decode routing) carries over
unchanged. This release is additive: all new behavior is **opt-in or default-equivalent**;
no existing route changes when the new envs stay off.

## What's new

### V4.1 prefill scratch memory bounded — block-compact candidate masks (PR #27, @a775828b-dot)
Eager prefill used to retain the **position-expanded** candidate mask per indexer layer
(one bool per position per layer). The candidate selector only ever decides per *block*,
so masks are now kept at **block granularity** and the current chunk alone is expanded at
consumption (bit-exact: the selector's mask is a `repeat_interleave` of block bits, so
block-compact ↔ expanded is lossless both directions; graph-decode tensor masks are
untouched). Additionally `SGLANG_DSV41_INDEXER_SCORE_BUDGET_BYTES` (default 1 GiB, the
old hard-coded constant) now also gates the **dense FP4** indexer's FP32 score allocation:
over-budget calls take the existing torch fallback instead of a single huge alloc.
Lowering the budget trades prefill speed for peak memory. Validated 192
shape/block/top-k/tie combinations per device against the original selector.

### NVMe Engram backend + SM120 b12x small-batch MXFP8 dispatch (PR #28, @a775828b-dot)
Two independent opt-ins:
- **`SGLANG_ENABLE_DSV41_ENGRAM_NVME=1`** — Engram FP8 rows + E8M0 scales stay in the
  immutable safetensors shards; a direct-I/O reader (native AIO, bounded per-layer row
  cache + pinned staging) fetches only the rows each forward needs, so host RAM no longer
  bounds the table size. TP=1, single worker/request, fixed-width DSpark (gamma 3/5,
  verify width 4/6); every unsupported speculative/compact combination fails closed at
  init. Mutually exclusive with the host-table option.
- **`SGLANG_SM120_MXFP8_B12X_SMALL_BATCH=1`** — on SM120 only, the FlashInfer MXFP8
  linear wrapper's `cutlass` calls at flattened **M=1/4/6** (decode / fixed-width DSpark
  verify) route to `b12x`; everything else keeps its dispatch. Author's SM120 component
  bench: 7-shape bundle 0.88 ms → 0.21 ms (4.2x, input quantization included). Note this
  is component time, not model t/s: hosts whose decode is bound by CPU-expert traffic
  will see little change.
See `examples/runtime/deepseek_v4/README.engram_nvme.md` and `README.sm120_mxfp8.md`.

### Engram host table: pin *after* page folding (regression fix)
`cudaHostRegister` ran inside the table constructor, which **GUP-pins whatever 4K
fallback pages fault-time THP happened to leave** — after that neither khugepaged nor
`MADV_COLLAPSE` (absent on 5.14/EL9 kernels, EINVAL) can ever fold them, so a single
unlucky fault window cost ~10x lookup speed for the process lifetime. Registration is
now deferred to the end of `finish_load`: collapse attempt first, then a bounded
khugepaged window (20 s) when still under the 98% huge-page bar, and pin last. The
resident/huge-page line in the log now reflects the final backing before pinning.

### Huge pages for the Engram host table (launch guidance)
The 48 GiB/layer host tables want 2 MiB pages; whether you get them is mostly **when**
you start, not **what flags** you pass:
- The loader already drops checkpoint page cache before pre-faulting — but any *other*
  process reading the same checkpoint refills it and steals the contiguous 2 MiB blocks.
  **Stop the co-resident server during model load**, or at least:
  `sync; echo 1 > /proc/sys/vm/compact_memory` right before starting.
- `transparent_hugepage/enabled=always` + `defrag=madvise` is the supported setting
  (both are the distro default on the reference hosts).
- Kernel ≥ 6.1 folds synchronously via `MADV_COLLAPSE`; on RHEL9's 5.14 the post-fault
  `khugepaged` window above is the only folder.
- Final fix on 1 TiB hosts: reserve 1G pages at boot
  (`default_hugepagesz=1G hugepagesz=1G hugepages=128`) — the hugetlb-backed table
  layout is planned for a follow-up release.
The startup log always prints the verdict: `... MiB in huge pages (NN%)` — 0% with
`pinned` means the window was lost; expect engram lookups ~10x slower.

## What's new since v1.5.3 (carried from v1.5.4/v1.5.5, unchanged in this release)

### Mixed-arch TP=4 CUDA-graph capture — root-caused and fixed
The capture loop iterates a **candidate-variant axis** that was gated on a *local*
`major >= 10` device check: on a mixed host the SM120 ranks captured several variants per
batch size while SM86 ranks captured one, so the per-bs collective counts diverged and NCCL
busy-spin deadlocked the warmup. The gate is now a one-time **TP-group AND** (`all_gather`
of majors, unconditional on every rank): homogeneous hosts keep their native variant tables
byte-identically, mixed hosts take the common denominator on *every* capture axis
(c128 buckets + candidate variants). Verified: mixed 2×3090 + 2×5060Ti TP=4 now captures
and serves with a fully cold triton/flashinfer cache.
- Plus the remaining **cold-cubin prewarms** at load time (rank-uniform, failure-safe):
  block-fp8 w8a16 M-buckets, CuTe fused-RMSNorm first-init, `wo_a` bf16 cublasLt heuristic,
  MoE vision-gate router specializations, and the Marlin single-token align kernels.

### V4.1 low-ratio indexer no longer OOMs long-prompt extends on 16 GB cards
`Indexer.scores` materialized the full `[rows, 32, lc]` head-concat bf16 cube
(≈1 GiB per row chunk, ~2.8 GiB transient per low-ratio layer — a 16k-token extend was
deterministically fatal on RTX 5060Ti-class cards under `--mem-fraction-static 0.95`).
Columns are now scored in 32 MiB tiles with the head axis reduced per tile (short tail
tile masked by its own width), so the full cube never exists; peak measured
**2766 MiB → 154 MiB**, outputs bit-identical on SM86 and SM120, decode/verify and
short prompts keep the original one-shot path byte-for-byte (~1.0–1.14× on tiled
prefills only).

### SM89 dense block-fp8 selects kernel by M (RTX 4090/4080 dspark)
`_is_ada_sm89` → `w8a16` regressed DSPARK verify (bf16 math leaves the Ada fp8 tensor cores
idle at verify-M). The SM89 dispatch now picks by token count: `M <= SGLANG_SM89_W8A16_MAX_M`
(default 4) → w8a16 (decode, weight-bandwidth bound), larger M → fp8-dot Triton (verify).
Verified 4× RTX 4080 SUPER: plain decode 44 t/s **and** DSPARK back to **65 t/s**.

### MXFP8 symbol registration no longer depends on import-time device
The FlashInfer MXFP8 dense ops are registered when *any* Blackwell GPU is visible
(`CUDA_VISIBLE_DEVICES` scan) instead of "current device is Blackwell at import", fixing a
latent `NameError` on SM120 ranks whose process imports before `set_device` (mixed hosts).

### DeepSeek-V4.1 plain-decode throughput (SM89 + SM120)
On non-SM80 archs the dense block-fp8 linears (block `[32, 32]` + `ue8m0` scales = MXFP8
semantics: `wqkv_a`, `wq_b`, `wo_b`, `wkv`, the indexer `wq_b`, the shared-expert and
engram projections) fell back to the Triton block kernel's **untuned default config**
(`BLOCK_N=32`, no split-K outside SM90). At decode-M=1 the weight stream saturates on too
few CTAs, so only **plain decode** looked slow (SM120 ~15 t/s, SM89 ~18 t/s); DSPARK and
prefill hid it behind accept-amortization.
- **SM120**: `auto` now resolves these weights to **FlashInfer MXFP8 (CUTLASS)**. The
  server pre-resolves `auto → cutlass` for the 128×128 block path; that pre-resolution is
  now treated as `auto` for this *separate* MXFP8 route, so a server process routes exactly
  like a bare import. Verified 2× RTX 5060Ti TP=2: plain decode **15 → 21 t/s**.
- **SM89**: routed to the tuned **`w8a16`** kernel (the proven SM80/86 path with automatic
  split-K) instead of the fp8-dot Triton kernel (no split-K tuned config on Ada).
  Verified on 2× RTX 4080 SUPER.
- **#36655 backport**: SM120 sparse-MLA decode uses the exact per-rank head count instead
  of padding to 64 heads when the installed FlashInfer advertises it (fail-closed; the
  padded path stays the fallback for older builds).
- New diagnostic `SGLANG_DSV41_DISABLE_DECODE_SIDE_STREAMS=1` (default off) collapses the
  three decode-only side streams to the SM80/86 single-stream structure for A/B.

### Mixed-arch TP group structure (2× RTX 3090 + 2× RTX 5060Ti, TP=4)
- **Forward structure is now TP-group uniform.** Every arch predicate that changes the
  graph shape (Blackwell split-K sinkhorn, `wo_a` rope fusion, `is_sm90`, `_IS_SM8`) is
  resolved via a one-time TP-group `all_reduce` and cached, so ranks with different
  chips capture identical graphs (common-denominator path on mixed hosts; single-arch
  hosts byte-identical).
- **Cold-cubin prewarm**: both HC-mix paths (fused sinkhorn and split aten sinkhorn)
  *and every engram-hash specialization* (decode/verify/extend modes, the verify
  `BLOCK` constexpr, `HIST_VIA_SLOTS`, and the `num_real` buckets, plus the verify
  commit kernel) are eagerly warmed at load, so a first `cuModuleLoadData` can never
  land inside a capture collective. This closes the cold-cache first-run capture
  deadlock where one arch hit its warm triton disk cache while its mixed-arch peer
  cold-loaded inside the capture warmup.
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
| **86** | RTX 3090/3080, A6000 | 01 + 02 | ✅ verified: V4.1-Flash on 2×3090 (TP=2). Mixed 4-GPU TP=4: captures and serves, but output correctness under investigation (see Known issues) |
| **89** | RTX 4090/4060, L40/L40S | 01 (02 for GPU-resident MoE) | ✅ verified: V4.1-Flash on 2× RTX 4080 SUPER, 60 t/s (DSPARK); plain decode via `w8a16` block-fp8 route since v1.5.4 |
| **90** | H100/H200/H800/H20 | 01 | upstream-native path (DeepGEMM FP8/FP4, trtllm MoE) |
| **100** | B200/GB200 | 01 | upstream-native Blackwell path (FlashInfer FP4, split-K sinkhorn) |
| **120** | RTX 5060Ti/5080/5090, RTX PRO 6000 | 01 (02 for mixed-arch TP + the sparse-MLA prefill fast path) | ✅ verified: 2×5060Ti TP=2 (plain decode → FlashInfer MXFP8 CUTLASS; optional `SGLANG_SM120_MXFP8_B12X_SMALL_BATCH=1` routes M=1/4/6 to b12x). Mixed 4-GPU TP=4: output correctness under investigation (see Known issues) |

✅ = measured on reference hardware (dual-EPYC host). Other rows: enabled-by-construction
on the shared code paths, not individually bench-tested here — please report issues.
SM8x ranks mixed with SM12x ranks run on the common-denominator path (see What's new).

## Known issues

### Mixed-arch TP=4 (SM86 + SM120) output correctness — OPEN
On the 2× RTX 3090 + 2× RTX 5060Ti TP=4 host, V4.1-Flash captures, serves, and streams
decode (cuda graph True, ~30 t/s) but long generations occasionally emit a garbled token
(a missing/invalid UTF-8 character, or a wrong identifier inside code output). Every
single-arch config is clean: **2×3090, 2×5060Ti, and 4×4080S all produce correct output**;
the fault appears only when SM86 and SM120 ranks share one TP group.
- Every flag-isolable axis has been ruled out by A/B or offline numeric proof: `_tp_all`
  group forcing (`SGLANG_DSV4_TP_ARCH_LOCAL=1` — still broken), the TileLang indexer
  dispatch, the CUTLASS 576→640 MoE padding (byte/bit-exact vs a dequant reference), the
  MoE backend choice (`--moe-runner-backend marlin` — still broken), and the SM120 dense
  MXFP8 route (`--fp8-gemm-backend triton` — still broken).
- Narrowing to the SM86-vs-SM120 attention / KV-selection kernel split under an aligned
  collective schedule; needs a greedy + `return_logprob` first-divergent-token capture to
  localize. Single-arch users are unaffected; on a mixed host, use TP=2 pairs for now.

## Install / build
```bash
pip install lsglang==1.5.6          # or build the wheel from tag lsglang-v1.5.6
```

## Patches (`patches/` Note: these patches are only meant to document the diff against upstream sglang. They are already included in the lsglang install and can be ignored.)
| Patch | Applies to | Contents |
|-------|-----------|----------|
| [`01_lk_moe__dsv4.1.patch`](./patches/01_lk_moe__dsv4.1.patch) | clean sglang `dsv4.1` (`1aa0e962b`) | pure lk_moe MOE hybrid inference |

```bash
git apply patches/01_lk_moe__dsv4.1.patch            # SM90/SM100 (and SM120 basic) stop here
```

SM80/86 ports, SM120 mixed-arch & prefill fast path ship in the wheel only (no standalone patch).

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
  --disable-shared-experts-fusion \
   --speculative-algo DSPARK \
  --speculative-dspark-block-size 5 \
  --speculative-attention-mode decode  \
  --enable-decoder-swa-bounded-replay 
```

Adjust `--model` path, `CUDA_VISIBLE_DEVICES` and `LK_THREADS` to your host.
For a mixed SM86+SM120 host use the same command with `--tensor-parallel-size 4` and
all four GPUs in `CUDA_VISIBLE_DEVICES` (`CUDA_DEVICE_ORDER=PCI_BUS_ID` is required so
each rank's arch probe sees its own chip).

## Launch — DeepSeek-V4.1-Flash (SM120, 2× RTX 5060Ti)

**~21 t/s plain decode** (65k context config) — Test environment: Dual EPYC 7642,
16-channel DDR4 3200, 2× RTX 5060Ti 16 GB (SM120, TP=2), `Lsglang-v1.5.6`.
The host tables are ~48 GiB per engram layer on a 1 TiB host: keep
`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` and read the **huge-pages note** below the
command before blaming the GPU for a slow first hour.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,2 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_GPU_PREFETCH_WINDOW=1 \
LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_SKIP_P2P_CHECK=1 \
LK_POWER_SAVING=1 \
sglang serve \
  --model ~/Models/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port 8070 --trust-remote-code \
  --tensor-parallel-size 2 --max-running-requests 2 \
  --chunked-prefill-size 1024 --max-total-tokens 65536 --mem-fraction-static 0.92 \
  --dist-timeout 180 \
  --cuda-graph-backend-prefill disabled --disable-shared-experts-fusion \
  --enable-decoder-swa-bounded-replay \
  --speculative-algo DSPARK --speculative-dspark-block-size 5 \
  --speculative-attention-mode decode
```

16 GB cards: `--max-total-tokens` is an **explicit** KV reservation, unlike the
automatic budget on some engines — 256000 over-reserves ~8.5 GiB and OOMs the engram
host-table path; 65536 is the verified 5060Ti value. `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0`
disables the GPU-MoE prefill pool (saves the largest single slice of card memory on
16 GB cards); raise it to 1024+ only on 24 GB+ cards. Optional decode/verify speedups:
`SGLANG_SM120_MXFP8_B12X_SMALL_BATCH=1` (see What's new; matters most when the GPU,
not the CPU expert path, is the bottleneck).

One caveat: `LVLLM_GPU_RESIDENT_MOE_LAYERS` is left unset above, i.e. every expert layer
stays CPU-resident — that is the ~30 t/s config measured. It is a **layer-index list, not a
count**, so `=0` would make *layer 0* GPU-resident, and on SM80/86 that needs
`--moe-runner-backend marlin` (the default `flashinfer_mxfp4` resolves to the TRT-LLM path,
whose FP4 kernels are SM90+ only, so weight loading dies with `ValueError: Invalid backend: 86`).

## Additional support branches (v0.5.19 series)

### DeepSeek V4.1 Flash
| Branch | Arch |
|--------|------|
| https://github.com/usrlocalben/Lsglang/tree/ds41 by usrlocalben | SM120 |

Related issue: [guqiong96/Lsglang#25](https://github.com/guqiong96/Lsglang/issues/25)

### DeepSeek V4 (SM80+)

| Branch | Arch |
|--------|------|
| https://github.com/guqiong96/Lsglang/tree/0.5.19-lkmoe-deepseekv4-sm80plus | SM80+ |

### GLM-5.3 Flash (SM80+ / SM120+)

| Branch | Arch |
|--------|------|
| https://github.com/usrlocalben/Lsglang/commits/pr-20-22-contd/ by usrlocalben | SM120+ |
| https://github.com/guqiong96/Lsglang/tree/lkmoe-glm5.3-flash-sm80plus by a775828b-dot and guqiong96 | SM80+ |

Related issue: [guqiong96/Lsglang#21](https://github.com/guqiong96/Lsglang/issues/21)

### Qwen3.8-Flash-Next (SM89+ / SM120+)

`feat/qwen38-flash-next` (by lovedheart); lk_moe integrated on top, released as `lsglang-v1.4.13`.
SM89 requires the flash-attention PR #2751 patch (prebuilt `flash_attn-2.8.4+pr2751` wheel).

| Branch | Arch | Author |
|--------|------|--------|
| [https://github.com/lovedheart/sglang/tree/feat/qwen38-flash-next | SM89+ / SM120+ | lovedheart |
