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

Note 2: **The first launch is slow because of JIT compilation.** Expect heavy compilation
during startup — you can see `cicc`, `cc1plus`, `ptxas` (and `nvcc`) processes in `top`.
The second launch reuses the JIT cache and starts much faster. Some models also JIT extra
kernel variants on their first request; this is the same behavior and those variants are
cached for subsequent runs as well.

---

## Layer compute modes

The modes below are **per-MoE-layer runtime roles**: within one model, different layers can run
in different modes at the same time (attention is always on the GPU). Speculative decoding is the
exception — it is a global switch; GPU Pool is global today as well (per-layer pool control is on
the roadmap).

### Layer compute modes (per layer)

| # | Mode | Decode path | Prefill path | Control | When to use |
|---|---|---|---|---|---|
| 1 | **GPU** (stock sglang path) | weights resident in VRAM, original GPU kernels | same | `LK_GPU_RESIDENT_MOE_LAYERS` | Large VRAM; saves host memory, lifts decode and prefill together |
| 2 | **CPU** (hybrid) | MoE on CPU (NUMA host weights), attention on GPU | MoE on CPU, chunk by chunk | default layer role when `LK_MOE_HYBRID_ENABLED=1` | **Recommended for high-performance CPU subsystems** (many cores + high memory bandwidth); zero extra VRAM |
| 3 | **CPU + Speculative decoding** | speculative verification × CPU MoE | same | `LK_MOE_HYBRID_ENABLED=1` + `--speculative-algorithm NEXTN` / `DSPARK` | **Recommended for high-performance CPU subsystems** |
| 4 | **CPU + GPU Pool** (expert cache) | pool-resident experts on GPU, the rest on CPU | same split | `LK_POOL_SLOTS` (e.g. 64) | **Recommended for low-power CPU subsystems + high-performance GPUs** (spare GPU compute + high-bandwidth PCIe) |
| 5 | **CPU + GPU Pool + Speculative decoding** | speculative verification × pool's GPU/CPU halves | same split | `LK_POOL_SLOTS` (e.g. 64) + `--speculative-algorithm NEXTN` / `DSPARK` | **Recommended**: the draft rows ride the residency the real tokens build (`LK_VERIFY_RESIDENT`, default on) — measured +17.6% over the CPU-only path on DeepSeek-V4-Flash-0731 / DSPARK, acceptance unchanged |

**Overlay — Super Mix Prefill**: GPU prefill with super-chunk sizing (decoupled from the
token budget), runs in parallel with CPU-side decode at near-100% GPU utilization;
coordinates automatically with modes 2-5.
`LK_GPU_PREFILL_MIN_TOKENS` (e.g. 1024) + `LK_GPU_PREFILL_SUB_M` (e.g. 16384).

> **Selection principle**: whenever VRAM has headroom, keep prefill on the GPU
> (Super Mix Prefill and/or pool/GPU-resident modes).

---

## Getting started

### Installation

Prerequisite: **CUDA 13.2.1** (driver + toolkit) — matches the `torch==2.14.1` CUDA build below.

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
pip install torchaudio triton torchvision torch==2.14.1
pip install grpcio-tools wheel-stub
MAX_JOBS=32 NVCC_THREADS=1 CMAKE_BUILD_TYPE=Release \
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" \
pip install -e "python" --no-build-isolation -vvv
```

(`MAX_JOBS=32 NVCC_THREADS=1`: reduce compile memory; `CMAKE_BUILD_TYPE=Release`: perf option.)

### Quick start (Qwen3.8-Flash-Next)

Two production recipes cover the common hardware pairs. They are the exact scripts the
benchmark rows above were measured with, so copy the script instead of pasting a command:

| Hardware | Script | Decode |
|---|---|---|
| RTX 3090 x2 (24 GB each, GPUs 0,3) | [`launch/sglang_serve_qwen38_mtp_tp2_3090.sh`](launch/sglang_serve_qwen38_mtp_tp2_3090.sh) | NEXTN speculative; plain-decode twin: [`sglang_serve_qwen38_tp2_3090.sh`](launch/sglang_serve_qwen38_tp2_3090.sh) |
| RTX 5060 Ti x2 (16 GB each, GPUs 1,2) | [`launch/sglang_serve_qwen38_mtp_tp2_5060ti.sh`](launch/sglang_serve_qwen38_mtp_tp2_5060ti.sh) | NEXTN speculative; plain-decode twin: [`sglang_serve_qwen38_tp2_5060ti.sh`](launch/sglang_serve_qwen38_tp2_5060ti.sh) |

```bash
bash launch/sglang_serve_qwen38_mtp_tp2_3090.sh     # 3090 pair
bash launch/sglang_serve_qwen38_mtp_tp2_5060ti.sh   # 5060 Ti pair
```

Both keep the expert cache inside one VRAM pool and print `auto extended LK_POOL_SLOTS X to Y`
at boot when the configured cache was below what one prefill pass needs (the rows are free --
they ride in the same allocation).

---

## Configuration parameters

All lk_moe switches use the unified `LK_*` prefix. Legacy `LVLLM_*` names are still accepted: they are auto-renamed at startup with a one-line warning (if both spellings are set, `LK_*` wins). Our knobs never use the `SGLANG_*` prefix.

### New server flag (this fork): `--moe-runner-backend sm12x_nvfp4_bf16`

Adds `sm12x_nvfp4_bf16` to the upstream `--moe-runner-backend` choices (all
upstream values unchanged). Use it for NVFP4 (modelopt) MoE checkpoints on
SM120-family GPUs - RTX 5090 / 5060 Ti / RTX PRO 6000: MoE layers whose expert
weights are GPU-resident run SM120 native FP4 tensor-core grouped GEMMs
instead of the generic triton fallback.

Scope: layers served by lk_moe (CPU-offloaded, pooled, GPU-prefill tier) never
route through the sglang runner, so this flag does not affect them - it is
safe, and recommended, to set it together with any `LK_*` configuration. The
`launch/` recipes targeting 5060 Ti cards already pin it. It is independent of
`LK_GPU_PREFILL_SM120_W4A4`: that knob is about the pool tier's prefill
activations, this one about resident-layer GEMMs.

### Switch linkage (one master gate)

`LK_MOE_HYBRID_ENABLED` is the single master switch: with it off (the default) `lk_moe` is never imported and **every `LK_*` below is inert** — the server runs the stock sglang MoE path bit-for-bit. Every downstream tier is ANDed with it (switches only — perf knobs never appear in a linkage):

```
LK_MOE_HYBRID_ENABLED=1                     master
 ├─ pool on            LK_POOL_SLOTS (e.g. 64)      (dual knob: value itself is the switch, 0 = off)
 │   └─ pool decode    LK_POOL_DECODE=1     (unset = follows pool on; explicit `1`/`0` wins)
 ├─ GPU prefill on     LK_GPU_PREFILL_MIN_TOKENS (e.g. 1024)   (dual knob, same rule)
 │   ├─ engine         LK_GPU_PREFILL_ENGINE=cutlass|triton|flashinfer (default cutlass; triton = opt-out;
 │   │                                             flashinfer = SM120/SM121 + NVFP4 checkpoints only)
 │   ├─ w4a4 level     LK_GPU_PREFILL_SM120_W4A4=1   (NVFP4 activations; SM120-family + flashinfer engine)
 │   ├─ wave           LK_GPU_PREFILL_WAVE_EXPERTS (cutlass only; -1 auto / 0 single-shot / >0 cap)
 │   └─ super mix      LK_GPU_PREFILL_SUB_M (e.g. 16384)  (>0 super-chunk staging; 0 = per-chunk)
 ├─ nvme filler on    LK_NVME_DIRECT=1          (needs the feature's own backend flag: --ple-offload-backend nvme / engram host table;
 └─ resident layers   LK_GPU_RESIDENT_MOE_LAYERS*   (complementary: a resident layer never pools)
LK_MOE_HYBRID_ENABLED=0  →  every branch above unreachable = original sglang
( LK_NVME_DIRECT with the engram/PLE nvme backends is the one knob that also works without the LK master )
```

### Core

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_MOE_HYBRID_ENABLED` | core | `0` | **master switch**: `1`-on, `0`-off (= stock sglang, all `LK_*` inert) |

### CPU / NUMA / power

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_THREADS` | perf | cores/GPU | host decode thread count (read directly by lk_moe at bootstrap) |
| `LK_NUMA_ILP` | perf | `0` | `1`: interleaved ILP mode in the NUMA allocator (pairs with `LK_NUMA_HUGE`; read by lk_moe) |
| `LK_THREAD_BINDING` | perf | `CPU_CORE` | `CPU_CORE` bind by core, `NUMA_NODE` bind by node |
| `LK_NUMA_INTERLEAVE` | perf | 1 | `1`: interleave host weights across NUMA nodes (avoids node OOM; required for the pump's per-node copy streams); `0` off |
| `LK_NUMA_HUGE` | perf | `0` | `1`: opt-in 2M THP-aligned host slabs (single-node / THP-friendly boxes); NUMA-interleave, not page size, is the usual win |
| `LK_POWER_SAVING` | power | 0 | `1`: an idle worker `yield()`s instead of spinning (and sleeps 4 s instead of 2 s after 120 s idle) — **standby only**, hands CPU back while the server waits for work; nothing in the compute path reads it, so it does not lower power while inferring |
| `LK_HF_MBOX` | perf | `1` | mailbox early-start of host-side decode (non-`0` on, `0` off) |
| `OMP_NUM_THREADS` | perf | - | set to 1 to avoid slow model loading |

### GPU-resident layers

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_GPU_RESIDENT_MOE_LAYERS` | GPU | none | expert layers resident in VRAM, e.g. `0`, `0-1`, `0,9` |
| `LK_GPU_RESIDENT_MOE_LAYERS_SPEC` | GPU | none | draft model layers resident in GPU |
| `LK_GPU_RESIDENT_MOE_LAYERS_MTP` / `LK_GPU_RESIDENT_MOE_LAYERS_DSPARK` | GPU | follow `_SPEC` | per-draft-role override (MTP / DSpark workers) |

### Super Mix Prefill

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_GPU_PREFILL_MIN_TOKENS` | prefill | 1024 (on) | **tier switch AND batch gate**: the value itself switches (e.g. 1024 turns the tier on; `0` = pure CPU prefill) = minimum token batch for the GPU path. On by default; set `0` for pure CPU prefill (16 GB cards) |
| `LK_GPU_PREFILL_ENGINE` | prefill | `cutlass` | which GPU prefill path runs above the token gate. `cutlass`: fastest on long prompts, works with every pooled dtype; rounds activations to bf16 — long-context logits can shift and borderline EOS can flip; `triton`: output bit-matches the CPU reference, slightly slower; `flashinfer` (SM120/SM121 + **NVFP4 checkpoints only** — the CuTe kernels are NVFP4-only, an fp8/mxfp4 pool layer fails its init probe and the boot aborts by design): stages the weights as FP4 and runs the FP4 tensor cores — works standalone at W4A16 (CPU-matching) numerics, add `LK_GPU_PREFILL_SM120_W4A4=1` to run activations in FP4 too; forwarded via cfg, the C++ library itself does not read this env |
| `LK_GPU_PREFILL_SM120_W4A4` | prefill | 0 = off | `0`: prefill activations run as BF16 (W4A16) — numerics equal the CPU path on every GPU; `1`: activations run as FP4 (W4A4) — visibly faster prefill, slightly different numerics. Needs the `flashinfer` engine (the only engine implementing W4A4) *and* an SM120-family card; otherwise a one-time warning is logged and prefill stays W4A16 |
| `LK_GPU_PREFILL_WAVE_EXPERTS` | prefill | `-1` auto | tier wave width, ONE geometry for all three engines, frozen at init: `-1`/unset = one wave over all E experts (the gate-tested default), `>0` = W experts per wave - a per-wave gemm loop only the cutlass engine has, so `0 < W < E` on a staging engine (triton/flashinfer) aborts at init; appears as `wave=` in the `[tier]` boot line |
| `LK_GPU_PREFILL_D_SLOTS` | prefill | 0 = off | tier activation window in slot rows, frozen at init (the `act_rows=` figure in the `[tier]` boot line); **all three engines** pace their prefill passes to hold it, at the cost of extra kernel launches, result unchanged. The two staging engines never walk more than their built-in 8192-slot block, so the knob only *tightens* their walk: a value at or above that block (and `0`) leaves them byte-identical to before, and the triton scratch follows a tighter walk. The staging arena still reports `act 0 B` -- its activations live in the driver buffers, not in the arena. `0` = one window per pass (cutlass) |
| `LK_GPU_PREFILL_SUB_M` | prefill | 0 | super-mix staging size (tokens); only meaningful while the tier is on (`LK_GPU_PREFILL_MIN_TOKENS > 0`): `>0` enables super-chunk staging (e.g. 65536), `0` = per-chunk staging; the super-chunk attention wrapper only arms when SUB_M exceeds `--chunked-prefill-size`. Bigger values move more of each expert's weights per token (faster) but raise the token budget of a single forward - the `CAP=` in the `Super-chunk prefill wrapper active` boot line - and with it the scratch that scales with that budget, incl. a pre-reserved output buffer of ~ SUB_M × hidden × 4 bytes per rank that is already counted against the KV budget; halving it roughly halves that scratch |

### GPU Pool (expert cache)

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_POOL_SLOTS` | pool | 0 = off | **GPU Pool switch AND expert-cache capacity**: the value itself switches (e.g. 64 turns the pool on; `0` = off, all `LK_POOL_*` inert) = resident expert slots per MOE layer. If it is below what one prefill pass needs it is **auto-extended** at boot (free: those rows sit in the same allocation) and the log prints `auto extended LK_POOL_SLOTS X to Y`; a larger value is never shrunk. It sizes the CACHE only: the domain's total VRAM is `max(cache, prefill working set)` — see [VRAM sizing](#vram-sizing-pool--gpu-prefill). Must be `>= top_k` when non-zero |
| `LK_POOL_DECODE` | pool | follow pool | decode GPU half: pool-resident experts go to the GPU resident-gemv half. `1` on / `0` off explicitly; unset follows `LK_POOL_SLOTS` > 0 |
| `LK_POOL_MBOX_SPLIT` | pool | 0 = off | Start the decode GPU half before joining the CPU half instead of in one step. Nothing measured at one or two rows per step (7.43 vs 7.07-7.35 ms/step at one row), so the `launch/` recipes no longer set it |
| `LK_POOL_WINDOW` | pool | 32 | Expected-reuse window in steps: an evicted expert is pumped back only if it is seen again within the window; illegal values clamp to min `8`, never disables |
| `LK_POOL_INFLIGHT` | pool | `16` | in-flight DMA jobs / event-ring depth (backpressure cap = 8x; lower values false-positive under small batches) |
| `LK_POOL_LAYERS` | pool | -- | RETIRED: pool membership is global (`LK_POOL_SLOTS` > 0); the parser is dead code and per-layer mixing with GPU prefill is not supported |
| `LK_POOL_NO_STAGE` | pool | 1 | Fill path for an admitted expert. `1` (default): each NUMA fragment is DMA'd straight from the pinned host pages into its slot -- on an 8-node host that is ~16 async copies and no host copy (1.1 ms -> 0.13 ms per admission here). `0`: the coalesced staging ring (`LK_POOL_STAGE`), which memcpys the whole expert into pinned scratch first. **Which one wins depends on what the step waits on**: 0731/DSPARK prefers the default, dsv4.1/3090 the slab -- see [fill path](#gpu-pool-expert-cache) |
| `LK_VERIFY_RESIDENT` | spec | 1 | Ride-share for verify batches: the draft rows read the residency the real tokens built (resident expert -> GPU half, miss -> CPU). Their ids never enter the admission ledger unless `LK_VERIFY_PUMP` is set. `0` = draft rows always run on the CPU |

**Fill path (`LK_POOL_NO_STAGE`).** Both paths are correctness-equivalent; the winner follows what the step waits on. Measured here on the same build: DeepSeek-V4-Flash-0731 / DSPARK prefers the default per-fragment DMA (an admission's host memcpy was the cost: 1.1 ms -> 0.13 ms, gather 1669-2494 -> 0 us); DeepSeek-V4.1-Flash-1017 TP2/3090 prefers the slab -- decode median **27.0 -> 21.1 tok/s** with the per-fragment default, **26.0** back with `LK_POOL_NO_STAGE=0` (same recipe, same log, 284 vs 11 samples). The three `launch/sglang_serve_dsv41_*.sh` recipes therefore pin `LK_POOL_NO_STAGE=0`; leave it unset (default `1`) everywhere else.

### Debug / diagnostics

Knobs in this section change what the engine *prints* or are only meaningful when debugging a specific regression; none of them is needed to run the engine.

| Env var | Type | Default | Description |
|--------|------|--------|------|
| `LK_POOL_LOG` | debug | 0 | windowed `[pool]` stats line every N decode steps (0 = silent) |
| `LK_VERIFY_PUMP` | debug | 0 | `1` also feeds the draft rows' ids to the admission ledger. Only sensible when the cache can hold a whole verify batch; on a small cache it evicts what it just brought in |
| `LK_POOL_DET_SPLIT` | debug | 0 | deterministic split: publish the claimed batch and make the GPU half wait on that batch's copy fence before reading slots |
| `LK_POOL_SPLIT_PROBE` | debug | 0 | log the frozen split (claim/ready/snapshot counts) per pool layer for batches of <=32 routed ids; skipped inside capture |

### VRAM sizing (pool + GPU prefill)

Each weight type owns one static VRAM domain that holds two overlapping
demands — the expert cache (`LK_POOL_SLOTS` x layers) and the prefill working
set (wave / `LK_GPU_PREFILL_D_SLOTS` / `LK_GPU_PREFILL_SUB_M`) — and the domain
is `max(cache, prefill)`, never their sum. So `LK_POOL_SLOTS` buys cache only,
and a prefill larger than the cache grows the domain and eats KV one-for-one.
The server logs the plan at boot:

```
arena: 4.75 GiB total = max(cache 4.75 GiB [720 rows], prefill 4.32 GiB [192 map + 464 act rows])
```

How to size each knob, which wave to use (and why wave is not a performance
knob), the shipped per-recipe budgets and the pitfalls are in
**[docs/vram_sizing.md](docs/vram_sizing.md)** — read it before changing
`LK_POOL_SLOTS`, `LK_GPU_PREFILL_WAVE_EXPERTS`, `LK_GPU_PREFILL_D_SLOTS` or
`LK_GPU_PREFILL_SUB_M`.

### Speculative decode

The pool and speculative decoding now compose: `LK_VERIFY_RESIDENT` (default
`1`) lets the draft rows read the residency the real tokens built, while their
ids stay out of the admission ledger (see `LK_VERIFY_PUMP` under
[Debug / diagnostics](#debug--diagnostics)). Requires a pool
(`LK_POOL_SLOTS > 0`) and a NEXTN/DSPARK draft.

### PLE / NVMe offload

| Knob | Type | Default | Description |
|------|--------|--------|------|
| `--ple-offload-backend nvme` | ple (CLI) | pinned | The PLE table lives on disk: host RAM never holds it, only a row cache |
| `LK_PLE_BLOCK_MB` | ple | unset | MB budget for PLE prefill temporaries; unset = derived from the chunked-prefill size |
| `LK_NVME_DIRECT` | nvme | `0` | `1`: table rows are read straight from disk, so serving them cannot evict model/page cache (host RAM stays flat); `0` reads through the OS page cache |
| `LK_NVME_CACHE_MB` | nvme | `1024` | MB of hot rows kept for the direct-read path (inert unless `LK_NVME_DIRECT=1` and the feature's backend is on) |
| `LK_ENGRAM_NVME` | nvme | `0` | DeepSeek-V4.1 Engram served directly from the checkpoint file: zero host RAM for the table |
| `LK_ENGRAM_NVME_CACHE_MB` / `LK_ENGRAM_NVME_STAGING_MB` | nvme | `1024` / `128` | Engram NVMe store cache / staging budgets |

**Engram backend: prefer the host table when RAM allows.** Same build, DeepSeek-V4.1-Flash-1017 TP2/3090, real text, exclusive run: cold prefill 5.2K / 16.9K / 31.9K = **524 / 688 / 766 tok/s on NVMe** against **666 / 967 / 1091 with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`** (+27 / +41 / +42%), decode 26 vs 28 tok/s at 4K. The NVMe path is **latency-bound, not bandwidth-bound**: during a 32K prefill the disks carried at most 14.7 MB/s (median 0), so bandwidth counters say nothing. The table is ~48 GB in host RAM; use `LK_ENGRAM_NVME=1` only when that RAM is not available.

### Diagnostics (all default off; no effect on the compute path)

| Env var | Area | Description |
|--------|------|------|
| `LK_ROUTE_PROBE` | routing | per-step `[RP]` routing traces for offline calibration (requires the pool on) |
| `LK_POOL_DET_SPLIT` | pool | force the GPU/CPU split timing to be identical every step — for reproducing pool bugs; tiny decode slowdown |
| `LK_POOL_SPLIT_PROBE` | pool | print per-layer split counts for small batches (skipped inside CUDA-graph capture) |
| `LK_SPEC_PROBE` | spec | print per-step draft/verify stage timings |
| `LK_PLE_PROBE` | ple | PLE row-locality/traffic probe (`LK_PLE_PROBE_DUMP=<path>` dumps rows) |
| `LK_PP_DEBUG` | prefill | extra `[prefill]` tracing lines |
| `LK_DECODE_TS` / `LK_TS_SKIPGEMM` | host decode | decode-path timestamp trace / skip-the-gemm A/B (read by lk_moe) |
| `LK_MBOX_STOPFILE` | mbox | stop-file for orderly teardown (read by lk_moe) |
| `LK_DISABLE_QSA_SHARE` | draft | draft worker: disable QSA share (A/B isolation only) |
| `SGLANG_RS_DEBUG` | upstream | rejection-sampling traces; upstream sglang knob, kept as-is |

## Tuning recipes

All recipes assume `LK_MOE_HYBRID_ENABLED=1` (off = stock sglang).

| Mode | Parameter recipe |
|---|---|
| **1 GPU-resident layers** | `LK_GPU_RESIDENT_MOE_LAYERS=0-<last>` (format `0,1,8-9`; some models start at a non-zero layer, e.g. Step-3.5-Flash at layer 3); draft model: `LK_GPU_RESIDENT_MOE_LAYERS_SPEC=0-2` |
| **2 CPU hybrid** | default layer role, no extra parameters; tune the CPU side: `LK_THREADS=(physical cores / #GPUs)`, `LK_THREAD_BINDING=CPU_CORE`; give the saved VRAM to the KV cache (`--mem-fraction-static` as high as it fits) |
| **4 CPU + GPU Pool** | `LK_POOL_SLOTS=<from spare VRAM>` (e.g. 64; `>0` is itself the pool switch — there is no separate `LK_POOL` flag; membership covers every non-resident layer); check `LK_POOL_LOG=100` and watch `[pool] win hit=` — raise `LK_POOL_SLOTS` or `LK_POOL_WINDOW` if the hit rate is low |
| **3/5 Speculative decoding** | `--speculative-algorithm NEXTN` (or DSPARK flags) + resident draft: `LK_GPU_RESIDENT_MOE_LAYERS_SPEC`; when combined with GPU Pool, check VRAM and measure first |
| **Super Mix Prefill** | `LK_GPU_PREFILL_SUB_M=65536` + `--chunked-prefill-size 8192` (SUB_M must EXCEED the chunked size or the super-chunk wrapper never arms; the `[super-chunk]`/`Super-chunk prefill wrapper active` boot line shows CAP/inner); on 16 GB-class cards: `LK_GPU_PREFILL_MIN_TOKENS=0` + `--chunked-prefill-size 4096` |

What is actually active:

- hybrid: the lk_moe load line in the startup log;
- GPU Pool: `[pool] ... hit=` window line with `LK_POOL_LOG=100`.

Machine-level notes:

- **BIOS NUMA**: AMD EPYC NPS4 / Intel XEON SNC4; use 2,4,8 nodes (multiple of GPU count is best), up to 32.
- **Thread count**: HT on -> physical cores / GPUs; HT off -> (physical cores-2) / GPUs.
- **CPU power saving**: `LK_POWER_SAVING=1` — standby only, cuts the CPU time the idle workers hold while waiting for requests; no effect on decode power at load, and it costs wake-up latency.

---

## Benchmarks

### Performance

| Model | GPU | Prefill 32K | Prefill 64K | Prefill 128K | Prefill 256K | Decode | Spec. decode | Command |
|-------|-----|-------------|-------------|--------------|--------------|--------|--------------|---------|
| DeepSeek-V4.1-Flash-1017 | 3090 x2 | **1499 t/s** (22.0 s) | **1461 t/s** (44.9 s) | **1454 t/s** (90.3 s) | **1109 t/s** (228 s) | 30.9 t/s | 48.5-52.8 t/s | [plain](launch/sglang_serve_dsv41_tp2_3090.sh) · [DSPARK](launch/sglang_serve_dsv41_tp2_3090_dspark.sh) |
| DeepSeek-V4-Flash-0731 | 3090 x2 | **1148 t/s** (28.7 s) | **1103 t/s** (59.4 s) | — | — | — | 43.9-46.5 t/s | [DSPARK](launch/sglang_serve_0731_dspark_tp2_3090.sh) |
| Qwen3.8-Flash-Next | 3090 x2 | **2214 t/s** | **2524 t/s** | — | **2333 t/s** | 54.8 t/s | 72.2-74.3 t/s | [plain](launch/sglang_serve_qwen38_tp2_3090.sh) · [NEXTN](launch/sglang_serve_qwen38_mtp_tp2_3090.sh) |
| Qwen3.8-Flash-Next | 5060Ti x2 | **2189 t/s** | **2066 t/s** | — | — | 37.2 t/s | 46.2-49.5 t/s | [plain](launch/sglang_serve_qwen38_tp2_5060ti.sh) · [NEXTN](launch/sglang_serve_qwen38_mtp_tp2_5060ti.sh) |
| Qwen3.8-Flash-Next | 3090 x2 + 5060Ti x2 (TP4) | **1249 t/s** | — | **1247 t/s** | — | — | 64.0 t/s | [NEXTN](launch/sglang_serve_qwen38_mtp_tp4.sh) |
| Qwen3.8-Flash-Next | 3090 x2 + 5060Ti x2, PLE on NVMe | **1166 t/s** | — | — | — | — | 66.6 t/s | [NVMe](launch/sglang_serve_qwen38_mtp_tp4_nvme.sh) |
| GLM-5.3-Flash-NVFP4 | 3090 x2 | **892 t/s** | — | — | — | 20.6 t/s | — | [plain](launch/sglang_serve_glm53_tp2_3090.sh) |
| Qwen3.6-35B-A3B | 3090 x2 | **6026 t/s** | — | — | — | 126.7 t/s | — | [plain](launch/sglang_serve_qwen36_tp2_3090.sh) |

All rows measured on one machine with **lsglang 1.7.3 + lk_moe 2.8.4**: 2x AMD EPYC 7642
(96 cores / 192 threads, AVX2 only), 16-channel DDR4-3200, 1 TB, CUDA 13.0 / torch 2.14.1 /
driver 610.57.04. **Prefill** = client wall time of one request at that input length (bracketed
figure = wall time); **Decode** = 256 output tokens after a short prompt; **Spec. decode** =
DSPARK (DeepSeek) or NEXTN (Qwen3.8) on the same recipe, as the range over repeated
runs (low-median to best). `—` = not measured for that
configuration, not a failure. A DSPARK run's prefill is within ~3% of its plain twin.

Experimental DeepSeek-V4.1 storage option: exact NVMe Engram lookup (`LK_ENGRAM_NVME=1`; host-table fallback `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`; budgets `LK_ENGRAM_NVME_CACHE_MB` (default 1024) / `LK_ENGRAM_NVME_STAGING_MB` (default 128)).

### Supported models & quant formats

Models:

Legend: ✅ tested · `ok` supported, not yet tested · blank = to be filled.

| Model | sm80 | sm86 | sm89 | sm90 | sm100 | sm120+ | Launch | NVMe |
|-------|:----:|:----:|:----:|:----:|:-----:|:------:|--------|------|
| [Qwen3.8-Flash-Next](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_qwen38_tp2_3090.sh)  | [PLE ON NVME](launch/sglang_serve_qwen38_mtp_tp4_nvme.sh) |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_dsv41_tp2_3090.sh) · | [ENGRAM ON NVME](launch/sglang_serve_dsv41_tp2_3090_nvme.sh) |
| [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_dsv41_tp2_3090.sh) | [ENGRAM ON NVME](launch/sglang_serve_dsv41_tp2_3090_nvme.sh) |
| [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_0731_tp2_3090.sh) | — |
| [GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_glm53_tp2_3090.sh) | — |
| [Qwen3.6-35B-A3B](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | ok | ✅ | ok | ok | ok | ✅ | [launch](launch/sglang_serve_qwen36_tp2_3090.sh) | — |
| Qwen3-Coder-Next / 30B-A3B | | | | | | | | |
| gemma-4-26B-A4B-it | | | | | | | | |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | | | | | | | | |
| Qwen3.5-35B-A3B | | | | | | | | |
| Qwen3.5-122B-A10B | | | | | | | | |
| Qwen3.5-397B-A17B | | | | | | | | |
| Qwen3-VL-30B | | | | | | | | |
| MiniMax-M2.7 / 2.5 / 2.1 | | | | | | | | |
| GLM-5.2-NVFP4 | | | | | | | | |
| GLM-5.1 / 5.0-FP8 | | | | | | | | |
| GLM-4.7(-Flash) / 4.6V | | | | | | | | |
| Kimi k2.6 / k2.5 | | | | | | | | |

Quant formats:

| Format | Status |
|--------|--------|
| BF16 | tested |
| FP8 | tested |
| NVFP4 — compressed-tensors / modelopt | tested |
| MXFP4 | tested |
| AWQ INT4 (symmetric, per-group) | tested |

The MoE computation is delegated to lk_moe's quantized kernels; attention, shared experts and
all other layers run on the original sglang GPU path.

---

## Integration & development

lk_moe integrates into any inference framework as a drop-in MOE backend. The integration surface
(layer roles, feature gates, config fields, weight placement) is spread over the model, pool and
scheduler code paths and moves with them, so this repository no longer ships a single guide for it:
read the launch recipes under `launch/` for the supported configurations, and `lk_moe`'s README for
the knobs.

---

## License

lk_moe is provided under a proprietary software license agreement. See [LICENSE](./LICENSE) for
terms and conditions.
