# lsglang v1.6.6 — Host Pins Fall Back, They No Longer Fail

**Base:** upstream sglang main @ `06342acad2`; **lk_moe v2.7.4**.

## What Changed

- **Every host-memory pin can now fall back instead of failing** (PR #30):
  the engram host table and the HiCache host KV pools register their NUMA
  arenas with one `cudaHostRegister` call where the kernel allows it, and
  granule- / aligned-chunk-sized registration takes over where it does not
  (pre-6.13 kernels, `expandable_segments` setups). A pin that cannot be
  established says so on one line instead of dying or running silently
  unpinned. lk_moe v2.7.4 carries the same rule in its C++ helper.
- **`get_device_name()` answers for the current device** when called without
  an index; the three-key discipline (locate by id, gate by cap, look the
  name up in a table) is now enforced at every call site that used to guess
  from a name string.
- **Group configs gather over the CPU (gloo) group**, so heterogeneous-rank
  launches cannot wedge the CUDA collective during init.
- **The HiCache storage hash JIT finds conda's openssl** instead of the
  system libcrypto.
- **All serve scripts pass `--sleep-on-idle`**: idle schedulers park.
- **`--disable-custom-all-reduce` retires from the 3090 TP2 serve scripts**:
  custom allreduce ran clean on the production box (2000+ decode steps over
  several windows, plus a 49K-token dspark prefill with it on).
- **The 5060Ti and 0731-dspark scripts carry the measured boot shapes**:
  5060Ti qwen38 at frac 0.88 / KV 65536; 0731 dspark at super-chunk 16384
  (8192 starved prefill staging on this box) / KV 262144.
- Optional **radix tracing of host hits and storage-prefetch outcomes** for
  cache debugging (log-only, default off).
- lk_moe v2.7.4 companion: `frozen_chunk` thaws every pool it froze when a
  mid-freeze error fires (the partial-freeze leak the teardown-deadlock
  audit convicted), plus a faster MXFP4 gemv (fp16-LUT decode, e8m0 folded
  into A) -- +32-36% in the kernel microbenchmark; the end-to-end decode
  gain measured on production shapes is well short of that.

## Verified

- 8070 production (2x3090, qwen3.8): zero pin WARN lines at boot = every
  one-shot register succeeded on this box; 2000+ decode steps, zero hangs.
- Fallback ceiling measured with `LK_PIN_FORCE_FAIL=1`: 6.5k-token prefill
  8.48 s -> 9.62 s (-13.5%) is what granule-first costs when it fires.
- PR #30 chain merged with the author's attribution and ran clean on the
  3090 dsv4.1 NVMe production box.
- 49K-token dspark prefill with superchunk loops (this tree, 3090):
  `wrapper active` + LOOP lines present, no stall, custom allreduce on.

## Commits

Everything added since v1.6.5: `git log --oneline lsglang-v1.6.5..HEAD`.

# lsglang v1.6.5 — The NVMe Engram Can No Longer Fail Silently

**Base:** upstream sglang main @ `06342acad2`; **lk_moe v2.7.3**.

## What Changed

- **The NVMe gather gate never releases on timeout** (lk_moe v2.7.3): the
  300 ms release handed merges stage bytes from an older ring lap -- stale
  rows entered the model as embeddings (garble), and lost tickets made every
  later gate burn the full budget (hang-like stalls). The soft limit now only
  raises a sticky slow-disk warning; a fault that cannot resolve is loud.
- **Stale rows emit zero, never the slot's previous bytes** (merge kernel).
- **Faults are checked on every processed tick**, not one tick in sixteen.
- **The `[engram-nvme]` health line actually prints** (_tick queried an event
  recorded microseconds ago -- it never read back; graph replay also skipped
  the old %1024 gate).
- **`LK_NVME_CACHE_MB` is now a whole-machine budget**, divided over
  (engram layers x tp) stores at attach; the 256GB-class NVMe target no
  longer silently multiplies it x4.

## Verified

- Store-level A/B on a 5060Ti with the real 1017 shard: legacy release =
  82,304 silently wrong rows, zero errors; forever-wait = zero wrong rows,
  warnings visible.
- 3090 model-level A/B vs the host-table arm: long-prefill and multi-turn
  opencode shapes bitwise identical; gsm8k cold 357 s (was 454), decode
  within 2% of all-RAM, 94% cache hits.

## Commits

Everything added since v1.6.4: `git log --oneline lsglang-v1.6.4..HEAD`.

# lsglang v1.6.4 — Smaller Super-Chunk Transients

**Base:** upstream sglang main @ `06342acad2`; **lk_moe v2.7.2** (unchanged).

## What Changed

- **`_combine` runs per sub-chunk** instead of allocating the whole chunk at
  once: transient 320 -> 190 MiB at 16,384 rows, +0.4% of a forward, bitwise
  identical (skipped on sm90/sm100, where the fused kernel would change).
- **Indexer `scores` peak: 4.11x -> 1.05x of its budget** -- `relu_()` in place
  and rows sized against the measured 3.1x peak.
- **DeepSeek-V4.1 scripts: `SUB_M=12288`** (was 8192) and `LK_POOL_LOG=0`. A
  250k-token request now runs in 21 chunks / 676 s with no OOM.

`expandable_segments` is gone from every launch script: with the card full it
only logged `memory mapping failed ... (free: 6 MiB)` -- physical pages, not
fragmentation.

## Commits

Everything added since v1.6.3: `git log --oneline lsglang-v1.6.3..HEAD`.

# lsglang v1.6.3 — The Super-Chunk Now Fits Its Own Reservation

**Base:** upstream sglang main @ `06342acad2`; **lk_moe v2.7.2** (unchanged).

## Problems This Release Fixes

- **`LK_GPU_PREFILL_SUB_M` was only partly honoured on sliding-window models.**
  The super-chunk raises the per-forward token budget, but the sliding-window
  pool a chunk is measured against was reserved from `--chunked-prefill-size`
  alone. On DeepSeek-V4.1 that asked for 16,384-token chunks and delivered
  5,376-token ones -- a third of the configured size, with no log line saying
  why. The reservation now covers the larger of the two (6,912 -> 37,632 slots at
  `SUB_M=16384`, 20,736 at the 8,192 the scripts now ship). Nothing to set; no
  change where the GPU-prefill tier is off, and none on models with no
  sliding-window pool (Qwen3.8, GLM-5.3).

## What You Need To Change

Nothing to set. Two things to know if you run a sliding-window model on a 24 GB
card with `LK_GPU_PREFILL_SUB_M` set:

- The larger sliding-window pool is taken from the same budget as the
  compressed-KV pools. On the DeepSeek-V4.1 TP2 setup it costs 0.82 GB, and the
  16,384-token chunks triple the per-forward activations on top of that: the run
  died at `full token usage 0.38` with a 160 MB activation (`16,384 x 5,120 x 2`)
  it could not get. The launch scripts therefore ship `SUB_M=8192`, which halves
  both.
- Do not buy the headroom with `--mem-fraction-static`. That fraction is the
  budget the KV pools live in: at 0.74 the DeepSeek-V4.1 setup has 1.06 GB left
  for pools that need 1.25 GB, so the boot stops with
  "leaves no room for the full KV pool" instead of running smaller. The launch
  scripts raise `--max-total-tokens` to 294,912 instead, which costs 54 MB and
  leaves room to generate after a 256k-token prompt (memory alone allows
  451,328 tokens there, so 256k was never the constraint -- the flag was).

## Known Issues

- `enable_dynamic_chunking` on a pipeline-parallel group sizes chunks from a
  profiled model the reservation cannot see, so a sliding-window model in that
  configuration can still be clamped. Not reachable at `--pp-size 1`.

## Commits

Everything added since v1.6.2: `git log --oneline lsglang-v1.6.2..HEAD`.


# lsglang v1.6.2 — Long-Context Correctness, Deterministic Output, Faster Prefill

**Base:** upstream sglang main @ `06342acad2` (v1.6.0 base: `583fada954`);
**lk_moe v2.7.2**.

Everything on top of upstream is the lk_moe integration and its upgrades.

## Problems This Release Fixes

- **Long prompts no longer stall silently.** On sliding-window models (in this
  tree: DeepSeek-V4.1), the longest prompt you could send was decided by an
  internal sliding-window slot budget, not by `--max-total-tokens` or the model
  context. At `--max-total-tokens 262144` with `--swa-full-tokens-ratio 0.1` a
  request past **25,856 tokens** was accepted, prefilled up to that point, and
  then waited forever -- no error, no log line, the client just hangs. Window
  slots are now reclaimed while the prompt is still being prefilled, so the
  ceiling is the configured context budget again. No knob to set -- it applies
  on upgrade, and it only affects sliding-window models.
- **Repeated / runaway output and run-to-run differences on Qwen3.8 and
  DeepSeek V4.x.** Async host-to-device copies could read a host buffer that
  had already been recycled underneath them. Every such site now stages
  through a deep parked ring, which removes that whole family of symptoms
  (greedy decoding producing loops, the same prompt answering differently).
- **Servers that deadlocked at startup on mixed GPU models (TP4, 3090 +
  5060Ti).** CUDA-graph capture is now decided by a group-consensus arch check
  instead of per rank, so a heterogeneous group either captures together or
  runs eager together.
- **The first request paid a multi-second JIT stall** on DeepSeek V4.x. The
  sampling kernel and the small decode/prefill shapes are now prewarmed at
  model load.
- **The determinism switch did nothing.** `LK_POOL_DET_SPLIT=1` looked armed but
  its GPU-side wait was never actually reached, so results were no more
  reproducible than with the knob off. It now takes effect -- and fails loudly
  at startup on a build that cannot honour it, instead of pretending.
- **Silently missing expert contributions.** When a decode step fell outside
  the expert pool's coverage, the CPU half could skip experts the GPU half had
  not actually computed for that step. In the worst case a token's MoE output
  was missing half of its experts, with nothing logged.

## What Got Better

- **Prefill throughput on long prompts.** The CUTLASS grouped-GEMM engine is the
  default (set `LK_GPU_PREFILL_ENGINE=triton` to opt out). It plans whole slabs
  at a time, pipelines the activation region through a bounded window
  (`LK_GPU_PREFILL_D_SLOTS`) instead of allocating scratch off the books, and
  sizes its waves automatically (`LK_GPU_PREFILL_WAVE_EXPERTS=-1`) -- measured
  here at 6 waves for a 384-expert model with no change in output vs the CPU
  reference (24/24 retrieval through 13k tokens, 24.6k tokens at the old wall).
- **Decode throughput on SM8x cards** (3090 class): block-FP8 group-32 Marlin
  W8A16 kernels, prewarmed before graph capture, plus a fused inverse-RoPE path
  in the DeepSeek-V4 decode attention.
- **Deterministic FP8** (`SGLANG_DETERMINISTIC_FP8`): fixed split-K and a bypass
  of the SM12x code path, so FP8 layers stop reordering reductions between runs.
- **Weight loading without host-RAM residency**: DeepSeek-V4.1 Engram and
  Qwen3.8 PLE tables stream from NVMe into the VRAM hot cache; boot cost no
  longer scales with checkpoint size. Each TP rank now reads exactly the rows it
  owns.
- **Cache correctness**: mid-chunk radix-tree insertion is disabled by default
  (an aborted request could leave freed pages referenced); recycled Mamba
  ping-pong slots are cleared lazily so stale state cannot leak into the tree;
  sparse-attention top-k ties are broken by index order so retrieval is stable.
- **Upstream**: sglang main folded in through `06342acad2` -- unified memory pool
  (asymmetric KV rows, fused draft / sliding-window / recurrent-state layers),
  scheduler and MemCache refactors, Rust gRPC/router work, and the DeepSeek-V4.1
  late-layer pinned-row perf path.

## What You Need To Change

Nothing to stay on the safe path: the defaults are conservative and the GPU
prefill tier stays off until you switch it on. To reproduce what this build was
validated with (any of the 16 launch scripts already encodes it):

```sh
LK_GPU_PREFILL_MIN_TOKENS=1024      # this value is the switch: 0 (default) = CPU prefill
LK_GPU_PREFILL_SUB_M=16384          # super-chunk size; the value is also the switch
LK_GPU_PREFILL_ENGINE=cutlass       # only matters when the tier is on
LK_GPU_PREFILL_D_SLOTS=16384        # pipelined act window (0 = one window per pass)
LK_POOL_SLOTS=12                    # resident expert slots per MoE layer
```

| Variable | Default | Effect |
|---|---|---|
| `LK_GPU_PREFILL_MIN_TOKENS` | **0 (tier off)** | Batch size at which GPU prefill engages; setting it > 0 is what turns the tier on. |
| `LK_GPU_PREFILL_SUB_M` | **0 (off)** | Super-chunk for long prefills; > 0 enables the super-chunk loop. |
| `LK_GPU_PREFILL_ENGINE` | `cutlass` | Which GPU-prefill implementation a powered tier uses. `triton` = legacy staging path (fp32 activations, the reference the CPU tier agrees with). Inert when the tier is off. |
| `LK_GPU_PREFILL_D_SLOTS` | **0 = one window per pass** | > 0 pipelines reorder/gemm/fuse per window inside each pass, cutting activation VRAM. Ignored by triton. |
| `LK_GPU_PREFILL_WAVE_EXPERTS` | `-1` = auto | Wave width: `0` single-shot, `>0` explicit cap, `-1` size it from the slab. Ignored by triton. |
| `LK_GPU_PREFETCH_WINDOW` | **1** | Staging ring depth: `2` overlaps host-to-device with the gemm -- only pays off on fast PCIe. Ignored by cutlass. |
| `LK_POOL_SLOTS` | 0 | Resident expert slots per MoE layer; the value is the switch, and it also turns on the concurrent CPU+GPU decode half unless `LK_POOL_DECODE=0`. |
| `LK_POOL_DET_SPLIT` | 0 | Freeze the split before the GPU half reads its slots. Now actually enforced; startup aborts if the build cannot honour it. |

Behaviour changes worth knowing: `LK_GPU_PREFILL_BORROW_MB` is gone (row
borrowing is no longer budgeted), and enabling `LK_POOL_DET_SPLIT` on an older
build now stops the boot with an error rather than running unenforced.

## Known Issues

- With speculative decoding and the expert pool both enabled, a decode batch
  whose row count happens to divide evenly by the draft width can be read as one
  request with draft rows instead of several requests. Not reachable without
  speculative decoding.
- DeepSeek-V4.1 dynamic split at 512-token tracks is still not reproducible; the
  prefill down-GEMM keeps a ~3e-3 bf16 hi/lo tail.
- `reasoning_content` is dropped from `tool_calls` responses when speculative
  decoding is on.
- Three speculative-decoding kernels still JIT on the first request;
  `--warmups prefill_shapes` absorbs them.

## Commits

Full list of what we added on top of the upstream base:
`git log --oneline --first-parent 583fada954..HEAD`. The ones that matter:

```
2924198f55 fix(moe): do not log an ERROR when lk_moe is simply switched off
6e1c0621de chore(deps): lk_moe 2.7.1 -> 2.7.2
d833b41d8f docs(releasenotes): user-facing problems, env table, commit delta
1ae927a69f feat(swa): reclaim out-of-window SWA during prefill by default
d8e24fb154 chore(release): v1.6.2
f4e8f864d3 chore(launch): default the cutlass engine, pin LK_GPU_PREFILL_D_SLOTS=16384
997a02ac9c fix(moe): make the pool-decode fence real, clear the gpu-half flag
a27d5ea5cc feat(swa): reclaim out-of-window SWA during chunked prefill (opt-in)   # superseded by 1ae927a69f
1945a0a71f fix(engram): serve each TP rank its exact shard_range rows from NVMe
4f97ab6484 debug(lk_moe): LK_PP_DEBUG=1 compares the GPU prefill tier against the CPU tier per layer
972824c91d fix(mem_cache): default-disable mid-chunk radix insert for all models (abort dangles freed pages, #38319)
cbca589a66 feat(lk_moe): LK_GPU_PREFILL_D_SLOTS -> cfg.prefill_d_slots (cutlass act slice window)
60f6658524 fix(lk_moe): default LK_GPU_PREFILL_ENGINE back to triton   # superseded by f4e8f864d3
76bd58545f chore(dsv4.1): drop obsolete dspark TP2/TP4 launch scripts
```

lk_moe v2.7.1 -> v2.7.2: CUTLASS fp32 intermediates and the
single N=2I gate/up grouped gemm, the slot-window activation slice pipeline,
per-window padding fill, **every wave's staging rows published** (this is the
384-expert long-context fix), the fp32 prefill accumulator booked inside the
VRAM ledger, a geom reproducer that sweeps wave / d_slots / pool_slots, and an
atomic `.so` republish that never touches a mapped library.


# lsglang v1.6.0 — GPU Expert Pool, NVMe Weight Offload, Super-Mix Prefill

**Base:** upstream sglang @ `583fada954` (main, 2026-09-27)
**lk_moe:** v2.6.0
Everything on top of the base is the lk_moe integration and its upgrades.

## What This Release Solves

- **MoE decode is no longer limited to whichever device holds the experts.** The
  GPU expert pool keeps hot experts resident on the GPU and streams the rest
  from the host, and a single decode step serves **both halves at the same
  time**: resident experts through the GPU path, the remainder on CPU. The
  mailbox decode transport also removes the silent hangs seen on slow hosts at
  TP4.
- **Multi-hundred-GiB embedding tables no longer have to fit in host RAM.**
  NVMe weight offload streams Engram (DSV4.1) and PLE (Qwen3.8) rows from disk
  into the VRAM hot cache through a pinned ring, so boot time no longer scales
  with checkpoint size -- a 475 GiB checkpoint boots on a host with far less
  free RAM than that, and only a small part of its tables is ever resident.
- **Long prefills (~20k+ tokens) no longer saturate memory** — Super-Mix Prefill
  processes long prompts through one expert-union staging with a sub-M loop,
  bounding block-scale temporaries while keeping the GPU pipeline saturated.

## Model Support

All models run on SM80+. Every model supports the GPU Expert Pool and Super-Mix
Prefill; models with big embedding/Engram tables additionally support NVMe
Weight Offload.

| Model | GPU Expert Pool | Super-Mix Prefill | NVMe Weight Offload |
|---|---|---|---|
| Qwen3.8-Flash-Next (NVFP4) | ✓ | ✓ | ✓ (PLE table) |
| DeepSeek V4.1 (FP8) | ✓ | ✓ | ✓ (Engram table) |
| GLM-5.3 Flash (NVFP4) | ✓ | ✓ | — |
| DeepSeek V4 Flash (0731) | ✓ | ✓ | — |

## Environment Variables & Tuning

| Variable | Default | What it does |
|---|---|---|
| `LK_POOL_SLOTS` | 0 | GPU expert pool: `>0` turns on concurrent CPU+GPU decode (e.g. `=64` for 64 resident slots per MoE layer), `0` = staging prefill only. |
| `LK_POOL_DECODE` | auto | Run GPU decode half alongside CPU. Default on when pool is on. |
| `LK_GPU_PREFILL_MIN_TOKENS` | 1024 | Minimum prompt tokens to use GPU prefill. |
| `LK_GPU_PREFILL_SUB_M` | 16384 | Sub-M loop for chunked prefill (`0` = per-chunk staging). |
| `LK_GPU_PREFETCH_WINDOW` | 0 | Overlap H2D copy with GEMM (`2` on fast PCIe). |
| `LK_ENGRAM_NVME` | 0 | Enable NVMe offload for DeepSeek-V4.1 Engram tables (`1` = on). |
| `--ple-offload-embedding` | — | Enable NVMe offload for Qwen3.8 PLE tables. |
| `--ple-offload-backend nvme` | — | Select NVMe backend for PLE offload. |
| `LK_NVME_DIRECT` | 0 | O_DIRECT reads bypassing page cache, backed by a pinned direct-mapped host cache (`LK_NVME_CACHE_MB`, `LK_NVME_STAGING_MB`). |

## Upgrade Notes

- Environment naming unified under `LK_*`; legacy `LVLLM*` names removed (an
  old `LVLLM_*` setting is now simply ignored -- move it to the `LK_*` name).
- Tuning parameters have been optimized and unified — see the
  [README tuning guide](https://github.com/guqiong96/Lsglang/blob/lsglang-v1.6.0/README.md)
  for the full `LK_*` reference and recommended presets per scenario.
