# lsglang 1.7.6 (from 1.7.5)

Needs lk_moe 2.8.5.

```
pip install lsglang==1.7.6
```

## Correctness fixes

- **Banded auxiliary captures now tile the column they were built for.** A band that
  skipped rows left allocator bytes in the draft model's input, and one that overlapped
  rewrote rows that were already correct; both are refused instead of silently handing
  the model wrong hidden states.
- **DeepSeek-V4.1 with DSPARK and bounded replay boots again.** Those recipes used to
  build a capture packer sized for CAP rows, then refuse to write the replay tail; the
  layer loop and the collector now read the same tail predicate.

## QSA K/V host offload (Qwen3.8)

`--enable-qsa-kv-offload` keeps the full-attention K/V of the QSA layers in pinned host
memory and holds only the blocks the indexer selects on the card, in a per-request hot
cache. On 2x RTX 3090 the KV pool drops from 19.6 GB to 16.2 GB per card at the same
262,144 tokens, and the logical pool can grow to 524,288. Weights, indexer and prefix
reuse are untouched, and the packed K/V is bit-identical to the fully device-resident
path (checked against the pool gather).

```
--enable-qsa-kv-offload
--qsa-kv-offload-host-size 8        # GB per rank
--qsa-kv-offload-hot-blocks 2048    # draft tokens x 512 with MTP, otherwise 1024
```

It cannot be combined with `--enable-hierarchical-cache`, and it does not offload the
Mamba state, so a hybrid model's prefix reuse is still bounded by the Mamba pool.

## Recipes

- **Every recipe pins the tree core to Python** (`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`).
  The Rust tree core is compiled from source on demand whenever a checkout and cargo are
  present, which silently changed the backend between machines and between days for the
  same version.
- **Hybrid linear-attention recipes enable the hierarchical cache** (Qwen3.8, Qwen3.6,
  GLM-5.3, ratio 4). A radix hit there needs a donated Mamba state; the 11-slot device
  pool cannot hold one per long context, and the host tier keeps those states
  recoverable instead of silently recomputing the whole prefix.
- **Qwen3.6's MTP recipes declare the KV dtype explicitly** (`--kv-cache-dtype fp8_e4m3`
  with `--speculative-draft-kv-cache-dtype fp8_e4m3`) so the draft pool matches the
  target's fp8 cache; the hierarchical cache refuses to boot otherwise.
- **DeepSeek-V4.1 recipes enable the same hierarchical cache** (ratio 4). Mind the host
  budget: one TP rank already takes about 338 GB for the CPU expert pool, so the tier
  fits only when the machine is not also running another engine.
- New: `SGLANG_DEBUG_MAMBA_CHECKPOINT=1` logs what a finished request donates to the
  radix cache and what each match reuses. Off by default.

## Measured quality

Same gate, same frozen corpus, 2x RTX 3090: Qwen3.6 MTP 8.105 (8.108 with the
hierarchical cache), 0731 DSpark 6.983, GLM-5.3 6.503, all `GATE PASSED`.

# lsglang 1.7.5 (from 1.7.4)

Needs lk_moe 2.8.5.

```
pip install lsglang==1.7.5
```

The deep_gemm build with the SM120 kernels is not a package dependency: PyPI refuses
requirements that point at a URL, so installing this package cannot pull it for you.
Install it once by hand (Python 3.12, linux x86_64) — skip this if it already works:

```
pip install https://github.com/guqiong96/Lsglang/releases/download/lsglang-v1.7.4/deep_gemm-2.8.0+local-cp312-cp312-linux_x86_64.whl
```

On an environment that still has `sgl-deep-gemm` from an older release, remove it
first — both provide the `deep_gemm` module:

```
pip uninstall -y sgl-deep-gemm
```

## FP4 x FP4 prefill on SM120

The 2x RTX 5060 Ti recipes now run prefill with 4-bit activations against 4-bit weights:
packed codes and scale factors go straight to the card's block-scaled GEMM instead of
being dequantized to bf16 first. It is one knob, resolved per checkpoint:

```
LK_GPU_PREFILL_QUANT=w4a4     # auto | w4a4 | w4a8 | w4a16 | w8a8
```

Only an engine that carries the W4A4 arm can serve that request;
`LK_GPU_PREFILL_ENGINE` picks the engine (`cutlass` by default):

| engine | `w4a4` on SM120 | checkpoints |
|--------|-----------------|-------------|
| `cutlass` | yes | NVFP4 and MXFP4 — the new arm in this release |
| `triton` | — | never had it |
| `flashinfer` | yes | NVFP4 only |

`auto` is the default and keeps each checkpoint on a mode it can run. A mode the engine
does not implement for these weights degrades to the safe one and says so at boot, so an
older `lk_moe` or an older card costs you speed, not the boot. The boot line reports the
mode that is actually live; no W4A4 line means the recipe asked for it and did not get
it.

## Thinking conversations keep their cache

Two prefix-cache faults, both only visible inside a real conversation:

* A thinking conversation stopped extending its cache at the first assistant turn -- the
  cached length froze while the prompt kept growing, so the hit rate slid as the chat got
  longer (#36). Reasoning the client sends back is now replayed as the turn that produced
  it: over four turns on DeepSeek-V4-Flash-0731, 0, 0, 0 cached tokens became 256, 512,
  512, with non-thinking conversations unchanged. The V4.1 format renders the same turn
  the same way and carries the same fix, covered by the same tests.
* A conversation that started with a partial hit on another conversation's prefix
  contributed nothing of its own, so the cache never grew for it, however many turns it
  ran (#37).

Both are covered by CPU tests, each with a case that must not match; the thinking fix is
additionally measured on hardware, the #37 one is not yet measured with two live
conversations on this build.

## 2x RTX 5060 Ti serves the 0731 checkpoint

The recipe boots and serves: 262,144 tokens of KV, **792 t/s** prefill at 32K and
**862 t/s** at 128K, **56-69 t/s** decode sampled from live OpenWebUI turns at
3.5K-11.3K context. A 256K prompt still does not fit two 16 GB cards, so treat 128K as
the ceiling there and use a 3090 recipe for 256K.

## Long-prompt memory

Scoring memory for a long prompt used to be counted when the plan was made and taken
later; it is now held from the start, so a long prefill no longer dies on memory its own
plan had already counted. What a long prompt costs is unchanged.

# lsglang 1.7.4 (from 1.7.3)

Needs lk_moe 2.8.4.

```
pip install lsglang==1.7.4
```

The deep_gemm build with the SM120 kernels comes with the package (Python 3.12,
linux x86_64), and can also be installed by hand:

```
pip install https://github.com/guqiong96/Lsglang/releases/download/lsglang-v1.7.4/deep_gemm-2.8.0+local-cp312-cp312-linux_x86_64.whl
```

On an environment that still has `sgl-deep-gemm` from an older release, remove it
first — both provide the `deep_gemm` module:

```
pip uninstall -y sgl-deep-gemm
```

## Faster decode on SM120

DeepSeek-V4.1 decode now runs the card's own FP8 paths end to end — the dense
GEMMs, the wo_a absorb and the sparse indexer — instead of portable fallbacks.
The published 32x32 block-scale checkpoints are used as they are, and keeping
wo_a in FP8 leaves room for a 256K context on 16 GB cards.

## Prefix cache works with the default SWA replay

Long conversations reuse their prefix again instead of re-prefilling it. Reuse is
only enabled where the model can restore its state; where it cannot, the previous
behaviour is kept, so answers stay correct. Thanks to @Orange4483 (PR #33) for the
report and the first fix.

## 2x RTX 5060 Ti recipe

New: `launch/sglang_serve_dsv41_tp2_5060ti.sh` — DeepSeek-V4.1 on the 5060 Ti pair
with 256K context.

## Notes

- Retaining the SWA window behind the replay boundary is back-ported from
  sgl-project/sglang#46008, still open upstream.
- `SGLANG_SWA_REPLAY_REBUILD_WINDOW` is a debug switch, off by default.
- Perplexity 5.49 here against 5.19 in the 1.7.3 note: those were different builds,
  not a regression from these changes.

# lsglang 1.7.3 (from 1.7.2)

Needs lk_moe 2.8.3.

Measured on DeepSeek-V4.1-Flash-1017: 2x RTX 3090 24 GB (TP2), 2x AMD EPYC 7642
(96 cores / 192 threads, AVX2 only -- Zen 2 has no AVX-512), 1 TB DDR4,
8 NUMA nodes, CUDA 13.0 / torch 2.14.1 / driver 610.57.04.

## Long context works

| context | prefill | throughput |
|---|---|---|
| 32K | 22.0 s | 1499 tok/s |
| 64K | 44.9 s | 1461 tok/s |
| 128K | 90.3 s | 1454 tok/s |
| **256K** | **228 s** | **1109 tok/s** |

256K (253,565 tokens) peaks at 24.1 GB per card on a 24 GB 3090. Decode:
45-53 tok/s at 4K with DSPARK, 41 tok/s measured at 2-4K on a real request.

## Correctness

- **Per-token scores (prompt logprobs) were wrong on long prompts**; correct
  now: PPL 38,593 -> 5.24, NLL 5.43 -> 0.039 at 440 rows. Same recipe.
- **The first cache-hit request after a restart** no longer stalls for minutes:
  593 s -> 1.6-2.6 s (reported on 4090 + RTX PRO 5000, TP2, DSPARK, HiCache).
  Thanks to @Orange4483 (PR #32).
- Quality gate on 2x3090 / DeepSeek-V4.1: PPL 5.19, NLL 0.04/0.04/0.02/0.01.

## Recipes

- **DeepSeek-V4.1 + DSPARK** (`launch/sglang_serve_dsv41_tp2_3090_dspark.sh`):
  45-53 tok/s at 4K against 26-28 plain, with 4.65 GB left per card.
- **Triton prefill variants** (`*_triton.sh`): same recipe, higher quality
  (PPL 5.02 against 5.19) at ~30% lower prefill throughput. Use it to check
  whether a quality question is a kernel or a recipe problem.
- **`--max-prefill-tokens` is gone** from the dsv4.1 recipes. It caps one batch
  while `LK_GPU_PREFILL_SUB_M` only splits inside a batch, so a `SUB_M` larger
  than the cap does nothing: raise the cap only when you raise `SUB_M`, and
  never past it. Dropping it lets a 32K prompt run as two batches instead of
  three, which is faster.

## Build

`docs/integration.md` is gone: the integration surface follows the model, pool
and scheduler code, so a single guide aged into a liability. Use the recipes
under `launch/`.
