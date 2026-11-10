# lsglang 1.7.5 (from 1.7.4)

Needs lk_moe 2.8.5.

```
pip install lsglang==1.7.5
```

The deep_gemm build is unchanged from 1.7.4, so nothing to reinstall if that already
works; it ships with the package (Python 3.12, linux x86_64) and can also be installed
by hand:

```
pip install https://github.com/guqiong96/Lsglang/releases/download/lsglang-v1.7.4/deep_gemm-2.8.0+local-cp312-cp312-linux_x86_64.whl
```

On an environment that still has `sgl-deep-gemm` from an older release, remove it
first — both provide the `deep_gemm` module:

```
pip uninstall -y sgl-deep-gemm
```

Upgrade `lk_moe` before starting a 2x RTX 5060 Ti recipe: an older one leaves that path
unusable and the server falls back quietly, which then surfaces as a long prompt failing
to start rather than as an error at boot.

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

The recipe boots and serves: 262,144 tokens of KV, the card's own block-scaled prefill
active, **792 t/s** prefill at 32K and **862 t/s** at 128K, **56-69 t/s** decode sampled
from live OpenWebUI turns at 3.5K-11.3K context. A 256K prompt still does not fit two
16 GB cards, so treat 128K as the ceiling there and use a 3090 recipe for 256K.

## Long-prompt scoring memory is held rather than promised

The paged indexer's score matrix used to be counted in a long prompt's plan when the plan
was built, and allocated later, once per piece, when that piece was scored. It is now
taken once and kept by the process; reusing it was checked to give exactly the scores a
fresh allocation would. That removes one way a long prefill can die on an allocation its
own plan had already accounted for. It does not reduce what a long prompt costs -- on two
16 GB cards a 256K prompt is still out of reach, and 128K is the largest shape measured
here (**862 t/s**).

A prompt split into pieces now also captures the speculative draft's inputs for every
piece instead of only the first; the 32K and 128K numbers above were taken with DSPARK
enabled.

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
