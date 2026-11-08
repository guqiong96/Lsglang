# lsglang 1.7.4 (from 1.7.3)

Needs lk_moe 2.8.4.

## 2x RTX 5060 Ti (SM120)

DeepSeek-V4.1 now serves on the 5060 Ti pair with the card's own paths: the sparse
indexer, the FP8 decode GEMMs and wo_a all stay on the GPU instead of falling back,
and a 256K context fits in 16 GB per card. New recipe:
`launch/sglang_serve_dsv41_tp2_5060ti.sh`.

New debug variable, default off: `SGLANG_SWA_REPLAY_REBUILD_WINDOW` (see the
environment-variable reference).

## Long prompts and cache reuse

- Long prefills no longer run out of memory as the KV pool fills.
- Prefix reuse stays available with the default decoder SWA replay mode; reuse of a
  state-less tombstone is limited to models whose state can be restored.

## Notes

- Retaining the SWA window behind the replay boundary is back-ported from
  sgl-project/sglang#46008, which is still open upstream.
- Perplexity on the frozen corpus reads 5.49 on this build; the 1.7.3 note of 5.19 was
  measured on an earlier build, so treat it as a build-to-build difference rather than a
  regression from these changes.

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
