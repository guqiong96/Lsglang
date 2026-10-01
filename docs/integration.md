# Integrating lk_moe into an inference framework

lk_moe is a pip-installable package (`pip install lk-moe`). It exposes a small set of C++ kernel
classes (`MOE_WNA16`, `MOE_FP8`, `MOE_MXFP4`, ...) driven by a `MOEConfigV2` config.
The engine handles expert weight placement (VRAM / pinned NUMA host memory), NUMA-aware scheduling,
and quantized kernel execution internally.

The integration work in a serving framework is therefore **only about routing each MOE layer to
lk_moe** (which layers stay on GPU, which go hybrid, which quant kernel to use) and **keeping the
feature optional** so the branch stays 100% compatible with stock behavior when disabled.

## Core integration principle

> **Every MOE layer can be one of three roles.** The role is decided by a few env vars, and the
> rest of the engine is unchanged.

| Role | Meaning | Decision |
|---|---|---|
| GPU-resident layer | all weights in VRAM, original GPU path | `LK_GPU_RESIDENT_MOE_LAYERS` |
| CPU layer (hybrid) | MoE weights in memory, attn in VRAM; GPU computes attn + CPU computes MoE | default when enabled |
| GPU-prefill layer | large batches on GPU, small batches on CPU | `LK_GPU_PREFILL_MIN_BATCH_SIZE` |

## Minimal integration checklist

1. **Add the dependency** — `lk_moe` in `python/pyproject.toml` (for sglang) / `requirements` (for vllm).
2. **Add a feature gate** — `is_lk_moe_feature_enabled()` (reads `LK_MOE_HYBRID_ENABLED`) so all
   hybrid behavior is off by default and the branch behaves exactly like stock sglang/vllm.
3. **Wire the MOE layer** — in the fused-MoE layer, resolve each layer's role, build a
   `lk_moe.MOEConfigV2`, instantiate the quant-appropriate `MOE_*` class, and call it in `forward`.
   Pool identity is resolved here too, as config fields only (`layer_id`, `gpu_pool`,
   `route_probe`) — the engine reads no pool-membership env vars, and per-layer membership
   collapses into one predicate (`gpu_pool = non-draft AND non-resident AND in LK_POOL_LAYERS`).
4. **Register per-quantization kernels** — each quant method exposes its own LK MoE kernel class.
5. **Handle weight loading / placement** — keep CPU-resident weights off the GPU device.
6. **(Optional) extras** — NUMA thread binding.

## Case study — Lsglang (sglang) file-by-file

| File | What it does |
|---|---|
| `python/pyproject.toml` | adds `lk_moe` dependency |
| `srt/utils/common.py` | the feature-gate helpers: `is_lk_moe_feature_enabled`, `is_lk_moe_gpu_resident_layer`, `is_lk_pool_layer`, `get_gpu_prefetch_window`, ... |
| `srt/layers/moe/fused_moe_triton/layer.py` | **the core**: resolve layer role, build `MOEConfigV2`, instantiate `MOE_WNA16` / `MOE_FP8` / `MOE_MXFP4` per quant, and dispatch in `run_moe_core` (GPU resident → `quant_method.apply`; hybrid → `_cpu_decode` / `_cpu_prefill` / `_gpu_prefill`) |
| `srt/layers/quantization/{fp8,unquant,modelopt_quant,mxfp4_*}.py` | each quant method registers its LK MoE kernel (e.g. `MOE_FP8`, `MOE_MXFP4`) |
| `srt/layers/quantization/compressed_tensors/schemes/*` | compressed-tensors W8A8-FP8 / W4A4-NVFP4 / WNA16 MoE each register their LK kernel |
| `srt/model_loader/loader.py` | keep CPU-resident MoE layers off the GPU device; run `process_weights_after_loading` / `clean_weights_after_loading` for lk_moe layers |
| `srt/utils/numa_utils.py` | when `LK_NUMA_INTERLEAVE=1`, launch workers under `numactl --interleave=all` |
