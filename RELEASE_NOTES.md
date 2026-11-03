# lsglang 1.7.2 (from 1.7.1)

Needs lk_moe 2.8.2.

## Super-chunk prefill: correctness

- **Attention sub-chunks now hand the TP reduction over exactly as the caller
  asked.** The model-side projection loop called itself without
  `defer_all_reduce`, so every sub-chunk reduced eagerly and the layer took the
  non-collective post: whenever a batch was cut, the whole model ran a different
  reduction path. Measured on DeepSeek-V4-Flash-0731 (same text, same ids, first
  1024 positions): mean |dlogprob| **2.28 -> 0.175** at `--chunked-prefill-size
  1024`, PPL median 42.6 -> 6.32 (CPU reference 6.35).

## Expert cache: speculative decoding now benefits from it

- **Verify draft rows ride the cache** (`LK_VERIFY_RESIDENT`, now default on).
  Measured +17.6% median throughput (33.3 -> 39.1 tok/s, peak 51.7) with
  acceptance unchanged, once the pool stopped charging for the draft rows'
  admission (`LK_VERIFY_PUMP` stays off by default) and the staging copy was
  replaced by per-fragment DMA (`LK_POOL_NO_STAGE`, default on). **The dsv4.1
  recipes pin that back to the slab (`LK_POOL_NO_STAGE=0`)**: on dsv4.1-1017
  TP2/3090 the per-fragment fill cost decode throughput against the pre-1.7.2
  slab (decode median 27.0 -> 21.1 tok/s, 26.0 back with `=0`; same recipe,
  same log). Which path wins follows what the step waits on, so the README now
  states the rule per model instead of leaving it to the default.
- Canonical pool slots are only requested when the SM12x decode kernel will read
  them, so a gemv decode tier no longer repacks every fill.

- **`LK_POOL_DECODE_ENGINE` is gone** (env, accessor and the SM12x resident-half
  kernel), together with the canonical pool slot layout it required. The kernel
  measured no better than the gemv trio on its own shapes and forced a CPU
  repack of every expert fill, so it was cost without a return. Setting the env
  now does nothing; the resident half is always the gemv trio.

- **Ada (SM89, RTX 4080/4090) now runs the Ampere fp8 kernels everywhere.**
  The only paths a 40-series card took differently were the fp8-dot block linear
  (fp8-dot at prefill M, split-K w8a16 at decode M), the e4m3 Triton helpers
  (`fp8e4nv` vs the int/fp32 emulation) and the group-32 Marlin gate. A 4x4090
  field report has DeepSeek-V4.1-Flash-1017 (fp8 block 32x32 + ue8m0 scales)
  producing garbage through them, and nothing here has validated that route.
  Ada is now treated as Ampere on all three: in-register bf16 upcast GEMM,
  emulated e4m3 helpers, and the group-32 Marlin W8A16 path the 30-series cards
  already run. Costs the fp8 tensor-core prefill speed on Ada; correctness first.

## Fixes

- Warmup feeds real token ids instead of `np.random.randint(2**16)`. A random
  sweep is the worst access pattern an engine with off-device embedding tables
  can be handed: the DSV4.1 engram table is read from NVMe by token id and the
  N1 path keeps no RAM cache, so the shape ladder read tens of thousands of
  cold rows per step. Measured on dsv41-1017 (TP2/3090, `LK_ENGRAM_NVME=1`):
  3072/6144/12288-token shapes took **76/82/161 s** (40.5/75.0/76.3 tok/s) while
  the neighbouring power-of-2 shapes took 10/14 s (418/582 tok/s), the GPU gates
  spinning at ~150 W for rows the filler could not stage in time. Warmup now
  tokenizes one fixed passage and repeats it: same M coverage, with the n-gram
  locality a real request has. No knob; a missing tokenizer or a failed encode
  falls back to the old sweep, so warmup can never take the boot down.
- Qwen3.8 logprob OOM: the logprob chunk is clamped to the chunked-prefill size
  instead of the upstream 2048-row env default.
- The pool pump and mailbox workers bind to the model's device (`gpu_id`), not to
  a `std::thread` default of 0 -- on a TP pair that default is the other rank's
  GPU.
- `prefill_shapes` also warms prefix-cache chunks (a few new tokens behind a
  1024/2048-token hit); those extend shapes were first compiled mid-serving,
  where a lazy module load can park both ranks.

## Recipes

- **DeepSeek-V4.1: serve the Engram table from host RAM, not from NVMe.**
  Measured on dsv4.1-1017 TP2/3090 (same build, real text, exclusive): cold
  prefill 5.2K / 16.9K / 31.9K = **524 / 688 / 766 tok/s with `LK_ENGRAM_NVME=1`**
  against **666 / 967 / 1091 with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`**
  (+27 / +41 / +42%), decode 26 vs 28 tok/s at 4K. The NVMe path is
  latency-bound, not bandwidth-bound (a 32K prefill pulled at most 14.7 MB/s off
  the disks), so bandwidth counters will not show it. The table is ~48 GB of
  host RAM.

- **Every 5060 Ti (SM120) recipe now runs the flashinfer prefill engine with
  W4A4 armed** (`LK_GPU_PREFILL_ENGINE=flashinfer`,
  `LK_GPU_PREFILL_SM120_W4A4=1`). The MTP recipe was the one left on cutlass,
  whose `w4a4_support()` is a stub that returns False -- so that recipe ran the
  whole prefill on W4A16 while its siblings ran W4A4. `LK_GPU_PREFILL_WAVE_EXPERTS`
  is a cutlass-only knob and is gone from the flashinfer recipe.

## Notes

- `LK_POOL_MBOX_SPLIT` measured no gain and is no longer set by the recipes.
- The one-shot super-chunk diagnostics (`[sc-dbg]`: the hc plan the layer took,
  why the wrapper declined, the engram loop plan) sit behind `LK_SC_DEBUG`, so a
  production boot log no longer carries them.
- Parameter docs now separate operating knobs from debug-only ones: see
  "Debug / diagnostics" in the README.

# lsglang 1.7.1 (from 1.7.0)

Needs lk_moe 2.8.1.

## MoE on consumer Blackwell (SM120: RTX 5090 / 5060 Ti / RTX PRO 6000)

- **`LK_POOL_DECODE_ENGINE`** (new; default `gemv`, unchanged). With a GPU expert
  pool, chooses the kernel the decode GPU half uses. `sm12x` runs the FP4 kernel:
  +5.4% per step at one row and +3.3% at two rows (2x5060 Ti, same build), and
  keeps its intermediates in FP16, so it is a numerics class rather than a free
  speed switch - output differs in the third decimal. Needs
  `LK_GPU_PREFILL_ENGINE=flashinfer` and an intermediate size divisible by 128;
  otherwise it logs the reason and stays on `gemv`.
- **`LK_POOL_MBOX_SPLIT`** (new; code default off, `1` in the `launch/`
  recipes). Starts the decode GPU half before the CPU half is joined. Nothing
  measured at one or two rows per step, so it is a knob rather than a win, and
  the shipped recipes run with it on.

## Stability

- A long prompt is no longer an abort risk when another request joins it in
  mid-prefill; the server used to die on an assertion in that path. No parameter
  change.
- When a sparse-attention check does fire, the log line names what it was
  guarding instead of printing an empty assertion message.

---

# lsglang 1.7.0 (from 1.6.6)

## MoE on consumer Blackwell (SM120: RTX 5090 / 5060 Ti / RTX PRO 6000)

- **Native NVFP4 MoE speed (flag added by this fork).**
  `--moe-runner-backend sm12x_nvfp4_bf16` makes MoE layers with GPU-resident
  weights run SM120 hardware FP4 kernels instead of generic fallbacks; CPU
  offload/pool layers are untouched by it, so it is safe to set alongside any
  `LK_*` configuration.  by @lovedheart
- **Faster long prompts.** Turn on the GPU prefill tier for GPU-pooled MoE
  with `LK_GPU_PREFILL_MIN_TOKENS` (e.g. 1024) and
  `LK_GPU_PREFILL_ENGINE=flashinfer` (SM120/SM121, NVFP4 checkpoints): expert
  weights run on the FP4 tensor cores at W4A16 (CPU-matching) numerics. FP4 activations
  (`LK_GPU_PREFILL_SM120_W4A4=1`) pairs with this engine specifically - it is
  the only engine implementing W4A4 - and adds more speed at slightly
  different numerics; on its own flashinfer stays safe, and W4A4 without the
  engine or the hardware degrades back to W4A16 with a warning. Defaults are
  unchanged; output is unchanged at defaults.  wiring by @lovedheart

## GPU memory

- **GPU-pooled MoE experts.** Expert staging no longer reserves VRAM of its
  own: it runs inside the memory the pool already owns.
- **Memory is planned at startup.** All MoE VRAM regions are sized and
  committed during boot (one `[tier]` line prints the geometry); nothing is
  allocated or resized while serving.

## Stability

- Super-chunk prefill no longer splits a sequence across chunk boundaries
  (a source of degraded output in mixed long batches). Fixes #17.
- Speculative decode no longer stalls on the first sampled request
  (EAGLE / NEXTN and DSpark). No parameter change.
- Long prompts no longer pay a one-off compile. No parameter change.
- Parameters that were tightened to stay clear of those stalls can be restored:
  put `--mem-fraction-static` and `--max-total-tokens` back to your usual
  values on speculative-decode deployments.
- `LK_GPU_PREFILL_WAVE_EXPERTS` applies to the cutlass engine only. With
  `LK_GPU_PREFILL_ENGINE=triton` or `flashinfer` it is ignored with a warning
  instead of aborting startup: remove the line from those launch scripts, or
  leave it in place.

## Docs

- README: refreshed lk_moe env-knob reference (dead knobs removed) and two
  ready-to-run 5060 Ti launch recipes.
