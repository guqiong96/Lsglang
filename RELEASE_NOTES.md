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
