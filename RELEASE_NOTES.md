# Release Notes — Lsglang-v1.5.2

**Base Version:** sglang `dsv4.1` (upstream branch, commit `1aa0e962b`) + lk_moe v2.4.3
**Release Type:** Feature integration release — adds **DeepSeek-V4.1 SM80/SM86** support

Use this wheel for **all** GPUs (SM80 → SM120). SM89+/SM120+ behaviour is byte-identical to the
pure-lk_moe release (`v1.5.1`); SM80/SM86 (Ampere / RTX 30) DeepSeek-V4.1 support is added on top.

## Patches (`patches/`)

| Patch | Applies to | Contents |
|-------|-----------|----------|
| [`01_lk_moe__dsv4.1.patch`](./patches/01_lk_moe__dsv4.1.patch) | clean sglang `dsv4.1` (`1aa0e962b`) | pure lk_moe MOE hybrid inference |
| [`02_sm80_support__dsv4.1.patch`](./patches/02_sm80_support__dsv4.1.patch) | after 01 | DeepSeek-V4.1 on SM80/SM86 (optional, only needed for Ampere / RTX 30) |

```bash
git apply patches/01_lk_moe__dsv4.1.patch            # SM89+/SM120+ stop here
git apply patches/02_sm80_support__dsv4.1.patch       # + SM80/SM86 support
```

## What's new for SM80/SM86 (patch 02)

- DeepSeek-V4.1 runs end-to-end on RTX 30 (SM86): c128 multi-bucket target-verify graphs, `mhc`
  split-k generalized to `hc_hidden_size=20480`, block-fp8 `w8a16` GEMM (with split-K) where Triton
  cannot emit `fp8e4nv`, and bit-exact fp8 decode/round in a shared `fp8_emulate` module (engram,
  RoPE FP4 fake-quant). SM89+/SM120+ code paths untouched.
- SM120 (RTX 50 / Blackwell) MoE note: at TP=4 the FlashInfer **CUTLASS** MXFP4 MoE path requires
  `intermediate % 128 == 0`; if it trips, use `--moe-runner-backend flashinfer_trtllm` (auto-pads)
  or `marlin`, or keep MoE off pure-TP (EP / TP=2). This is an upstream constraint, not a regression.

## Supported models

General sglang model list applies, plus the DeepSeek-V4 / GLM / Qwen-Flash lines below.

## Additional support branches (v0.5.19 series)

New-model and architecture-specific support is provided by the following in-repo branches
(based on the v0.5.19 line), grouped by model:

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

`feat/qwen38-flash-next` is an external sglang branch by lovedheart; lk_moe was integrated on top
of it and released as `lsglang-v1.4.13`. SM89 support requires the flash-attention PR #2751 patch
(prebuilt `flash_attn-2.8.4+pr2751` wheel).

| Branch | Arch | Author |
|--------|------|--------|
| [feat/qwen38-flash-next](https://github.com/lovedheart/sglang/tree/feat/qwen38-flash-next) | SM89+ / SM120+ | lovedheart |
