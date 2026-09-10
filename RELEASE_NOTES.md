# Release Notes — Lsglang-v1.5.1 (dsv4.1)

**Base Version:** sglang `dsv4.1` (upstream branch, commit `1aa0e962b`) + lk_moe v2.4.3
**Branch:** `dsv4.1-lkmoe`
**Release Type:** Feature integration release

The lk_moe integration is captured as a single portable patch:
[`patches/01_lk_moe__dsv4.1.patch`](./patches/01_lk_moe__dsv4.1.patch)

## Additional support branches (v0.5.19 series)

New-model and architecture-specific support is provided by the following
in-repo branches (based on the v0.5.19 line), grouped by model:

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

`feat/qwen38-flash-next` is an external sglang branch by lovedheart; lk_moe was
integrated on top of it and released as `lsglang-v1.4.13`. SM89 support requires
the flash-attention PR #2751 patch (prebuilt `flash_attn-2.8.4+pr2751` wheel).

| Branch | Arch | Author |
|--------|------|--------|
| [feat/qwen38-flash-next](https://github.com/lovedheart/sglang/tree/feat/qwen38-flash-next) | SM89+ / SM120+ | lovedheart |
