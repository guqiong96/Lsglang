#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 on 2x RTX 5060 Ti (SM120), TP2.
# 5060 Ti sibling of sglang_serve_dsv41_tp2_3090.sh.
#
# Cards: under CUDA_DEVICE_ORDER=PCI_BUS_ID the 5060 Ti pair is 1,2 (0,3 are the
# 3090s). Index form is required: sglang's custom allreduce and lk_moe both
# int()-parse CUDA_VISIBLE_DEVICES; a UUID form silently degrades both (~40%
# decode loss, measured 2026-11-08).
#
# VRAM accounting (16 GB/rank vs 24 GB on the 3090 recipe), measured on 1.7.3:
#   weights + fixed  -> leaves 2.07 GB free after the lk_moe arena (Memory pool
#                       end log); device KV is only 1630 B/token, so 65536 tokens
#                       cost ~0.32 GB incl. the fixed SWA ring.
#   pool arena row   = 8.96 MiB; cache rows = 40 layers x LK_POOL_SLOTS.
#     slots=12 -> 4.20 GiB, slots=14 -> 4.90 GiB (+0.70 GB), 16 -> 5.60 GiB (+1.40,
#     only 0.35 GB left: the boot that failed at mem-fraction 0.95).
#   SUB_M raises the prefill side two ways: act rows (measured 319 -> 339 going
#     2048 -> 8192, driven by D_SLOTS=16384) and the staging buffer
#     SUB_M*H*4 B (=0.17 GB at 8192). arena = max(cache, prefill), so slots do
#     not grow with SUB_M; the free rows only exist when prefill > cache.
#   chosen: slots=12 + SUB_M=8192 + 128K context. SWA fixed bytes scale with
#           max-total-tokens (0.20 GB at 64K, 0.46 GB at 256K). 256K needs the
#           SM120 wo_a fp8 path (32x32 scales, on by arch) and still leaves the
#           next 8K prefill 320 MiB short, so it stays at 128K. Never add
#           PYTORCH_CUDA_ALLOC_CONF=expandable_segments here: its VMM handles
#           cannot be exported for the custom all-reduce graph buffers.
#
# Decoder SWA bounded replay is upstream's default for V4.1 (the hook resolves it
# to True unless explicitly disabled) and it is what keeps prefill fast, so it
# stays. What it costs: a prefilled prompt is stored as a Full-only tombstone, so
# a long conversation re-prefills instead of reusing (cached_tokens = 0). That is
# upstream behaviour; restoring reuse would need the low-ratio (c4/c128) KV rows
# and indexer keys to be cached per node, which they are not (per-request state
# slots). Reusing the tombstone anyway is measured to break output.
#           mem-fraction-static 0.90.
#
# Acceleration (all on): the SM120 model hook defaults the indexer to the torch
# paged-MQA fallback and leaves the two SM120 DeepGEMM knobs off, because older
# wheels lack the kernels. With a build that ships them (deep_gemm 2.8.0+nv_dev)
# the native sparse indexer and both knobs are enabled below.
#
# Dense block-FP8 GEMM: the checkpoint is 32x32/ue8m0, and sglang gates the
# DeepGEMM 1x32x32 recipe to SM90, so SM120 fell back to the untuned triton
# kernel (52% of decode kernel time). fp8_utils._deepgemm_block_fp8_supported
# now also accepts [32,32] on SM120 for M<=8 (decode); measured 1.0-3.6x faster
# and bit-identical at M=1. Prefill-sized batches keep the previous path.
LOG="${LOG:-/tmp/opencode/serve_logs/sglang_serve_dsv41_tp2_5060ti.log}"
mkdir -p /tmp/opencode/serve_logs
echo "===== $(date "+%F %T") launch pid=$$ =====" >> "$LOG"
if [ -t 1 ]; then
    # interactive terminal: console and log both live
    exec > >(tee -a "$LOG") 2>&1
else
    # detached/tool launch: plain file append, no coprocess a teardown can kill
    exec >> "$LOG" 2>&1
fi
export PATH="/home/guqiong/.conda/envs/Lsglang/bin:$PATH"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
CUDA_VISIBLE_DEVICES=1,2 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=cutlass \
LK_GPU_PREFILL_SUB_M=8192 \
LK_GPU_PREFILL_D_SLOTS=16384 \
LK_GPU_PREFILL_WAVE_EXPERTS=64 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_POOL_SLOTS=12 \
LK_POOL_NO_STAGE=0 \
LK_POOL_LOG=0 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_LOGPROB_CHUNK_SIZE=64 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_SKIP_P2P_CHECK=1 \
FLASHINFER_USE_CUDA_NORM=1 \
SGLANG_ENABLE_JIT_DEEPGEMM=1 \
SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
SGLANG_TORCH_PROFILER_DIR=/tmp/opencode/profiles \
SGLANG_OPT_FP8_WO_A_GEMM=1 \
SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
SGLANG_DEBUG_SWA_REPLAY_MATCH=1 \
sglang serve --sleep-on-idle \
  --model ~/Models/DeepSeek-V4.1-Flash-1017 \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 \
  --port 8072 \
  --trust-remote-code \
  --enable-decoder-swa-bounded-replay \
  --tensor-parallel-size 2 \
  --max-running-requests 2 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 1024 \
  --max-total-tokens 262144 \
  --mem-fraction-static 0.90 \
  --warmups prefill_shapes \
  --cuda-graph-backend-prefill disabled
