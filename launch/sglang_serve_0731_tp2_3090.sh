#!/usr/bin/env bash
LOG="${LOG:-/tmp/opencode/serve_logs/sglang_serve_0731_tp2_3090.log}"
mkdir -p /tmp/opencode/serve_logs
echo "===== $(date "+%F %T") launch pid=$$ =====" >> "$LOG"
if [ -t 1 ]; then
    # interactive terminal: console and log both live (closing the
    # terminal is a deliberate stop)
    exec > >(tee -a "$LOG") 2>&1
else
    # detached/tool launch: plain file append, no coprocess that a
    # session teardown could kill (EPIPE would take the engine down)
    exec >> "$LOG" 2>&1
fi
# DeepSeek-V4-Flash-0731 · 2× RTX 3090 (SM86), TP2, plain decode.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=0,3 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=cutlass \
LK_GPU_PREFILL_SUB_M=16384 \
LK_GPU_PREFILL_D_SLOTS=16384 \
LK_GPU_PREFILL_WAVE_EXPERTS=128 \
LK_POOL_MBOX_SPLIT=1 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_POOL_SLOTS=32 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_SM8X_MARLIN_GROUP32=1 \
SGLANG_SM8X_MARLIN_GROUP32_ONLY=1 \
FLASHINFER_USE_CUDA_NORM=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
sglang serve --sleep-on-idle \
  --model ~/Downloads/DeepSeek-V4-Flash-0731 \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --host 0.0.0.0 \
  --port 8070 \
  --trust-remote-code \
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
