#!/usr/bin/env bash
LOG="${LOG:-/tmp/opencode/serve_logs/sglang_serve_dsv41_tp4.log}"
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
# DeepSeek-V4.1-Flash-1017 · 2× RTX 3090 (SM86), 2x RTX 5060Ti, TP4.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
LK_POOL_MBOX_SPLIT=1 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=16 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_SM8X_MARLIN_GROUP32=1 \
SGLANG_SM8X_MARLIN_GROUP32_ONLY=1 \
FLASHINFER_USE_CUDA_NORM=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
sglang serve --sleep-on-idle \
  --model ~/Models/DeepSeek-V4.1-Flash-1017 \
  --served-model-name DeepSeek-V4.1-Flash-1017 \
  --host 0.0.0.0 \
  --port 8070 \
  --trust-remote-code \
  --enable-decoder-swa-bounded-replay \
  --tensor-parallel-size 4 \
  --max-running-requests 2 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 1024 \
  --max-total-tokens 262144 \
  --mem-fraction-static 0.80 \
  --warmups prefill_shapes \
  --speculative-attention-mode decode
