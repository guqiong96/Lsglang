#!/usr/bin/env bash
LOG="${LOG:-/tmp/opencode/serve_logs/sglang_serve_glm53_tp4_3090.log}"
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
# GLM-5.3-Flash-NVFP4 · 4× RTX 3090 (SM86), TP4, plain decode.

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=cutlass \
LK_GPU_PREFILL_SUB_M=16384 \
LK_GPU_PREFILL_D_SLOTS=16384 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_POOL_SLOTS=0 \
FLASHINFER_USE_CUDA_NORM=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
/home/guqiong/.conda/envs/Lsglang/bin/python /home/guqiong/Downloads/sglang/python/sglang/launch_server.py --sleep-on-idle \
  --model ~/Models/GLM-5.3-Flash-NVFP4 \
  --served-model-name GLM-5.3-Flash-NVFP4 \
  --host 0.0.0.0 \
  --port 8070 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-running-requests 2 \
  --max-mamba-cache-size 11 \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --chunked-prefill-size 1024 \
  --max-total-tokens 262144 \
  --mem-fraction-static 0.80 \
  --warmups prefill_shapes \
  --cuda-graph-backend-prefill disabled
