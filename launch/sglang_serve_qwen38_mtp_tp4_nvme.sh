#!/usr/bin/env bash
LOG="${LOG:-/tmp/opencode/serve_logs/sglang_serve_qwen38_mtp_tp4.log}"
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
# Qwen3.8-Flash-Next-NVFP4 · 2x RTX 3090 (SM86) + 2x RTX 5060 Ti (SM120), TP4, MTP (NEXTN).

CUDA_DEVICE_ORDER=PCI_BUS_ID \
GLIBC_TUNABLES=glibc.malloc.thp_enabled=0 \
GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
LK_MOE_HYBRID_ENABLED=1 \
LK_THREADS=48 \
OMP_NUM_THREADS=1 \
LK_THREAD_BINDING=NUMA_NODE \
LK_GPU_PREFETCH_WINDOW=1 \
LK_GPU_PREFILL_MIN_TOKENS=1024 \
LK_GPU_PREFILL_ENGINE=cutlass \
LK_GPU_PREFILL_SUB_M=32768 \
LK_GPU_PREFILL_D_SLOTS=16384 \
LK_NUMA_INTERLEAVE=1 \
LK_POWER_SAVING=1 \
LK_GPU_RESIDENT_MOE_LAYERS="" \
LK_GPU_RESIDENT_MOE_LAYERS_SPEC="" \
LK_POOL_SLOTS=16 \
LK_POOL_LOG=0 \
LK_NVME_DIRECT=1 \
LK_NVME_CACHE_MB=4096 \
FLASHINFER_USE_CUDA_NORM=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
/home/guqiong/.conda/envs/Lsglang/bin/sglang serve --sleep-on-idle \
  --model ~/Models/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4 \
  --served-model-name Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 \
  --port 8070 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-running-requests 2 \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --ple-offload-embedding \
  --ple-offload-backend nvme \
  --ple-offload-dir /home/guqiong/Models/.ple_cache/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4 \
  --chunked-prefill-size 1024 \
  --max-total-tokens 262144 \
  --mem-fraction-static 0.80 \
  --warmups prefill_shapes \
  --mm-feature-transport=cpu \
  --mm-process-config '{"image":{"max_pixels":3686400,"min_pixels":200704}}' \
  --mm-attention-backend flashinfer_cudnn \
  --image-processor-backend pil \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-max-states-per-path 2 \
  --mamba-track-interval 256 \
  --max-mamba-cache-size 11 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"medium"}' \
  --moe-runner-backend marlin
