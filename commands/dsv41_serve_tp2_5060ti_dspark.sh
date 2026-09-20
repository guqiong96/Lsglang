#!/usr/bin/env bash
# DeepSeek-V4.1-Flash - 2x RTX 5060 Ti (SM120, PCI idx 1,2), TP2, DSPARK.
# Adjust MODEL / CUDA_VISIBLE_DEVICES / LK_THREADS to your host.
# kill: pkill -f "launch_server.bi[n]"
set -e
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
LVLLM_MOE_NUMA_ENABLED=1 \
LK_THREAD_BINDING=CPU_CORE LK_THREADS=48 OMP_NUM_THREADS=1 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 \
LVLLM_GPU_PREFETCH_WINDOW=1 LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=0 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
LK_POWER_SAVING=1 \
SGLANG_SKIP_P2P_CHECK=1 \
SGLANG_DSV4_TP_ARCH_LOCAL=1 \
exec env -u PYTHONPATH /home/guqiong/.conda/envs/Lsglang/bin/python \
  "$SGIT/python/sglang/launch_server.py" \
  --model "${MODEL:-$HOME/Models/DeepSeek-V4.1-Flash}" \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 0.0.0.0 --port 8070 --trust-remote-code \
  --tensor-parallel-size 2 --max-running-requests 2 \
  --chunked-prefill-size 1024 --max-total-tokens 65536 --mem-fraction-static 0.92 \
  --dist-timeout 180 \
  --cuda-graph-backend-prefill disabled --disable-shared-experts-fusion \
  --enable-decoder-swa-bounded-replay \
  --speculative-algo DSPARK --speculative-dspark-block-size 5 \
  --speculative-attention-mode decode
