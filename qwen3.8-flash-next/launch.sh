#!/usr/bin/env bash
# Parameterized Qwen3.8-Flash-Next-FP8 launcher for autoresearch sweeps.
# Env knobs: SPEC_CONFIG, MAX_MODEL_LEN, GPU_MEM_UTIL, MAX_NUM_SEQS, PORT,
#            PLE_OFFLOAD, KV_CACHE_DTYPE, MAX_NUM_BATCHED_TOKENS, EXTRA_ENV,
#            EXTRA_ARGS
# Mounts patches/connector.py: sm120 workaround for the PLE-offload warmup
# deadlock (vllm#53960) — drops the cross-stream event wait in _copy_cuda_inputs.
set -euo pipefail
IMAGE="vllm/vllm-openai:qwen38-flash-next"
NAME="qwen38-flash-next-prod"
MODEL="${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}"
PORT="${PORT:-30001}"
SPEC_CONFIG="${SPEC_CONFIG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"
PLE_OFFLOAD="${PLE_OFFLOAD:-1}"

SPEC_ARGS=()
if [[ -n "$SPEC_CONFIG" ]]; then
  SPEC_ARGS=(--speculative-config "$SPEC_CONFIG")
fi

KV_ARGS=()
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  KV_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

MBT_ARGS=()
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  MBT_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

ENV_ARGS=()
if [[ -n "$EXTRA_ENV" ]]; then
  for kv in $EXTRA_ENV; do ENV_ARGS+=(-e "$kv"); done
fi

CACHE=$HOME/.cache/vllm-docker-qwen38
mkdir -p "$CACHE"/{vllm,triton,flashinfer}
docker rm -f "$NAME" 2>/dev/null || true

docker run -d \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  -p "$PORT":"$PORT" \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_PLE_CPU_OFFLOAD="$PLE_OFFLOAD" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${EXEC_TIMEOUT:-3600}" \
  "${ENV_ARGS[@]}" \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:rw \
  -v "$CACHE/vllm":/root/.cache/vllm:rw \
  -v "$CACHE/triton":/root/.triton:rw \
  -v "$CACHE/flashinfer":/root/.cache/flashinfer:rw \
  -v $HOME/qwen3.8-flash-next/patches/connector.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/connector.py:ro \
  "$IMAGE" \
  "$MODEL" \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  "${SPEC_ARGS[@]}" \
  "${KV_ARGS[@]}" \
  "${MBT_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}" \
  --compilation-config "{\"cudagraph_capture_sizes\":[1,2,4,8,12,16,20,24,32,48,64]}" \
  $EXTRA_ARGS
echo "launched $NAME port=$PORT spec=${SPEC_CONFIG:-none} ple_offload=$PLE_OFFLOAD kv=${KV_CACHE_DTYPE:-auto} mbt=${MAX_NUM_BATCHED_TOKENS:-default} util=${GPU_MEM_UTIL:-0.90}"
