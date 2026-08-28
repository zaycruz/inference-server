#!/usr/bin/env bash
# Deploy GLM-5.3-Flash-NVFP4 inference server (vLLM glm53-flash image)
#
# Template only. Do NOT run while DeepSeek-V4-Flash-0731 occupies the GPUs
# (port 30000). Deploy when the cards are free:
#   sudo bash $HOME/glm-5.3-flash-nvfp4/docker-run.sh
#
# Serves on port 30001. Model: LibertAIDAI/GLM-5.3-Flash-NVFP4 (181 GB).
set -euo pipefail

IMAGE="vllm/vllm-openai:glm53-flash-x86_64-cu130"
NAME="glm-5.3-flash-nvfp4-prod"
# MTP spec decoding: the checkpoint ships the MTP draft layer (layer 45).
# Recipe: vllm recipes zai-org/GLM-5.3-Flash. Set SPEC_CONFIG="" to disable
# if the NVFP4 MTP path fails at deploy time.
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":5}'
SPEC_ARGS=()
if [[ -n "${SPEC_CONFIG:-}" ]]; then
  SPEC_ARGS=(--speculative-config "$SPEC_CONFIG")
fi

# Fresh kernel-cache dirs for this vLLM build (do not share DeepSeek's caches).
CACHE=$HOME/.cache/vllm-docker-glm53
mkdir -p "$CACHE"/{vllm,triton,flashinfer}

# Stop and remove any previous instance of this container.
docker rm -f "$NAME" 2>/dev/null || true

docker run -d \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --restart unless-stopped \
  -p "$PORT":"$PORT" \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:rw \
  -v "$CACHE/vllm":/root/.cache/vllm:rw \
  -v "$CACHE/triton":/root/.triton:rw \
  -v "$CACHE/flashinfer":/root/.cache/flashinfer:rw \
  "$IMAGE" \
  serve "$MODEL" \
  --trust-remote-code \
  --served-model-name glm-5.3-flash-nvfp4 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.97 \
  --max-model-len 32768 \
  --max-num-seqs 24 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  "${SPEC_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --disable-custom-all-reduce

# Fresh kernel-cache dirs for this vLLM build (do not share DeepSeek's caches).
CACHE=$HOME/.cache/vllm-docker-glm53
mkdir -p "$CACHE"/{vllm,triton,flashinfer}

# Stop and remove any previous instance of this container.
docker rm -f "$NAME" 2>/dev/null || true

docker run -d \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --restart unless-stopped \
  -p "$PORT":"$PORT" \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:rw \
  -v "$CACHE/vllm":/root/.cache/vllm:rw \
  -v "$CACHE/triton":/root/.triton:rw \
  -v "$CACHE/flashinfer":/root/.cache/flashinfer:rw \
  "$IMAGE" \
  serve "$MODEL" \
  --trust-remote-code \
  --served-model-name glm-5.3-flash-nvfp4 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.97 \
  --max-model-len 32768 \
  --max-num-seqs 24 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --disable-custom-all-reduce

echo "Deployed $NAME on port $PORT."
echo "Check: docker logs -f $NAME"
echo "Smoke: curl http://127.0.0.1:$PORT/v1/models"
