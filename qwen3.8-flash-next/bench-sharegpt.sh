#!/usr/bin/env bash
# ShareGPT (real workload) bench for qwen38-flash-next-prod.
# Usage: bash bench-sharegpt.sh <tag> [concurrencies...]
set -euo pipefail
TAG="$1"; shift
CONCS=("$@")
if [ ${#CONCS[@]} -eq 0 ]; then CONCS=(1 2 4 8); fi
DIR=$HOME/qwen3.8-flash-next/results/$TAG
mkdir -p "$DIR"
for C in "${CONCS[@]}"; do
  echo "=== $TAG conc=$C sharegpt ==="
  docker exec qwen38-flash-next-prod vllm bench serve \
    --backend openai-chat \
    --host 127.0.0.1 --port 30001 \
    --endpoint /v1/chat/completions \
    --model qwen3.8-flash-next \
    --tokenizer "${TOKENIZER:-Qwen/Qwen3.8-Flash-Next-FP8}" \
    --dataset-name sharegpt \
    --dataset-path /tmp/sharegpt.json \
    --num-prompts $((C * 8)) \
    --max-concurrency "$C" \
    --ignore-eos \
    --percentile-metrics "50,90,99" \
    --save-result --result-filename "$DIR/conc${C}_sharegpt.json" \
    2>&1 | tee "$DIR/conc${C}_sharegpt.log" | grep -E "throughput|Throughput|TTFT|TPOT|ITL|Successful|input len|output len" || true
done
echo "done -> $DIR"
