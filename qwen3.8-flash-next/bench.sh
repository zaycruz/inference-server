#!/usr/bin/env bash
# vllm bench serve runner for qwen38-flash-next-prod.
# Usage: bash bench.sh <tag> <input_len> <output_len> [concurrencies...]
# Results tee'd to results/<tag>/
set -euo pipefail
TAG="$1"; IN="$2"; OUT="$3"; shift 3
CONCS=("$@")
if [ ${#CONCS[@]} -eq 0 ]; then CONCS=(1 2 4 8 16 24); fi
DIR=$HOME/qwen3.8-flash-next/results/$TAG
mkdir -p "$DIR"
for C in "${CONCS[@]}"; do
  echo "=== $TAG conc=$C in=$IN out=$OUT ==="
  docker exec qwen38-flash-next-prod vllm bench serve \
    --backend openai-chat \
    --host 127.0.0.1 --port 30001 \
    --endpoint /v1/chat/completions \
    --model qwen3.8-flash-next \
    --tokenizer "${TOKENIZER:-Qwen/Qwen3.8-Flash-Next-FP8}" \
    --dataset-name random \
    --random-input-len "$IN" --random-output-len "$OUT" \
    --num-prompts $((C * 4)) \
    --max-concurrency "$C" \
    --ignore-eos \
    --percentile-metrics "50,90,99" \
    --save-result --result-filename "$DIR/conc${C}_in${IN}_out${OUT}.json" \
    2>&1 | tee "$DIR/conc${C}_in${IN}_out${OUT}.log" | grep -E "throughput|Throughput|TTFT|TPOT|ITL|Successful" || true
done
echo "done -> $DIR"
