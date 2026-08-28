#!/usr/bin/env bash
# MTP sweep: for each num_speculative_tokens, restart server, wait for API,
# run short bench (all concs) + long bench (conc 8,24). Logs spec-accept stats.
set -uo pipefail
cd $HOME/qwen3.8-flash-next
for N in "$@"; do
  echo "########## MTP num_speculative_tokens=$N ##########"
  SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":$N}" bash launch.sh
  UP=0
  for i in $(seq 1 60); do
    if curl -s -m 2 localhost:30001/v1/models > /dev/null 2>&1; then UP=1; break; fi
    sleep 15
  done
  if [ "$UP" != "1" ]; then
    echo "SERVER FAILED TO BOOT for N=$N, skipping"
    docker logs qwen38-flash-next-prod 2>&1 | tail -20
    continue
  fi
  echo "server up for N=$N"
  bash bench.sh "mtp${N}-short" 512 256 1 2 4 8 16 24 2>&1 | grep -E "===|Output token throughput|Successful"
  bash bench.sh "mtp${N}-long" 8192 512 8 24 2>&1 | grep -E "===|Output token throughput|Successful"
  echo "--- spec decode metrics for N=$N:"
  docker logs qwen38-flash-next-prod 2>&1 | grep -iE "accept" | tail -3
done
echo "SWEEP DONE"
