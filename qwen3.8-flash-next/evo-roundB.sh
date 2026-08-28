#!/usr/bin/env bash
# Evo round B: quantify the PLE-offload PCIe tax via eager-mode A/B.
# pleoff-eager vs pleon-eager isolates the per-step N-gram fetch cost.
set -uo pipefail
cd $HOME/qwen3.8-flash-next
LOG=results/evo-roundB.log

run_cfg() {
  local NAME="$1"; shift
  echo "########## ROUND-B $NAME : $* ##########" | tee -a "$LOG"
  env "$@" bash launch.sh
  local UP=0
  for i in $(seq 1 80); do
    if ! docker ps --format '{{.Names}}' | grep -q '^qwen38-flash-next-prod$'; then break; fi
    if curl -s -m 2 localhost:30001/v1/models > /dev/null 2>&1; then UP=1; break; fi
    sleep 15
  done
  if [ "$UP" != "1" ]; then
    echo "ROUND-B $NAME BOOT FAILED" | tee -a "$LOG"
    docker logs qwen38-flash-next-prod 2>&1 | grep -iE "error|assert" | tail -2 | tee -a "$LOG"
    return
  fi
  echo "ROUND-B $NAME up" | tee -a "$LOG"
  docker cp probe.py qwen38-flash-next-prod:/tmp/probe.py
  docker exec qwen38-flash-next-prod python3 /tmp/probe.py short 5 128 | tee -a "$LOG"
  docker exec qwen38-flash-next-prod python3 /tmp/probe.py long 3 128 | tee -a "$LOG"
  bash bench.sh "evoB-${NAME}-short" 512 256 1 8 2>&1 | grep -E "===|Output token" | tee -a "$LOG"
}

run_cfg pleoff-eager PLE_OFFLOAD=0 GPU_MEM_UTIL=0.97 \
  EXTRA_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  EXTRA_ARGS=--enforce-eager
run_cfg pleon-eager EXTRA_ARGS=--enforce-eager
echo "ROUND-B DONE" | tee -a "$LOG"
