#!/usr/bin/env bash
# Evo round A: MoE backend sweep, all with MTP=3 held constant.
set -uo pipefail
cd $HOME/qwen3.8-flash-next
LOG=results/evo-roundA.log
SUM=results/evo-roundA-summary.csv
echo "config,conc,in,out,output_toks" > "$SUM"

run_cfg() {
  local NAME="$1"; shift
  echo "########## ROUND-A $NAME : $* ##########" | tee -a "$LOG"
  env SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}' "$@" bash launch.sh
  local UP=0
  for i in $(seq 1 60); do
    if ! docker ps --format '{{.Names}}' | grep -q '^qwen38-flash-next-prod$'; then break; fi
    if curl -s -m 2 localhost:30001/v1/models > /dev/null 2>&1; then UP=1; break; fi
    sleep 15
  done
  if [ "$UP" != "1" ]; then
    echo "ROUND-A $NAME BOOT FAILED" | tee -a "$LOG"
    docker logs qwen38-flash-next-prod 2>&1 | grep -iE "error|assert" | tail -2 | tee -a "$LOG"
    echo "$NAME,BOOT_FAILED,,," >> "$SUM"
    return
  fi
  echo "ROUND-A $NAME up" | tee -a "$LOG"
  bash bench.sh "evoA-${NAME}-short" 512 256 1 8 24 2>&1 | grep -E "===|Output token" | tee -a "$LOG"
  docker cp probe.py qwen38-flash-next-prod:/tmp/probe.py
  docker exec qwen38-flash-next-prod python3 /tmp/probe.py short 5 128 | tee -a "$LOG"
  python3 summarize.py "evoA-${NAME}-short" "$NAME" >> "$SUM"
}

run_cfg deepgemm
run_cfg triton EXTRA_ARGS=--moe-backend=triton
run_cfg marlin EXTRA_ARGS=--moe-backend=marlin
run_cfg fi_cutlass EXTRA_ARGS=--moe-backend=flashinfer_cutlass
run_cfg batched_triton EXTRA_ARGS=--moe-backend=batched_triton
echo "ROUND-A DONE" | tee -a "$LOG"
cat "$SUM"
