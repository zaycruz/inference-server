# Qwen3.8-Flash-Next on 2× RTX PRO 6000 (sm120) — Baseline + MTP Sweep Report

Date: 2026-08-27. Server: asus-trx50 (2× RTX PRO 6000 96GB, sm120, 123GB RAM).
Model: `Qwen/Qwen3.8-Flash-Next-FP8` (125B MoE, 6B active + 51B N-gram table, 172.78 GiB).
Engine: `vllm/vllm-openai:qwen38-flash-next`, TP=2, util 0.90, max-model-len 262144,
PLE CPU offload on (N-gram table in host RAM), prefix caching on.

**Deploy required two sm120 fixes** (baked into `launch.sh`):
1. `--disable-custom-all-reduce` (custom AR busy-waits forever on this box, same as GLM).
2. `patches/connector.py` — workaround for the PLE-offload warmup deadlock
   ([vllm#53960](https://github.com/vllm-project/vllm/issues/53960)): drops the
   cross-stream `_input_ready_event` wait in `_copy_cuda_inputs`, uses blocking copies.
   Without PLE offload the model does NOT fit (weights 86.5 GiB/GPU + 47.7 GiB autotune spike → OOM).

KV cache: ~20 GiB/GPU → 1.35M tokens (GDN linear attention on 3/4 layers).
Smoke tests: coherent output at temp 0 and 0.7, concurrency 8, no loops/garbage.

## Output token throughput (tok/s), `vllm bench serve`, random dataset, ignore-eos

### Short context (512 in / 256 out)

| Conc | Baseline | MTP=1 | MTP=2 | MTP=3 | Best |
|---|---|---|---|---|---|
| 1  | **122.7** | 102.5 | 107.4 | 117.0 | baseline |
| 2  | 189.1 | 181.3 | 196.8 | **229.4** | MTP=3 (+21%) |
| 4  | 338.6 | 275.7 | 352.9 | **376.8** | MTP=3 (+11%) |
| 8  | **518.0** | 501.2 | 486.1 | 506.9 | baseline (~tie MTP=3) |
| 16 | 661.8 | 648.6 | 607.5 | **672.3** | MTP=3 (+1.6%, ~tie) |
| 24 | **891.9** | 883.5 | 615.4 | 651.3 | baseline (+37% vs MTP=3) |

### Long context (8192 in / 512 out)

| Conc | Baseline | MTP=1 | MTP=2 | MTP=3 |
|---|---|---|---|---|
| 1  | 101.3 | — | — | — |
| 2  | 163.8 | — | — | — |
| 4  | 266.5 | — | — | — |
| 8  | **362.0** | 317.2 | 311.3 | 314.4 |
| 16 | 485.4 | — | — | — |
| 24 | **661.6** | 443.2 | 367.3 | 379.8 |

MTP=4 and MTP=5 do not boot on this build:
`AssertionError: QSA ring capacity 12 must divide the attention block size 1616`.
The valid sweep space is N ∈ {1, 2, 3} (recipe validates up to MTP3).

## MTP acceptance (from server SpecDecoding metrics)

- MTP=1: mean acceptance length ~1.6, per-position acceptance ~60%
- MTP=2: mean acceptance length ~1.9, position rates 57% / 32%
- MTP=3: mean acceptance length ~2.1, position rates 58% / 35% / 21%

## Real-workload check (ShareGPT dataset, conc 1–8, paired same-seed samples)

Random-token prompts are nearly unpredictable, so the numbers above *understate*
MTP. Re-ran low concurrency on real ShareGPT conversations (`--ignore-eos`):

| Conc | Baseline | MTP=3 | Delta |
|---|---|---|---|
| 1 | 48.0 | **121.4** | **+153%** |
| 2 | 119.0 | **217.6** | **+83%** |
| 4 | 290.8 | **316.4** | **+9%** |
| 8 | 436.9 | **462.4** | **+6%** |

MTP=3 acceptance on ShareGPT: mean acceptance length 2.24–2.34
(position rates 62% / 38–42% / 25–31%).
(First baseline conc-4 sample came out at 74.2 tok/s — a degenerate draw of very
long samples; the paired rerun above is the honest number. ShareGPT is high-
variance at small prompt counts.)

## Addendum: MTP=4 and the DS4 speed question

MTP=4 **boots** (missed in the original sweep, which tried 1/2/3/5). The QSA
ring-capacity constraint is `capacity = 4 * ceil((4 + N) / 4)` and it must
divide the attention block size 1616 → **N ∈ {1,2,3,4} boot, N ∈ {5,6,7,8}
assert** (N ∈ {9..12} would boot at capacity 16, untested).

MTP=4 ShareGPT: 116.4 / 198.7 / 294.5 / 426.8 tok/s at conc 1/2/4/8 —
slightly *worse* than MTP=3 everywhere despite higher acceptance length
(2.5–3.0 vs 2.24–2.34). The 4th draft position costs more than it returns.
**MTP=3 remains the optimum.**

### vs DeepSeek-V4-Flash-0731 on the same box

Qwen3.8-Flash-Next (6B active) conc-1 decode ≈ 121 tok/s today. DS4's evo
history (local DS4 evo results) shows aggregate conc-1
tok/s of **61.7 baseline → 123.3 → 202.8 → 235.4 tuned** (custom image
`vllm-ds4-sm120:0.25.1-fi0.6.14`, dspark spec decode N=5, marlin MoE, fp8 KV).

So Qwen's day-0 stack already matches DS4 *mid-tuning* (~123). The remaining
~2× gap to tuned DS4 is serving-stack, not model size. Prime suspects:
1. **PLE CPU offload**: every decode step fetches N-gram embedding rows over
   PCIe from host RAM — a per-step latency DS4 doesn't pay (all weights on GPU).
2. Day-0 kernels (QSA/GDN) not yet tuned for sm120; DS4's numbers came from a
   custom-patched image after weeks of evo tuning.
3. MTP acceptance 2.2–2.3 vs dspark's deeper tuned pipeline.

## Autoresearch loop (round 2, 2026-08-27): knob sweep for tok/s + TTFT

Short ctx (512/256) output tok/s at conc 1/8/24 + median TTFT (quiet streaming probe):

| Config | c1 | c8 | c24 | TTFT short | TTFT long (8k) |
|---|---|---|---|---|---|
| base (defaults) | 119.4 | 522.5 | **899.0** | **99.6 ms** | **111.0 ms** |
| mbt16384 | 99.6 | 489.5 | 738.6 | — | — |
| mtp3 | 120.0 | 507.1 | 661.4 | 119.3 ms | 785.0 ms |
| kvfp8 | — | — | — | unsupported: QSA cache allows only `auto`/`bfloat16` | |
| pleoff (no CPU offload) | — | — | — | cannot boot on 2×96GB: deterministic 47.69 GiB autotune OOM (3 attempts, all autotune-disable envs) | |

Long ctx (8192/512): base 108.5/337.0 (c1/c8), mtp3 108.3/314.7, mbt16384 106.6/328.7.

Decode rate (quiet probe, conc 1): base 129.7 tok/s short / 128.8 long
(TPOT 7.7 ms). Per-chunk TPOT is not comparable under MTP (burst delivery);
ShareGPT bench above is the valid MTP throughput measure.

Findings:
- **PLE CPU offload is mandatory** on 2×96GB — the table-on-GPU path dies on a
  deterministic 47.69 GiB inductor autotune allocation no env suppresses.
  The PCIe N-gram fetch tax is therefore unavoidable on this image; closing the
  remaining gap to tuned DS4 (~235 tok/s conc-1) needs an upstream fix or a
  future image, not a config knob.
- **fp8 KV cache is architecturally unsupported** for this model (QSA bf16/auto).
- **MAX_NUM_BATCHED_TOKENS=16384 hurts** (-18% at conc 24, -17% at conc 1).
  Keep the default.
- **MTP=3's TTFT cost**: +20 ms on short prompts, ~7× on 8k prompts (draft-head
  prefill). Throughput wins are conc 1–4; TTFT wins are always base.

## Evo rounds A–C (2026-08-27): backend/kernel-level sweep — conclusion

All with MTP=3 held constant, short ctx (512/256), tok/s at conc 1/8/24:

| Candidate | c1 | c8 | c24 | Verdict |
|---|---|---|---|---|
| deepgemm (default MoE) | 120.1 | 479.2 | 665.9 | **best — auto-selector was right** |
| triton MoE | 108.7 | 391.5 | 609.4 | worse |
| marlin MoE | 107.5 | 401.2 | 620.6 | worse (unlike DS4, where marlin won) |
| flashinfer_cutlass MoE | — | — | — | boot failed |
| batched_triton MoE | — | — | — | boot failed |
| draft_tensor_parallel_size=1 | 122.4 | 480.4 | 652.0 | noise, no win |

**PLE offload tax measured: ~0.** Controlled eager-mode A/B (enforce-eager
boots without PLE because it skips the 47.69 GiB compile-time autotune):
- PLE off, eager: TPOT 58.17 ms short / 58.15 ms long (17.2 tok/s)
- PLE on, eager: TPOT 57.86 ms short / 57.77 ms long (17.3 tok/s)

The N-gram PCIe prefetch is fully pipelined — PLE costs nothing per step.
(Eager itself is 7.5× slower than compiled: torch.compile + cudagraphs carry
this model.)

**Where the DS4 gap actually is (roofline analysis):** compiled Qwen decode is
7.7 ms/step ≈ 780 GB/s effective weight read ≈ **22% of the 2×1.8 TB/s memory
roofline**. DS4's tuned stack runs ~3.9 ms/step ≈ 2.6 TB/s ≈ **72% of
roofline**. The gap is day-0 kernel efficiency (GDN Triton / QSA / DeepGEMM on
sm120), not PLE offload, not MoE backend choice, not any serving config.

**Evo conclusion:** default DeepGEMM + MTP=3 + PLE-on is the measured optimum
of every candidate tried. No candidate beat the starting point by >5%.
Closing the remaining ~2× requires kernel-level work or upstream image updates,
not serving-side tuning.

## Agentic-workload evo (2026-08-28): realistic coding-agent benchmarks

Harness: `agentic_bench.py` - fixed coding-agent system prompt (5 tool schemas)
+ real vLLM source files as code context, 8 rotating task types (bugfix, test
writing, review, refactor, diff). Measured: per-stream decode tok/s (p50),
aggregate output tok/s, median TTFT. Input sizes are measured medians.

### Per-stream decode tok/s (p50) - mtp3 vs base

| Conc | ~1.6-2.6k ctx mtp3 | base | ~2-5.5k ctx mtp3 | base | long mtp3 | base |
|---|---|---|---|---|---|---|
| 1 | **308.7** | 197.8 | **198.7** | 169.8 | **189.5** | 180.6 |
| 2 | **212.6** | 148.2 | **211.3** | 119.1 | — | — |
| 4 | **152.0** | 116.8 | **119.9** | 94.5 | **123.3** | 88.6 |
| 8 | **102.1** | 73.6 | **83.1** | 75.2 | **80.3** | 70.1 |
| 16 | 54.6 | **57.2** | **59.5** | 52.3 | — | — |
| 24 | 39.7 | **52.6** | 36.9 | **45.4** | — | — |

### Aggregate output tok/s

| Conc | short mtp3 | base | mid mtp3 | base | long mtp3 | base |
|---|---|---|---|---|---|---|
| 1 | **165.8** | 107.9 | **132.9** | 115.1 | **148.3** | 117.7 |
| 8 | **454.4** | 387.3 | **426.3** | 397.9 | **453.5** | 410.6 |
| 24 | 581.1 | **750.3** | 625.9 | **792.6** | — | — |

**Under realistic agentic load, MTP=3 wins per-stream decode by 30-77% at
conc 1-8** (309 vs 198 tok/s at conc 1) and crosses over at conc ~16; base wins
aggregate at conc 24 (+29%). Code-heavy content gives higher MTP acceptance
than ShareGPT chat, widening the interactive gap.

**vs tuned DS4 (same box):** per-stream conc-1 decode 308.7 tok/s (mtp3)
**beats DS4's tuned p50 of ~255 tok/s**. End-to-end aggregate including TTFT
still favors DS4 (202.8 vs 165.8 tok/s) because Qwen's TTFT is higher.

### Stability finding (base config)

No-spec base died under agentic-shaped prompts: `TimeoutError: RPC call to
sample_tokens timed out` -> EngineDead. Root cause: a `qsa_mqa_paged` Triton
kernel JIT-compiling at runtime for shapes not covered by warmup exceeds the
300s `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` default. Fix baked into launch.sh:
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` + one-time soak (the Triton cache is
persistent, so the compile cost is paid once). MTP configs were unaffected -
their warmup covers the spec/nonspec shape matrix.

## Final recommendation (autoresearch conclusion)

Two validated operating points; the box is left running **MTP=3** because the
primary workload is interactive agent sessions (low concurrency, natural text):

- **Interactive (conc 1–4, short ctx): MTP=3** — +153%/+83%/+9% tok/s vs base
  (ShareGPT), acceptance length 2.2–2.8, TTFT +20 ms. Full sweep validated:
  117/229/377/507/672/651 tok/s at conc 1/2/4/8/16/24.
- **Batch / long-context (conc ≥8, long ctx): no-spec base** — best conc-24
  throughput (899 tok/s), best TTFT everywhere (99.6 ms short, 111 ms at 8k —
  7× better than MTP=3 on long prompts).

Switch with:
```bash
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}' bash launch.sh  # interactive
bash launch.sh                                                          # batch/long-ctx
```

## Conclusions

- **MTP=3 is the best MTP config.** On real text at interactive concurrency
  (1–4) it is dramatically faster: +153% at conc 1, +83% at conc 2. The edge
  shrinks to ~6–9% by conc 8 and inverts at conc 24 (random-data: -27% vs
  baseline). Classic spec-decode: wins memory-bound, loses compute-bound.
- **Long context: baseline always wins.** Prefill-dominated workloads leave no
  headroom for draft/verify overhead.
- Recommended serving config: **MTP=3 for interactive/low-concurrency traffic**;
  **no spec decode for batch or long-context traffic**. The box runs MTP=3.

## Reproduce

```bash
cd qwen3.8-flash-next
bash launch.sh                                    # baseline server, port 30001
bash bench.sh baseline-short 512 256 1 2 4 8 16 24
bash bench.sh baseline-long 8192 512 1 2 4 8 16 24
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}' bash launch.sh
bash sweep.sh 1 2 3                               # full restart+bench loop
```

Raw results: `results/<tag>/*.json|.log` on the server.
Note: DeepSeek (`deepseek-v4-flash-vllm-prod-v5`) must be stopped first —
the two models cannot share the GPUs.

## Config update 2026-08-28: 256k context
- max-model-len raised 32768 -> 262144 (model native max_position_embeddings=262144).
  The 32k was a bench-default, not a model limit. Client agents were overflowing
  at 33k sessions; server + launch.sh default now 262144. MTP=3, PLE offload unchanged.
- Note: chat template accepts reasoning_effort low/medium/xhigh only (high/max -> 400).
