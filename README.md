# trx50 inference server

Self-hosted LLM serving on a single workstation: **2× NVIDIA RTX PRO 6000
Blackwell (96 GB each, sm120)**, Ubuntu, vLLM, Tensor Parallel 2.
Everything in this repo is a measured, reproducible serving recipe — launch
scripts, benchmark harnesses, and full reports with the numbers.

## Headline: Qwen3.8-Flash-Next (FP8) — baseline vs MTP=3

`Qwen/Qwen3.8-Flash-Next-FP8` (125B MoE, 6B active + 51B N-gram table),
vLLM TP=2, 256k context, MTP speculative decoding.

**Realistic agentic coding workloads** (tool schemas + real code context,
per-stream decode tok/s p50):

| Concurrency | MTP=3 | Base | Delta |
|---|---|---|---|
| 1 | **308.7** | 197.8 | **+56%** |
| 2 | **212.6** | 148.2 | **+43%** |
| 4 | **152.0** | 116.8 | **+30%** |
| 8 | **102.1** | 73.6 | **+39%** |
| 16 | 54.6 | **57.2** | crossover |
| 24 | 39.7 | **52.6** | base wins |

Same box, vs a tuned DeepSeek-V4-Flash stack: Qwen MTP=3 **wins per-stream
decode** (309 vs ~255 tok/s p50), loses end-to-end aggregate including TTFT
(166 vs 203 tok/s) — DS4 prefills faster.

MTP sweep findings: **N=3 is the optimum** (N=4 slower despite higher
acceptance, N≥5 fails a QSA ring-capacity assert). Spec decode is an
interactive-latency play: +30–77% per-stream at conc 1–8, inverts at high
batch and on prefill-dominated long-context traffic.

Full data — baseline sweeps at short/long context, MTP acceptance rates,
backend/kernel evo rounds, roofline analysis, and the DS4 comparison:
**[qwen3.8-flash-next/REPORT.md](qwen3.8-flash-next/REPORT.md)**

### Reproduce

```bash
cd qwen3.8-flash-next
# launch (MTP=3 interactive config; drop SPEC_CONFIG for no-spec base)
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":3}' bash launch.sh
bash bench.sh baseline-short 512 256 1 2 4 8 16 24   # throughput sweep
bash sweep.sh 1 2 3                                   # full MTP restart+bench loop
python3 agentic_bench.py                              # realistic agentic bench
```

### sm120 (RTX PRO 6000) deploy gotchas

- `--disable-custom-all-reduce` required (custom AR busy-waits on this box).
- PLE CPU offload is mandatory to fit the model, and hits a warmup deadlock
  ([vllm#53960](https://github.com/vllm-project/vllm/issues/53960)) — worked
  around by `patches/connector.py`, mounted over the installed module.
- fp8 KV cache unsupported for this model (QSA allows `auto`/`bfloat16` only).
- A `qsa_mqa_paged` Triton JIT on cold shapes can exceed vLLM's 300 s exec
  timeout → set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` (in `launch.sh`).
- Chat template accepts `reasoning_effort` of `low`/`medium`/`xhigh` only.

## Also in this repo

- **[glm-5.3-flash-nvfp4/](glm-5.3-flash-nvfp4/)** — deploy template +
  failure matrix for `LibertAIDAI/GLM-5.3-Flash-NVFP4` on the same box.
  Boots and serves on sm120 after several patches, but generation is broken
  (FP4 MoE kernels + KDA/mHC attention stack miscompute on sm120). Documented
  so the next person doesn't have to rediscover it.

## Layout

```
qwen3.8-flash-next/   launch.sh, MTP/kernel sweeps, agentic bench, REPORT.md
glm-5.3-flash-nvfp4/  GLM-5.3 deploy template + sm120 failure matrix
```
