# GLM-5.3-Flash-NVFP4 — deploy template

Serving template for `LibertAIDAI/GLM-5.3-Flash-NVFP4` on the ASUS TRX50
inference server.

## Status

- vLLM image `vllm/vllm-openai:glm53-flash-x86_64-cu130`: pulled, verified
  (`glm5_next` support present, flashinfer 0.6.17, CUDA 13.0.1).
- Model checkpoint: downloaded to `$HOME/.cache/huggingface/hub/`
  (181 GB, 120 shards, validated).
- **Deployed 2026-08-26 (first 2× RTX PRO 6000 attempt). Server boots and
  serves, but generation is broken on sm120 — do not use for benchmarks.**
  Verified failure matrix:
  - fp8 KV cache: hard crash (`concat_and_cache_mla: pe_dim must be 64 for
    fp8_ds_mla`). GLM-5.3 is NoPE (`qk_rope_head_dim=0`); unfixable in config.
    bf16 KV works.
  - Sparse indexer: only sm120 backend (`FLASHINFER_MLA_SPARSE_SM120`)
    requires fp8_ds_mla → dead. Workaround: null `text_config.index_topk` in
    the snapshot config (dense MLA; exact at ≤2048-token contexts) + patch
    `glm5next model.py` to skip indexer weights on load.
  - MLA prefill: no kernel supports dims (256,0,256) on sm120 (selector treats
    sm120 as "Hopper and older" → FLASH_ATTN only, dims not whitelisted).
    Patched whitelist in `mla/prefill/flash_attn.py`.
  - Multimodal profiling eats several GiB → `--limit-mm-per-prompt 0` needed.
  - With all fixes + bf16 KV: serves, KV ~132k tokens @ util 0.99, BUT output
    is garbage: `locklocklock` via FLASHINFER_CUTLASS and VLLM_CUTLASS FP4 MoE
    (marlin OOMs at repack); echo-loops via EMULATION MoE (exact dequant).
    → FP4 MoE kernels broken on sm120 AND KDA/mHC attention stack miscomputes.
  - Working launcher: `launch.sh` (not `docker-run.sh` — that one is buggy:
    duplicate run block, unset PORT/MODEL, extra `serve` word vs image
    entrypoint). Patches in `patches/`, mounted read-only.
  - Upstream: vllm#53906 and sglang#36507 both open/unmerged (2026-08-26).

## Deploy

```bash
sudo bash $HOME/glm-5.3-flash-nvfp4/docker-run.sh
```

Serves on port **30001** as `glm-5.3-flash-nvfp4`.

## Verify after deploy

```bash
docker logs -f glm-5.3-flash-nvfp4-prod
curl http://127.0.0.1:30001/v1/models
```

## Configuration notes

- `--tensor-parallel-size 2` — both RTX PRO 6000 GPUs.
- `--gpu-memory-utilization 0.97` — 181.3 GB weights in 195.8 GB VRAM leaves
  ~9 GB for KV cache + activations. Tight; tune down `--max-model-len` if
  OOM on first load.
- `--kv-cache-dtype fp8` — matches the DeepSeek server; saves KV memory.
- `--max-model-len 32768` — conservative default. Raise only after verifying
  actual KV capacity on first deploy.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'` — MTP
  spec decoding. The checkpoint ships the MTP draft layer (layer 45); the
  glm53 image implements `Glm5NextMTP`. Recipe: vllm recipes
  `zai-org/GLM-5.3-Flash`. Set `SPEC_CONFIG=""` in `docker-run.sh` to disable.
- Kernel caches use fresh dirs (`$HOME/.cache/vllm-docker-glm53/`) so this
  build never reuses DeepSeek's compiled kernels.
- Env mirrors the DeepSeek container: `NCCL_P2P_DISABLE=1`,
  `NCCL_CUMEM_ENABLE=0`, `FLASHINFER_DISABLE_VERSION_CHECK=1`.

## Impact vs DeepSeek-V4-Flash-0731 (current prod)

- Weights: GLM 181.3 GB vs DS4 156 GB (FP8). GLM is 25 GB heavier.
- KV headroom: GLM ~9 GB (0.97 util) vs DS4 ~28 GB (0.94 util) — ~3x less.
- Concurrency: expect roughly 1/3 the concurrent streams at equal context.
  To keep 24 streams, context must drop to ~1/3 of DS4's 262k. Measure actual
  KV capacity on first deploy before raising `--max-model-len`.
## Known risks (deploy-time checks)

- **fp8 KV cache crash on sm120 (RTX PRO 6000).** HF discussion on the base
  model (zai-org/GLM-5.3-Flash #19): a 4× RTX 6000 Pro Blackwell user hit
  `concat_and_cache_mla ... pe_dim must be 64 for fp8_ds_mla` with the
  glm53-flash image, even with `--kv-cache-dtype auto` (the SM120 backend
  defaults to fp8_ds_mla). Their config differed (EP, 256 seqs, CPU offload);
  ours may not trigger it. If it does, try `--kv-cache-dtype bf16` (costs KV
  capacity) or drop to `--kv-cache-dtype fp8_e5m2`.
- **No confirmed 2× RTX 6000 run of the NVFP4 checkpoint.** HF discussion on
  the NVFP4 repo (#3) asks exactly this and is unanswered. SGLang's verified
  GLM-5.3-Flash hardware list (H100/H200/B200/B300/GB200/GB300) excludes
  sm120. First deploy is the first verification.
- **MTP tok/s: no public GLM-5.3-Flash numbers.** Closest data is GLM-5.2 MTP
  on the same sm120 hardware (4× RTX PRO 6000): 73/64/52 tok/s short,
  44/41/32 @123K with MTP vs 55/51/41 base (jarrelscy/vllm-glm52-sm120).
  vLLM's GLM.md recommends `num_speculative_tokens 1` for GLM-4.x (acceptance
  drops at higher ns); the recipe default is 5. SGLang recommends disabling
  spec decode for heavily batched traffic. Expect to tune ns at deploy.
- Kernel caches use fresh dirs (`$HOME/.cache/vllm-docker-glm53/`) so this
  build never reuses DeepSeek's compiled kernels.
- Env mirrors the DeepSeek container: `NCCL_P2P_DISABLE=1`,
  `NCCL_CUMEM_ENABLE=0`, `FLASHINFER_DISABLE_VERSION_CHECK=1`.

## Impact vs DeepSeek-V4-Flash-0731 (current prod)

- Weights: GLM 181.3 GB vs DS4 156 GB (FP8). GLM is 25 GB heavier.
- KV headroom: GLM ~9 GB (0.97 util) vs DS4 ~28 GB (0.94 util) — ~3x less.
- Concurrency: expect roughly 1/3 the concurrent streams at equal context.
  To keep 24 streams, context must drop to ~1/3 of DS4's 262k. Measure actual
  KV capacity on first deploy before raising `--max-model-len`.
- Spec decoding: MTP weights CONFIRMED in this checkpoint (layer 45, 2617
  tensors: eh_proj/enorm/hnorm/experts; the earlier note claiming no MTP
  weights was wrong). The glm53 image implements `Glm5NextMTP`. Untested on
  sm120 — blocked behind the kernel failures above.

## Model

- Repo: `LibertAIDAI/GLM-5.3-Flash-NVFP4` (weight-only NVFP4, 181 GB)
- Base: `zai-org/GLM-5.3-Flash` (320B / 18B-active MoE, multimodal)
- License: MIT
- Runtime: `glm5_next` not in vLLM main yet (PR vllm#53906 in flight); this
  image is the dedicated per-model build.
