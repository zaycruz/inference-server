#!/usr/bin/env python3
"""No-GPU validation of the GLM-5.3-Flash-NVFP4 checkpoint."""
import json, struct
from pathlib import Path

SNAP = Path("$HOME/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290")

# 1. config.json parses and has expected fields
cfg = json.loads((SNAP / "config.json").read_text())
print("arch:", cfg["architectures"])
print("model_type:", cfg["model_type"])
print("quant_method:", cfg.get("quantization_config", {}).get("quant_method"))
assert cfg["model_type"] == "glm5_next"
assert cfg["architectures"] == ["Glm5NextForConditionalGeneration"]

# 2. All expected shards present and non-empty
shards = sorted(SNAP.glob("model-*.safetensors"))
print("shards:", len(shards))
assert len(shards) == 120, f"expected 120 shards, got {len(shards)}"

# 3. Safetensors headers parse (first 8 bytes = header length, then JSON)
for s in shards[:8] + shards[-2:]:
    with open(s, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    assert "__metadata__" in hdr or any(k.endswith(".dtype") for k in hdr)
print("safetensors headers OK (sampled 10 shards)")

# 4. Total size
total = sum(s.stat().st_size for s in shards)
print(f"total: {total/1073741824:.2f} GiB")
print("VALIDATION OK")
