#!/usr/bin/env python3
"""TTFT/TPOT probe: streaming requests at conc 1, reports medians.
Runs inside the container (needs `openai` package). Usage:
  python3 probe.py <short|long> [n_requests] [max_tokens]
"""
import json
import statistics
import sys
import time

from openai import OpenAI

profile = sys.argv[1] if len(sys.argv) > 1 else "short"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30001/v1")

if profile == "long":
    # ~8k tokens of varied prose
    base = ("The history of computation spans mechanical calculators, vacuum tubes, "
            "transistors, integrated circuits, and modern accelerators. Each era brought "
            "new abstractions and new bottlenecks. ")
    prompt = "Summarize the following text.\n\n" + base * 260
else:
    prompt = ("Explain how speculative decoding works in large language model serving, "
              "including draft models, verification, and acceptance rates. ") * 8

ttfts, tpots = [], []
for i in range(n):
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model="qwen3.8-flash-next",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
    )
    t_first = None
    ntok = 0
    t_last = t0
    for chunk in stream:
        delta = chunk.choices[0].delta
        piece = (
            (getattr(delta, "reasoning", None) or "")
            + (getattr(delta, "reasoning_content", None) or "")
            + (delta.content or "")
        )
        if piece:
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            ntok += 1
            t_last = now
    if t_first is None:
        continue
    ttfts.append((t_first - t0) * 1000)
    if ntok > 1:
        tpots.append((t_last - t_first) * 1000 / (ntok - 1))

out = {
    "profile": profile,
    "n": n,
    "median_ttft_ms": round(statistics.median(ttfts), 1) if ttfts else None,
    "median_tpot_ms": round(statistics.median(tpots), 2) if tpots else None,
    "decode_tok_s": round(1000 / statistics.median(tpots), 1) if tpots else None,
}
print("PROBE " + json.dumps(out))
