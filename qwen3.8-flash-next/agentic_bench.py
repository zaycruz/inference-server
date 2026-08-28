#!/usr/bin/env python3
"""Agentic-workload benchmark: realistic coding-agent prompts, streaming metrics.

Runs inside the container. Builds prompts from a fixed coding-agent system
prompt (with tool schemas) + real vLLM source files as code context, then
fires concurrent streaming requests and reports TTFT, per-stream decode tok/s,
and aggregate throughput.

Usage: python3 agentic_bench.py <tag> <profile:s|m|l> <concurrencies...>
"""
import asyncio
import glob
import json
import statistics
import sys
import time

from openai import AsyncOpenAI

TAG = sys.argv[1]
PROFILE = sys.argv[2]
CONCS = [int(x) for x in sys.argv[3:]] or [1, 2, 4, 8, 16, 24]

TARGET_CHARS = {"s": 8_000, "m": 32_000, "l": 64_000}[PROFILE]  # ~2k/8k/16k tokens
MAX_TOKENS = 400

SYSTEM_PROMPT = """You are an expert coding agent operating in a terminal. You help users with software engineering tasks by reading code, making edits, running commands, and verifying results.

You have access to the following tools:

[
  {"type": "function", "function": {"name": "read_file", "description": "Read a file from the filesystem with optional line offset and count", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute path to the file"}, "offset": {"type": "integer"}, "count": {"type": "integer"}}, "required": ["path"]}}},
  {"type": "function", "function": {"name": "edit_file", "description": "Replace an exact string in a file with new content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
  {"type": "function", "function": {"name": "run_command", "description": "Execute a bash command and return stdout and stderr", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_seconds": {"type": "integer"}}, "required": ["command"]}}},
  {"type": "function", "function": {"name": "search_code", "description": "Search file contents using a regular expression", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "file_glob": {"type": "string"}}, "required": ["pattern"]}}},
  {"type": "function", "function": {"name": "write_file", "description": "Create or fully overwrite a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}
]

Guidelines:
- Make minimal, targeted changes. Do not refactor beyond what the task requires.
- Verify every change by running the relevant tests or a focused check.
- When diagnosing a failure, form a hypothesis first, then test it with the cheapest experiment.
- Keep code style consistent with the surrounding file: naming, comment density, structure.
- Never commit, push, or otherwise mutate version control state unless explicitly asked.
- When you finish, summarize what changed and how you verified it.
"""

TASKS = [
    "Find the bug in this code and explain the root cause in one paragraph, then show the minimal fix as a diff.",
    "Write a focused unit test for the most important function in this file.",
    "This code has a race condition. Identify where it is and explain how to fix it without changing the public interface.",
    "Refactor the longest function in this file for readability. Keep behavior identical.",
    "Explain what this module does, its key abstractions, and any design flaws you notice.",
    "Write the diff to add proper error handling to the I/O paths in this code.",
    "Review this code as a senior engineer. List the three most important issues, ordered by severity.",
    "Add type annotations to the public functions in this file. Show the result as a diff.",
]


def build_prompts():
    files = sorted(
        glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/v1/**/*.py", recursive=True)
    )
    texts = []
    for f in files:
        try:
            with open(f) as fh:
                texts.append((f, fh.read()))
        except Exception:
            pass
    prompts = []
    need = TARGET_CHARS
    i = 0
    while len(prompts) < 32:
        path, text = texts[i % len(texts)]
        chunk = text[:need]
        task = TASKS[i % len(TASKS)]
        user = (
            f"Here is the file `{path}`:\n\n```python\n{chunk}\n```\n\n"
            f"Task: {task}"
        )
        prompts.append(user)
        i += 1
    return prompts


async def one_request(client, system, user, results, idx):
    t0 = time.perf_counter()
    try:
        stream = await client.chat.completions.create(
            model="qwen3.8-flash-next",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.6,
            stream=True,
            stream_options={"include_usage": True},
        )
        t_first = None
        t_last = t0
        usage = None
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            piece = (
                (getattr(d, "reasoning", None) or "")
                + (getattr(d, "reasoning_content", None) or "")
                + (d.content or "")
            )
            if piece:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
        wall = time.perf_counter() - t0
        out_tokens = usage.completion_tokens if usage else MAX_TOKENS
        in_tokens = usage.prompt_tokens if usage else 0
        decode_s = (t_last - t_first) if t_first else wall
        results.append({
            "idx": idx,
            "ttft_ms": ((t_first - t0) * 1000) if t_first else None,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "decode_tok_s": (out_tokens / decode_s) if decode_s > 0 else None,
            "wall_s": wall,
        })
    except Exception as e:
        results.append({"idx": idx, "error": str(e)[:120]})


async def run_conc(client, system, prompts, conc):
    n = max(conc * 3, 6)
    selected = [prompts[i % len(prompts)] for i in range(n)]
    sem = asyncio.Semaphore(conc)
    results = []

    async def guarded(u, idx):
        async with sem:
            await one_request(client, system, u, results, idx)

    t0 = time.perf_counter()
    await asyncio.gather(*(guarded(u, i) for i, u in enumerate(selected)))
    wall = time.perf_counter() - t0
    ok = [r for r in results if "error" not in r and r.get("decode_tok_s")]
    total_out = sum(r["out_tokens"] for r in ok)
    row = {
        "conc": conc,
        "n": len(ok),
        "errors": len(results) - len(ok),
        "median_in_tokens": int(statistics.median(r["in_tokens"] for r in ok)) if ok else 0,
        "median_ttft_ms": round(statistics.median(r["ttft_ms"] for r in ok), 1) if ok else None,
        "p50_decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in ok), 1) if ok else None,
        "agg_output_tok_s": round(total_out / wall, 1) if wall > 0 else None,
    }
    print("AGENTIC " + json.dumps({"tag": TAG, "profile": PROFILE, **row}), flush=True)
    return row


async def main():
    client = AsyncOpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30001/v1")
    prompts = build_prompts()
    for c in CONCS:
        await run_conc(client, SYSTEM_PROMPT, prompts, c)


asyncio.run(main())
