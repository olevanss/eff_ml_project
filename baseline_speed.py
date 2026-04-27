"""
Run this against a sglang server launched WITHOUT speculative decoding.
The result goes into BASELINE_TOK_S in metrics_1.py.

Example launch (no spec decoding):
  python -m sglang.launch_server --model-path /path/to/model --port 30001
"""

import itertools
import json
import statistics
import time

import requests

SERVER_URL = "http://127.0.0.1:30000"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_TOKENS = 256
TEMPERATURE = 0.0
N_RUNS = 20
OUT_FILE = "baseline_results.jsonl"

PROMPTS = [
    "Implement binary search in Python and add a few unit tests.",
    "Write a thread-safe LRU cache class in Python.",
    "Explain what the GIL is and how asyncio works around it.",
    "What's the difference between a process and a thread? Give a concrete example.",
    "Write a simple TCP echo server in Python without any frameworks.",
    "What is the CAP theorem and why does it matter for distributed systems?",
    "Implement merge sort in Python and explain its time complexity.",
    "Write a producer-consumer pattern using asyncio.Queue.",
    "Explain Python's memory model and how garbage collection works in CPython.",
    "What are context managers? Show how to write a custom one with __enter__/__exit__.",
    "Write a function that detects cycles in a linked list.",
    "Explain the difference between optimistic and pessimistic locking in databases.",
]

PROMPTS = list(itertools.islice(itertools.cycle(PROMPTS), N_RUNS))


def warmup():
    try:
        requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=60,
        )
    except Exception:
        pass


def run_request(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    t0 = time.perf_counter()
    ttft = None
    text = ""

    with requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                text += delta

    elapsed_s = time.perf_counter() - t0
    n_toks = len(text.split())
    return {
        "ttft_ms": ttft,
        "total_ms": elapsed_s * 1000,
        "tokens": n_toks,
        "tok_s": n_toks / elapsed_s if elapsed_s > 0 else 0.0,
    }


def main():
    print(f"server : {SERVER_URL}  (NO speculative decoding)")
    print(f"model  : {MODEL}")
    print(f"runs   : {N_RUNS}")
    print("-" * 55)

    print("warmup... ", end="", flush=True)
    warmup()
    print("done\n")

    results = []
    for i, prompt in enumerate(PROMPTS):
        r = run_request(prompt)
        results.append(r)
        print(
            f"[{i+1:2d}/{N_RUNS}]  "
            f"{r['tok_s']:6.1f} tok/s  |  "
            f"ttft {r['ttft_ms']:.0f} ms  |  "
            f"total {r['total_ms']:.0f} ms"
        )

    speeds = [r["tok_s"] for r in results]
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    mean_speed = statistics.mean(speeds)

    with open(OUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== baseline (autoregressive) ===")
    print(f"  mean tok/s : {mean_speed:.1f}")
    print(f"  p50  tok/s : {sorted(speeds)[len(speeds)//2]:.1f}")
    print(f"  mean ttft  : {statistics.mean(ttfts):.0f} ms")
    print(f"\nsaved → {OUT_FILE}")

    print(f"\n{'='*55}")
    print(f"  paste into metrics_1.py:")
    print(f"  BASELINE_TOK_S = {mean_speed:.2f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
