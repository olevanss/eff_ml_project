"""
Benchmark sglang server with speculative decoding.
Run baseline_speed.py first (against a server WITHOUT spec decoding),
then paste the result into BASELINE_TOK_S below.
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
OUT_FILE = "spec_results.jsonl"

BASELINE_TOK_S = None

DRAFT_GAMMA = 5

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


def fetch_spec_metrics():
    try:
        resp = requests.get(f"{SERVER_URL}/metrics", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [warn] /metrics unreachable: {e}")
        return {}

    out = {}
    for line in resp.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if "spec" not in line.lower():
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        key = parts[0].split("{")[0].strip()
        try:
            out[key] = float(parts[1])
        except ValueError:
            pass
    return out


def bench():
    results = []

    print(f"server        : {SERVER_URL}  (speculative decoding)")
    print(f"model         : {MODEL}")
    print(f"runs          : {N_RUNS}  |  max_tokens={MAX_TOKENS}  |  gamma={DRAFT_GAMMA}")
    if BASELINE_TOK_S is not None:
        print(f"baseline      : {BASELINE_TOK_S:.1f} tok/s  (from baseline_speed.py)")
    else:
        print(f"baseline      : not set — speedup won't be shown")
    print("-" * 60)

    print("warmup... ", end="", flush=True)
    warmup()
    print("done\n")

    for i, prompt in enumerate(PROMPTS):
        r = run_request(prompt)
        results.append(r)
        print(
            f"[{i+1:2d}/{N_RUNS}]  "
            f"{r['tok_s']:6.1f} tok/s  |  "
            f"ttft {r['ttft_ms']:.0f} ms  |  "
            f"total {r['total_ms']:.0f} ms"
        )

    return results


def summary(results):
    speeds = [r["tok_s"] for r in results]
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    mean_speed = statistics.mean(speeds)

    s = sorted(speeds)
    p50 = s[len(s) // 2]
    p95 = s[max(0, int(len(s) * 0.95) - 1)]

    print("\n=== throughput (with spec decoding) ===")
    print(f"  mean tok/s  : {mean_speed:.1f}")
    print(f"  p50  tok/s  : {p50:.1f}")
    print(f"  p95  tok/s  : {p95:.1f}")
    if ttfts:
        print(f"  mean ttft   : {statistics.mean(ttfts):.0f} ms")
        print(f"  p95  ttft   : {sorted(ttfts)[max(0, int(len(ttfts)*0.95)-1)]:.0f} ms")

    if BASELINE_TOK_S is not None and BASELINE_TOK_S > 0:
        speedup = mean_speed / BASELINE_TOK_S
        print(f"\n=== speedup ===")
        print(f"  baseline    : {BASELINE_TOK_S:.1f} tok/s")
        print(f"  with spec   : {mean_speed:.1f} tok/s")
        print(f"  speedup     : {speedup:.2f}x")

    spec = fetch_spec_metrics()

    print("\n=== speculative decoding metrics ===")
    if not spec:
        print("  no spec metrics — is speculative decoding enabled on this server?")
        print("  (launch with --speculative-algo EAGLE or similar)")
        return

    for key, val in spec.items():
        label = key.split(":")[-1]
        print(f"  {label}: {val:.4f}")

    acc_key = next((k for k in spec if "accept_length_mean" in k), None)
    if acc_key:
        mean_acc = spec[acc_key]
        acceptance_rate = min(mean_acc / DRAFT_GAMMA, 1.0)
        print(f"\n  mean accepted tokens/step : {mean_acc:.2f}  (gamma={DRAFT_GAMMA})")
        print(f"  acceptance rate           : {acceptance_rate:.2%}")


if __name__ == "__main__":
    results = bench()

    with open(OUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nsaved → {OUT_FILE}")

    summary(results)
