import itertools
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

PY = "python3"
BENCH_SCRIPT = "bench_russian.py"

MAIN_MODEL = "/home/jovyan/T-pro-it-2.1"
DRAFT_MODEL = "/home/jovyan/T-pro-it-2.0-eagle"
SPECULATIVE_ALGORITHM = "EAGLE"

OUTPUT = "results.jsonl"
NUM_PROMPTS = 16

# --- Grid ---
BATCH_SIZE = 1
INCLUDE_BASELINE = True
STEPS = [1, 2, 3, 4]
TOPK = [1, 2, 4, 8]
NUM_DRAFT_TOKENS = [2, 4, 8, 16, 32]

MAX_RUNS: Optional[int] = None

CONTINUE_ON_ERROR = True
MAX_RETRIES = 0
RETRY_SLEEP_SEC = 3

TOP_N_REPORT = 10


@dataclass(frozen=True)
class SpecConfig:
    batch_size: int
    steps: int
    topk: int
    num_draft_tokens: int


def is_valid_spec_config(steps: int, topk: int, num_draft_tokens: int) -> bool:
    if steps <= 0:
        max_verify_tokens = 0
    else:
        max_verify_tokens = 1 + topk + (steps - 1) * (topk**2)

    if num_draft_tokens > max_verify_tokens:
        return False
    if (steps == 0 or topk == 0 or num_draft_tokens == 0) and (
        steps + topk + num_draft_tokens != 0
    ):
        return False
    return True


def config_key(c: SpecConfig) -> Tuple:
    return (c.batch_size, c.steps, c.topk, c.num_draft_tokens)


def load_existing_records(path: str) -> Tuple[Dict[Tuple, dict], List[dict]]:
    by_key: Dict[Tuple, dict] = {}
    records: List[dict] = []
    if not os.path.exists(path):
        return by_key, records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            records.append(rec)
            key = (
                int(rec.get("batch_size", 0)),
                int(rec.get("steps", 0)),
                int(rec.get("topk", 0)),
                int(rec.get("num_draft_tokens", 0)),
            )
            by_key[key] = rec
    return by_key, records


def score_record(r: dict, baseline_speed: Optional[float]) -> float:
    spd = float(r.get("speed", 0.0))
    if baseline_speed and baseline_speed > 0:
        return spd / baseline_speed
    return spd


def fmt_cfg(r: dict) -> str:
    parts = [
        f"bs={r.get('batch_size')}",
        f"steps={r.get('steps')}",
        f"topk={r.get('topk')}",
        f"ndt={r.get('num_draft_tokens')}",
    ]
    return " ".join(parts)


def iter_configs(
    batch_size: int,
    steps: Sequence[int],
    topk: Sequence[int],
    num_draft_tokens: Sequence[int],
    include_baseline: bool,
) -> List[SpecConfig]:
    cfgs: List[SpecConfig] = []

    if include_baseline:
        cfgs.append(SpecConfig(batch_size=batch_size, steps=0, topk=0, num_draft_tokens=0))

    for s, k, ndt in itertools.product(steps, topk, num_draft_tokens):
        if not is_valid_spec_config(s, k, ndt):
            continue
        if s == 0 and k == 0 and ndt == 0:
            continue
        cfgs.append(SpecConfig(batch_size=batch_size, steps=s, topk=k, num_draft_tokens=ndt))

    cfgs.sort(key=lambda c: config_key(c))
    return cfgs


def build_cmd(cfg: SpecConfig) -> List[str]:
    return [
        PY,
        BENCH_SCRIPT,
        "--model-path", MAIN_MODEL,
        "--speculative-draft-model-path", DRAFT_MODEL,
        "--speculative-algorithm", SPECULATIVE_ALGORITHM,
        "--batch-size", str(cfg.batch_size),
        "--steps", str(cfg.steps),
        "--topk", str(cfg.topk),
        "--num_draft_tokens", str(cfg.num_draft_tokens),
        "--num-prompts", str(NUM_PROMPTS),
        "--output", OUTPUT,
    ]


def refresh_and_report(batch_size: int, include_baseline: bool) -> None:
    by_key, recs = load_existing_records(OUTPUT)

    baseline_speed: Optional[float] = None
    if include_baseline:
        base_key = (batch_size, 0, 0, 0)
        if base_key in by_key and "speed" in by_key[base_key]:
            baseline_speed = float(by_key[base_key]["speed"])

    scored = []
    for r in recs:
        if "acc_length" not in r or "speed" not in r:
            continue
        s = score_record(r, baseline_speed)
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    print("\nBest by score:")
    for s, r in scored[:TOP_N_REPORT]:
        print(
            f"  score={s:.4f} acc_length={float(r['acc_length']):.3f} "
            f"speed={float(r['speed']):.3f} tok/s step_time={float(r.get('step_time', 0))*1000:.2f}ms | {fmt_cfg(r)}"
        )


def run() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)) or ".", exist_ok=True)

    existing_by_key, _ = load_existing_records(OUTPUT)

    cfgs = iter_configs(
        batch_size=BATCH_SIZE,
        steps=STEPS,
        topk=TOPK,
        num_draft_tokens=NUM_DRAFT_TOKENS,
        include_baseline=INCLUDE_BASELINE,
    )

    # --- Resume: skip already completed configs ---
    todo = [c for c in cfgs if config_key(c) not in existing_by_key]

    if MAX_RUNS is not None:
        todo = todo[: max(0, int(MAX_RUNS))]

    total = len(cfgs)
    skipped = len(cfgs) - len(todo)
    print(f"OUTPUT={os.path.abspath(OUTPUT)}")
    print(f"Total valid configs: {total}. To run now: {len(todo)}. Skipped(resume): {skipped}.")

    if not todo:
        print("Nothing to run (everything already in OUTPUT).")
        refresh_and_report(batch_size=BATCH_SIZE, include_baseline=INCLUDE_BASELINE)
        return

    failures: List[Tuple[Tuple, int]] = []
    for idx, cfg in enumerate(todo, start=1):
        print("=" * 100)
        print(
            f"[{idx}/{len(todo)}] bs={cfg.batch_size} steps={cfg.steps} topk={cfg.topk} ndt={cfg.num_draft_tokens}"
        )
        print("=" * 100)

        cmd = build_cmd(cfg=cfg)
        attempt = 0
        while True:
            try:
                subprocess.run(cmd, check=True)
                break
            except subprocess.CalledProcessError as e:
                attempt += 1
                key = config_key(cfg)
                failures.append((key, e.returncode))
                print(f"[error] failed config {key} (exit={e.returncode}), attempt={attempt}/{MAX_RETRIES + 1}")
                if attempt > MAX_RETRIES:
                    if CONTINUE_ON_ERROR:
                        print("[error] continue_on_error=True, skipping this config")
                        break
                    raise
                time.sleep(RETRY_SLEEP_SEC)

        refresh_and_report(batch_size=BATCH_SIZE, include_baseline=INCLUDE_BASELINE)

    print("Done.")
    if failures:
        print(f"Failures: {len(failures)} (showing up to 20)")
        for key, code in failures[:20]:
            print(f" - {key} exit={code}")
    refresh_and_report(batch_size=BATCH_SIZE, include_baseline=INCLUDE_BASELINE)


if __name__ == "__main__":
    run()
