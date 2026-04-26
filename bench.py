import argparse
import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import List

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.bench_serving import DatasetRow, benchmark, set_global_args
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)

PROMPTS_FILE = "prompts.json"


def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty JSON array in {path}")
    return data


def send_one_batch(base_url, num_prompts, batch_size, tokenizer, prompts):
    padded = (prompts * ((num_prompts + len(prompts) - 1) // len(prompts)))[:num_prompts]
    input_requests: List[DatasetRow] = [DatasetRow(p, 0, 512) for p in padded]

    set_global_args(SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=False,
        return_logprob=False,
        backend="sglang",
        dataset_name="custom",
        return_routed_experts=False,
        plot_throughput=False,
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
        warmup_requests=1,
        output_details=False,
    ))

    results = asyncio.run(
        benchmark(
            backend="sglang",
            api_url=f"{base_url}/generate",
            base_url=base_url,
            model_id="default",
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=float("inf"),
            max_concurrency=batch_size,
            disable_tqdm=False,
            lora_names=None,
            lora_request_distribution=None,
            lora_zipf_alpha=None,
            extra_request_body={},
            profile=None,
        )
    )

    assert results["completed"] == len(input_requests)
    acc_length = results["accept_length"] or 1.0

    real_speed = results.get("output_throughput")
    if real_speed is None:
        duration_s = results.get("duration")
        if duration_s:
            real_speed = results["total_output_tokens"] / duration_s

    server_info = requests.get(base_url + "/get_server_info").json()
    step_time = np.percentile(
        server_info["internal_states"][0]["step_time_dict"][str(batch_size)], 20
    )
    speed = acc_length / step_time

    return (
        round(acc_length, 3),
        round(step_time, 5),
        round(speed, 3),
        round(real_speed, 3) if real_speed is not None else None,
        results["total_output_tokens"] / results["completed"],
    )


def main(args, server_args):
    base_url = "http://127.0.0.1:20000"

    # --- Load prompts ---
    prompts = load_prompts(PROMPTS_FILE)

    configs = []
    for batch_size in args.batch_size:
        for steps in args.steps:
            for topk in args.topk:
                for num_draft_tokens in args.num_draft_tokens:
                    if steps <= 0:
                        max_verify_tokens = 0
                    else:
                        max_verify_tokens = 1 + topk + (steps - 1) * (topk**2)
                    if num_draft_tokens > max_verify_tokens:
                        continue
                    if (steps == 0 or topk == 0 or num_draft_tokens == 0) and (
                        steps + topk + num_draft_tokens != 0
                    ):
                        continue
                    configs.append((batch_size, steps, topk, num_draft_tokens))

    print(f"Configs to run: {len(configs)}")

    for i in range(args.start, args.end or len(configs)):
        batch_size, steps, topk, num_draft_tokens = configs[i]
        print(f"[{i}] {batch_size=}, {steps=}, {topk=}, {num_draft_tokens=}")

        if steps == 0:
            other_args = []
        else:
            other_args = [
                "--speculative-num-steps", steps,
                "--speculative-eagle-topk", topk,
                "--speculative-num-draft-tokens", num_draft_tokens,
            ]
            if server_args.speculative_draft_model_path is not None:
                other_args.extend([
                    "--speculative-draft-model-path", server_args.speculative_draft_model_path,
                    "--speculative-algorithm", server_args.speculative_algorithm,
                ])

        process = popen_launch_server(
            args.model_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={"SGLANG_RECORD_STEP_TIME": "1", **os.environ},
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=server_args.trust_remote_code
        )

        try:
            # --- Warmup ---
            send_one_batch(base_url, batch_size, batch_size, tokenizer, prompts)

            # --- Benchmark ---
            acc_length, step_time, speed, real_speed, completion_tokens = send_one_batch(
                base_url, max(args.num_prompts, batch_size), batch_size, tokenizer, prompts
            )
        finally:
            kill_process_tree(process.pid)

        print(
            f"  acc_length={acc_length:.3f}, speed={speed:.2f} tok/s/req, "
            f"real_speed={real_speed if real_speed is not None else 'n/a'} tok/s, "
            f"step_time={step_time * 1000:.2f} ms"
        )

        record = {
            "batch_size": batch_size,
            "steps": steps,
            "topk": topk,
            "num_draft_tokens": num_draft_tokens,
            "acc_length": acc_length,
            "step_time": step_time,
            "speed": speed,
            "speed_real": real_speed,
            "completion_tokens": completion_tokens,
        }

        with open(args.output, "a") as fout:
            fout.write(json.dumps(record) + "\n")

        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--batch-size", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--steps", type=int, nargs="+", default=(0, 1, 3, 5, 7))
    parser.add_argument("--topk", type=int, nargs="+", default=(0, 1, 2, 4, 8))
    parser.add_argument("--num_draft_tokens", type=int, nargs="+", default=(0, 2, 4, 8, 16, 32))
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output", type=str, default="output.jsonl")
    args = parser.parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)

    main(args, server_args)
