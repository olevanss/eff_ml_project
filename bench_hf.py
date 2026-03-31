import gc
import json
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig


PROMPTS = [
    "Human: Give me a fully functional FastAPI server. Show the full, long python code without stop.\n\nAssistant:",
    "Human: Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. Develop a Solidity smart contract for this purpose, including the necessary functions. Please provide the code and explanations.\n\nAssistant:",
    "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
    "Human: I want you to act as a storyteller. Come up with entertaining stories that are engaging, imaginative and captivating for the audience. Answer in more than 5000 words. My first request is 'I need an interesting story on perseverance.'\n\nAssistant:",
    "Human: Solve x^2 = -1. Think step-by-step. Give me a long detailed explanation.\n\nAssistant:",
    "Human: Tell me about the president of the USA in wikipedia style.\n\nAssistant:",
    "Human: Hello? Who are you? Write code, math, and poem to explain yourself.\n\nAssistant:",
]


class GenerationTracker:
    def __init__(self, model):
        self.model = model
        self.forward_count = 0
        self._orig = model.forward

    def __enter__(self):
        self.forward_count = 0
        tracker = self

        def _fwd(*args, **kwargs):
            tracker.forward_count += 1
            return tracker._orig(*args, **kwargs)

        self.model.forward = _fwd
        return self

    def __exit__(self, *exc):
        self.model.forward = self._orig
        return False


def _fix_gen_config(model):
    cfg = model.generation_config
    for attr in ("max_length", "min_length", "min_new_tokens"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, None)


def load_model(model_path, quantization=None):
    kw = {"device_map": "auto", "torch_dtype": "auto"}
    if quantization == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    model.eval()
    return model


def load_tokenizer(model_path):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1),
    }


def get_model_memory_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2


def warmup(model, tokenizer, draft_model=None, num_assistant_tokens=5):
    inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
    gen_kwargs = {"generation_config": GenerationConfig(max_new_tokens=16, max_length=None, min_length=None)}
    if draft_model is not None:
        _fix_gen_config(draft_model)
        gen_kwargs["assistant_model"] = draft_model
        gen_kwargs["num_assistant_tokens"] = num_assistant_tokens
        gen_kwargs["num_assistant_tokens_schedule"] = "constant"
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()


def run_benchmark(target_model, tokenizer, prompts, draft_model=None, num_assistant_tokens=5, max_new_tokens=256):
    gen_kwargs = {
        "generation_config": GenerationConfig(max_new_tokens=max_new_tokens, max_length=None, min_length=None)
    }
    if draft_model is not None:
        _fix_gen_config(draft_model)
        gen_kwargs["assistant_model"] = draft_model
        gen_kwargs["num_assistant_tokens"] = num_assistant_tokens
        gen_kwargs["num_assistant_tokens_schedule"] = "constant"

    total_tokens = 0
    total_time = 0.0
    total_fwd = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(target_model.device)
        input_len = inputs.input_ids.shape[1]
        torch.cuda.synchronize()

        with GenerationTracker(target_model) as tracker:
            t0 = time.perf_counter()
            with torch.no_grad():
                out = target_model.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        total_tokens += out.shape[1] - input_len
        total_time += t1 - t0
        total_fwd += tracker.forward_count

    decode_steps = total_fwd - len(prompts)
    acc = total_tokens / decode_steps if decode_steps > 0 else 1.0
    step_ms = (total_time / decode_steps) * 1000 if decode_steps > 0 else None
    speed = total_tokens / total_time

    return {
        "acc_length": round(acc, 3),
        "step_time_ms": round(step_ms, 3) if step_ms else None,
        "speed_tok_s": round(speed, 2),
        "total_output_tokens": int(total_tokens),
        "total_time_s": round(total_time, 2),
        "target_forward_passes": int(total_fwd),
        "num_prompts": len(prompts),
        "avg_output_tokens": round(total_tokens / len(prompts), 1),
    }


def run_benchmark_suite(
    target_model, tokenizer, draft_model=None, draft_label="",
    num_assistant_tokens_list=None, max_new_tokens=256, num_prompts=8,
    output_file=None, do_warmup=True,
):
    if num_assistant_tokens_list is None:
        num_assistant_tokens_list = [0, 3, 5, 8]

    prompts = (PROMPTS * 10)[:num_prompts]
    results = []

    for nat in num_assistant_tokens_list:
        is_baseline = nat == 0 or draft_model is None
        tag = "baseline" if is_baseline else f"nat={nat}"
        print(f"\n--- {tag} ---")

        cur_draft = None if is_baseline else draft_model
        cur_nat = 0 if is_baseline else nat

        if do_warmup:
            warmup(target_model, tokenizer, cur_draft, cur_nat)

        metrics = run_benchmark(target_model, tokenizer, prompts, cur_draft, cur_nat, max_new_tokens)
        record = {"draft_label": draft_label, "num_assistant_tokens": nat, "is_baseline": is_baseline, **metrics}
        results.append(record)

        st_s = f", step={record['step_time_ms']:.1f}ms" if record["step_time_ms"] else ""
        print(f"{record['speed_tok_s']:.1f} tok/s, acc={record['acc_length']:.3f}{st_s}")

        if output_file:
            with open(output_file, "a") as f:
                f.write(json.dumps(record) + "\n")

    return results


def run_multi_draft_benchmark(
    target_model, tokenizer, draft_configs,
    num_assistant_tokens_list=None, max_new_tokens=256,
    num_prompts=8, output_file="bench_results.jsonl",
):
    if num_assistant_tokens_list is None:
        num_assistant_tokens_list = [0, 3, 5, 8]

    all_results = []

    for dc in draft_configs:
        model_path = dc.get("model_path")
        label = dc.get("label", model_path or "baseline")
        print(f"\n=== {label} ===")

        draft_model = None
        if model_path:
            draft_model = load_model(model_path, quantization=dc.get("quantization"))
            print(f"{get_model_memory_mb(draft_model):.0f} MB, GPU: {get_gpu_memory_info()}")

        try:
            nat_list = num_assistant_tokens_list if model_path else [0]
            results = run_benchmark_suite(
                target_model, tokenizer,
                draft_model=draft_model, draft_label=label,
                num_assistant_tokens_list=nat_list,
                max_new_tokens=max_new_tokens,
                num_prompts=num_prompts,
                output_file=output_file,
            )
            all_results.extend(results)
        finally:
            if draft_model is not None:
                del draft_model
                gc.collect()
                torch.cuda.empty_cache()

    return all_results
