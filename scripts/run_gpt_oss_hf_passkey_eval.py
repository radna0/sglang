#!/usr/bin/env python3

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt_oss_hf_loader import prepare_hf_model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pure-HF synthetic long-context passkey retrieval eval."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--contexts", default="2048,4096,8192,16384,32768")
    parser.add_argument("--positions", default="start,middle,end")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-memory-per-gpu", default="78GiB")
    parser.add_argument("--max-cpu-memory", default="512GiB")
    parser.add_argument("--offload-folder", default="./offload")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _resolve_model_ref(model_path: str) -> str:
    candidate = Path(model_path)
    if candidate.exists():
        return str(candidate.resolve())
    return model_path


def _infer_input_device(model) -> torch.device:
    if hasattr(model, "device") and isinstance(model.device, torch.device) and model.device.type != "meta":
        return model.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def _build_prompt_ids(
    tokenizer,
    *,
    target_context_len: int,
    position: str,
    passkey: str,
) -> torch.Tensor:
    prefix_ids = tokenizer(
        "Read the full document carefully. A secret passkey appears exactly once. "
        "At the end, answer with only the passkey digits.\n\n",
        add_special_tokens=False,
    )["input_ids"]
    key_ids = tokenizer(
        f"\nImportant note: the secret passkey is {passkey}. Keep it exact.\n",
        add_special_tokens=False,
    )["input_ids"]
    suffix_ids = tokenizer(
        "\n\nQuestion: What is the secret passkey? Answer with only the digits.\nAnswer:",
        add_special_tokens=False,
    )["input_ids"]
    filler_ids = tokenizer(
        " Background details follow. This sentence is irrelevant to the secret key.",
        add_special_tokens=False,
    )["input_ids"]
    if not filler_ids:
        filler_ids = [tokenizer.eos_token_id or 0]

    static_len = len(prefix_ids) + len(key_ids) + len(suffix_ids)
    if target_context_len <= static_len + 16:
        raise ValueError(
            f"Target context {target_context_len} is too small for static prompt length {static_len}."
        )

    filler_budget = target_context_len - static_len
    repeated = (filler_ids * ((filler_budget // len(filler_ids)) + 2))[:filler_budget]
    if position == "start":
        split = min(len(repeated) // 8, len(repeated))
    elif position == "end":
        split = max(len(repeated) - len(repeated) // 8, 0)
    elif position == "middle":
        split = len(repeated) // 2
    else:
        raise ValueError(f"Unsupported passkey position: {position}")
    prompt_ids = prefix_ids + repeated[:split] + key_ids + repeated[split:] + suffix_ids
    return torch.tensor(prompt_ids, dtype=torch.long)


def _extract_passkey(text: str) -> str | None:
    match = re.search(r"(\d{4,12})", text)
    return match.group(1) if match else None


def main() -> None:
    args = _parse_args()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    model_ref, trust_remote_code = prepare_hf_model_path(_resolve_model_ref(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref, trust_remote_code=trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        dtype=_dtype_from_name(args.dtype),
        device_map="auto" if args.device == "cuda" else None,
        max_memory=(
            {
                **{i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())},
                "cpu": args.max_cpu_memory,
            }
            if args.device == "cuda" and torch.cuda.device_count() > 0
            else None
        ),
        offload_folder=args.offload_folder,
    )
    model.eval()
    input_device = _infer_input_device(model)

    rng = random.Random(args.seed)
    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    positions = [x.strip() for x in args.positions.split(",") if x.strip()]

    results = []
    summary: dict[str, dict[str, float | int]] = {}
    for context_len in contexts:
        for position in positions:
            correct = 0
            actual_lengths: list[int] = []
            generations = []
            for sample_idx in range(int(args.num_samples)):
                passkey = f"{rng.randrange(0, 10**6):06d}"
                input_ids = _build_prompt_ids(
                    tokenizer,
                    target_context_len=context_len,
                    position=position,
                    passkey=passkey,
                ).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                actual_lengths.append(int(input_ids.shape[1]))
                input_ids = input_ids.to(input_device)
                attention_mask = attention_mask.to(input_device)
                do_sample = bool(args.temperature > 0)
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "do_sample": do_sample,
                    "max_new_tokens": int(args.max_new_tokens),
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if do_sample:
                    generation_kwargs["temperature"] = float(args.temperature)

                with torch.no_grad():
                    generated = model.generate(**generation_kwargs)
                new_tokens = generated[:, input_ids.shape[1] :]
                text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
                predicted = _extract_passkey(text)
                matched = predicted == passkey
                correct += int(matched)
                generations.append(
                    {
                        "sample_idx": sample_idx,
                        "target_context_len": int(context_len),
                        "actual_input_tokens": int(input_ids.shape[1]),
                        "position": position,
                        "passkey": passkey,
                        "prediction": predicted,
                        "matched": matched,
                        "raw_text": text,
                    }
                )

            accuracy = correct / max(int(args.num_samples), 1)
            bucket_key = f"{context_len}:{position}"
            summary[bucket_key] = {
                "accuracy": accuracy,
                "correct": correct,
                "num_samples": int(args.num_samples),
                "mean_input_tokens": sum(actual_lengths) / max(len(actual_lengths), 1),
            }
            results.extend(generations)

    payload = {
        "model_path": model_ref,
        "contexts": contexts,
        "positions": positions,
        "summary": summary,
        "samples": results,
    }
    _save_json(out_json, payload)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
