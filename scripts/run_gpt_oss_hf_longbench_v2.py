#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt_oss_hf_loader import prepare_hf_model_path


DEFAULT_DATASET = "THUDM/LongBench-v2:train"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pure-HF LongBench-v2 eval for GPT-OSS checkpoints."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--categories", default=None)
    parser.add_argument("--min-context-length", type=int, default=None)
    parser.add_argument("--max-context-length", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-memory-per-gpu", default="78GiB")
    parser.add_argument("--max-cpu-memory", default="512GiB")
    parser.add_argument("--offload-folder", default="./offload")
    parser.add_argument("--max-new-tokens", type=int, default=64)
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


def _load_local_data(path: str) -> list[dict[str, Any]]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".json", ".jsonl"}:
        with open(path, "r", encoding="utf-8") as fh:
            if suffix == ".jsonl":
                data = [json.loads(line) for line in fh if line.strip()]
            else:
                data = json.load(fh)
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8") as fh:
            data = list(csv.DictReader(fh))
    else:
        raise ValueError(f"Unsupported LongBench-v2 local file: {path}")
    if isinstance(data, dict):
        data = data.get("data", [])
    return [dict(row) for row in data]


def _load_dataset(identifier: str) -> list[dict[str, Any]]:
    if os.path.exists(identifier):
        return _load_local_data(identifier)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "LongBench-v2 from Hugging Face requires `datasets`. Install it or provide a local dataset file."
        ) from exc

    parts = identifier.split(":", maxsplit=1)
    dataset_name = parts[0]
    split = parts[1] if len(parts) == 2 else "train"
    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def _normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(example)
    for letter in ["A", "B", "C", "D"]:
        choice_key = f"choice_{letter}"
        if letter not in normalized and choice_key in normalized:
            normalized[letter] = normalized[choice_key]
    if "category" not in normalized and "domain" in normalized:
        normalized["category"] = normalized["domain"]

    answer = normalized.get("answer")
    if isinstance(answer, str):
        normalized["answer"] = answer.strip().upper()
    elif isinstance(answer, int) and 0 <= answer < 4:
        normalized["answer"] = ["A", "B", "C", "D"][answer]
    return normalized


def _format_question(row: dict[str, Any]) -> str:
    context = row.get("context", "")
    question = row.get("question", "")
    if "choices" in row:
        choices = row["choices"]
        choice_a = choices[0] if len(choices) > 0 else ""
        choice_b = choices[1] if len(choices) > 1 else ""
        choice_c = choices[2] if len(choices) > 2 else ""
        choice_d = choices[3] if len(choices) > 3 else ""
    else:
        choice_a = row.get("A", row.get("choice_A", ""))
        choice_b = row.get("B", row.get("choice_B", ""))
        choice_c = row.get("C", row.get("choice_C", ""))
        choice_d = row.get("D", row.get("choice_D", ""))

    return f"""
Please read the following text and answer the question below.
<text>
{context.strip()}
</text>

What is the correct answer to this question: {question.strip()}
Choices:
(A) {str(choice_a).strip()}
(B) {str(choice_b).strip()}
(C) {str(choice_c).strip()}
(D) {str(choice_d).strip()}

Format your response as follows: "The correct answer is (insert answer here)".""".strip()


def _extract_answer(response: str) -> str | None:
    response = response.replace("*", "")
    patterns = [
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
        r"answer\s+is\s*\(?([A-D])\)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r"\(([A-D])\)", response)
    if match:
        return match.group(1).upper()
    return None


def main() -> None:
    args = _parse_args()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    model_ref, trust_remote_code = prepare_hf_model_path(_resolve_model_ref(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=trust_remote_code, use_fast=True)
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

    raw_examples = _load_dataset(args.dataset_path)
    examples = [_normalize_example(ex) for ex in raw_examples]
    if args.categories:
        allowed = {x.strip() for x in args.categories.split(",") if x.strip()}
        examples = [ex for ex in examples if ex.get("category") in allowed]
    if args.num_examples is not None:
        examples = examples[: int(args.num_examples)]

    results = []
    category_stats: dict[str, dict[str, int]] = {}
    total_correct = 0
    total_seen = 0

    for idx, row in enumerate(examples):
        prompt = _format_question(row)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        token_len = int(input_ids.shape[1])
        if args.min_context_length is not None and token_len < int(args.min_context_length):
            continue
        if args.max_context_length is not None and token_len > int(args.max_context_length):
            continue

        input_ids = input_ids.to(input_device)
        attention_mask = torch.ones_like(input_ids)
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
        response_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        predicted = _extract_answer(response_text)

        answer = row.get("answer", "")
        if isinstance(answer, str):
            answer = answer.strip().upper()
        matched = predicted == answer
        total_correct += int(matched)
        total_seen += 1

        category = str(row.get("category", "unknown"))
        bucket = category_stats.setdefault(category, {"correct": 0, "total": 0})
        bucket["correct"] += int(matched)
        bucket["total"] += 1

        results.append(
            {
                "idx": idx,
                "category": category,
                "difficulty": row.get("difficulty"),
                "context_tokens": token_len,
                "answer": answer,
                "prediction": predicted,
                "matched": matched,
                "response_text": response_text,
            }
        )

    summary = {
        "overall_accuracy": total_correct / max(total_seen, 1),
        "total_examples": total_seen,
        "correct": total_correct,
        "by_category": {
            key: {
                "accuracy": value["correct"] / max(value["total"], 1),
                "correct": value["correct"],
                "total": value["total"],
            }
            for key, value in sorted(category_stats.items())
        },
    }
    payload = {
        "model_path": model_ref,
        "dataset_path": args.dataset_path,
        "summary": summary,
        "results": results,
    }
    _save_json(out_json, payload)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
