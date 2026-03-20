#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_gpt_oss_kv_covariance import _install_mxfp4_preswizzle_cache
from gpt_oss_hf_loader import prepare_hf_model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sharded pure-HF perplexity eval for GPT-OSS checkpoints."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True, choices=["wikitext", "c4"])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--progress-every-docs", type=int, default=256)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mxfp4-preswizzle-dir",
        default=os.environ.get("GPTOSS_MXFP4_PRESWIZZLE_DIR", ""),
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _load_task_dataset(task: str):
    if task == "wikitext":
        ds = load_dataset(
            "EleutherAI/wikitext_document_level",
            "wikitext-2-raw-v1",
            split="test",
        )
        return ds, "page"
    if task == "c4":
        ds = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            verification_mode="no_checks",
        )
        return ds, "text"
    raise ValueError(f"Unsupported task: {task}")


def _detok_wikitext(text: str) -> str:
    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    text = text.replace(" : ", ": ")
    text = text.replace(" ; ", "; ")
    text = text.replace(" . ", ". ")
    text = text.replace(" ! ", "! ")
    text = text.replace(" ? ", "? ")
    text = text.replace(" , ", ", ")
    text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
    text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
    text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
    text = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
    text = text.replace("= = = =", "====")
    text = text.replace("= = =", "===")
    text = text.replace("= =", "==")
    text = text.replace(" " + chr(176) + " ", chr(176))
    text = text.replace(" \n", "\n")
    text = text.replace("\n ", "\n")
    text = text.replace(" N ", " 1 ")
    text = text.replace(" 's", "'s")
    return text


def _detok_c4(text: str) -> str:
    return _detok_wikitext(text)


def _maybe_init_dist() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return world_size, rank, local_rank, enabled


def _sync_stats(
    device: torch.device,
    enabled: bool,
    processed_docs: int,
    total_words: int,
    total_bytes: int,
    total_tokens: int,
    total_nll: float,
) -> tuple[int, int, int, int, float]:
    if not enabled:
        return processed_docs, total_words, total_bytes, total_tokens, total_nll
    payload = torch.tensor(
        [processed_docs, total_words, total_bytes, total_tokens, total_nll],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    return (
        int(payload[0].item()),
        int(payload[1].item()),
        int(payload[2].item()),
        int(payload[3].item()),
        float(payload[4].item()),
    )


def _compute_doc_nll(
    model,
    input_ids: torch.Tensor,
    max_length: int,
    stride: int,
) -> tuple[float, int]:
    seq_len = int(input_ids.shape[0])
    if seq_len < 2:
        return 0.0, 0

    total_nll = 0.0
    total_tokens = 0
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        if trg_len <= 0:
            continue
        window = input_ids[begin_loc:end_loc].unsqueeze(0)
        with torch.inference_mode():
            outputs = model(window, use_cache=False)
        shift_logits = outputs.logits[:, :-1, :].float().contiguous()
        shift_labels = window[:, 1:].contiguous()
        valid_positions = shift_labels.shape[1] - trg_len
        if valid_positions > 0:
            shift_labels[:, :valid_positions] = -100
        nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        if not torch.isfinite(nll):
            continue
        total_nll += float(nll.item())
        total_tokens += int((shift_labels != -100).sum().item())
        if end_loc >= seq_len:
            break
    return total_nll, total_tokens


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ref, trust_remote_code = prepare_hf_model_path(args.model_path)

    world_size, rank, local_rank, dist_enabled = _maybe_init_dist()
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    stride = int(args.stride or args.max_length)
    dataset, text_key = _load_task_dataset(args.task)
    total_docs = len(dataset)
    if args.limit_docs is not None:
        total_docs = min(total_docs, int(args.limit_docs))

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    _install_mxfp4_preswizzle_cache(args.mxfp4_preswizzle_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        dtype=_dtype_from_name(args.dtype),
        device_map={"": str(device)} if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if args.task == "wikitext":
        detok = _detok_wikitext
    else:
        detok = _detok_c4

    local_total = math.ceil(max(total_docs - rank, 0) / max(world_size, 1))
    pbar = None
    last_synced_docs = 0
    if rank == 0:
        pbar = tqdm(total=total_docs, desc=f"{args.task} docs", dynamic_ncols=True)

    local_processed_docs = 0
    local_words = 0
    local_bytes = 0
    local_tokens = 0
    local_nll = 0.0
    started = time.time()

    for doc_idx in range(rank, total_docs, world_size):
        row = dataset[doc_idx]
        original_text = row[text_key]
        detok_text = detok(original_text)
        encoded = tokenizer(detok_text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0].to(device)

        doc_nll, doc_tokens = _compute_doc_nll(model, encoded, args.max_length, stride)
        if doc_tokens > 0:
            local_words += len(re.split(r"\s+", original_text))
            local_bytes += len(original_text.encode("utf-8"))
            local_tokens += doc_tokens
            local_nll += doc_nll
        local_processed_docs += 1

        if (
            local_processed_docs % max(args.progress_every_docs, 1) == 0
            or local_processed_docs == local_total
        ):
            (
                global_docs,
                global_words,
                global_bytes,
                global_tokens,
                global_nll,
            ) = _sync_stats(
                device,
                dist_enabled,
                local_processed_docs,
                local_words,
                local_bytes,
                local_tokens,
                local_nll,
            )
            if rank == 0 and pbar is not None:
                pbar.update(max(global_docs - last_synced_docs, 0))
                last_synced_docs = global_docs
                word_ppl = math.exp(global_nll / global_words) if global_words > 0 else float("nan")
                print(
                    json.dumps(
                        {
                            "task": args.task,
                            "progress_docs": global_docs,
                            "total_docs": total_docs,
                            "progress_pct": round(100.0 * global_docs / total_docs, 3)
                            if total_docs
                            else 0.0,
                            "elapsed_s": round(time.time() - started, 2),
                            "world_size": world_size,
                            "word_ppl": word_ppl,
                            "tokens": global_tokens,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    (
        global_docs,
        global_words,
        global_bytes,
        global_tokens,
        global_nll,
    ) = _sync_stats(
        device,
        dist_enabled,
        local_processed_docs,
        local_words,
        local_bytes,
        local_tokens,
        local_nll,
    )

    if rank == 0 and pbar is not None:
        pbar.update(max(global_docs - last_synced_docs, 0))
        pbar.close()

    if rank == 0:
        word_ppl = math.exp(global_nll / global_words) if global_words > 0 else float("nan")
        byte_ppl = math.exp(global_nll / global_bytes) if global_bytes > 0 else float("nan")
        bits_per_byte = (
            global_nll / (global_bytes * math.log(2.0)) if global_bytes > 0 else float("nan")
        )
        payload = {
            "summary": {
                args.task: {
                    "word_perplexity,none": word_ppl,
                    "word_perplexity_stderr,none": "N/A",
                    "byte_perplexity,none": byte_ppl,
                    "byte_perplexity_stderr,none": "N/A",
                    "bits_per_byte,none": bits_per_byte,
                    "bits_per_byte_stderr,none": "N/A",
                }
            },
            "raw": {
                "task": args.task,
                "world_size": world_size,
                "docs_processed": global_docs,
                "words": global_words,
                "bytes": global_bytes,
                "tokens": global_tokens,
                "total_nll": global_nll,
                "max_length": args.max_length,
                "stride": stride,
            },
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)

    if dist_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
