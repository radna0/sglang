#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_gpt_oss_kv_covariance import _install_mxfp4_preswizzle_cache
from gpt_oss_hf_loader import prepare_hf_model_path

_HARMONY_MARKER_RE = re.compile(r"<\|start\|>|<\|message\|>|<\|channel\|>|<\|recipient\|>")

try:
    import openai_harmony
except ImportError:  # pragma: no cover - optional dependency
    openai_harmony = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GPT-OSS teacher vs MLA student next-token logits on a fixed JSONL pack."
    )
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--pack-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--positions-per-sample",
        type=int,
        default=64,
        help="Compare up to this many next-token positions from the tail of each sample.",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--harmony-normalize-input",
        default="auto",
        choices=["auto", "on", "off"],
    )
    parser.add_argument(
        "--mxfp4-preswizzle-dir",
        default=os.environ.get("GPTOSS_MXFP4_PRESWIZZLE_DIR", ""),
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _looks_like_harmony_transcript(text: str) -> bool:
    return bool(_HARMONY_MARKER_RE.search(text))


@lru_cache(maxsize=1)
def _get_harmony_encoding():
    if openai_harmony is None:
        return None
    return openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )


def _maybe_harmony_normalize(text: str, mode: str) -> str:
    if mode == "off":
        return text
    encoding = _get_harmony_encoding()
    if encoding is None:
        return text
    if mode == "auto" and not _looks_like_harmony_transcript(text):
        return text
    try:
        tokens = encoding.encode(text, allowed_special="all", disallowed_special=())
        messages = encoding.parse_messages_from_completion_tokens(tokens)
        conversation = openai_harmony.Conversation.from_messages(messages)
        render_config = openai_harmony.RenderConversationConfig(
            auto_drop_analysis=False
        )
        canonical_tokens = encoding.render_conversation_for_training(
            conversation, config=render_config
        )
        return encoding.decode(canonical_tokens)
    except Exception:
        return text


def _load_rows(pack_path: Path, max_samples: int) -> list[dict]:
    rows = []
    with pack_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No rows found in {pack_path}")
    return rows


def _extract_text(row: dict, harmony_normalize_input: str) -> str:
    text = str(row.get("text") or row.get("prompt") or row.get("input") or "")
    if not text:
        text = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return _maybe_harmony_normalize(text, harmony_normalize_input)


def _load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    preswizzle_dir: str,
):
    model_ref, trust_remote_code = prepare_hf_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref, trust_remote_code=trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    _install_mxfp4_preswizzle_cache(preswizzle_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map={"": str(device)} if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model_ref, model, tokenizer


def _sample_position_slice(seq_len: int, positions_per_sample: int) -> slice:
    usable = max(seq_len - 1, 0)
    if usable <= 0:
        return slice(0, 0)
    start = max(0, usable - positions_per_sample)
    return slice(start, usable)


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path = Path(args.pack_jsonl).resolve()
    rows = _load_rows(pack_path, args.max_samples)

    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dtype = _dtype_from_name(args.dtype)
    teacher_ref, teacher_model, teacher_tokenizer = _load_model_and_tokenizer(
        args.teacher_model_path,
        dtype=dtype,
        device=device,
        preswizzle_dir=args.mxfp4_preswizzle_dir,
    )
    student_ref, student_model, student_tokenizer = _load_model_and_tokenizer(
        args.student_model_path,
        dtype=dtype,
        device=device,
        preswizzle_dir=args.mxfp4_preswizzle_dir,
    )

    if teacher_tokenizer.get_vocab() != student_tokenizer.get_vocab():
        raise ValueError("Teacher and student tokenizers do not match.")

    summary = {
        "samples": 0,
        "compared_positions": 0,
        "mean_kl_teacher_to_student": 0.0,
        "mean_top1_agreement": 0.0,
        "mean_topk_overlap": 0.0,
        "mean_teacher_nll_under_student": 0.0,
    }
    sample_reports = []

    for row in tqdm(rows, desc="logit-projection"):
        text = _extract_text(row, args.harmony_normalize_input)
        encoded = teacher_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(args.max_length),
        )
        input_ids = encoded["input_ids"].to(device)
        if int(input_ids.shape[1]) < 2:
            continue

        with torch.inference_mode():
            teacher_logits = teacher_model(input_ids, use_cache=False).logits.float()[0]
            student_logits = student_model(input_ids, use_cache=False).logits.float()[0]

        pos_slice = _sample_position_slice(
            int(input_ids.shape[1]), int(args.positions_per_sample)
        )
        teacher_shift = teacher_logits[:-1][pos_slice]
        student_shift = student_logits[:-1][pos_slice]
        labels = input_ids[0, 1:][pos_slice]
        if teacher_shift.numel() == 0:
            continue

        teacher_log_probs = F.log_softmax(teacher_shift, dim=-1)
        student_log_probs = F.log_softmax(student_shift, dim=-1)
        teacher_probs = teacher_log_probs.exp()

        kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
            log_target=False,
        )
        teacher_top1 = teacher_shift.argmax(dim=-1)
        student_top1 = student_shift.argmax(dim=-1)
        top1_agreement = (teacher_top1 == student_top1).float().mean()

        k = int(args.topk)
        teacher_topk = teacher_shift.topk(k, dim=-1).indices
        student_topk = student_shift.topk(k, dim=-1).indices
        topk_overlap = (
            (teacher_topk.unsqueeze(-1) == student_topk.unsqueeze(-2))
            .any(dim=-1)
            .float()
            .mean()
        )
        teacher_nll_under_student = F.nll_loss(
            student_log_probs,
            labels,
            reduction="mean",
        )

        compared_positions = int(teacher_shift.shape[0])
        summary["samples"] += 1
        summary["compared_positions"] += compared_positions
        summary["mean_kl_teacher_to_student"] += float(kl.item())
        summary["mean_top1_agreement"] += float(top1_agreement.item())
        summary["mean_topk_overlap"] += float(topk_overlap.item())
        summary["mean_teacher_nll_under_student"] += float(
            teacher_nll_under_student.item()
        )
        sample_reports.append(
            {
                "id": row.get("id"),
                "compared_positions": compared_positions,
                "kl_teacher_to_student": float(kl.item()),
                "top1_agreement": float(top1_agreement.item()),
                "topk_overlap": float(topk_overlap.item()),
                "teacher_nll_under_student": float(teacher_nll_under_student.item()),
            }
        )

    if summary["samples"] > 0:
        denom = float(summary["samples"])
        for key in (
            "mean_kl_teacher_to_student",
            "mean_top1_agreement",
            "mean_topk_overlap",
            "mean_teacher_nll_under_student",
        ):
            summary[key] /= denom
    else:
        for key in (
            "mean_kl_teacher_to_student",
            "mean_top1_agreement",
            "mean_topk_overlap",
            "mean_teacher_nll_under_student",
        ):
            summary[key] = math.nan

    payload = {
        "teacher_model_path": teacher_ref,
        "student_model_path": student_ref,
        "pack_jsonl": str(pack_path),
        "max_samples": int(args.max_samples),
        "max_length": int(args.max_length),
        "positions_per_sample": int(args.positions_per_sample),
        "topk": int(args.topk),
        "summary": summary,
        "samples": sample_reports,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
