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

_HARMONY_CONTROL_RE = re.compile(r"<\|[^>]+?\|>")
_HARMONY_MARKER_RE = re.compile(r"<\|start\|>|<\|message\|>|<\|channel\|>|<\|recipient\|>")

try:
    import openai_harmony
except ImportError:  # pragma: no cover - optional dependency
    openai_harmony = None

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPL on a fixed local JSONL eval pack for GPT-OSS checkpoints."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pack-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--contexts", default="2048,8192")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--harmony-visible-text",
        default="auto",
        choices=["auto", "on", "off"],
        help="Use openai_harmony parsing for visible-text word/byte metrics.",
    )
    parser.add_argument(
        "--harmony-normalize-input",
        default="auto",
        choices=["auto", "on", "off"],
        help="Round-trip Harmony transcripts through openai_harmony before tokenization.",
    )
    parser.add_argument(
        "--mxfp4-preswizzle-dir",
        default=os.environ.get("GPTOSS_MXFP4_PRESWIZZLE_DIR", ""),
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _visible_text(text: str) -> str:
    if "<|" not in text:
        return text
    text = _HARMONY_CONTROL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _content_text(content) -> str:
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    if hasattr(content, "model_dump"):
        dumped = content.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return json.dumps(dumped, ensure_ascii=False, sort_keys=True)
    return str(content)


def _looks_like_harmony_transcript(text: str) -> bool:
    return bool(_HARMONY_MARKER_RE.search(text))


@lru_cache(maxsize=1)
def _get_harmony_encoding():
    if openai_harmony is None:
        return None
    return openai_harmony.load_harmony_encoding(
        openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
    )


def _maybe_harmony_parse(text: str):
    encoding = _get_harmony_encoding()
    if encoding is None or not _looks_like_harmony_transcript(text):
        return None, None
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
        canonical_text = encoding.decode(canonical_tokens)
        return messages, canonical_text
    except Exception:
        return None, None


def _resolve_text_views(
    text: str,
    *,
    harmony_visible_text_mode: str,
    harmony_normalize_input_mode: str,
) -> tuple[str, str]:
    messages, canonical_text = _maybe_harmony_parse(text)
    use_harmony_visible = harmony_visible_text_mode == "on" or (
        harmony_visible_text_mode == "auto" and messages is not None
    )
    use_harmony_normalize = harmony_normalize_input_mode == "on" or (
        harmony_normalize_input_mode == "auto" and canonical_text is not None
    )

    model_text = canonical_text if use_harmony_normalize and canonical_text is not None else text
    if use_harmony_visible and messages is not None:
        chunks = []
        for message in messages:
            for content in message.content:
                value = _content_text(content).strip()
                if value:
                    chunks.append(value)
        visible_text = "\n\n".join(chunks).strip()
        return model_text, visible_text or _visible_text(text)
    return model_text, _visible_text(text)


def _compute_doc_nll(
    model,
    input_ids: torch.Tensor,
    *,
    max_length: int,
) -> tuple[float, int]:
    seq_len = int(input_ids.shape[0])
    if seq_len < 2:
        return 0.0, 0

    stride = max_length
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


def _summarize(total_nll: float, total_tokens: int, total_words: int, total_bytes: int) -> dict:
    if total_tokens <= 0 or total_words <= 0 or total_bytes <= 0:
        return {
            "bits_per_byte": math.nan,
            "byte_perplexity": math.nan,
            "log_byte_perplexity": math.nan,
            "token_perplexity": math.nan,
            "log_token_perplexity": math.nan,
            "word_perplexity": math.nan,
            "log_word_perplexity": math.nan,
        }
    bits_per_byte = total_nll / math.log(2) / total_bytes
    log_byte_perplexity = bits_per_byte * math.log(2)
    log_token_perplexity = total_nll / total_tokens
    log_word_perplexity = total_nll / total_words
    try:
        byte_perplexity = math.exp(log_byte_perplexity)
    except OverflowError:
        byte_perplexity = math.inf
    try:
        token_perplexity = math.exp(log_token_perplexity)
    except OverflowError:
        token_perplexity = math.inf
    try:
        word_perplexity = math.exp(log_word_perplexity)
    except OverflowError:
        word_perplexity = math.inf
    return {
        "bits_per_byte": bits_per_byte,
        "byte_perplexity": byte_perplexity,
        "log_byte_perplexity": log_byte_perplexity,
        "token_perplexity": token_perplexity,
        "log_token_perplexity": log_token_perplexity,
        "word_perplexity": word_perplexity,
        "log_word_perplexity": log_word_perplexity,
    }


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path = Path(args.pack_jsonl).resolve()
    model_ref, trust_remote_code = prepare_hf_model_path(args.model_path)

    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    rows = []
    with pack_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {pack_path}")

    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

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

    payload = {
        "model_path": model_ref,
        "pack_jsonl": str(pack_path),
        "contexts": contexts,
        "rows": len(rows),
        "results": {},
    }

    for max_length in contexts:
        per_source: dict[str, dict[str, float | int]] = {}
        per_doc = []
        source_stats: dict[str, dict[str, float]] = {}
        pbar = tqdm(rows, desc=f"fixed-pack ppl@{max_length}", dynamic_ncols=True)
        for row in pbar:
            text = str(row["text"])
            model_text, visible_text = _resolve_text_views(
                text,
                harmony_visible_text_mode=args.harmony_visible_text,
                harmony_normalize_input_mode=args.harmony_normalize_input,
            )
            source = str(row["source"])
            sample_id = str(row["sample_id"])
            encoded = tokenizer(
                model_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0].to(device)
            try:
                doc_nll, doc_tokens = _compute_doc_nll(model, encoded, max_length=max_length)
            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                doc_nll, doc_tokens = math.nan, 0
            doc_words = max(len(re.findall(r"\\S+", text)), 1)
            doc_bytes = max(len(text.encode("utf-8")), 1)
            record = {
                "source": source,
                "sample_id": sample_id,
                "char_len": int(row.get("char_len", len(text))),
                "model_char_len": len(model_text),
                "visible_char_len": len(visible_text),
                "token_count": int(doc_tokens),
                "nll": float(doc_nll) if math.isfinite(doc_nll) else None,
            }
            per_doc.append(record)
            stats = source_stats.setdefault(
                source,
                {"nll": 0.0, "tokens": 0, "words": 0, "bytes": 0, "docs": 0},
            )
            if math.isfinite(doc_nll) and doc_tokens > 0:
                doc_words = max(len(re.findall(r"\S+", visible_text)), 1)
                doc_bytes = max(len(visible_text.encode("utf-8")), 1)
                stats["nll"] += float(doc_nll)
                stats["tokens"] += int(doc_tokens)
                stats["words"] += int(doc_words)
                stats["bytes"] += int(doc_bytes)
                stats["docs"] += 1

        for source, stats in sorted(source_stats.items()):
            metrics = _summarize(
                float(stats["nll"]),
                int(stats["tokens"]),
                int(stats["words"]),
                int(stats["bytes"]),
            )
            metrics["docs"] = int(stats["docs"])
            metrics["tokens"] = int(stats["tokens"])
            per_source[source] = metrics

        all_nll = sum(float(stats["nll"]) for stats in source_stats.values())
        all_tokens = sum(int(stats["tokens"]) for stats in source_stats.values())
        all_words = sum(int(stats["words"]) for stats in source_stats.values())
        all_bytes = sum(int(stats["bytes"]) for stats in source_stats.values())
        payload["results"][str(max_length)] = {
            "summary": _summarize(all_nll, all_tokens, all_words, all_bytes),
            "per_source": per_source,
            "per_doc": per_doc,
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(payload["results"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
