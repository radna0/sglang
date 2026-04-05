import re
from pathlib import Path


def _read_repo_file(rel_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / rel_path).read_text(encoding="utf-8")


def test_gpt_oss_prefill_populates_index_cache_without_sparse_attention():
    text = _read_repo_file("python/sglang/srt/models/gpt_oss.py")
    assert "if forward_batch.forward_mode.is_extend_without_speculative():" in text
    assert "return_indices=False" in text
    assert re.search(
        r"elif\s+forward_batch\.forward_mode\.is_decode_or_idle\(\):\s+topk_indices\s*=\s*self\.indexer",
        text,
        flags=re.DOTALL,
    )


def test_flashattention_sparse_path_is_guarded_to_decode_only():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "GPT-OSS GQA DSA topk_indices are decode-only" in text
    assert "forward_batch.forward_mode.is_decode_or_idle()" in text
