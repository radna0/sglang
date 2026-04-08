import os
import re
from pathlib import Path

import pytest


if os.name == "nt":
    pytest.skip("SGLang SRT tests require Linux (resource module).", allow_module_level=True)


def _read_repo_file(rel_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / rel_path).read_text(encoding="utf-8")


def test_fa3_backend_forward_decode_accepts_topk_indices():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    # Keep this intentionally lightweight and import-free (flash-attn may be absent in CPU CI).
    m = re.search(r"def\\s+forward_decode\\s*\\(.*?\\):", text, flags=re.DOTALL)
    assert m is not None
    assert "topk_indices" in m.group(0)


def test_fa3_backend_exposes_get_indexer_metadata():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "def get_indexer_metadata(" in text

