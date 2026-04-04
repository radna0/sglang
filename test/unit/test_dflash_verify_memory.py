import ast
from pathlib import Path

import torch


def _load_top_p_helper():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
        / "dflash_info.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    wanted = {"_top_p_is_effectively_disabled"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace, namespace)
    return namespace["_top_p_is_effectively_disabled"]


_top_p_is_effectively_disabled = _load_top_p_helper()


def test_top_p_is_effectively_disabled_for_unfiltered_rows():
    assert _top_p_is_effectively_disabled(torch.tensor([1.0], dtype=torch.float32))
    assert _top_p_is_effectively_disabled(
        torch.tensor([1.0, 1.0, 1.000001], dtype=torch.float32)
    )


def test_top_p_is_effectively_disabled_detects_real_filtering():
    assert not _top_p_is_effectively_disabled(
        torch.tensor([1.0, 0.95], dtype=torch.float32)
    )
