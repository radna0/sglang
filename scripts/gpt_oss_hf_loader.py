#!/usr/bin/env python3

import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


_REMOTE_CODE_FILES = ("modeling_gpt_oss_mla.py", "__init__.py")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _atomic_save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def is_gpt_oss_mla_checkpoint(model_path: str | Path) -> bool:
    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    config = _load_json(config_path)
    architectures = config.get("architectures") or []
    if "GptOssMlaForCausalLM" in architectures:
        return True
    return bool(
        "kv_lora_rank" in config
        and "qk_nope_head_dim" in config
        and "qk_rope_head_dim" in config
    )


def _assert_clean_gpt_oss_mla_checkpoint(path: Path) -> None:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return
    index = _load_json(index_path)
    weight_map = index.get("weight_map", {})
    stale = [
        name
        for name in weight_map
        if ".self_attn.k_proj." in name or ".self_attn.v_proj." in name
    ]
    if not stale:
        return
    sample = ", ".join(stale[:4])
    raise ValueError(
        "Mixed GPT-OSS MLA checkpoint detected: dense k_proj/v_proj tensors are still "
        f"present in the safetensors index ({sample}). Repair the checkpoint first or "
        "use a clean converted artifact such as converted_checkpoint_clean."
    )


def ensure_gpt_oss_mla_remote_code(model_path: str | Path) -> bool:
    path = Path(model_path).resolve()
    if not is_gpt_oss_mla_checkpoint(path):
        return False
    _assert_clean_gpt_oss_mla_checkpoint(path)

    source_dir = Path(__file__).resolve().parent / "hf_gpt_oss_mla"
    config_path = path / "config.json"
    lock_path = path / ".gpt_oss_mla_loader.lock"

    with _file_lock(lock_path):
        config = _load_json(config_path)
        expected = "modeling_gpt_oss_mla.GptOssMlaForCausalLM"
        auto_map = dict(config.get("auto_map") or {})
        has_files = all((path / filename).exists() for filename in _REMOTE_CODE_FILES)
        if auto_map.get("AutoModelForCausalLM") == expected and has_files:
            return True

        for filename in _REMOTE_CODE_FILES:
            shutil.copy2(source_dir / filename, path / filename)

        auto_map["AutoModelForCausalLM"] = expected
        config["auto_map"] = auto_map
        config["architectures"] = ["GptOssMlaForCausalLM"]
        if "v_head_dim" not in config:
            config["v_head_dim"] = int(config.get("head_dim", 64))
        _atomic_save_json(config_path, config)
    return True


def prepare_hf_model_path(model_path: str | Path) -> tuple[str, bool]:
    path = Path(model_path)
    if path.exists():
        resolved = str(path.resolve())
        trust_remote_code = ensure_gpt_oss_mla_remote_code(resolved)
        return resolved, trust_remote_code
    return str(model_path), False
