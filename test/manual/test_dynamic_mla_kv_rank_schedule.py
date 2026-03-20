from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"
import sys  # noqa: E402

sys.path.insert(0, str(_PYTHON_ROOT))

torch = pytest.importorskip("torch")


def test_model_config_parses_kv_lora_rank_per_layer_schedule():
    try:
        from sglang.srt.configs.model_config import AttentionArch, ModelConfig
    except ModuleNotFoundError as e:
        if e.name in ("triton", "resource"):
            pytest.skip("Requires a Linux + CUDA SGLang environment.")
        raise

    mc = object.__new__(ModelConfig)
    mc.attention_arch = AttentionArch.MLA
    mc.kv_lora_rank = 512
    mc.kv_lora_rank_per_layer = None
    mc.num_hidden_layers = 3
    mc.hf_text_config = SimpleNamespace(kv_lora_rank_per_layer=[256, 512, 256])
    mc.hf_config = SimpleNamespace()

    mc._maybe_init_dynamic_mla_kv_ranks()

    assert mc.kv_lora_rank_per_layer == [256, 512, 256]
    assert mc.kv_lora_rank == 512
    assert mc.has_dynamic_mla_kv_lora_rank() is True
    assert mc.get_unique_mla_kv_lora_ranks() == [256, 512]
    assert mc.get_mla_kv_lora_rank_slice(0, 2) == [256, 512]


def test_mla_token_to_kv_pool_supports_per_layer_rank_cpu():
    try:
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    except ModuleNotFoundError as e:
        if e.name in ("triton", "resource"):
            pytest.skip("Requires a Linux + CUDA SGLang environment.")
        raise

    pool = MLATokenToKVPool(
        size=8,
        page_size=2,
        dtype=torch.float16,
        kv_lora_rank=[128, 256, 128],
        qk_rope_head_dim=64,
        layer_num=3,
        device="cpu",
        enable_memory_saver=False,
    )

    assert pool.kv_cache_dim_per_layer == [192, 320, 192]
    assert pool.kv_buffer[0].shape[-1] == 192
    assert pool.kv_buffer[1].shape[-1] == 320
    assert pool.kv_buffer[2].shape[-1] == 192

    assert pool.get_value_buffer(0).shape[-1] == 128
    assert pool.get_value_buffer(1).shape[-1] == 256
    assert pool.get_value_buffer(2).shape[-1] == 128
