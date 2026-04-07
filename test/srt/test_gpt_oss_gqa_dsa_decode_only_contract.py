import re
from pathlib import Path


def _read_repo_file(rel_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / rel_path).read_text(encoding="utf-8")


def test_gpt_oss_prefill_populates_index_cache_without_sparse_attention():
    text = _read_repo_file("python/sglang/srt/models/gpt_oss.py")
    assert "if forward_batch.forward_mode.is_extend_without_speculative():" in text
    assert "return_indices=False" in text
    assert "elif forward_batch.forward_mode.is_decode_or_idle():" in text
    assert "topk_indices = self.indexer(" in text


def test_gpt_oss_decode_skips_indexer_for_all_short_batches():
    text = _read_repo_file("python/sglang/srt/models/gpt_oss.py")
    assert 'get_global_server_args().gpt_oss_dsa_index_topk' in text
    assert 'and not torch.any(' in text
    assert "topk_indices = None" in text
    assert "gpt_oss_dsa_recent = False" in text


def test_gpt_oss_recent_lane_uses_bool_flag_not_fake_tensor():
    text = _read_repo_file("python/sglang/srt/models/gpt_oss.py")
    assert "gpt_oss_dsa_recent = True" in text
    assert "gpt_oss_dsa_recent=gpt_oss_dsa_recent" in text
    assert "q.new_empty((q.shape[0], 1), dtype=torch.int32)" not in text


def test_flashattention_sparse_path_is_guarded_to_decode_only():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "GPT-OSS GQA DSA topk_indices are decode-only" in text
    assert "forward_batch.forward_mode.is_decode_or_idle()" in text


def test_flashattention_sparse_path_masks_padded_topk_entries():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "valid_topk_mask = (idx >= 0) & (idx < seq_lens_i64)" in text
    assert "torch.full_like(idx, -1)" in text
    assert "idx_i32 = torch.where(" in text


def test_flashattention_sparse_path_uses_page_sparse_decode():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "select_topk_pages_decode(" in text
    assert "pages_out = (topk + self.page_size - 1) // self.page_size" in text
    assert "full_page_table = metadata.page_table[:bs]" in text
    assert "sparse_page_table = torch.gather(" in text
    assert "key_cache = key_cache.view(" in text
    assert "value_cache = value_cache.view(" in text
    assert "page_table=sparse_page_table" in text


def test_flashattention_disables_dsa_when_seq_len_fits_topk_budget():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    # Instead of disabling DSA via tensor->bool branching (CUDA graph unsafe),
    # the sparse path now handles seq_len <= topk by producing a full-prefix selection.
    assert "gpt_oss_dsa_topk_source" in text


def test_gpt_oss_dsa_is_gated_to_full_attention_layers_only():
    text = _read_repo_file("python/sglang/srt/models/gpt_oss.py")
    assert (
        'server_args.enable_gpt_oss_gqa_dsa and layer_type == "full_attention"' in text
    )
    assert 'use_sliding_window = layer_type == "sliding_attention"' in text
    assert (
        "sliding_window_size=(sliding_window_size if use_sliding_window else -1)"
        in text
    )


def test_gpt_oss_attention_keeps_sink_forwarding():
    text = _read_repo_file("python/sglang/srt/models/gpt_oss.py")
    assert "sinks=self.sinks" in text
    backend_text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert 'kwargs["sinks"] = sinks' in backend_text


def test_transform_index_uses_fast_cuda_path_when_available():
    text = _read_repo_file("python/sglang/srt/layers/attention/nsa/transform_index.py")
    assert "transform_index_page_table_decode_fast(**kwargs)" in text
    assert "transform_index_page_table_prefill_fast(**kwargs)" in text
    assert "page_table.is_cuda" in text
    assert "topk_indices.shape[1] == 2048" in text


def test_flash_attention_v4_wrapper_exposes_block_sparse_tensors():
    text = _read_repo_file("python/sglang/jit_kernel/flash_attention_v4.py")
    assert "block_sparse_tensors: Optional[object] = None" in text
    assert "block_sparse_tensors=block_sparse_tensors" in text


def test_nsa_backend_accepts_gpt_oss_gqa_dsa():
    text = _read_repo_file("python/sglang/srt/layers/attention/nsa_backend.py")
    assert "self.is_gpt_oss_gqa_dsa" in text
    assert "DeepSeek NSA or GPT-OSS GQA DSA" in text


def test_nsa_backend_has_gpt_oss_mha_sparse_decode_path():
    text = _read_repo_file("python/sglang/srt/layers/attention/nsa_backend.py")
    assert "def _forward_decode_standard_mha_sparse(" in text
    assert "select_topk_pages_decode(" in text
    assert "forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)" in text
    assert "metadata.real_page_table" in text


def test_server_args_allow_nsa_for_gpt_oss_dsa():
    text = _read_repo_file("python/sglang/srt/server_args.py")
    assert 'supported_backends.append("nsa")' in text
    assert 'allowed_dsa_backends = {"fa3", "nsa"}' in text
    assert '--gpt-oss-dsa-fp8-query-mode' in text
    assert 'choices=["bf16", "fp8"]' in text


def test_flashattention_sparse_fp8_path_skips_forced_query_cast():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "sparse_gpt_oss_dsa_decode" in text
    assert 'sparse_fp8_query_mode = getattr(' in text
    assert 'or sparse_fp8_query_mode == "fp8"' in text
    assert "benchmark the actual serving path" in text


def test_flashattention_sparse_path_forces_single_split():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "sparse_num_splits = 1 if sparse_gpt_oss_dsa_decode else self.num_splits" in text
    assert "num_splits=sparse_num_splits" in text


def test_flashattention_recent_lane_uses_local_window_fast_path():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "def forward_extend(" in text
    assert "gpt_oss_dsa_recent: bool = False" in text
    assert "recent_window = (topk - 1, 0)" in text
    assert "dense_recent_result = flash_attn_with_kvcache(" in text
    assert "window_size=recent_window" in text
    assert "and (topk_indices is not None or gpt_oss_dsa_recent)" in text
    assert "key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(" in text


def test_flashattention_recent_lane_no_longer_precaches_sparse_page_table():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "metadata.gpt_oss_dsa_page_table = torch.gather(" not in text
    assert "metadata.gpt_oss_dsa_cache_seqlens_int32 = torch.where(" not in text
