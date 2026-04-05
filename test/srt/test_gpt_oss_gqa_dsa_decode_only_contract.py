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
    assert "valid_topk_mask = (idx >= 0) & (idx <= row_max)" in text
    assert "torch.full_like(idx, -1)" in text
    assert "seqlens_k_t = (sparse_page_table != -1).sum(dim=1).to(" in text


def test_flashattention_sparse_path_uses_sparse_page_table_decode():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "transform_index_page_table_decode(" in text
    assert "page_size=1" in text
    assert "key_cache_sparse = key_buf.view(" in text
    assert "value_cache_sparse = value_buf.view(" in text
    assert "page_table=sparse_page_table" in text


def test_flashattention_disables_dsa_when_seq_len_fits_topk_budget():
    text = _read_repo_file(
        "python/sglang/srt/layers/attention/flashattention_backend.py"
    )
    assert "sparse_rows = valid_rows & (seq_lens > topk)" in text
    assert "if not torch.any(sparse_rows):" in text
    assert "topk_indices = None" in text
    assert "dense_rows = valid_rows & ~sparse_rows" in text


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
