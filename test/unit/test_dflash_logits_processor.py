from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _make_processor():
    proc = LogitsProcessor.__new__(LogitsProcessor)
    nn.Module.__init__(proc)
    proc._dflash_verify_lm_head_pad_buffers = {}
    return proc


def test_dflash_target_verify_lm_head_padding_expands_to_default_width():
    proc = _make_processor()
    hidden_states = torch.arange(20 * 4, dtype=torch.float32).view(20, 4)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.TARGET_VERIFY,
        dflash_draft_token_num=5,
        dflash_default_draft_token_num=16,
    )

    padded, raw_rows = proc._maybe_pad_dflash_target_verify_hidden_states(
        hidden_states, metadata
    )

    assert raw_rows == 20
    assert tuple(padded.shape) == (64, 4)
    assert torch.equal(padded[:20], hidden_states)
    assert torch.count_nonzero(padded[20:]) == 0


def test_dflash_target_verify_lm_head_padding_skips_full_width():
    proc = _make_processor()
    hidden_states = torch.arange(64 * 4, dtype=torch.float32).view(64, 4)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.TARGET_VERIFY,
        dflash_draft_token_num=16,
        dflash_default_draft_token_num=16,
    )

    padded, raw_rows = proc._maybe_pad_dflash_target_verify_hidden_states(
        hidden_states, metadata
    )

    assert raw_rows is None
    assert padded is hidden_states


def test_dflash_target_verify_lm_head_padding_skips_non_dflash_metadata():
    proc = _make_processor()
    hidden_states = torch.arange(20 * 4, dtype=torch.float32).view(20, 4)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.TARGET_VERIFY,
        dflash_draft_token_num=0,
        dflash_default_draft_token_num=16,
    )

    padded, raw_rows = proc._maybe_pad_dflash_target_verify_hidden_states(
        hidden_states, metadata
    )

    assert raw_rows is None
    assert padded is hidden_states


def test_dflash_target_verify_lm_head_padding_can_be_preallocated():
    proc = _make_processor()
    proc.reserve_dflash_target_verify_lm_head_pad_buffers(
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden_size=4,
        actual_width=5,
        default_width=16,
        batch_sizes=[1, 4],
    )

    assert ("cpu", torch.float32, 4, 16) in proc._dflash_verify_lm_head_pad_buffers
    assert ("cpu", torch.float32, 4, 64) in proc._dflash_verify_lm_head_pad_buffers


def test_logits_metadata_extracts_dflash_widths_from_forward_batch():
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        capture_hidden_mode=None,
        next_token_logits_buffer=None,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        extend_logprob_start_lens_cpu=None,
        extend_input_logprob_token_ids_gpu=None,
        padded_static_len=-1,
        is_prefill_only=False,
        global_num_tokens_gpu=None,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_dp_buffer_len=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_tokens_for_logprob_gpu=None,
        mm_input_embeds=None,
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: True),
        spec_info=SimpleNamespace(draft_token_num=5),
    )

    from sglang.srt.layers import logits_processor as logits_processor_module

    old_get_global_server_args = logits_processor_module.get_global_server_args
    logits_processor_module.get_global_server_args = lambda: SimpleNamespace(
        speculative_num_draft_tokens=16
    )
    try:
        metadata = LogitsMetadata.from_forward_batch(forward_batch)
    finally:
        logits_processor_module.get_global_server_args = old_get_global_server_args

    assert metadata.dflash_draft_token_num == 5
    assert metadata.dflash_default_draft_token_num == 16
