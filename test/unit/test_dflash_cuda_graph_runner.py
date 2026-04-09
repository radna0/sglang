from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    DFlashMultiWidthCudaGraphRunner,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _make_runner(*, capture_forward_mode, num_tokens_per_bs, max_bs=8):
    runner = CudaGraphRunner.__new__(CudaGraphRunner)
    runner.model_runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DFLASH)
    runner.capture_forward_mode = capture_forward_mode
    runner.num_tokens_per_bs = int(num_tokens_per_bs)
    runner.require_mlp_tp_gather = False
    runner.disable_padding = False
    runner.max_bs = int(max_bs)
    runner.graphs = {}
    runner.enable_pdmux = False
    runner.require_mlp_sync = False
    runner.is_encoder_decoder = False
    runner.capture_hidden_mode = CaptureHiddenMode.NULL
    runner.enable_two_batch_overlap = False
    return runner


def _make_forward_batch(*, forward_mode, batch_size, input_tokens, draft_token_num):
    return SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=int(batch_size),
        input_ids=torch.arange(int(input_tokens), dtype=torch.int64),
        spec_info=SimpleNamespace(
            capture_hidden_mode=CaptureHiddenMode.NULL,
            draft_token_num=int(draft_token_num),
        ),
        capture_hidden_mode=CaptureHiddenMode.NULL,
        encoder_lens=torch.ones((int(batch_size),), dtype=torch.int32),
        can_run_tbo=True,
    )


def test_dflash_target_verify_graph_rejects_reduced_width_batches():
    runner = _make_runner(
        capture_forward_mode=ForwardMode.TARGET_VERIFY,
        num_tokens_per_bs=16,
    )
    forward_batch = _make_forward_batch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=4,
        input_tokens=32,
        draft_token_num=8,
    )
    assert runner.can_run(forward_batch) is False


def test_dflash_target_verify_graph_accepts_full_width_batches():
    runner = _make_runner(
        capture_forward_mode=ForwardMode.TARGET_VERIFY,
        num_tokens_per_bs=16,
    )
    forward_batch = _make_forward_batch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=4,
        input_tokens=64,
        draft_token_num=16,
    )
    assert runner.can_run(forward_batch) is True


def test_dflash_decode_graph_still_accepts_plain_decode_batches():
    runner = _make_runner(
        capture_forward_mode=ForwardMode.DECODE,
        num_tokens_per_bs=1,
    )
    forward_batch = _make_forward_batch(
        forward_mode=ForwardMode.DECODE,
        batch_size=4,
        input_tokens=4,
        draft_token_num=1,
    )
    assert runner.can_run(forward_batch) is True


def test_dflash_multiwidth_cuda_graph_runner_selects_matching_width():
    wrapper = DFlashMultiWidthCudaGraphRunner.__new__(DFlashMultiWidthCudaGraphRunner)
    runner5 = SimpleNamespace(can_run=lambda fb: True, replay=lambda *a, **k: "w5", bs=5)
    runner16 = SimpleNamespace(can_run=lambda fb: True, replay=lambda *a, **k: "w16", bs=16)
    wrapper.default_width = 16
    wrapper.runners = {5: runner5, 16: runner16}
    wrapper.default_runner = runner16
    forward_batch = _make_forward_batch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=4,
        input_tokens=20,
        draft_token_num=5,
    )
    assert wrapper.can_run(forward_batch) is True
    assert wrapper.replay(forward_batch) == "w5"
    assert wrapper.bs == 5


def test_dflash_multiwidth_cuda_graph_runner_rejects_missing_width():
    wrapper = DFlashMultiWidthCudaGraphRunner.__new__(DFlashMultiWidthCudaGraphRunner)
    runner16 = SimpleNamespace(can_run=lambda fb: True, replay=lambda *a, **k: "w16", bs=16)
    wrapper.default_width = 16
    wrapper.runners = {16: runner16}
    wrapper.default_runner = runner16
    forward_batch = _make_forward_batch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=4,
        input_tokens=20,
        draft_token_num=5,
    )
    assert wrapper.can_run(forward_batch) is False
