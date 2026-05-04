import torch
import pytest
from types import SimpleNamespace


class _FakeGrammar:
    def __init__(self):
        self.accepted = []

    def accept_token(self, token_id):
        self.accepted.append(int(token_id))


class _FakeReq:
    def __init__(
        self,
        *,
        max_new_tokens=16,
        stop_token_ids=None,
        ignore_eos=False,
        grammar=None,
        output_ids=None,
        origin_input_ids=None,
        eos_token_ids=None,
        finish_after=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        min_p=0.0,
    ):
        self.output_ids = list(output_ids or [])
        self.origin_input_ids = list(origin_input_ids or [1])
        self.fill_ids = self.origin_input_ids + self.output_ids
        self.eos_token_ids = list(eos_token_ids or [])
        self.tokenizer = SimpleNamespace(
            eos_token_id=None,
            additional_stop_token_ids=[],
        )
        self.vocab_size = None
        self.grammar = grammar
        self.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens,
            stop_strs=[],
            stop_regex_strs=[],
            ignore_eos=ignore_eos,
            stop_token_ids=list(stop_token_ids or []),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        self._finished = False
        self._finish_after = finish_after
        self.check_finished_calls = []
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0
        self.hist = []
        self.finished_reason = None
        self.finished_len = None
        self.to_finish = None
        self.req_pool_idx = None
        self.is_retracted = False
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.decode_batch_idx = 0
        self.time_stats = SimpleNamespace(completion_time=None)

    def check_finished(self, new_accepted_len=None):
        self.check_finished_calls.append(new_accepted_len)
        if self._finish_after is not None and len(self.output_ids) >= int(self._finish_after):
            self._finished = True
            return
        if len(self.output_ids) >= int(self.sampling_params.max_new_tokens):
            self.finished_reason = ("length", int(self.sampling_params.max_new_tokens))
            self.finished_len = int(self.sampling_params.max_new_tokens)
            return
        if not self.sampling_params.ignore_eos and new_accepted_len:
            new_tokens = self.output_ids[-int(new_accepted_len):]
            stop_ids = set(int(tok) for tok in self.sampling_params.stop_token_ids)
            stop_ids.update(int(tok) for tok in self.eos_token_ids)
            if self.tokenizer.eos_token_id is not None:
                stop_ids.add(int(self.tokenizer.eos_token_id))
            if self.tokenizer.additional_stop_token_ids:
                stop_ids.update(int(tok) for tok in self.tokenizer.additional_stop_token_ids)
            for token_id in new_tokens:
                if int(token_id) in stop_ids:
                    self.finished_reason = ("token", int(token_id))
                    return

    def finished(self):
        return self._finished or self.finished_reason is not None

    def update_spec_acceptance_histogram(self, accepted_draft_tokens):
        self.hist.append(int(accepted_draft_tokens))


class _FakeAllocator:
    def __init__(self):
        self.freed = []
        self.freed_pages = []

    def free(self, free_index):
        self.freed.append(free_index.clone())

    def free_page_indices(self, free_page_indices):
        self.freed_pages.append(free_page_indices.clone())


class _FakeEvent:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True


class _FakeBatch:
    def __init__(self):
        self.token_to_kv_pool_allocator = _FakeAllocator()
        self.out_cache_loc = None


class _FakeScheduleBatchForDecode:
    def __init__(self):
        self.forward_mode = SimpleNamespace(is_idle=lambda: False)
        self.evicted = False
        self.waited = False
        self.input_ids = None
        self.seq_lens = torch.tensor([1, 2], dtype=torch.int32)
        self.seq_lens_cpu = torch.tensor([1, 2], dtype=torch.int32)

    def maybe_evict_swa(self):
        self.evicted = True

    def maybe_wait_verify_done(self):
        self.waited = True


class _FakeModelWorkerBatch:
    def __init__(self, spec_info):
        self.spec_info = spec_info
        self.input_ids = torch.empty((0,), dtype=torch.int64)


class _FakeTreeCache:
    def __init__(self):
        self.cached = []

    def cache_unfinished_req(self, req):
        self.cached.append(req)


class _FakeReqToTokenPool:
    def __init__(self, rows: torch.Tensor):
        self.req_to_token = rows.clone()

    def write(self, indices, values):
        self.req_to_token[indices] = values


def test_build_target_layer_ids_gptoss120b_k5():
    from sglang.srt.speculative.dflash_utils import build_target_layer_ids

    # GPT-OSS-120B: 36 layers; typical K=5 draft selects evenly spaced ids.
    assert build_target_layer_ids(36, 5) == [1, 9, 17, 25, 33]


def test_parse_dflash_draft_config_basic():
    from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

    cfg = {
        "num_hidden_layers": 5,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "dflash_config": {
            "block_size": 16,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "mask_token": "<|MASK|>",
        },
    }

    parsed = parse_dflash_draft_config(draft_hf_config=cfg)
    assert parsed.require_num_layers() == 5
    assert parsed.block_size == 16
    assert parsed.resolve_target_layer_ids(target_num_layers=36, draft_num_layers=5) == [
        1,
        9,
        17,
        25,
        33,
    ]
    assert parsed.mask_token == "<|MASK|>"


def test_resolve_dflash_capture_contract_reports_plus_one_capture_mapping():
    from sglang.srt.speculative.dflash_utils import resolve_dflash_capture_contract

    cfg = {
        "num_hidden_layers": 5,
        "dflash_config": {
            "block_size": 16,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "mask_token": "<|MASK|>",
            "mask_token_id": 200019,
        },
    }

    contract = resolve_dflash_capture_contract(
        draft_hf_config=cfg,
        runtime_target_num_layers=36,
    )

    assert contract.runtime_target_num_layers == 36
    assert contract.draft_num_layers == 5
    assert contract.block_size == 16
    assert contract.mask_token_id == 200019
    assert contract.target_layer_ids == [1, 9, 17, 25, 33]
    assert contract.capture_layer_ids == [2, 10, 18, 26, 34]


def test_compute_dflash_accept_len_and_bonus():
    from sglang.srt.speculative.dflash_utils import compute_dflash_accept_len_and_bonus

    # block_size=5: candidates include current token at index 0.
    # Accept rule compares candidates[:,1:] to target_predict[:,:-1] consecutively.
    candidates = torch.tensor([[10, 11, 12, 99, 99]], dtype=torch.int64)
    target_predict = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int64)

    accept_len, bonus = compute_dflash_accept_len_and_bonus(
        candidates=candidates, target_predict=target_predict
    )
    # Accept tokens 11 and 12 (2 tokens), then stop at mismatch (99 != 13).
    assert accept_len.tolist() == [2]
    # Bonus token is target_predict at index accept_len (2) => 13.
    assert bonus.tolist() == [13]


def test_dflash_supports_spec_v2_and_selects_v2_worker():
    from types import SimpleNamespace

    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    assert SpeculativeAlgorithm.DFLASH.supports_spec_v2() is True
    worker_cls = SpeculativeAlgorithm.DFLASH.create_worker(
        SimpleNamespace(disable_overlap_schedule=False)
    )
    assert worker_cls is DFlashWorkerV2


def test_dflash_tree_supports_spec_v2_and_selects_tree_workers():
    from types import SimpleNamespace

    from sglang.srt.speculative.dflash_tree_worker import DFlashTreeWorker
    from sglang.srt.speculative.dflash_tree_worker_v2 import DFlashTreeWorkerV2
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    assert SpeculativeAlgorithm.DFLASH_TREE.supports_spec_v2() is True
    worker_cls = SpeculativeAlgorithm.DFLASH_TREE.create_worker(
        SimpleNamespace(disable_overlap_schedule=True)
    )
    assert worker_cls is DFlashTreeWorker
    overlap_worker_cls = SpeculativeAlgorithm.DFLASH_TREE.create_worker(
        SimpleNamespace(disable_overlap_schedule=False)
    )
    assert overlap_worker_cls is DFlashTreeWorkerV2


def test_snapshot_dflash_request_sampling_params_builds_tensors():
    from sglang.srt.speculative.dflash_utils import (
        snapshot_dflash_request_sampling_params,
    )

    reqs = [
        _FakeReq(temperature=0.7, top_p=0.9, top_k=17, min_p=0.0),
        _FakeReq(temperature=1.3, top_p=0.95, top_k=5, min_p=0.02),
    ]

    temperatures, top_ps, top_ks, min_ps, need_min_p_sampling = (
        snapshot_dflash_request_sampling_params(reqs)
    )

    assert temperatures.dtype == torch.float32
    assert top_ps.dtype == torch.float32
    assert top_ks.dtype == torch.int32
    assert min_ps.dtype == torch.float32
    assert torch.allclose(temperatures, torch.tensor([0.7, 1.3], dtype=torch.float32))
    assert torch.allclose(top_ps, torch.tensor([0.9, 0.95], dtype=torch.float32))
    assert top_ks.tolist() == [17, 5]
    assert torch.allclose(min_ps, torch.tensor([0.0, 0.02], dtype=torch.float32))
    assert need_min_p_sampling is True


def test_snapshot_dflash_request_sampling_params_device_cpu_path():
    from sglang.srt.speculative.dflash_utils import (
        snapshot_dflash_request_sampling_params,
    )

    reqs = [
        _FakeReq(temperature=0.8, top_p=0.92, top_k=9, min_p=0.0),
        _FakeReq(temperature=1.1, top_p=0.99, top_k=3, min_p=0.05),
    ]

    temperatures, top_ps, top_ks, min_ps, need_min_p_sampling = (
        snapshot_dflash_request_sampling_params(reqs, device="cpu")
    )

    assert temperatures.device.type == "cpu"
    assert top_ps.device.type == "cpu"
    assert top_ks.device.type == "cpu"
    assert min_ps.device.type == "cpu"
    assert torch.allclose(temperatures, torch.tensor([0.8, 1.1], dtype=torch.float32))
    assert torch.allclose(top_ps, torch.tensor([0.92, 0.99], dtype=torch.float32))
    assert top_ks.tolist() == [9, 3]
    assert torch.allclose(min_ps, torch.tensor([0.0, 0.05], dtype=torch.float32))
    assert need_min_p_sampling is True


def test_pack_dflash_target_only_commits():
    from sglang.srt.speculative.dflash_utils import pack_dflash_target_only_commits

    target_predict = torch.tensor(
        [
            [11, 12, 13, 14],
            [21, 22, 23, 24],
        ],
        dtype=torch.int64,
    )
    accept_len = torch.tensor([2, 0], dtype=torch.int32)

    packed = pack_dflash_target_only_commits(
        target_predict=target_predict,
        accept_len=accept_len,
    )
    assert packed.commit_lens.tolist() == [3, 1]
    assert packed.proposed_flat.tolist() == [11, 12, 13, 21]
    assert packed.commit_offsets.tolist() == [0, 3, 4]
    assert packed.default_new_verified_id.tolist() == [13, 21]


def test_pack_dflash_indexed_commits():
    from sglang.srt.speculative.dflash_utils import pack_dflash_indexed_commits

    predict = torch.tensor([41, 42, 43, 44, 51, 52], dtype=torch.int64)
    accept_index = torch.tensor(
        [
            [0, 2, 3, -1],
            [4, -1, -1, -1],
        ],
        dtype=torch.int32,
    )

    packed = pack_dflash_indexed_commits(
        predict=predict,
        accept_index=accept_index,
    )
    assert packed.proposed_flat.tolist() == [41, 43, 44, 51]
    assert packed.commit_lens.tolist() == [3, 1]
    assert packed.commit_offsets.tolist() == [0, 3, 4]
    assert packed.default_new_verified_id.tolist() == [44, 51]


def test_verify_dflash_tree_greedy_fallback_matches_kernel_example():
    from sglang.srt.speculative.dflash_utils import (
        verify_dflash_tree_greedy_fallback,
    )

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
    )
    target_predict = torch.tensor(
        [
            [3, 18, 18, 4, 5, 18],
            [11, 18, 18, 18, 12, 18],
        ],
        dtype=torch.int64,
    )

    predicts, accept_index, accept_token_num = verify_dflash_tree_greedy_fallback(
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
        num_speculative_tokens=4,
        device="cpu",
    )

    assert predicts.tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
    assert accept_index.tolist() == [[0, 3, 4, 5], [6, 10, 11, -1]]
    assert accept_token_num.tolist() == [3, 2]


def test_build_dflash_tree_candidates_from_per_step_topk_matches_generic_builder():
    from sglang.srt.speculative.dflash_utils import (
        build_dflash_tree_candidates_from_per_step_topk,
    )
    from sglang.srt.speculative.eagle_utils import organize_draft_results
    from sglang.srt.speculative.spec_utils import select_top_k_tokens

    topk_p = torch.tensor(
        [
            [
                [0.9, 0.8],
                [0.7, 0.6],
                [0.5, 0.4],
            ]
        ],
        dtype=torch.float32,
    )
    topk_index = torch.tensor(
        [
            [
                [10, 11],
                [20, 21],
                [30, 31],
            ]
        ],
        dtype=torch.int64,
    )
    topk = int(topk_p.shape[-1])

    score_list = []
    token_list = []
    parents_list = []
    scores = None
    hidden_dummy = torch.empty((0, 0), dtype=torch.float32)

    for i in range(int(topk_p.shape[1])):
        step_p = topk_p[:, i, :]
        step_ids = topk_index[:, i, :]
        if i == 0:
            _, hidden_dummy, scores, tree_info = select_top_k_tokens(
                i, step_p, step_ids, hidden_dummy, scores, topk
            )
        else:
            _, hidden_dummy, scores, tree_info = select_top_k_tokens(
                i,
                step_p.repeat_interleave(topk, dim=0),
                step_ids.repeat_interleave(topk, dim=0),
                hidden_dummy,
                scores,
                topk,
            )
        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])

    ref_parent_list, ref_top_scores_index, ref_draft_tokens = organize_draft_results(
        score_list, token_list, parents_list, num_draft_token=4
    )
    got_parent_list, got_top_scores_index, got_draft_tokens = (
        build_dflash_tree_candidates_from_per_step_topk(
            topk_p=topk_p,
            topk_index=topk_index,
            num_verify_tokens=4,
        )
    )

    assert torch.equal(got_parent_list, ref_parent_list)
    assert torch.equal(got_top_scores_index, ref_top_scores_index)
    assert torch.equal(got_draft_tokens, ref_draft_tokens)


@pytest.mark.parametrize(
    ("bs", "step_count", "topk", "num_verify_tokens"),
    [
        (2, 3, 2, 4),
        (3, 4, 2, 5),
        (2, 4, 4, 8),
        (1, 5, 4, 12),
    ],
)
def test_build_dflash_tree_candidates_from_per_step_topk_matches_generic_builder_randomized(
    bs, step_count, topk, num_verify_tokens
):
    from sglang.srt.speculative.dflash_utils import (
        build_dflash_tree_candidates_from_per_step_topk,
    )
    from sglang.srt.speculative.eagle_utils import organize_draft_results
    from sglang.srt.speculative.spec_utils import select_top_k_tokens

    g = torch.Generator().manual_seed(1234 + bs * 100 + step_count * 10 + topk)
    raw_scores = torch.rand((bs, step_count, topk), generator=g, dtype=torch.float32)
    topk_p = raw_scores / raw_scores.sum(dim=-1, keepdim=True)

    # Use distinct token ids per request/step/branch so mismatches are easy to spot.
    token_base = torch.arange(bs * step_count * topk, dtype=torch.int64).view(
        bs, step_count, topk
    )
    topk_index = token_base + 1000

    score_list = []
    token_list = []
    parents_list = []
    scores = None
    hidden_dummy = torch.empty((0, 0), dtype=torch.float32)

    for i in range(step_count):
        step_p = topk_p[:, i, :]
        step_ids = topk_index[:, i, :]
        if i == 0:
            _, hidden_dummy, scores, tree_info = select_top_k_tokens(
                i, step_p, step_ids, hidden_dummy, scores, topk
            )
        else:
            _, hidden_dummy, scores, tree_info = select_top_k_tokens(
                i,
                step_p.repeat_interleave(topk, dim=0),
                step_ids.repeat_interleave(topk, dim=0),
                hidden_dummy,
                scores,
                topk,
            )
        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])

    ref_parent_list, ref_top_scores_index, ref_draft_tokens = organize_draft_results(
        score_list, token_list, parents_list, num_draft_token=num_verify_tokens
    )
    got_parent_list, got_top_scores_index, got_draft_tokens = (
        build_dflash_tree_candidates_from_per_step_topk(
            topk_p=topk_p,
            topk_index=topk_index,
            num_verify_tokens=num_verify_tokens,
        )
    )

    assert torch.equal(got_parent_list, ref_parent_list)
    assert torch.equal(got_top_scores_index, ref_top_scores_index)
    assert torch.equal(got_draft_tokens, ref_draft_tokens)


def test_build_dflash_tree_candidates_from_per_step_topk_reuses_output_buffers():
    from sglang.srt.speculative.dflash_utils import (
        build_dflash_tree_candidates_from_per_step_topk,
    )

    topk_p = torch.tensor(
        [
            [
                [0.9, 0.8],
                [0.7, 0.6],
                [0.5, 0.4],
            ]
        ],
        dtype=torch.float32,
    )
    topk_index = torch.tensor(
        [
            [
                [10, 11],
                [20, 21],
                [30, 31],
            ]
        ],
        dtype=torch.int64,
    )

    score_buf = torch.full((1, 10), -1.0, dtype=torch.float32)
    token_buf = torch.full((1, 10), -1, dtype=torch.int64)
    parent_buf = torch.full((1, 5), -9, dtype=torch.int64)
    index_buf = torch.full((1, 3), -1, dtype=torch.int64)

    parent_list, top_scores_index, draft_tokens = (
        build_dflash_tree_candidates_from_per_step_topk(
            topk_p=topk_p,
            topk_index=topk_index,
            num_verify_tokens=4,
            candidate_scores_buf=score_buf,
            candidate_tokens_buf=token_buf,
            parent_list_buf=parent_buf,
            top_scores_index_buf=index_buf,
        )
    )

    assert parent_list.data_ptr() == parent_buf.data_ptr()
    assert top_scores_index.data_ptr() == index_buf.data_ptr()
    assert parent_list.tolist() == [[-1, 0, 1, 2, 4]]
    assert top_scores_index.tolist() == [[0, 1, 2]]
    assert draft_tokens.tolist() == [[10, 11, 20]]


def test_sample_dflash_tree_branch_candidates_vectorized_paths():
    from sglang.srt.speculative.dflash_utils import (
        sample_dflash_tree_branch_candidates,
    )

    probs = torch.tensor(
        [
            [0.40, 0.30, 0.20, 0.10],  # enough support, no replacement
            [1.00, 0.00, 0.00, 0.00],  # replacement required
            [0.00, 0.00, 0.00, 0.00],  # fallback to logits
        ],
        dtype=torch.float32,
    )
    logits = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0],
            [9.0, 0.0, 0.0, 0.0],
            [0.5, 3.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(0)
    sampled_probs, sampled_ids = sample_dflash_tree_branch_candidates(
        probs=probs,
        logits=logits,
        topk=2,
    )

    assert sampled_probs.shape == (3, 2)
    assert sampled_ids.shape == (3, 2)

    row0_probs = probs[0].gather(0, sampled_ids[0])
    assert torch.allclose(sampled_probs[0], row0_probs, atol=1e-6)
    assert sampled_probs[0, 0] >= sampled_probs[0, 1]
    assert int(torch.unique(sampled_ids[0]).numel()) == 2

    assert sampled_ids[1].tolist() == [0, 0]
    assert torch.allclose(sampled_probs[1], torch.tensor([1.0, 1.0]), atol=1e-6)

    ref_vals, ref_ids = torch.topk(logits[2], k=2, dim=-1)
    ref_probs = torch.softmax(ref_vals, dim=-1)
    assert torch.equal(sampled_ids[2], ref_ids.to(torch.int64))
    assert torch.allclose(sampled_probs[2], ref_probs.to(torch.float32), atol=1e-6)


def test_sample_dflash_tree_branch_candidates_from_support():
    from sglang.srt.speculative.dflash_utils import (
        sample_dflash_tree_branch_candidates_from_support,
    )

    probs = torch.tensor(
        [
            [0.60, 0.30, 0.10, 0.00],
            [1.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )
    token_ids = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=torch.int64,
    )
    fallback_probs = torch.tensor(
        [
            [0.60, 0.30, 0.10, 0.00],
            [1.00, 0.00, 0.00, 0.00],
            [0.70, 0.20, 0.10, 0.00],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(0)
    sampled_probs, sampled_ids = sample_dflash_tree_branch_candidates_from_support(
        probs=probs,
        token_ids=token_ids,
        topk=2,
        fallback_probs=fallback_probs,
    )

    assert sampled_probs.shape == (3, 2)
    assert sampled_ids.shape == (3, 2)
    assert sampled_probs[0, 0] >= sampled_probs[0, 1]
    assert int(torch.unique(sampled_ids[0]).numel()) == 2
    assert sampled_ids[1].tolist() == [20, 20]
    assert sampled_probs[1].tolist() == [1.0, 1.0]
    assert sampled_ids[2].tolist() == [30, 31]
    assert torch.allclose(sampled_probs[2], torch.tensor([0.7777778, 0.2222222]), atol=1e-6)


def test_dflash_tree_topk_from_vocab_parallel_head_fast_path(monkeypatch):
    from sglang.srt.speculative.dflash_tree_worker import DFlashTreeWorker
    import sglang.srt.speculative.dflash_tree_worker as dflash_tree_worker_mod

    class _FakeTpGroup:
        world_size = 1

    monkeypatch.setattr(dflash_tree_worker_mod, "get_tp_group", lambda: _FakeTpGroup())

    lm_head = SimpleNamespace(
        weight=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        shard_indices=SimpleNamespace(
            num_org_elements=3,
            num_org_elements_padded=3,
            num_added_elements=0,
            org_vocab_start_index=0,
            added_vocab_start_index=3,
        ),
    )
    hidden_states = torch.tensor(
        [
            [2.0, 0.5],
            [0.1, 3.0],
        ],
        dtype=torch.float32,
    )

    probs, token_ids = DFlashTreeWorker._topk_from_vocab_parallel_head(
        object(),
        hidden_states=hidden_states,
        lm_head=lm_head,
        topk=2,
        chunk_size=8,
    )

    logits = hidden_states @ lm_head.weight.T
    ref_vals, ref_ids = torch.topk(logits, k=2, dim=-1)
    ref_probs = torch.softmax(ref_vals, dim=-1)

    assert torch.equal(token_ids, ref_ids.to(torch.int64))
    assert torch.allclose(probs, ref_probs.to(torch.float32), atol=1e-6)

    logits_out, logits_ids = DFlashTreeWorker._topk_from_vocab_parallel_head(
        object(),
        hidden_states=hidden_states,
        lm_head=lm_head,
        topk=2,
        chunk_size=8,
        return_logits=True,
    )
    assert torch.equal(logits_ids, ref_ids.to(torch.int64))
    assert torch.allclose(logits_out, ref_vals.to(torch.float32), atol=1e-6)


def test_resolve_dflash_overlap_token_ids():
    from sglang.srt.speculative.dflash_utils import resolve_dflash_overlap_token_ids

    got = resolve_dflash_overlap_token_ids(
        flat_token_ids=torch.tensor([11, 12, 13, 21], dtype=torch.int64),
        accept_lens=torch.tensor([3, 1], dtype=torch.int32),
    )
    assert got == [[11, 12, 13], [21]]


def test_resolve_dflash_indexed_accept_indices():
    from sglang.srt.speculative.dflash_utils import (
        resolve_dflash_indexed_accept_indices,
    )

    accept_index = torch.tensor(
        [
            [0, 2, 3, -1],
            [4, 5, -1, -1],
        ],
        dtype=torch.int32,
    )
    commit_lens = torch.tensor([2, 1], dtype=torch.int32)

    got = resolve_dflash_indexed_accept_indices(
        accept_index=accept_index,
        commit_lens=commit_lens,
    )
    assert got.tolist() == [0, 2, 4]


def test_materialize_dflash_target_only_commit_metadata():
    from sglang.srt.speculative.dflash_utils import (
        DFlashTargetOnlyCommitResult,
        materialize_dflash_target_only_commit_metadata,
    )

    metadata = materialize_dflash_target_only_commit_metadata(
        commit_results=[
            DFlashTargetOnlyCommitResult(
                commit_len=3,
                new_verified_token=13,
                accepted_draft_tokens=2,
                used_device_defaults=False,
            ),
            DFlashTargetOnlyCommitResult(
                commit_len=0,
                new_verified_token=55,
                accepted_draft_tokens=0,
                used_device_defaults=False,
            ),
        ],
        device=torch.device("cpu"),
    )
    assert metadata.commit_lens.tolist() == [3, 0]
    assert metadata.new_verified_id.tolist() == [13, 55]


def test_materialize_dflash_target_only_commit_metadata_uses_device_defaults():
    from sglang.srt.speculative.dflash_utils import (
        DFlashTargetOnlyCommitResult,
        materialize_dflash_target_only_commit_metadata,
    )

    default_commit_lens = torch.tensor([3, 1], dtype=torch.int32)
    default_new_verified_id = torch.tensor([13, 21], dtype=torch.int64)
    metadata = materialize_dflash_target_only_commit_metadata(
        commit_results=[
            DFlashTargetOnlyCommitResult(
                commit_len=3,
                new_verified_token=13,
                accepted_draft_tokens=2,
                used_device_defaults=True,
            ),
            DFlashTargetOnlyCommitResult(
                commit_len=1,
                new_verified_token=21,
                accepted_draft_tokens=0,
                used_device_defaults=True,
            ),
        ],
        device=torch.device("cpu"),
        default_commit_lens=default_commit_lens,
        default_new_verified_id=default_new_verified_id,
    )
    assert metadata.commit_lens is default_commit_lens
    assert metadata.new_verified_id is default_new_verified_id


def test_materialize_dflash_target_only_commit_metadata_mixed_overrides():
    from sglang.srt.speculative.dflash_utils import (
        DFlashTargetOnlyCommitResult,
        materialize_dflash_target_only_commit_metadata,
    )

    metadata = materialize_dflash_target_only_commit_metadata(
        commit_results=[
            DFlashTargetOnlyCommitResult(
                commit_len=3,
                new_verified_token=13,
                accepted_draft_tokens=2,
                used_device_defaults=True,
            ),
            DFlashTargetOnlyCommitResult(
                commit_len=0,
                new_verified_token=55,
                accepted_draft_tokens=0,
                used_device_defaults=False,
            ),
        ],
        device=torch.device("cpu"),
        default_commit_lens=torch.tensor([3, 1], dtype=torch.int32),
        default_new_verified_id=torch.tensor([13, 21], dtype=torch.int64),
    )
    assert metadata.commit_lens.tolist() == [3, 0]
    assert metadata.new_verified_id.tolist() == [13, 55]


def test_build_dflash_target_only_cache_plan_page_size_1():
    from sglang.srt.speculative.dflash_utils import build_dflash_target_only_cache_plan

    plan = build_dflash_target_only_cache_plan(
        out_cache_loc=torch.tensor([100, 101, 102, 103, 200, 201, 202, 203], dtype=torch.int64),
        commit_lens=torch.tensor([3, 1], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        draft_token_num=4,
        page_size=1,
    )
    assert plan.keep_mask.tolist() == [
        [True, True, True, False],
        [True, False, False, False],
    ]
    assert plan.accepted_indices.tolist() == [0, 1, 2, 4]
    assert plan.compact_out_cache_loc.tolist() == [100, 101, 102, 200]
    assert plan.evicted_slots.tolist() == [103, 201, 202, 203]
    assert plan.evicted_pages is None
    assert plan.clear_start.tolist() == [8, 8]
    assert plan.clear_end.tolist() == [9, 11]
    assert plan.clear_token_count == 4


def test_build_dflash_target_only_cache_plan_page_size_2_cpu_alignment():
    from sglang.srt.speculative.dflash_utils import build_dflash_target_only_cache_plan

    plan = build_dflash_target_only_cache_plan(
        out_cache_loc=torch.tensor([100, 101, 102, 103], dtype=torch.int64),
        commit_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        draft_token_num=4,
        page_size=2,
    )
    assert plan.keep_mask.tolist() == [[True, False, False, False]]
    assert plan.accepted_indices.tolist() == [0]
    assert plan.compact_out_cache_loc.tolist() == [100]
    assert plan.evicted_slots.tolist() == [102, 103]
    assert plan.evicted_pages.tolist() == [51]
    assert plan.clear_start.tolist() == [5]
    assert plan.clear_end.tolist() == [8]
    assert plan.clear_token_count == 3


def test_build_dflash_shared_pool_append_plan():
    from sglang.srt.speculative.dflash_utils import (
        build_dflash_shared_pool_append_plan,
    )

    plan = build_dflash_shared_pool_append_plan(
        draft_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        commit_lens=torch.tensor([3, 1], dtype=torch.int32),
        compact_out_cache_loc=torch.tensor([100, 101, 102, 200], dtype=torch.int64),
    )
    assert plan.total_ctx == 4
    assert plan.ctx_positions.tolist() == [5, 6, 7, 7]
    assert plan.ctx_cache_loc.tolist() == [100, 101, 102, 200]


def test_build_dflash_shared_pool_append_plan_from_flat_positions():
    from sglang.srt.speculative.dflash_utils import (
        build_dflash_shared_pool_append_plan_from_flat_positions,
    )

    plan = build_dflash_shared_pool_append_plan_from_flat_positions(
        positions=torch.tensor([5, 6, 7, 8, 7, 8, 9, 10], dtype=torch.int64),
        accepted_indices=torch.tensor([0, 1, 2, 4], dtype=torch.int64),
        compact_out_cache_loc=torch.tensor([100, 101, 102, 200], dtype=torch.int64),
    )
    assert plan.total_ctx == 4
    assert plan.ctx_positions.tolist() == [5, 6, 7, 7]
    assert plan.ctx_cache_loc.tolist() == [100, 101, 102, 200]


def test_apply_dflash_shared_pool_verify_append_updates_draft_input_and_calls_writer():
    from sglang.srt.speculative.dflash_utils import apply_dflash_shared_pool_verify_append

    draft_input = SimpleNamespace(
        draft_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        new_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        ctx_lens=torch.tensor([0, 0], dtype=torch.int32),
        target_hidden=torch.randn(8, 4),
    )
    cache_plan = SimpleNamespace(
        accepted_indices=torch.tensor([0, 1, 2, 4], dtype=torch.int64),
        compact_out_cache_loc=torch.tensor([100, 101, 102, 200], dtype=torch.int64),
    )
    verify_positions = torch.tensor([5, 6, 7, 8, 7, 8, 9, 10], dtype=torch.int64)
    hidden_states = torch.randn(8, 4)
    commit_lens = torch.tensor([3, 1], dtype=torch.int32)
    calls = {}

    def _writer(*, hidden_states, accepted_indices, ctx_positions, ctx_cache_loc):
        calls["hidden_rows"] = int(hidden_states.shape[0])
        calls["accepted_indices"] = accepted_indices.tolist()
        calls["ctx_positions"] = ctx_positions.tolist()
        calls["ctx_cache_loc"] = ctx_cache_loc.tolist()

    plan = apply_dflash_shared_pool_verify_append(
        draft_input=draft_input,
        verify_positions=verify_positions,
        hidden_states=hidden_states,
        cache_plan=cache_plan,
        commit_lens=commit_lens,
        write_selected_hidden=_writer,
    )
    assert plan.total_ctx == 4
    assert calls["accepted_indices"] == [0, 1, 2, 4]
    assert calls["ctx_positions"] == [5, 6, 7, 7]
    assert calls["ctx_cache_loc"] == [100, 101, 102, 200]
    assert draft_input.draft_seq_lens.tolist() == [8, 8]
    assert draft_input.new_seq_lens.tolist() == [8, 8]
    assert draft_input.ctx_lens.tolist() == [0, 0]
    assert tuple(draft_input.target_hidden.shape) == (0, 4)


def test_apply_dflash_target_only_req_kv_accounting_commits_lengths():
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_target_only_req_kv_accounting,
    )

    req0 = _FakeReq()
    req1 = _FakeReq()
    req0.kv_committed_len = 11
    req0.kv_allocated_len = 11
    req1.kv_committed_len = 7
    req1.kv_allocated_len = 7

    apply_dflash_target_only_req_kv_accounting(
        reqs=[req0, req1],
        commit_lens_cpu=[2, 1],
    )

    assert req0.decode_batch_idx == 2
    assert req0.kv_committed_len == 13
    assert req0.kv_allocated_len == 13
    assert req1.decode_batch_idx == 1
    assert req1.kv_committed_len == 8
    assert req1.kv_allocated_len == 8


def test_apply_dflash_target_only_req_kv_accounting_preserves_overlap_tail():
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_target_only_req_kv_accounting,
    )

    req = _FakeReq()
    req.kv_committed_len = 11
    req.kv_allocated_len = 27

    apply_dflash_target_only_req_kv_accounting(
        reqs=[req],
        commit_lens_cpu=[2],
        preserve_allocated_len=True,
    )

    assert req.decode_batch_idx == 2
    assert req.kv_committed_len == 13
    assert req.kv_allocated_len == 27


def test_apply_dflash_target_only_cache_plan_page_size_1():
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_target_only_cache_plan,
        build_dflash_target_only_cache_plan,
    )

    batch = _FakeBatch()
    batch.out_cache_loc = torch.tensor(
        [100, 101, 102, 103, 200, 201, 202, 203], dtype=torch.int64
    )
    plan = build_dflash_target_only_cache_plan(
        out_cache_loc=batch.out_cache_loc,
        commit_lens=torch.tensor([3, 1], dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        draft_token_num=4,
        page_size=1,
    )
    apply_dflash_target_only_cache_plan(batch=batch, cache_plan=plan, page_size=1)
    assert len(batch.token_to_kv_pool_allocator.freed) == 1
    assert batch.token_to_kv_pool_allocator.freed[0].tolist() == [103, 201, 202, 203]
    assert batch.out_cache_loc.tolist() == [100, 101, 102, 200]


def test_apply_dflash_target_only_cache_plan_page_size_2_uses_page_free():
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_target_only_cache_plan,
        build_dflash_target_only_cache_plan,
    )

    batch = _FakeBatch()
    batch.out_cache_loc = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
    plan = build_dflash_target_only_cache_plan(
        out_cache_loc=batch.out_cache_loc,
        commit_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        draft_token_num=4,
        page_size=2,
    )
    apply_dflash_target_only_cache_plan(batch=batch, cache_plan=plan, page_size=2)
    assert len(batch.token_to_kv_pool_allocator.freed_pages) == 1
    assert batch.token_to_kv_pool_allocator.freed_pages[0].tolist() == [51]
    assert batch.out_cache_loc.tolist() == [100]


def test_dflash_draft_input_future_filter_and_merge():
    from sglang.srt.managers.overlap_utils import FutureIndices
    from sglang.srt.speculative.dflash_info import DFlashDraftInput

    draft = DFlashDraftInput(
        verified_id=torch.tensor([11, 22], dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.zeros((2,), dtype=torch.int32),
        draft_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        future_indices=FutureIndices(indices=torch.tensor([101, 102], dtype=torch.int64)),
        new_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
    )
    draft.filter_batch(torch.tensor([1], dtype=torch.int64))
    assert draft.future_indices.indices.tolist() == [102]
    assert draft.verified_id.tolist() == [22]
    assert draft.draft_seq_lens.tolist() == [7]
    assert draft.new_seq_lens.tolist() == [7]

    other = DFlashDraftInput(
        verified_id=torch.tensor([33], dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.zeros((1,), dtype=torch.int32),
        draft_seq_lens=torch.tensor([9], dtype=torch.int32),
        future_indices=FutureIndices(indices=torch.tensor([103], dtype=torch.int64)),
        new_seq_lens=torch.tensor([9], dtype=torch.int32),
    )
    draft.merge_batch(other)
    assert draft.future_indices.indices.tolist() == [102, 103]
    assert draft.verified_id.tolist() == [22, 33]
    assert draft.draft_seq_lens.tolist() == [7, 9]
    assert draft.new_seq_lens.tolist() == [7, 9]


def test_future_map_dflash_roundtrip_uses_post_append_state():
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.speculative.dflash_info import DFlashDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    future_map = FutureMap(
        max_running_requests=4,
        chunked_prefill_size=0,
        context_len=16,
        device=torch.device("cpu"),
        spec_algo=SpeculativeAlgorithm.DFLASH,
    )
    future_indices = future_map.alloc_future_indices(2)
    stored = DFlashDraftInput(
        verified_id=torch.tensor([11, 22], dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.zeros((2,), dtype=torch.int32),
        draft_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        new_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
    )
    future_map.store_to_map_for_new_batch(future_indices, stored)

    unresolved = DFlashDraftInput(
        verified_id=torch.empty((0,), dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.empty((0,), dtype=torch.int32),
        draft_seq_lens=torch.empty((0,), dtype=torch.int32),
        future_indices=future_indices,
        new_seq_lens=torch.empty((0,), dtype=torch.int32),
    )
    future_map.resolve_future(_FakeModelWorkerBatch(unresolved))

    assert unresolved.verified_id.tolist() == [11, 22]
    assert unresolved.draft_seq_lens.tolist() == [5, 7]
    assert unresolved.new_seq_lens.tolist() == [5, 7]
    assert unresolved.ctx_lens.tolist() == [0, 0]
    assert unresolved.target_hidden.numel() == 0


def test_future_map_dflash_roundtrip_preserves_verify_done():
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.speculative.dflash_info import DFlashDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    future_map = FutureMap(
        max_running_requests=4,
        chunked_prefill_size=0,
        context_len=16,
        device=torch.device("cpu"),
        spec_algo=SpeculativeAlgorithm.DFLASH,
    )
    future_indices = future_map.alloc_future_indices(2)
    sentinel = object()
    stored = DFlashDraftInput(
        verified_id=torch.tensor([11, 22], dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.zeros((2,), dtype=torch.int32),
        draft_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        new_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        verify_done=sentinel,
    )
    future_map.store_to_map_for_new_batch(future_indices, stored)

    unresolved = DFlashDraftInput(
        verified_id=torch.empty((0,), dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.empty((0,), dtype=torch.int32),
        draft_seq_lens=torch.empty((0,), dtype=torch.int32),
        future_indices=future_indices,
        new_seq_lens=torch.empty((0,), dtype=torch.int32),
    )
    future_map.resolve_future(_FakeModelWorkerBatch(unresolved))

    assert unresolved.verify_done is sentinel


def test_future_map_dflash_tree_roundtrip_preserves_verify_done():
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.speculative.dflash_info import DFlashDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    future_map = FutureMap(
        max_running_requests=4,
        chunked_prefill_size=0,
        context_len=16,
        device=torch.device("cpu"),
        spec_algo=SpeculativeAlgorithm.DFLASH_TREE,
    )
    future_indices = future_map.alloc_future_indices(1)
    sentinel = object()
    stored = DFlashDraftInput(
        verified_id=torch.tensor([19], dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.zeros((1,), dtype=torch.int32),
        draft_seq_lens=torch.tensor([7], dtype=torch.int32),
        new_seq_lens=torch.tensor([7], dtype=torch.int32),
        verify_done=sentinel,
    )
    future_map.store_to_map_for_new_batch(future_indices, stored)

    unresolved = DFlashDraftInput(
        verified_id=torch.empty((0,), dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.empty((0,), dtype=torch.int32),
        draft_seq_lens=torch.empty((0,), dtype=torch.int32),
        future_indices=future_indices,
        new_seq_lens=torch.empty((0,), dtype=torch.int32),
    )
    future_map.resolve_future(_FakeModelWorkerBatch(unresolved))

    assert unresolved.verify_done is sentinel
    assert unresolved.verified_id.tolist() == [19]
    assert unresolved.new_seq_lens.tolist() == [7]


def test_future_map_dflash_rejects_pre_append_state():
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.speculative.dflash_info import DFlashDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    future_map = FutureMap(
        max_running_requests=4,
        chunked_prefill_size=0,
        context_len=16,
        device=torch.device("cpu"),
        spec_algo=SpeculativeAlgorithm.DFLASH,
    )
    future_indices = future_map.alloc_future_indices(1)
    pre_append = DFlashDraftInput(
        verified_id=torch.tensor([11], dtype=torch.int64),
        target_hidden=torch.randn((1, 4), dtype=torch.float32),
        ctx_lens=torch.tensor([1], dtype=torch.int32),
        draft_seq_lens=torch.tensor([5], dtype=torch.int32),
        new_seq_lens=torch.tensor([6], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="post-append draft state"):
        future_map.store_to_map_for_new_batch(future_indices, pre_append)


def test_dflash_draft_input_prepare_for_decode_uses_explicit_new_seq_lens():
    from sglang.srt.speculative.dflash_info import DFlashDraftInput

    batch = _FakeScheduleBatchForDecode()
    draft = DFlashDraftInput(
        verified_id=torch.tensor([11, 22], dtype=torch.int64),
        target_hidden=torch.empty((0,), dtype=torch.float32),
        ctx_lens=torch.zeros((2,), dtype=torch.int32),
        draft_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        new_seq_lens=torch.tensor([6, 8], dtype=torch.int32),
    )
    draft.prepare_for_decode(batch)
    assert batch.evicted is True
    assert batch.waited is True
    assert batch.input_ids.tolist() == [11, 22]
    assert batch.seq_lens.tolist() == [6, 8]
    assert batch.seq_lens_cpu.tolist() == [6, 8]
    assert batch.seq_lens_sum == 14


def test_dflash_overlap_preprocessed_unfinished_req_is_cached_back_to_tree():
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    class _DummyScheduler(SchedulerOutputProcessorMixin):
        def __init__(self):
            self.enable_overlap = True
            self.tree_cache = _FakeTreeCache()

    scheduler = _DummyScheduler()
    req = _FakeReq(output_ids=[9], origin_input_ids=[1, 2, 3])
    req.req_pool_idx = 7
    req.is_retracted = False
    req.fill_ids = [1, 2, 3]
    req.customized_info = None

    scheduler._handle_dflash_overlap_preprocessed_req(
        req=req,
        i=0,
        logits_output=None,
    )

    assert scheduler.tree_cache.cached == [req]
    assert req.fill_ids == [1, 2, 3, 9]


def test_dflash_overlap_preprocessed_finished_req_rebuilds_fill_ids_before_release():
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    events = []

    class _DummyScheduler(SchedulerOutputProcessorMixin):
        def __init__(self):
            self.enable_overlap = True
            self.tree_cache = _FakeTreeCache()
            self.server_args = SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            )

        def maybe_collect_routed_experts(self, req):
            events.append(("collect", list(req.output_ids)))

        def _release_kv_cache_and_draft(self, req, *, is_insert=True):
            events.append(("release", list(req.fill_ids), bool(is_insert)))
            req.kv_committed_freed = True
            req.kv_overallocated_freed = True

    scheduler = _DummyScheduler()
    req = _FakeReq(output_ids=[4, 5], origin_input_ids=[1, 2, 3], finish_after=2)
    req.req_pool_idx = 11
    req.fill_ids = [1, 2, 3, 4]
    req.check_finished()

    scheduler._handle_dflash_overlap_preprocessed_req(
        req=req,
        i=0,
        logits_output=None,
    )

    assert req.fill_ids == [1, 2, 3, 4, 5]
    assert scheduler.tree_cache.cached == []
    assert events == [
        ("collect", [4, 5]),
        ("release", [1, 2, 3, 4, 5], False),
    ]
    assert req.time_stats.completion_time is not None


def test_dflash_overlap_preprocessed_finished_req_skips_refresh_when_fully_cached():
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    events = []

    class _DummyScheduler(SchedulerOutputProcessorMixin):
        def __init__(self):
            self.enable_overlap = True
            self.tree_cache = _FakeTreeCache()
            self.server_args = SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            )

        def maybe_collect_routed_experts(self, req):
            events.append(("collect", list(req.output_ids)))

        def _release_kv_cache_and_draft(self, req, *, is_insert=True):
            events.append(("release", list(req.fill_ids), bool(is_insert)))
            req.kv_committed_freed = True
            req.kv_overallocated_freed = True

    scheduler = _DummyScheduler()
    req = _FakeReq(output_ids=[4, 5], origin_input_ids=[1, 2, 3], finish_after=2)
    req.req_pool_idx = 11
    req.fill_ids = [1, 2, 3, 4]
    req.prefix_indices = torch.tensor([101, 102, 103, 104, 105], dtype=torch.int64)
    req.cache_protected_len = 5
    req.check_finished()

    scheduler._handle_dflash_overlap_preprocessed_req(
        req=req,
        i=0,
        logits_output=None,
    )

    assert req.fill_ids == [1, 2, 3, 4, 5]
    assert scheduler.tree_cache.cached == []
    assert events == [
        ("collect", [4, 5]),
        ("release", [1, 2, 3, 4, 5], False),
    ]
    assert req.time_stats.completion_time is not None


def test_dflash_overlap_preprocessed_finished_req_syncs_req_to_token_prefix_after_refresh():
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    class _TreeCacheWithPrefixRewrite(_FakeTreeCache):
        def cache_unfinished_req(self, req):
            super().cache_unfinished_req(req)
            req.prefix_indices = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)
            req.cache_protected_len = 5

    class _DummyScheduler(SchedulerOutputProcessorMixin):
        def __init__(self):
            self.enable_overlap = True
            self.tree_cache = _TreeCacheWithPrefixRewrite()
            self.req_to_token_pool = _FakeReqToTokenPool(
                torch.tensor([[10, 11, 12, 97, 98]], dtype=torch.int64)
            )
            self.server_args = SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            )

        def maybe_collect_routed_experts(self, req):
            return None

        def _release_kv_cache_and_draft(self, req, *, is_insert=True):
            req.kv_committed_freed = True
            req.kv_overallocated_freed = True

    scheduler = _DummyScheduler()
    req = _FakeReq(output_ids=[4, 5], origin_input_ids=[1, 2, 3], finish_after=2)
    req.req_pool_idx = 0
    req.fill_ids = [1, 2, 3, 4]
    req.prefix_indices = torch.tensor([10, 11, 12, 97], dtype=torch.int64)
    req.cache_protected_len = 4
    req.kv_committed_len = 5
    req.kv_allocated_len = 5
    req.check_finished()

    scheduler._handle_dflash_overlap_preprocessed_req(
        req=req,
        i=0,
        logits_output=None,
    )

    assert scheduler.tree_cache.cached == []
    assert req.prefix_indices.tolist() == [10, 11, 12, 97, 98]
    assert (
        scheduler.req_to_token_pool.req_to_token[0, :5].tolist()
        == [10, 11, 12, 97, 98]
    )


def test_dflash_overlap_preprocessed_finished_req_restores_verify_allocation_before_release():
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    events = []

    class _DummyScheduler(SchedulerOutputProcessorMixin):
        def __init__(self):
            self.enable_overlap = True
            self.tree_cache = _FakeTreeCache()
            self.server_args = SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            )

        def maybe_collect_routed_experts(self, req):
            events.append(("collect", list(req.output_ids)))

        def _release_kv_cache_and_draft(self, req, *, is_insert=True):
            events.append(
                ("release", int(req.kv_committed_len), int(req.kv_allocated_len))
            )
            req.kv_committed_freed = True
            req.kv_overallocated_freed = True

    scheduler = _DummyScheduler()
    req = _FakeReq(output_ids=[4, 5], origin_input_ids=[1, 2, 3], finish_after=2)
    req.req_pool_idx = 11
    req.fill_ids = list(range(1, 15))
    req.prefix_indices = torch.tensor([101, 102, 103, 104, 105], dtype=torch.int64)
    req.cache_protected_len = 5
    req.kv_committed_len = 13
    req.kv_allocated_len = 13
    req.spec_accept_length_step_last = 1
    req.spec_dflash_effective_draft_token_num_last = 8
    req.check_finished()

    scheduler._handle_dflash_overlap_preprocessed_req(
        req=req,
        i=0,
        logits_output=None,
    )

    assert events == [
        ("collect", [4, 5]),
        ("release", 13, 13),
    ]


def test_dflash_overlap_preprocessed_finished_req_keeps_cache_protected_len_for_release():
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    observed = []

    class _DummyScheduler(SchedulerOutputProcessorMixin):
        def __init__(self):
            self.enable_overlap = True
            self.tree_cache = _FakeTreeCache()
            self.server_args = SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            )

        def maybe_collect_routed_experts(self, req):
            observed.append(("collect", int(req.cache_protected_len)))

        def _release_kv_cache_and_draft(self, req, *, is_insert=True):
            observed.append(("release", int(req.cache_protected_len)))
            req.kv_committed_freed = True
            req.kv_overallocated_freed = True

    scheduler = _DummyScheduler()
    req = _FakeReq(output_ids=[4, 5], origin_input_ids=[1, 2, 3], finish_after=2)
    req.req_pool_idx = 11
    req.fill_ids = [1, 2, 3, 4]
    req.prefix_indices = torch.tensor([101, 102, 103, 104, 105], dtype=torch.int64)
    req.cache_protected_len = 5
    req.kv_committed_len = 13
    req.kv_allocated_len = 13
    req.spec_accept_length_step_last = 1
    req.spec_dflash_effective_draft_token_num_last = 8
    req.check_finished()

    scheduler._handle_dflash_overlap_preprocessed_req(
        req=req,
        i=0,
        logits_output=None,
    )

    assert observed == [
        ("collect", 5),
        ("release", 5),
    ]
    assert req.cache_protected_len == 5


def test_prepare_spec_v2_overlap_state_keeps_finished_reqs_until_output_processing():
    from sglang.srt.managers.overlap_utils import FutureIndices
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.speculative.dflash_info import DFlashDraftInput

    class _FakeSpecAlg:
        def is_dflash_family(self):
            return True

    class _FakeBatch:
        def __init__(self, reqs, spec_info):
            self.reqs = reqs
            self.spec_info = spec_info
            self.spec_algorithm = _FakeSpecAlg()
            self.seq_lens = None
            self.seq_lens_cpu = None
            self.seq_lens_sum = 0
            self.filtered_keep_indices = None

        def filter_batch(self, keep_indices):
            self.filtered_keep_indices = list(keep_indices)
            keep = torch.tensor(keep_indices, dtype=torch.int64)
            self.reqs = [self.reqs[i] for i in keep_indices]
            self.spec_info.filter_batch(keep)
            self.seq_lens = self.seq_lens[keep]
            self.seq_lens_cpu = self.seq_lens_cpu[keep]
            self.seq_lens_sum = int(self.seq_lens.sum().item())

    req_finished = _FakeReq(output_ids=[10], finish_after=1)
    req_finished.check_finished()
    req_finished.is_retracted = False

    req_running = _FakeReq(output_ids=[20])
    req_running.is_retracted = False

    next_draft_input = DFlashDraftInput(
        verified_id=torch.tensor([101, 202], dtype=torch.int64),
        target_hidden=torch.empty((0, 4), dtype=torch.float32),
        ctx_lens=torch.zeros((2,), dtype=torch.int32),
        draft_seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        new_seq_lens=torch.tensor([6, 8], dtype=torch.int32),
    )
    batch = _FakeBatch([req_finished, req_running], next_draft_input)
    next_out_cache_loc = torch.tensor([301, 302, 401, 402], dtype=torch.int64)
    result = SimpleNamespace(
        next_draft_input=next_draft_input,
        next_out_cache_loc=next_out_cache_loc,
    )
    future_indices = FutureIndices(indices=torch.tensor([11, 22], dtype=torch.int64))

    Scheduler._prepare_spec_v2_overlap_state(
        SimpleNamespace(),
        batch,
        result,
        future_indices,
    )

    assert batch.filtered_keep_indices is None
    assert len(batch.reqs) == 2
    assert batch.reqs[0] is req_finished
    assert batch.reqs[1] is req_running
    assert batch.spec_info.future_indices.indices.tolist() == [11, 22]
    assert batch.spec_info.verified_id.tolist() == [101, 202]
    assert batch.seq_lens.tolist() == [6, 8]
    assert batch.seq_lens_cpu.tolist() == [6, 8]
    assert batch.seq_lens_sum == 14
    assert batch.out_cache_loc.tolist() == [301, 302, 401, 402]


def test_dflash_verify_input_create_idle_input():
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    idle = DFlashVerifyInput.create_idle_input(
        device=torch.device("cpu"),
        draft_token_num=8,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )
    assert idle.draft_token.numel() == 0
    assert idle.positions.numel() == 0
    assert idle.draft_token_num == 8
    assert idle.num_tokens_per_batch == 8


def test_build_and_apply_dflash_indexed_cache_plan():
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_indexed_cache_plan,
        build_dflash_indexed_cache_plan,
    )

    batch = _FakeBatch()
    batch.out_cache_loc = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64)
    plan = build_dflash_indexed_cache_plan(
        out_cache_loc=batch.out_cache_loc,
        accepted_indices=torch.tensor([1, 3], dtype=torch.int64),
    )
    assert plan.compact_out_cache_loc.tolist() == [101, 103]
    assert plan.evicted_slots.tolist() == [100, 102, 104]
    apply_dflash_indexed_cache_plan(batch=batch, cache_plan=plan)
    assert len(batch.token_to_kv_pool_allocator.freed) == 1
    assert batch.token_to_kv_pool_allocator.freed[0].tolist() == [100, 102, 104]
    assert batch.out_cache_loc.tolist() == [101, 103]


def test_build_and_apply_dflash_indexed_cache_plan_page_size_2():
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_indexed_cache_plan,
        build_dflash_indexed_cache_plan,
    )

    batch = _FakeBatch()
    batch.out_cache_loc = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64)
    plan = build_dflash_indexed_cache_plan(
        out_cache_loc=batch.out_cache_loc,
        accepted_indices=torch.tensor([1, 3], dtype=torch.int64),
        page_size=2,
    )
    assert plan.compact_out_cache_loc.tolist() == [101, 103]
    assert plan.evicted_slots.tolist() == [100, 102, 104]
    assert plan.evicted_pages.tolist() == [52]
    apply_dflash_indexed_cache_plan(batch=batch, cache_plan=plan, page_size=2)
    assert len(batch.token_to_kv_pool_allocator.freed_pages) == 1
    assert batch.token_to_kv_pool_allocator.freed_pages[0].tolist() == [52]
    assert batch.out_cache_loc.tolist() == [101, 103]


def test_build_and_apply_dflash_indexed_cache_plan_borrowed_out_cache_loc():
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_indexed_cache_plan,
        build_dflash_indexed_cache_plan,
    )

    batch = _FakeBatch()
    batch.out_cache_loc = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64)
    plan = build_dflash_indexed_cache_plan(
        out_cache_loc=batch.out_cache_loc,
        accepted_indices=torch.tensor([1, 3], dtype=torch.int64),
        borrowed_out_cache_loc=True,
    )
    assert plan.compact_out_cache_loc.tolist() == [101, 103]
    assert plan.evicted_slots.numel() == 0
    assert plan.evicted_pages is None
    apply_dflash_indexed_cache_plan(batch=batch, cache_plan=plan)
    assert batch.token_to_kv_pool_allocator.freed == []
    assert batch.token_to_kv_pool_allocator.freed_pages == []
    assert batch.out_cache_loc.tolist() == [101, 103]


def test_build_dflash_indexed_cache_plan_rejects_unsorted_and_duplicate_indices():
    from sglang.srt.speculative.dflash_utils import build_dflash_indexed_cache_plan

    out_cache_loc = torch.tensor([100, 101, 102, 103], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="sorted"):
        build_dflash_indexed_cache_plan(
            out_cache_loc=out_cache_loc,
            accepted_indices=torch.tensor([2, 0], dtype=torch.int64),
        )

    with pytest.raises(RuntimeError, match="duplicate"):
        build_dflash_indexed_cache_plan(
            out_cache_loc=out_cache_loc,
            accepted_indices=torch.tensor([1, 1], dtype=torch.int64),
        )


def test_build_dflash_indexed_cache_plan_rejects_duplicate_or_nonpositive_slots():
    from sglang.srt.speculative.dflash_utils import build_dflash_indexed_cache_plan

    with pytest.raises(RuntimeError, match="duplicate KV slots"):
        build_dflash_indexed_cache_plan(
            out_cache_loc=torch.tensor([100, 101, 101, 103], dtype=torch.int64),
            accepted_indices=torch.tensor([0, 1], dtype=torch.int64),
        )

    with pytest.raises(RuntimeError, match="non-positive KV slots"):
        build_dflash_indexed_cache_plan(
            out_cache_loc=torch.tensor([100, 0, 102, 103], dtype=torch.int64),
            accepted_indices=torch.tensor([0, 2], dtype=torch.int64),
        )


def test_gather_dflash_committed_hidden_uses_keep_mask():
    from sglang.srt.speculative.dflash_utils import gather_dflash_committed_hidden

    hidden = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ]
    )
    keep_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )
    gathered = gather_dflash_committed_hidden(
        hidden_states=hidden,
        keep_mask=keep_mask,
        draft_token_num=3,
        accepted_indices=torch.tensor([0, 1, 3], dtype=torch.int64),
    )
    assert gathered.tolist() == [[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]


def test_gather_dflash_committed_hidden_uses_flat_indices_when_provided():
    from sglang.srt.speculative.dflash_utils import gather_dflash_committed_hidden

    hidden = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    gathered = gather_dflash_committed_hidden(
        hidden_states=hidden,
        keep_mask=None,
        draft_token_num=None,
        accepted_indices=torch.tensor([2, 0], dtype=torch.int64),
    )
    assert gathered.tolist() == [[3.0, 30.0], [1.0, 10.0]]


def test_update_and_resolve_dflash_verify_append_path_stats():
    from sglang.srt.speculative.dflash_utils import (
        resolve_dflash_verify_append_path,
        update_dflash_verify_append_path_stats,
    )

    reqs = [SimpleNamespace(), SimpleNamespace()]
    append_path = resolve_dflash_verify_append_path(
        appended_from_verify=True,
        fused_helper_active=False,
    )
    assert append_path == "shared_sequential"
    update_dflash_verify_append_path_stats(reqs=reqs, append_path=append_path)
    for req in reqs:
        assert req.spec_dflash_verify_append_path_last == "shared_sequential"
        assert req.spec_dflash_verify_append_path_fused_ct == 0
        assert req.spec_dflash_verify_append_path_direct_ct == 1
        assert req.spec_dflash_verify_append_path_staged_ct == 0


def test_update_dflash_req_verify_bookkeeping_tracks_tree_style_stats():
    from sglang.srt.speculative.dflash_utils import (
        update_dflash_req_verify_bookkeeping,
    )

    reqs = [
        SimpleNamespace(spec_verify_ct=11),
        SimpleNamespace(spec_verify_ct=7),
    ]

    update_dflash_req_verify_bookkeeping(
        reqs=reqs,
        accept_length_per_req_cpu=[3, 1],
        verify_mode="tree_target_only",
        append_path="shared_fused",
        default_max_steps=7,
        default_effective_draft_token_num=8,
        default_effective_step_count=7,
    )

    assert reqs[0].spec_dflash_verify_mode_last == "tree_target_only"
    assert reqs[0].spec_accept_length_step_last == 3
    assert reqs[0].spec_accept_length_step_min == 3
    assert reqs[0].spec_accept_length_step_max == 3
    assert reqs[0].spec_dflash_max_steps_last == 7
    assert reqs[0].spec_dflash_effective_draft_token_num_last == 8
    assert reqs[0].spec_dflash_effective_step_count_last == 7
    assert reqs[0].spec_dflash_total_draft_token_num == 7
    assert reqs[0].spec_dflash_verify_append_path_last == "shared_fused"
    assert reqs[0].spec_dflash_verify_append_path_fused_ct == 1
    assert reqs[0].dflash_difficulty_state.accept_len_last == 3.0
    assert reqs[0].dflash_difficulty_state.verify_ct_last == 11

    assert reqs[1].spec_accept_length_step_last == 1
    assert reqs[1].spec_dflash_verify_append_path_direct_ct == 1
    assert reqs[1].spec_dflash_verify_append_path_staged_ct == 0
    assert reqs[1].dflash_difficulty_state.accept_len_last == 1.0
    assert reqs[1].dflash_difficulty_state.verify_ct_last == 7


def test_project_target_hidden_selected_matches_manual_index_select():
    from sglang.srt.models.dflash import DFlashDraftModel

    class _Dummy:
        pass

    dummy = _Dummy()
    dummy.fc = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        dummy.fc.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )
        )
    dummy.hidden_norm = torch.nn.Identity()

    hidden = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    accepted_indices = torch.tensor([2, 0], dtype=torch.int64)

    projected = DFlashDraftModel.project_target_hidden_selected(
        dummy,
        hidden,
        accepted_indices,
    )
    expected = dummy.fc(hidden.index_select(0, accepted_indices))
    assert torch.equal(projected, expected)


def test_dflash_set_embed_disables_input_embeds_requirement():
    from sglang.srt.models.dflash import DFlashDraftModel

    class _Dummy:
        pass

    dummy = _Dummy()
    dummy.embed_tokens = None
    dummy.requires_input_embeds = True
    embed = torch.nn.Embedding(8, 4)

    DFlashDraftModel.set_embed(dummy, embed)

    assert dummy.embed_tokens is embed
    assert dummy.requires_input_embeds is False


def test_dflash_forward_uses_shared_embed_when_input_embeds_missing():
    from sglang.srt.models.dflash import DFlashDraftModel

    class _Dummy:
        pass

    dummy = _Dummy()
    dummy.embed_tokens = torch.nn.Embedding(8, 4)
    with torch.no_grad():
        dummy.embed_tokens.weight.copy_(
            torch.arange(32, dtype=torch.float32).view(8, 4)
        )
    dummy.layers = []
    dummy.norm = torch.nn.Identity()

    input_ids = torch.tensor([1, 3, 5], dtype=torch.int64)
    out = DFlashDraftModel.forward(
        dummy,
        input_ids=input_ids,
        positions=torch.arange(3, dtype=torch.int64),
        forward_batch=SimpleNamespace(),
        input_embeds=None,
    )

    expected = dummy.embed_tokens(input_ids)
    assert torch.equal(out, expected)


def test_compute_dflash_sampling_accept_len_and_bonus_honors_max_steps_and_returns_prefix(
    monkeypatch,
):
    from sglang.srt.speculative import dflash_utils as du

    def fake_tree_kernel(
        *,
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    ):
        predicts[:5] = torch.tensor([11, 12, 13, -99, 14], dtype=torch.int32)
        accept_index.fill_(-1)
        accept_index[0, :4] = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
        accept_token_num[0] = 3

    monkeypatch.setattr(du, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True)
    monkeypatch.setattr(du, "tree_speculative_sampling_target_only", fake_tree_kernel)

    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([1.0], dtype=torch.float32),
        top_ks=torch.tensor([1], dtype=torch.int32),
        top_ps=torch.tensor([1.0], dtype=torch.float32),
        min_ps=torch.tensor([0.0], dtype=torch.float32),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )
    candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
    logits = torch.zeros((4, 8), dtype=torch.float32)

    accept_len, bonus, proposed = du.compute_dflash_sampling_accept_len_and_bonus(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=sampling_info,
        threshold_single=1.0,
        threshold_acc=1.0,
        return_proposed_tokens=True,
    )
    assert accept_len.tolist() == [3]
    assert bonus.tolist() == [14]
    assert proposed.tolist() == [[11, 12, 13, 14]]

    accept_len, bonus, proposed = du.compute_dflash_sampling_accept_len_and_bonus(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=sampling_info,
        max_steps_per_req=torch.tensor([1], dtype=torch.int32),
        threshold_single=1.0,
        threshold_acc=1.0,
        return_proposed_tokens=True,
    )
    assert accept_len.tolist() == [1]
    assert bonus.tolist() == [12]
    assert proposed.tolist() == [[11, 12, 0, 0]]


def test_compute_dflash_sampling_accept_len_and_bonus_clamps_kernel_accept_count(
    monkeypatch,
):
    from sglang.srt.speculative import dflash_utils as du

    def fake_tree_kernel(
        *,
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    ):
        predicts[:5] = torch.tensor([11, 12, 13, 14, 16], dtype=torch.int32)
        accept_index.fill_(-1)
        accept_index[0, :4] = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
        # Simulate an out-of-range accepted count from the kernel-side path.
        accept_token_num[0] = 99

    monkeypatch.setattr(du, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True)
    monkeypatch.setattr(du, "tree_speculative_sampling_target_only", fake_tree_kernel)

    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([1.0], dtype=torch.float32),
        top_ks=torch.tensor([1], dtype=torch.int32),
        top_ps=torch.tensor([1.0], dtype=torch.float32),
        min_ps=torch.tensor([0.0], dtype=torch.float32),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )
    candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
    logits = torch.zeros((4, 8), dtype=torch.float32)

    accept_len, bonus, proposed = du.compute_dflash_sampling_accept_len_and_bonus(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=sampling_info,
        threshold_single=1.0,
        threshold_acc=1.0,
        return_proposed_tokens=True,
    )
    assert accept_len.tolist() == [3]
    assert bonus.tolist() == [16]
    assert proposed.tolist() == [[11, 12, 13, 16]]


def test_compute_dflash_sampling_accept_len_and_bonus_ignores_garbage_trailing_accept_indices(
    monkeypatch,
):
    from sglang.srt.speculative import dflash_utils as du

    def fake_tree_kernel(
        *,
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    ):
        predicts[:5] = torch.tensor([11, 12, 13, 14, 15], dtype=torch.int32)
        accept_index.fill_(999999)
        accept_index[0, :3] = torch.tensor([0, 1, 3], dtype=torch.int32)
        accept_token_num[0] = 2

    monkeypatch.setattr(du, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True)
    monkeypatch.setattr(du, "tree_speculative_sampling_target_only", fake_tree_kernel)

    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([1.0], dtype=torch.float32),
        top_ks=torch.tensor([1], dtype=torch.int32),
        top_ps=torch.tensor([1.0], dtype=torch.float32),
        min_ps=torch.tensor([0.0], dtype=torch.float32),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
    )
    candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
    logits = torch.zeros((4, 8), dtype=torch.float32)

    accept_len, bonus, proposed = du.compute_dflash_sampling_accept_len_and_bonus(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=sampling_info,
        threshold_single=1.0,
        threshold_acc=1.0,
        return_proposed_tokens=True,
    )
    assert accept_len.tolist() == [2]
    assert bonus.tolist() == [14]
    assert proposed.tolist() == [[11, 12, 14, 0]]


def test_compute_dflash_sampling_accept_len_and_bonus_applies_min_p_filter(
    monkeypatch,
):
    from sglang.srt.speculative import dflash_utils as du

    captured = {}

    def fake_tree_kernel(
        *,
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    ):
        captured["target_probs"] = target_probs.detach().clone()
        predicts[:2] = torch.tensor([0, 0], dtype=torch.int32)
        accept_index.fill_(-1)
        accept_index[0, :1] = torch.tensor([0], dtype=torch.int32)
        accept_token_num[0] = 0

    monkeypatch.setattr(du, "_DFLASH_SAMPLING_VERIFY_AVAILABLE", True)
    monkeypatch.setattr(du, "tree_speculative_sampling_target_only", fake_tree_kernel)

    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([1.0], dtype=torch.float32),
        top_ks=torch.tensor([3], dtype=torch.int32),
        top_ps=torch.tensor([1.0], dtype=torch.float32),
        min_ps=torch.tensor([0.6], dtype=torch.float32),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=True,
    )
    candidates = torch.tensor([[10, 11]], dtype=torch.int64)
    logits = torch.tensor(
        [[4.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    du.compute_dflash_sampling_accept_len_and_bonus(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=sampling_info,
        threshold_single=1.0,
        threshold_acc=1.0,
    )
    probs = captured["target_probs"][0, 0]
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert probs[0] > 0
    assert probs[1].item() == 0.0
    assert probs[2].item() == 0.0


def test_commit_dflash_proposed_tokens_to_req_fast_path_truncates_and_updates_stats():
    from sglang.srt.speculative.dflash_utils import commit_dflash_proposed_tokens_to_req

    req = _FakeReq(max_new_tokens=8, stop_token_ids=[13], origin_input_ids=[99])
    outcome = commit_dflash_proposed_tokens_to_req(
        req=req,
        proposed=[11, 12, 13, 14],
    )

    assert req.output_ids == [11, 12, 13]
    assert outcome.commit_len == 3
    assert outcome.accepted_draft_tokens == 2
    assert outcome.new_verified_token == 13
    assert outcome.used_device_defaults is False
    assert req.spec_verify_ct == 1
    assert req.spec_accepted_tokens == 2
    assert req.hist == [2]
    assert req.check_finished_calls == [3]


def test_commit_dflash_proposed_tokens_to_req_marks_device_defaults_when_untruncated():
    from sglang.srt.speculative.dflash_utils import commit_dflash_proposed_tokens_to_req

    req = _FakeReq(max_new_tokens=8, stop_token_ids=[], origin_input_ids=[99])
    outcome = commit_dflash_proposed_tokens_to_req(
        req=req,
        proposed=[11, 12, 14],
    )

    assert req.output_ids == [11, 12, 14]
    assert outcome.commit_len == 3
    assert outcome.accepted_draft_tokens == 2
    assert outcome.new_verified_token == 14
    assert outcome.used_device_defaults is True


def test_commit_dflash_proposed_tokens_to_req_grammar_path_and_empty_fallback():
    from sglang.srt.speculative.dflash_utils import commit_dflash_proposed_tokens_to_req

    grammar = _FakeGrammar()
    req = _FakeReq(
        grammar=grammar,
        output_ids=[],
        origin_input_ids=[7],
        finish_after=2,
    )
    outcome = commit_dflash_proposed_tokens_to_req(
        req=req,
        proposed=[21, 22, 23],
        empty_error_prefix="DFLASH_TREE verify",
    )

    assert req.output_ids == [21, 22]
    assert grammar.accepted == [21]
    assert outcome.commit_len == 2
    assert outcome.accepted_draft_tokens == 1
    assert outcome.new_verified_token == 22
    assert outcome.used_device_defaults is False
    assert req.spec_verify_ct == 1
    assert req.spec_accepted_tokens == 1
    assert req.hist == [1]
    assert req.check_finished_calls == [None, None]

    empty_req = _FakeReq(max_new_tokens=0, output_ids=[], origin_input_ids=[55])
    empty_outcome = commit_dflash_proposed_tokens_to_req(
        req=empty_req,
        proposed=[91, 92],
    )
    assert empty_req.output_ids == []
    assert empty_outcome.commit_len == 0
    assert empty_outcome.accepted_draft_tokens == 0
    assert empty_outcome.new_verified_token == 55
    assert empty_outcome.used_device_defaults is False


def test_commit_dflash_target_only_batch_matches_individual_fast_path():
    from sglang.srt.speculative.dflash_utils import (
        commit_dflash_proposed_tokens_to_req,
        commit_dflash_target_only_batch,
        pack_dflash_target_only_commits,
    )

    target_predict = torch.tensor(
        [
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
        ],
        dtype=torch.int64,
    )
    accept_len = torch.tensor([2, 1, 3], dtype=torch.int32)
    packed = pack_dflash_target_only_commits(
        target_predict=target_predict,
        accept_len=accept_len,
    )
    proposed_flat_cpu = packed.proposed_flat.cpu()
    commit_offsets_cpu = packed.commit_offsets.cpu()

    reqs_batch = [
        _FakeReq(max_new_tokens=8, stop_token_ids=[13], origin_input_ids=[99]),
        _FakeReq(max_new_tokens=8, stop_token_ids=[], origin_input_ids=[99]),
        _FakeReq(max_new_tokens=2, stop_token_ids=[], origin_input_ids=[99]),
    ]
    reqs_ref = [
        _FakeReq(max_new_tokens=8, stop_token_ids=[13], origin_input_ids=[99]),
        _FakeReq(max_new_tokens=8, stop_token_ids=[], origin_input_ids=[99]),
        _FakeReq(max_new_tokens=2, stop_token_ids=[], origin_input_ids=[99]),
    ]

    got = commit_dflash_target_only_batch(
        reqs=reqs_batch,
        proposed_flat_cpu=proposed_flat_cpu,
        commit_offsets_cpu=commit_offsets_cpu,
    )

    ref = []
    for i, req in enumerate(reqs_ref):
        start_offset = int(commit_offsets_cpu[i].item())
        end_offset = int(commit_offsets_cpu[i + 1].item())
        ref.append(
            commit_dflash_proposed_tokens_to_req(
                req=req,
                proposed=proposed_flat_cpu[start_offset:end_offset].tolist(),
            )
        )

    assert [r.output_ids for r in reqs_batch] == [r.output_ids for r in reqs_ref]
    assert [r.spec_verify_ct for r in reqs_batch] == [r.spec_verify_ct for r in reqs_ref]
    assert [r.spec_accepted_tokens for r in reqs_batch] == [
        r.spec_accepted_tokens for r in reqs_ref
    ]
    assert [r.commit_len for r in got] == [r.commit_len for r in ref]
    assert [r.new_verified_token for r in got] == [r.new_verified_token for r in ref]
    assert [r.accepted_draft_tokens for r in got] == [
        r.accepted_draft_tokens for r in ref
    ]


def test_commit_dflash_target_only_batch_falls_back_for_grammar_requests():
    from sglang.srt.speculative.dflash_utils import (
        commit_dflash_target_only_batch,
        pack_dflash_target_only_commits,
    )

    grammar = _FakeGrammar()
    target_predict = torch.tensor([[41, 42, 43]], dtype=torch.int64)
    accept_len = torch.tensor([1], dtype=torch.int32)
    packed = pack_dflash_target_only_commits(
        target_predict=target_predict,
        accept_len=accept_len,
    )
    req = _FakeReq(grammar=grammar, finish_after=2, origin_input_ids=[7])
    got = commit_dflash_target_only_batch(
        reqs=[req],
        proposed_flat_cpu=packed.proposed_flat.cpu(),
        commit_offsets_cpu=packed.commit_offsets.cpu(),
    )

    assert req.output_ids == [41, 42]
    assert grammar.accepted == [41]
    assert got[0].commit_len == 2
    assert got[0].accepted_draft_tokens == 1


def test_commit_dflash_target_only_batch_marks_exact_fit_length_finish():
    from sglang.srt.speculative.dflash_utils import (
        commit_dflash_target_only_batch,
        pack_dflash_target_only_commits,
    )

    target_predict = torch.tensor([[41, 42]], dtype=torch.int64)
    accept_len = torch.tensor([1], dtype=torch.int32)
    packed = pack_dflash_target_only_commits(
        target_predict=target_predict,
        accept_len=accept_len,
    )
    req = _FakeReq(max_new_tokens=2, stop_token_ids=[], origin_input_ids=[7])

    got = commit_dflash_target_only_batch(
        reqs=[req],
        proposed_flat_cpu=packed.proposed_flat.cpu(),
        commit_offsets_cpu=packed.commit_offsets.cpu(),
    )

    assert req.output_ids == [41, 42]
    assert req.finished_len == 2
    assert req.finished()
    assert type(req.finished_reason).__name__ == "FINISH_LENGTH"
    assert got[0].commit_len == 2
    assert got[0].used_device_defaults is True


def test_generation_batch_result_copy_to_cpu_skips_missing_tokens_for_dflash_overlap():
    from sglang.srt.managers.utils import GenerationBatchResult

    fake_event = _FakeEvent()
    result = GenerationBatchResult(
        logits_output=SimpleNamespace(
            hidden_states=None,
            next_token_logprobs=None,
            input_token_logprobs=None,
        ),
        next_token_ids=None,
        dflash_overlap_preprocessed=True,
        copy_done=fake_event,
    )

    result.copy_to_cpu(return_logprob=False)

    assert fake_event.recorded is True


def test_compute_adaptive_max_steps_for_req_reacts_to_low_last_accept_early():
    from sglang.srt.speculative.dflash_controller import (
        DFlashReqDifficultyState,
        compute_adaptive_max_steps_for_req,
    )

    req_state = DFlashReqDifficultyState(
        accept_len_last=1.0,
        accept_len_ema=4.5,
        verify_ct_last=2,
    )

    got = compute_adaptive_max_steps_for_req(
        req_state,
        step_count=16,
        verify_ct_ge=8,
        last_verify_ct_ge=2,
        accept_ema_hard_le=2.0,
        accept_ema_medium_le=5.0,
        accept_last_hard_le=1.0,
        accept_last_medium_le=2.0,
        hard_cap_steps=1,
        medium_cap_steps=4,
    )

    assert got == 1


def test_compute_adaptive_max_steps_for_req_uses_last_accept_medium_cap():
    from sglang.srt.speculative.dflash_controller import (
        DFlashReqDifficultyState,
        compute_adaptive_max_steps_for_req,
    )

    req_state = DFlashReqDifficultyState(
        accept_len_last=2.0,
        accept_len_ema=6.0,
        verify_ct_last=3,
    )

    got = compute_adaptive_max_steps_for_req(
        req_state,
        step_count=16,
        verify_ct_ge=8,
        last_verify_ct_ge=2,
        accept_ema_hard_le=2.0,
        accept_ema_medium_le=5.0,
        accept_last_hard_le=1.0,
        accept_last_medium_le=2.0,
        hard_cap_steps=1,
        medium_cap_steps=4,
    )

    assert got == 4


def test_compute_adaptive_max_steps_for_req_does_not_hard_cap_from_last_accept_when_disabled():
    from sglang.srt.speculative.dflash_controller import (
        DFlashReqDifficultyState,
        compute_adaptive_max_steps_for_req,
    )

    req_state = DFlashReqDifficultyState(
        accept_len_last=1.0,
        accept_len_ema=4.5,
        verify_ct_last=2,
    )

    got = compute_adaptive_max_steps_for_req(
        req_state,
        step_count=16,
        verify_ct_ge=8,
        last_verify_ct_ge=2,
        accept_ema_hard_le=2.0,
        accept_ema_medium_le=4.0,
        accept_last_hard_le=-1.0,
        accept_last_medium_le=2.0,
        hard_cap_steps=1,
        medium_cap_steps=4,
    )

    assert got == 4


def test_compute_adaptive_max_steps_for_req_keeps_full_width_without_enough_history():
    from sglang.srt.speculative.dflash_controller import (
        DFlashReqDifficultyState,
        compute_adaptive_max_steps_for_req,
    )

    req_state = DFlashReqDifficultyState(
        accept_len_last=1.0,
        accept_len_ema=1.0,
        verify_ct_last=1,
    )

    got = compute_adaptive_max_steps_for_req(
        req_state,
        step_count=16,
        verify_ct_ge=8,
        last_verify_ct_ge=2,
        accept_ema_hard_le=2.0,
        accept_ema_medium_le=5.0,
        accept_last_hard_le=1.0,
        accept_last_medium_le=2.0,
        hard_cap_steps=1,
        medium_cap_steps=4,
    )

    assert got == 16
