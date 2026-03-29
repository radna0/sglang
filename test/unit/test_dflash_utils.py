import torch
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
    ):
        self.output_ids = list(output_ids or [])
        self.origin_input_ids = list(origin_input_ids or [1])
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
        )
        self._finished = False
        self._finish_after = finish_after
        self.check_finished_calls = []
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0
        self.hist = []

    def check_finished(self, new_accepted_len=None):
        self.check_finished_calls.append(new_accepted_len)
        if self._finish_after is not None and len(self.output_ids) >= int(self._finish_after):
            self._finished = True

    def finished(self):
        return self._finished

    def update_spec_acceptance_histogram(self, accepted_draft_tokens):
        self.hist.append(int(accepted_draft_tokens))


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
            ),
            DFlashTargetOnlyCommitResult(
                commit_len=0,
                new_verified_token=55,
                accepted_draft_tokens=0,
            ),
        ],
        device=torch.device("cpu"),
    )
    assert metadata.commit_lens.tolist() == [3, 0]
    assert metadata.new_verified_id.tolist() == [13, 55]


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
    assert req.spec_verify_ct == 1
    assert req.spec_accepted_tokens == 2
    assert req.hist == [2]
    assert req.check_finished_calls == [3]


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
