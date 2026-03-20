import torch


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

