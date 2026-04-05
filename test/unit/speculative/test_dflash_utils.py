import unittest

import torch

from sglang.srt.speculative.dflash_utils import (
    DEFAULT_DFLASH_MASK_TOKEN,
    build_target_layer_ids,
    compute_dflash_accept_len_and_bonus,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
    scale_kv_cell_size_per_token_for_dflash,
)


class _BackendStub:
    pass


class _WrappedBackendStub:
    def __init__(self, backend):
        self.full_attn_backend = backend


class TestScaleKVCellSizePerTokenForDFlash(unittest.TestCase):
    def test_scales_target_and_draft_layers(self):
        scaled = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=100,
            target_num_layers=40,
            draft_num_layers=10,
        )
        self.assertEqual(scaled, 125)

    def test_uses_explicit_draft_cell_size_when_provided(self):
        scaled = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=100,
            target_num_layers=40,
            draft_num_layers=10,
            draft_cell_size_per_token=24,
        )
        self.assertEqual(scaled, 124)

    def test_rejects_non_positive_target_cell_size(self):
        with self.assertRaisesRegex(ValueError, "target_cell_size_per_token"):
            scale_kv_cell_size_per_token_for_dflash(
                target_cell_size_per_token=0,
                target_num_layers=40,
                draft_num_layers=10,
            )


class TestBuildTargetLayerIds(unittest.TestCase):
    def test_single_draft_layer_uses_midpoint(self):
        self.assertEqual(build_target_layer_ids(40, 1), [20])

    def test_multiple_draft_layers_are_spread_across_target(self):
        self.assertEqual(build_target_layer_ids(40, 4), [1, 13, 25, 37])

    def test_rejects_too_few_target_layers(self):
        with self.assertRaisesRegex(ValueError, "num_target_layers >= 4"):
            build_target_layer_ids(3, 2)


class TestParseDFlashDraftConfig(unittest.TestCase):
    def test_parses_top_level_and_dflash_config_fields(self):
        cfg = parse_dflash_draft_config(
            draft_hf_config={
                "text_config": {"num_hidden_layers": 12},
                "block_size": 16,
                "dflash_config": {
                    "num_target_layers": 40,
                    "target_layer_ids": [1, 8, 16, 24, 32, 39],
                    "mask_token": "<|CUSTOM_MASK|>",
                    "mask_token_id": 320001,
                },
            }
        )
        self.assertEqual(cfg.num_hidden_layers, 12)
        self.assertEqual(cfg.num_target_layers, 40)
        self.assertEqual(cfg.block_size, 16)
        self.assertEqual(cfg.target_layer_ids, [1, 8, 16, 24, 32, 39])
        self.assertEqual(cfg.mask_token, "<|CUSTOM_MASK|>")
        self.assertEqual(cfg.mask_token_id, 320001)

    def test_uses_default_mask_token(self):
        cfg = parse_dflash_draft_config(
            draft_hf_config={
                "text_config": {"num_hidden_layers": 8},
                "dflash_config": {},
            }
        )
        self.assertEqual(cfg.mask_token, DEFAULT_DFLASH_MASK_TOKEN)
        self.assertIsNone(cfg.mask_token_id)

    def test_rejects_invalid_mask_token_id(self):
        with self.assertRaisesRegex(ValueError, "mask_token_id must be an integer"):
            parse_dflash_draft_config(
                draft_hf_config={
                    "text_config": {"num_hidden_layers": 8},
                    "dflash_config": {"mask_token_id": "not-an-int"},
                }
            )

    def test_resolve_target_layer_ids_uses_explicit_list(self):
        cfg = parse_dflash_draft_config(
            draft_hf_config={
                "text_config": {"num_hidden_layers": 6},
                "dflash_config": {"target_layer_ids": [2, 7, 11]},
            }
        )
        self.assertEqual(
            cfg.resolve_target_layer_ids(target_num_layers=16, draft_num_layers=6),
            [2, 7, 11],
        )

    def test_resolve_target_layer_ids_falls_back_to_derived_layout(self):
        cfg = parse_dflash_draft_config(
            draft_hf_config={
                "text_config": {"num_hidden_layers": 4},
                "dflash_config": {},
            }
        )
        self.assertEqual(
            cfg.resolve_target_layer_ids(target_num_layers=40),
            [1, 13, 25, 37],
        )


class TestResolveDFlashVerifyMaskPolicy(unittest.TestCase):
    def test_unknown_backend_keeps_custom_mask(self):
        backend_name, build_custom_mask = resolve_dflash_verify_mask_policy(
            _BackendStub()
        )
        self.assertEqual(backend_name, "_BackendStub")
        self.assertTrue(build_custom_mask)

    def test_wrapped_backend_uses_inner_backend_name(self):
        wrapped = _WrappedBackendStub(_BackendStub())
        backend_name, build_custom_mask = resolve_dflash_verify_mask_policy(wrapped)
        self.assertEqual(backend_name, "_BackendStub")
        self.assertTrue(build_custom_mask)


class TestComputeDFlashAcceptLenAndBonus(unittest.TestCase):
    def test_accepts_full_chain_when_predictions_match(self):
        candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        target_predict = torch.tensor([[11, 12, 13, 99]], dtype=torch.long)
        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        self.assertEqual(accept_len.tolist(), [3])
        self.assertEqual(bonus.tolist(), [99])

    def test_stops_at_first_mismatch_and_returns_bonus(self):
        candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        target_predict = torch.tensor([[11, 77, 13, 99]], dtype=torch.long)
        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        self.assertEqual(accept_len.tolist(), [1])
        self.assertEqual(bonus.tolist(), [77])

    def test_supports_batched_inputs(self):
        candidates = torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
            ],
            dtype=torch.long,
        )
        target_predict = torch.tensor(
            [
                [11, 12, 13, 90],
                [99, 21, 22, 91],
            ],
            dtype=torch.long,
        )
        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        self.assertEqual(accept_len.tolist(), [3, 0])
        self.assertEqual(bonus.tolist(), [90, 99])


if __name__ == "__main__":
    unittest.main()
