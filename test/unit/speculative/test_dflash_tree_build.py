import unittest

import torch

from sglang.srt.speculative.sampling_utils import min_p_renorm_prob
from sglang.srt.speculative.eagle_utils import organize_draft_results


def _select_top_k_tokens_ref(
    *,
    step_idx: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    scores: torch.Tensor | None,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference (non-compiled) port of `select_top_k_tokens` for unit tests.

    This keeps the unit test CPU-only and avoids `torch.compile` overhead.
    """
    if step_idx == 0:
        input_ids = topk_index.flatten()
        scores = topk_p
        tree_info = (
            topk_p.unsqueeze(1),  # (b, 1, topk)
            topk_index,  # (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=topk_p.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # (b, topk + 1)
        )
        return input_ids, scores, tree_info

    if scores is None:
        raise ValueError("scores must be set for step_idx > 0")

    # expand_scores: (b, topk, topk)
    expand_scores = scores.unsqueeze(2) * topk_p.reshape(-1, topk, topk)
    topk_cs_p, topk_cs_index = torch.topk(expand_scores.flatten(start_dim=1), topk, dim=-1)
    scores = topk_cs_p

    flat_index = topk_index.reshape(-1, topk**2)
    input_ids = torch.gather(flat_index, index=topk_cs_index, dim=1).flatten()
    tree_info = (
        expand_scores,  # (b, topk, topk)
        flat_index,  # (b, topk * topk)
        topk_cs_index + (topk**2 * (step_idx - 1) + topk),  # (b, topk)
    )
    return input_ids, scores, tree_info


class TestDFlashTreeBuild(unittest.TestCase):
    def test_tree_build_shapes_cpu(self):
        # A tiny, deterministic setup that mirrors the DFLASH_TREE tree-building loop:
        # per-step top-k -> beam-style expansion -> organize_draft_results().
        bs = 2
        step_count = 3
        topk = 2
        num_verify_tokens = 6  # includes root; so selects num_verify_tokens-1 draft nodes.

        # topk_p/topk_index per step: (bs, step_count, topk).
        topk_p = torch.tensor(
            [
                [[0.9, 0.1], [0.6, 0.4], [0.7, 0.3]],
                [[0.8, 0.2], [0.55, 0.45], [0.65, 0.35]],
            ],
            dtype=torch.float32,
        )
        topk_index = torch.tensor(
            [
                [[11, 12], [21, 22], [31, 32]],
                [[13, 14], [23, 24], [33, 34]],
            ],
            dtype=torch.long,
        )

        score_list: list[torch.Tensor] = []
        token_list: list[torch.Tensor] = []
        parents_list: list[torch.Tensor] = []
        scores: torch.Tensor | None = None

        for i in range(step_count):
            step_p = topk_p[:, i, :]
            step_ids = topk_index[:, i, :]
            if i == 0:
                _, scores, tree_info = _select_top_k_tokens_ref(
                    step_idx=i,
                    topk_p=step_p,
                    topk_index=step_ids,
                    scores=scores,
                    topk=topk,
                )
            else:
                step_p_rep = step_p.repeat_interleave(topk, dim=0)
                step_ids_rep = step_ids.repeat_interleave(topk, dim=0)
                _, scores, tree_info = _select_top_k_tokens_ref(
                    step_idx=i,
                    topk_p=step_p_rep,
                    topk_index=step_ids_rep,
                    scores=scores,
                    topk=topk,
                )

            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, num_verify_tokens
        )

        self.assertEqual(tuple(parent_list.shape), (bs, topk * (step_count - 1) + 1))
        self.assertEqual(tuple(top_scores_index.shape), (bs, num_verify_tokens - 1))
        self.assertEqual(tuple(draft_tokens.shape), (bs, num_verify_tokens - 1))

        # Selected tokens must come from the provided per-step candidate ids.
        all_candidates = set(int(x) for x in topk_index.flatten().tolist())
        selected = set(int(x) for x in draft_tokens.flatten().tolist())
        self.assertTrue(
            selected.issubset(all_candidates),
            f"draft_tokens contains ids outside candidate set: {selected - all_candidates}",
        )

    def test_min_p_renorm_prob_cpu(self):
        probs = torch.tensor(
            [
                [0.7, 0.2, 0.1, 0.0],
                [0.4, 0.3, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        min_ps = torch.tensor([0.5, 0.25], dtype=torch.float32)

        out = min_p_renorm_prob(probs.clone(), min_ps)

        # Row 0: threshold = 0.7 * 0.5 = 0.35 -> keep only 0.7 -> renorm to 1.0
        self.assertTrue(torch.allclose(out[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))

        # Row 1: threshold = 0.4 * 0.25 = 0.1 -> keep all (>= 0.1) -> unchanged
        self.assertTrue(torch.allclose(out[1], probs[1]))

        # Each row should still sum to 1.
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
