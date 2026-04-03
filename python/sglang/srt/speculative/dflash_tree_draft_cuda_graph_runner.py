from __future__ import annotations

import bisect
import os
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    build_dflash_tree_candidates_from_per_step_topk,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.dflash_tree_worker import DFlashTreeWorker


class DFlashTreeDraftCudaGraphRunner:
    """Dedicated CUDA graph replay for pure DFLASH_TREE draft+topk.

    This keeps the hot non-overlap tree path off the generic model-runner decode
    wrapper and captures the actual block forward + per-step vocab top-k extraction
    together. It is intentionally limited to the current single-model draft lane.
    """

    def __init__(self, tree_worker: "DFlashTreeWorker"):
        self.tree_worker = tree_worker
        self.model_runner = model_runner = tree_worker.draft_model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        self.block_size = int(tree_worker.block_size)
        self.spec_steps = int(tree_worker.spec_steps)
        self.topk = int(tree_worker.topk)
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.num_tokens_per_bs = self.block_size
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.candidate_count = self.topk + max(0, self.spec_steps - 1) * (
            self.topk * self.topk
        )
        self.parent_count = (
            0
            if self.spec_steps == 1
            else (self.topk + 1) + max(0, self.spec_steps - 2) * self.topk
        )
        self.verify_count = max(0, int(self.tree_worker.num_verify_tokens) - 1)
        self.capture_builder_in_graph = (
            (os.environ.get("SGLANG_DFLASH_TREE_CAPTURE_BUILDER_IN_GRAPH") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        self.capture_topk_in_graph = (
            (os.environ.get("SGLANG_DFLASH_TREE_CAPTURE_TOPK_IN_GRAPH") or "")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        if not os.environ.get("SGLANG_DFLASH_TREE_CAPTURE_TOPK_IN_GRAPH"):
            # The explicit-input-embeds lane is the production-aligned raw-target path.
            # Keep graph replay for the draft forward, but run LM-head top-k eagerly after
            # replay until the captured top-k path is proven stable there.
            self.capture_topk_in_graph = not bool(
                getattr(tree_worker, "_force_explicit_input_embeds", False)
            )

        self.model_runner.attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        hidden_size = int(self.model_runner.model_config.hidden_size)
        self.requires_input_embeds = bool(
            getattr(self.model_runner.model, "requires_input_embeds", False)
        )
        with torch.device(model_runner.device):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.topk_p = torch.zeros(
                (self.max_bs, self.spec_steps, self.topk), dtype=torch.float32
            )
            self.topk_index = torch.zeros(
                (self.max_bs, self.spec_steps, self.topk), dtype=torch.int64
            )
            self.candidate_scores_buf = torch.zeros(
                (self.max_bs, self.candidate_count), dtype=torch.float32
            )
            self.candidate_tokens_buf = torch.zeros(
                (self.max_bs, self.candidate_count), dtype=torch.int64
            )
            self.parent_list = torch.zeros(
                (self.max_bs, self.parent_count), dtype=torch.int64
            )
            self.top_scores_index = torch.zeros(
                (self.max_bs, self.verify_count), dtype=torch.int64
            )
            self.draft_tokens = torch.zeros(
                (self.max_bs, self.verify_count), dtype=torch.int64
            )
            self.hidden_states = torch.zeros(
                (self.max_num_token, hidden_size), dtype=self.model_runner.dtype
            )
            self.input_embeds = (
                torch.zeros(
                    (self.max_num_token, hidden_size), dtype=self.model_runner.dtype
                )
                if self.requires_input_embeds
                else None
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    self.global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    self.global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                self.global_num_tokens_gpu = None
                self.global_num_tokens_for_logprob_gpu = None

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _maybe_debug_sync(self, label: str) -> None:
        enabled = (
            (os.environ.get("SGLANG_DFLASH_TREE_DEBUG_FASTPATH_SYNC") or "")
            .strip()
            .lower()
            not in ("", "0", "false", "off", "no")
        )
        if not enabled or not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize(self.model_runner.device)
        except Exception as e:
            raise RuntimeError(
                "DFLASH_TREE fastpath failed after "
                f"{label}: block_size={self.block_size} spec_steps={self.spec_steps} "
                f"topk={self.topk} capture_topk_in_graph={self.capture_topk_in_graph} "
                f"capture_builder_in_graph={self.capture_builder_in_graph}"
            ) from e

    def _cache_loc_dtype(self):
        return torch.int64

    def can_run(self, forward_batch: ForwardBatch):
        cuda_graph_bs = forward_batch.batch_size
        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )
        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph
        return is_bs_supported

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self, _forward_batch: ForwardBatch):
        self.graphs[self.bs].replay()

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, bs: int, forward: Callable, stream_idx: int = 0
    ):
        graph = self._create_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        seq_lens_cpu = self.seq_lens_cpu[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        input_embeds = (
            self.input_embeds[:num_tokens] if self.input_embeds is not None else None
        )
        topk_p = self.topk_p[:bs]
        topk_index = self.topk_index[:bs]
        candidate_scores_buf = self.candidate_scores_buf[:bs]
        candidate_tokens_buf = self.candidate_tokens_buf[:bs]
        parent_list = self.parent_list[:bs]
        top_scores_index = self.top_scores_index[:bs]
        draft_tokens_buf = self.draft_tokens[:bs]

        if self.require_gathered_buffer:
            fill = [num_tokens] * self.dp_size if self.require_mlp_tp_gather else [num_tokens]
            self.global_num_tokens_gpu.copy_(
                torch.tensor(fill, dtype=torch.int32, device=input_ids.device)
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(fill, dtype=torch.int32, device=input_ids.device)
            )
            global_dp_buffer_len = (
                num_tokens * self.dp_size if self.require_mlp_tp_gather else num_tokens
            )
        else:
            global_dp_buffer_len = None

        spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=input_ids.device),
            positions=torch.empty((0,), dtype=torch.int64, device=input_ids.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            positions=positions,
            input_embeds=input_embeds,
            global_num_tokens_gpu=self.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=self.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            None,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        lm_head = self.tree_worker.target_worker.model_runner.model.lm_head

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)
            forward_kwargs = {}
            if self.requires_input_embeds and input_embeds is not None:
                forward_kwargs["input_embeds"] = input_embeds
            draft_hidden = forward(
                input_ids,
                positions,
                forward_batch,
                **forward_kwargs,
            )
            self.hidden_states[:num_tokens].copy_(draft_hidden)
            if not self.capture_topk_in_graph:
                return self.hidden_states[:num_tokens]
            step_hidden = self.hidden_states[:num_tokens].view(
                bs, self.block_size, -1
            )[:, 1 : 1 + self.spec_steps, :].reshape(-1, draft_hidden.shape[-1])
            topk_p_flat, topk_index_flat = self.tree_worker._topk_from_vocab_parallel_head(
                hidden_states=step_hidden,
                lm_head=lm_head,
                topk=self.topk,
            )
            topk_p.copy_(topk_p_flat.view(bs, self.spec_steps, self.topk))
            topk_index.copy_(topk_index_flat.view(bs, self.spec_steps, self.topk))
            if self.capture_builder_in_graph:
                built_parent_list, built_top_scores_index, built_draft_tokens = (
                    build_dflash_tree_candidates_from_per_step_topk(
                        topk_p=topk_p,
                        topk_index=topk_index,
                        num_verify_tokens=int(self.tree_worker.num_verify_tokens),
                        candidate_scores_buf=candidate_scores_buf,
                        candidate_tokens_buf=candidate_tokens_buf,
                        parent_list_buf=parent_list,
                        top_scores_index_buf=top_scores_index,
                    )
                )
                if self.tree_worker._assert_builder_equiv:
                    self.tree_worker._assert_tree_builder_equiv(
                        topk_p=topk_p,
                        topk_index=topk_index,
                        parent_list=built_parent_list,
                        top_scores_index=built_top_scores_index,
                        draft_tokens=built_draft_tokens,
                    )
                draft_tokens_buf[:, : built_draft_tokens.shape[1]].copy_(built_draft_tokens)
                return (
                    built_parent_list,
                    built_top_scores_index,
                    draft_tokens_buf[:, : built_draft_tokens.shape[1]],
                )
            return topk_p, topk_index

        self.deepep_adapter.capture(is_extend_in_batch=False)
        self._capture_init(run_once)
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )
        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def replay(self, forward_batch: ForwardBatch):
        raw_bs = int(forward_batch.batch_size)
        raw_num_token = int(len(forward_batch.input_ids))
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = int(self.capture_bs[index])
        if bs != raw_bs:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.out_cache_loc.zero_()
            self.positions.zero_()
            self.input_ids.zero_()
            if self.input_embeds is not None:
                self.input_embeds.zero_()

        num_tokens = bs * self.num_tokens_per_bs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        if self.input_embeds is not None and forward_batch.input_embeds is not None:
            self.input_embeds[:raw_num_token].copy_(forward_batch.input_embeds)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(self.seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        seq_lens_sum = int(forward_batch.seq_lens_sum) + (
            bs - raw_bs
        ) * int(self.seq_len_fill_value)
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            seq_lens_sum,
            None,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu[:bs],
        )

        orig_batch_size = forward_batch.batch_size
        orig_seq_lens = forward_batch.seq_lens
        orig_req_pool_indices = forward_batch.req_pool_indices
        orig_positions = forward_batch.positions
        orig_seq_lens_cpu = forward_batch.seq_lens_cpu

        if bs != raw_bs:
            forward_batch.batch_size = bs
            forward_batch.seq_lens = self.seq_lens[:bs]
            forward_batch.req_pool_indices = self.req_pool_indices[:bs]
            forward_batch.positions = self.positions[:num_tokens]
            forward_batch.seq_lens_cpu = self.seq_lens_cpu[:bs]

        self.raw_bs = raw_bs
        self.bs = bs
        self._replay(forward_batch)
        self._maybe_debug_sync("draft_graph_replay")
        output = self.output_buffers[bs]

        if bs != raw_bs:
            forward_batch.batch_size = orig_batch_size
            forward_batch.seq_lens = orig_seq_lens
            forward_batch.req_pool_indices = orig_req_pool_indices
            forward_batch.positions = orig_positions
            forward_batch.seq_lens_cpu = orig_seq_lens_cpu
        if self.capture_builder_in_graph:
            parent_list, top_scores_index, draft_tokens = output
            if bs != raw_bs:
                return (
                    parent_list[:raw_bs],
                    top_scores_index[:raw_bs],
                    draft_tokens[:raw_bs],
                )
            return parent_list, top_scores_index, draft_tokens

        if self.capture_topk_in_graph:
            topk_p, topk_index = output
            if bs != raw_bs:
                topk_p = topk_p[:raw_bs]
                topk_index = topk_index[:raw_bs]
        else:
            step_hidden = self.hidden_states[: raw_bs * self.block_size].view(
                raw_bs, self.block_size, -1
            )[:, 1 : 1 + self.spec_steps, :].reshape(-1, self.hidden_states.shape[-1])
            topk_p_flat, topk_index_flat = self.tree_worker._topk_from_vocab_parallel_head(
                hidden_states=step_hidden,
                lm_head=self.tree_worker.target_worker.model_runner.model.lm_head,
                topk=self.topk,
            )
            topk_p = topk_p_flat.view(raw_bs, self.spec_steps, self.topk)
            topk_index = topk_index_flat.view(raw_bs, self.spec_steps, self.topk)
        self._maybe_debug_sync("fastpath_topk")
        # Break aliasing with graph-owned output buffers before eager candidate assembly.
        topk_p = topk_p.clone()
        topk_index = topk_index.clone()
        parent_list, top_scores_index, draft_tokens = (
            build_dflash_tree_candidates_from_per_step_topk(
                topk_p=topk_p,
                topk_index=topk_index,
                num_verify_tokens=int(self.tree_worker.num_verify_tokens),
                candidate_scores_buf=self.candidate_scores_buf[:raw_bs],
                candidate_tokens_buf=self.candidate_tokens_buf[:raw_bs],
                parent_list_buf=self.parent_list[:raw_bs],
                top_scores_index_buf=self.top_scores_index[:raw_bs],
            )
        )
        self._maybe_debug_sync("fastpath_candidate_builder")
        if self.tree_worker._assert_builder_equiv:
            self.tree_worker._assert_tree_builder_equiv(
                topk_p=topk_p,
                topk_index=topk_index,
                parent_list=parent_list,
                top_scores_index=top_scores_index,
                draft_tokens=draft_tokens,
            )
        return parent_list, top_scores_index, draft_tokens
