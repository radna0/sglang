"""
Support attention backend for FlashMLA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
import triton
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.flex_utils import apply_attention_sinks
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


PAGE_SIZE = 64
FLASHMLA_SPARSE_TOPK_ALIGNMENT = 128


@dataclass
class FlashMLADecodeMetadata:
    flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    num_splits: Optional[torch.Tensor] = None
    block_kv_indices: Optional[torch.Tensor] = None

    def __init__(
        self,
        flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_splits: Optional[torch.Tensor] = None,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_kv_indices = block_kv_indices


class FlashMLABackend(FlashInferMLAAttnBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        self._logger = logging.getLogger(__name__)
        if getattr(
            model_runner.model_config, "has_dynamic_mla_kv_lora_rank", lambda: False
        )():
            raise NotImplementedError(
                "Dynamic per-layer kv_lora_rank schedules (CARE-E) are not supported by flashmla yet. "
                "Use `--attention-backend flashinfer` for dynamic-rank MLA."
            )
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.forward_metadata: Union[FlashMLADecodeMetadata] = None
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # Align to 64 for FlashMLA kernels (which often require 576)
        if self.kv_cache_dim % 64 != 0:
            self.kv_cache_dim = (self.kv_cache_dim + 63) // 64 * 64

        self.is_fp8_kvcache = self.data_type in {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.cuda_graph_kv_indices = None
        self.cuda_graph_mla_metadata = None
        self.cuda_graph_num_splits = None
        self.cuda_graph_mla_metadata_view = None
        self.cuda_graph_num_splits_view = None
        self.device_sm_major = torch.cuda.get_device_properties(
            self.req_to_token.device
        ).major

    def _reshape_q_all(
        self, q: torch.Tensor, layer: RadixAttention, q_rope: Optional[torch.Tensor]
    ) -> torch.Tensor:
        q_head_num = layer.tp_q_head_num
        q_nope = q.view(-1, q_head_num, q.shape[-1])
        if q_rope is not None:
            q_rope_dim = q_rope.shape[-1]
            if q_rope_dim > 0:
                q_rope = q_rope.view(-1, q_head_num, q_rope_dim)
                expected_nope_dim = layer.head_dim - layer.qk_rope_head_dim
                if expected_nope_dim > 0 and q_nope.shape[-1] not in {
                    expected_nope_dim,
                    layer.v_head_dim,
                }:
                    raise ValueError(
                        f"FlashMLA q_nope dim mismatch for layer={layer.layer_id}: "
                        f"got {q_nope.shape[-1]}, expected {expected_nope_dim} "
                        f"or {layer.v_head_dim}"
                    )
                if q_rope_dim != layer.qk_rope_head_dim:
                    raise ValueError(
                        f"FlashMLA q_rope dim mismatch for layer={layer.layer_id}: "
                        f"got {q_rope_dim}, expected {layer.qk_rope_head_dim}"
                    )
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            else:
                q_all = q_nope
        else:
            q_all = q_nope
        
        # Pad head_dim to multiple of 64 for FlashMLA kernel compatibility
        if q_all.shape[-1] > self.kv_cache_dim:
            raise ValueError(
                f"FlashMLA q width exceeds kv_cache_dim: q={q_all.shape[-1]}, "
                f"kv_cache_dim={self.kv_cache_dim} at layer={layer.layer_id}"
            )
        if q_all.shape[-1] < self.kv_cache_dim:
            q_padded = q_all.new_zeros((*q_all.shape[:-1], self.kv_cache_dim))
            q_padded[..., :q_all.shape[-1]] = q_all
            return q_padded
        return q_all

    def _align_sparse_topk(self, topk: int) -> int:
        if topk <= 0:
            return FLASHMLA_SPARSE_TOPK_ALIGNMENT
        return triton.cdiv(topk, FLASHMLA_SPARSE_TOPK_ALIGNMENT) * (
            FLASHMLA_SPARSE_TOPK_ALIGNMENT
        )

    def _build_decode_sliding_indices(
        self,
        seq_lens_cpu: torch.Tensor,
        sliding_window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        seq_lens_list = [int(x) for x in seq_lens_cpu.tolist()]
        raw_topk = min(max(seq_lens_list), int(sliding_window_size) + 1)
        topk = self._align_sparse_topk(raw_topk)
        indices = torch.full(
            (len(seq_lens_list), 1, topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for batch_idx, seq_len in enumerate(seq_lens_list):
            keep = min(seq_len, raw_topk)
            start = max(0, seq_len - keep)
            indices[batch_idx, 0, :keep] = torch.arange(
                start, seq_len, dtype=torch.int32, device=device
            )
        return indices

    def _build_extend_dense_kv_and_indices(
        self,
        k_cache: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        req_pool_indices = [int(x) for x in forward_batch.req_pool_indices.tolist()]
        seq_lens_cpu = [int(x) for x in forward_batch.seq_lens_cpu.tolist()]
        extend_prefix_lens_cpu = [int(x) for x in forward_batch.extend_prefix_lens_cpu]
        extend_seq_lens_cpu = [int(x) for x in forward_batch.extend_seq_lens_cpu]

        kv_segments = []
        sparse_rows = []
        kv_base = 0
        raw_topk = 0
        use_sliding = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )

        for req_idx, seq_len, prefix_len, extend_len in zip(
            req_pool_indices,
            seq_lens_cpu,
            extend_prefix_lens_cpu,
            extend_seq_lens_cpu,
        ):
            token_locs = self.req_to_token[req_idx, :seq_len].to(torch.long)
            kv_segments.append(k_cache[token_locs])

            for rel in range(extend_len):
                q_pos = prefix_len + rel
                start = (
                    max(0, q_pos - int(layer.sliding_window_size))
                    if use_sliding
                    else 0
                )
                row = torch.arange(
                    start + kv_base,
                    q_pos + kv_base + 1,
                    dtype=torch.int32,
                    device=k_cache.device,
                )
                sparse_rows.append(row)
                raw_topk = max(raw_topk, int(row.numel()))
            kv_base += seq_len

        if not kv_segments:
            raise ValueError(
                "FlashMLA sparse extend had no KV segments; check seq_lens and extend settings."
            )

        kv_dense = torch.cat(kv_segments, dim=0)
        topk = self._align_sparse_topk(raw_topk)
        indices = torch.full(
            (len(sparse_rows), 1, topk),
            -1,
            dtype=torch.int32,
            device=k_cache.device,
        )
        for row_idx, row in enumerate(sparse_rows):
            indices[row_idx, 0, : row.numel()] = row

        return kv_dense, indices

    def _build_decode_dense_kv_and_indices(
        self,
        k_cache: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        req_pool_indices = [int(x) for x in forward_batch.req_pool_indices.tolist()]
        seq_lens_cpu = [int(x) for x in forward_batch.seq_lens_cpu.tolist()]

        kv_segments = []
        sparse_rows = []
        kv_base = 0
        raw_topk = 0

        for req_idx, seq_len in zip(req_pool_indices, seq_lens_cpu):
            token_locs = self.req_to_token[req_idx, :seq_len].to(torch.long)
            kv_segments.append(k_cache[token_locs])

            if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
                start = max(0, seq_len - (int(layer.sliding_window_size) + 1))
            else:
                start = 0
            row = torch.arange(
                start + kv_base,
                seq_len + kv_base,
                dtype=torch.int32,
                device=k_cache.device,
            )
            sparse_rows.append(row)
            raw_topk = max(raw_topk, int(row.numel()))
            kv_base += seq_len

        if not kv_segments:
            raise ValueError(
                "FlashMLA sparse decode had no KV segments; check seq_lens and req_pool_indices."
            )

        kv_dense = torch.cat(kv_segments, dim=0)
        topk = self._align_sparse_topk(raw_topk)
        indices = torch.full(
            (len(sparse_rows), 1, topk),
            -1,
            dtype=torch.int32,
            device=k_cache.device,
        )
        for row_idx, row in enumerate(sparse_rows):
            indices[row_idx, 0, : row.numel()] = row

        return kv_dense, indices

    def _forward_sparse_prefill(
        self,
        q_all: torch.Tensor,
        kv_dense: torch.Tensor,
        indices: torch.Tensor,
        layer: RadixAttention,
        sinks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        if kv_dense.dim() != 3:
            raise ValueError(f"kv_dense must be 3D (tokens, heads, dim), got {kv_dense.dim()}.")
        if indices.dim() != 3:
            raise ValueError(
                f"indices must be 3D (tokens, 1, topk), got {indices.dim()}."
            )

        num_tokens, num_heads, head_dim = q_all.shape
        required_padding = 128 if self.device_sm_major >= 10 else 64
        need_padding = num_heads % required_padding != 0

        if need_padding:
            assert required_padding % num_heads == 0, (
                f"num_heads {num_heads} cannot be padded to {required_padding}"
            )
            q_padded = q_all.new_zeros((num_tokens, required_padding, head_dim))
            q_padded[:, :num_heads, :] = q_all
            q_input = q_padded
        else:
            q_input = q_all

        o, _, lse = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv_dense,
            indices=indices,
            sm_scale=layer.scaling,
            d_v=layer.v_head_dim,
        )

        if need_padding:
            o = o[:, :num_heads, :]
            lse = lse[:, :num_heads]

        if sinks is not None and sinks.numel() > 0:
            o = apply_attention_sinks(o, lse, sinks, base=2.0)
        return o

    def _bind_cuda_graph_decode_metadata(
        self,
        bs: int,
        max_seqlen_pad: int,
        actual_num_sm_parts: int,
        mla_metadata: torch.Tensor,
        num_splits: torch.Tensor,
    ) -> FlashMLADecodeMetadata:
        if self.cuda_graph_mla_metadata is None or self.cuda_graph_num_splits is None:
            raise RuntimeError(
                "CUDA graph metadata buffers were not initialized; call "
                "init_cuda_graph_state() before replay."
            )
        if self.cuda_graph_kv_indices is None:
            raise RuntimeError(
                "CUDA graph KV block indices were not initialized; call "
                "init_cuda_graph_state() before replay."
            )
        if actual_num_sm_parts > self.cuda_graph_mla_metadata.shape[0]:
            raise RuntimeError(
                "CUDA graph metadata overflow: "
                f"actual_num_sm_parts={actual_num_sm_parts}, "
                f"capacity={self.cuda_graph_mla_metadata.shape[0]}"
            )
        if bs + 1 > self.cuda_graph_num_splits.shape[0]:
            raise RuntimeError(
                "CUDA graph num_splits overflow: "
                f"bs={bs}, capacity={self.cuda_graph_num_splits.shape[0] - 1}"
            )

        self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
        self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
        self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
            :actual_num_sm_parts
        ]
        self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]
        decode_metadata = FlashMLADecodeMetadata(
            self.cuda_graph_mla_metadata_view,
            self.cuda_graph_num_splits_view,
            self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
        )
        self.forward_metadata = decode_metadata
        return decode_metadata

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            max_seqlen_pad = triton.cdiv(
                forward_batch.seq_lens_cpu.max().item(), PAGE_SIZE
            )
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = get_mla_metadata(
                forward_batch.seq_lens.to(torch.int32),
                self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
        elif forward_batch.forward_mode.is_target_verify():
            seq_lens_cpu = forward_batch.seq_lens_cpu + self.num_draft_tokens
            seq_lens = forward_batch.seq_lens + self.num_draft_tokens

            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            block_kv_indices = torch.full(
                (bs, max_seqlen_pad),
                -1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                seq_lens,
                None,
                block_kv_indices,
                self.req_to_token.stride(0),
                max_seqlen_pad,
            )
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
            )
        else:
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        if block_kv_indices is None:
            self.cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            self.cuda_graph_kv_indices = block_kv_indices

        device_props = torch.cuda.get_device_properties(self.req_to_token.device)
        max_num_sm_parts = device_props.multi_processor_count

        self.cuda_graph_mla_metadata = torch.empty(
            (max_num_sm_parts, 8),
            dtype=torch.int32,
            device="cuda",
        )
        self.cuda_graph_num_splits = torch.empty(
            max_bs + 1,
            dtype=torch.int32,
            device="cuda",
        )

        self.cuda_graph_mla_metadata_view = None
        self.cuda_graph_num_splits_view = None

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            num_q_heads = self.num_q_heads

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0], (
                f"num_sm_parts {actual_num_sm_parts} exceeds preallocated max "
                f"{self.cuda_graph_mla_metadata.shape[0]}"
            )

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                :actual_num_sm_parts
            ]
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata_view,
                self.cuda_graph_num_splits_view,
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )

        elif forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            max_seqlen_pad = triton.cdiv(seq_lens.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)

            self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                :actual_num_sm_parts
            ]
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata_view,
                self.cuda_graph_num_splits_view,
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            )
        else:
            super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            seq_lens = seq_lens[:bs]
            seq_lens_cpu = seq_lens_cpu[:bs]
            if seq_lens_cpu.numel() == 0:
                return
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )
            num_q_heads = self.num_q_heads

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            if self.cuda_graph_mla_metadata_view is not None and (
                actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]
            ):
                self._logger.warning(
                    f"num_sm_parts mismatch in CUDA Graph replay: "
                    f"capture={self.cuda_graph_mla_metadata_view.shape[0]}, "
                    f"replay={actual_num_sm_parts}. "
                    f"This may indicate batch size changed between capture and replay."
                )
            self._bind_cuda_graph_decode_metadata(
                bs=bs,
                max_seqlen_pad=max_seqlen_pad,
                actual_num_sm_parts=actual_num_sm_parts,
                mla_metadata=mla_metadata,
                num_splits=num_splits,
            )

        elif forward_mode.is_target_verify():
            if seq_lens_cpu is None:
                raise RuntimeError(
                    "seq_lens_cpu is required for target_verify replay metadata."
                )
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            seq_lens_cpu = seq_lens_cpu[:bs] + self.num_draft_tokens
            if seq_lens_cpu.numel() == 0:
                return
            max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)

            create_flashmla_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            if self.cuda_graph_mla_metadata_view is not None and (
                actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]
            ):
                self._logger.warning(
                    f"num_sm_parts mismatch in CUDA Graph replay target-verify: "
                    f"capture={self.cuda_graph_mla_metadata_view.shape[0]}, "
                    f"replay={actual_num_sm_parts}. "
                    f"This may indicate batch size changed between capture and replay."
                )
            self._bind_cuda_graph_decode_metadata(
                bs=bs,
                max_seqlen_pad=max_seqlen_pad,
                actual_num_sm_parts=actual_num_sm_parts,
                mla_metadata=mla_metadata,
                num_splits=num_splits,
            )
        else:
            super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                if k_rope is not None:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        v,
                    )
        bs = forward_batch.batch_size
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        q_all = self._reshape_q_all(q, layer, q_rope)
        q_width = q_all.shape[-1]
        reshape_q = q_all.view(
            bs, -1, layer.tp_q_head_num, q_width
        )
        use_sliding = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        use_sparse_decode = use_sliding or self.kv_cache_dim != (512 + 64)
        if use_sparse_decode:
            kv_dense, indices = self._build_decode_dense_kv_and_indices(
                k_cache, layer, forward_batch
            )
            o = self._forward_sparse_prefill(
                q_all=reshape_q.reshape(-1, layer.tp_q_head_num, q_width),
                kv_dense=kv_dense,
                indices=indices,
                layer=layer,
                sinks=sinks,
            )
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        decode_metadata = self.forward_metadata
        decode_indices = None
        causal = True
        effective_fp8_kvcache = self.is_fp8_kvcache
        k_cache_for_kernel = k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim)
        if self.is_fp8_kvcache:
            if layer.k_scale is not None:
                q_scale = layer.k_scale
                descale_q = layer.k_scale.reshape(1)
                descale_k = layer.k_scale.reshape(1)
            else:
                q_scale = torch.ones((1,), dtype=torch.float32, device=reshape_q.device)
                descale_q = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )
                descale_k = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )

            q_shape = reshape_q.shape
            reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
            reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
            reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
            o, lse = flash_mla_with_kvcache(
                q=reshape_q_fp8,
                k_cache=k_cache_for_kernel,
                block_table=decode_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=decode_metadata.flashmla_metadata,
                num_splits=decode_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=causal,
                descale_q=descale_q,
                descale_k=descale_k,
                indices=decode_indices,
            )
            if sinks is not None and sinks.numel() > 0:
                o = apply_attention_sinks(
                    o.view(-1, self.num_q_heads, layer.v_head_dim),
                    lse.reshape(-1, self.num_q_heads),
                    sinks,
                    base=2.0,
                ).view(o.shape)

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            o, lse = flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache_for_kernel,
                block_table=decode_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=decode_metadata.flashmla_metadata,
                num_splits=decode_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=causal,
                is_fp8_kvcache=effective_fp8_kvcache,
                indices=decode_indices,
            )
            if sinks is not None and sinks.numel() > 0:
                o = apply_attention_sinks(
                    o.view(-1, self.num_q_heads, layer.v_head_dim),
                    lse.reshape(-1, self.num_q_heads),
                    sinks,
                    base=2.0,
                ).view(o.shape)

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
            or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
        ):
            cache_loc = forward_batch.out_cache_loc
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    if k_rope is not None:
                        forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                            layer,
                            cache_loc,
                            k,
                            k_rope,
                        )
                    else:
                        forward_batch.token_to_kv_pool.set_kv_buffer(
                            layer,
                            cache_loc,
                            k,
                            v,
                        )

            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            q_all = self._reshape_q_all(q, layer, q_rope)
            kv_dense, indices = self._build_extend_dense_kv_and_indices(
                k_cache, layer, forward_batch
            )
            o = self._forward_sparse_prefill(q_all, kv_dense, indices, layer, sinks)
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            cache_loc = forward_batch.out_cache_loc

            if k is not None:
                assert v is not None
                if save_kv_cache:
                    if k_rope is not None:
                        forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                            layer, cache_loc, k, k_rope
                        )
                    else:
                        forward_batch.token_to_kv_pool.set_kv_buffer(
                            layer, cache_loc, k, v
                        )

            bs = forward_batch.batch_size
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            q_all = self._reshape_q_all(q, layer, q_rope)
            q_width = q_all.shape[-1]
            reshape_q = q_all.view(
                bs, -1, layer.tp_q_head_num, q_width
            )
            if self.is_fp8_kvcache:
                if layer.k_scale is not None:
                    q_scale = layer.k_scale
                    descale_q = layer.k_scale.reshape(1)
                    descale_k = layer.k_scale.reshape(1)
                else:
                    q_scale = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_q = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_k = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )

                q_shape = reshape_q.shape
                reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
                reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
                reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
                o, lse = flash_mla_with_kvcache(
                    q=reshape_q_fp8,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.num_draft_tokens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                    descale_q=descale_q,
                    descale_k=descale_k,
                )
                if sinks is not None and sinks.numel() > 0:
                    o = apply_attention_sinks(
                        o.view(-1, self.num_q_heads, layer.v_head_dim),
                        lse.reshape(-1, self.num_q_heads),
                        sinks,
                        base=2.0,
                    ).view(o.shape)
            else:
                o, lse = flash_mla_with_kvcache(
                    q=reshape_q,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.num_draft_tokens,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
                if sinks is not None and sinks.numel() > 0:
                    o = apply_attention_sinks(
                        o.view(-1, self.num_q_heads, layer.v_head_dim),
                        lse.reshape(-1, self.num_q_heads),
                        sinks,
                        base=2.0,
                    ).view(o.shape)
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class FlashMLAMultiStepDraftBackend:
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        if topk > 1:
            raise ValueError(
                "Currently FlashMLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashMLABackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=None,
                )
            )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        call_fn: Callable,
    ):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, block_kv_indices=None
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            # EAGLE draft worker uses DECODE mode for draft steps
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            # Create a dummy forward_mode for draft step
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, call_fn)
