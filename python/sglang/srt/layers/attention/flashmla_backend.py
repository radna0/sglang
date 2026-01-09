from __future__ import annotations

"""
Support attention backend for FlashMLA.
"""

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch
import triton
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


# FlashMLA only supports pagesize=64
PAGE_SIZE = 64

# FlashMLA FP8 issue: https://github.com/deepseek-ai/FlashMLA/issues/56


@dataclass
class FlashMLADecodeMetadata:
    flashmla_metadata: torch.Tensor | None = None
    num_splits: torch.Tensor | None = None
    block_kv_indices: torch.Tensor | None = None
    # Optional "indices/topk" mode metadata for SWA layers (GPT-OSS).
    swa_indices: torch.Tensor | None = None
    flashmla_metadata_swa: torch.Tensor | None = None
    num_splits_swa: torch.Tensor | None = None
    swa_topk: int = 0

    def __init__(
        self,
        flashmla_metadata: torch.Tensor | None = None,
        num_splits: torch.Tensor | None = None,
        block_kv_indices: torch.Tensor | None = None,
        swa_indices: torch.Tensor | None = None,
        flashmla_metadata_swa: torch.Tensor | None = None,
        num_splits_swa: torch.Tensor | None = None,
        swa_topk: int = 0,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_kv_indices = block_kv_indices
        self.swa_indices = swa_indices
        self.flashmla_metadata_swa = flashmla_metadata_swa
        self.num_splits_swa = num_splits_swa
        self.swa_topk = int(swa_topk or 0)


class FlashMLABackend(FlashInferMLAAttnBackend):
    """Flashmla attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        self._gptoss_transmla_force_mla = bool(
            getattr(getattr(model_runner.model_config, "hf_config", None), "architectures", None)
            and "GptOssForCausalLM" in model_runner.model_config.hf_config.architectures
            and bool(getattr(model_runner.model_config.hf_config, "use_transmla", False))
            and model_runner.use_mla_backend
        )
        # Optional: implement GPT-OSS sliding-window layers via FlashMLA "indices/topk"
        # mode by passing last-W indices (decode-only).
        #
        # NOTE: This is NOT DeepSeek semantic sparsity; this is deterministic SWA.
        self._gptoss_transmla_flashmla_swa_indices = False
        self._gptoss_transmla_swa_window_len = 0
        if self._gptoss_transmla_force_mla:
            enable = os.getenv(
                "SGLANG_GPTOSS_TRANSMLA_FLASHMLA_SWA_INDICES", "0"
            ).lower() in ("1", "true", "yes")
            window_cfg = getattr(model_runner.model_config.hf_config, "sliding_window", 0)
            try:
                window_len = int(window_cfg or 0)
            except Exception:
                window_len = 0
            if enable and window_len > 0:
                self._gptoss_transmla_flashmla_swa_indices = True
                self._gptoss_transmla_swa_window_len = window_len
        self._logged_swa_indices_once = False
        self._logged_swa_indices_unavailable_once = False
        # GPT-OSS uses alternating sliding/full attention layers. FlashMLA does not
        # natively support sliding-window masking, so for correctness we fall back to
        # FA3 on sliding layers (and for all prefill). Full-attention layers keep
        # using FlashMLA for decode.
        self._fa3_backend: FlashAttentionBackend | None = (
            FlashAttentionBackend(model_runner) if self._gptoss_transmla_force_mla else None
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
        # Check if KV cache is FP8 (supports both e4m3 and e5m2)
        self.is_fp8_kvcache = self.data_type in {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }
        # FlashMLA sparse/indices decode uses a packed FP8 KV cache layout
        # (see sgl-kernel/tests/test_flashmla.py: quantize_k_cache), where
        # bytes_per_token = kv_lora_rank + 4*(kv_lora_rank/128) + rope_bytes.
        #
        # Our GPT-OSS TransMLA KV cache is stored as (kv_lora_rank + qk_rope_head_dim)
        # elements, so unless the pool is explicitly allocated in the packed layout,
        # indices mode will crash. Disable early to keep the fallback path working.
        self._flashmla_fp8_sparse_bytes_per_token = 0
        if self.kv_lora_rank % 128 == 0:
            rope_bytes = torch.empty((), dtype=self.q_data_type).element_size()
            self._flashmla_fp8_sparse_bytes_per_token = int(
                self.kv_lora_rank
                + (self.kv_lora_rank // 128) * 4
                + self.qk_rope_head_dim * rope_bytes
            )
        if self._gptoss_transmla_flashmla_swa_indices and self.is_fp8_kvcache:
            expected = int(self._flashmla_fp8_sparse_bytes_per_token or 0)
            if expected <= 0 or self.kv_cache_dim != expected:
                if not self._logged_swa_indices_unavailable_once:
                    print(
                        "[FLASHMLA] GPT-OSS SWA via indices disabled: "
                        f"requires packed FP8 KV cache bytes_per_token={expected}, "
                        f"but kv_cache_dim={self.kv_cache_dim}"
                    )
                    self._logged_swa_indices_unavailable_once = True
                self._gptoss_transmla_flashmla_swa_indices = False

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.cuda_graph_swa_indices: torch.Tensor | None = None
        self.cuda_graph_mla_metadata_swa: torch.Tensor | None = None
        self.cuda_graph_num_splits_swa: torch.Tensor | None = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if self._fa3_backend is not None:
            self._fa3_backend.init_forward_metadata(forward_batch)

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
            swa_indices = None
            mla_metadata_swa = None
            num_splits_swa = None
            swa_topk = 0
            if (
                self._gptoss_transmla_flashmla_swa_indices
                and not self.num_draft_tokens
                and self._gptoss_transmla_swa_window_len > 0
            ):
                if not self.is_fp8_kvcache:
                    # FlashMLA sparse/indices decode is not supported for BF16 KV on SM90.
                    # Keep the existing FA3 fallback for SWA layers.
                    if not self._logged_swa_indices_unavailable_once:
                        print(
                            f"[FLASHMLA] GPT-OSS SWA via indices disabled (kv_cache_dtype={self.data_type})"
                        )
                        self._logged_swa_indices_unavailable_once = True
                    # Disable so we don't keep attempting sparse metadata construction.
                    self._gptoss_transmla_flashmla_swa_indices = False
                else:
                    swa_topk = int(self._gptoss_transmla_swa_window_len)
                    seq_lens_i32 = forward_batch.seq_lens.to(torch.int32)
                    window_lens = torch.minimum(
                        seq_lens_i32,
                        torch.full_like(seq_lens_i32, swa_topk, dtype=torch.int32),
                    )
                    start_idx = (seq_lens_i32 - window_lens).to(torch.int32)
                    offs = torch.arange(
                        swa_topk, dtype=torch.int32, device=seq_lens_i32.device
                    ).view(1, -1)
                    idx = start_idx.view(-1, 1) + offs
                    idx = torch.where(
                        offs < window_lens.view(-1, 1),
                        idx,
                        torch.full_like(idx, -1),
                    )
                    swa_indices = idx.unsqueeze(1)  # [bs, 1, topk]
                    try:
                        mla_metadata_swa, num_splits_swa = get_mla_metadata(
                            seq_lens_i32,
                            self.num_q_heads,
                            1,
                            num_heads_q=self.num_q_heads,
                            is_fp8_kvcache=self.is_fp8_kvcache,
                            topk=swa_topk,
                        )
                    except Exception as e:
                        if not self._logged_swa_indices_unavailable_once:
                            print(
                                f"[FLASHMLA] GPT-OSS SWA via indices disabled: {type(e).__name__}: {e}"
                            )
                            self._logged_swa_indices_unavailable_once = True
                        self._gptoss_transmla_flashmla_swa_indices = False
                        swa_indices = None
                        mla_metadata_swa = None
                        num_splits_swa = None
                        swa_topk = 0

            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_kv_indices,
                swa_indices=swa_indices,
                flashmla_metadata_swa=mla_metadata_swa,
                num_splits_swa=num_splits_swa,
                swa_topk=swa_topk,
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

            # Use FlashMLADecodeMetadata which has the attributes forward_extend expects
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
        # Hybrid GPT-OSS TransMLA mode may dispatch some layers to FA3 even when the
        # selected backend is FlashMLA. When CUDA graph capture/replay is enabled,
        # FA3 must have its own graph buffers/metadata initialized too.
        if self._fa3_backend is not None:
            self._fa3_backend.init_cuda_graph_state(max_bs, max_num_tokens)

        if block_kv_indices is None:
            cuda_graph_kv_indices = torch.full(
                (max_bs, (self.max_context_len + PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = block_kv_indices

        if self.num_draft_tokens:
            self.cuda_graph_mla_metadata, self.cuda_graph_num_splits = get_mla_metadata(
                torch.ones(
                    max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device
                ),
                self.num_draft_tokens * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
        else:
            self.cuda_graph_mla_metadata, self.cuda_graph_num_splits = get_mla_metadata(
                torch.ones(
                    max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device
                ),
                self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
        self.cuda_graph_kv_indices = cuda_graph_kv_indices
        if (
            self._gptoss_transmla_flashmla_swa_indices
            and not self.num_draft_tokens
            and self._gptoss_transmla_swa_window_len > 0
        ):
            if not self.is_fp8_kvcache:
                if not self._logged_swa_indices_unavailable_once:
                    print(
                        f"[FLASHMLA] GPT-OSS SWA via indices disabled (kv_cache_dtype={self.data_type})"
                    )
                    self._logged_swa_indices_unavailable_once = True
                self._gptoss_transmla_flashmla_swa_indices = False
                return
            swa_topk = int(self._gptoss_transmla_swa_window_len)
            self.cuda_graph_swa_indices = torch.full(
                (max_bs, 1, swa_topk),
                -1,
                dtype=torch.int32,
                device=cuda_graph_kv_indices.device,
            )
            try:
                self.cuda_graph_mla_metadata_swa, self.cuda_graph_num_splits_swa = (
                    get_mla_metadata(
                        torch.ones(
                            max_bs,
                            dtype=torch.int32,
                            device=cuda_graph_kv_indices.device,
                        ),
                        self.num_q_heads,
                        1,
                        num_heads_q=self.num_q_heads,
                        is_fp8_kvcache=self.is_fp8_kvcache,
                        topk=swa_topk,
                    )
                )
            except Exception as e:
                if not self._logged_swa_indices_unavailable_once:
                    print(
                        f"[FLASHMLA] GPT-OSS SWA via indices disabled: {type(e).__name__}: {e}"
                    )
                    self._logged_swa_indices_unavailable_once = True
                self._gptoss_transmla_flashmla_swa_indices = False
                self.cuda_graph_swa_indices = None
                self.cuda_graph_mla_metadata_swa = None
                self.cuda_graph_num_splits_swa = None
                return

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
        if self._fa3_backend is not None:
            # Initialize FA3 capture metadata for any mode where we might fall back.
            self._fa3_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

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
            num_q_heads = self.num_q_heads * (self.num_draft_tokens or 1)
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            swa_indices = None
            flashmla_metadata_swa = None
            num_splits_swa = None
            swa_topk = 0
            if (
                self.cuda_graph_swa_indices is not None
                and self.cuda_graph_mla_metadata_swa is not None
                and self.cuda_graph_num_splits_swa is not None
            ):
                swa_topk = int(self._gptoss_transmla_swa_window_len)
                seq_lens_i32 = seq_lens.to(torch.int32)
                window_lens = torch.minimum(
                    seq_lens_i32,
                    torch.full_like(seq_lens_i32, swa_topk, dtype=torch.int32),
                )
                start_idx = (seq_lens_i32 - window_lens).to(torch.int32)
                offs = torch.arange(
                    swa_topk, dtype=torch.int32, device=seq_lens_i32.device
                ).view(1, -1)
                idx = start_idx.view(-1, 1) + offs
                idx = torch.where(
                    offs < window_lens.view(-1, 1),
                    idx,
                    torch.full_like(idx, -1),
                )
                self.cuda_graph_swa_indices[:bs, 0, :].copy_(idx)
                try:
                    mla_metadata_swa, ns_swa = get_mla_metadata(
                        seq_lens_i32,
                        self.num_q_heads,
                        1,
                        num_heads_q=self.num_q_heads,
                        is_fp8_kvcache=self.is_fp8_kvcache,
                        topk=swa_topk,
                    )
                except Exception as e:
                    if not self._logged_swa_indices_unavailable_once:
                        print(
                            f"[FLASHMLA] GPT-OSS SWA via indices disabled: {type(e).__name__}: {e}"
                        )
                        self._logged_swa_indices_unavailable_once = True
                    self._gptoss_transmla_flashmla_swa_indices = False
                    self.cuda_graph_swa_indices = None
                    self.cuda_graph_mla_metadata_swa = None
                    self.cuda_graph_num_splits_swa = None
                    swa_indices = None
                    flashmla_metadata_swa = None
                    num_splits_swa = None
                    swa_topk = 0
                else:
                    self.cuda_graph_mla_metadata_swa.copy_(mla_metadata_swa)
                    self.cuda_graph_num_splits_swa[: bs + 1].copy_(ns_swa)
                    swa_indices = self.cuda_graph_swa_indices[:bs, :, :]
                    flashmla_metadata_swa = self.cuda_graph_mla_metadata_swa
                    num_splits_swa = self.cuda_graph_num_splits_swa[: bs + 1]

            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
                swa_indices=swa_indices,
                flashmla_metadata_swa=flashmla_metadata_swa,
                num_splits_swa=num_splits_swa,
                swa_topk=swa_topk,
            )
        elif forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            if self._fa3_backend is not None:
                # Re-init with the adjusted seq_lens used in target-verify.
                self._fa3_backend.init_forward_metadata_capture_cuda_graph(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                )
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
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
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
        if self._fa3_backend is not None:
            self._fa3_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            seq_lens = seq_lens[:bs]
            seq_lens_cpu = seq_lens_cpu[:bs]
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
            num_q_heads = self.num_q_heads * (self.num_draft_tokens or 1)
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]
            if (
                self.cuda_graph_swa_indices is not None
                and self.cuda_graph_mla_metadata_swa is not None
                and self.cuda_graph_num_splits_swa is not None
                and self._gptoss_transmla_swa_window_len > 0
                and not self.num_draft_tokens
            ):
                swa_topk = int(self._gptoss_transmla_swa_window_len)
                seq_lens_i32 = seq_lens.to(torch.int32)
                window_lens = torch.minimum(
                    seq_lens_i32,
                    torch.full_like(seq_lens_i32, swa_topk, dtype=torch.int32),
                )
                start_idx = (seq_lens_i32 - window_lens).to(torch.int32)
                offs = torch.arange(
                    swa_topk, dtype=torch.int32, device=seq_lens_i32.device
                ).view(1, -1)
                idx = start_idx.view(-1, 1) + offs
                idx = torch.where(
                    offs < window_lens.view(-1, 1),
                    idx,
                    torch.full_like(idx, -1),
                )
                self.cuda_graph_swa_indices[:bs, 0, :].copy_(idx)
                try:
                    mla_metadata_swa, ns_swa = get_mla_metadata(
                        seq_lens_i32,
                        self.num_q_heads,
                        1,
                        num_heads_q=self.num_q_heads,
                        is_fp8_kvcache=self.is_fp8_kvcache,
                        topk=swa_topk,
                    )
                except Exception as e:
                    if not self._logged_swa_indices_unavailable_once:
                        print(
                            f"[FLASHMLA] GPT-OSS SWA via indices disabled: {type(e).__name__}: {e}"
                        )
                        self._logged_swa_indices_unavailable_once = True
                    self._gptoss_transmla_flashmla_swa_indices = False
                    self.cuda_graph_swa_indices = None
                    self.cuda_graph_mla_metadata_swa = None
                    self.cuda_graph_num_splits_swa = None
                    self.forward_metadata.swa_indices = None
                    self.forward_metadata.flashmla_metadata_swa = None
                    self.forward_metadata.num_splits_swa = None
                    self.forward_metadata.swa_topk = 0
                else:
                    self.cuda_graph_mla_metadata_swa.copy_(mla_metadata_swa)
                    self.cuda_graph_num_splits_swa[: bs + 1].copy_(ns_swa)
                    self.forward_metadata.swa_indices = self.cuda_graph_swa_indices[
                        :bs, :, :
                    ]
                    self.forward_metadata.flashmla_metadata_swa = (
                        self.cuda_graph_mla_metadata_swa
                    )
                    self.forward_metadata.num_splits_swa = (
                        self.cuda_graph_num_splits_swa[: bs + 1]
                    )
                    self.forward_metadata.swa_topk = swa_topk
        elif forward_mode.is_target_verify():
            seq_lens = seq_lens[:bs] + self.num_draft_tokens
            seq_lens_cpu = seq_lens_cpu[:bs] + self.num_draft_tokens
            if self._fa3_backend is not None:
                # Re-init with the adjusted seq_lens used in target-verify.
                self._fa3_backend.init_forward_metadata_replay_cuda_graph(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    seq_lens_sum,
                    encoder_lens,
                    forward_mode,
                    spec_info,
                    seq_lens_cpu,
                )
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
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_kv_indices = self.cuda_graph_kv_indices[
                :bs, :max_seqlen_pad
            ]
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
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        if (
            self._fa3_backend is not None
            and is_swa_layer
            and (
                not self._gptoss_transmla_flashmla_swa_indices
                or self.forward_metadata.swa_indices is None
                or self.forward_metadata.flashmla_metadata_swa is None
                or self.forward_metadata.num_splits_swa is None
                or self.forward_metadata.swa_topk <= 0
            )
        ):
            return self._fa3_backend.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )

        LOG2E = 1.4426950408889634
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

        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            reshape_q = torch.cat([q_nope, q_rope], dim=-1).view(
                bs, -1, layer.tp_q_head_num, layer.head_dim
            )
        else:
            reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)

        # GPT-OSS SWA via FlashMLA indices/topk mode (decode-only).
        if (
            is_swa_layer
            and self._gptoss_transmla_flashmla_swa_indices
            and self.forward_metadata.swa_indices is not None
            and self.forward_metadata.flashmla_metadata_swa is not None
            and self.forward_metadata.num_splits_swa is not None
            and self.forward_metadata.swa_topk > 0
        ):
            bytes_per_token = int(k_cache.shape[-1])
            expected = int(self._flashmla_fp8_sparse_bytes_per_token or 0)
            if expected > 0 and bytes_per_token != expected:
                if not self._logged_swa_indices_unavailable_once:
                    print(
                        "[FLASHMLA] GPT-OSS SWA via indices disabled: "
                        f"expected bytes_per_token={expected}, got {bytes_per_token}"
                    )
                    self._logged_swa_indices_unavailable_once = True
                self._gptoss_transmla_flashmla_swa_indices = False
                if self._fa3_backend is not None:
                    return self._fa3_backend.forward_decode(
                        q,
                        k,
                        v,
                        layer,
                        forward_batch,
                        save_kv_cache=save_kv_cache,
                        q_rope=q_rope,
                        k_rope=k_rope,
                        sinks=sinks,
                    )
                raise RuntimeError(
                    "FlashMLA SWA indices disabled but FA3 fallback is unavailable."
                )

            if not self._logged_swa_indices_once:
                print(
                    f"[FLASHMLA] GPT-OSS SWA via indices enabled (topk={self.forward_metadata.swa_topk})"
                )
                self._logged_swa_indices_once = True

            try:
                o, softmax_lse = flash_mla_with_kvcache(
                    q=reshape_q,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, bytes_per_token),
                    block_table=self.forward_metadata.block_kv_indices[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata_swa,
                    num_splits=self.forward_metadata.num_splits_swa,
                    softmax_scale=layer.scaling,
                    causal=False,
                    is_fp8_kvcache=self.is_fp8_kvcache,
                    indices=self.forward_metadata.swa_indices,
                )
            except Exception as e:
                if not self._logged_swa_indices_unavailable_once:
                    print(
                        f"[FLASHMLA] GPT-OSS SWA via indices disabled: {type(e).__name__}: {e}"
                    )
                    self._logged_swa_indices_unavailable_once = True
                self._gptoss_transmla_flashmla_swa_indices = False
                if self._fa3_backend is not None:
                    return self._fa3_backend.forward_decode(
                        q,
                        k,
                        v,
                        layer,
                        forward_batch,
                        save_kv_cache=save_kv_cache,
                        q_rope=q_rope,
                        k_rope=k_rope,
                        sinks=sinks,
                    )
                raise
            if sinks is not None:
                sinks_f = sinks.to(torch.float32)
                if sinks_f.ndim == 1:
                    sinks_f = sinks_f.view(1, 1, -1)
                lse_f = softmax_lse.permute(0, 2, 1).to(torch.float32)
                if sinks_f.shape[-1] != lse_f.shape[-1]:
                    raise ValueError(
                        f"sinks shape mismatch: sinks={tuple(sinks_f.shape)} lse={tuple(lse_f.shape)}"
                    )
                scale = 1.0 / (1.0 + torch.exp2(sinks_f * LOG2E - lse_f))
                o = o * scale.to(o.dtype).unsqueeze(-1)
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

        if self.is_fp8_kvcache:
            # For FP8 KV cache, Q needs to be converted to FP8 for FlashMLA kernel
            # In SGLang, we use layer.k_scale for both q and k scales
            if layer.k_scale is not None:
                q_scale = layer.k_scale
                descale_q = layer.k_scale.reshape(1)
                descale_k = layer.k_scale.reshape(1)
            else:
                # Fallback to 1.0 if k_scale is not initialized
                q_scale = torch.ones((1,), dtype=torch.float32, device=reshape_q.device)
                descale_q = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )
                descale_k = torch.ones(
                    (1,), dtype=torch.float32, device=reshape_q.device
                )

            # Reshape to 2D for scaled_fp8_quant (which requires 2D input)
            q_shape = reshape_q.shape
            reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
            reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
            reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
            o, softmax_lse = flash_mla_with_kvcache(
                q=reshape_q_fp8,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
                descale_q=descale_q,
                descale_k=descale_k,
            )
            if sinks is not None:
                sinks_f = sinks.to(torch.float32)
                if sinks_f.ndim == 1:
                    sinks_f = sinks_f.view(1, 1, -1)
                lse_f = softmax_lse.permute(0, 2, 1).to(torch.float32)
                if sinks_f.shape[-1] != lse_f.shape[-1]:
                    raise ValueError(
                        f"sinks shape mismatch: sinks={tuple(sinks_f.shape)} lse={tuple(lse_f.shape)}"
                    )
                scale = 1.0 / (1.0 + torch.exp2(sinks_f * LOG2E - lse_f))
                o = o * scale.to(o.dtype).unsqueeze(-1)

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        else:
            # todo: need check all causal True or False?
            o, softmax_lse = flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_kv_indices[:bs],
                cache_seqlens=forward_batch.seq_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
            )
            if sinks is not None:
                sinks_f = sinks.to(torch.float32)
                if sinks_f.ndim == 1:
                    sinks_f = sinks_f.view(1, 1, -1)
                lse_f = softmax_lse.permute(0, 2, 1).to(torch.float32)
                if sinks_f.shape[-1] != lse_f.shape[-1]:
                    raise ValueError(
                        f"sinks shape mismatch: sinks={tuple(sinks_f.shape)} lse={tuple(lse_f.shape)}"
                    )
                scale = 1.0 / (1.0 + torch.exp2(sinks_f * LOG2E - lse_f))
                o = o * scale.to(o.dtype).unsqueeze(-1)

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
            if self._fa3_backend is not None:
                return self._fa3_backend.forward_extend(
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                    q_rope=q_rope,
                    k_rope=k_rope,
                    sinks=sinks,
                )
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
            )
        else:
            LOG2E = 1.4426950408889634
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
                            layer, cache_loc, k, v
                        )

            bs = forward_batch.batch_size
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
                reshape_q = torch.cat([q_nope, q_rope], dim=-1).view(
                    bs, -1, layer.tp_q_head_num, layer.head_dim
                )
            else:
                reshape_q = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            if self.is_fp8_kvcache:
                # For FP8 KV cache, Q needs to be converted to FP8 for FlashMLA kernel
                # In SGLang, we use layer.k_scale for both q and k scales
                if layer.k_scale is not None:
                    q_scale = layer.k_scale
                    descale_q = layer.k_scale.reshape(1)
                    descale_k = layer.k_scale.reshape(1)
                else:
                    # Fallback to 1.0 if k_scale is not initialized
                    q_scale = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_q = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )
                    descale_k = torch.ones(
                        (1,), dtype=torch.float32, device=reshape_q.device
                    )

                # Quantize Q using scaled_fp8_quant (matching vLLM's approach)
                # Reshape to 2D for scaled_fp8_quant (which requires 2D input)
                q_shape = reshape_q.shape
                reshape_q_2d = reshape_q.reshape(-1, q_shape[-1])
                reshape_q_fp8_2d, _ = scaled_fp8_quant(reshape_q_2d, q_scale)
                reshape_q_fp8 = reshape_q_fp8_2d.reshape(q_shape)
                o, softmax_lse = flash_mla_with_kvcache(
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
            else:
                o, softmax_lse = flash_mla_with_kvcache(
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
            if sinks is not None:
                sinks_f = sinks.to(torch.float32)
                if sinks_f.ndim == 1:
                    sinks_f = sinks_f.view(1, 1, -1)
                lse_f = softmax_lse.permute(0, 2, 1).to(torch.float32)
                if sinks_f.shape[-1] != lse_f.shape[-1]:
                    raise ValueError(
                        f"sinks shape mismatch: sinks={tuple(sinks_f.shape)} lse={tuple(lse_f.shape)}"
                    )
                scale = 1.0 / (1.0 + torch.exp2(sinks_f * LOG2E - lse_f))
                o = o * scale.to(o.dtype).unsqueeze(-1)
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


# TODO: multi step kv indices optimization
class FlashMLAMultiStepDraftBackend:
    """
    Wrap multiple flashmla attention backends as one for multiple consecutive
    draft decoding steps.
    """

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
