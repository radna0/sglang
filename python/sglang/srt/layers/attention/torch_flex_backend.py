from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import json
import logging
import os
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flex_utils import (
    apply_attention_sinks,
    make_extend_causal_mask_mod,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)

_TRUE_ENVS = {"1", "true", "True", "yes", "YES", "on", "ON"}


class TorchFlexAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner, *, kernel_options: dict | None = None):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        if (os.environ.get("SGLANG_FLEX_IMPORT_DEBUG") or "").strip() in _TRUE_ENVS:
            try:
                print("[flex] TorchFlexAttnBackend __init__ begin", flush=True)
            except Exception:
                pass
        # torch.compile knobs:
        # - For long-context inference we often run with a fairly stable shape set, so
        #   allowing operators to disable `dynamic` can improve specialization and
        #   reduce compile cache churn.
        # - Some PyTorch builds disallow specifying both `mode` and `options`; handle
        #   this the same way as other stacks (best-effort fallback).
        dynamic = (os.environ.get("SGLANG_FLEX_COMPILE_DYNAMIC") or "1").strip().lower() not in {
            "",
            "0",
            "false",
            "off",
            "no",
        }
        disable_compile = (os.environ.get("SGLANG_FLEX_DISABLE_COMPILE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        mode = (os.environ.get("SGLANG_FLEX_COMPILE_MODE") or "").strip() or None
        if mode is not None and mode.lower() in {"none", "null", "default"}:
            mode = None
        if disable_compile:
            self.flex_attention = flex_attention
        else:
            require_compile = (os.environ.get("SGLANG_FLEX_REQUIRE_COMPILE") or "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            try:
                if mode:
                    self.flex_attention = torch.compile(
                        flex_attention, dynamic=bool(dynamic), mode=mode
                    )
                else:
                    self.flex_attention = torch.compile(
                        flex_attention, dynamic=bool(dynamic)
                    )
            except Exception as exc:
                # NOTE: FlexAttention performance relies on successful compilation. However, we prefer
                # a functional (but slower) server over a hard crash during startup.
                logger.warning(
                    "torch.compile(flex_attention) failed (%s): %s",
                    type(exc).__name__,
                    str(exc)[:300],
                )
                if require_compile:
                    raise
                logger.warning("Falling back to eager FlexAttention (no torch.compile).")
                self.flex_attention = flex_attention
        # Avoid touching torch._dynamo unless we actually attempt to compile; importing
        # torch._dynamo can be expensive and has occasionally been implicated in startup
        # issues for containerized multi-process servers.
        if not disable_compile:
            torch._dynamo.config.cache_size_limit = 1024
            torch._dynamo.config.accumulated_cache_size_limit = 1024

        if kernel_options is None:
            raw = (os.environ.get("SGLANG_FLEX_KERNEL_OPTIONS") or "").strip()
            if raw:
                try:
                    kernel_options = json.loads(raw)
                except Exception as e:
                    raise ValueError(
                        f"Invalid JSON in SGLANG_FLEX_KERNEL_OPTIONS={raw!r}: {e}"
                    ) from e
            else:
                kernel_options = {}
        self.flex_kernel_options = kernel_options
        if (os.environ.get("SGLANG_FLEX_LOG_KERNEL_OPTIONS") or "").strip().lower() in {"1", "true", "yes"}:
            try:
                print(f"[flex] kernel_options={self.flex_kernel_options!r}", flush=True)
            except Exception:
                pass
        # Cache block masks because create_block_mask can be expensive and we often
        # reuse the same (q_len, kv_len, window_left) under batching/cuda-graphs.
        self._block_mask_cache: dict[tuple, object] = {}
        self._kernel_options_disabled = False
        self._require_kernel_options = (
            (os.environ.get("SGLANG_FLEX_REQUIRE_KERNEL_OPTIONS") or "").strip() in _TRUE_ENVS
        )

    def _flex(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        score_mod=None,
        block_mask=None,
        scale=None,
        enable_gqa: bool = False,
        return_lse: bool = False,
    ):
        """Call FlexAttention with safe kernel_options fallback."""
        # FP8 KV-cache compatibility:
        # Some serving stacks store KV cache in FP8 (e.g. fp8_e4m3). FlexAttention kernels
        # may not support float8 K/V directly on all PyTorch builds. For correctness, we
        # upcast K/V to the query dtype (bf16/fp16) before calling flex_attention.
        #
        # Note: this may allocate, so the recommended fast path is still to use FA3/FA4
        # for FP8 KV-cache regimes. This is a compatibility fallback for Flex backends.
        try:
            if key.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                key = key.to(dtype=query.dtype)
            if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                value = value.to(dtype=query.dtype)
        except Exception:
            # If float8 dtypes are not available (older torch), ignore.
            pass
        try:
            return self.flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
                return_lse=return_lse,
                kernel_options=self.flex_kernel_options if not self._kernel_options_disabled else {},
            )
        except Exception as e:
            if (
                not self._kernel_options_disabled
                and isinstance(self.flex_kernel_options, dict)
                and len(self.flex_kernel_options) > 0
            ):
                if bool(self._require_kernel_options):
                    logger.error(
                        "FlexAttention kernel_options=%r failed (%s) and SGLANG_FLEX_REQUIRE_KERNEL_OPTIONS=1, refusing to fall back.",
                        self.flex_kernel_options,
                        type(e).__name__,
                    )
                    raise
                logger.warning(
                    "FlexAttention kernel_options=%r failed (%s). Disabling kernel_options and retrying.",
                    self.flex_kernel_options,
                    type(e).__name__,
                )
                self._kernel_options_disabled = True
                return self.flex_attention(
                    query,
                    key,
                    value,
                    score_mod=score_mod,
                    block_mask=block_mask,
                    scale=scale,
                    enable_gqa=enable_gqa,
                    return_lse=return_lse,
                    kernel_options={},
                )
            raise

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        # TODO: find a more elegant way to save memory
        # Currently maintain the same memory as torch_native_backend
        torch.cuda.empty_cache()

        # FlexAttention masks are layer-dependent (sliding window differs per layer),
        # so we build/cached them at call-time in forward_extend/forward_decode.
        # Keep any per-forward metadata minimal here.
        self._last_forward_mode_is_extend = forward_batch.forward_mode.is_extend()

    # ------------------------
    # CUDA graph integration
    # ------------------------
    # FlexAttention backends historically disabled CUDA graphs in SGLang because
    # older implementations allocated per-step metadata (BlockMasks / paging math)
    # and relied on Python control flow. Our V2 backend is designed to keep all
    # decode metadata in fixed-shape buffers, so we allow CUDA graphs behind an
    # explicit opt-in (see SGLANG_FLEX_ALLOW_CUDA_GRAPH).
    #
    # These hooks are required by `CudaGraphRunner`. For FlexAttention, we can
    # keep them as no-ops because all metadata needed for a forward pass is
    # constructed inside the captured graph (and reads from the graph input
    # buffers, which are updated in-place before replay).

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self._cuda_graph_max_bs = int(max_bs)
        self._cuda_graph_max_num_tokens = int(max_num_tokens)

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
        return None

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
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        # Most backends use 1 so kv_len computations remain valid under padding.
        return 1

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # FlexAttention does not require verify buffers (tree mask/positions) here.
        return None

    def _get_or_create_extend_block_mask(
        self,
        *,
        q_len: int,
        kv_len: int,
        q_kv_offset: int,
        block_size: int | tuple[int, int] = 128,
    ):
        # q_kv_offset is the absolute-position delta between q_idx=0 and kv_idx=0.
        # If abs_q0 is the absolute index of the first query token, and abs_kv0 is the
        # absolute index of kv_idx=0, then q_kv_offset = abs_q0 - abs_kv0.
        # Causal constraint becomes: kv_idx <= q_kv_offset + q_idx.
        key = ("extend", q_len, kv_len, int(q_kv_offset), block_size, str(self.device))
        cached = self._block_mask_cache.get(key)
        if cached is not None:
            return cached

        mask_mod = make_extend_causal_mask_mod(q_kv_offset=int(q_kv_offset))

        block_mask = create_block_mask(
            mask_mod,
            None,
            None,
            q_len,
            kv_len,
            device=str(self.device),
            BLOCK_SIZE=block_size,
            _compile=False,
        )
        self._block_mask_cache[key] = block_mask
        return block_mask

    def _run_flex_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        sinks: torch.Tensor | None = None,
        window_left: int = -1,
    ):
        """Run the extend forward by using torch flex attention op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        # Avoid repeated device syncs if lens tensors live on CUDA by making a single
        # CPU copy for the Python control-flow below.
        seq_lens_cpu = seq_lens
        extend_prefix_lens_cpu = extend_prefix_lens
        extend_seq_lens_cpu = extend_seq_lens
        try:
            if getattr(seq_lens, "is_cuda", False):
                seq_lens_cpu = seq_lens.to("cpu")
            if getattr(extend_prefix_lens, "is_cuda", False):
                extend_prefix_lens_cpu = extend_prefix_lens.to("cpu")
            if getattr(extend_seq_lens, "is_cuda", False):
                extend_seq_lens_cpu = extend_seq_lens.to("cpu")
        except Exception:
            pass

        start_q = 0
        for seq_idx in range(seq_lens_cpu.shape[0]):
            extend_seq_len_q = int(extend_seq_lens_cpu[seq_idx])
            prefill_seq_len_q = int(extend_prefix_lens_cpu[seq_idx])
            seq_len_kv = int(seq_lens_cpu[seq_idx])

            if extend_seq_len_q == 0:
                continue

            # By construction, extend_prefix_lens + extend_seq_lens == seq_lens.
            # q_idx=0 corresponds to absolute position prefill_seq_len_q.
            abs_q0 = prefill_seq_len_q
            abs_q_last = seq_len_kv - 1
            if abs_q0 + extend_seq_len_q != seq_len_kv:
                raise ValueError(
                    f"TorchFlexAttnBackend: unexpected extend lengths: "
                    f"prefix={prefill_seq_len_q}, extend={extend_seq_len_q}, seq_len={seq_len_kv}"
                )

            if not causal:
                raise NotImplementedError("Non-causal mode is not yet implemented.")

            # Apply sliding-window by truncating KV to the last window that any query token can see.
            wl = int(window_left)
            if wl >= 0:
                kv_abs_start = max(0, abs_q_last - wl)
            else:
                kv_abs_start = 0
            kv_abs_end = abs_q_last
            kv_len = kv_abs_end - kv_abs_start + 1
            q_kv_offset = abs_q0 - kv_abs_start

            end_q = start_q + extend_seq_len_q
            per_req_query = query[:, start_q:end_q, :]

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, kv_abs_start : kv_abs_end + 1]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            block_mask = self._get_or_create_extend_block_mask(
                q_len=int(extend_seq_len_q),
                kv_len=int(kv_len),
                q_kv_offset=int(q_kv_offset),
            )

            need_sinks = sinks is not None and sinks.numel() > 0
            if need_sinks:
                out, lse = self._flex(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=True,
                )
                out = apply_attention_sinks(out, lse, sinks)
            else:
                out = self._flex(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    block_mask=block_mask,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=False,
                )

            per_req_out = out.squeeze(0).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q
        return output

    def _run_flex_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        sinks: torch.Tensor | None = None,
        window_left: int = -1,
    ):
        """Run the decode forward by using torch flex attention op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        seq_lens_cpu = seq_lens
        req_pool_indices_cpu = req_pool_indices
        try:
            if getattr(seq_lens, "is_cuda", False):
                seq_lens_cpu = seq_lens.to("cpu")
            if getattr(req_pool_indices, "is_cuda", False):
                req_pool_indices_cpu = req_pool_indices.to("cpu")
        except Exception:
            pass

        start_q = 0
        for seq_idx in range(seq_lens_cpu.shape[0]):
            seq_len_q = 1
            seq_len_kv = int(seq_lens_cpu[seq_idx])
            end_q = start_q + seq_len_q

            per_req_query = query[:, start_q:end_q, :]

            # Apply sliding-window by truncating KV to the last window for this layer.
            wl = int(window_left)
            abs_qpos = seq_len_kv - 1
            if wl >= 0:
                kv_abs_start = max(0, abs_qpos - wl)
            else:
                kv_abs_start = 0
            kv_abs_end = abs_qpos

            req_pool_idx = req_pool_indices_cpu[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, kv_abs_start : kv_abs_end + 1]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            need_sinks = sinks is not None and sinks.numel() > 0
            if need_sinks:
                out, lse = self._flex(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    block_mask=None,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=True,
                )
                out = apply_attention_sinks(out, lse, sinks)
            else:
                out = self._flex(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    block_mask=None,
                    scale=scaling,
                    enable_gqa=enable_gqa,
                    return_lse=False,
                )

            per_req_out = out.squeeze(0).movedim(query.dim() - 2, 0)

            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            raise NotImplementedError(
                "TorchFlexAttnBackend does not support non-causal attention for now."
            )

        self._run_flex_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
            sinks=kwargs.get("sinks", None),
            window_left=int(getattr(layer, "sliding_window_size", -1)),
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_flex_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
            sinks=kwargs.get("sinks", None),
            window_left=int(getattr(layer, "sliding_window_size", -1)),
        )

        return o

    def support_triton(self):
        return False
