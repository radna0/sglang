# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Inference-only GptOss model compatible with HuggingFace weights."""

import logging
import math
import os
import time
from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.compilation.piecewise_context_manager import (
    get_forward_context,
    is_in_piecewise_cuda_graph,
)
from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import dequant_mxfp4
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import LazyValue, add_prefix, is_cuda, is_npu, make_layers
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()
_is_npu = is_npu()


class GptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


logger = logging.getLogger(__name__)


def _fa3_tensor_stats(tensor: torch.Tensor) -> dict:
    detached = tensor.detach()
    return {
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "contiguous": bool(detached.is_contiguous()),
        "finite": bool(torch.isfinite(detached).all().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.float().mean().item()),
        "std": float(detached.float().std(unbiased=False).item()),
    }


def _fa3_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@lru_cache(maxsize=None)
def _fa3_int_set(name: str) -> frozenset[int]:
    raw = os.environ.get(name, "")
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.add(int(part))
        except ValueError:
            logger.warning("[FA3SinkPolicy] ignoring invalid integer %r for %s", part, name)
    return frozenset(values)


def _fa3_trace_mlp_enabled(layer_id: int) -> bool:
    if not _fa3_flag("SGLANG_FA3_TRACE_MLP_DETAILS"):
        return False
    trace_layers = _fa3_int_set("SGLANG_FA3_TRACE_MLP_LAYER_IDS")
    return not trace_layers or layer_id in trace_layers


def _fa3_topk_output_stats(topk_output) -> dict:
    stats: dict[str, object] = {"format": type(topk_output).__name__}

    if hasattr(topk_output, "router_logits"):
        router_logits = getattr(topk_output, "router_logits")
        if isinstance(router_logits, torch.Tensor):
            stats["router_logits"] = _fa3_tensor_stats(router_logits)

    if hasattr(topk_output, "topk_weights"):
        topk_weights = getattr(topk_output, "topk_weights")
        if isinstance(topk_weights, torch.Tensor):
            weights_sum = topk_weights.sum(dim=-1)
            stats["topk_weights"] = _fa3_tensor_stats(topk_weights)
            stats["topk_weights_row_sum"] = _fa3_tensor_stats(weights_sum)

    if hasattr(topk_output, "topk_ids"):
        topk_ids = getattr(topk_output, "topk_ids")
        if isinstance(topk_ids, torch.Tensor):
            ids_detached = topk_ids.detach()
            stats["topk_ids_shape"] = tuple(ids_detached.shape)
            stats["topk_ids_min"] = int(ids_detached.min().item())
            stats["topk_ids_max"] = int(ids_detached.max().item())
            stats["topk_ids_unique"] = int(torch.unique(ids_detached).numel())
            stats["topk_ids_head"] = ids_detached[: min(4, ids_detached.shape[0])].cpu().tolist()

    return stats


@lru_cache(maxsize=1)
def _get_gpt_oss_moe_runtime():
    """Import the MoE stack lazily.

    GPT-OSS registration should not fail just because optional MoE backends
    (FlashInfer/TVM-FFI-adjacent paths) are unavailable at import time.
    """

    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopK

    return get_moe_a2a_backend, get_moe_impl_class, FusedMoE, TopK


@lru_cache(maxsize=1)
def _get_gpt_oss_communicator_runtime():
    """Import communicator pieces lazily for the same reason as MoE."""

    from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes

    return LayerCommunicator, LayerScatterModes


# Aligned with HF's implementation, using sliding window inclusive with the last token
# SGLang assumes exclusive
def get_attention_sliding_window_size(config):
    return config.sliding_window - 1


class GptOssSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        _, get_moe_impl_class, _, TopK = _get_gpt_oss_moe_runtime()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.gemm1_alpha = getattr(config, "hidden_act_alpha", 1.702)
        self.gemm1_clamp_limit = config.swiglu_limit

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            layer_id=layer_id,
        )

        self.top_k = config.num_experts_per_tok
        experts_type = get_moe_impl_class(quant_config)
        extra_kwargs = {}
        if experts_type.__name__ == "FusedMoE":
            quant_config_name = (
                quant_config.get_name() if quant_config is not None else None
            )
            extra_kwargs = {
                # for moe gate_up_proj and down_proj and their bias loading
                "use_weight_loader_fused": quant_config_name
                != "mxfp4"
            }

        self.experts = experts_type(
            num_experts=config.num_local_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            activation=self.activation,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            with_bias=True,
            prefix=add_prefix("experts", prefix),
            **extra_kwargs,
        )

        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=True,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
            params_dtype=config.torch_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        get_moe_a2a_backend, _, _, _ = _get_gpt_oss_moe_runtime()
        if not get_moe_a2a_backend().is_deepep():
            return self.forward_normal(hidden_states, should_allreduce_fusion)
        else:
            raise Exception("forward_deepep branch not implemented yet")

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        trace_mlp = _fa3_trace_mlp_enabled(self.layer_id)
        if trace_mlp:
            logger.info(
                "[FA3MLP] layer=%d input=%s",
                self.layer_id,
                _fa3_tensor_stats(hidden_states),
            )
        if is_in_piecewise_cuda_graph():
            final_hidden_states = moe_impl(self.layer_id, hidden_states)
        else:
            router_logits, _ = self.router(hidden_states)
            if trace_mlp:
                logger.info(
                    "[FA3MLP] layer=%d router_logits=%s",
                    self.layer_id,
                    _fa3_tensor_stats(router_logits),
                )
            topk_output = self.topk(hidden_states, router_logits)
            if trace_mlp:
                logger.info(
                    "[FA3MLP] layer=%d topk=%s",
                    self.layer_id,
                    _fa3_topk_output_stats(topk_output),
                )
            final_hidden_states = self.experts(hidden_states, topk_output)
            if trace_mlp:
                logger.info(
                    "[FA3MLP] layer=%d experts_out=%s",
                    self.layer_id,
                    _fa3_tensor_stats(final_hidden_states),
                )

        if self.tp_size > 1 and not should_allreduce_fusion:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
            if trace_mlp:
                logger.info(
                    "[FA3MLP] layer=%d allreduce_out=%s",
                    self.layer_id,
                    _fa3_tensor_stats(final_hidden_states),
                )

        ans = final_hidden_states.view(num_tokens, hidden_dim)
        if trace_mlp:
            logger.info(
                "[FA3MLP] layer=%d output=%s",
                self.layer_id,
                _fa3_tensor_stats(ans),
            )
        return ans


@register_custom_op(out_shape="hidden_states")
def moe_impl(layer_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
    forward_context = get_forward_context()
    moe_fusion = forward_context.moe_fusions[layer_id]
    router_logits, _ = moe_fusion.router(hidden_states)
    topk_output = moe_fusion.topk(hidden_states, router_logits)
    final_hidden_states = moe_fusion.experts(hidden_states, topk_output)
    return final_hidden_states


class GptOssAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window_size: int = -1,  # if -1, normal attention, else, window attention.
        layer_type: str = "",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.sliding_window_size = sliding_window_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        # Choose dtype of sinks based on attention backend: trtllm_mha requires float32,
        # others can use bfloat16
        attn_backend = get_global_server_args().attention_backend
        sinks_dtype = torch.float32 if attn_backend == "trtllm_mha" else torch.bfloat16
        self.sinks = nn.Parameter(
            torch.empty(self.num_heads, dtype=sinks_dtype), requires_grad=False
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            params_dtype=params_dtype,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        assert layer_type in {"sliding_attention", "full_attention"}
        self.layer_type = layer_type
        use_sliding_window = layer_type == "sliding_attention"
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=(sliding_window_size if use_sliding_window else -1),
        )
        self.layer_id = layer_id

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        fa3_trace_attn_details = _fa3_flag("SGLANG_FA3_TRACE_ATTN_DETAILS")
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        if fa3_trace_attn_details and self.layer_id >= 26:
            logger.info(
                "[FA3Attn] layer=%d qkv_proj=%s",
                self.layer_id,
                _fa3_tensor_stats(qkv),
            )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if fa3_trace_attn_details and self.layer_id >= 26:
            logger.info(
                "[FA3Attn] layer=%d q=%s k=%s v=%s sinks=%s",
                self.layer_id,
                _fa3_tensor_stats(q),
                _fa3_tensor_stats(k),
                _fa3_tensor_stats(v),
                _fa3_tensor_stats(self.sinks),
            )

        extra_args = {}
        if not _is_npu:
            extra_args = {
                "fused_set_kv_buffer_arg": (
                    create_fused_set_kv_buffer_arg(
                        value=v,
                        layer=self.attn,
                        forward_batch=forward_batch,
                    )
                    if enable_fused_set_kv_buffer(forward_batch)
                    else None
                ),
            }

        q, k = self.rotary_emb(positions, q, k, **extra_args)
        if fa3_trace_attn_details and self.layer_id >= 26:
            logger.info(
                "[FA3Attn] layer=%d after_rope q=%s k=%s",
                self.layer_id,
                _fa3_tensor_stats(q),
                _fa3_tensor_stats(k),
            )
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        fa3_trace_attn_details = _fa3_flag("SGLANG_FA3_TRACE_ATTN_DETAILS")
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        sink_policy = "normal"
        if _fa3_flag("SGLANG_GPTOSS_DISABLE_SINKS"):
            sinks = None
            sink_policy = "global_off"
        elif self.layer_id in _fa3_int_set("SGLANG_GPTOSS_DISABLE_SINKS_LAYER_IDS"):
            sinks = None
            sink_policy = "layer_off"
        elif (
            self.layer_type == "full_attention"
            and _fa3_flag("SGLANG_GPTOSS_DISABLE_SINKS_FULL_ATTENTION")
        ):
            sinks = None
            sink_policy = "full_attention_off"
        elif _fa3_flag("SGLANG_GPTOSS_CLAMP_SINKS_NONNEG"):
            sinks = self.sinks.clamp_min(0)
            sink_policy = "clamp_nonneg"
        else:
            sinks = self.sinks
        if _fa3_flag("SGLANG_FA3_TRACE_SINK_POLICY") and self.layer_id >= 24:
            logger.info(
                "[FA3SinkPolicy] layer=%d type=%s policy=%s sinks=%s",
                self.layer_id,
                self.layer_type,
                sink_policy,
                "None" if sinks is None else _fa3_tensor_stats(sinks),
            )
        attn_output = self.attn(
            *inner_state,
            sinks=sinks,
            save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
        )
        if fa3_trace_attn_details and self.layer_id >= 26:
            logger.info(
                "[FA3Attn] layer=%d attn_output=%s",
                self.layer_id,
                _fa3_tensor_stats(attn_output),
            )
        output, _ = self.o_proj(attn_output)
        if fa3_trace_attn_details and self.layer_id >= 26:
            logger.info(
                "[FA3Attn] layer=%d o_proj=%s",
                self.layer_id,
                _fa3_tensor_stats(output),
            )
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return self.forward_core(s)


class GptOssDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias

        if sliding_window_size is None:
            self.sliding_window_size = get_attention_sliding_window_size(self.config)
        else:
            self.sliding_window_size = sliding_window_size

        self.self_attn = GptOssAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=self.sliding_window_size,
            layer_type=config.layer_types[layer_id],
            params_dtype=config.torch_dtype,
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # GptOss all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        self.is_nextn = False
        is_previous_layer_sparse = True
        is_next_layer_sparse = True

        LayerCommunicator, LayerScatterModes = _get_gpt_oss_communicator_runtime()
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = GptOssSparseMoeBlock(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            raise NotImplementedError(
                "Dense MLP is not implemented for GptOssDecoderLayer. "
                "Please use GptOssSparseMoeBlock instead."
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            is_last_layer=(
                self.is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
            ),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fa3_sync_after_attn = _fa3_flag("SGLANG_FA3_SYNC_AFTER_ATTN")
        fa3_trace_layer_io = _fa3_flag("SGLANG_FA3_TRACE_LAYER_IO")
        fa3_sync_after_mlp = _fa3_flag("SGLANG_FA3_SYNC_AFTER_MLP")
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        if fa3_trace_layer_io:
            logger.info(
                "[FA3LayerIO] after_attn layer=%d stats=%s",
                self.layer_id,
                _fa3_tensor_stats(hidden_states),
            )
            if fa3_sync_after_attn and hidden_states.is_cuda:
                if torch.cuda.is_current_stream_capturing():
                    if fa3_trace_layer_io:
                        logger.info(
                            "[FA3LayerIO] sync_after_attn_skipped_capture layer=%d",
                            self.layer_id,
                        )
                else:
                    torch.cuda.synchronize(hidden_states.device)
                    if fa3_trace_layer_io:
                        logger.info(
                            "[FA3LayerIO] sync_after_attn_ok layer=%d", self.layer_id
                        )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        if fa3_trace_layer_io:
            logger.info(
                "[FA3LayerIO] before_mlp layer=%d hidden=%s residual=%s",
                self.layer_id,
                _fa3_tensor_stats(hidden_states),
                None if residual is None else _fa3_tensor_stats(residual),
            )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        skip_mlp_for_fa3_warmup = str(
            os.environ.get("SGLANG_FA3_WARMUP_SKIP_GPTOSS_MLP", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if skip_mlp_for_fa3_warmup:
            if self.layer_id == 0:
                logger.info(
                    "[FA3Warmup] skipping GPT-OSS MLP during FA3 precapture warmup."
                )
            should_allreduce_fusion = False
        else:
            hidden_states = self.mlp(
                hidden_states, forward_batch, should_allreduce_fusion
            )
        if fa3_trace_layer_io:
            logger.info(
                "[FA3LayerIO] after_mlp layer=%d stats=%s",
                self.layer_id,
                _fa3_tensor_stats(hidden_states),
            )
        if fa3_sync_after_mlp and hidden_states.is_cuda:
            if torch.cuda.is_current_stream_capturing():
                if fa3_trace_layer_io:
                    logger.info(
                        "[FA3LayerIO] sync_after_mlp_skipped_capture layer=%d",
                        self.layer_id,
                    )
            else:
                torch.cuda.synchronize(hidden_states.device)
                if fa3_trace_layer_io:
                    logger.info("[FA3LayerIO] sync_after_mlp_ok layer=%d", self.layer_id)

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True

        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
            if fa3_trace_layer_io:
                logger.info(
                    "[FA3LayerIO] after_postprocess layer=%d hidden=%s residual=%s",
                    self.layer_id,
                    _fa3_tensor_stats(hidden_states),
                    None if residual is None else _fa3_tensor_stats(residual),
                )

        return hidden_states, residual


class GptOssModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = GptOssDecoderLayer,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if _is_npu:
            config.hidden_act = "npu_swiglu_oai"

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to GptOssDecoderLayer
        decoder_layer_type = decoder_layer_type or GptOssDecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        fa3_trace_model_tail = _fa3_flag("SGLANG_FA3_TRACE_MODEL_TAIL")
        fa3_trace_first_bad_layer = _fa3_flag("SGLANG_FA3_TRACE_FIRST_BAD_LAYER")
        fa3_sync_after_model = _fa3_flag("SGLANG_FA3_SYNC_AFTER_MODEL")
        fa3_reported_bad_layer = False
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(hidden_states + residual)
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions, hidden_states, forward_batch, residual
                )
                if fa3_trace_first_bad_layer and not fa3_reported_bad_layer:
                    hidden_ok = bool(torch.isfinite(hidden_states).all().item())
                    residual_ok = (
                        True
                        if residual is None
                        else bool(torch.isfinite(residual).all().item())
                    )
                    if not hidden_ok or not residual_ok:
                        logger.info(
                            "[FA3BadLayer] layer=%s forward_mode=%s hidden_ok=%s residual_ok=%s hidden=%s residual=%s",
                            i,
                            int(forward_batch.forward_mode),
                            hidden_ok,
                            residual_ok,
                            _fa3_tensor_stats(hidden_states),
                            None if residual is None else _fa3_tensor_stats(residual),
                        )
                        fa3_reported_bad_layer = True
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
            if fa3_trace_model_tail:
                logger.info(
                    "[FA3ModelTail] after_norm %s",
                    _fa3_tensor_stats(hidden_states),
                )
            if fa3_sync_after_model and hidden_states.is_cuda:
                if torch.cuda.is_current_stream_capturing():
                    logger.info("[FA3ModelTail] sync_after_norm_skipped_capture")
                else:
                    torch.cuda.synchronize(hidden_states.device)
                    logger.info("[FA3ModelTail] sync_after_norm_ok")
        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states



class GptOssForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = GptOssModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            # quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, GptOssSparseMoeBlock)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        fa3_trace_model_tail = _fa3_flag("SGLANG_FA3_TRACE_MODEL_TAIL")
        fa3_sync_after_logits = _fa3_flag("SGLANG_FA3_SYNC_AFTER_LOGITS")
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            final_hidden_states = hidden_states
            if fa3_trace_model_tail:
                logger.info(
                    "[FA3ModelTail] before_logits %s",
                    _fa3_tensor_stats(hidden_states),
                )
                logger.info(
                    "[FA3ModelTail] entering_logits_processor cls=%s",
                    type(self.logits_processor).__name__,
                )
            logits = self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
            if (
                self.capture_aux_hidden_states
                and forward_batch.forward_mode.is_target_verify()
                and hasattr(logits, "__dict__")
            ):
                # DFLASH keeps aux hidden features in `logits_output.hidden_states` for
                # draft-KV append. Preserve the final LM-head-ready hidden states on a
                # private attribute so tree verify can bypass `next_token_logits` safely.
                logits._dflash_final_hidden_states = final_hidden_states
            if _fa3_flag("SGLANG_FA3_TRACE_TOP_LOGITS") and not isinstance(
                logits, torch.Tensor
            ):
                next_token_logits = getattr(logits, "next_token_logits", None)
                if (
                    isinstance(next_token_logits, torch.Tensor)
                    and next_token_logits.dim() == 2
                    and next_token_logits.shape[0] > 0
                    and next_token_logits.shape[1] > 0
                ):
                    row0 = next_token_logits[0]
                    topn = min(16, row0.shape[0])
                    top_vals, top_ids = torch.topk(row0, k=topn)
                    logger.info(
                        "[FA3TopLogits] ids=%s vals=%s argmax_id=%s argmax_val=%s",
                        top_ids.tolist(),
                        [float(x) for x in top_vals.tolist()],
                        int(top_ids[0].item()),
                        float(top_vals[0].item()),
                    )
            if fa3_trace_model_tail:
                if isinstance(logits, torch.Tensor):
                    logger.info(
                        "[FA3ModelTail] after_logits tensor shape=%s dtype=%s contiguous=%s",
                        tuple(logits.shape),
                        logits.dtype,
                        logits.is_contiguous(),
                    )
                else:
                    next_token_logits = getattr(logits, "next_token_logits", None)
                    logger.info(
                        "[FA3ModelTail] after_logits output_cls=%s next_token_logits=%s",
                        type(logits).__name__,
                        None
                        if next_token_logits is None
                        else {
                            "shape": tuple(next_token_logits.shape),
                            "dtype": str(next_token_logits.dtype),
                            "contiguous": bool(next_token_logits.is_contiguous()),
                        },
                    )
            sync_tensor = logits if isinstance(logits, torch.Tensor) else getattr(
                logits, "next_token_logits", None
            )
            if (
                fa3_sync_after_logits
                and isinstance(sync_tensor, torch.Tensor)
                and sync_tensor.is_cuda
            ):
                if torch.cuda.is_current_stream_capturing():
                    logger.info("[FA3ModelTail] sync_after_logits_skipped_capture")
                else:
                    torch.cuda.synchronize(sync_tensor.device)
                    logger.info("[FA3ModelTail] sync_after_logits_ok")
            return logits
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer


    def _get_default_weight_mapping(self):
        """Generate default weight name mapping for GptOss safetensors."""
        weight_mapping = {}

        # Map router weights to gate
        weight_mapping["embedding.weight"] = "model.embed_tokens.weight"
        weight_mapping["unembedding.weight"] = "lm_head.weight"
        weight_mapping["norm.scale"] = "model.norm.weight"
        for layer_id in range(self.config.num_hidden_layers):
            weight_mapping[f"block.{layer_id}.attn.q_proj.weight"] = (
                f"model.layers.{layer_id}.self_attn.q_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.q_proj.bias"] = (
                f"model.layers.{layer_id}.self_attn.q_proj.bias"
            )

            weight_mapping[f"block.{layer_id}.attn.k_proj.weight"] = (
                f"model.layers.{layer_id}.self_attn.k_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.k_proj.bias"] = (
                f"model.layers.{layer_id}.self_attn.k_proj.bias"
            )

            weight_mapping[f"block.{layer_id}.attn.v_proj.weight"] = (
                f"model.layers.{layer_id}.self_attn.v_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.v_proj.bias"] = (
                f"model.layers.{layer_id}.self_attn.v_proj.bias"
            )

            weight_mapping[f"block.{layer_id}.attn.out.weight"] = (
                f"model.layers.{layer_id}.self_attn.o_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.out.bias"] = (
                f"model.layers.{layer_id}.self_attn.o_proj.bias"
            )
            weight_mapping[f"block.{layer_id}.attn.sinks"] = (
                f"model.layers.{layer_id}.self_attn.sinks"
            )
            weight_mapping[f"block.{layer_id}.attn.norm.scale"] = (
                f"model.layers.{layer_id}.input_layernorm.weight"
            )

            weight_mapping[f"block.{layer_id}.mlp.gate.weight"] = (
                f"model.layers.{layer_id}.mlp.router.weight"
            )
            weight_mapping[f"block.{layer_id}.mlp.gate.bias"] = (
                f"model.layers.{layer_id}.mlp.router.bias"
            )
            weight_mapping[f"block.{layer_id}.mlp.norm.scale"] = (
                f"model.layers.{layer_id}.post_attention_layernorm.weight"
            )
            weight_mapping[f"block.{layer_id}.mlp.experts.gate_up_proj"] = (
                f"model.layers.{layer_id}.mlp.experts.gate_up_proj"
            )
            weight_mapping[f"block.{layer_id}.mlp.gate_up_proj_bias"] = (
                f"model.layers.{layer_id}.mlp.experts.gate_up_proj_bias"
            )
            weight_mapping[f"block.{layer_id}.mlp.down_proj"] = (
                f"model.layers.{layer_id}.mlp.experts.mlp2_weight"
            )
            weight_mapping[f"block.{layer_id}.mlp.down_proj_bias"] = (
                f"model.layers.{layer_id}.mlp.experts.mlp2_bias"
            )

        return weight_mapping

    # TODO beautify code
    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
        weight_name_mapping: dict = None,
    ):
        quant_config_name = (
            self.quant_config.get_name() if self.quant_config is not None else None
        )
        if quant_config_name != "mxfp4":
            self._load_normal_weights(
                weights, is_nextn=is_nextn, weight_name_mapping=weight_name_mapping
            )
        else:
            self._load_weights_mxfp4(
                weights, is_nextn=is_nextn, weight_name_mapping=weight_name_mapping
            )

    def _load_weights_mxfp4(self, weights, is_nextn, weight_name_mapping):
        mxfp4_weights = []
        normal_weights = []
        partition_log_progress = (
            os.environ.get("SGLANG_GPT_OSS_NORMAL_PROGRESS", "").strip().lower()
            in ("1", "true", "yes", "on")
        ) or (
            os.environ.get("SGLANG_GPT_OSS_MXFP4_PROGRESS", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        partition_every = int(
            os.environ.get("SGLANG_GPT_OSS_NORMAL_PROGRESS_EVERY", "32") or "32"
        )
        partition_start = time.perf_counter()
        total_weights = 0

        for total_weights, (name, weight) in enumerate(weights, start=1):
            if (
                ".experts" in name
                and self.quant_config is not None
                and self.quant_config.get_name() == "mxfp4"
            ):
                mxfp4_weights.append((name, weight))
            else:
                normal_weights.append((name, weight))
            if partition_log_progress and (
                total_weights == 1
                or (partition_every > 0 and total_weights % partition_every == 0)
            ):
                logging.info(
                    "GPT-OSS load partition progress: %d total mxfp4=%d normal=%d last=%s elapsed_s=%.2f",
                    total_weights,
                    len(mxfp4_weights),
                    len(normal_weights),
                    name,
                    time.perf_counter() - partition_start,
                )

        if partition_log_progress:
            logging.info(
                "GPT-OSS load partition end: total=%d mxfp4=%d normal=%d elapsed_s=%.2f",
                total_weights,
                len(mxfp4_weights),
                len(normal_weights),
                time.perf_counter() - partition_start,
            )

        mxfp4_loaded_params = self._load_mxfp4_experts_weights(mxfp4_weights)
        self._load_normal_weights(
            normal_weights,
            is_nextn=is_nextn,
            weight_name_mapping=weight_name_mapping,
            other_loaded_param_names=mxfp4_loaded_params,
        )

    def _load_mxfp4_experts_weights(self, weights):

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        mxfp4_block = 32
        log_progress = (
            os.environ.get("SGLANG_GPT_OSS_MXFP4_PROGRESS", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        eager_cuda = (
            os.environ.get("SGLANG_GPT_OSS_MXFP4_TRANSFER_MODE", "eager_cuda")
            .strip()
            .lower()
            != "direct_copy"
        )
        progress_every = int(
            os.environ.get("SGLANG_GPT_OSS_MXFP4_PROGRESS_EVERY", "4") or "4"
        )

        moe_tp_rank = get_moe_tensor_parallel_rank()
        moe_tp_size = get_moe_tensor_parallel_world_size()
        moe_ep_rank = get_moe_expert_parallel_rank()
        moe_ep_size = get_moe_expert_parallel_world_size()

        intermediate_size = self.config.intermediate_size
        assert (
            intermediate_size % mxfp4_block == 0
        ), f"{intermediate_size=} must be divisible by {mxfp4_block=}"
        intermediate_size_block = intermediate_size // mxfp4_block

        per_rank_intermediate_size_block = math.ceil(
            intermediate_size_block / moe_tp_size
        )

        per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

        # Calculate common slicing bounds for current rank
        assert self.config.num_local_experts % moe_ep_size == 0
        moe_num_global_experts = self.config.num_local_experts
        moe_num_local_experts = self.config.num_local_experts // moe_ep_size

        moe_tp_rank_start = moe_tp_rank * per_rank_intermediate_size
        moe_tp_rank_end = min(
            (moe_tp_rank + 1) * per_rank_intermediate_size, intermediate_size
        )

        moe_ep_rank_start = moe_ep_rank * moe_num_local_experts
        moe_ep_rank_end = (moe_ep_rank + 1) * moe_num_local_experts

        total_weights = len(weights)
        total_bytes = sum(weight.numel() * weight.element_size() for _, weight in weights)
        bytes_done = 0
        transfer_s_total = 0.0
        loader_s_total = 0.0
        overall_start = time.perf_counter()

        if log_progress:
            logging.info(
                "GPT-OSS MXFP4 load begin: tensors=%d total_gb=%.3f tp=%d ep=%d transfer_mode=%s",
                total_weights,
                total_bytes / (1024**3),
                moe_tp_size,
                moe_ep_size,
                "eager_cuda" if eager_cuda else "direct_copy",
            )

        for idx, (name, weight) in enumerate(weights, start=1):
            weight_bytes = weight.numel() * weight.element_size()
            transfer_start = time.perf_counter()
            if eager_cuda and weight.device.type != "cuda":
                weight = weight.cuda(non_blocking=False)
                if log_progress:
                    torch.cuda.synchronize(weight.device)
            transfer_s_total += time.perf_counter() - transfer_start

            if "gate_up_proj_blocks" in name:
                # Handle MLP gate and up projection weights
                new_name = name.replace("gate_up_proj_blocks", "w13_weight")

                # flat weight from (E, 2 * N, block_size, entry_per_block)
                # to (E, 2 * N, -1), shouldn't trigger copy for contiguous
                weight = weight.view(
                    moe_num_global_experts, 2 * intermediate_size, -1
                ).contiguous()

                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
                    ...,
                ]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loader_start = time.perf_counter()
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
                if log_progress:
                    torch.cuda.synchronize(param.device)
                loader_s_total += time.perf_counter() - loader_start
                loaded_params.add(new_name)

            elif "down_proj_blocks" in name:
                # Handle MLP down projection weights
                new_name = name.replace("down_proj_blocks", "w2_weight")
                # same flatten here, but since 2 mx4 value are packed in 1
                # uint8, divide by 2
                weight = weight.view(
                    moe_num_global_experts, -1, intermediate_size // 2
                ).contiguous()
                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    ...,
                    moe_tp_rank_start // 2 : moe_tp_rank_end // 2,
                ]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loader_start = time.perf_counter()
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
                if log_progress:
                    torch.cuda.synchronize(param.device)
                loader_s_total += time.perf_counter() - loader_start
                loaded_params.add(new_name)

            elif "gate_up_proj_scales" in name:
                # Handle MLP gate and up projection weights scale
                new_name = name.replace("gate_up_proj_scales", "w13_weight_scale")
                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
                    ...,
                ]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loader_start = time.perf_counter()
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
                if log_progress:
                    torch.cuda.synchronize(param.device)
                loader_s_total += time.perf_counter() - loader_start
                loaded_params.add(new_name)

            elif "down_proj_scales" in name:
                # Handle MLP down projection weights
                new_name = name.replace("down_proj_scales", "w2_weight_scale")
                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    ...,
                    moe_tp_rank_start // mxfp4_block : moe_tp_rank_end // mxfp4_block,
                ]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loader_start = time.perf_counter()
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
                if log_progress:
                    torch.cuda.synchronize(param.device)
                loader_s_total += time.perf_counter() - loader_start
                loaded_params.add(new_name)
            elif "gate_up_proj_bias" in name:
                # Handle MLP gate and up projection biases
                new_name = name.replace("gate_up_proj_bias", "w13_weight_bias")

                narrow_weight = weight[
                    moe_ep_rank_start:moe_ep_rank_end,
                    2 * moe_tp_rank_start : 2 * moe_tp_rank_end,
                ]

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loader_start = time.perf_counter()
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
                if log_progress:
                    torch.cuda.synchronize(param.device)
                loader_s_total += time.perf_counter() - loader_start
                loaded_params.add(new_name)

            elif "down_proj_bias" in name:
                narrow_weight = weight[moe_ep_rank_start:moe_ep_rank_end, ...]
                if moe_tp_rank != 0:
                    narrow_weight = torch.zeros_like(narrow_weight)

                # Handle MLP down projection bias
                new_name = name.replace("down_proj_bias", "w2_weight_bias")
                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loader_start = time.perf_counter()
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
                if log_progress:
                    torch.cuda.synchronize(param.device)
                loader_s_total += time.perf_counter() - loader_start
                loaded_params.add(new_name)

            bytes_done += weight_bytes
            if log_progress and (
                idx == 1 or idx == total_weights or (progress_every > 0 and idx % progress_every == 0)
            ):
                logging.info(
                    "GPT-OSS MXFP4 load progress: %d/%d last=%s bytes_gb=%.3f/%.3f transfer_s=%.2f loader_s=%.2f elapsed_s=%.2f",
                    idx,
                    total_weights,
                    name,
                    bytes_done / (1024**3),
                    total_bytes / (1024**3),
                    transfer_s_total,
                    loader_s_total,
                    time.perf_counter() - overall_start,
                )

        if log_progress:
            logging.info(
                "GPT-OSS MXFP4 load end: tensors=%d total_gb=%.3f transfer_s=%.2f loader_s=%.2f elapsed_s=%.2f",
                total_weights,
                total_bytes / (1024**3),
                transfer_s_total,
                loader_s_total,
                time.perf_counter() - overall_start,
            )

        return loaded_params

    def _load_normal_weights(
        self,
        weights,
        is_nextn: bool,
        weight_name_mapping: dict,
        other_loaded_param_names=[],
    ):
        log_progress = (
            os.environ.get("SGLANG_GPT_OSS_NORMAL_PROGRESS", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        progress_every = int(
            os.environ.get("SGLANG_GPT_OSS_NORMAL_PROGRESS_EVERY", "32") or "32"
        )
        phase_start = time.perf_counter()
        tp_rank = get_tensor_model_parallel_rank()
        if is_nextn:
            logging.warning(
                "Loading weights for nextn is currently not supported in GptOssForCausalLM. "
            )
            return
        weights = _canonicalize_weights(self.config, weights)
        weights = sorted(weights, key=lambda x: x[0])  # Sort by name for consistency
        if log_progress:
            logging.info(
                "GPT-OSS normal load begin: canonicalized=%d elapsed_s=%.2f",
                len(weights),
                time.perf_counter() - phase_start,
            )

        new_weights = []
        for name, p in weights:
            if "qkv.weight" in name:
                q_proj, k_proj, v_proj = p.split(
                    [
                        self.config.num_attention_heads * self.config.head_dim,
                        self.config.num_key_value_heads * self.config.head_dim,
                        self.config.num_key_value_heads * self.config.head_dim,
                    ],
                    dim=0,
                )
                new_weights.append(
                    (f"{name.replace('qkv.weight', 'q_proj.weight')}", q_proj)
                )
                new_weights.append(
                    (f"{name.replace('qkv.weight', 'k_proj.weight')}", k_proj)
                )
                new_weights.append(
                    (f"{name.replace('qkv.weight', 'v_proj.weight')}", v_proj)
                )
            elif "qkv.bias" in name:
                q_bias, k_bias, v_bias = p.split(
                    [
                        self.config.num_attention_heads * self.config.head_dim,
                        self.config.num_key_value_heads * self.config.head_dim,
                        self.config.num_key_value_heads * self.config.head_dim,
                    ],
                    dim=0,
                )
                new_weights.append(
                    (f"{name.replace('qkv.bias', 'q_proj.bias')}", q_bias)
                )
                new_weights.append(
                    (f"{name.replace('qkv.bias', 'k_proj.bias')}", k_bias)
                )
                new_weights.append(
                    (f"{name.replace('qkv.bias', 'v_proj.bias')}", v_bias)
                )
            else:
                new_weights.append((name, p))
        weights = new_weights
        if log_progress:
            logging.info(
                "GPT-OSS normal load after_qkv_split=%d elapsed_s=%.2f",
                len(weights),
                time.perf_counter() - phase_start,
            )

        # Use provided weight name mapping if available, otherwise use default
        if weight_name_mapping is None:
            weight_name_mapping = self._get_default_weight_mapping()
        else:
            # Merge with default mapping
            default_mapping = self._get_default_weight_mapping()
            default_mapping.update(weight_name_mapping)
            weight_name_mapping = default_mapping

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        _, _, FusedMoE, _ = _get_gpt_oss_moe_runtime()
        expert_params_mapping = FusedMoE.make_expert_params_mapping_fused(
            ckpt_gate_up_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_gate_up_proj_bias_name="gate_up_proj_bias",
            ckpt_down_proj_bias_name="down_proj_bias",
        )

        params_dict = dict(self.named_parameters())

        total_weights = len(weights)
        for idx, (name, loaded_weight) in enumerate(weights, start=1):
            loaded_weight = _WeightCreator.maybe_materialize(loaded_weight)

            # Apply weight name mapping if provided
            if weight_name_mapping and name in weight_name_mapping:
                name = weight_name_mapping[name]

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    if "bias" not in name:
                        loaded_weight = loaded_weight.transpose(-2, -1)
                    if "w2_weight_bias" in name and get_moe_tensor_parallel_rank() != 0:
                        loaded_weight = loaded_weight.zero_()

                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    if name in params_dict.keys():
                        param = params_dict[name]
                        if "sinks" in name:
                            start = get_attention_tp_rank() * param.numel()
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )
                        else:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

            if log_progress and (
                idx == 1
                or idx == total_weights
                or (progress_every > 0 and idx % progress_every == 0)
            ):
                logging.info(
                    "GPT-OSS normal load progress: %d/%d last=%s elapsed_s=%.2f",
                    idx,
                    total_weights,
                    name,
                    time.perf_counter() - phase_start,
                )
        if log_progress:
            logging.info(
                "GPT-OSS normal load end: total=%d elapsed_s=%.2f",
                total_weights,
                time.perf_counter() - phase_start,
            )

        # Fallback for FP8 KV Cache: initialize no-scale metadata to 1.0.
        # RadixAttention always defines these attributes as None, so hasattr()
        # is not a valid initialization check here.
        device = next(self.parameters()).device
        for layer_idx in range(self.config.num_hidden_layers):
            layer = self.model.layers[layer_idx]
            attn = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attn is not None:
                if getattr(attn, "k_scale", None) is None:
                    attn.k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
                if getattr(attn, "v_scale", None) is None:
                    attn.v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
                if getattr(attn, "k_scale_float", None) is None:
                    attn.k_scale_float = 1.0
                if getattr(attn, "v_scale_float", None) is None:
                    attn.v_scale_float = 1.0

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        logger.warning(
            "Ignoring KV scale file for GPT-OSS FP8 path: %s. "
            "The current GPT-OSS KV-scale loader is disabled because it is "
            "numerically unstable in the validated DFlash FP8 serving path.",
            quantization_param_path,
        )
        return

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [val + 1 for val in layer_ids]

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)


def _canonicalize_weights(config, weights_in: Iterable[Tuple[str, torch.Tensor]]):
    weights_out_dict = dict(weights_in)

    for layer_id in range(config.num_hidden_layers):
        for name_chunk in ["mlp1_weight", "mlp2_weight"]:
            name_prefix = f"block.{layer_id}.mlp.{name_chunk}"
            w_blocks = weights_out_dict.pop(f"{name_prefix}.blocks", None)
            w_scales = weights_out_dict.pop(f"{name_prefix}.scales", None)
            if w_blocks is not None:
                weights_out_dict[name_prefix] = _WeightCreator(
                    partial(
                        _dequant_mlp_weight,
                        debug_name=name_prefix,
                        w_blocks=w_blocks,
                        w_scales=w_scales,
                    )
                )

    return list(weights_out_dict.items())


def _dequant_mlp_weight(debug_name, w_blocks, w_scales):
    if get_tensor_model_parallel_rank() == 0:
        logger.info(f"Dequantize {debug_name} start")

    original_device = w_blocks.device

    w_blocks = w_blocks.cuda()
    w_scales = w_scales.cuda()

    w_bf16 = dequant_mxfp4(w_block=w_blocks, w_scale=w_scales, out_dtype=torch.bfloat16)
    w_bf16 = w_bf16.transpose(-2, -1).contiguous()

    if get_tensor_model_parallel_rank() == 0:
        logger.info(
            f"Dequantize {debug_name} end {w_blocks.shape=} {w_scales.shape=} {w_bf16.shape=}"
        )

    return w_bf16.to(original_device)


class _WeightCreator:
    def __init__(self, fn):
        self._fn = fn

    @staticmethod
    def maybe_materialize(obj):
        if isinstance(obj, _WeightCreator):
            output = obj._fn()
            obj._fn = None
            return output

        return obj


EntryClass = GptOssForCausalLM
