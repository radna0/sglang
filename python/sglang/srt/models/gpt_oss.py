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
from collections.abc import Iterable
from functools import partial
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
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
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
from sglang.srt.model_loader.weight_utils import default_weight_loader, kv_cache_scales_loader
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import LazyValue, add_prefix, is_npu, make_layers
from sglang.srt.utils.custom_op import register_custom_op

_is_npu = is_npu()


class GptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


logger = logging.getLogger(__name__)


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
        if is_in_piecewise_cuda_graph():
            final_hidden_states = moe_impl(self.layer_id, hidden_states)
        else:
            router_logits, _ = self.router(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            final_hidden_states = self.experts(hidden_states, topk_output)

        if self.tp_size > 1 and not should_allreduce_fusion:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        ans = final_hidden_states.view(num_tokens, hidden_dim)
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
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

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
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(
            *inner_state,
            sinks=self.sinks,
            save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
        )
        output, _ = self.o_proj(attn_output)
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


class GptOssMLAAttention(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: Optional[int] = None,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window_size: int = -1,
        layer_type: str = "",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.sliding_window_size = sliding_window_size
        self.qk_nope_head_dim = int(getattr(config, "qk_nope_head_dim"))
        self.qk_rope_head_dim = int(getattr(config, "qk_rope_head_dim"))
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(getattr(config, "v_head_dim"))
        self.total_mla_rope_num_kv_heads = int(
            getattr(config, "mla_rope_num_kv_heads", 1)
        )

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        if self.total_mla_rope_num_kv_heads < 1:
            raise ValueError(
                f"mla_rope_num_kv_heads must be >= 1, got {self.total_mla_rope_num_kv_heads}"
            )
        self.use_native_mha_anchor = bool(
            getattr(config, "mla_attention_mode", None) == "mha_explicit"
            or self.total_mla_rope_num_kv_heads > 1
        )
        if self.total_num_heads % self.total_mla_rope_num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} is not divisible by "
                f"mla_rope_num_kv_heads={self.total_mla_rope_num_kv_heads}"
            )
        self.query_heads_per_rope_kv = (
            self.total_num_heads // self.total_mla_rope_num_kv_heads
        )
        self.global_q_head_start = attn_tp_rank * self.num_heads
        self.global_q_head_end = self.global_q_head_start + self.num_heads
        if self.use_native_mha_anchor and (
            self.global_q_head_start % self.query_heads_per_rope_kv != 0
            or self.global_q_head_end % self.query_heads_per_rope_kv != 0
        ):
            raise ValueError(
                "GPT-OSS native MLA MHA-mode requires attention TP shards to align "
                "with rope-KV head groups."
            )
        if self.use_native_mha_anchor:
            self.local_rope_head_start = (
                self.global_q_head_start // self.query_heads_per_rope_kv
            )
            self.local_rope_head_end = (
                self.global_q_head_end // self.query_heads_per_rope_kv
            )
            self.mla_rope_num_kv_heads = (
                self.local_rope_head_end - self.local_rope_head_start
            )
        else:
            self.local_rope_head_start = 0
            self.local_rope_head_end = 1
            self.mla_rope_num_kv_heads = 1
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        if kv_lora_rank is None:
            kv_lora_rank = int(getattr(config, "kv_lora_rank"))
        self.kv_lora_rank = int(kv_lora_rank)
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.qk_head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            params_dtype=params_dtype,
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            self.kv_lora_rank
            + self.total_mla_rope_num_kv_heads * self.qk_rope_head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            params_dtype=params_dtype,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            params_dtype=params_dtype,
        )

        attn_backend = get_global_server_args().attention_backend
        sinks_dtype = torch.float32 if attn_backend == "trtllm_mha" else torch.bfloat16
        self.sinks = nn.Parameter(
            torch.zeros(self.num_heads, dtype=sinks_dtype), requires_grad=False
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            params_dtype=params_dtype,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = (
            get_rope(
                self.qk_rope_head_dim,
                rotary_dim=self.qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
            if self.qk_rope_head_dim > 0
            else None
        )

        assert layer_type in {"sliding_attention", "full_attention"}
        use_sliding_window = layer_type == "sliding_attention"
        if self.use_native_mha_anchor:
            self.attn = RadixAttention(
                self.num_heads,
                self.qk_head_dim,
                self.scaling,
                num_kv_heads=self.num_heads,
                layer_id=layer_id,
                prefix=add_prefix("attn", prefix),
                v_head_dim=self.v_head_dim,
                sliding_window_size=(
                    sliding_window_size if use_sliding_window else -1
                ),
            )
        else:
            self.attn = RadixAttention(
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
                self.scaling,
                num_kv_heads=1,
                layer_id=layer_id,
                prefix=add_prefix("attn", prefix),
                v_head_dim=self.kv_lora_rank,
                sliding_window_size=(
                    sliding_window_size if use_sliding_window else -1
                ),
            )

        self.w_kc = None
        self.w_vc = None

    def _ensure_postprocessed_weights(self) -> None:
        if self.w_kc is not None and self.w_vc is not None:
            return
        if not hasattr(self, "kv_b_proj"):
            raise RuntimeError(
                "GptOssMLAAttention weights not post-processed and kv_b_proj is unavailable."
            )
        w_kc, w_vc = self.kv_b_proj.weight.unflatten(
            0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.w_kc = w_kc.contiguous()
        self.w_vc = w_vc.contiguous().transpose(1, 2)
        del self.kv_b_proj

    def _prepare_q_pe(
        self, positions: torch.Tensor, q_pe: torch.Tensor, k_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_emb is None:
            return q_pe, k_pe
        if q_pe.shape[0] == 0:
            return q_pe, k_pe
        q_shape = q_pe.shape
        k_shape = k_pe.shape
        q_pe, k_pe = self.rotary_emb(
            positions,
            q_pe.reshape(-1, q_shape[1] * q_shape[2]),
            k_pe.reshape(-1, k_shape[1] * k_shape[2]),
        )
        return q_pe.view(q_shape), k_pe.view(k_shape)

    def _expand_local_k_rope(self, k_rope: torch.Tensor) -> torch.Tensor:
        if self.mla_rope_num_kv_heads == self.num_heads:
            return k_rope
        repeat_factor = self.num_heads // self.mla_rope_num_kv_heads
        return k_rope.repeat_interleave(repeat_factor, dim=1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        self._ensure_postprocessed_weights()

        q = self.q_proj(hidden_states)[0].view(-1, self.num_heads, self.qk_head_dim)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        if self.use_native_mha_anchor:
            kv_latent = latent_cache[..., : self.kv_lora_rank]
            q_nope = q[..., : self.qk_nope_head_dim]
            q_pe = q[..., self.qk_nope_head_dim :]
            k_pe_all = latent_cache[..., self.kv_lora_rank :]
            k_pe_all = k_pe_all.view(
                -1,
                self.total_mla_rope_num_kv_heads,
                self.qk_rope_head_dim,
            )
            k_pe = k_pe_all[
                :, self.local_rope_head_start : self.local_rope_head_end, :
            ].contiguous()

            if self.qk_rope_head_dim > 0:
                q_pe, k_pe = self._prepare_q_pe(positions, q_pe, k_pe)
                k_pe = self._expand_local_k_rope(k_pe)

            k_nope = torch.einsum("tr,hdr->thd", kv_latent, self.w_kc)
            value_states = torch.einsum("tr,hrv->thv", kv_latent, self.w_vc)
            key_states = (
                torch.cat((k_nope, k_pe), dim=-1)
                if self.qk_rope_head_dim > 0
                else k_nope
            )
            query_states = (
                torch.cat((q_nope, q_pe), dim=-1)
                if self.qk_rope_head_dim > 0
                else q_nope
            )

            attn_dtype = getattr(
                getattr(forward_batch, "token_to_kv_pool", None), "dtype", None
            )
            if attn_dtype is not None and query_states.dtype != attn_dtype:
                query_states = query_states.to(attn_dtype)
                key_states = key_states.to(attn_dtype)
                value_states = value_states.to(attn_dtype)

            attn_output = self.attn(
                query_states.flatten(1, 2),
                key_states,
                value_states,
                forward_batch,
                sinks=self.sinks,
            )
            output, _ = self.o_proj(attn_output)
            return output

        q_input = hidden_states.new_zeros(
            hidden_states.shape[0],
            self.num_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        v_input = latent_cache[..., : self.kv_lora_rank].unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)

        if self.qk_nope_head_dim > 0:
            q_nope = q[..., : self.qk_nope_head_dim]
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
            q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

        if self.qk_rope_head_dim > 0:
            q_pe = q[..., self.qk_nope_head_dim :]
            k_pe = k_input[..., self.kv_lora_rank :].clone()
            q_pe, k_pe = self._prepare_q_pe(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe

        attn_dtype = getattr(getattr(forward_batch, "token_to_kv_pool", None), "dtype", None)
        if attn_dtype is not None and q_input.dtype != attn_dtype:
            q_input = q_input.to(attn_dtype)

        attn_output = self.attn(
            q_input,
            k_input,
            v_input,
            forward_batch,
            sinks=self.sinks,
        )
        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        if attn_output.dtype != self.w_vc.dtype:
            attn_output = attn_output.to(self.w_vc.dtype)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)
        return output


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
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        hidden_states = self.mlp(hidden_states, forward_batch, should_allreduce_fusion)

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True

        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


class GptOssMLADecoderLayer(GptOssDecoderLayer):
    def __init__(
        self,
        config: GptOssConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window_size: int | None = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            sliding_window_size=sliding_window_size,
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        kv_rank_schedule = getattr(config, "kv_lora_rank_per_layer", None)
        if isinstance(kv_rank_schedule, (list, tuple)) and len(kv_rank_schedule) == int(
            getattr(config, "num_hidden_layers", 0) or 0
        ):
            kv_lora_rank = int(kv_rank_schedule[int(layer_id)])
        else:
            kv_lora_rank = int(config.kv_lora_rank)

        self.self_attn = GptOssMLAAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            kv_lora_rank=kv_lora_rank,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            attention_bias=config.attention_bias,
            # Mirror the stock GPT-OSS attention path: dense attention projections are
            # instantiated as regular linears even when expert weights use MXFP4.
            # Passing the MXFP4 quant config into these MLA-specific projections hits an
            # unsupported LinearBase quant path during model construction.
            quant_config=None,
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=self.sliding_window_size,
            layer_type=config.layer_types[layer_id],
            params_dtype=config.torch_dtype,
        )


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
        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state.
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            layer = self.layers[layer_idx]
            if isinstance(layer, nn.Identity) or isinstance(layer, PPMissingLayer):
                # Not a local layer on this PP rank.
                continue

            layer_self_attn = getattr(layer, "self_attn", None)
            if layer_self_attn is None or not hasattr(layer_self_attn, "attn"):
                raise RuntimeError(
                    f"Layer {layer_idx} has no self_attn.attn; cannot set KV scales."
                )

            attn = layer_self_attn.attn
            if not hasattr(attn, "k_scale"):
                raise RuntimeError(
                    "Self attention has no KV cache scaling factor attribute!"
                )

            # FA3 backend expects k_scale/v_scale to be a Tensor so it can `.expand(...)`.
            # Keep *_float for backends that read a Python float.
            scaling_factor = float(scaling_factor)
            attn.k_scale_float = scaling_factor
            attn.v_scale_float = scaling_factor

            device = next(layer_self_attn.parameters()).device
            scale_t = torch.tensor(scaling_factor, dtype=torch.float32, device=device)
            attn.k_scale = scale_t
            attn.v_scale = scale_t


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
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

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

        for name, weight in weights:
            if (
                ".experts" in name
                and self.quant_config is not None
                and self.quant_config.get_name() == "mxfp4"
            ):
                mxfp4_weights.append((name, weight))
            else:
                normal_weights.append((name, weight))

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

        for name, weight in weights:
            weight = weight.cuda()

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
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
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
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
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
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
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
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
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
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(new_name)

            elif "down_proj_bias" in name:
                narrow_weight = weight[moe_ep_rank_start:moe_ep_rank_end, ...]
                if moe_tp_rank != 0:
                    narrow_weight = torch.zeros_like(narrow_weight)

                # Handle MLP down projection bias
                new_name = name.replace("down_proj_bias", "w2_weight_bias")
                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=new_name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(new_name)

        return loaded_params

    def _load_normal_weights(
        self,
        weights,
        is_nextn: bool,
        weight_name_mapping: dict,
        other_loaded_param_names=[],
    ):
        tp_rank = get_tensor_model_parallel_rank()
        if is_nextn:
            logging.warning(
                "Loading weights for nextn is currently not supported in GptOssForCausalLM. "
            )
            return
        weights = _canonicalize_weights(self.config, weights)
        weights = sorted(weights, key=lambda x: x[0])  # Sort by name for consistency

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
        expert_params_mapping = FusedMoE.make_expert_params_mapping_fused(
            ckpt_gate_up_proj_name="gate_up_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_gate_up_proj_bias_name="gate_up_proj_bias",
            ckpt_down_proj_bias_name="down_proj_bias",
        )

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
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

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state.
    #
    # Note: SGLang's FA3 FP8 path expects `attn.k_scale` / `attn.v_scale` to be
    # tensors (so `.expand()` works). The QuantParamSchema stores Python floats,
    # so we materialize 0-dim float32 tensors on the model device.
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        device = next(self.parameters()).device

        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if layer_idx < self.model.start_layer or layer_idx >= self.model.end_layer:
                continue
            layer = self.model.layers[layer_idx]
            attn = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attn is None:
                continue

            scale_f = float(scaling_factor)
            scale_t = torch.tensor(scale_f, dtype=torch.float32, device=device)

            for name in ("k_scale", "v_scale"):
                cur = getattr(attn, name, None)
                if isinstance(cur, torch.nn.Parameter):
                    cur.data.copy_(scale_t)  # 0-dim
                elif isinstance(cur, torch.Tensor):
                    try:
                        cur.copy_(scale_t)
                    except Exception:
                        setattr(attn, name, scale_t)
                else:
                    setattr(attn, name, scale_t)

            attn.k_scale_float = scale_f
            attn.v_scale_float = scale_f

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


class GptOssMlaForCausalLM(GptOssForCausalLM):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = GptOssModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            decoder_layer_type=GptOssMLADecoderLayer,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
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

    def _get_legacy_weight_mapping(self):
        weight_mapping = {
            "embedding.weight": "model.embed_tokens.weight",
            "unembedding.weight": "lm_head.weight",
            "norm.scale": "model.norm.weight",
        }
        for layer_id in range(self.config.num_hidden_layers):
            weight_mapping[f"block.{layer_id}.attn.q_proj.weight"] = (
                f"model.layers.{layer_id}.self_attn.q_proj.weight"
            )
            weight_mapping[f"block.{layer_id}.attn.q_proj.bias"] = (
                f"model.layers.{layer_id}.self_attn.q_proj.bias"
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

    def _maybe_load_absorbed_weight(
        self, name: str, loaded_weight: torch.Tensor
    ) -> bool:
        if not name.startswith("model.layers.") or ".self_attn." not in name:
            return False
        if not (name.endswith(".self_attn.w_kc") or name.endswith(".self_attn.w_vc")):
            return False

        parts = name.split(".")
        if len(parts) < 5:
            return False
        layer_id = int(parts[2])
        if (
            hasattr(self.model, "start_layer")
            and (
                layer_id < self.model.start_layer
                or layer_id >= self.model.end_layer
            )
        ):
            return True

        layer = self.model.layers[layer_id]
        if isinstance(layer, nn.Identity) or isinstance(layer, PPMissingLayer):
            return True

        self_attn = layer.self_attn
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        if loaded_weight.shape[0] % attn_tp_size != 0:
            raise ValueError(
                f"Absorbed weight {name} has incompatible head dimension {loaded_weight.shape[0]} "
                f"for attention tp_size={attn_tp_size}."
            )
        heads_per_rank = loaded_weight.shape[0] // attn_tp_size
        start = attn_tp_rank * heads_per_rank
        end = start + heads_per_rank
        target = loaded_weight[start:end].to(
            device=self_attn.q_proj.weight.device,
            dtype=self_attn.q_proj.weight.dtype,
        )

        if name.endswith(".self_attn.w_kc"):
            self_attn.w_kc = target.contiguous()
        else:
            self_attn.w_vc = target.contiguous()
        return True

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
        weight_name_mapping: dict = None,
    ):
        if is_nextn:
            logging.warning(
                "Loading weights for nextn is currently not supported in GptOssMlaForCausalLM."
            )
            return

        weights = _canonicalize_weights(self.config, weights)
        weights = sorted(weights, key=lambda x: x[0])
        params_dict = dict(self.named_parameters())
        legacy_mapping = self._get_legacy_weight_mapping()
        if weight_name_mapping:
            legacy_mapping.update(weight_name_mapping)

        for name, loaded_weight in weights:
            loaded_weight = _WeightCreator.maybe_materialize(loaded_weight)

            if self._maybe_load_absorbed_weight(name, loaded_weight):
                continue

            if "qkv.weight" in name:
                q_proj, _, _ = loaded_weight.split(
                    [
                        self.config.num_attention_heads * self.config.head_dim,
                        self.config.num_key_value_heads * self.config.head_dim,
                        self.config.num_key_value_heads * self.config.head_dim,
                    ],
                    dim=0,
                )
                loaded_weight = q_proj
                name = name.replace("qkv.weight", "q_proj.weight")
            elif "qkv.bias" in name:
                q_bias, _, _ = loaded_weight.split(
                    [
                        self.config.num_attention_heads * self.config.head_dim,
                        self.config.num_key_value_heads * self.config.head_dim,
                        self.config.num_key_value_heads * self.config.head_dim,
                    ],
                    dim=0,
                )
                loaded_weight = q_bias
                name = name.replace("qkv.bias", "q_proj.bias")

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

            target_name = None
            if name in params_dict:
                target_name = name
            elif name in legacy_mapping:
                target_name = legacy_mapping[name]

            if target_name is None or target_name not in params_dict:
                continue

            param = params_dict[target_name]
            if target_name.endswith("sinks"):
                start = get_attention_tp_rank() * param.numel()
                param.data.copy_(loaded_weight[start : start + param.numel()])
                continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def post_load_weights(self):
        for layer_id in range(self.config.num_hidden_layers):
            layer = self.model.layers[layer_id]
            if isinstance(layer, nn.Identity) or isinstance(layer, PPMissingLayer):
                continue
            self_attn = layer.self_attn
            if self_attn.w_kc is not None and self_attn.w_vc is not None:
                if hasattr(self_attn, "kv_b_proj"):
                    del self_attn.kv_b_proj
                continue
            if not hasattr(self_attn, "kv_b_proj"):
                continue
            w_kc, w_vc = self_attn.kv_b_proj.weight.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.contiguous()
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
            del self_attn.kv_b_proj


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


EntryClass = [GptOssForCausalLM, GptOssMlaForCausalLM]
