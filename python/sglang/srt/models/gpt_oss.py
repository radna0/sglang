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

import json
import logging
import math
import os
from collections.abc import Iterable
from functools import partial
from pathlib import Path
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
from sglang.srt.model_loader.weight_utils import default_weight_loader
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


_GPTOSS_DEEPSEEK_LATENT_KV_CKPT_INDEX_ENV = (
    "SGLANG_GPTOSS_DEEPSEEK_LATENT_KV_CKPT_INDEX_JSON"
)

_GPTOSS_DEEPSEEK_LATENT_KV_MLA_ENV = "SGLANG_GPTOSS_DEEPSEEK_LATENT_KV_MLA"
_GPTOSS_DEEPSEEK_LATENT_KV_MLA_RANK_ENV = "SGLANG_GPTOSS_DEEPSEEK_LATENT_KV_MLA_RANK"
_GPTOSS_DEEPSEEK_LATENT_KV_MLA_ROPE_DIM_ENV = "SGLANG_GPTOSS_DEEPSEEK_LATENT_KV_MLA_ROPE_DIM"
_GPTOSS_DEEPSEEK_MLA_ROPEK_CKPT_INDEX_ENV = "SGLANG_GPTOSS_DEEPSEEK_MLA_ROPEK_CKPT_INDEX_JSON"


def _load_deepseek_latent_kv_ckpt_index(path: str) -> dict[int, str]:
    """
    Load a DeepSeek-native latent-KV checkpoint index JSON.

    Expected format (matching `kaggle_scripts/train_deepseek_sharedown_end2end.py`):
      { "<layer_idx>": "<ckpt_path>", ... }

    Paths may be absolute (e.g. `/models/out/.../deepseek_latent_kv.pt`) or relative
    to the index file's directory.
    """

    p = str(path or "").strip()
    if not p:
        raise ValueError("empty ckpt-index path")
    raw = json.loads(Path(p).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ckpt-index.json must be a JSON object mapping layer_idx -> ckpt_path")

    base_dir = Path(p).expanduser().resolve().parent
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            li = int(k)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Bad ckpt index key: {k!r}") from e
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Bad ckpt index value for layer {li}: {v!r}")
        v_path = Path(v).expanduser()
        if not v_path.is_absolute():
            v_path = (base_dir / v_path).resolve()
        out[li] = str(v_path)
    return out


def _load_ropek_ckpt_index(path: str) -> dict[int, str]:
    """
    Load a RoPE-K synthesis checkpoint index JSON.

    Expected format:
      { "<layer_idx>": "<ckpt_path>", ... }

    Paths may be absolute (e.g. `/models/out/.../rope_k_head_synth.pt`) or relative
    to the index file's directory.
    """

    p = str(path or "").strip()
    if not p:
        raise ValueError("empty ropek ckpt-index path")
    raw = json.loads(Path(p).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ropek ckpt-index.json must be a JSON object mapping layer_idx -> ckpt_path")

    base_dir = Path(p).expanduser().resolve().parent
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            li = int(k)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Bad ropek ckpt index key: {k!r}") from e
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Bad ropek ckpt path for layer {li}: {v!r}")
        vv = str(v).strip()
        if not (vv.startswith("/") or (len(vv) >= 2 and vv[1] == ":")):
            vv = str(base_dir / vv)
        out[int(li)] = vv
    return out


def _bool_env(name: str) -> bool:
    v = os.getenv(str(name), "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _infer_deepseek_latent_kv_rank_from_ckpt_index(idx_path: str) -> int:
    idx = _load_deepseek_latent_kv_ckpt_index(idx_path)
    if not idx:
        raise ValueError("ckpt-index.json is empty")

    ckpt_path = idx[min(idx.keys())]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Bad DeepSeek latent-KV ckpt at {ckpt_path!r} (expected dict)")

    k_lat = int(ckpt.get("k_latent_rank", 0) or 0)
    v_lat = int(ckpt.get("v_latent_rank", 0) or 0)
    if k_lat <= 0 or v_lat <= 0:
        raise ValueError(
            f"Bad latent ranks in {ckpt_path!r}: k_latent_rank={k_lat} v_latent_rank={v_lat}"
        )
    if k_lat != v_lat:
        raise ValueError(
            f"Latent rank mismatch in {ckpt_path!r}: k_latent_rank={k_lat} v_latent_rank={v_lat}"
        )
    return k_lat


def _slice_kv_heads_for_attn_tp(
    t: torch.Tensor,
    *,
    total_kv_heads: int,
    attn_tp_rank: int,
    attn_tp_size: int,
) -> torch.Tensor:
    """
    Match QKVParallelLinear's KV head partitioning/replication behavior:

    - if attn_tp_size >= total_kv_heads: KV heads are replicated, each rank holds 1 head
    - else: KV heads are partitioned contiguously across ranks
    """

    total_kv_heads = int(total_kv_heads)
    attn_tp_rank = int(attn_tp_rank)
    attn_tp_size = int(attn_tp_size)
    if total_kv_heads <= 0:
        raise ValueError(f"bad total_kv_heads={total_kv_heads}")
    if attn_tp_size <= 0:
        raise ValueError(f"bad attn_tp_size={attn_tp_size}")

    if attn_tp_size >= total_kv_heads:
        replicas = attn_tp_size // total_kv_heads
        if replicas <= 0 or (attn_tp_size % total_kv_heads) != 0:
            raise ValueError(
                f"KV replication requires attn_tp_size divisible by total_kv_heads "
                f"(got attn_tp_size={attn_tp_size} total_kv_heads={total_kv_heads})"
            )
        head_id = attn_tp_rank // replicas
        if head_id < 0 or head_id >= total_kv_heads:  # pragma: no cover
            raise ValueError(f"bad KV head_id={head_id} for rank={attn_tp_rank}")
        return t.narrow(0, int(head_id), 1).contiguous()

    if (total_kv_heads % attn_tp_size) != 0:
        raise ValueError(
            f"KV partition requires total_kv_heads divisible by attn_tp_size "
            f"(got total_kv_heads={total_kv_heads} attn_tp_size={attn_tp_size})"
        )
    heads_per_rank = total_kv_heads // attn_tp_size
    start = attn_tp_rank * heads_per_rank
    return t.narrow(0, int(start), int(heads_per_rank)).contiguous()


class _DeepSeekLowRankPreconditioner(nn.Module):
    """
    Optional low-rank preconditioner used by our DeepSeek-native shared-down K/V modules.

      x0 = x * scale
      x' = x0 + (x0 @ v) @ u^T
    """

    def __init__(self, *, hidden_size: int, rank: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.scale = nn.Parameter(torch.ones(self.hidden_size), requires_grad=False)
        self.v = nn.Parameter(torch.zeros(self.hidden_size, self.rank), requires_grad=False)
        self.u = nn.Parameter(torch.zeros(self.hidden_size, self.rank), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x * self.scale
        xr = torch.matmul(x0, self.v)
        xlr = torch.matmul(xr, self.u.t())
        return x0 + xlr


class _DeepSeekLatentPerHeadProj2D(nn.Module):
    """
    DeepSeek-native per-head K/V projection from a shared latent space.

    This is the "Stage-1 injection" module: it produces standard-shaped K/V tensors
    (flattened) so we can run inference with the existing RadixAttention + KV cache.

    It does NOT change the KV cache format to store compressed latents. That is Stage-2.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_kv_heads: int,
        head_dim: int,
        latent_rank: int,
        bias: bool,
        precond: nn.Module | None = None,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.latent_rank = int(latent_rank)
        self.precond = precond
        self.down = nn.Linear(self.hidden_size, self.latent_rank, bias=False)
        self.up = nn.Parameter(
            torch.empty(self.num_kv_heads, self.latent_rank, self.head_dim),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(torch.empty(self.num_kv_heads, self.head_dim), requires_grad=False)
            if bool(bias)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, hidden]
        if self.precond is not None:
            x = self.precond(x)
        z = self.down(x)  # [N, r]
        y = torch.einsum("nr,hrd->nhd", z, self.up)  # [N, Hkv, D]
        if self.bias is not None:
            y = y + self.bias.view(1, self.num_kv_heads, self.head_dim)
        return y.reshape(int(x.shape[0]), self.num_kv_heads * self.head_dim)


class _DeepSeekSharedRopeKFromHeadSynth(nn.Module):
    """
    Shared RoPE-K generator for GPT-OSS MLA conversion.

    We consume checkpoints produced by `kaggle_scripts/train_rope_k_head_synth.py`.
    That script trains a per-KV-head generator (base + delta). For DeepSeek-style MLA we need
    a *single shared* RoPE-K vector, so we take the mean over KV heads:
      k_shared = mean_h ( base(z) + delta_h(z) )  = (base + mean_h delta_h)(z)

    This module implements that shared mapping:
      z = down(x)
      k_shared = up(z)
    """

    def __init__(self, *, hidden_size: int, latent_rank: int, rope_dim: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_rank = int(latent_rank)
        self.rope_dim = int(rope_dim)
        self.down = nn.Linear(self.hidden_size, self.latent_rank, bias=False)
        self.up = nn.Linear(self.latent_rank, self.rope_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


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
        self._deepseek_latent_kv_mla = _bool_env(_GPTOSS_DEEPSEEK_LATENT_KV_MLA_ENV)
        self._deepseek_mla_kv_lora_rank: int | None = None
        self._deepseek_mla_rope_head_dim: int | None = None
        self._deepseek_mla_rotary_emb = None
        if self._deepseek_latent_kv_mla:
            rank_s = os.getenv(_GPTOSS_DEEPSEEK_LATENT_KV_MLA_RANK_ENV, "").strip()
            kv_lora_rank = int(rank_s) if rank_s else 0
            if kv_lora_rank <= 0:
                idx_path = os.getenv(_GPTOSS_DEEPSEEK_LATENT_KV_CKPT_INDEX_ENV, "").strip()
                if not idx_path:
                    raise RuntimeError(
                        "MLA mode enabled but no ckpt-index was provided. Set "
                        f"{_GPTOSS_DEEPSEEK_LATENT_KV_CKPT_INDEX_ENV} or "
                        f"{_GPTOSS_DEEPSEEK_LATENT_KV_MLA_RANK_ENV}."
                    )
                kv_lora_rank = _infer_deepseek_latent_kv_rank_from_ckpt_index(idx_path)
            rope_s = os.getenv(_GPTOSS_DEEPSEEK_LATENT_KV_MLA_ROPE_DIM_ENV, "").strip()
            rope_dim = int(rope_s) if rope_s else int(self.head_dim)
            if rope_dim <= 0:
                raise RuntimeError(f"bad rope_dim={rope_dim} for GPT-OSS MLA mode")
            self._deepseek_mla_kv_lora_rank = int(kv_lora_rank)
            self._deepseek_mla_rope_head_dim = int(rope_dim)
            # DeepSeek MLA uses decoupled RoPE on only the "rope" slice (qk_rope_head_dim).
            # GPT-OSS uses full-head RoPE, so we keep the base params but build a rope-only
            # embedding for the MLA path.
            self._deepseek_mla_rotary_emb = get_rope(
                int(rope_dim),
                rotary_dim=int(rope_dim),
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
            # IMPORTANT (FlashMLA correctness):
            # In MLA mode, the attention kernel consumes concatenated (q_nope_out, q_rope) where
            # head_dim_k = kv_lora_rank + rope_dim (e.g. 512 + 64 = 576). The softmax scaling must
            # match this effective dot-product dimension; using the original GPT-OSS head_dim (64)
            # over-scales logits and can collapse decoding (e.g. repetitive "1.1.1..." outputs).
            mla_head_dim = int(kv_lora_rank) + int(rope_dim)
            mla_scaling = float(mla_head_dim**-0.5)
            self.attn = RadixAttention(
                self.num_heads,
                mla_head_dim,
                mla_scaling,
                # For DeepSeek-style MLA we store a single shared latent KV (+ shared RoPE key)
                # and let Q/O modules provide head/group specificity.
                num_kv_heads=1,
                layer_id=layer_id,
                v_head_dim=int(kv_lora_rank),
                prefix=add_prefix("attn_mla", prefix),
                sliding_window_size=(sliding_window_size if use_sliding_window else -1),
            )
        else:
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
        # Optional DeepSeek-native latent KV patch modules (loaded after HF weights).
        self._deepseek_latent_k: _DeepSeekLatentPerHeadProj2D | None = None
        self._deepseek_latent_v: _DeepSeekLatentPerHeadProj2D | None = None
        self._deepseek_q: nn.Linear | None = None
        self._deepseek_o: nn.Linear | None = None
        self._deepseek_mla_ropek: _DeepSeekSharedRopeKFromHeadSynth | None = None
        self._deepseek_mla_qkc: torch.Tensor | None = None
        # Optional CARE decoupled-RoPE healing add-on (Q_rope / K_rope projections).
        self._deepseek_rope_q: nn.Linear | None = None
        self._deepseek_rope_k: nn.Linear | None = None
        self._deepseek_rope_dim: int = 0
        self._deepseek_rope_k_shared: bool = False

    def enable_deepseek_mla_ropek_patch(
        self,
        ckpt: Dict[str, Any],
        *,
        params_dtype: torch.dtype,
    ) -> None:
        """
        Attach a shared RoPE-K synthesis module for DeepSeek-style MLA KV cache.

        Expected checkpoint format: `kaggle_scripts/train_rope_k_head_synth.py` output
        (`rope_k_head_synth.pt`). We support `arch=linear` only.
        """

        if not isinstance(ckpt, dict):
            raise TypeError("ckpt must be a dict (torch.load result)")
        state_dict = ckpt.get("state_dict", None)
        if not isinstance(state_dict, dict):
            raise ValueError("ckpt missing state_dict")

        arch = str(ckpt.get("arch", "linear") or "linear").strip().lower()
        if arch != "linear":
            raise ValueError(f"Unsupported ropek synth arch={arch!r} (only 'linear' is supported)")

        hidden = int(ckpt.get("hidden_size", 0) or 0) or int(self.hidden_size)
        rope_dim = int(ckpt.get("head_dim", 0) or 0) or int(self.head_dim)
        r = int(ckpt.get("latent_rank", 0) or 0)
        if hidden <= 0 or rope_dim <= 0 or r <= 0:
            raise ValueError(f"bad ropek meta: hidden={hidden} rope_dim={rope_dim} latent_rank={r}")

        down_w = state_dict.get("down.weight", None)
        base_w = state_dict.get("base.weight", None)
        delta_w = state_dict.get("delta.weight", None)
        if not isinstance(down_w, torch.Tensor) or not isinstance(base_w, torch.Tensor) or not isinstance(delta_w, torch.Tensor):
            raise ValueError("ropek ckpt missing down.weight/base.weight/delta.weight tensors")

        if down_w.ndim != 2 or base_w.ndim != 2 or delta_w.ndim != 2:
            raise ValueError(
                f"ropek ckpt weight dims: down={tuple(down_w.shape)} base={tuple(base_w.shape)} delta={tuple(delta_w.shape)}"
            )
        if int(down_w.shape[0]) != r or int(down_w.shape[1]) != hidden:
            raise ValueError(f"ropek down.weight shape mismatch: got {tuple(down_w.shape)} expected ({r}, {hidden})")
        if int(base_w.shape[0]) != rope_dim or int(base_w.shape[1]) != r:
            raise ValueError(f"ropek base.weight shape mismatch: got {tuple(base_w.shape)} expected ({rope_dim}, {r})")
        if int(delta_w.shape[1]) != r:
            raise ValueError(f"ropek delta.weight shape mismatch: got {tuple(delta_w.shape)} expected (*, {r})")
        if int(delta_w.shape[0]) % rope_dim != 0:
            raise ValueError(f"ropek delta.weight first dim not divisible by rope_dim: {int(delta_w.shape[0])} % {rope_dim} != 0")
        kvh = int(delta_w.shape[0]) // int(rope_dim)
        if kvh <= 0:
            raise ValueError(f"bad kvh inferred from delta.weight: kvh={kvh}")

        # Build a shared RoPE-K mapping by averaging per-KV-head deltas.
        delta_mean = delta_w.view(kvh, rope_dim, r).mean(dim=0)  # [rope_dim, r]
        up_w = base_w + delta_mean

        device = self.sinks.device
        mod = _DeepSeekSharedRopeKFromHeadSynth(hidden_size=hidden, latent_rank=r, rope_dim=rope_dim).to(
            device=device, dtype=params_dtype
        )
        with torch.no_grad():
            mod.down.weight.copy_(down_w.to(device=device, dtype=params_dtype))
            mod.up.weight.copy_(up_w.to(device=device, dtype=params_dtype))
        mod.requires_grad_(False)
        self._deepseek_mla_ropek = mod

    def enable_deepseek_latent_kv_patch(
        self,
        ckpt: Dict[str, Any],
        *,
        attn_tp_rank: int,
        attn_tp_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        """
        Attach DeepSeek-native latent K/V modules to this layer (inference-only).

        The checkpoint format matches `kaggle_scripts/train_deepseek_sharedown_end2end.py`'s
        `deepseek_latent_kv.pt`.
        """

        if not isinstance(ckpt, dict):
            raise TypeError("ckpt must be a dict (torch.load result)")
        state_dict = ckpt.get("state_dict", None)
        if not isinstance(state_dict, dict):
            raise ValueError("ckpt missing state_dict")
        k_state = state_dict.get("k")
        v_state = state_dict.get("v")
        if not isinstance(k_state, dict) or not isinstance(v_state, dict):
            raise ValueError("ckpt missing state_dict['k'] or state_dict['v']")

        ckpt_hidden = int(ckpt.get("hidden_size", -1) or -1)
        ckpt_head_dim = int(ckpt.get("head_dim", -1) or -1)
        ckpt_total_kv_heads = int(ckpt.get("num_kv_heads", -1) or -1)
        if ckpt_hidden != int(self.hidden_size):
            raise ValueError(f"ckpt hidden_size={ckpt_hidden} != model hidden_size={self.hidden_size}")
        if ckpt_head_dim != int(self.head_dim):
            raise ValueError(f"ckpt head_dim={ckpt_head_dim} != model head_dim={self.head_dim}")
        if ckpt_total_kv_heads != int(self.total_num_kv_heads):
            raise ValueError(
                f"ckpt num_kv_heads={ckpt_total_kv_heads} != model num_kv_heads={self.total_num_kv_heads}"
            )

        device = self.sinks.device
        share_down = bool(ckpt.get("share_down", False))
        k_lat = int(ckpt.get("k_latent_rank", 0) or 0)
        v_lat = int(ckpt.get("v_latent_rank", 0) or 0)
        if k_lat <= 0 or v_lat <= 0:
            raise ValueError(f"bad latent ranks: k_latent_rank={k_lat} v_latent_rank={v_lat}")

        # Build optional preconditioner if present in the checkpoint state dict.
        has_precond = any(str(k).startswith("precond.") for k in k_state.keys())
        precond: _DeepSeekLowRankPreconditioner | None = None
        if has_precond:
            v_t = k_state.get("precond.v", None)
            u_t = k_state.get("precond.u", None)
            if not isinstance(v_t, torch.Tensor) or not isinstance(u_t, torch.Tensor):
                raise ValueError("ckpt has precond.* keys but missing precond.v/precond.u tensors")
            precond_rank = int(v_t.shape[1])
            precond = _DeepSeekLowRankPreconditioner(hidden_size=self.hidden_size, rank=precond_rank).to(
                device=device, dtype=params_dtype
            )

        # Localize K/V per-head up/bias tensors for attention TP.
        def _localize_latent_state(st: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = dict(st)
            up = out.get("up", None)
            if isinstance(up, torch.Tensor):
                out["up"] = _slice_kv_heads_for_attn_tp(
                    up, total_kv_heads=ckpt_total_kv_heads, attn_tp_rank=attn_tp_rank, attn_tp_size=attn_tp_size
                )
            b = out.get("bias", None)
            if isinstance(b, torch.Tensor):
                out["bias"] = _slice_kv_heads_for_attn_tp(
                    b, total_kv_heads=ckpt_total_kv_heads, attn_tp_rank=attn_tp_rank, attn_tp_size=attn_tp_size
                )
            # Move/cast tensors.
            for k, v in list(out.items()):
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device=device, dtype=params_dtype)
            return out

        k_has_bias = bool(ckpt.get("k_has_bias", False))
        v_has_bias = bool(ckpt.get("v_has_bias", False))

        k_mod = _DeepSeekLatentPerHeadProj2D(
            hidden_size=self.hidden_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            latent_rank=k_lat,
            bias=k_has_bias,
            precond=precond,
        ).to(device=device, dtype=params_dtype)

        v_mod = _DeepSeekLatentPerHeadProj2D(
            hidden_size=self.hidden_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            latent_rank=v_lat,
            bias=v_has_bias,
            precond=(precond if share_down else None),
        ).to(device=device, dtype=params_dtype)

        # Load K strictly (must match).
        incompat_k = k_mod.load_state_dict(
            _localize_latent_state(k_state),
            strict=False,
        )
        if incompat_k.unexpected_keys:
            raise ValueError(f"unexpected keys in K state dict: {incompat_k.unexpected_keys}")
        if incompat_k.missing_keys:
            raise ValueError(f"missing keys in K state dict: {incompat_k.missing_keys}")
        # V: when share_down, we alias down+precond to K's module and ignore duplicate weights.
        if share_down:
            v_mod.down = k_mod.down
            v_mod.precond = k_mod.precond
            v_local = _localize_latent_state(v_state)
            for kk in list(v_local.keys()):
                if str(kk).startswith("down.") or str(kk).startswith("precond."):
                    v_local.pop(kk, None)
            incompat_v = v_mod.load_state_dict(v_local, strict=False)
            if incompat_v.unexpected_keys:
                raise ValueError(f"unexpected keys in V state dict: {incompat_v.unexpected_keys}")
            bad_missing = [
                k for k in incompat_v.missing_keys if not (str(k).startswith("down.") or str(k).startswith("precond."))
            ]
            if bad_missing:
                raise ValueError(f"missing keys in V state dict: {bad_missing}")
        else:
            incompat_v = v_mod.load_state_dict(_localize_latent_state(v_state), strict=False)
            if incompat_v.unexpected_keys:
                raise ValueError(f"unexpected keys in V state dict: {incompat_v.unexpected_keys}")
            if incompat_v.missing_keys:
                raise ValueError(f"missing keys in V state dict: {incompat_v.missing_keys}")

        # Optional Q/O full-linears (rarely needed for init-only runs, but supported).
        q_mod: nn.Linear | None = None
        if bool(ckpt.get("patch_qproj", False)) and isinstance(state_dict.get("q"), dict):
            q_state = state_dict["q"]
            if isinstance(q_state.get("weight"), torch.Tensor):
                w_full = q_state["weight"]
                start = int(attn_tp_rank) * int(self.q_size)
                w_local = w_full.narrow(0, start, int(self.q_size)).contiguous()
                b_full = q_state.get("bias", None)
                b_local = (
                    b_full.narrow(0, start, int(self.q_size)).contiguous()
                    if isinstance(b_full, torch.Tensor)
                    else None
                )
                q_mod = nn.Linear(self.hidden_size, self.q_size, bias=b_local is not None).to(
                    device=device, dtype=params_dtype
                )
                with torch.no_grad():
                    q_mod.weight.copy_(w_local.to(device=device, dtype=params_dtype))
                    if b_local is not None and q_mod.bias is not None:
                        q_mod.bias.copy_(b_local.to(device=device, dtype=params_dtype))

        o_mod: nn.Linear | None = None
        if bool(ckpt.get("patch_oproj", False)) and isinstance(state_dict.get("o"), dict):
            o_state = state_dict["o"]
            if isinstance(o_state.get("weight"), torch.Tensor):
                w_full = o_state["weight"]  # [hidden, total_q]
                start = int(attn_tp_rank) * int(self.q_size)
                w_local = w_full.narrow(1, start, int(self.q_size)).contiguous()
                b_full = o_state.get("bias", None)
                # Match RowParallelLinear bias semantics: only rank0 applies bias.
                use_bias = bool(isinstance(b_full, torch.Tensor) and int(attn_tp_rank) == 0)
                o_mod = nn.Linear(self.q_size, self.hidden_size, bias=use_bias).to(
                    device=device, dtype=params_dtype
                )
                with torch.no_grad():
                    o_mod.weight.copy_(w_local.to(device=device, dtype=params_dtype))
                    if use_bias and o_mod.bias is not None and isinstance(b_full, torch.Tensor):
                        o_mod.bias.copy_(b_full.to(device=device, dtype=params_dtype))

        qkc_w: torch.Tensor | None = None
        qkc_state = state_dict.get("qkc")
        if isinstance(qkc_state, dict) and isinstance(qkc_state.get("weight"), torch.Tensor):
            qkc_w = qkc_state["weight"]
            qkc_w = _slice_kv_heads_for_attn_tp(
                qkc_w, total_kv_heads=ckpt_total_kv_heads, attn_tp_rank=attn_tp_rank, attn_tp_size=attn_tp_size
            ).to(device=device, dtype=params_dtype)

        # Optional decoupled RoPE healing modules (train_deepseek_sharedown_end2end.py).
        rope_q: nn.Linear | None = None
        rope_k: nn.Linear | None = None
        rope_dim_ckpt = int(ckpt.get("decoupled_rope_dim", 0) or 0)
        rope_k_shared = False
        if rope_dim_ckpt > 0 and isinstance(state_dict.get("rope_q"), dict) and isinstance(state_dict.get("rope_k"), dict):
            rq_state = state_dict.get("rope_q", {})
            rk_state = state_dict.get("rope_k", {})
            wq_full = rq_state.get("weight")
            wk_full = rk_state.get("weight")
            bq_full = rq_state.get("bias")
            bk_full = rk_state.get("bias")
            if not isinstance(wq_full, torch.Tensor) or wq_full.ndim != 2:
                raise ValueError("ckpt rope_q missing weight")
            if not isinstance(wk_full, torch.Tensor) or wk_full.ndim != 2:
                raise ValueError("ckpt rope_k missing weight")

            # rope_q is stored as [total_heads*dr, hidden] (head-major); slice by attention TP rank.
            dr = int(rope_dim_ckpt)
            if int(wq_full.shape[0]) != int(self.total_num_heads) * int(dr) or int(wq_full.shape[1]) != int(self.hidden_size):
                raise ValueError(
                    f"rope_q.weight shape mismatch: got {tuple(wq_full.shape)} expected ({int(self.total_num_heads)*int(dr)}, {int(self.hidden_size)})"
                )
            local_heads = int(self.num_heads)
            start_head = int(attn_tp_rank) * int(local_heads)
            wq_local = torch.empty((local_heads * dr, int(self.hidden_size)), dtype=wq_full.dtype, device=wq_full.device)
            bq_local: torch.Tensor | None = None
            if isinstance(bq_full, torch.Tensor):
                if bq_full.ndim != 1 or int(bq_full.shape[0]) != int(self.total_num_heads) * int(dr):
                    raise ValueError(
                        f"rope_q.bias shape mismatch: got {tuple(bq_full.shape)} expected ({int(self.total_num_heads)*int(dr)},)"
                    )
                bq_local = torch.empty((local_heads * dr,), dtype=bq_full.dtype, device=bq_full.device)
            for hidx in range(local_heads):
                src0 = (start_head + hidx) * dr
                dst0 = hidx * dr
                wq_local[dst0 : dst0 + dr].copy_(wq_full[src0 : src0 + dr])
                if bq_local is not None:
                    bq_local[dst0 : dst0 + dr].copy_(bq_full[src0 : src0 + dr])
            rope_q = nn.Linear(int(self.hidden_size), int(local_heads) * int(dr), bias=bq_local is not None).to(
                device=device, dtype=params_dtype
            )
            with torch.no_grad():
                rope_q.weight.copy_(wq_local.to(device=device, dtype=params_dtype))
                if bq_local is not None and rope_q.bias is not None:
                    rope_q.bias.copy_(bq_local.to(device=device, dtype=params_dtype))
            rope_q.requires_grad_(False)

            # rope_k is either shared [dr, hidden] or per-KV-head [total_kv_heads*dr, hidden].
            if int(wk_full.shape[1]) != int(self.hidden_size):
                raise ValueError(f"rope_k.weight hidden mismatch: got {tuple(wk_full.shape)} hidden={int(self.hidden_size)}")
            if int(wk_full.shape[0]) == int(dr):
                rope_k_shared = True
                use_bias = bool(isinstance(bk_full, torch.Tensor))
                if use_bias:
                    if bk_full.ndim != 1 or int(bk_full.shape[0]) != int(dr):
                        raise ValueError(f"rope_k.bias shape mismatch: got {tuple(bk_full.shape)} expected ({dr},)")
                rope_k = nn.Linear(int(self.hidden_size), int(dr), bias=use_bias).to(device=device, dtype=params_dtype)
                with torch.no_grad():
                    rope_k.weight.copy_(wk_full.to(device=device, dtype=params_dtype))
                    if use_bias and rope_k.bias is not None and isinstance(bk_full, torch.Tensor):
                        rope_k.bias.copy_(bk_full.to(device=device, dtype=params_dtype))
                rope_k.requires_grad_(False)
            elif int(wk_full.shape[0]) == int(ckpt_total_kv_heads) * int(dr):
                rope_k_shared = False
                wk3 = wk_full.view(int(ckpt_total_kv_heads), int(dr), int(self.hidden_size))
                wk3_local = _slice_kv_heads_for_attn_tp(
                    wk3, total_kv_heads=ckpt_total_kv_heads, attn_tp_rank=attn_tp_rank, attn_tp_size=attn_tp_size
                )
                if wk3_local.ndim != 3 or int(wk3_local.shape[1]) != int(dr) or int(wk3_local.shape[2]) != int(self.hidden_size):
                    raise ValueError(f"rope_k sliced shape mismatch: {tuple(wk3_local.shape)}")
                wk_local = wk3_local.reshape(int(wk3_local.shape[0]) * int(dr), int(self.hidden_size)).contiguous()
                bk_local: torch.Tensor | None = None
                if isinstance(bk_full, torch.Tensor):
                    if bk_full.ndim != 1 or int(bk_full.shape[0]) != int(ckpt_total_kv_heads) * int(dr):
                        raise ValueError(
                            f"rope_k.bias shape mismatch: got {tuple(bk_full.shape)} expected ({int(ckpt_total_kv_heads)*int(dr)},)"
                        )
                    bk3 = bk_full.view(int(ckpt_total_kv_heads), int(dr))
                    bk3_local = _slice_kv_heads_for_attn_tp(
                        bk3, total_kv_heads=ckpt_total_kv_heads, attn_tp_rank=attn_tp_rank, attn_tp_size=attn_tp_size
                    )
                    if bk3_local.ndim != 2 or int(bk3_local.shape[1]) != int(dr):
                        raise ValueError(f"rope_k.bias sliced shape mismatch: {tuple(bk3_local.shape)}")
                    bk_local = bk3_local.reshape(int(bk3_local.shape[0]) * int(dr)).contiguous()
                rope_k = nn.Linear(int(self.hidden_size), int(wk3_local.shape[0]) * int(dr), bias=bk_local is not None).to(
                    device=device, dtype=params_dtype
                )
                with torch.no_grad():
                    rope_k.weight.copy_(wk_local.to(device=device, dtype=params_dtype))
                    if bk_local is not None and rope_k.bias is not None:
                        rope_k.bias.copy_(bk_local.to(device=device, dtype=params_dtype))
                rope_k.requires_grad_(False)
            else:
                raise ValueError(
                    f"rope_k.weight out_features mismatch: got {int(wk_full.shape[0])} expected {dr} or {int(ckpt_total_kv_heads)*int(dr)}"
                )

        self._deepseek_latent_k = k_mod
        self._deepseek_latent_v = v_mod
        self._deepseek_q = q_mod
        self._deepseek_o = o_mod
        self._deepseek_mla_qkc = qkc_w
        self._deepseek_rope_q = rope_q
        self._deepseek_rope_k = rope_k
        self._deepseek_rope_dim = int(rope_dim_ckpt) if rope_q is not None and rope_k is not None else 0
        self._deepseek_rope_k_shared = bool(rope_k_shared)

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
        if self._deepseek_latent_kv_mla:
            if self._deepseek_latent_k is None or self._deepseek_latent_v is None:
                raise RuntimeError(
                    "GPT-OSS MLA mode requires DeepSeek latent-KV patch modules, but they are missing. "
                    "Set SGLANG_GPTOSS_DEEPSEEK_LATENT_KV_CKPT_INDEX_JSON and ensure weights were loaded."
                )
            if self._deepseek_latent_v.down is not self._deepseek_latent_k.down:
                raise RuntimeError(
                    "GPT-OSS MLA mode requires share_down=True (single latent for K/V). "
                    "Got separate K/V down projections."
                )

            if self._deepseek_q is not None:
                q = self._deepseek_q(hidden_states)

            k_mod = self._deepseek_latent_k
            x = hidden_states
            if k_mod.precond is not None:
                x = k_mod.precond(x)
            z = k_mod.down(x)  # [N, r]
            n = int(z.shape[0])
            r = int(self._deepseek_mla_kv_lora_rank or z.shape[-1])
            if int(z.shape[-1]) != r:
                raise RuntimeError(
                    f"latent rank mismatch: down produced {int(z.shape[-1])}, expected {r}"
                )

            kvh = int(self.num_kv_heads)
            if kvh <= 0:
                raise RuntimeError(f"bad num_kv_heads={kvh}")

            # DeepSeek-style MLA cache format:
            # - K/V "nope" is the shared latent z (1 KV head).
            # - Q "nope" is head/group-specific via W_UK^T (we reuse the checkpoint's K up matrices).
            k_nope = z.unsqueeze(1)  # [N, 1, r]
            v_nope = k_nope

            rope_dim = int(self._deepseek_mla_rope_head_dim or 0)
            if rope_dim <= 0:
                raise RuntimeError("GPT-OSS MLA mode missing rope_dim")
            if rope_dim > int(self.head_dim):
                raise RuntimeError(
                    f"bad rope_dim={rope_dim} for head_dim={int(self.head_dim)}"
                )
            nope_dim = int(self.head_dim) - rope_dim

            q_heads = q.view(n, int(self.num_heads), int(self.head_dim))
            # Split query into decoupled slices (DeepSeek semantics):
            # - q_nope participates in the latent dot-product term.
            # - q_pe participates in the RoPE term.
            q_nope_src = q_heads[:, :, :nope_dim] if nope_dim > 0 else None
            if self._deepseek_rope_q is not None:
                if int(self._deepseek_rope_dim) != int(rope_dim):
                    raise RuntimeError(
                        f"ckpt decoupled_rope_dim={int(self._deepseek_rope_dim)} but MLA rope_dim={int(rope_dim)}. "
                        f"Set {_GPTOSS_DEEPSEEK_LATENT_KV_MLA_ROPE_DIM_ENV}={int(self._deepseek_rope_dim)}."
                    )
                q_pe_src = self._deepseek_rope_q(hidden_states).view(n, int(self.num_heads), int(rope_dim))
            else:
                q_pe_src = q_heads[:, :, nope_dim:]  # [N, Hq, rope_dim] (rope_dim may == head_dim)

            # Compute Q_nope_out: q_nope @ W_UK^T -> [N, Hq, r]
            q_heads = q.view(n, int(self.num_heads), int(self.head_dim))
            if int(self.num_heads) % kvh != 0:
                raise RuntimeError(
                    f"q_head_num ({int(self.num_heads)}) must be divisible by kv_head_num ({kvh})"
                )
            group = int(self.num_heads) // kvh
            uk = k_mod.up  # [Hkv, r, D]
            qkc = self._deepseek_mla_qkc
            if qkc is None:
                qkc = uk
            q_nope = q_heads.new_empty((n, int(self.num_heads), r))
            if nope_dim <= 0:
                # When rope_dim consumes the entire original head_dim (GPT-OSS head_dim=64),
                # we *still* need head/group-specific content scores. Zeroing q_nope collapses
                # all KV-head groups onto the shared RoPE key and degenerates quality under
                # FlashMLA.
                #
                # Use the full query vector to produce the latent score term:
                #   q_nope_out[h] = q[h] @ W_UK[ kv_group(h) ]^T   (shape: [r])
                # so that q_nope_out · z matches q · (z @ W_UK) without materializing K.
                for i in range(kvh):
                    sl = slice(i * group, (i + 1) * group)
                    # (N, group, D) @ (D, r) -> (N, group, r)
                    q_nope[:, sl, :] = torch.matmul(q_heads[:, sl, :], qkc[i].t())
            else:
                if int(uk.shape[-1]) < nope_dim:
                    raise RuntimeError(
                        f"latent uk dim {int(uk.shape[-1])} < nope_dim {nope_dim}"
                    )
                for i in range(kvh):
                    sl = slice(i * group, (i + 1) * group)
                    uk_nope = qkc[i][:, :nope_dim]  # [r, nope_dim]
                    # (N, group, nope_dim) @ (nope_dim, r) -> (N, group, r)
                    q_nope[:, sl, :] = torch.matmul(q_nope_src[:, sl, :], uk_nope.t())

            # Shared K_pe (RoPE key):
            # - Derive it from the patched key projection space (z @ W_UK) and share across KV heads.
            #   This avoids depending on the original K projection (which may not exist in a fully
            #   converted checkpoint) and keeps K_pe consistent with the latent z.
            if self._deepseek_rope_k is not None:
                if int(self._deepseek_rope_dim) != int(rope_dim):
                    raise RuntimeError(
                        f"ckpt decoupled_rope_dim={int(self._deepseek_rope_dim)} but MLA rope_dim={int(rope_dim)}. "
                        f"Set {_GPTOSS_DEEPSEEK_LATENT_KV_MLA_ROPE_DIM_ENV}={int(self._deepseek_rope_dim)}."
                    )
                if bool(self._deepseek_rope_k_shared):
                    k_pe_src = self._deepseek_rope_k(hidden_states).view(n, 1, int(rope_dim))
                else:
                    kvh_local = int(self.num_kv_heads)
                    k_raw = self._deepseek_rope_k(hidden_states).view(n, kvh_local, int(rope_dim))
                    k_pe_src = k_raw.mean(dim=1, keepdim=True).contiguous()
            elif self._deepseek_mla_ropek is not None:
                k_pe_src = self._deepseek_mla_ropek(hidden_states).view(n, 1, rope_dim)
            elif nope_dim <= 0:
                uk_mean = uk.mean(dim=0)  # [r, D]
                k_pe_src = torch.matmul(z, uk_mean)  # [N, D]
                if k_mod.bias is not None:
                    k_pe_src = k_pe_src + k_mod.bias.mean(dim=0).view(1, int(self.head_dim))
                k_pe_src = k_pe_src.view(n, 1, rope_dim)  # [N, 1, rope_dim==head_dim]
            else:
                uk_rope = uk[:, :, nope_dim:]  # [Hkv, r, rope_dim]
                k_pe_src = torch.einsum("nr,hrd->nhd", z, uk_rope)  # [N, Hkv, rope_dim]
                if k_mod.bias is not None:
                    k_pe_src = k_pe_src + k_mod.bias[:, nope_dim:].view(1, kvh, rope_dim)
                k_pe_src = k_pe_src.mean(dim=1, keepdim=True)  # [N, 1, rope_dim]

            # RoPE only the decoupled slices (DeepSeek semantics).
            rotary = self._deepseek_mla_rotary_emb
            if rotary is None:
                raise RuntimeError("GPT-OSS MLA mode missing mla rotary embedding")
            q_rope, k_rope = rotary(positions, q_pe_src, k_pe_src)

            inner_state = q_nope, k_nope, v_nope, forward_batch, q_rope, k_rope
            return None, forward_batch, inner_state

        if self._deepseek_latent_k is not None:
            # Replace K/V with DeepSeek-native latent projections (flattened per-head).
            k = self._deepseek_latent_k(hidden_states)
            if self._deepseek_latent_v is not None:
                v = self._deepseek_latent_v(hidden_states)
            if self._deepseek_q is not None:
                q = self._deepseek_q(hidden_states)

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
        # Debug knob: disable GPT-OSS attention sinks (used to isolate FlashMLA issues).
        disable_sinks = _bool_env("SGLANG_GPTOSS_DISABLE_SINKS")
        sinks = None if disable_sinks else self.sinks
        if self._deepseek_latent_kv_mla:
            q_nope, k_nope, v_nope, forward_batch, q_rope, k_rope = inner_state
            attn_latent = self.attn(
                q_nope,
                k_nope,
                v_nope,
                forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
                sinks=sinks,
                save_kv_cache=True,
            )
            v_mod = self._deepseek_latent_v
            if v_mod is None:
                raise RuntimeError("GPT-OSS MLA mode requires V latent module")

            r = int(self._deepseek_mla_kv_lora_rank or 0)
            if r <= 0:
                raise RuntimeError("GPT-OSS MLA mode missing kv_lora_rank")
            attn_latent_3d = attn_latent.view(-1, int(self.num_heads), r)

            kvh = int(self.num_kv_heads)
            if kvh <= 0:
                raise RuntimeError(f"bad num_kv_heads={kvh}")
            if int(self.num_heads) % kvh != 0:
                raise RuntimeError(
                    f"q_head_num ({int(self.num_heads)}) must be divisible by kv_head_num ({kvh})"
                )
            group = int(self.num_heads) // kvh

            out_v = attn_latent_3d.new_empty(
                (int(attn_latent_3d.shape[0]), int(self.num_heads), int(self.head_dim))
            )
            up = v_mod.up
            bias = v_mod.bias
            for i in range(kvh):
                sl = slice(i * group, (i + 1) * group)
                out_v[:, sl, :] = torch.matmul(attn_latent_3d[:, sl, :], up[i])
                if bias is not None:
                    out_v[:, sl, :] = out_v[:, sl, :] + bias[i].view(1, 1, int(self.head_dim))
            attn_output = out_v.reshape(int(out_v.shape[0]), int(self.num_heads) * int(self.head_dim))
        else:
            attn_output = self.attn(
                *inner_state,
                sinks=sinks,
                save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
            )
        if self._deepseek_o is not None:
            output = self._deepseek_o(attn_output)
        else:
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
        logger.warning(
            "Ignoring KV scale file for GPT-OSS FP8 path: %s. "
            "The current GPT-OSS KV-scale loader is disabled because it is "
            "numerically unstable in the validated DFlash FP8 serving path.",
            quantization_param_path,
        )
        return

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

    def _maybe_apply_deepseek_latent_kv_patch(self) -> None:
        """
        If requested via env var, load a DeepSeek-native latent-KV ckpt-index and
        patch the in-memory GPT-OSS model (inference-only).
        """

        if getattr(self, "_deepseek_latent_kv_patched", False):
            return
        idx_path = os.getenv(_GPTOSS_DEEPSEEK_LATENT_KV_CKPT_INDEX_ENV, "").strip()
        if not idx_path:
            return
        idx = _load_deepseek_latent_kv_ckpt_index(idx_path)

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        params_dtype = getattr(self.config, "torch_dtype", torch.bfloat16)

        patched = 0
        for layer_id in range(self.start_layer, self.end_layer):
            ckpt_path = idx.get(int(layer_id))
            if not ckpt_path:
                continue
            layer = self.model.layers[layer_id]
            if isinstance(layer, PPMissingLayer):
                continue
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            layer.self_attn.enable_deepseek_latent_kv_patch(
                ckpt,
                attn_tp_rank=attn_tp_rank,
                attn_tp_size=attn_tp_size,
                params_dtype=params_dtype,
            )
            patched += 1

        verbose = os.getenv("SGLANG_GPTOSS_DEEPSEEK_LATENT_KV_VERBOSE", "").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        if verbose and str(os.environ.get("RANK", "0")) == "0":
            # Use stdout (not logger) so it is visible even when SGLang sets log_level=error.
            print(
                f"[GPTOSS_DEEPSEEK_LATENT_KV] patched_layers={int(patched)} ckpt_index={idx_path}",
                flush=True,
            )

        logger.info(
            "Applied DeepSeek latent-KV patch to %d GPT-OSS layers (ckpt-index=%s)",
            int(patched),
            idx_path,
        )
        setattr(self, "_deepseek_latent_kv_patched", True)

    def _maybe_apply_deepseek_mla_ropek_patch(self) -> None:
        """
        If requested via env var, load a RoPE-K synth ckpt-index and attach the shared
        RoPE-K generator used by the GPT-OSS MLA cache path.
        """

        if getattr(self, "_deepseek_mla_ropek_patched", False):
            return
        idx_path = os.getenv(_GPTOSS_DEEPSEEK_MLA_ROPEK_CKPT_INDEX_ENV, "").strip()
        if not idx_path:
            return
        idx = _load_ropek_ckpt_index(idx_path)

        params_dtype = getattr(self.config, "torch_dtype", torch.bfloat16)
        patched = 0
        for layer_id in range(self.start_layer, self.end_layer):
            ckpt_path = idx.get(int(layer_id))
            if not ckpt_path:
                continue
            layer = self.model.layers[layer_id]
            if isinstance(layer, PPMissingLayer):
                continue
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            layer.self_attn.enable_deepseek_mla_ropek_patch(ckpt, params_dtype=params_dtype)
            patched += 1

        verbose = os.getenv("SGLANG_GPTOSS_DEEPSEEK_LATENT_KV_VERBOSE", "").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        if verbose and str(os.environ.get("RANK", "0")) == "0":
            print(
                f"[GPTOSS_DEEPSEEK_MLA_ROPEK] patched_layers={int(patched)} ckpt_index={idx_path}",
                flush=True,
            )

        logger.info(
            "Applied GPT-OSS MLA RoPE-K synth patch to %d layers (ckpt-index=%s)",
            int(patched),
            idx_path,
        )
        setattr(self, "_deepseek_mla_ropek_patched", True)

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
        # Optional: DeepSeek-native latent KV injection for inference.
        self._maybe_apply_deepseek_latent_kv_patch()
        self._maybe_apply_deepseek_mla_ropek_patch()

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
