import copy
import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    ALL_ATTENTION_FUNCTIONS,
    GptOssDecoderLayer,
    GptOssMLP,
    GptOssModel,
    GptOssPreTrainedModel,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    apply_rotary_pos_emb,
    create_causal_mask,
    create_sliding_window_causal_mask,
    eager_attention_forward,
    load_balancing_loss_func,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache


_UNSLOTH_FLEX_ATTENTION_ERROR = None


def _load_unsloth_flex_attention_with_sink():
    global _UNSLOTH_FLEX_ATTENTION_ERROR

    local_unsloth_zoo = os.environ.get("UNSLOTH_ZOO_LOCAL_PATH", "/workspace/unsloth_zoo_repo")
    local_unsloth = os.environ.get("UNSLOTH_LOCAL_PATH", "/workspace/unsloth_repo")
    if os.path.isdir(local_unsloth) and local_unsloth not in sys.path:
        sys.path.insert(0, local_unsloth)
    if os.path.isdir(local_unsloth_zoo) and local_unsloth_zoo not in sys.path:
        sys.path.insert(0, local_unsloth_zoo)

    try:
        if os.environ.get("UNSLOTH_IS_PRESENT") != "1":
            import unsloth  # noqa: F401
        from unsloth_zoo.flex_attention import flex_attention_with_sink

        return flex_attention_with_sink
    except Exception as exc:  # pragma: no cover - optional acceleration path
        _UNSLOTH_FLEX_ATTENTION_ERROR = repr(exc)
        return None


class GptOssMlaAttention(nn.Module):
    """HF-native MLA attention for converted GPT-OSS checkpoints.

    This follows the DeepSeek MLA decomposition pattern while preserving GPT-OSS
    sinks and alternating sliding/full attention behavior.

    GPT-OSS-specific contract:

    - q_proj is assumed to already live in the exported decoupled {NoPE, RoPE}
      basis.
    - kv_b_proj is interpreted as [all K_nope rows][all V rows], not per-head
      [K,V] interleaving.
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.qk_rope_head_dim = int(getattr(config, "qk_rope_head_dim"))
        self.qk_nope_head_dim = int(getattr(config, "qk_nope_head_dim"))
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.v_head_dim = int(getattr(config, "v_head_dim", getattr(config, "head_dim")))
        self.mla_rope_num_kv_heads = int(getattr(config, "mla_rope_num_kv_heads", 1))
        self.gpt_oss_mla_contract = getattr(config, "gpt_oss_mla_contract", None)
        if self.num_heads % self.mla_rope_num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_heads} is not divisible by "
                f"mla_rope_num_kv_heads={self.mla_rope_num_kv_heads}"
            )
        if config.model_type == "gpt_oss" and self.gpt_oss_mla_contract is not None:
            status = self.gpt_oss_mla_contract.get("status")
            query_basis_status = self.gpt_oss_mla_contract.get("query_basis_status")
            if status == "bridge_per_kv_head_rope":
                warnings.warn(
                    "GPT-OSS MLA checkpoint is in bridge mode with per-KV-head rope. "
                    "This is useful for debugging but is not the final shared-k_R "
                    "target contract.",
                    stacklevel=2,
                )
            if query_basis_status != "rewritten_decoupled":
                warnings.warn(
                    "GPT-OSS MLA checkpoint does not declare a rewritten decoupled "
                    "query basis. Quality-sensitive evaluation should treat this as "
                    "an experimental artifact, not the final target contract.",
                    stacklevel=2,
                )

        per_layer_rank = getattr(config, "kv_lora_rank_per_layer", None)
        if per_layer_rank is not None:
            self.kv_lora_rank = int(per_layer_rank[layer_idx])
        else:
            self.kv_lora_rank = int(getattr(config, "kv_lora_rank"))

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.qk_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.mla_rope_num_kv_heads * self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.mla_train_attention_backend = getattr(
            config,
            "mla_train_attention_backend",
            "eager",
        )

        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, self.num_heads, self.qk_head_dim)
        key_shape = (batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rope = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.mla_rope_num_kv_heads * self.qk_rope_head_dim],
            dim=-1,
        )
        kv_expanded = self.kv_b_proj(kv_latent)
        k_flat, v_flat = torch.split(
            kv_expanded,
            [self.num_heads * self.qk_nope_head_dim, self.num_heads * self.v_head_dim],
            dim=-1,
        )
        k_nope = k_flat.view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)
        value_states = v_flat.view(batch_size, seq_length, self.num_heads, self.v_head_dim).transpose(1, 2)

        if self.qk_rope_head_dim > 0:
            k_rope = k_rope.reshape(
                batch_size,
                self.mla_rope_num_kv_heads,
                seq_length,
                self.qk_rope_head_dim,
            )
            cos, sin = position_embeddings
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
            if self.mla_rope_num_kv_heads != self.num_heads:
                k_rope = repeat_kv(k_rope, self.num_heads // self.mla_rope_num_kv_heads)
            query_states = torch.cat((q_nope, q_rope), dim=-1)
            key_states = torch.cat((k_nope, k_rope), dim=-1)
        else:
            query_states = q_nope
            key_states = k_nope

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        use_unsloth_flex_attention = (
            self.training
            and past_key_values is None
            and self.config._attn_implementation == "eager"
            and (
                self.mla_train_attention_backend == "unsloth_flex_attention"
                or os.environ.get("GPT_OSS_MLA_USE_UNSLOTH_FLEX_ATTENTION", "0") == "1"
            )
        )

        if use_unsloth_flex_attention:
            flex_attention_with_sink = _load_unsloth_flex_attention_with_sink()
            if flex_attention_with_sink is None:
                raise RuntimeError(
                    "GPT-OSS MLA training requested Unsloth Flex Attention, but "
                    f"unsloth_zoo.flex_attention.flex_attention_with_sink could not be loaded: "
                    f"{_UNSLOTH_FLEX_ATTENTION_ERROR}"
                )
            kernel_dtype = query_states.dtype
            if kernel_dtype not in (torch.float16, torch.bfloat16):
                kernel_dtype = torch.bfloat16 if query_states.is_cuda else torch.float32
            if query_states.dtype != kernel_dtype:
                query_states = query_states.to(kernel_dtype)
            if key_states.dtype != kernel_dtype:
                key_states = key_states.to(kernel_dtype)
            if value_states.dtype != kernel_dtype:
                value_states = value_states.to(kernel_dtype)
            attn_output = flex_attention_with_sink(
                self,
                query_states,
                key_states,
                value_states,
            )
            if attn_output.dtype != self.o_proj.weight.dtype:
                attn_output = attn_output.to(self.o_proj.weight.dtype)
            attn_weights = None
        else:
            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,
                **kwargs,
            )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GptOssMlaDecoderLayer(GptOssDecoderLayer):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GptOssMlaAttention(config=config, layer_idx=layer_idx)


class GptOssMlaModel(GptOssPreTrainedModel):
    _no_split_modules = ["GptOssMlaDecoderLayer"]

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GptOssMlaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rotary_config = copy.deepcopy(config)
        rotary_config.head_dim = int(
            getattr(config, "qk_rope_head_dim", getattr(config, "head_dim"))
        )
        self.rotary_emb = GptOssRotaryEmbedding(config=rotary_config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GptOssMlaForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _no_split_modules = ["GptOssMlaDecoderLayer"]
    config_class = GptOssConfig

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GptOssMlaAttention):
            module.sinks.data.normal_(mean=0.0, std=self.config.initializer_range)

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.model = GptOssMlaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["GptOssMlaForCausalLM"]
