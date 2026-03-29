from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.utils import is_cuda

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"

_DFLASH_SAMPLING_VERIFY_AVAILABLE = False
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)


if is_cuda():
    try:
        from sgl_kernel import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        _DFLASH_SAMPLING_VERIFY_AVAILABLE = True
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
        tree_speculative_sampling_target_only = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def scale_kv_cell_size_per_token_for_dflash(
    *,
    target_cell_size_per_token: int,
    target_num_layers: int,
    draft_num_layers: int,
    draft_cell_size_per_token: Optional[int] = None,
) -> int:
    """Compute bytes/token budget for combined target+draft KV pools (DFLASH).

    DFLASH runs a separate draft runner with its own KV pool. The target runner's
    token capacity must fit both pools in aggregate.

    Returns:
        Approximate per-token bytes for (target KV + draft KV), expressed as a
        scaled version of `target_cell_size_per_token`, unless an explicit
        `draft_cell_size_per_token` is provided (in which case we sum them).
    """
    if target_cell_size_per_token <= 0:
        raise ValueError(
            "target_cell_size_per_token must be positive, "
            f"got {target_cell_size_per_token}."
        )

    if draft_cell_size_per_token is not None:
        draft_cell_size_per_token = int(draft_cell_size_per_token)
        if draft_cell_size_per_token <= 0:
            raise ValueError(
                "draft_cell_size_per_token must be positive when provided, "
                f"got {draft_cell_size_per_token}."
            )
        return int(target_cell_size_per_token) + int(draft_cell_size_per_token)

    if target_num_layers <= 0 or draft_num_layers <= 0:
        return int(target_cell_size_per_token)

    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers) + int(target_num_layers) - 1
    ) // int(target_num_layers)


def resolve_dflash_verify_mask_policy(attn_backend: Any) -> tuple[str, bool]:
    backend = attn_backend
    for _ in range(4):
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    backend_name = type(backend).__name__
    return backend_name, (backend_name not in _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS)


def _get_or_create_chain_verify_buffers(
    *,
    bs: int,
    draft_token_num: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    key = (device.index, int(draft_token_num))
    cached = _DFLASH_CHAIN_VERIFY_BUFFERS.get(key)
    cap_bs = 0 if cached is None else int(cached["cap_bs"])
    if cap_bs < bs:
        new_cap = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        retrieve_index = torch.arange(
            new_cap * draft_token_num, dtype=torch.int64, device=device
        ).view(new_cap, draft_token_num)
        row_next = torch.arange(
            1, draft_token_num + 1, dtype=torch.int64, device=device
        )
        row_next[-1] = -1
        retrieve_next_token = row_next.unsqueeze(0).expand(new_cap, -1).clone()
        retrieve_next_sibling = torch.full(
            (new_cap, draft_token_num), -1, dtype=torch.int64, device=device
        )
        predicts = torch.empty(
            (new_cap * draft_token_num + 1,), dtype=torch.int32, device=device
        )
        accept_index = torch.empty(
            (new_cap, draft_token_num), dtype=torch.int32, device=device
        )
        accept_token_num = torch.empty((new_cap,), dtype=torch.int32, device=device)
        cached = {
            "cap_bs": int(new_cap),
            "retrieve_index": retrieve_index,
            "retrieve_next_token": retrieve_next_token,
            "retrieve_next_sibling": retrieve_next_sibling,
            "predicts": predicts,
            "accept_index": accept_index,
            "accept_token_num": accept_token_num,
        }
        _DFLASH_CHAIN_VERIFY_BUFFERS[key] = cached

    assert cached is not None
    retrieve_index = cached["retrieve_index"][:bs]
    retrieve_next_token = cached["retrieve_next_token"][:bs]
    retrieve_next_sibling = cached["retrieve_next_sibling"][:bs]
    predicts = cached["predicts"][: bs * draft_token_num + 1]
    accept_index = cached["accept_index"][:bs]
    accept_token_num = cached["accept_token_num"][:bs]
    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Select target layer indices used to build DFlash context features.

    Args:
        num_target_layers: Number of transformer layers in the runtime target model.
        num_draft_layers: Number of layers in the DFlash draft model.

    Returns:
        A list of 0-based target layer indices of length `num_draft_layers`.

    Notes:
        - DFlash uses hidden states after each selected target layer (HF-style).
        - SGLang captures "before layer i", so the model hook will typically add +1
          when mapping to capture points.
    """
    if num_target_layers <= 0:
        raise ValueError(
            f"num_target_layers must be positive, got {num_target_layers}."
        )
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}.")

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    if end < start:
        raise ValueError(
            "DFlash layer selection requires num_target_layers >= 4. "
            f"Got num_target_layers={num_target_layers}."
        )

    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_dflash_config(config: Any) -> dict:
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg

    try:
        return dict(cfg)
    except Exception:
        return {}


def _parse_optional_int(
    value: Any,
    *,
    field_name: str,
    min_value: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DFlashDraftConfig:
    num_hidden_layers: Optional[int]
    num_target_layers: Optional[int]
    block_size: Optional[int]
    target_layer_ids: Optional[List[int]]
    mask_token: str
    mask_token_id: Optional[int]

    def require_num_layers(self) -> int:
        if self.num_hidden_layers is None:
            raise ValueError(
                "DFLASH requires draft num_hidden_layers in config. "
                "Got config without num_hidden_layers."
            )
        return int(self.num_hidden_layers)

    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.block_size if self.block_size is not None else default

    def resolve_target_layer_ids(
        self,
        *,
        target_num_layers: int,
        draft_num_layers: Optional[int] = None,
    ) -> List[int]:
        target_num_layers = int(target_num_layers)
        if target_num_layers <= 0:
            raise ValueError(
                f"target_num_layers must be positive, got {target_num_layers}."
            )

        if self.target_layer_ids is None:
            if draft_num_layers is None:
                draft_num_layers = self.require_num_layers()
            return build_target_layer_ids(target_num_layers, int(draft_num_layers))

        resolved = list(self.target_layer_ids)
        if len(resolved) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(resolved)}."
            )
        for idx, val in enumerate(resolved):
            if val < 0 or val >= target_num_layers:
                raise ValueError(
                    "DFLASH target_layer_ids contains an out-of-range layer id. "
                    f"target_layer_ids[{idx}]={val}, target_num_layers={target_num_layers}."
                )
        return resolved


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    dflash_cfg = _get_dflash_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_hf_config, "num_hidden_layers", None),
        field_name="DFLASH draft num_hidden_layers",
        min_value=1,
    )
    raw_num_target_layers = dflash_cfg.get(
        "num_target_layers",
        _cfg_get(draft_hf_config, "num_target_layers", None),
    )
    num_target_layers = _parse_optional_int(
        raw_num_target_layers,
        field_name="DFLASH draft num_target_layers",
        min_value=1,
    )

    # Keep support for current checkpoints where block_size is top-level.
    raw_block_size = dflash_cfg.get(
        "block_size",
        _cfg_get(draft_hf_config, "block_size", None),
    )
    block_size = _parse_optional_int(
        raw_block_size,
        field_name="DFLASH block_size",
        min_value=1,
    )

    layer_ids = dflash_cfg.get(
        "target_layer_ids",
        _cfg_get(draft_hf_config, "target_layer_ids", None),
    )
    parsed_target_layer_ids: Optional[List[int]]
    if layer_ids is None:
        parsed_target_layer_ids = None
    else:
        if not isinstance(layer_ids, (list, tuple)):
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be a list of ints, "
                f"got type={type(layer_ids).__name__}."
            )
        parsed_target_layer_ids = [int(x) for x in layer_ids]
        if len(parsed_target_layer_ids) <= 0:
            raise ValueError(
                "DFLASH dflash_config.target_layer_ids must be non-empty. "
                f"Got len(target_layer_ids)={len(parsed_target_layer_ids)}."
            )

    mask_token = dflash_cfg.get("mask_token", None)
    if mask_token is None:
        mask_token = DEFAULT_DFLASH_MASK_TOKEN
    if not isinstance(mask_token, str) or not mask_token:
        raise ValueError(
            "DFLASH dflash_config.mask_token must be a non-empty string, "
            f"got {mask_token!r}."
        )

    mask_token_id = dflash_cfg.get("mask_token_id", None)
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be an integer, "
                f"got {mask_token_id!r} (type={type(mask_token_id).__name__})."
            )
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0:
            raise ValueError(
                "DFLASH dflash_config.mask_token_id must be non-negative, "
                f"got {mask_token_id}."
            )

    return DFlashDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=parsed_target_layer_ids,
        mask_token=mask_token,
        mask_token_id=mask_token_id,
    )


def resolve_dflash_target_layer_ids(
    *,
    draft_hf_config: Any,
    target_num_layers: int,
    draft_num_layers: Optional[int] = None,
) -> List[int]:
    draft_cfg = parse_dflash_draft_config(draft_hf_config=draft_hf_config)
    resolved_target_num_layers = int(target_num_layers)
    if draft_cfg.num_target_layers is not None:
        resolved_target_num_layers = int(draft_cfg.num_target_layers)
    return draft_cfg.resolve_target_layer_ids(
        target_num_layers=resolved_target_num_layers,
        draft_num_layers=draft_num_layers,
    )


def resolve_dflash_block_size(
    *,
    draft_hf_config: Any | None = None,
    draft_config_json: Any | None = None,
    default: Optional[int] = None,
) -> Optional[int]:
    """Resolve the DFlash block size from either HF config or raw config.json.

    This keeps compatibility with both older callsites that pass
    `draft_config_json=...` and newer ones that pass `draft_hf_config=...`.
    """
    cfg_src = draft_hf_config if draft_hf_config is not None else draft_config_json
    if cfg_src is None:
        return default
    return parse_dflash_draft_config(draft_hf_config=cfg_src).resolve_block_size(
        default=default
    )


@dataclass(frozen=True)
class DFlashTargetOnlyCommitResult:
    commit_len: int
    new_verified_token: int
    accepted_draft_tokens: int


def commit_dflash_proposed_tokens_to_req(
    *,
    req: Any,
    proposed: List[int],
    empty_error_prefix: str = "DFLASH verify",
) -> DFlashTargetOnlyCommitResult:
    """Apply a committed speculative token prefix to a request on the CPU path.

    This helper centralizes the current production CPU-side request mutation used by
    DFlash `target_only` verify. It intentionally preserves the existing semantics:

    - fast path for plain stop-token / max-new-token handling
    - slow path for grammar / stop string / regex interaction
    - spec-acceptance accounting
    - fallback to the current token when no new token is appended
    """
    appended = 0
    if (
        req.grammar is None
        and not req.sampling_params.stop_strs
        and not req.sampling_params.stop_regex_strs
    ):
        remaining = int(req.sampling_params.max_new_tokens) - len(req.output_ids)
        if remaining > 0:
            tokens = proposed[:remaining]
            if not req.sampling_params.ignore_eos:
                stop_token_ids = req.sampling_params.stop_token_ids
                eos_token_ids = req.eos_token_ids
                tokenizer = req.tokenizer
                tokenizer_eos = tokenizer.eos_token_id if tokenizer is not None else None
                additional_stop = (
                    tokenizer.additional_stop_token_ids if tokenizer is not None else None
                )
                vocab_size = getattr(req, "vocab_size", None)

                for j, token_id in enumerate(tokens):
                    if vocab_size is not None and (
                        int(token_id) > int(vocab_size) or int(token_id) < 0
                    ):
                        tokens = tokens[: j + 1]
                        break
                    if stop_token_ids and token_id in stop_token_ids:
                        tokens = tokens[: j + 1]
                        break
                    if eos_token_ids and token_id in eos_token_ids:
                        tokens = tokens[: j + 1]
                        break
                    if tokenizer_eos is not None and int(token_id) == int(tokenizer_eos):
                        tokens = tokens[: j + 1]
                        break
                    if additional_stop and token_id in additional_stop:
                        tokens = tokens[: j + 1]
                        break

            req.output_ids.extend(int(tok) for tok in tokens)
            appended = len(tokens)
            if appended > 0:
                req.check_finished(new_accepted_len=appended)
    else:
        for tok in proposed:
            req.output_ids.append(int(tok))
            appended += 1
            req.check_finished()
            if req.finished():
                break
            if req.grammar is not None:
                req.grammar.accept_token(int(tok))

    if req.output_ids:
        new_verified_token = int(req.output_ids[-1])
    elif req.origin_input_ids:
        new_verified_token = int(req.origin_input_ids[-1])
    else:
        raise RuntimeError(
            f"{empty_error_prefix} cannot determine current token: both output_ids and "
            "origin_input_ids are empty."
        )

    accepted_draft_tokens = max(0, appended - 1)
    req.spec_verify_ct += 1
    req.spec_accepted_tokens += accepted_draft_tokens
    if hasattr(req, "update_spec_acceptance_histogram"):
        req.update_spec_acceptance_histogram(accepted_draft_tokens)

    return DFlashTargetOnlyCommitResult(
        commit_len=appended,
        new_verified_token=new_verified_token,
        accepted_draft_tokens=accepted_draft_tokens,
    )


def can_dflash_slice_qkv_weight(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether DFlash can slice KV weights from a fused QKV linear layer."""
    quant_method = getattr(qkv_proj, "quant_method", None)
    # Avoid importing the full quantization package at module import time (it may
    # depend on optional GPU-only deps like triton). We only need to distinguish
    # "unquantized" vs "quantized" here.
    if quant_method is not None and type(quant_method).__name__ != "UnquantizedLinearMethod":
        return (
            False,
            "quantized qkv_proj is not supported for this path "
            f"(quant_method={type(quant_method).__name__})",
        )
    if not hasattr(qkv_proj, "weight"):
        return False, "qkv weight tensor is missing"
    return True, ""


def can_dflash_use_fused_qkv_proj(qkv_proj: Any) -> Tuple[bool, str]:
    """Validate whether a QKV layer is eligible for DFlash fused KV materialization."""
    eligible, reason = can_dflash_slice_qkv_weight(qkv_proj)
    if not eligible:
        return False, reason
    return True, ""


def compute_dflash_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    max_steps_per_req: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        accept_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index accept_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    if max_steps_per_req is not None:
        if max_steps_per_req.ndim != 1 or int(max_steps_per_req.shape[0]) != int(bs):
            raise ValueError(
                "max_steps_per_req must be a 1D tensor with shape [bs]. "
                f"Got shape={tuple(max_steps_per_req.shape)} for bs={bs}."
            )
        caps = max_steps_per_req.to(device=matches.device, dtype=torch.int64)
        caps = caps.clamp(min=0, max=int(block_size - 1))
        step_ids = torch.arange(int(block_size - 1), device=matches.device, dtype=torch.int64)
        matches = matches & (step_ids.unsqueeze(0) < caps.unsqueeze(1))
    accept_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), accept_len]
    return accept_len, bonus.to(torch.int64)


def pack_dflash_target_only_commits(
    *,
    target_predict: torch.Tensor,
    accept_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compact the committed target-only token prefix for CPU-side request updates.

    Args:
        target_predict: [bs, draft_token_num] target-predicted tokens.
        accept_len: [bs] accepted draft-token count, excluding the bonus token.

    Returns:
        proposed_flat: flattened committed token prefixes, row-major by request.
        commit_lens: [bs] committed token counts including the bonus token.
    """
    if target_predict.ndim != 2:
        raise ValueError(
            f"target_predict must be 2D, got shape={tuple(target_predict.shape)}"
        )
    if accept_len.ndim != 1 or int(accept_len.shape[0]) != int(target_predict.shape[0]):
        raise ValueError(
            "accept_len must be a 1D tensor with shape [bs]. "
            f"Got shape={tuple(accept_len.shape)} for bs={int(target_predict.shape[0])}."
        )

    bs, draft_token_num = target_predict.shape
    commit_lens = accept_len.to(device=target_predict.device, dtype=torch.int32).clamp(
        min=0, max=int(draft_token_num - 1)
    )
    commit_lens = commit_lens + 1
    keep_mask = torch.arange(
        draft_token_num, device=target_predict.device, dtype=torch.int32
    )[None, :] < commit_lens.unsqueeze(1)
    proposed_flat = target_predict[keep_mask].to(torch.int64)
    return proposed_flat, commit_lens


def compute_dflash_sampling_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    max_steps_per_req: Optional[torch.Tensor] = None,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
    return_proposed_tokens: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens for non-greedy sampling.

    This is a chain-specialized variant of speculative target-only verification:
      - DFlash proposals are linear (topk == 1), so each verify level has at most one candidate.
      - When a candidate is rejected at a level, the final token is sampled from
        `relu(q - p)` where `p` has only the rejected candidate mass.
    """
    if not _DFLASH_SAMPLING_VERIFY_AVAILABLE:
        raise RuntimeError(
            "DFLASH non-greedy verification is unavailable on this build/device."
        )
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must be 2D, "
            f"got shape={tuple(next_token_logits.shape)}."
        )

    bs, draft_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}.")
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError(
            "next_token_logits row count mismatch. "
            f"Expected {bs * draft_token_num}, got {next_token_logits.shape[0]}."
        )
    if candidates.device != next_token_logits.device:
        raise ValueError(
            "candidates and next_token_logits must be on the same device, "
            f"got {candidates.device} and {next_token_logits.device}."
        )

    if threshold_single is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_single = get_global_server_args().speculative_accept_threshold_single
    if threshold_acc is None:
        from sglang.srt.server_args import get_global_server_args

        threshold_acc = get_global_server_args().speculative_accept_threshold_acc
    threshold_single = float(threshold_single)
    threshold_acc = max(float(threshold_acc), 1e-9)

    device = next_token_logits.device

    if uniform_samples is None:
        uniform_samples = torch.rand(
            (bs, draft_token_num), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples.shape != (bs, draft_token_num):
            raise ValueError(
                "uniform_samples shape mismatch. "
                f"Expected {(bs, draft_token_num)}, got {tuple(uniform_samples.shape)}."
            )
        uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)

    if uniform_samples_for_final_sampling is None:
        uniform_samples_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=device
        )
    else:
        if uniform_samples_for_final_sampling.shape != (bs,):
            raise ValueError(
                "uniform_samples_for_final_sampling shape mismatch. "
                f"Expected {(bs,)}, got {tuple(uniform_samples_for_final_sampling.shape)}."
            )
        uniform_samples_for_final_sampling = uniform_samples_for_final_sampling.to(
            device=device,
            dtype=torch.float32,
        )

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    # Build target distribution once over all verify rows.
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature.view(-1, 1)
    sparse_topk_applied = False

    if use_sparse_topk and need_top_k:
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        max_top_k = int(repeated_top_ks.max().item())

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        if 0 < max_top_k < vocab_size:
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            if not torch.all(repeated_top_ks == max_top_k):
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = top_p_renorm_prob(topk_probs, repeated_top_ps)
            if bool(getattr(sampling_info, "need_min_p_sampling", False)):
                repeated_min_ps = torch.repeat_interleave(
                    sampling_info.min_ps, draft_token_num, dim=0
                ).to(device=device, dtype=topk_probs.dtype)
                base_probs = topk_probs
                min_p_thresholds = (
                    topk_probs.max(dim=-1, keepdim=True).values
                    * repeated_min_ps.view(-1, 1)
                )
                topk_probs = topk_probs.masked_fill(topk_probs < min_p_thresholds, 0.0)
                denom = topk_probs.sum(dim=-1, keepdim=True)
                topk_probs = torch.where(
                    denom > 0,
                    topk_probs / denom.clamp_min(1e-20),
                    base_probs,
                )

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
        if bool(getattr(sampling_info, "need_min_p_sampling", False)):
            expanded_min_ps = torch.repeat_interleave(
                sampling_info.min_ps, draft_token_num, dim=0
            ).to(device=device, dtype=target_probs.dtype)
            base_probs = target_probs
            min_p_thresholds = (
                target_probs.max(dim=-1, keepdim=True).values
                * expanded_min_ps.view(-1, 1)
            )
            target_probs = target_probs.masked_fill(
                target_probs < min_p_thresholds, 0.0
            )
            denom = target_probs.sum(dim=-1, keepdim=True)
            target_probs = torch.where(
                denom > 0,
                target_probs / denom.clamp_min(1e-20),
                base_probs,
            )
    target_probs = target_probs.view(bs, draft_token_num, -1).contiguous()
    draft_probs = torch.zeros_like(target_probs)

    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=draft_token_num,
        device=device,
    )
    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates_i64,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    accept_len = accept_token_num
    if max_steps_per_req is not None:
        if max_steps_per_req.ndim != 1 or int(max_steps_per_req.shape[0]) != int(bs):
            raise ValueError(
                "max_steps_per_req must be a 1D tensor with shape [bs]. "
                f"Got shape={tuple(max_steps_per_req.shape)} for bs={bs}."
            )
        caps = max_steps_per_req.to(device=device, dtype=torch.int32).clamp(
            min=0, max=int(draft_token_num - 1)
        )
        accept_len = torch.minimum(accept_len, caps)
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, accept_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    if not return_proposed_tokens:
        return accept_len, bonus

    gather_index = accept_index.clamp(min=0).to(torch.long)
    proposed_tokens = predicts[gather_index].to(torch.int64)
    valid_prefix = torch.arange(draft_token_num, device=device, dtype=torch.int32)[
        None, :
    ] <= accept_len.unsqueeze(1)
    proposed_tokens = proposed_tokens.masked_fill(~valid_prefix, 0)
    return accept_len, bonus, proposed_tokens
