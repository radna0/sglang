from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Callable, List, Optional, Tuple

import os
import logging
import torch
import torch.nn.functional as F
from sglang.srt.layers.sampler import (
    _sanitize_sampling_probs_for_multinomial_,
    multinomial_with_seed,
    prepare_top_k_top_p_min_p_sorted_probs_torch,
)
from sglang.srt.speculative.dflash_controller import (
    DFlashDifficultySignals,
    DFlashReqDifficultyState,
)
from sglang.srt.utils import is_cuda

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"

_DFLASH_SAMPLING_VERIFY_AVAILABLE = is_cuda()
_DFLASH_CHAIN_VERIFY_BUFFERS: dict[tuple[Optional[int], int], dict[str, Any]] = {}
_DFLASH_SAMPLED_DEBUG_CT: int = 0
_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset(
    {
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "FlashAttentionBackend",
        "TRTLLMHAAttnBackend",
        "TRTLLMMLABackend",
    }
)

logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


if is_cuda():
    try:
        from sgl_kernel import (  # type: ignore[import-not-found]
            tree_speculative_sampling_target_only,
        )
    except Exception:
        tree_speculative_sampling_target_only = None
else:
    tree_speculative_sampling_target_only = None


def is_dflash_sampling_verify_available() -> bool:
    return _DFLASH_SAMPLING_VERIFY_AVAILABLE


def build_dflash_filtered_sampling_distribution_from_probs(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs_sort, probs_idx = prepare_top_k_top_p_min_p_sorted_probs_torch(
        probs,
        top_ks,
        top_ps,
        min_ps,
        need_min_p_sampling,
    )
    _sanitize_sampling_probs_for_multinomial_(probs_sort)
    return probs_sort, probs_idx.to(torch.long)


def build_dflash_filtered_sampling_distribution_from_logits(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape={tuple(logits.shape)}.")
    if int(logits.shape[0]) == 0:
        empty_idx = torch.empty_like(logits, dtype=torch.long)
        return logits, empty_idx

    vocab_size = int(logits.shape[-1])
    top_ks_view = top_ks.view(-1).to(device=logits.device, dtype=torch.int64)
    use_bounded_topk = top_ks_view.numel() > 0 and not torch.any(
        top_ks_view >= vocab_size
    )
    if not use_bounded_topk:
        probs = F.softmax(logits, dim=-1)
        return build_dflash_filtered_sampling_distribution_from_probs(
            probs,
            top_ks.to(device=logits.device, dtype=torch.int32),
            top_ps.to(device=logits.device, dtype=torch.float32),
            min_ps.to(device=logits.device, dtype=torch.float32),
            need_min_p_sampling,
        )

    max_top_k = int(top_ks_view.max().item())
    max_top_k = max(1, min(max_top_k, vocab_size))
    topk_logits, topk_indices = torch.topk(
        logits,
        k=max_top_k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    if not torch.all(top_ks_view == max_top_k):
        ranks = torch.arange(max_top_k, device=logits.device, dtype=torch.int64)[None, :]
        valid = ranks < top_ks_view.unsqueeze(1)
        topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

    probs_sort = F.softmax(topk_logits, dim=-1)
    probs_sort.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    cumulative = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(cumulative - probs_sort) > top_ps.view(-1, 1)] = 0.0
    if need_min_p_sampling:
        min_thresholds = probs_sort[:, :1] * min_ps.to(
            device=logits.device, dtype=probs_sort.dtype
        ).view(-1, 1)
        probs_sort[probs_sort < min_thresholds] = 0.0
    _sanitize_sampling_probs_for_multinomial_(probs_sort)
    return probs_sort, topk_indices.to(torch.long)


def sample_dflash_filtered_distribution(
    *,
    probs_sort: torch.Tensor,
    probs_idx: torch.Tensor,
    sampling_seed: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_seed is None:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    else:
        sampled_index = multinomial_with_seed(
            torch.log(probs_sort.to(torch.float64)),
            sampling_seed,
            positions,
        )
    sampled_prob = torch.gather(probs_sort, dim=1, index=sampled_index).view(-1)
    sampled_token = torch.gather(
        probs_idx.to(torch.long), dim=1, index=sampled_index
    ).view(-1)
    return sampled_token, sampled_prob


def scatter_dflash_filtered_distribution_to_dense(
    probs_sort: torch.Tensor,
    probs_idx: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    dense = torch.zeros(
        (int(probs_sort.shape[0]), int(vocab_size)),
        dtype=probs_sort.dtype,
        device=probs_sort.device,
    )
    dense.scatter_(1, probs_idx.to(torch.long), probs_sort)
    return dense


def _sample_dflash_dense_probs_with_uniform(
    probs: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    probs = probs.clone()
    _sanitize_sampling_probs_for_multinomial_(probs)
    cdf = probs.cumsum(dim=-1)
    sampled = torch.sum(
        cdf < uniform.to(device=probs.device, dtype=probs.dtype).unsqueeze(1),
        dim=-1,
        dtype=torch.int64,
    )
    sampled.clamp_(max=int(probs.shape[-1] - 1))
    return sampled


def _sample_dflash_sparse_probs_with_uniform(
    probs_sort: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    probs_sort = probs_sort.clone()
    _sanitize_sampling_probs_for_multinomial_(probs_sort)
    cdf = probs_sort.cumsum(dim=-1)
    sampled = torch.sum(
        cdf < uniform.to(device=probs_sort.device, dtype=probs_sort.dtype).unsqueeze(1),
        dim=-1,
        dtype=torch.int64,
    )
    sampled.clamp_(max=int(probs_sort.shape[-1] - 1))
    return sampled


def snapshot_dflash_request_sampling_params(
    reqs: List[Any],
    *,
    device: Optional[torch.device | str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    temperatures_cpu = torch.tensor(
        [float(getattr(req.sampling_params, "temperature", 1.0)) for req in reqs],
        dtype=torch.float32,
    )
    top_ps_cpu = torch.tensor(
        [float(getattr(req.sampling_params, "top_p", 1.0)) for req in reqs],
        dtype=torch.float32,
    )
    top_ks_cpu = torch.tensor(
        [int(getattr(req.sampling_params, "top_k", 1)) for req in reqs],
        dtype=torch.int32,
    )
    min_ps_cpu = torch.tensor(
        [float(getattr(req.sampling_params, "min_p", 0.0)) for req in reqs],
        dtype=torch.float32,
    )
    need_min_p_sampling = bool(torch.any(min_ps_cpu > 0).item())

    if device is None:
        return (
            temperatures_cpu,
            top_ps_cpu,
            top_ks_cpu,
            min_ps_cpu,
            need_min_p_sampling,
        )

    temperatures = temperatures_cpu.to(device=device, dtype=torch.float32)
    top_ps = top_ps_cpu.to(device=device, dtype=torch.float32)
    top_ks = top_ks_cpu.to(device=device, dtype=torch.int32)
    min_ps = min_ps_cpu.to(device=device, dtype=torch.float32)
    return temperatures, top_ps, top_ks, min_ps, need_min_p_sampling


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


def dflash_sampling_info_uses_sampled_target(
    sampling_info: Any | None,
    *,
    reqs: Optional[List[Any]] = None,
) -> bool:
    """Return True when the active batch should use sampled target verification.

    Some call-sites historically relied only on ``sampling_info.is_all_greedy``.
    That flag can be misleading for prewarm / merged batches if it gets carried
    forward before the tensor fields are fully updated. The DFlash worker should
    resolve sampled-vs-greedy from the actual batched sampling tensors.
    """
    if sampling_info is None:
        return False

    # Avoid host synchronizations on device tensors in the decode hot path. SamplingBatchInfo
    # already computes CPU-resident flags for greedy vs sampled requests.
    is_all_greedy = getattr(sampling_info, "is_all_greedy", None)
    if isinstance(is_all_greedy, bool):
        return not is_all_greedy

    if reqs:
        for req in reqs:
            sampling_params = getattr(req, "sampling_params", None)
            if sampling_params is None:
                continue
            try:
                top_k = int(getattr(sampling_params, "top_k", 1))
            except Exception:
                top_k = 1
            try:
                top_p = float(getattr(sampling_params, "top_p", 1.0))
            except Exception:
                top_p = 1.0
            try:
                min_p = float(getattr(sampling_params, "min_p", 0.0))
            except Exception:
                min_p = 0.0
            try:
                temperature = float(getattr(sampling_params, "temperature", 1.0))
            except Exception:
                temperature = 1.0

            if top_k <= 0 or top_k > 1:
                return True
            if top_p < 0.999999:
                return True
            if temperature > 0.0 and top_k <= 0:
                return True
            if min_p > 0.0 and top_k > 1:
                return True

    return False


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


def _get_text_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("text_config", config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            resolved = get_text_config()
            if resolved is not None:
                return resolved
        except TypeError:
            pass
    return config


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


@dataclass(frozen=True)
class DFlashCaptureContract:
    runtime_target_num_layers: int
    draft_num_layers: int
    trained_target_num_layers: Optional[int]
    block_size: Optional[int]
    mask_token: str
    mask_token_id: Optional[int]
    target_layer_ids: List[int]
    capture_layer_ids: List[int]

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "runtime_target_num_layers": int(self.runtime_target_num_layers),
            "draft_num_layers": int(self.draft_num_layers),
            "trained_target_num_layers": (
                None
                if self.trained_target_num_layers is None
                else int(self.trained_target_num_layers)
            ),
            "block_size": None if self.block_size is None else int(self.block_size),
            "mask_token": self.mask_token,
            "mask_token_id": (
                None if self.mask_token_id is None else int(self.mask_token_id)
            ),
            "target_layer_ids": [int(x) for x in self.target_layer_ids],
            "capture_layer_ids": [int(x) for x in self.capture_layer_ids],
        }


def parse_dflash_draft_config(*, draft_hf_config: Any) -> DFlashDraftConfig:
    """Parse and validate DFLASH draft config fields from HF config/dict."""
    dflash_cfg = _get_dflash_config(draft_hf_config)
    draft_text_config = _get_text_config(draft_hf_config)

    num_hidden_layers = _parse_optional_int(
        _cfg_get(draft_text_config, "num_hidden_layers", None),
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


def resolve_dflash_capture_contract(
    *,
    draft_hf_config: Any,
    runtime_target_num_layers: int,
) -> DFlashCaptureContract:
    runtime_target_num_layers = int(runtime_target_num_layers)
    if runtime_target_num_layers <= 0:
        raise ValueError(
            "runtime_target_num_layers must be positive, "
            f"got {runtime_target_num_layers}."
        )

    draft_config = parse_dflash_draft_config(draft_hf_config=draft_hf_config)
    draft_num_layers = draft_config.require_num_layers()
    target_layer_ids = draft_config.resolve_target_layer_ids(
        target_num_layers=runtime_target_num_layers,
        draft_num_layers=draft_num_layers,
    )
    override = (os.environ.get("SGLANG_DFLASH_CAPTURE_BOUNDARIES_OVERRIDE") or "").strip()
    if override:
        try:
            boundaries = [int(x.strip()) for x in override.split(",") if x.strip()]
        except Exception:
            boundaries = []
        if boundaries:
            logger.warning(
                "DFLASH capture boundary override enabled via SGLANG_DFLASH_CAPTURE_BOUNDARIES_OVERRIDE=%r. "
                "This bypasses the default +1 mapping and can reduce acceptance if it mismatches the checkpoint.",
                override,
            )
            return DFlashCaptureContract(
                runtime_target_num_layers=runtime_target_num_layers,
                draft_num_layers=draft_num_layers,
                trained_target_num_layers=draft_config.num_target_layers,
                block_size=draft_config.block_size,
                mask_token=draft_config.mask_token,
                mask_token_id=draft_config.mask_token_id,
                target_layer_ids=target_layer_ids,
                capture_layer_ids=[int(x) for x in boundaries],
            )
    return DFlashCaptureContract(
        runtime_target_num_layers=runtime_target_num_layers,
        draft_num_layers=draft_num_layers,
        trained_target_num_layers=draft_config.num_target_layers,
        block_size=draft_config.block_size,
        mask_token=draft_config.mask_token,
        mask_token_id=draft_config.mask_token_id,
        target_layer_ids=target_layer_ids,
        capture_layer_ids=[int(x) + 1 for x in target_layer_ids],
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


def resolve_dflash_mask_token(*, draft_hf_config: Any) -> str:
    return parse_dflash_draft_config(draft_hf_config=draft_hf_config).mask_token


def resolve_dflash_mask_token_id(*, draft_hf_config: Any) -> Optional[int]:
    return parse_dflash_draft_config(draft_hf_config=draft_hf_config).mask_token_id


@dataclass(frozen=True)
class DFlashTargetOnlyCommitResult:
    commit_len: int
    new_verified_token: int
    accepted_draft_tokens: int
    used_device_defaults: bool


@dataclass(frozen=True)
class DFlashPackedTargetOnlyCommits:
    proposed_flat: torch.Tensor
    commit_lens: torch.Tensor
    commit_offsets: torch.Tensor
    default_new_verified_id: torch.Tensor


@dataclass(frozen=True)
class DFlashPackedIndexedCommits:
    proposed_flat: torch.Tensor
    commit_lens: torch.Tensor
    commit_offsets: torch.Tensor
    default_new_verified_id: torch.Tensor


@dataclass(frozen=True)
class DFlashTargetOnlyCommitMetadata:
    commit_lens: torch.Tensor
    new_verified_id: torch.Tensor


@dataclass(frozen=True)
class DFlashTargetOnlyCachePlan:
    keep_mask: torch.Tensor
    accepted_indices: torch.Tensor
    compact_out_cache_loc: torch.Tensor
    evicted_slots: torch.Tensor
    evicted_pages: torch.Tensor | None
    clear_start: torch.Tensor
    clear_end: torch.Tensor
    clear_token_count: int


@dataclass(frozen=True)
class DFlashIndexedCachePlan:
    accepted_indices: torch.Tensor
    compact_out_cache_loc: torch.Tensor
    evicted_slots: torch.Tensor
    evicted_pages: torch.Tensor | None


@dataclass(frozen=True)
class DFlashSharedPoolAppendPlan:
    ctx_positions: torch.Tensor
    ctx_cache_loc: torch.Tensor
    total_ctx: int


@dataclass(frozen=True)
class _DFlashFastFinishPolicy:
    stop_ids: tuple[int, ...]
    vocab_size: int | None
    nan_replacement_token: int | None


def _build_dflash_fast_finish_policy(req: Any) -> _DFlashFastFinishPolicy | None:
    if getattr(req, "to_finish", None) is not None:
        return None
    if req.finished():
        return None
    if req.grammar is not None:
        return None
    if req.sampling_params.stop_strs or req.sampling_params.stop_regex_strs:
        return None

    # Cache policy per-request to reduce per-verify CPU overhead (called every verify step).
    try:
        cached = getattr(req, "_dflash_fast_finish_policy_cache", None)
        cached_sig = getattr(req, "_dflash_fast_finish_policy_sig", None)
    except Exception:
        cached = None
        cached_sig = None

    stop_ids: set[int] = set()
    nan_replacement_token = None
    if not req.sampling_params.ignore_eos:
        if req.sampling_params.stop_token_ids:
            stop_ids.update(int(tok) for tok in req.sampling_params.stop_token_ids)
            nan_replacement_token = next(iter(req.sampling_params.stop_token_ids))
        if req.eos_token_ids:
            stop_ids.update(int(tok) for tok in req.eos_token_ids)
            if nan_replacement_token is None:
                nan_replacement_token = next(iter(req.eos_token_ids))
        tokenizer = req.tokenizer
        if tokenizer is not None:
            if tokenizer.eos_token_id is not None:
                stop_ids.add(int(tokenizer.eos_token_id))
            additional_stop = getattr(tokenizer, "additional_stop_token_ids", None)
            if additional_stop:
                stop_ids.update(int(tok) for tok in additional_stop)

    vocab_size = getattr(req, "vocab_size", None)
    if vocab_size is not None:
        vocab_size = int(vocab_size)

    sig = None
    try:
        tok = getattr(req, "tokenizer", None)
        sig = (
            bool(getattr(req.sampling_params, "ignore_eos", False)),
            tuple(int(x) for x in (getattr(req.sampling_params, "stop_token_ids", None) or ())),
            tuple(int(x) for x in (getattr(req, "eos_token_ids", None) or ())),
            int(getattr(tok, "eos_token_id", -1)) if tok is not None else -1,
            tuple(int(x) for x in (getattr(tok, "additional_stop_token_ids", None) or ())) if tok is not None else (),
            int(vocab_size) if vocab_size is not None else None,
        )
    except Exception:
        sig = None

    if cached is not None and cached_sig is not None and sig is not None and cached_sig == sig:
        return cached

    policy = _DFlashFastFinishPolicy(
        stop_ids=tuple(sorted(stop_ids)),
        vocab_size=vocab_size,
        nan_replacement_token=(
            int(nan_replacement_token) if nan_replacement_token is not None else None
        ),
    )
    if sig is not None:
        try:
            setattr(req, "_dflash_fast_finish_policy_cache", policy)
            setattr(req, "_dflash_fast_finish_policy_sig", sig)
        except Exception:
            pass
    return policy


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
        used_device_defaults=(appended == len(proposed)),
    )


def commit_dflash_target_only_batch(
    *,
    reqs: List[Any],
    proposed_flat_cpu: torch.Tensor | None = None,
    proposed_dense_cpu: torch.Tensor | None = None,
    commit_lens_cpu: torch.Tensor,
    empty_error_prefix: str = "DFLASH verify",
) -> List[DFlashTargetOnlyCommitResult]:
    """Batch the common plain-request target_only commit path.

    Requests that need grammar, stop-string, stop-regex, or explicit pre-finished
    handling fall back to the exact legacy helper. The common plain path groups
    requests by finish-policy signature and computes truncation/finish events in
    batched CPU tensor form before applying the final per-request mutations.
    """
    from sglang.srt.managers.schedule_batch import (
        FINISH_LENGTH,
        FINISH_MATCHED_STR,
        FINISH_MATCHED_TOKEN,
    )

    if (proposed_flat_cpu is None) == (proposed_dense_cpu is None):
        raise ValueError(
            "commit_dflash_target_only_batch expects exactly one of proposed_flat_cpu "
            "or proposed_dense_cpu."
        )
    if proposed_flat_cpu is not None and proposed_flat_cpu.device.type != "cpu":
        raise ValueError(
            "commit_dflash_target_only_batch expects proposed_flat_cpu on CPU, "
            f"got device={proposed_flat_cpu.device}."
        )
    if proposed_dense_cpu is not None:
        if proposed_dense_cpu.device.type != "cpu":
            raise ValueError(
                "commit_dflash_target_only_batch expects proposed_dense_cpu on CPU, "
                f"got device={proposed_dense_cpu.device}."
            )
        if proposed_dense_cpu.ndim != 2:
            raise ValueError(
                "commit_dflash_target_only_batch expects proposed_dense_cpu to be 2D, "
                f"got shape={tuple(proposed_dense_cpu.shape)}."
            )
    if commit_lens_cpu.device.type != "cpu":
        raise ValueError(
            "commit_dflash_target_only_batch expects commit_lens_cpu on CPU, "
            f"got device={commit_lens_cpu.device}."
        )

    num_reqs = len(reqs)
    if int(commit_lens_cpu.numel()) != num_reqs:
        raise ValueError(
            "commit_lens_cpu size mismatch: "
            f"expected {num_reqs}, got {int(commit_lens_cpu.numel())}."
        )
    if proposed_dense_cpu is not None and int(proposed_dense_cpu.shape[0]) != num_reqs:
        raise ValueError(
            "proposed_dense_cpu batch size mismatch: "
            f"expected {num_reqs}, got {int(proposed_dense_cpu.shape[0])}."
        )

    # Ultra-fast path: plain decode-only requests (no grammar, no stop strings/regex,
    # no stop-token ids) with max_new_tokens truncation only. This is the dominant
    # DFLASH benchmark lane and avoids per-step policy construction and large tolist()
    # materializations.
    try:
        all_plain = True
        for req in reqs:
            sp = req.sampling_params
            if req.grammar is not None:
                all_plain = False
                break
            if sp.stop_strs or sp.stop_regex_strs:
                all_plain = False
                break
            if not bool(getattr(sp, "ignore_eos", False)):
                all_plain = False
                break
            # NOTE: stop_token_ids are semantically independent from ignore_eos in SGLang.
            if getattr(sp, "stop_token_ids", None):
                all_plain = False
                break

        if all_plain:
            results: list[DFlashTargetOnlyCommitResult] = []
            flat_offset = 0
            for i, req in enumerate(reqs):
                default_len = int(commit_lens_cpu[i].item())
                old_output_len = len(req.output_ids)
                remaining = int(req.sampling_params.max_new_tokens) - old_output_len
                effective_len = min(default_len, max(remaining, 0))
                if effective_len > 0:
                    if proposed_dense_cpu is not None:
                        tokens = proposed_dense_cpu[i, :effective_len].tolist()
                    else:
                        assert proposed_flat_cpu is not None
                        tokens = proposed_flat_cpu[flat_offset : flat_offset + effective_len].tolist()
                        flat_offset += default_len
                    req.output_ids.extend(tokens)
                    req.check_finished(new_accepted_len=effective_len)
                else:
                    if proposed_flat_cpu is not None:
                        flat_offset += default_len

                if req.output_ids:
                    new_verified_token = int(req.output_ids[-1])
                elif req.origin_input_ids:
                    new_verified_token = int(req.origin_input_ids[-1])
                else:
                    raise RuntimeError(
                        f"{empty_error_prefix} cannot determine current token: both output_ids and "
                        "origin_input_ids are empty."
                    )

                accepted_draft_tokens = max(0, effective_len - 1)
                req.spec_verify_ct += 1
                req.spec_accepted_tokens += accepted_draft_tokens
                if hasattr(req, "update_spec_acceptance_histogram"):
                    req.update_spec_acceptance_histogram(accepted_draft_tokens)

                results.append(
                    DFlashTargetOnlyCommitResult(
                        commit_len=effective_len,
                        new_verified_token=new_verified_token,
                        accepted_draft_tokens=accepted_draft_tokens,
                        used_device_defaults=bool(effective_len == default_len),
                    )
                )
            return results
    except Exception:
        # Fall back to the fully general path below.
        pass

    # Avoid materializing the full packed proposal list in Python for every step.
    # We keep the CPU tensors and slice per-request (segments are small: <= block_size).
    commit_lens_list = [int(x) for x in commit_lens_cpu.tolist()]
    commit_offsets_list: list[int] | None = None
    if proposed_flat_cpu is not None:
        commit_offsets_list = [0] * (num_reqs + 1)
        for i, width in enumerate(commit_lens_list):
            commit_offsets_list[i + 1] = commit_offsets_list[i] + int(width)

    def _slice_proposed_tokens(req_index: int, width: int) -> list[int]:
        width = int(width)
        if width <= 0:
            return []
        if proposed_dense_cpu is not None:
            return [int(x) for x in proposed_dense_cpu[req_index, :width].tolist()]
        assert proposed_flat_cpu is not None and commit_offsets_list is not None
        start_offset = int(commit_offsets_list[req_index])
        end_offset = start_offset + width
        return [int(x) for x in proposed_flat_cpu[start_offset:end_offset].tolist()]

    results: list[DFlashTargetOnlyCommitResult | None] = [None] * num_reqs
    grouped_fast_indices: dict[_DFlashFastFinishPolicy, list[int]] = {}

    for i, req in enumerate(reqs):
        policy = _build_dflash_fast_finish_policy(req)
        if policy is None:
            proposed = _slice_proposed_tokens(i, commit_lens_list[i])
            results[i] = commit_dflash_proposed_tokens_to_req(
                req=req,
                proposed=proposed,
                empty_error_prefix=empty_error_prefix,
            )
            continue
        grouped_fast_indices.setdefault(policy, []).append(i)

    for policy, group_indices in grouped_fast_indices.items():
        stop_ids = set(int(tok) for tok in policy.stop_ids)
        vocab_hi = int(policy.vocab_size) if policy.vocab_size is not None else None
        replacement_token = (
            int(policy.nan_replacement_token)
            if policy.nan_replacement_token is not None
            else None
        )
        for req_index in group_indices:
            req = reqs[req_index]
            default_len = int(commit_lens_list[req_index])
            old_output_len = len(req.output_ids)
            remaining = int(req.sampling_params.max_new_tokens) - old_output_len
            effective_len = min(default_len, max(remaining, 0))
            tokens = _slice_proposed_tokens(req_index, effective_len)
            finish_kind = "none"
            finish_pos = -1

            if stop_ids and tokens:
                for pos, tok in enumerate(tokens):
                    if int(tok) in stop_ids:
                        tokens = tokens[: pos + 1]
                        finish_kind = "token"
                        finish_pos = pos
                        break

            if finish_kind == "none" and vocab_hi is not None and tokens:
                for pos, tok in enumerate(tokens):
                    if int(tok) < 0 or int(tok) > vocab_hi:
                        tokens = tokens[: pos + 1]
                        if replacement_token is not None:
                            tokens[-1] = replacement_token
                        finish_kind = "vocab"
                        finish_pos = pos
                        break

            commit_len = len(tokens)
            if (
                finish_kind == "none"
                and commit_len > 0
                and remaining > 0
                and commit_len == remaining
            ):
                finish_kind = "length"
                finish_pos = commit_len - 1

            if tokens:
                req.output_ids.extend(int(tok) for tok in tokens)

            if finish_kind == "token" and tokens:
                matched_token = int(tokens[-1])
                req.finished_reason = FINISH_MATCHED_TOKEN(matched=matched_token)
                req.finished_len = old_output_len + finish_pos + 1
            elif finish_kind == "vocab":
                req.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                req.finished_len = old_output_len + finish_pos + 1
            elif finish_kind == "length":
                req.finished_reason = FINISH_LENGTH(
                    length=req.sampling_params.max_new_tokens
                )
                req.finished_len = req.sampling_params.max_new_tokens

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    f"{empty_error_prefix} cannot determine current token: both output_ids and "
                    "origin_input_ids are empty."
                )

            accepted_draft_tokens = max(0, commit_len - 1)
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accepted_draft_tokens
            if hasattr(req, "update_spec_acceptance_histogram"):
                req.update_spec_acceptance_histogram(accepted_draft_tokens)

            results[req_index] = DFlashTargetOnlyCommitResult(
                commit_len=commit_len,
                new_verified_token=new_verified_token,
                accepted_draft_tokens=accepted_draft_tokens,
                used_device_defaults=bool(
                    commit_len == default_len and finish_kind != "vocab"
                ),
            )

    assert all(result is not None for result in results)
    return results  # type: ignore[return-value]


def materialize_dflash_target_only_commit_metadata(
    *,
    commit_results: List[DFlashTargetOnlyCommitResult],
    device: torch.device,
    default_commit_lens: torch.Tensor | None = None,
    default_new_verified_id: torch.Tensor | None = None,
) -> DFlashTargetOnlyCommitMetadata:
    """Convert CPU-side target-only commit outcomes back into device tensors."""
    if default_commit_lens is not None or default_new_verified_id is not None:
        if default_commit_lens is None or default_new_verified_id is None:
            raise ValueError(
                "Both default_commit_lens and default_new_verified_id must be provided together."
            )

        if len(commit_results) != int(default_commit_lens.shape[0]) or len(
            commit_results
        ) != int(default_new_verified_id.shape[0]):
            raise ValueError(
                "Target-only commit defaults must match the number of commit results."
            )

        override_idx = [
            i
            for i, result in enumerate(commit_results)
            if not result.used_device_defaults
        ]
        if not override_idx:
            return DFlashTargetOnlyCommitMetadata(
                commit_lens=default_commit_lens.to(device=device, dtype=torch.int32),
                new_verified_id=default_new_verified_id.to(
                    device=device, dtype=torch.int64
                ),
            )

        commit_lens = default_commit_lens.to(device=device, dtype=torch.int32).clone()
        new_verified_id = default_new_verified_id.to(
            device=device, dtype=torch.int64
        ).clone()
        override_index_tensor = torch.tensor(
            override_idx, dtype=torch.int64, device=device
        )
        commit_lens[override_index_tensor] = torch.tensor(
            [int(commit_results[i].commit_len) for i in override_idx],
            dtype=torch.int32,
            device=device,
        )
        new_verified_id[override_index_tensor] = torch.tensor(
            [int(commit_results[i].new_verified_token) for i in override_idx],
            dtype=torch.int64,
            device=device,
        )
        return DFlashTargetOnlyCommitMetadata(
            commit_lens=commit_lens,
            new_verified_id=new_verified_id,
        )

    commit_lens = torch.tensor(
        [int(result.commit_len) for result in commit_results],
        dtype=torch.int32,
        device=device,
    )
    new_verified_id = torch.tensor(
        [int(result.new_verified_token) for result in commit_results],
        dtype=torch.int64,
        device=device,
    )
    return DFlashTargetOnlyCommitMetadata(
        commit_lens=commit_lens,
        new_verified_id=new_verified_id,
    )


def _align_dflash_evict_mask_to_page_size_fallback(
    *,
    seq_lens: torch.Tensor,
    evict_mask: torch.Tensor,
    page_size: int,
    num_draft_tokens: int,
) -> torch.Tensor:
    evict_mask = evict_mask.view(-1, int(num_draft_tokens)).clone()
    for bid in range(int(seq_lens.shape[0])):
        seq_len = int(seq_lens[bid].item())
        mask_row = evict_mask[bid]
        num_trues = int(mask_row.sum().item())
        num_false = int(num_draft_tokens) - num_trues
        start = ((seq_len + num_false - 1) // int(page_size)) * int(page_size) - seq_len
        for i in range(max(start, 0), min(start + int(page_size), int(num_draft_tokens))):
            mask_row[i] = False
    return evict_mask.reshape(-1).contiguous()


def build_dflash_target_only_cache_plan(
    *,
    out_cache_loc: torch.Tensor,
    commit_lens: torch.Tensor,
    commit_lens_cpu: List[int] | None = None,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    page_size: int,
) -> DFlashTargetOnlyCachePlan:
    """Derive target-only cache free/compact/clear metadata from committed lengths."""
    if commit_lens.ndim != 1:
        raise ValueError(f"commit_lens must be 1D, got shape={tuple(commit_lens.shape)}")
    bs = int(commit_lens.shape[0])
    if out_cache_loc.ndim == 1:
        out_cache_loc_rows = out_cache_loc.view(bs, int(draft_token_num))
    elif out_cache_loc.ndim == 2 and tuple(out_cache_loc.shape) == (bs, int(draft_token_num)):
        out_cache_loc_rows = out_cache_loc
    else:
        raise ValueError(
            "out_cache_loc must be shaped as [bs * draft_token_num] or [bs, draft_token_num]. "
            f"Got shape={tuple(out_cache_loc.shape)} for bs={bs}, draft_token_num={draft_token_num}."
        )

    keep_mask = torch.arange(
        int(draft_token_num), device=commit_lens.device, dtype=torch.int32
    )[None, :] < commit_lens.to(torch.int32).unsqueeze(1)
    accepted_indices = torch.nonzero(
        keep_mask.reshape(-1), as_tuple=False
    ).reshape(-1).to(torch.int64)
    compact_out_cache_loc = out_cache_loc_rows.reshape(-1)[accepted_indices]
    clear_start = seq_lens + commit_lens.to(seq_lens.dtype)
    clear_end = seq_lens + int(draft_token_num)
    if commit_lens_cpu is not None:
        if len(commit_lens_cpu) != bs:
            raise ValueError(
                "commit_lens_cpu size mismatch: "
                f"expected {bs}, got {len(commit_lens_cpu)}."
            )
        clear_token_count = int(bs * int(draft_token_num) - sum(int(x) for x in commit_lens_cpu))
    else:
        clear_token_count = int((clear_end - clear_start).sum().item())

    if int(page_size) == 1:
        return DFlashTargetOnlyCachePlan(
            keep_mask=keep_mask,
            accepted_indices=accepted_indices,
            compact_out_cache_loc=compact_out_cache_loc,
            evicted_slots=out_cache_loc_rows[~keep_mask],
            evicted_pages=None,
            clear_start=clear_start,
            clear_end=clear_end,
            clear_token_count=clear_token_count,
        )

    evict_mask = (~keep_mask).reshape(-1).contiguous()
    if out_cache_loc_rows.is_cuda:
        from sglang.srt.speculative.spec_utils import align_evict_mask_to_page_size
        from sglang.srt.utils.common import next_power_of_2

        align_evict_mask_to_page_size[(bs,)](
            seq_lens,
            evict_mask,
            int(page_size),
            int(draft_token_num),
            next_power_of_2(int(draft_token_num)),
        )
    else:
        evict_mask = _align_dflash_evict_mask_to_page_size_fallback(
            seq_lens=seq_lens,
            evict_mask=evict_mask,
            page_size=int(page_size),
            num_draft_tokens=int(draft_token_num),
        )

    evicted_slots = out_cache_loc_rows.reshape(-1)[evict_mask]
    evicted_pages = None
    if int(evicted_slots.numel()) % int(page_size) == 0 and int(evicted_slots.numel()) > 0:
        evicted_pages = evicted_slots.view(-1, int(page_size))[:, 0] // int(page_size)

    return DFlashTargetOnlyCachePlan(
        keep_mask=keep_mask,
        accepted_indices=accepted_indices,
        compact_out_cache_loc=compact_out_cache_loc,
        evicted_slots=evicted_slots,
        evicted_pages=evicted_pages,
        clear_start=clear_start,
        clear_end=clear_end,
        clear_token_count=clear_token_count,
    )


def build_dflash_shared_pool_append_plan(
    *,
    draft_seq_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    compact_out_cache_loc: torch.Tensor,
) -> DFlashSharedPoolAppendPlan:
    """Build direct append inputs for the shared-pool target->draft KV path.

    This is the common fast path for GPT-OSS DFlash on the shared req_to_token pool:
    after verify compacts the accepted target slots, we can append the committed target
    hidden states into the draft KV cache by reusing those compact cache locations
    directly instead of rebuilding them from req_to_token again.
    """
    if draft_seq_lens.ndim != 1:
        raise ValueError(
            f"draft_seq_lens must be 1D, got shape={tuple(draft_seq_lens.shape)}"
        )
    if commit_lens.ndim != 1:
        raise ValueError(
            f"commit_lens must be 1D, got shape={tuple(commit_lens.shape)}"
        )
    if int(draft_seq_lens.shape[0]) != int(commit_lens.shape[0]):
        raise ValueError(
            "draft_seq_lens / commit_lens batch size mismatch: "
            f"{tuple(draft_seq_lens.shape)} vs {tuple(commit_lens.shape)}"
        )

    total_ctx = int(commit_lens.to(torch.int64).sum().item())
    device = draft_seq_lens.device
    if total_ctx == 0:
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
        return DFlashSharedPoolAppendPlan(
            ctx_positions=empty_i64,
            ctx_cache_loc=empty_i64,
            total_ctx=0,
        )

    max_ctx = int(commit_lens.max().item())
    r = torch.arange(max_ctx, device=device, dtype=torch.int64)[None, :]
    pos2d = draft_seq_lens.to(torch.int64)[:, None] + r
    mask = r < commit_lens.to(torch.int64)[:, None]
    ctx_positions = pos2d[mask]
    ctx_cache_loc = compact_out_cache_loc.to(torch.int64)
    if int(ctx_cache_loc.numel()) != int(total_ctx):
        raise RuntimeError(
            "DFLASH shared-pool append plan size mismatch: "
            f"total_ctx={total_ctx} compact_cache_locs={int(ctx_cache_loc.numel())}"
        )
    if int(ctx_positions.numel()) != int(total_ctx):
        raise RuntimeError(
            "DFLASH shared-pool append position mismatch: "
            f"total_ctx={total_ctx} positions={int(ctx_positions.numel())}"
        )
    return DFlashSharedPoolAppendPlan(
        ctx_positions=ctx_positions,
        ctx_cache_loc=ctx_cache_loc,
        total_ctx=total_ctx,
    )


def build_dflash_shared_pool_append_plan_from_flat_positions(
    *,
    positions: torch.Tensor,
    accepted_indices: torch.Tensor,
    compact_out_cache_loc: torch.Tensor,
) -> DFlashSharedPoolAppendPlan:
    """Build direct append inputs from the verify-time flat position layout.

    This is the preferred DFlash shared-pool append representation because it uses
    the same compact accepted-index view that already drives hidden selection and
    compact cache slots.
    """
    accepted_indices = accepted_indices.to(dtype=torch.int64, device=positions.device)
    total_ctx = int(accepted_indices.numel())
    if total_ctx == 0:
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=positions.device)
        return DFlashSharedPoolAppendPlan(
            ctx_positions=empty_i64,
            ctx_cache_loc=empty_i64,
            total_ctx=0,
        )

    if positions.ndim != 1:
        positions = positions.reshape(-1)
    ctx_positions = positions.index_select(0, accepted_indices)
    ctx_cache_loc = compact_out_cache_loc.to(torch.int64)
    if int(ctx_cache_loc.numel()) != int(total_ctx):
        raise RuntimeError(
            "DFLASH shared-pool append(flat positions) size mismatch: "
            f"total_ctx={total_ctx} compact_cache_locs={int(ctx_cache_loc.numel())}"
        )
    if int(ctx_positions.numel()) != int(total_ctx):
        raise RuntimeError(
            "DFLASH shared-pool append(flat positions) mismatch: "
            f"total_ctx={total_ctx} positions={int(ctx_positions.numel())}"
        )
    return DFlashSharedPoolAppendPlan(
        ctx_positions=ctx_positions,
        ctx_cache_loc=ctx_cache_loc,
        total_ctx=total_ctx,
    )


def apply_dflash_shared_pool_verify_append(
    *,
    draft_input: Any,
    verify_positions: torch.Tensor,
    hidden_states: torch.Tensor,
    cache_plan: Any,
    commit_lens: torch.Tensor,
    write_selected_hidden: Callable[..., None],
) -> DFlashSharedPoolAppendPlan:
    """Apply the shared-pool direct verify-append contract for DFlash.

    This is the single batched contract both the linear and tree workers should use:
    1. compact accepted flat indices drive position/cache selection
    2. accepted verify hidden rows are projected/materialized into the draft KV pool
    3. draft input state is advanced without staging target_hidden for the next step
    """
    if hidden_states is None:
        raise RuntimeError(
            "DFLASH shared-pool verify append requires verify hidden states, but got None."
        )

    append_plan = build_dflash_shared_pool_append_plan_from_flat_positions(
        positions=verify_positions,
        accepted_indices=cache_plan.accepted_indices,
        compact_out_cache_loc=cache_plan.compact_out_cache_loc,
    )
    if append_plan.total_ctx > 0:
        write_selected_hidden(
            hidden_states=hidden_states,
            accepted_indices=cache_plan.accepted_indices,
            ctx_positions=append_plan.ctx_positions,
            ctx_cache_loc=append_plan.ctx_cache_loc,
        )

    ctx_lens = commit_lens.to(
        dtype=torch.int32,
        device=draft_input.draft_seq_lens.device,
    )
    draft_input.draft_seq_lens = draft_input.draft_seq_lens + ctx_lens
    draft_input.new_seq_lens = draft_input.draft_seq_lens.clone()
    draft_input.ctx_lens = torch.zeros_like(ctx_lens)
    draft_input.target_hidden = hidden_states[:0]
    return append_plan


def resolve_dflash_verify_append_path(
    *,
    appended_from_verify: bool,
    fused_helper_active: bool,
) -> str:
    if not bool(appended_from_verify):
        return "staged"
    return "shared_fused" if bool(fused_helper_active) else "shared_sequential"


def update_dflash_verify_append_path_stats(
    *,
    reqs: list[Any],
    append_path: str,
) -> None:
    for req in reqs:
        req.spec_dflash_verify_append_path_last = append_path
        req.spec_dflash_verify_append_path_fused_ct = int(
            getattr(req, "spec_dflash_verify_append_path_fused_ct", 0)
        ) + (1 if append_path == "shared_fused" else 0)
        req.spec_dflash_verify_append_path_direct_ct = int(
            getattr(req, "spec_dflash_verify_append_path_direct_ct", 0)
        ) + (1 if append_path in ("shared_fused", "shared_sequential") else 0)
        req.spec_dflash_verify_append_path_staged_ct = int(
            getattr(req, "spec_dflash_verify_append_path_staged_ct", 0)
        ) + (1 if append_path == "staged" else 0)


def _update_dflash_req_running_mean(req: Any, attr_name: str, value: float | None) -> None:
    if value is None:
        return
    try:
        value_f = float(value)
    except Exception:
        return
    if not math.isfinite(value_f):
        return
    sum_attr = f"{attr_name}_sum"
    ct_attr = f"{attr_name}_ct"
    new_sum = float(getattr(req, sum_attr, 0.0)) + value_f
    new_ct = int(getattr(req, ct_attr, 0)) + 1
    setattr(req, sum_attr, new_sum)
    setattr(req, ct_attr, new_ct)
    setattr(req, attr_name, new_sum / float(new_ct))


def update_dflash_req_verify_bookkeeping(
    *,
    reqs: list[Any],
    accept_length_per_req_cpu: list[int],
    verify_mode: str,
    append_path: str | None = None,
    verify_ct_attr: str = "spec_verify_ct",
    dflash_debug: dict[str, Any] | None = None,
    draft_conf_debug: dict[str, Any] | None = None,
    max_steps_per_req_cpu: list[int] | None = None,
    default_max_steps: int | None = None,
    effective_draft_token_nums: list[int] | None = None,
    default_effective_draft_token_num: int | None = None,
    effective_step_counts: list[int] | None = None,
    default_effective_step_count: int | None = None,
) -> None:
    verify_mode = str(verify_mode or "target_only")
    sig = None
    if dflash_debug is not None or draft_conf_debug is not None:
        try:
            sig = DFlashDifficultySignals.from_debug(
                verify_mode=verify_mode,
                dflash_debug=dflash_debug,
                draft_conf_debug=draft_conf_debug,
            )
        except Exception:
            sig = None

    debug_scalar_keys = (
        "accept_ratio_mean",
        "tv_mean",
        "p_entropy_mean",
        "q_entropy_mean",
        "p_max_mean",
        "q_max_mean",
    )
    draft_scalar_keys = (
        "q_max_mean_first",
        "q_max_min_first",
        "q_ent_mean_first",
    )

    if append_path is not None:
        update_dflash_verify_append_path_stats(reqs=reqs, append_path=append_path)

    for i, req in enumerate(reqs):
        req.spec_dflash_debug_stat_ct = int(
            getattr(req, "spec_dflash_debug_stat_ct", 0)
        ) + (1 if dflash_debug is not None else 0)
        req.spec_dflash_verify_mode_last = verify_mode

        st = getattr(req, "dflash_difficulty_state", None)
        if st is None:
            st = DFlashReqDifficultyState()
            setattr(req, "dflash_difficulty_state", st)

        accept_len_i = int(accept_length_per_req_cpu[i])
        st.update(
            accept_len=accept_len_i,
            verify_ct=int(getattr(req, verify_ct_attr, 0)),
        )
        if sig is not None:
            st.update_verify_debug(sig)

        req.spec_accept_length_step_last = accept_len_i
        prev_min = getattr(req, "spec_accept_length_step_min", None)
        prev_max = getattr(req, "spec_accept_length_step_max", None)
        req.spec_accept_length_step_min = (
            accept_len_i if prev_min is None else min(int(prev_min), accept_len_i)
        )
        req.spec_accept_length_step_max = (
            accept_len_i if prev_max is None else max(int(prev_max), accept_len_i)
        )

        if max_steps_per_req_cpu is not None and i < len(max_steps_per_req_cpu):
            max_steps_i = int(max_steps_per_req_cpu[i])
        elif default_max_steps is not None:
            max_steps_i = int(default_max_steps)
        else:
            max_steps_i = None

        if (
            effective_draft_token_nums is not None
            and i < len(effective_draft_token_nums)
        ):
            effective_draft_token_num_i = int(effective_draft_token_nums[i])
        elif default_effective_draft_token_num is not None:
            effective_draft_token_num_i = int(default_effective_draft_token_num)
        else:
            effective_draft_token_num_i = None

        if effective_step_counts is not None and i < len(effective_step_counts):
            effective_step_count_i = int(effective_step_counts[i])
        elif default_effective_step_count is not None:
            effective_step_count_i = int(default_effective_step_count)
        elif effective_draft_token_num_i is not None:
            effective_step_count_i = max(0, int(effective_draft_token_num_i) - 1)
        else:
            effective_step_count_i = None

        if max_steps_i is not None:
            req.spec_dflash_max_steps_last = max_steps_i
            prev_min_steps = getattr(req, "spec_dflash_max_steps_min", None)
            prev_max_steps = getattr(req, "spec_dflash_max_steps_max", None)
            req.spec_dflash_max_steps_min = (
                max_steps_i
                if prev_min_steps is None
                else min(int(prev_min_steps), max_steps_i)
            )
            req.spec_dflash_max_steps_max = (
                max_steps_i
                if prev_max_steps is None
                else max(int(prev_max_steps), max_steps_i)
            )
            _update_dflash_req_running_mean(req, "spec_dflash_max_steps_mean", max_steps_i)

        if effective_draft_token_num_i is not None:
            req.spec_dflash_effective_draft_token_num_last = effective_draft_token_num_i
            prev_min_token_num = getattr(
                req, "spec_dflash_effective_draft_token_num_min", None
            )
            prev_max_token_num = getattr(
                req, "spec_dflash_effective_draft_token_num_max", None
            )
            req.spec_dflash_effective_draft_token_num_min = (
                effective_draft_token_num_i
                if prev_min_token_num is None
                else min(int(prev_min_token_num), effective_draft_token_num_i)
            )
            req.spec_dflash_effective_draft_token_num_max = (
                effective_draft_token_num_i
                if prev_max_token_num is None
                else max(int(prev_max_token_num), effective_draft_token_num_i)
            )
            _update_dflash_req_running_mean(
                req,
                "spec_dflash_effective_draft_token_num_mean",
                effective_draft_token_num_i,
            )

        if effective_step_count_i is not None:
            req.spec_dflash_effective_step_count_last = effective_step_count_i
            prev_min_eff_steps = getattr(
                req, "spec_dflash_effective_step_count_min", None
            )
            prev_max_eff_steps = getattr(
                req, "spec_dflash_effective_step_count_max", None
            )
            req.spec_dflash_effective_step_count_min = (
                effective_step_count_i
                if prev_min_eff_steps is None
                else min(int(prev_min_eff_steps), effective_step_count_i)
            )
            req.spec_dflash_effective_step_count_max = (
                effective_step_count_i
                if prev_max_eff_steps is None
                else max(int(prev_max_eff_steps), effective_step_count_i)
            )
            _update_dflash_req_running_mean(
                req,
                "spec_dflash_effective_step_count_mean",
                effective_step_count_i,
            )
            req.spec_dflash_total_draft_token_num = int(
                getattr(req, "spec_dflash_total_draft_token_num", 0)
            ) + int(effective_step_count_i)

        if dflash_debug is not None:
            for key in debug_scalar_keys:
                _update_dflash_req_running_mean(
                    req, f"spec_dflash_{key}", dflash_debug.get(key)
                )
        if sig is not None:
            _update_dflash_req_running_mean(
                req, "spec_dflash_accept_ratio_mean", sig.accept_mean
            )
            _update_dflash_req_running_mean(req, "spec_dflash_tv_mean", sig.tv_mean)
            _update_dflash_req_running_mean(
                req, "spec_dflash_p_entropy_mean", sig.p_entropy_mean
            )
            _update_dflash_req_running_mean(
                req, "spec_dflash_q_entropy_mean", sig.q_entropy_mean
            )
            _update_dflash_req_running_mean(
                req, "spec_dflash_p_max_mean", sig.p_max_mean
            )
            _update_dflash_req_running_mean(
                req, "spec_dflash_q_max_mean", sig.q_max_mean
            )
        if draft_conf_debug is not None:
            for key in draft_scalar_keys:
                _update_dflash_req_running_mean(
                    req, f"spec_dflash_{key}", draft_conf_debug.get(key)
                )


def apply_dflash_target_only_cache_plan(
    *,
    batch: Any,
    cache_plan: DFlashTargetOnlyCachePlan,
    page_size: int,
    debug_page_free: bool = False,
) -> None:
    """Apply the target-only cache free/compact plan to the batch allocator state."""
    if int(page_size) == 1:
        batch.token_to_kv_pool_allocator.free(cache_plan.evicted_slots)
        batch.out_cache_loc = cache_plan.compact_out_cache_loc
        return

    if int(cache_plan.evicted_slots.numel()) > 0:
        fast_free_used = False
        if (
            hasattr(batch.token_to_kv_pool_allocator, "free_page_indices")
            and cache_plan.evicted_pages is not None
        ):
            if debug_page_free:
                page_rows = cache_plan.evicted_slots.view(-1, int(page_size))
                same_page = (page_rows // int(page_size)) == cache_plan.evicted_pages[:, None]
                if not bool(torch.all(same_page)):
                    raise RuntimeError(
                        "DFLASH paged fast-free invariant failed: evicted slots are not page-aligned."
                    )
            batch.token_to_kv_pool_allocator.free_page_indices(cache_plan.evicted_pages)
            fast_free_used = True
        if not fast_free_used:
            batch.token_to_kv_pool_allocator.free(cache_plan.evicted_slots)

    batch.out_cache_loc = cache_plan.compact_out_cache_loc


def apply_dflash_target_only_req_kv_accounting(
    *,
    reqs: List[Any],
    commit_lens_cpu: List[int],
    preserve_allocated_len: bool = False,
) -> None:
    for req, commit_len in zip(reqs, commit_lens_cpu, strict=True):
        req.decode_batch_idx += int(commit_len)
        req.kv_committed_len += int(commit_len)
        if preserve_allocated_len:
            req.kv_allocated_len = max(
                int(req.kv_allocated_len), int(req.kv_committed_len)
            )
        else:
            req.kv_allocated_len = req.kv_committed_len


def _assign_dflash_req_to_token_packed_direct(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    lengths: torch.Tensor,
    out_cache_loc: torch.Tensor,
    max_len_hint: int | None = None,
) -> None:
    if int(lengths.numel()) == 0 or int(out_cache_loc.numel()) == 0:
        return
    lengths_i64 = lengths.to(device=req_pool_indices.device, dtype=torch.int64)
    max_len = (
        int(max_len_hint)
        if max_len_hint is not None
        else int(lengths_i64.max().item())
    )
    if max_len <= 0:
        return
    offs = torch.arange(max_len, device=req_pool_indices.device, dtype=torch.int64)
    mask = offs.unsqueeze(0) < lengths_i64.unsqueeze(1)
    row_ids = req_pool_indices.to(torch.int64).unsqueeze(1).expand(-1, max_len)[mask]
    col_ids = (
        start_offset.to(device=req_pool_indices.device, dtype=torch.int64)
        .unsqueeze(1)
        .add(offs.unsqueeze(0))
    )[mask]
    if int(row_ids.numel()) != int(out_cache_loc.numel()):
        raise RuntimeError(
            "DFLASH packed req_to_token write size mismatch: "
            f"rows={int(row_ids.numel())} values={int(out_cache_loc.numel())}."
        )
    req_to_token[row_ids, col_ids] = out_cache_loc.to(
        device=req_to_token.device, dtype=req_to_token.dtype
    )


def apply_dflash_commit_mapping_updates(
    *,
    batch: Any,
    commit_lens: torch.Tensor,
    commit_lens_cpu: List[int],
    commit_lens_cpu_tensor: torch.Tensor | None = None,
    clear_start: torch.Tensor | None = None,
    clear_end: torch.Tensor | None = None,
    clear_token_count: int = 0,
    draft_token_num: int | None = None,
) -> None:
    from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

    bs = int(commit_lens.shape[0])
    end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
    use_direct_packed_write = (
        bs <= 16 and int(batch.out_cache_loc.numel()) <= 512
    )
    if use_direct_packed_write:
        _assign_dflash_req_to_token_packed_direct(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            lengths=commit_lens.to(batch.seq_lens.dtype),
            out_cache_loc=batch.out_cache_loc,
            max_len_hint=max((int(x) for x in commit_lens_cpu), default=0),
        )
    else:
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )
    if int(clear_token_count) > 0:
        if clear_start is None or clear_end is None:
            raise ValueError(
                "clear_start and clear_end must be provided when clear_token_count > 0."
            )
        pad_locs = torch.zeros(
            (int(clear_token_count),),
            dtype=torch.int64,
            device=commit_lens.device,
        )
        clear_lengths = clear_end - clear_start
        if use_direct_packed_write:
            _assign_dflash_req_to_token_packed_direct(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=batch.req_to_token_pool.req_to_token,
                start_offset=clear_start,
                lengths=clear_lengths,
                out_cache_loc=pad_locs,
                max_len_hint=(
                    max(
                        (
                            max(0, int(draft_token_num) - int(commit_len))
                            for commit_len in commit_lens_cpu
                        ),
                        default=0,
                    )
                    if draft_token_num is not None
                    else None
                ),
            )
        else:
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                clear_start,
                clear_end,
                pad_locs,
                bs,
            )

    batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
    if commit_lens_cpu_tensor is not None:
        batch.seq_lens_cpu.add_(
            commit_lens_cpu_tensor.to(dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += int(commit_lens_cpu_tensor.sum().item())
    else:
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)


def apply_dflash_target_only_mapping_updates(
    *,
    batch: Any,
    commit_lens: torch.Tensor,
    commit_lens_cpu: List[int],
    commit_lens_cpu_tensor: torch.Tensor | None = None,
    cache_plan: DFlashTargetOnlyCachePlan,
    draft_token_num: int | None = None,
) -> None:
    apply_dflash_commit_mapping_updates(
        batch=batch,
        commit_lens=commit_lens,
        commit_lens_cpu=commit_lens_cpu,
        commit_lens_cpu_tensor=commit_lens_cpu_tensor,
        clear_start=cache_plan.clear_start,
        clear_end=cache_plan.clear_end,
        clear_token_count=cache_plan.clear_token_count,
        draft_token_num=draft_token_num,
    )


def build_dflash_indexed_cache_plan(
    *,
    out_cache_loc: torch.Tensor,
    accepted_indices: torch.Tensor,
    page_size: int = 1,
    borrowed_out_cache_loc: bool = False,
) -> DFlashIndexedCachePlan:
    out_cache_loc = out_cache_loc.to(dtype=torch.int64, device=out_cache_loc.device)
    if out_cache_loc.ndim != 1:
        raise ValueError(
            f"out_cache_loc must be 1D, got shape={tuple(out_cache_loc.shape)}."
        )
    if bool((out_cache_loc <= 0).any().detach().cpu().item()):
        raise RuntimeError(
            "DFLASH indexed cache plan received non-positive KV slots: "
            f"slots={out_cache_loc.detach().to('cpu', non_blocking=False).tolist()}"
        )
    if int(out_cache_loc.numel()) > 1:
        sorted_slots = out_cache_loc.sort().values
        if bool((sorted_slots[1:] == sorted_slots[:-1]).any().detach().cpu().item()):
            raise RuntimeError(
                "DFLASH indexed cache plan received duplicate KV slots: "
                f"slots={out_cache_loc.detach().to('cpu', non_blocking=False).tolist()}"
            )

    accepted_indices = accepted_indices.to(dtype=torch.int64, device=out_cache_loc.device)
    if accepted_indices.ndim != 1:
        raise ValueError(
            f"accepted_indices must be 1D, got shape={tuple(accepted_indices.shape)}."
        )
    if bool((accepted_indices < 0).any().detach().cpu().item()):
        raise RuntimeError(
            "DFLASH indexed cache plan received negative accepted indices: "
            f"indices={accepted_indices.detach().to('cpu', non_blocking=False).tolist()}"
        )
    if bool((accepted_indices >= int(out_cache_loc.numel())).any().detach().cpu().item()):
        raise RuntimeError(
            "DFLASH indexed cache plan received out-of-range accepted indices: "
            f"indices={accepted_indices.detach().to('cpu', non_blocking=False).tolist()} "
            f"out_cache_loc_len={int(out_cache_loc.numel())}"
        )
    if int(accepted_indices.numel()) > 1:
        sorted_indices = accepted_indices.sort().values
        if not bool(torch.equal(sorted_indices, accepted_indices)):
            raise RuntimeError(
                "DFLASH indexed cache plan requires accepted_indices to be sorted. "
                f"indices={accepted_indices.detach().to('cpu', non_blocking=False).tolist()}"
            )
        if bool((sorted_indices[1:] == sorted_indices[:-1]).any().detach().cpu().item()):
            raise RuntimeError(
                "DFLASH indexed cache plan received duplicate accepted indices: "
                f"indices={accepted_indices.detach().to('cpu', non_blocking=False).tolist()}"
            )
    evict_mask = torch.ones_like(out_cache_loc, dtype=torch.bool)
    if accepted_indices.numel() > 0:
        evict_mask[accepted_indices] = False
    evicted_slots = out_cache_loc[evict_mask]
    evicted_pages = None
    if borrowed_out_cache_loc:
        evicted_slots = out_cache_loc[:0]
    elif int(page_size) != 1 and int(evicted_slots.numel()) > 0:
        evicted_page_candidates = torch.unique(evicted_slots // int(page_size))
        if accepted_indices.numel() > 0:
            kept_pages = torch.unique(
                out_cache_loc.index_select(0, accepted_indices) // int(page_size)
            )
            page_keep_mask = ~(
                evicted_page_candidates[:, None] == kept_pages[None, :]
            ).any(dim=1)
            evicted_pages = evicted_page_candidates[page_keep_mask]
        else:
            evicted_pages = evicted_page_candidates
    return DFlashIndexedCachePlan(
        accepted_indices=accepted_indices,
        compact_out_cache_loc=out_cache_loc[accepted_indices],
        evicted_slots=evicted_slots,
        evicted_pages=evicted_pages,
    )


def apply_dflash_indexed_cache_plan(
    *,
    batch: Any,
    cache_plan: DFlashIndexedCachePlan,
    page_size: int = 1,
) -> None:
    if int(cache_plan.evicted_slots.numel()) == 0:
        batch.out_cache_loc = cache_plan.compact_out_cache_loc
        return

    fast_free_used = False
    if (
        int(page_size) != 1
        and cache_plan.evicted_pages is not None
        and hasattr(batch.token_to_kv_pool_allocator, "free_page_indices")
    ):
        if int(cache_plan.evicted_pages.numel()) > 0:
            batch.token_to_kv_pool_allocator.free_page_indices(cache_plan.evicted_pages)
        fast_free_used = True
    if not fast_free_used:
        batch.token_to_kv_pool_allocator.free(cache_plan.evicted_slots)
    batch.out_cache_loc = cache_plan.compact_out_cache_loc


def gather_dflash_committed_hidden(
    *,
    hidden_states: torch.Tensor,
    keep_mask: torch.Tensor | None = None,
    draft_token_num: int | None = None,
    accepted_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather committed verify hidden states using the same compact keep mask."""
    if hidden_states is None:
        raise RuntimeError("DFLASH verify requires target hidden states, but got None.")
    if accepted_indices is not None:
        if hidden_states.ndim == 2:
            flat_hidden = hidden_states
        elif hidden_states.ndim == 3:
            flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        else:
            raise ValueError(
                f"hidden_states must be 2D or 3D, got shape={tuple(hidden_states.shape)}"
            )
        accepted_indices = accepted_indices.to(
            dtype=torch.int64, device=flat_hidden.device
        )
        return (
            flat_hidden.index_select(0, accepted_indices)
            if accepted_indices.numel() > 0
            else flat_hidden[:0]
        )
    if keep_mask is None or draft_token_num is None:
        raise ValueError(
            "keep_mask and draft_token_num are required when accepted_indices is not provided."
        )
    if hidden_states.ndim == 2:
        bs = int(keep_mask.shape[0])
        hidden = hidden_states.view(bs, int(draft_token_num), -1)
    elif hidden_states.ndim == 3:
        hidden = hidden_states
    else:
        raise ValueError(
            f"hidden_states must be 2D or 3D, got shape={tuple(hidden_states.shape)}"
        )
    return hidden[keep_mask]


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


def build_dflash_tree_candidates_from_per_step_topk(
    *,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    num_verify_tokens: int,
    candidate_scores_buf: torch.Tensor | None = None,
    candidate_tokens_buf: torch.Tensor | None = None,
    parent_list_buf: torch.Tensor | None = None,
    top_scores_index_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build DFLASH_TREE candidates directly from per-step top-k tensors.

    Unlike EAGLE's sequential draft loop, DFLASH_TREE already has one non-causal draft
    block and per-position top-k candidates. The tree builder therefore only needs to
    enumerate the bounded beam-style combinations across steps and rerank them into the
    verify-node budget.

    Returns:
        parent_list: tree parent encoding matching `build_tree_kernel_efficient`.
        top_scores_index: sorted flat candidate indices selected for verification.
        draft_tokens: token ids for the selected verify nodes (excluding the root token).
    """
    if topk_p.ndim != 3:
        raise ValueError(f"topk_p must be 3D, got shape={tuple(topk_p.shape)}.")
    if topk_index.ndim != 3:
        raise ValueError(
            f"topk_index must be 3D, got shape={tuple(topk_index.shape)}."
        )
    if tuple(topk_p.shape) != tuple(topk_index.shape):
        raise ValueError(
            "topk_p and topk_index must have the same shape, got "
            f"{tuple(topk_p.shape)} vs {tuple(topk_index.shape)}."
        )

    bs, step_count, topk = (int(topk_p.shape[0]), int(topk_p.shape[1]), int(topk_p.shape[2]))
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}.")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    if int(num_verify_tokens) <= 0:
        raise ValueError(
            f"num_verify_tokens must be positive, got {num_verify_tokens}."
        )

    candidate_count = topk + max(0, step_count - 1) * (topk * topk)
    max_verify_tokens = 1 + candidate_count
    if int(num_verify_tokens) > int(max_verify_tokens):
        raise ValueError(
            "DFLASH_TREE verify-node budget is too large for the provided per-step top-k tensors. "
            f"num_verify_tokens={int(num_verify_tokens)}, max_allowed={int(max_verify_tokens)}, "
            f"step_count={step_count}, topk={topk}."
        )

    if candidate_scores_buf is None:
        flat_scores_all = torch.empty(
            (bs, candidate_count), dtype=torch.float32, device=topk_p.device
        )
    else:
        if candidate_scores_buf.ndim != 2 or int(candidate_scores_buf.shape[0]) != bs:
            raise ValueError(
                "candidate_scores_buf must be 2D with shape [bs, >=candidate_count], got "
                f"{tuple(candidate_scores_buf.shape)} for bs={bs}."
            )
        if int(candidate_scores_buf.shape[1]) < int(candidate_count):
            raise ValueError(
                "candidate_scores_buf is too small: "
                f"need {candidate_count}, got {int(candidate_scores_buf.shape[1])}."
            )
        flat_scores_all = candidate_scores_buf[:, :candidate_count]

    if candidate_tokens_buf is None:
        flat_tokens_all = torch.empty(
            (bs, candidate_count), dtype=torch.int64, device=topk_index.device
        )
    else:
        if candidate_tokens_buf.ndim != 2 or int(candidate_tokens_buf.shape[0]) != bs:
            raise ValueError(
                "candidate_tokens_buf must be 2D with shape [bs, >=candidate_count], got "
                f"{tuple(candidate_tokens_buf.shape)} for bs={bs}."
            )
        if int(candidate_tokens_buf.shape[1]) < int(candidate_count):
            raise ValueError(
                "candidate_tokens_buf is too small: "
                f"need {candidate_count}, got {int(candidate_tokens_buf.shape[1])}."
            )
        flat_tokens_all = candidate_tokens_buf[:, :candidate_count]

    parent_count = 0 if step_count == 1 else (topk + 1) + max(0, step_count - 2) * topk
    if parent_count == 0:
        parent_list = torch.empty((bs, 0), dtype=torch.long, device=topk_index.device)
    elif parent_list_buf is None:
        parent_list = torch.empty(
            (bs, parent_count), dtype=torch.long, device=topk_index.device
        )
    else:
        if parent_list_buf.ndim != 2 or int(parent_list_buf.shape[0]) != bs:
            raise ValueError(
                "parent_list_buf must be 2D with shape [bs, >=parent_count], got "
                f"{tuple(parent_list_buf.shape)} for bs={bs}."
            )
        if int(parent_list_buf.shape[1]) < int(parent_count):
            raise ValueError(
                "parent_list_buf is too small: "
                f"need {parent_count}, got {int(parent_list_buf.shape[1])}."
            )
        parent_list = parent_list_buf[:, :parent_count]

    flat_scores_all[:, :topk].copy_(topk_p[:, 0, :].to(torch.float32))
    flat_tokens_all[:, :topk].copy_(topk_index[:, 0, :].to(torch.int64))
    beam_scores = flat_scores_all[:, :topk]

    if parent_count > 0:
        first_parents = torch.arange(
            -1, topk, dtype=torch.long, device=topk_index.device
        ).unsqueeze(0)
        parent_list[:, : topk + 1].copy_(first_parents.expand(bs, -1))
        parent_offset = topk + 1
    else:
        parent_offset = 0

    for step_idx in range(1, step_count):
        step_scores = topk_p[:, step_idx, :].to(torch.float32)
        start = topk + (step_idx - 1) * (topk * topk)
        end = start + topk * topk

        flat_scores = flat_scores_all[:, start:end]
        flat_scores_view = flat_scores.view(bs, topk, topk)
        flat_scores_view.copy_(beam_scores.unsqueeze(2) * step_scores.unsqueeze(1))

        flat_tokens = flat_tokens_all[:, start:end]
        flat_tokens_view = flat_tokens.view(bs, topk, topk)
        flat_tokens_view.copy_(
            topk_index[:, step_idx, :]
            .to(torch.int64)
            .unsqueeze(1)
            .expand(bs, topk, topk)
        )

        topk_cs_p, topk_cs_index = torch.topk(flat_scores, k=topk, dim=1)
        beam_scores = topk_cs_p

        if step_idx < step_count - 1:
            parent_list[:, parent_offset : parent_offset + topk].copy_(
                topk_cs_index.to(torch.long)
                + (topk * topk * (step_idx - 1) + topk)
            )
            parent_offset += topk

    top_scores_index = torch.topk(
        flat_scores_all, k=int(num_verify_tokens) - 1, dim=-1
    ).indices
    top_scores_index = torch.sort(top_scores_index, dim=-1).values
    if top_scores_index_buf is not None:
        if (
            top_scores_index_buf.ndim != 2
            or int(top_scores_index_buf.shape[0]) != bs
            or int(top_scores_index_buf.shape[1]) < int(num_verify_tokens) - 1
        ):
            raise ValueError(
                "top_scores_index_buf must be 2D with shape [bs, >=num_verify_tokens-1], got "
                f"{tuple(top_scores_index_buf.shape)} for bs={bs} and num_verify_tokens={num_verify_tokens}."
            )
        top_scores_index_buf[:, : int(num_verify_tokens) - 1].copy_(top_scores_index)
        top_scores_index = top_scores_index_buf[:, : int(num_verify_tokens) - 1]
    draft_tokens = torch.gather(flat_tokens_all, index=top_scores_index, dim=1)

    return parent_list, top_scores_index, draft_tokens


def sample_dflash_tree_branch_candidates(
    *,
    probs: torch.Tensor,
    logits: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-row tree branch candidates without a Python row loop."""
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape={tuple(probs.shape)}.")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape={tuple(logits.shape)}.")
    if tuple(probs.shape) != tuple(logits.shape):
        raise ValueError(
            "probs and logits must have the same shape, got "
            f"{tuple(probs.shape)} vs {tuple(logits.shape)}."
        )
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")

    row_count, vocab_size = int(probs.shape[0]), int(probs.shape[1])
    sampled_ids = torch.empty(
        (row_count, int(topk)), dtype=torch.int64, device=probs.device
    )
    sampled_probs = torch.empty(
        (row_count, int(topk)), dtype=torch.float32, device=probs.device
    )
    if row_count == 0:
        return sampled_probs, sampled_ids

    row_sums = probs.sum(dim=-1)
    has_mass = row_sums > 0
    if bool(torch.any(has_mass)):
        live_rows = torch.nonzero(has_mass, as_tuple=False).squeeze(1)
        live_probs = probs[live_rows]
        nonzero_counts = (live_probs > 0).sum(dim=-1)
        needs_replacement = nonzero_counts < int(topk)

        for replacement in (False, True):
            mode_mask = needs_replacement if replacement else ~needs_replacement
            if not bool(torch.any(mode_mask)):
                continue
            rows = live_rows[mode_mask]
            row_probs = probs[rows]
            row_ids = torch.multinomial(
                row_probs,
                num_samples=int(topk),
                replacement=replacement,
            )
            row_probs_out = row_probs.gather(1, row_ids)
            order = torch.argsort(row_probs_out, descending=True, dim=1)
            sampled_ids[rows] = row_ids.gather(1, order).to(torch.int64)
            sampled_probs[rows] = row_probs_out.gather(1, order).to(torch.float32)

    empty_rows = torch.nonzero(~has_mass, as_tuple=False).squeeze(1)
    if int(empty_rows.numel()) > 0:
        fallback_logits = logits[empty_rows].to(torch.float32)
        fallback_k = min(int(topk), vocab_size)
        fallback_vals, fallback_ids = torch.topk(
            fallback_logits, k=fallback_k, dim=-1
        )
        fallback_probs = torch.softmax(fallback_vals, dim=-1)
        if fallback_k < int(topk):
            pad = int(topk) - fallback_k
            fallback_ids = torch.cat(
                [
                    fallback_ids,
                    torch.zeros(
                        (int(empty_rows.numel()), pad),
                        dtype=torch.int64,
                        device=probs.device,
                    ),
                ],
                dim=1,
            )
            fallback_probs = torch.cat(
                [
                    fallback_probs,
                    torch.zeros(
                        (int(empty_rows.numel()), pad),
                        dtype=torch.float32,
                        device=probs.device,
                    ),
                ],
                dim=1,
            )
        sampled_ids[empty_rows] = fallback_ids.to(torch.int64)
        sampled_probs[empty_rows] = fallback_probs.to(torch.float32)

    return sampled_probs, sampled_ids


def sample_dflash_tree_branch_candidates_from_support(
    *,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    topk: int,
    fallback_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tree branch candidates from a preselected support set."""
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape={tuple(probs.shape)}.")
    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must be 2D, got shape={tuple(token_ids.shape)}."
        )
    if tuple(probs.shape) != tuple(token_ids.shape):
        raise ValueError(
            "probs and token_ids must have the same shape, got "
            f"{tuple(probs.shape)} vs {tuple(token_ids.shape)}."
        )
    if fallback_probs is None:
        fallback_probs = probs
    elif tuple(fallback_probs.shape) != tuple(probs.shape):
        raise ValueError(
            "fallback_probs must match probs shape, got "
            f"{tuple(fallback_probs.shape)} vs {tuple(probs.shape)}."
        )
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")

    row_count, support_k = int(probs.shape[0]), int(probs.shape[1])
    sampled_ids = torch.empty(
        (row_count, int(topk)), dtype=torch.int64, device=probs.device
    )
    sampled_probs = torch.empty(
        (row_count, int(topk)), dtype=torch.float32, device=probs.device
    )
    if row_count == 0:
        return sampled_probs, sampled_ids

    row_sums = probs.sum(dim=-1)
    has_mass = row_sums > 0
    if bool(torch.any(has_mass)):
        live_rows = torch.nonzero(has_mass, as_tuple=False).squeeze(1)
        live_probs = probs[live_rows]
        nonzero_counts = (live_probs > 0).sum(dim=-1)
        needs_replacement = nonzero_counts < int(topk)

        for replacement in (False, True):
            mode_mask = needs_replacement if replacement else ~needs_replacement
            if not bool(torch.any(mode_mask)):
                continue
            rows = live_rows[mode_mask]
            row_probs = probs[rows]
            row_local_ids = torch.multinomial(
                row_probs,
                num_samples=int(topk),
                replacement=replacement,
            )
            row_probs_out = row_probs.gather(1, row_local_ids)
            order = torch.argsort(row_probs_out, descending=True, dim=1)
            row_local_ids = row_local_ids.gather(1, order)
            row_probs_out = row_probs_out.gather(1, order)
            sampled_ids[rows] = token_ids[rows].gather(1, row_local_ids).to(torch.int64)
            sampled_probs[rows] = row_probs_out.to(torch.float32)

    empty_rows = torch.nonzero(~has_mass, as_tuple=False).squeeze(1)
    if int(empty_rows.numel()) > 0:
        fallback_vals, fallback_sel = torch.topk(
            fallback_probs[empty_rows].to(torch.float32),
            k=min(int(topk), support_k),
            dim=-1,
        )
        fallback_ids = token_ids[empty_rows].gather(1, fallback_sel.to(torch.int64))
        if int(fallback_vals.shape[1]) < int(topk):
            pad = int(topk) - int(fallback_vals.shape[1])
            fallback_vals = torch.cat(
                [
                    fallback_vals,
                    torch.zeros(
                        (int(empty_rows.numel()), pad),
                        dtype=torch.float32,
                        device=probs.device,
                    ),
                ],
                dim=1,
            )
            fallback_ids = torch.cat(
                [
                    fallback_ids,
                    torch.zeros(
                        (int(empty_rows.numel()), pad),
                        dtype=torch.int64,
                        device=probs.device,
                    ),
                ],
                dim=1,
            )
        fallback_probs_out = fallback_vals / fallback_vals.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-20)
        sampled_ids[empty_rows] = fallback_ids.to(torch.int64)
        sampled_probs[empty_rows] = fallback_probs_out.to(torch.float32)

    return sampled_probs, sampled_ids


def pack_dflash_target_only_commits(
    *,
    target_predict: torch.Tensor,
    accept_len: torch.Tensor,
) -> DFlashPackedTargetOnlyCommits:
    """Compact the committed target-only token prefix for CPU-side request updates.

    Args:
        target_predict: [bs, draft_token_num] target-predicted tokens.
        accept_len: [bs] accepted draft-token count, excluding the bonus token.

    Returns:
        proposed_flat: flattened committed token prefixes, row-major by request.
        commit_lens: [bs] committed token counts including the bonus token.
        commit_offsets: [bs + 1] flattened-span offsets for each request.
        default_new_verified_id: [bs] last committed token before any CPU-side truncation.
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

    device = target_predict.device
    bs, draft_token_num = target_predict.shape
    commit_lens = accept_len.to(device=device, dtype=torch.int32).clamp(
        min=0, max=int(draft_token_num - 1)
    )
    commit_lens = commit_lens + 1

    try:
        cpu_pack_max = max(
            0,
            int(
                (
                    os.environ.get("SGLANG_DFLASH_TARGET_ONLY_SMALL_CPU_PACK_MAX")
                    or "512"
                ).strip()
            ),
        )
    except Exception:
        cpu_pack_max = 512

    use_cpu_pack = bool(
        target_predict.is_cuda
        and (
            _env_truthy("SGLANG_DFLASH_TARGET_ONLY_PACK_ON_CPU")
            or int(target_predict.numel()) <= cpu_pack_max
        )
    )
    if use_cpu_pack:
        target_predict_cpu = target_predict.to(
            device="cpu", dtype=torch.int64, non_blocking=False
        ).contiguous()
        commit_lens_cpu = commit_lens.to(
            device="cpu", dtype=torch.int32, non_blocking=False
        ).contiguous()
        commit_offsets = torch.empty((bs + 1,), dtype=torch.int64, device="cpu")
        commit_offsets[0] = 0
        commit_offsets[1:].copy_(commit_lens_cpu.to(torch.int64).cumsum(0))
        total_commits = int(commit_offsets[-1].item()) if int(bs) > 0 else 0
        proposed_flat = torch.empty((total_commits,), dtype=torch.int64, device="cpu")
        write_offset = 0
        for row_idx, width in enumerate(commit_lens_cpu.tolist()):
            width = int(width)
            if width <= 0:
                continue
            proposed_flat[write_offset : write_offset + width].copy_(
                target_predict_cpu[row_idx, :width]
            )
            write_offset += width
        if int(bs) > 0:
            default_new_verified_id = target_predict_cpu[
                torch.arange(bs, device="cpu"),
                commit_lens_cpu.to(torch.int64) - 1,
            ].to(torch.int64)
        else:
            default_new_verified_id = torch.empty(
                (0,), dtype=torch.int64, device="cpu"
            )
        return DFlashPackedTargetOnlyCommits(
            proposed_flat=proposed_flat,
            commit_lens=commit_lens_cpu,
            commit_offsets=commit_offsets,
            default_new_verified_id=default_new_verified_id,
        )

    keep_mask = torch.arange(draft_token_num, device=device, dtype=torch.int32)[
        None, :
    ] < commit_lens.unsqueeze(1)
    proposed_flat = target_predict[keep_mask].to(torch.int64)
    commit_offsets = torch.zeros((bs + 1,), dtype=torch.int64, device=device)
    commit_offsets[1:].copy_(commit_lens.to(torch.int64).cumsum(0))
    default_new_verified_id = target_predict[
        torch.arange(bs, device=device),
        commit_lens.to(torch.int64) - 1,
    ].to(torch.int64)
    return DFlashPackedTargetOnlyCommits(
        proposed_flat=proposed_flat,
        commit_lens=commit_lens,
        commit_offsets=commit_offsets,
        default_new_verified_id=default_new_verified_id,
    )


def verify_dflash_tree_greedy_fallback(
    *,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    num_speculative_tokens: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Safe reference implementation of greedy tree verification.

    This mirrors the `verify_tree_greedy` CUDA kernel exactly, but computes on CPU-owned
    snapshots of the tiny tree metadata tensors. It is intended as a correctness fallback
    when the kernel path is unstable for large batched DFLASH_TREE runs.
    """
    if candidates.ndim != 2:
        raise ValueError(
            f"candidates must be 2D, got shape={tuple(candidates.shape)}."
        )
    for name, tensor in (
        ("retrive_index", retrive_index),
        ("retrive_next_token", retrive_next_token),
        ("retrive_next_sibling", retrive_next_sibling),
        ("target_predict", target_predict),
    ):
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}.")
        if tuple(tensor.shape) != tuple(candidates.shape):
            raise ValueError(
                f"{name} shape mismatch: expected {tuple(candidates.shape)}, got {tuple(tensor.shape)}."
            )

    bs, num_draft_tokens = candidates.shape
    if num_speculative_tokens <= 0:
        raise ValueError(
            f"num_speculative_tokens must be positive, got {num_speculative_tokens}."
        )

    out_device = torch.device(device) if device is not None else candidates.device

    candidates_cpu = candidates.detach().to("cpu", dtype=torch.int64, non_blocking=False)
    retrive_index_cpu = retrive_index.detach().to(
        "cpu", dtype=torch.int64, non_blocking=False
    )
    retrive_next_token_cpu = retrive_next_token.detach().to(
        "cpu", dtype=torch.int64, non_blocking=False
    )
    retrive_next_sibling_cpu = retrive_next_sibling.detach().to(
        "cpu", dtype=torch.int64, non_blocking=False
    )
    target_predict_cpu = target_predict.detach().to(
        "cpu", dtype=torch.int64, non_blocking=False
    )
    target_predict_flat_cpu = target_predict_cpu.reshape(-1)

    predicts_cpu = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device="cpu"
    )
    accept_index_cpu = torch.full(
        (bs, num_speculative_tokens), -1, dtype=torch.int32, device="cpu"
    )
    accept_token_num_cpu = torch.zeros((bs,), dtype=torch.int32, device="cpu")

    for bx in range(bs):
        last_accepted_retrive_idx = int(retrive_index_cpu[bx, 0].item())
        accept_index_cpu[bx, 0] = last_accepted_retrive_idx
        num_accepted_tokens = 0
        cur_index = 0

        for j in range(1, num_speculative_tokens):
            cur_index = int(retrive_next_token_cpu[bx, cur_index].item())
            while cur_index != -1:
                draft_index = int(retrive_index_cpu[bx, cur_index].item())
                draft_token_id = int(candidates_cpu[bx, cur_index].item())
                target_token_id = int(target_predict_flat_cpu[last_accepted_retrive_idx].item())

                if draft_token_id == target_token_id:
                    predicts_cpu[last_accepted_retrive_idx] = target_token_id
                    num_accepted_tokens += 1
                    accept_index_cpu[bx, num_accepted_tokens] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break

                cur_index = int(retrive_next_sibling_cpu[bx, cur_index].item())

            if cur_index == -1:
                break

        accept_token_num_cpu[bx] = num_accepted_tokens
        predicts_cpu[last_accepted_retrive_idx] = int(
            target_predict_flat_cpu[last_accepted_retrive_idx].item()
        )

    return (
        predicts_cpu.to(device=out_device, dtype=torch.int32),
        accept_index_cpu.to(device=out_device, dtype=torch.int32),
        accept_token_num_cpu.to(device=out_device, dtype=torch.int32),
    )


def pack_dflash_indexed_commits(
    *,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
) -> DFlashPackedIndexedCommits:
    """Pack tree/indexed verify proposals into compact per-request commit buffers.

    `accept_index` stores, per request, the accepted verify indices followed by the
    bonus token index and then `-1` padding. This helper converts that representation
    into the same compact flat layout the linear target-only path uses so the CPU
    commit layer can batch plain requests and reuse device defaults.
    """
    if predict.ndim != 1:
        raise ValueError(f"predict must be 1D, got shape={tuple(predict.shape)}.")
    if accept_index.ndim != 2:
        raise ValueError(
            f"accept_index must be 2D, got shape={tuple(accept_index.shape)}."
        )

    bs = int(accept_index.shape[0])
    predict_cpu = predict.detach().to("cpu", non_blocking=False)
    accept_index_cpu = accept_index.detach().to("cpu", non_blocking=False)

    valid_mask = accept_index_cpu != -1
    commit_lens = valid_mask.sum(dim=1, dtype=torch.int32)
    commit_offsets = torch.empty((bs + 1,), dtype=torch.int64, device="cpu")
    commit_offsets[0] = 0
    commit_offsets[1:].copy_(commit_lens.to(torch.int64).cumsum(0))

    valid_indices_cpu = accept_index_cpu[valid_mask].to(torch.int64)
    bad_mask = (valid_indices_cpu < 0) | (valid_indices_cpu >= int(predict_cpu.shape[0]))
    if bool(bad_mask.any().item()):
        bad = valid_indices_cpu[bad_mask].tolist()
        raise RuntimeError(
            "DFLASH indexed commit packing produced invalid predict indices: "
            f"indices={bad} predict_len={int(predict_cpu.shape[0])} "
            f"accept_index={accept_index_cpu.tolist()}"
        )
    proposed_flat = predict_cpu.index_select(0, valid_indices_cpu.to(torch.long)).to(
        torch.int64
    )

    last_pos = (commit_lens.to(torch.int64) - 1).clamp(min=0)
    last_offsets = commit_offsets[:-1] + last_pos
    default_new_verified_id = torch.where(
        commit_lens > 0,
        proposed_flat.index_select(0, last_offsets.to(torch.long)),
        torch.zeros((bs,), dtype=torch.int64, device="cpu"),
    )
    return DFlashPackedIndexedCommits(
        proposed_flat=proposed_flat,
        commit_lens=commit_lens,
        commit_offsets=commit_offsets,
        default_new_verified_id=default_new_verified_id,
    )


def resolve_dflash_indexed_accept_indices(
    *,
    accept_index: torch.Tensor,
    commit_lens: torch.Tensor,
) -> torch.Tensor:
    """Resolve final accepted flat verify indices after CPU-side commit truncation."""
    if accept_index.ndim != 2:
        raise ValueError(
            f"accept_index must be 2D, got shape={tuple(accept_index.shape)}."
        )
    if commit_lens.ndim != 1:
        raise ValueError(
            f"commit_lens must be 1D, got shape={tuple(commit_lens.shape)}."
        )
    if int(accept_index.shape[0]) != int(commit_lens.shape[0]):
        raise ValueError(
            "accept_index / commit_lens batch size mismatch: "
            f"{tuple(accept_index.shape)} vs {tuple(commit_lens.shape)}"
        )

    commit_lens = commit_lens.to(device=accept_index.device, dtype=torch.int32)
    cols = torch.arange(
        int(accept_index.shape[1]), device=accept_index.device, dtype=torch.int32
    )[None, :]
    valid_mask = cols < commit_lens.unsqueeze(1)
    resolved = accept_index[valid_mask].to(torch.int64)
    if bool((resolved < 0).any().detach().cpu().item()):
        bad = resolved.detach().cpu().tolist()
        raise RuntimeError(
            "DFLASH indexed accept resolution produced negative indices after commit truncation: "
            f"indices={bad}"
        )
    return resolved


def resolve_dflash_overlap_token_ids(
    *,
    flat_token_ids: torch.Tensor,
    accept_lens: torch.Tensor,
) -> list[list[int]]:
    """Slice compact DFLASH overlap outputs into per-request accepted-token lists."""
    if flat_token_ids.ndim != 1:
        raise ValueError(
            f"flat_token_ids must be 1D, got shape={tuple(flat_token_ids.shape)}"
        )
    if accept_lens.ndim != 1:
        raise ValueError(
            f"accept_lens must be 1D, got shape={tuple(accept_lens.shape)}"
        )

    flat_cpu = flat_token_ids.to(torch.int64).tolist()
    lens_cpu = accept_lens.to(torch.int64).tolist()
    out: list[list[int]] = []
    offset = 0
    for commit_len in lens_cpu:
        next_offset = offset + int(commit_len)
        out.append(flat_cpu[offset:next_offset])
        offset = next_offset

    if offset != len(flat_cpu):
        raise ValueError(
            "DFLASH overlap compact token layout mismatch: "
            f"consumed={offset} total={len(flat_cpu)} lens={lens_cpu}"
        )
    return out


def compute_dflash_sampling_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    linear_mode: str = "draft=greedy,target=sampled",
    draft_selected_probs: Optional[torch.Tensor] = None,
    draft_proposal_indices: Optional[torch.Tensor] = None,
    draft_proposal_probs: Optional[torch.Tensor] = None,
    max_steps_per_req: Optional[torch.Tensor] = None,
    threshold_single: Optional[float] = None,
    threshold_acc: Optional[float] = None,
    uniform_samples: Optional[torch.Tensor] = None,
    uniform_samples_for_final_sampling: Optional[torch.Tensor] = None,
    use_sparse_topk: bool = True,
    return_proposed_tokens: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact DFlash accept lengths and bonus tokens for sampled verification.

    Production linear DFlash uses exact proposal correction:
      - `draft=greedy,target=sampled`: accept with probability `p(x)` and on rejection
        sample from `p` with the greedy draft token removed.
      - `draft=sampled,target=sampled`: accept with probability `min(1, p(x) / q(x))`
        and on rejection sample from normalized `max(0, p - q)`.
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

    if (os.environ.get("SGLANG_DFLASH_DISABLE_SAMPLED_HELPER_SPARSE_TOPK") or "").strip().lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    ):
        use_sparse_topk = False

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    need_min_p_sampling = bool(getattr(sampling_info, "need_min_p_sampling", False))

    if linear_mode not in (
        "draft=greedy,target=sampled",
        "draft=sampled,target=sampled",
    ):
        raise ValueError(f"Unsupported DFLASH sampled linear mode: {linear_mode!r}")

    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    ).to(device=device, dtype=next_token_logits.dtype)
    if bool(torch.any(expanded_temperature <= 0).item()):
        raise RuntimeError(
            "DFLASH sampled verification received non-positive temperature. "
            "This batch should have stayed on the greedy target-verify path."
        )
    scaled_logits = next_token_logits / expanded_temperature.view(-1, 1)
    repeated_top_ks = torch.repeat_interleave(
        sampling_info.top_ks, draft_token_num, dim=0
    ).to(device=device, dtype=torch.int32)
    repeated_top_ps = torch.repeat_interleave(
        sampling_info.top_ps, draft_token_num, dim=0
    ).to(device=device, dtype=torch.float32)
    repeated_min_ps = torch.repeat_interleave(
        sampling_info.min_ps, draft_token_num, dim=0
    ).to(device=device, dtype=torch.float32)

    target_probs_sort, target_probs_idx = build_dflash_filtered_sampling_distribution_from_logits(
        scaled_logits,
        repeated_top_ks,
        repeated_top_ps,
        repeated_min_ps,
        need_min_p_sampling,
    )
    target_probs_sort = target_probs_sort.view(bs, draft_token_num, -1).contiguous()
    target_probs_idx = target_probs_idx.view(bs, draft_token_num, -1).contiguous()

    candidates_i64 = (
        candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
    )
    step_count = int(max(0, draft_token_num - 1))
    if step_count <= 0:
        accept_len = torch.zeros((bs,), dtype=torch.int32, device=device)
        bonus_local = _sample_dflash_sparse_probs_with_uniform(
            target_probs_sort[:, 0, :],
            uniform_samples_for_final_sampling,
        )
        bonus = target_probs_idx[:, 0, :].gather(1, bonus_local.unsqueeze(1)).view(-1)
        bonus = bonus.to(torch.int64)
        if not return_proposed_tokens:
            return accept_len, bonus
        proposed_tokens = bonus.unsqueeze(1)
        return accept_len, bonus, proposed_tokens

    proposal_tokens = candidates_i64[:, 1:]
    proposal_match = target_probs_idx[:, :-1, :] == proposal_tokens.unsqueeze(-1)
    proposal_target_probs = torch.where(
        proposal_match,
        target_probs_sort[:, :-1, :],
        torch.zeros_like(target_probs_sort[:, :-1, :]),
    ).sum(dim=-1)

    if linear_mode == "draft=sampled,target=sampled":
        if draft_selected_probs is None:
            raise RuntimeError(
                "DFLASH exact sampled-draft verify requires per-token draft proposal probabilities."
            )
        if draft_selected_probs.shape != proposal_target_probs.shape:
            raise ValueError(
                "draft_selected_probs shape mismatch: "
                f"expected {tuple(proposal_target_probs.shape)}, got {tuple(draft_selected_probs.shape)}"
            )
        if draft_proposal_indices is None or draft_proposal_probs is None:
            raise RuntimeError(
                "DFLASH exact sampled-draft verify requires sparse draft proposal support."
            )
        if draft_proposal_indices.ndim != 3 or draft_proposal_probs.ndim != 3:
            raise ValueError(
                "draft proposal support tensors must be 3D, got "
                f"indices={None if draft_proposal_indices is None else tuple(draft_proposal_indices.shape)} "
                f"probs={None if draft_proposal_probs is None else tuple(draft_proposal_probs.shape)}"
            )
        if tuple(draft_proposal_indices.shape[:2]) != tuple(proposal_target_probs.shape):
            raise ValueError(
                "draft proposal support batch/step shape mismatch: "
                f"expected {tuple(proposal_target_probs.shape)}, "
                f"got {tuple(draft_proposal_indices.shape[:2])}"
            )
        accept_prob = proposal_target_probs / draft_selected_probs.to(
            device=device, dtype=proposal_target_probs.dtype
        ).clamp_min(1e-20)
        accept_prob.clamp_(min=0.0, max=1.0)
    else:
        accept_prob = proposal_target_probs.clamp(min=0.0, max=1.0)

    # Optional debug to explain low acceptance: if the draft proposes tokens outside the
    # target's filtered sampling support, proposal_target_probs becomes 0 and acceptance
    # collapses. This log is intentionally rate-limited and off by default.
    global _DFLASH_SAMPLED_DEBUG_CT
    if _env_truthy("SGLANG_DFLASH_DEBUG_SAMPLED_VERIFY") and _DFLASH_SAMPLED_DEBUG_CT < 10:
        _DFLASH_SAMPLED_DEBUG_CT += 1
        with torch.no_grad():
            try:
                zero_frac = float((proposal_target_probs <= 0).to(torch.float32).mean().item())
                p_mean = float(proposal_target_probs.to(torch.float32).mean().item())
                a_mean = float(accept_prob.to(torch.float32).mean().item())
                logger.info(
                    "DFLASH sampled-verify debug: mode=%s bs=%d steps=%d proposal_p_zero_frac=%.3f proposal_p_mean=%.4g accept_prob_mean=%.4g",
                    linear_mode,
                    int(bs),
                    int(step_count),
                    zero_frac,
                    p_mean,
                    a_mean,
                )
            except Exception as e:
                logger.info("DFLASH sampled-verify debug failed: %s", e)

    accept_mask = uniform_samples[:, :step_count] <= accept_prob
    if max_steps_per_req is not None:
        if max_steps_per_req.ndim != 1 or int(max_steps_per_req.shape[0]) != int(bs):
            raise ValueError(
                "max_steps_per_req must be a 1D tensor with shape [bs]. "
                f"Got shape={tuple(max_steps_per_req.shape)} for bs={bs}."
            )
        caps = max_steps_per_req.to(device=device, dtype=torch.int64).clamp(
            min=0, max=int(step_count)
        )
        step_ids = torch.arange(step_count, device=device, dtype=torch.int64)
        accept_mask = accept_mask & (step_ids.unsqueeze(0) < caps.unsqueeze(1))
    accept_len = accept_mask.to(torch.int32).cumprod(dim=1).sum(dim=1)
    if max_steps_per_req is not None:
        accept_len = torch.minimum(
            accept_len,
            max_steps_per_req.to(device=device, dtype=torch.int32).clamp(
                min=0, max=int(step_count)
            ),
        )
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    reject_step = accept_len.to(torch.long).clamp(min=0, max=int(step_count))
    bonus_probs_sort = target_probs_sort[row_ids, reject_step].clone()
    bonus_token_ids = target_probs_idx[row_ids, reject_step].clone()

    rejected_mask = reject_step < int(step_count)
    if bool(rejected_mask.any().item()):
        rejected_rows = row_ids[rejected_mask]
        rejected_steps = reject_step[rejected_mask]
        if linear_mode == "draft=sampled,target=sampled":
            assert draft_proposal_indices is not None
            assert draft_proposal_probs is not None
            rejected_support_idx = draft_proposal_indices[
                rejected_rows, rejected_steps
            ].to(device=device, dtype=torch.long)
            rejected_support_probs = draft_proposal_probs[
                rejected_rows, rejected_steps
            ].to(device=device, dtype=bonus_probs_sort.dtype)
            overlap = (
                bonus_token_ids[rejected_mask].unsqueeze(2)
                == rejected_support_idx.unsqueeze(1)
            )
            subtract = (
                overlap.to(dtype=bonus_probs_sort.dtype)
                * rejected_support_probs.unsqueeze(1)
            ).sum(dim=-1)
            bonus_probs_sort[rejected_mask].sub_(subtract).clamp_min_(0.0)
        else:
            rejected_tokens = proposal_tokens[rejected_rows, rejected_steps].unsqueeze(1)
            zero_mask = bonus_token_ids[rejected_mask] == rejected_tokens
            bonus_probs_sort[rejected_mask] = torch.where(
                zero_mask,
                torch.zeros_like(bonus_probs_sort[rejected_mask]),
                bonus_probs_sort[rejected_mask],
            )

    bonus_local = _sample_dflash_sparse_probs_with_uniform(
        bonus_probs_sort,
        uniform_samples_for_final_sampling,
    )
    bonus = bonus_token_ids.gather(1, bonus_local.unsqueeze(1)).view(-1).to(torch.int64)
    if not return_proposed_tokens:
        return accept_len, bonus

    proposed_tokens = torch.zeros(
        (bs, draft_token_num), dtype=torch.int64, device=device
    )
    if step_count > 0:
        accepted_prefix = proposal_tokens
        prefix_cols = torch.arange(
            step_count, device=device, dtype=torch.int32
        )[None, :]
        prefix_mask = prefix_cols < accept_len.unsqueeze(1)
        proposed_tokens[:, :-1] = torch.where(
            prefix_mask,
            accepted_prefix,
            torch.zeros_like(accepted_prefix),
        )
    bonus_pos = accept_len.to(torch.long).clamp(min=0, max=int(draft_token_num - 1))
    proposed_tokens.scatter_(1, bonus_pos.unsqueeze(1), bonus.unsqueeze(1))
    return accept_len, bonus, proposed_tokens
