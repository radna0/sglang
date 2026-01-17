# Adapted from vLLM/SGLang, but keep module import lightweight:
# - Avoid importing `loader` (which pulls in HF Transformers) during plain module
#   imports (e.g. unit tests / draft-model utilities).

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in {"BaseModelLoader", "get_model_loader"}:
        from sglang.srt.model_loader.loader import BaseModelLoader, get_model_loader

        return {"BaseModelLoader": BaseModelLoader, "get_model_loader": get_model_loader}[name]
    if name in {"get_architecture_class_name", "get_model_architecture"}:
        from sglang.srt.model_loader.utils import (
            get_architecture_class_name,
            get_model_architecture,
        )

        return {
            "get_architecture_class_name": get_architecture_class_name,
            "get_model_architecture": get_model_architecture,
        }[name]
    raise AttributeError(name)


def get_model(
    *,
    model_config: "ModelConfig",
    load_config: "LoadConfig",
    device_config: "DeviceConfig",
) -> nn.Module:
    # Import lazily to avoid HF Transformers dependency unless we actually load a target model.
    from sglang.srt.model_loader.loader import get_model_loader

    loader = get_model_loader(load_config, model_config)
    return loader.load_model(model_config=model_config, device_config=device_config)


__all__ = ["get_model", "get_model_loader", "BaseModelLoader", "get_architecture_class_name", "get_model_architecture"]
