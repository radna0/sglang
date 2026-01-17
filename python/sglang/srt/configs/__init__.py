from __future__ import annotations

from typing import Any


_CONFIG_EXPORTS = {
    "ChatGLMConfig": ("sglang.srt.configs.chatglm", "ChatGLMConfig"),
    "DbrxConfig": ("sglang.srt.configs.dbrx", "DbrxConfig"),
    "DeepseekVL2Config": ("sglang.srt.configs.deepseekvl2", "DeepseekVL2Config"),
    "DotsOCRConfig": ("sglang.srt.configs.dots_ocr", "DotsOCRConfig"),
    "DotsVLMConfig": ("sglang.srt.configs.dots_vlm", "DotsVLMConfig"),
    "ExaoneConfig": ("sglang.srt.configs.exaone", "ExaoneConfig"),
    "FalconH1Config": ("sglang.srt.configs.falcon_h1", "FalconH1Config"),
    "MultiModalityConfig": ("sglang.srt.configs.janus_pro", "MultiModalityConfig"),
    "JetNemotronConfig": ("sglang.srt.configs.jet_nemotron", "JetNemotronConfig"),
    "JetVLMConfig": ("sglang.srt.configs.jet_vlm", "JetVLMConfig"),
    "KimiLinearConfig": ("sglang.srt.configs.kimi_linear", "KimiLinearConfig"),
    "KimiVLConfig": ("sglang.srt.configs.kimi_vl", "KimiVLConfig"),
    "MoonViTConfig": ("sglang.srt.configs.kimi_vl_moonvit", "MoonViTConfig"),
    "LongcatFlashConfig": ("sglang.srt.configs.longcat_flash", "LongcatFlashConfig"),
    "NemotronH_Nano_VL_V2_Config": (
        "sglang.srt.configs.nano_nemotron_vl",
        "NemotronH_Nano_VL_V2_Config",
    ),
    "NemotronHConfig": ("sglang.srt.configs.nemotron_h", "NemotronHConfig"),
    "Olmo3Config": ("sglang.srt.configs.olmo3", "Olmo3Config"),
    "Qwen3NextConfig": ("sglang.srt.configs.qwen3_next", "Qwen3NextConfig"),
    "Step3TextConfig": ("sglang.srt.configs.step3_vl", "Step3TextConfig"),
    "Step3VisionEncoderConfig": ("sglang.srt.configs.step3_vl", "Step3VisionEncoderConfig"),
    "Step3VLConfig": ("sglang.srt.configs.step3_vl", "Step3VLConfig"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    import importlib

    spec = _CONFIG_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(name)
    mod = importlib.import_module(spec[0])
    return getattr(mod, spec[1])


__all__ = list(_CONFIG_EXPORTS.keys())

