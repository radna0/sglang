import types
import unittest
from unittest import mock
import sys

import torch

if "triton" not in sys.modules:
    triton_mod = types.ModuleType("triton")
    triton_lang_mod = types.ModuleType("triton.language")
    triton_mod.language = triton_lang_mod
    triton_mod.jit = lambda fn=None, **_kw: fn
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = triton_lang_mod

import sglang.srt.layers.attention.torch_flex2_backend as flex2
from sglang.test.test_utils import CustomTestCase


class _FakeFlashAttentionBackend:
    def __init__(self, _model_runner, fa_impl_ver=3):
        self.fa_impl_ver = fa_impl_ver

    def init_forward_metadata(self, _forward_batch):
        return None

    def init_cuda_graph_state(self, _max_bs, _max_num_tokens):
        return None

    def init_forward_metadata_capture_cuda_graph(
        self,
        _bs,
        _num_tokens,
        _req_pool_indices,
        _seq_lens,
        _encoder_lens,
        _forward_mode,
        _spec_info,
    ):
        return None

    def init_forward_metadata_replay_cuda_graph(
        self,
        _bs,
        _req_pool_indices,
        _seq_lens,
        _seq_lens_sum,
        _encoder_lens,
        _forward_mode,
        _spec_info,
        _seq_lens_cpu,
    ):
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(self, _spec_info, _cuda_graph_bs):
        return None

    def forward_decode(
        self,
        _q,
        _k,
        _v,
        _layer,
        _forward_batch,
        save_kv_cache=True,
        q_rope=None,
        k_rope=None,
        sinks=None,
    ):
        _ = (save_kv_cache, q_rope, k_rope, sinks)
        return "fake_decode"

    def forward_extend(
        self,
        _q,
        _k,
        _v,
        _layer,
        _forward_batch,
        save_kv_cache=True,
        q_rope=None,
        k_rope=None,
        sinks=None,
    ):
        _ = (save_kv_cache, q_rope, k_rope, sinks)
        return "fake_extend"


class TestFlex2ForceFlashDelegate(CustomTestCase):
    def test_force_flash_creates_delegate_and_routes_decode(self):
        # TorchFlexAttnBackend uses torch.compile(flex_attention) at init; for this unit
        # test we only validate delegation wiring, so stub compile to avoid heavy work.
        with (
            mock.patch.object(flex2, "_load_flashattention_backend", lambda: _FakeFlashAttentionBackend),
            mock.patch.object(torch, "compile", lambda f, **_kw: f),
        ):
            model_runner = types.SimpleNamespace(
                device=torch.device("cpu"),
                server_args=types.SimpleNamespace(page_size=128),
            )
            backend = flex2.TorchFlexAttnBackendV2(model_runner, kernel_options={"force_flash": True})
            self.assertIsInstance(backend._fa_delegate, _FakeFlashAttentionBackend)
            out = backend.forward_decode(
                torch.empty((1, 8), dtype=torch.float16),
                None,
                None,
                object(),
                object(),
                save_kv_cache=False,
            )
            self.assertEqual(out, "fake_decode")

    def test_force_flash_delegate_impl_4_is_respected(self):
        with (
            mock.patch.object(flex2, "_load_flashattention_backend", lambda: _FakeFlashAttentionBackend),
            mock.patch.object(torch, "compile", lambda f, **_kw: f),
        ):
            model_runner = types.SimpleNamespace(
                device=torch.device("cpu"),
                server_args=types.SimpleNamespace(page_size=128),
            )
            backend = flex2.TorchFlexAttnBackendV2(
                model_runner,
                kernel_options={"force_flash": True, "delegate_fa": True, "delegate_fa_impl": 4},
            )
            self.assertEqual(backend._delegate_fa_impl, 4)
            self.assertIsInstance(backend._fa_delegate, _FakeFlashAttentionBackend)
            self.assertEqual(backend._fa_delegate.fa_impl_ver, 4)

    def test_force_flash_fp8_kv_fastpath_creates_delegate(self):
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            self.skipTest("float8_e4m3fn unavailable in this torch build")

        with (
            mock.patch.object(flex2, "_load_flashattention_backend", lambda: _FakeFlashAttentionBackend),
            mock.patch.object(torch, "compile", lambda f, **_kw: f),
        ):
            model_runner = types.SimpleNamespace(
                device=torch.device("cpu"),
                server_args=types.SimpleNamespace(page_size=128),
            )
            backend = flex2.TorchFlexAttnBackendV2(
                model_runner,
                kernel_options={"force_flash": True, "delegate_fa_impl": 4},
            )
            fake_fp8_cache = types.SimpleNamespace(dtype=fp8_dtype)
            routed = backend._maybe_fastpath_fp8_kv_delegate(
                k_cache=fake_fp8_cache,
                v_cache=fake_fp8_cache,
                reason="unit_fp8",
            )
            self.assertTrue(routed)
            self.assertIsInstance(backend._fa_delegate, _FakeFlashAttentionBackend)
            self.assertEqual(backend._fa_delegate.fa_impl_ver, 4)

    def test_no_force_flash_does_not_create_delegate(self):
        with (
            mock.patch.object(flex2, "_load_flashattention_backend", lambda: _FakeFlashAttentionBackend),
            mock.patch.object(torch, "compile", lambda f, **_kw: f),
        ):
            model_runner = types.SimpleNamespace(
                device=torch.device("cpu"),
                server_args=types.SimpleNamespace(page_size=128),
            )
            backend = flex2.TorchFlexAttnBackendV2(model_runner, kernel_options={})
            self.assertIsNone(backend._fa_delegate)


if __name__ == "__main__":
    unittest.main()
