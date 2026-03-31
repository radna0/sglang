from __future__ import annotations

import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

derive_mod = SourceFileLoader(
    "derive_gpt_oss_care_rank_schedule",
    str(REPO_ROOT / "scripts" / "derive_gpt_oss_care_rank_schedule.py"),
).load_module()
smoke_mod = SourceFileLoader(
    "run_gpt_oss_care_sglang_smoke",
    str(REPO_ROOT / "scripts" / "run_gpt_oss_care_sglang_smoke.py"),
).load_module()

from sglang.srt.configs.model_config import AttentionArch, ModelConfig  # noqa: E402
from sglang.srt.layers.radix_attention import RadixAttention  # noqa: E402
from sglang.srt.layers.utils.multi_platform import MultiPlatformOp  # noqa: E402
from sglang.srt.model_executor.mla_utils import (  # noqa: E402
    get_flashmla_mla_kv_cache_dim,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (  # noqa: E402
    ModelRunnerKVCacheMixin,
)


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs):
        self.calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape) if k is not None else None,
                "v_shape": tuple(v.shape) if v is not None else None,
                "save_kv_cache": save_kv_cache,
                "kwargs": kwargs,
                "layer_id": layer.layer_id,
                "sliding_window_size": layer.sliding_window_size,
            }
        )
        return torch.zeros_like(q)


class _FakeForwardMode:
    @staticmethod
    def is_extend() -> bool:
        return False


class _FakeForwardBatch:
    def __init__(self, attn_backend):
        self.forward_mode = _FakeForwardMode()
        self.attn_backend = attn_backend


class _TopKCompileProbe(MultiPlatformOp):
    def forward_native(self, *args, **kwargs):
        return "native"

    def forward_cuda(self, *args, **kwargs):
        return "cuda"


class TestGptOssCareCpuChecks(unittest.TestCase):
    def test_round_schedule_to_multiple_emits_mod8_schedule(self):
        raw = [413, 308, 417, 431]
        spectra = [list(range(4000, 0, -1)) for _ in raw]
        caps = [1024] * len(raw)

        rounded, metadata = derive_mod._round_schedule_to_multiple(
            raw,
            spectra,
            caps,
            total_budget=sum(raw),
            min_rank=128,
            round_multiple=8,
        )

        self.assertTrue(all(rank % 8 == 0 for rank in rounded))
        self.assertEqual(sum(rounded), metadata["rounded_total_rank"])
        self.assertEqual(metadata["round_multiple"], 8)
        self.assertGreaterEqual(min(rounded), 128)
        self.assertLessEqual(max(rounded), 1024)

    def test_model_config_dynamic_rank_helpers_work(self):
        mc = object.__new__(ModelConfig)
        mc.attention_arch = AttentionArch.MLA
        mc.kv_lora_rank = 512
        mc.kv_lora_rank_per_layer = None
        mc.num_hidden_layers = 4
        mc.hf_text_config = SimpleNamespace(
            kv_lora_rank_per_layer=[320, 512, 320, 640]
        )
        mc.hf_config = SimpleNamespace()

        mc._maybe_init_dynamic_mla_kv_ranks()

        self.assertEqual(mc.kv_lora_rank_per_layer, [320, 512, 320, 640])
        self.assertTrue(mc.has_dynamic_mla_kv_lora_rank())
        self.assertEqual(mc.get_mla_kv_lora_rank(3), 640)
        self.assertEqual(mc.get_unique_mla_kv_lora_ranks(), [320, 512, 640])

    def test_auto_backend_selection_locks_gptoss_mla_to_flashmla(self):
        config = {
            "architectures": ["GptOssMlaForCausalLM"],
            "kv_lora_rank_per_layer": [320, 512, 384, 640],
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        }

        backend = smoke_mod._select_attention_backend("auto", config)
        self.assertEqual(backend, "flashmla")

    def test_auto_backend_selection_keeps_native_anchor_on_flashmla_contract(self):
        config = {
            "architectures": ["GptOssMlaForCausalLM"],
            "mla_rope_num_kv_heads": 8,
            "layer_types": ["sliding_attention", "full_attention"],
        }

        backend = smoke_mod._select_attention_backend("auto", config)
        self.assertEqual(backend, "flashmla")

    def test_flashmla_metadata_marks_actual_mla_backend_path(self):
        config = {
            "architectures": ["GptOssMlaForCausalLM"],
            "kv_lora_rank_per_layer": [320, 512, 384, 640],
            "layer_types": ["sliding_attention", "full_attention"],
        }

        metadata = smoke_mod._backend_metadata("flashmla", config)

        self.assertFalse(metadata["experimental"])
        self.assertEqual(metadata["backend_family"], "flashmla")
        self.assertIn("actual MLA kernel path", metadata["reason"])

    def test_native_mha_anchor_metadata_is_flashmla_explicit(self):
        config = {
            "architectures": ["GptOssMlaForCausalLM"],
            "mla_rope_num_kv_heads": 8,
            "layer_types": ["sliding_attention", "full_attention"],
        }

        metadata = smoke_mod._backend_metadata("flashmla", config)

        self.assertTrue(metadata["uses_gpt_oss_native_mha_anchor"])
        self.assertIn("sink and sliding-window parity", metadata["reason"])

    def test_dynamic_flashmla_smoke_auto_disables_cuda_graph(self):
        config = {
            "architectures": ["GptOssMlaForCausalLM"],
            "kv_lora_rank_per_layer": [320, 512, 384, 640],
            "layer_types": ["sliding_attention", "full_attention"],
        }

        self.assertTrue(smoke_mod._should_disable_cuda_graph("flashmla", config))
        self.assertTrue(
            smoke_mod._should_disable_piecewise_cuda_graph("flashmla", config)
        )

    def test_non_flashmla_backends_are_rejected_for_gptoss_mla_smoke(self):
        config = {
            "architectures": ["GptOssMlaForCausalLM"],
            "kv_lora_rank_per_layer": [320, 512, 384, 640],
            "layer_types": ["sliding_attention", "full_attention"],
        }

        with self.assertRaises(ValueError):
            smoke_mod._select_attention_backend("flex_attention2", config)
        with self.assertRaises(ValueError):
            smoke_mod._select_attention_backend("triton", config)
        with self.assertRaises(ValueError):
            smoke_mod._select_attention_backend("flashinfer", config)

    def test_radix_attention_forwards_sink_and_rope_kwargs(self):
        backend = _FakeBackend()
        batch = _FakeForwardBatch(backend)
        layer = RadixAttention(
            num_heads=2,
            head_dim=4,
            scaling=0.5,
            num_kv_heads=1,
            layer_id=7,
            v_head_dim=2,
            sliding_window_size=31,
        )
        q = torch.randn(3, 8)
        k = torch.randn(3, 2)
        v = torch.randn(3, 1, 2)
        q_rope = torch.randn(3, 2, 2)
        k_rope = torch.randn(3, 1, 2)
        sinks = torch.randn(2)

        out = layer(
            q,
            k,
            v,
            batch,
            q_rope=q_rope,
            k_rope=k_rope,
            sinks=sinks,
        )

        self.assertEqual(tuple(out.shape), tuple(q.shape))
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertIs(call["kwargs"]["q_rope"], q_rope)
        self.assertIs(call["kwargs"]["k_rope"], k_rope)
        self.assertIs(call["kwargs"]["sinks"], sinks)
        self.assertEqual(call["sliding_window_size"], 31)

    def test_gpt_oss_source_keeps_dynamic_rank_sliding_and_sink_wiring(self):
        source = (REPO_ROOT / "python" / "sglang" / "srt" / "models" / "gpt_oss.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('kv_rank_schedule = getattr(config, "kv_lora_rank_per_layer", None)', source)
        self.assertIn("sinks=self.sinks", source)
        self.assertIn(
            'sliding_window_size=(sliding_window_size if use_sliding_window else -1)',
            source,
        )

    def test_flashmla_graph_capture_uses_cpu_req_pool_indices(self):
        flashmla_source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "attention"
            / "flashmla_backend.py"
        ).read_text(encoding="utf-8")
        forward_batch_source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "model_executor"
            / "forward_batch_info.py"
        ).read_text(encoding="utf-8")
        model_runner_source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "model_executor"
            / "model_runner.py"
        ).read_text(encoding="utf-8")
        cuda_graph_runner_source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "model_executor"
            / "cuda_graph_runner.py"
        ).read_text(encoding="utf-8")

        self.assertIn('req_pool_indices_cpu: Optional[torch.Tensor] = None', forward_batch_source)
        self.assertIn('req_pool_indices_cpu=batch.req_pool_indices.cpu()', forward_batch_source)
        self.assertIn('req_pool_indices_cpu=buffers.req_pool_indices.cpu()', model_runner_source)
        self.assertIn('req_pool_indices_cpu = req_pool_indices.cpu()', cuda_graph_runner_source)
        self.assertIn('req_pool_indices_cpu=req_pool_indices_cpu', cuda_graph_runner_source)
        self.assertIn('req_pool_indices_cpu = getattr(forward_batch, "req_pool_indices_cpu", None)', flashmla_source)
        self.assertIn('forward_batch.req_pool_indices.detach().cpu()', flashmla_source)

    def test_server_args_prefers_triton_kernel_for_gptoss_mxfp4_on_cuda(self):
        source = (REPO_ROOT / "python" / "sglang" / "srt" / "server_args.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "Detected GPT-OSS MXFP4 quantization on CUDA, enabling triton_kernels MOE kernel",
            source,
        )
        self.assertIn('self.moe_runner_backend = "triton_kernel"', source)

    def test_mxfp4_method_autoselects_triton_kernel_on_hopper(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "quantization"
            / "mxfp4.py"
        ).read_text(encoding="utf-8")
        self.assertIn("moe_backend = get_moe_runner_backend()", source)
        self.assertIn("moe_backend.is_auto()", source)
        self.assertIn("is_sm90_supported()", source)
        self.assertIn("Auto-selecting triton_kernels MXFP4 MoE backend on Hopper", source)

    def test_model_runner_records_max_running_requests_before_routed_experts_capturer(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "model_executor"
            / "model_runner_kv_cache_mixin.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.max_running_requests = max_num_reqs", source)

    def test_hybrid_swa_allocator_uses_mla_latent_width(self):
        class DummyRunner(ModelRunnerKVCacheMixin):
            pass

        runner = DummyRunner()
        runner.use_mla_backend = True
        runner.kv_cache_dtype = torch.bfloat16
        runner.sliding_window_size = 4096
        runner.max_total_num_tokens = 8192
        runner.server_args = SimpleNamespace(page_size=64, swa_full_tokens_ratio=0.8)
        runner.model_config = SimpleNamespace(
            full_attention_layer_ids=[0, 1],
            swa_attention_layer_ids=[2, 3],
        )
        runner.calculate_mla_kv_cache_dim = lambda: 576

        runner.set_num_tokens_hybrid_swa()

        self.assertGreater(runner.full_max_total_num_tokens, 0)
        self.assertGreater(runner.swa_max_total_num_tokens, 0)
        self.assertEqual(runner.max_total_num_tokens, runner.full_max_total_num_tokens)

    def test_flashmla_mla_cache_width_helper_preserves_shared_and_native_geometries(self):
        self.assertEqual(get_flashmla_mla_kv_cache_dim(512, 32, 1, "flashmla"), 576)
        self.assertEqual(get_flashmla_mla_kv_cache_dim(256, 32, 1, "triton"), 576)
        self.assertEqual(get_flashmla_mla_kv_cache_dim(512, 32, 8, "flashmla"), 768)

    def test_scheduler_prefers_triton_kernel_for_gptoss_mxfp4_on_hopper(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "managers"
            / "scheduler.py"
        ).read_text(encoding="utf-8")
        self.assertIn("selecting triton_kernels MoE backend before initialize_moe_config", source)
        self.assertIn('self.server_args.moe_runner_backend = "triton_kernel"', source)

    def test_topk_compile_mode_preserves_triton_kernel_backend(self):
        probe = _TopKCompileProbe()
        baseline = probe.forward()

        with patch("sglang.srt.layers.moe.get_moe_runner_backend") as get_backend:
            get_backend.return_value = SimpleNamespace(is_triton_kernels=lambda: True)
            probe.enter_torch_compile(num_tokens=1)

        self.assertEqual(probe.forward(), "cuda")
        probe.leave_torch_compile()
        self.assertEqual(probe.forward(), baseline)

    def test_flashmla_backend_source_handles_sinks_sliding_and_pure_prefill(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "attention"
            / "flashmla_backend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from sglang.srt.layers.attention.flex_utils import apply_attention_sinks", source)
        self.assertIn("def _build_decode_sliding_indices(", source)
        self.assertIn("def _build_decode_dense_kv_and_indices(", source)
        self.assertIn("def _build_extend_dense_kv_and_indices(", source)
        self.assertIn("def _forward_sparse_prefill(", source)
        self.assertIn("forward_batch.token_to_kv_pool.set_mla_kv_buffer(", source)
        self.assertIn("kv_dense, indices = self._build_decode_dense_kv_and_indices(", source)
        self.assertIn("q_all = self._reshape_q_all(q, layer, q_rope)", source)
        self.assertIn("q_width = q_all.shape[-1]", source)
        self.assertIn("reshape_q = q_all.view(", source)
        self.assertIn("o = apply_attention_sinks(", source)
        self.assertIn("lse.reshape(-1, self.num_q_heads)", source)

    def test_model_config_routes_gptoss_multi_rope_anchor_to_mha_arch(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "configs"
            / "model_config.py"
        ).read_text(encoding="utf-8")
        self.assertIn("gpt_oss_native_mha_anchor", source)
        self.assertIn(
            'getattr(self.hf_text_config, "mla_rope_num_kv_heads", 1)',
            source,
        )
        self.assertIn(
            "AttentionArch.MHA if gpt_oss_native_mha_anchor else AttentionArch.MLA",
            source,
        )

    def test_attention_registry_keeps_flex_flash4_in_flex2_backend_family(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "attention"
            / "attention_registry.py"
        ).read_text(encoding="utf-8")
        self.assertIn('@register_attention_backend("flex_attention2")', source)
        self.assertIn('@register_attention_backend("flex_flash4")', source)
        self.assertIn(
            "from sglang.srt.layers.attention.torch_flex2_backend import TorchFlexAttnBackendV2",
            source,
        )

    def test_req_time_stats_keeps_prefill_launch_delay_compat_alias(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "observability"
            / "req_time_stats.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def get_prefill_launch_delay(self) -> Optional[float]:", source)
        self.assertIn("return self.get_prefill_waiting_latency()", source)

    def test_model_runner_forces_bf16_kv_cache_instead_of_fp16_auto(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "model_executor"
            / "model_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.kv_cache_dtype = (", source)
        self.assertIn("torch.bfloat16 if self.dtype == torch.float16 else self.dtype", source)
        self.assertIn('attn_backend_name == "FlashInferMLAAttnBackend"', source)
        self.assertIn("Disable cuda graph for FlashInfer MLA because dynamic per-layer", source)
        self.assertIn("Disable piecewise CUDA graph for FlashInfer MLA because dynamic per-layer", source)
        self.assertIn("Disable piecewise CUDA graph because --disable-cuda-graph is set", source)
        self.assertIn("Disable piecewise CUDA graph because --cuda-graph-mode none is set", source)

    def test_scheduler_uses_routed_dp_rank_for_tokenized_requests(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "managers"
            / "scheduler.py"
        ).read_text(encoding="utf-8")
        self.assertIn('routed_dp_rank=getattr(', source)
        self.assertIn('"routed_dp_rank"', source)
        self.assertIn('"data_parallel_rank"', source)

    def test_hybrid_swa_mla_path_uses_mla_kv_pool_class_and_split_rank_schedules(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "model_executor"
            / "model_runner_kv_cache_mixin.py"
        ).read_text(encoding="utf-8")
        self.assertIn("token_to_kv_pool_class = (", source)
        self.assertIn("MLATokenToKVPoolFP4", source)
        self.assertIn("MLATokenToKVPool", source)
        self.assertIn("full_kv_lora_rank", source)
        self.assertIn("swa_kv_lora_rank", source)
        self.assertIn("full_token_to_kv_pool_kwargs=full_pool_kwargs", source)
        self.assertIn("swa_token_to_kv_pool_kwargs=swa_pool_kwargs", source)
        self.assertIn("and not self.is_hybrid_swa", source)
        self.assertIn("self.model_config.is_hybrid_swa and kv_rank_schedule is not None", source)

    def test_swa_pool_filters_kwargs_for_mla_inner_pools(self):
        source = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "mem_cache"
            / "swa_memory_pool.py"
        ).read_text(encoding="utf-8")
        self.assertIn("inspect.signature(pool_cls.__init__).parameters", source)
        self.assertIn("_filter_pool_init_kwargs(", source)
        self.assertIn("_normalize_kv_size_bytes(", source)
        self.assertIn("if v_size == 0 and v_size_swa == 0", source)
        self.assertIn("valid_params = set(inspect.signature(pool.set_kv_buffer).parameters)", source)
        self.assertIn('if "layer_id_override" in valid_params:', source)
        self.assertIn("layer.layer_id = layer_id_pool", source)


if __name__ == "__main__":
    unittest.main()
