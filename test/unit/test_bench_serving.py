import asyncio
from types import SimpleNamespace

import sglang.bench_serving as bench_serving
from sglang.bench_serving import (
    RequestFuncInput,
    _merge_request_extra_body,
    async_request_sglang_generate,
)


def test_merge_request_extra_body_preserves_nested_sampling_params():
    merged = _merge_request_extra_body(
        {
            "sampling_params": {
                "temperature": 0.0,
                "top_k": 1,
            },
            "return_logprob": False,
        },
        {
            "sampling_params": {
                "max_new_tokens": 7470,
                "ignore_eos": True,
            },
            "stream": False,
        },
    )

    assert merged["return_logprob"] is False
    assert merged["stream"] is False
    assert merged["sampling_params"] == {
        "temperature": 0.0,
        "top_k": 1,
        "max_new_tokens": 7470,
        "ignore_eos": True,
    }


def test_merge_request_extra_body_keeps_top_level_precedence():
    merged = _merge_request_extra_body(
        {"temperature": 0.0, "sampling_params": {"top_p": 1.0}},
        {"temperature": 1.0, "sampling_params": {"min_p": 0.02}},
    )

    assert merged["temperature"] == 1.0
    assert merged["sampling_params"] == {
        "top_p": 1.0,
        "min_p": 0.02,
    }


def test_async_request_sglang_generate_nonstream_reads_full_json(monkeypatch):
    class _FakeResponse:
        status = 200
        reason = "OK"

        def __init__(self, payload):
            self._payload = payload
            self.content = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return self._payload

    class _FakeSession:
        def __init__(self, payload):
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            return _FakeResponse(self._payload)

    payload = {
        "text": "x" * 2048,
        "meta_info": {
            "completion_tokens": 7470,
            "spec_accept_length": 2.83,
        },
    }

    monkeypatch.setattr(
        bench_serving,
        "_create_bench_client_session",
        lambda: _FakeSession(payload),
    )
    monkeypatch.setattr(
        bench_serving,
        "args",
        SimpleNamespace(
            disable_stream=True,
            disable_ignore_eos=False,
            return_logprob=False,
            return_routed_experts=False,
            header=None,
        ),
        raising=False,
    )

    request = RequestFuncInput(
        prompt="prompt",
        api_url="http://unit.test/generate",
        prompt_len=1,
        output_len=7470,
        model="default",
        lora_name=None,
        image_data=None,
        extra_request_body={
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 7470,
            },
            "stream": False,
        },
    )

    output = asyncio.run(async_request_sglang_generate(request))

    assert output.success is True
    assert len(output.generated_text) == 2048
    assert output.output_len == 7470
    assert output.meta_info["completion_tokens"] == 7470
