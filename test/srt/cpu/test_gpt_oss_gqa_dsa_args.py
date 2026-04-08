import argparse
import os

import pytest


if os.name == "nt":
    pytest.skip("SGLang SRT tests require Linux (resource module).", allow_module_level=True)


def test_gpt_oss_gqa_dsa_cli_args_parse():
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw = parser.parse_args(
        [
            "--model-path",
            "dummy",
            "--enable-gpt-oss-gqa-dsa",
            "--gpt-oss-dsa-index-topk",
            "1024",
            "--gpt-oss-dsa-index-head-dim",
            "128",
        ]
    )
    server_args = ServerArgs.from_cli_args(raw)
    assert server_args.enable_gpt_oss_gqa_dsa is True
    assert server_args.gpt_oss_dsa_index_topk == 1024
    assert server_args.gpt_oss_dsa_index_head_dim == 128
