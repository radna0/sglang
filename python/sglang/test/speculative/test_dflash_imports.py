import importlib
import sys
import types
import unittest


class TestDflashImportRobustness(unittest.TestCase):
    def setUp(self):
        if sys.platform == "win32":
            raise unittest.SkipTest("SGLang runtime is not supported on Windows.")

    def test_token_dispatcher_import_survives_broken_flashinfer(self):
        # Simulate a partially-broken FlashInfer install (present package, missing submodules).
        sys.modules["flashinfer"] = types.ModuleType("flashinfer")

        mod = importlib.import_module("sglang.srt.layers.moe.token_dispatcher")
        importlib.reload(mod)

    def test_server_args_exposes_dflash(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        # Ensure DFLASH is an accepted choice for the CLI.
        self.assertIn(
            "DFLASH",
            parser._option_string_actions["--speculative-algorithm"].choices,
        )
