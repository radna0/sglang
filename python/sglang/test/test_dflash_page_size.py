import unittest


class TestDFlashPageSize(unittest.TestCase):
    def test_dflash_allows_paged_kv(self):
        try:
            from sglang.srt.server_args import ServerArgs
        except ModuleNotFoundError as e:
            # Some dev environments don't have Triton installed; importing SGLang may fail.
            self.skipTest(f"SGLang import unavailable in this environment: {e}")

        # Use dummy model path so ServerArgs doesn't try to load HF configs.
        args = ServerArgs(model_path="dummy")

        # Configure just enough fields for _handle_speculative_decoding() to run.
        args.speculative_algorithm = "DFLASH"
        args.speculative_draft_model_path = "dummy-draft-path"
        args.pp_size = 1
        args.enable_dp_attention = False

        # Pretend the caller requested paged KV; DFLASH should keep it.
        args.page_size = 128

        # Avoid touching filesystem / HF by specifying the block size directly.
        args.speculative_dflash_block_size = 16
        args.speculative_num_draft_tokens = None

        args._handle_speculative_decoding()
        self.assertEqual(args.page_size, 128)


if __name__ == "__main__":
    unittest.main()
