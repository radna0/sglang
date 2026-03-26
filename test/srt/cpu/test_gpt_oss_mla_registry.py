import unittest
from pathlib import Path


class TestGptOssMlaRegistry(unittest.TestCase):
    def test_entryclass_and_mla_wrapper_are_present_in_source(self):
        source = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/models/gpt_oss.py"
        ).read_text()

        self.assertIn("class GptOssMLAAttention", source)
        self.assertIn("class GptOssMLADecoderLayer", source)
        self.assertIn("class GptOssMlaForCausalLM", source)
        self.assertIn(
            "EntryClass = [GptOssForCausalLM, GptOssMlaForCausalLM]", source
        )


if __name__ == "__main__":
    unittest.main()
