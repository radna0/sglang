# GPT-OSS GQA DSA (Lightning Indexer) in SGLang

This repo adds an **experimental** DeepSeek-style *Lightning Indexer* ("DSA/NSA-style") path for
**`GptOssForCausalLM`** that keeps GPT-OSS **native GQA/MHA** and applies sparse top-k attention
**only on `full_attention` layers**.

## What's implemented (today)

- **Attention backend:** `fa3` (FlashAttention v3) is required.
- **Sparse scope:** only GPT-OSS layers with `layer_type == "full_attention"`.
- **Sliding-window layers:** stay dense + unchanged.
- **Attention sinks:** forwarded via the existing `sinks=...` kwarg.
- **Indexer contract:** uses `sglang.srt.layers.attention.nsa.nsa_indexer.Indexer` and passes
  `topk_indices` into `RadixAttention`.
- **Sparse mode:** decode-only; prefill/extend remains dense but populates the index-K cache for later decode steps.

## Server flags

- Enable DSA:
  - `--enable-gpt-oss-gqa-dsa`
- Control sparse top-k:
  - `--gpt-oss-dsa-index-topk 2048`
  - `--gpt-oss-dsa-index-head-dim 128`
  - `--gpt-oss-dsa-index-block-size 128`

## Recommended launch (H100 / SM90)

Use FA3 for attention and `triton_kernel` for MoE (if available):

```bash
python -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --enable-gpt-oss-gqa-dsa \
  --gpt-oss-dsa-index-topk 2048 \
  --kv-cache-dtype bfloat16 \
  --page-size 64
```

## Notes / limitations

- CUDA runs require `deep_gemm` to be importable (the indexer calls `deep_gemm.get_num_sms()`).
- This patch does **not** provide trained indexer weights; the indexer parameters will be
  randomly initialized unless you load a checkpoint that includes them.
- Because prefill is still dense, the main speed win is expected on **long-context decode**
  (generation) rather than initial prompt ingestion.
