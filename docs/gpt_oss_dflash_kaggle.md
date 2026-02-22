# Kaggle: GPT-OSS-120B + DFlash (FA3) on H100

This fork branch `gpt-oss-dflash` includes DFlash speculative decoding support
(based on upstream PR #16818) plus GPT-OSS-specific guardrails.

## Install (Kaggle notebook cell)

```bash
# Use the fork branch (contains DFLASH + GPT-OSS support).
pip install -U "sglang@git+https://github.com/radna0/sglang.git@gpt-oss-dflash#subdirectory=python"

# Kernel package for FA3.
pip install -U "sgl-kernel"
```

Notes:

- This fork makes `decord2` optional (`pip install sglang[multimodal]` if you need it).
- GPT-OSS + DFLASH enforces `--attention-backend=fa3` and `--speculative-draft-attention-backend=fa3`.

## Baseline Server (FA3)

```bash
python -m sglang.launch_server \
  --model-path "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1" \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --tp-size 1 \
  --mem-fraction-static 0.95 \
  --page-size 1 \
  --max-running-requests 1 \
  --max-total-tokens 12000 \
  --port 30000
```

## DFlash Server (FA3)

```bash
python -m sglang.launch_server \
  --model-path "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1" \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --tp-size 1 \
  --mem-fraction-static 0.93 \
  --page-size 1 \
  --max-running-requests 1 \
  --max-total-tokens 12000 \
  --port 30001 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "/kaggle/working/dflash_draft/epoch_0_step_11200" \
  --speculative-draft-attention-backend fa3 \
  --speculative-dflash-block-size 16
```

## DFlash-Tree Server (FA3) (experimental)

`DFLASH_TREE` uses a DFlash block draft but verifies with SGLang’s fused-tree speculative kernels
(EAGLE-style tree mask + verify). This can help when acceptance drops under higher concurrency.

```bash
python -m sglang.launch_server \
  --model-path "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1" \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --tp-size 1 \
  --mem-fraction-static 0.93 \
  --page-size 1 \
  --max-running-requests 1 \
  --max-total-tokens 12000 \
  --port 30002 \
  --speculative-algorithm DFLASH_TREE \
  --speculative-draft-model-path "/kaggle/working/dflash_draft/epoch_0_step_11200" \
  --speculative-draft-attention-backend fa3 \
  --speculative-dflash-block-size 16 \
  --speculative-num-steps 8 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 64
```

Notes:

- `--speculative-num-steps` must satisfy `<= block_size - 1`.
- `--speculative-num-draft-tokens` is the tree node budget for verification. If you set it to `16`
  (the block size), you effectively get a mostly linear chain (little/no extra branching).

## Benchmark sweep (DFLASH vs DFLASH_TREE)

If you have a repo checkout (so you can run `scripts/`), use the dedicated sweeper:

```bash
python3 scripts/playground/bench_speculative_dflash.py \
  --model-path "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1" \
  --speculative-draft-model-path "/kaggle/working/dflash_draft/epoch_0_step_11200" \
  --attention-backend fa3 \
  --speculative-draft-attention-backend fa3 \
  --dtype bfloat16 \
  --context-length 131072 \
  --mem-fraction-static 0.93 \
  --max-total-tokens 120000 \
  --tp-size 1 \
  --algorithms DFLASH_TREE \
  --batch-size 1 8 \
  --decode-len 8192 65536 \
  --block-size 16 \
  --spec-steps 15 \
  --tree-topk 4 \
  --num-verify-tokens 16 64 \
  --sampling-temperature 1.0 \
  --sampling-top-p 1.0 \
  --sampling-min-p 0.02 \
  --sampling-top-k -1 \
  --output dflash_sweep.jsonl
```

Notes:

- `accept_length` is the server’s `avg_spec_accept_length` (higher generally means fewer verify passes).
- `output_tok_s` is the serving output token throughput from `bench_serving`.
- Prefer `DFLASH_TREE` for production-like sampling (`temperature/top_p/min_p`). `DFLASH` remains greedy-verify only.

## Common Failure Mode: long decode on TP=1

For GPT-OSS-120B on a single H100 80GB, very long decodes can exceed KV-cache
capacity (especially for DFlash). If you need 16k+ decode reliably, use `--tp-size 8`.

If you must stay on TP=1, try reducing KV memory pressure:

- Lower `--max-total-tokens` to the smallest value that still fits your decode target.
- Consider `--kv-cache-dtype=fp8_e4m3` or `--kv-cache-dtype=fp8_e5m2` if acceptable for your use case.
