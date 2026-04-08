# GPT-OSS GQA DSA (FA3) Modal Bench

This folder contains a small Modal harness to benchmark GPT‑OSS‑120B GQA decode
throughput with and without GPT‑OSS “DSA” (sparse decode via FA3 `page_table`).

Key properties:

- Prefill stays dense; sparse activates only during decode (and only for full‑attention layers).
- Default is `topk=2048` and `topk_source=recent` (capture‑friendly debug lane).
- Designed to saturate the GPU via `concurrent_requests=8` + `cuda_graph_bs=8`.
- Runs a BF16 KV cache benchmark and then repeats the same settings with FP8_E4M3 KV cache.

Run (from the SGLang repo root):

```bash
modal run benchmark/gpt_oss/gqa_dsa/modal_bench.py --prompt-len 65000 --allow-auto-truncate
```

Notes:

- The harness downloads the model locally with `hf download` excluding `metal/*` and `original/*`.
- If `--allow-auto-truncate` is enabled, the harness parses server logs to infer the effective prompt length.

