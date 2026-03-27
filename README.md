# GPT-OSS DFlash Status on `dflash`

This branch is currently focused on making GPT-OSS DFlash correct and fast on the `fa3` + `triton_kernel` path.

Current status:
- original GPT-OSS GQA model works on `fa3` attention + `triton_kernel` MoE
- GPT-OSS DFlash page-size handling is fixed
- GPT-OSS DFlash now auto-defaults to `draft_page_size=1` on paged targets
- GPT-OSS DFlash now defaults to `block_size=8` when the draft config does not specify a block size
- the current best short-context regime is:
  - target `page_size=256`
  - draft `page_size=1`
  - `share_pools=False`
  - `block_size=8`
  - target KV cache `fp8_e4m3`
  - draft KV cache `bfloat16`

## Why This Matters

The main GPT-OSS DFlash failure mode was not the draft checkpoint itself. It was page-size handling.

Bad regime:
- target paged KV
- draft inherits the same page size
- pool sharing stays enabled
- acceptance collapses to about `1.3`
- DFlash loses badly to baseline

Good regime:
- target stays paged
- draft runs non-paged with `draft_page_size=1`
- pool sharing is disabled
- acceptance returns to about `3.0`
- DFlash beats baseline

## Current Benchmark Table

Reference benchmark:
- local problems from `/root/reference.csv`
- target model: `/workspace/offload_root/gpt-oss-120b`
- draft model: `/root/epoch_65_step_23760`
- attention backend: `fa3`
- MoE backend: `triton_kernel`
- target KV dtype: `fp8_e4m3`
- draft KV dtype: `bfloat16`

| run | page | draft_page | block | accept | wall tok/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| target128_inherit | 128 | inherit=128 | 16 | 1.308 | 343.541 | 0.532x |
| target128_draft1 | 128 | 1 | 16 | 3.060 | 706.925 | 1.096x |
| target256_inherit | 256 | inherit=256 | 16 | 1.301 | 329.256 | 0.517x |
| target256_draft1 | 256 | 1 | 16 | 3.077 | 708.020 | 1.113x |
| target256_draft1_block4 | 256 | 1 | 4 | 2.581 | 778.190 | 1.223x |
| target256_draft1_block8 | 256 | 1 | 8 | 3.175 | 815.444 | 1.281x |
| target256_draft1_block16 | 256 | 1 | 16 | 3.077 | 719.152 | 1.130x |

Conclusion:
- `draft_page_size=1` is required for GPT-OSS DFlash on paged targets
- `block_size=8` is the best current default on the short reference benchmark

## Overhead Diagnosis

The remaining overhead is not from CPU commit or D2H pack.

One-shot verify timing detail in the good regime:

| run | mode | bs | tokens_committed | t_accept | t_pack_d2h | t_commit_cpu | t_kv_free | t_mapping | t_hidden | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| block4 | target_only | 1 | 1 | 0.013238 | 0.000079 | 0.000093 | 0.051702 | 0.000149 | 0.000259 | 0.065521 |
| block8 | target_only | 1 | 1 | 0.013768 | 0.000083 | 0.000083 | 0.050805 | 0.000115 | 0.000240 | 0.065096 |
| block16 | target_only | 1 | 1 | 0.014916 | 0.000078 | 0.000079 | 0.012335 | 0.000103 | 0.000219 | 0.027732 |

Interpretation:
- `t_pack_d2h` is negligible
- `t_commit_cpu` is negligible
- the real optimization targets are:
  - verify / accept path
  - KV free / allocator behavior
  - fused verify-side operations on the good `draft_page_size=1` regime

## Code Changes in This Branch

Relevant files:
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/speculative/dflash_worker.py`

Implemented:
- explicit `--speculative-draft-page-size`
- resolved draft page config logging
- GPT-OSS auto `draft_page_size=1` on paged targets
- GPT-OSS default `block_size=8` when unspecified

## Recommended Launch Regime

For GPT-OSS DFlash on this branch, prefer:

```bash
sglang serve \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 256 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /root/epoch_65_step_23760 \
  --speculative-draft-attention-backend fa3 \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --speculative-moe-runner-backend triton_kernel
```

Notes:
- GPT-OSS DFlash will auto-default `draft_page_size=1`
- GPT-OSS DFlash will auto-default `block_size=8` if the draft config does not specify one

## Benchmark Commands

Short reference benchmark:

```bash
export PYTHONPATH=/workspace/sglang-dflash-line/python
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

/venv/main/bin/python /workspace/sglang-dflash-gqa-fix/scripts/playground/bench_reference_dflash.py \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --draft-model-path /root/epoch_65_step_23760 \
  --reference-csv /root/reference.csv \
  --context-length 8192 \
  --decode-len 2048 \
  --concurrency 8 \
  --num-prompts 8 \
  --page-size 256 \
  --kv-cache-dtype fp8_e4m3 \
  --draft-kv-cache-dtype bfloat16 \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --draft-attention-backend fa3 \
  --speculative-moe-runner-backend triton_kernel \
  --mem-fraction-static 0.90
```

Long-context 65k benchmark to run next:

```bash
export PYTHONPATH=/workspace/sglang-dflash-line/python
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

/venv/main/bin/python /workspace/sglang-dflash-gqa-fix/scripts/playground/bench_reference_dflash.py \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --draft-model-path /root/epoch_65_step_23760 \
  --reference-csv /root/reference.csv \
  --context-length 65536 \
  --decode-len 8192 \
  --concurrency 1 \
  --num-prompts 3 \
  --page-size 256 \
  --kv-cache-dtype fp8_e4m3 \
  --draft-kv-cache-dtype bfloat16 \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --draft-attention-backend fa3 \
  --speculative-moe-runner-backend triton_kernel \
  --mem-fraction-static 0.90
```

Long-context 131k benchmark to run after that:

```bash
export PYTHONPATH=/workspace/sglang-dflash-line/python
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

/venv/main/bin/python /workspace/sglang-dflash-gqa-fix/scripts/playground/bench_reference_dflash.py \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --draft-model-path /root/epoch_65_step_23760 \
  --reference-csv /root/reference.csv \
  --context-length 131072 \
  --decode-len 8192 \
  --concurrency 1 \
  --num-prompts 3 \
  --page-size 256 \
  --kv-cache-dtype fp8_e4m3 \
  --draft-kv-cache-dtype bfloat16 \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --draft-attention-backend fa3 \
  --speculative-moe-runner-backend triton_kernel \
  --mem-fraction-static 0.90
```

## Artifact Paths

Current local results live under:
- `/workspace/dflash_pagesize_matrix_20260327`

Key artifacts:
- `/workspace/dflash_pagesize_matrix_20260327/target128_auto.json`
- `/workspace/dflash_pagesize_matrix_20260327/target128_draft1.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_auto.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block4.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block8.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block16_timed.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_auto_afterfix_quick.log`

## Immediate Next Work

1. Re-run the 65k reference benchmark on the corrected regime.
2. Re-run the 131k reference benchmark on the corrected regime.
3. Optimize verify / allocator behavior in the good `draft_page_size=1` regime.
4. Investigate fused verify-side operations for GPT-OSS, since CPU-side pack/commit is not the bottleneck.
