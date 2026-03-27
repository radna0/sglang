# GPT-OSS DFlash Status on `dflash`

This branch is focused on making GPT-OSS DFlash correct and fast on the stable:

- target attention backend: `fa3`
- target MoE backend: `triton_kernel`
- target KV cache: `fp8_e4m3`
- draft KV cache: `bfloat16`

The current best short-context regime is:

- target `page_size=256`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`

This is a greedy benchmark setup:

- `temperature=0.0`
- `top_p=1.0`
- `top_k=1`
- `min_p=0.0`
- `ignore_eos=True`

## Current Status

- original GPT-OSS GQA works on `fa3` + `triton_kernel`
- GPT-OSS DFlash page-size handling is fixed
- paged target + inherited draft paging is still bad
- paged target + non-paged draft (`draft_page_size=1`) is the good regime
- `block_size=8` is the best current default on the short reference benchmark
- DFlash overlap scheduling is still not production-ready on this branch

## Why GPT-OSS DFlash Was Slow

The main GPT-OSS DFlash failure mode was not the draft checkpoint. It was page-size handling.

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
- acceptance returns to about `3.0+`
- DFlash beats baseline

## Reference Benchmark Setup

Reference benchmark:

- local problems from `/root/reference.csv`
- target model: `/workspace/offload_root/gpt-oss-120b`
- draft model: `/root/epoch_65_step_23760`
- attention backend: `fa3`
- MoE backend: `triton_kernel`
- target KV dtype: `fp8_e4m3`
- draft KV dtype: `bfloat16`

## Page Size / Block Size Matrix

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
- `block_size=8` is the best current short-context default

## Fused vs Unfused KV Materialization

Short reference benchmark:

- `context_length=8192`
- `decode_len=2048`
- `concurrency=8`
- `num_prompts=8`
- target `page_size=256`
- draft `page_size=1`
- `block_size=8`

### Fused KV Enabled

Artifact:

- `/workspace/dflash_timing_ab_20260327/short_ctx8192_dec2048_fused.json`

Result:

- `wall_tok_s = 832.947`
- `speedup = 1.3088x`
- `accept_length = 3.150`
- `accept min/max step = 1 / 8`
- `accept draft-token min/max = 0 / 7`
- `verify_ct_sum = 5277`
- `verify_ct_avg = 659.625`
- `verify_ct_min/max = 577 / 798`
- `accept_token_sum = 11099`
- `step_time_p20_s = 0.016213`
- `output_tok_s_p20 = 194.274`

One-shot timing:

- `verify wall = 0.027609s`
- `draft_kv_append wall = 0.003420s`

### Fused KV Disabled

Artifact:

- `/workspace/dflash_timing_ab_20260327/short_ctx8192_dec2048_nofused.json`

Result:

- `wall_tok_s = 816.565`
- `speedup = 1.2830x`
- `accept_length = 3.175`
- `accept min/max step = 1 / 8`
- `accept draft-token min/max = 0 / 7`
- `verify_ct_sum = 5237`
- `verify_ct_avg = 654.625`
- `verify_ct_min/max = 558 / 798`
- `accept_token_sum = 11139`
- `step_time_p20_s = 0.016491`
- `output_tok_s_p20 = 192.543`

One-shot timing:

- `verify wall = 0.027493s`
- `draft_kv_append wall = 0.001636s`

### Interpretation

Fused KV is not the main bottleneck.

- fused is only about `2.0%` faster on wall tok/s than unfused
- fused slightly worsens acceptance
- fused slightly increases total verify count

So the current ceiling is still dominated by verify-side overhead, not by CPU commit or by the fused/unfused append path alone.

## Overhead Diagnosis

Why `accept_length ~= 3.1` does not become a `3x` throughput win:

- baseline `step_time_p20_s` is about `0.006838`
- DFlash `step_time_p20_s` is about `0.0162 - 0.0165`
- that means a speculative step still costs about `2.4x` a baseline step

So even with `accept_length ~= 3.1`, the achievable gain is much smaller than `3x`.

The current evidence says:

- CPU pack/commit is not the main issue
- fused KV materialization is not the main issue
- the real remaining cost is still in:
  - verify path
  - allocator / KV-free behavior
  - DFlash-specific scheduling overhead

## Overlap Scheduling Status

Overlap scheduling is still disabled for DFlash on the main path.

Reason:

- the current overlap scheduler is still Eagle/spec-v2 shaped
- it expects `next_draft_input`, `topk_p`, `topk_index`, and future-token payloads
- DFlash does not return that payload contract yet

So this is not just a hidden flag. It still needs real DFlash-specific overlap plumbing before it is safe to enable by default.

## Code Changes on This Branch

Relevant files:

- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_utils.py`
- `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- `python/sglang/bench_serving.py`
- `scripts/playground/bench_reference_dflash.py`

Implemented:

- explicit draft-page handling for GPT-OSS DFlash
- GPT-OSS fused KV path now supports QKV bias
- DFlash verify now records acceptance histograms correctly
- benchmark helper now preserves per-request `meta_info`
- reference benchmark now reports:
  - mean accept length
  - min/max step accept length
  - min/max accepted draft tokens
  - total/avg/min/max `spec_verify_ct`

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
  --speculative-draft-page-size 1 \
  --speculative-dflash-block-size 8 \
  --speculative-moe-runner-backend triton_kernel
```

## Benchmark Command

Short reference benchmark:

```bash
export PYTHONPATH=/workspace/sglang-dflash-line/python
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_DFLASH_TIMING=1

/venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/bench_reference_dflash.py \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --draft-model-path /root/epoch_65_step_23760 \
  --reference-csv /root/reference.csv \
  --context-length 8192 \
  --decode-len 2048 \
  --concurrency 8 \
  --num-prompts 8 \
  --page-size 256 \
  --draft-page-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --draft-kv-cache-dtype bfloat16 \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --draft-attention-backend fa3 \
  --speculative-moe-runner-backend triton_kernel \
  --speculative-dflash-block-size 8 \
  --mem-fraction-static 0.90
```

## Artifact Paths

Key artifacts:

- `/workspace/dflash_pagesize_matrix_20260327/target128_auto.json`
- `/workspace/dflash_pagesize_matrix_20260327/target128_draft1.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_auto.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block4.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block8.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block16_timed.json`
- `/workspace/dflash_timing_ab_20260327/short_ctx8192_dec2048_fused.json`
- `/workspace/dflash_timing_ab_20260327/short_ctx8192_dec2048_nofused.json`
- `/workspace/dflash_longctx_20260327_fixeddraft1_block8_v3/ctx65536_dec8192.json`
- `/workspace/dflash_longctx_20260327_fixeddraft1_block8_v3/ctx131072_dec8192.json`

## Immediate Next Work

1. Re-run the `65k` reference benchmark on the corrected `page=256 / draft_page=1 / block=8` regime and record verify counts there too.
2. Re-run the `131k` reference benchmark on the same regime.
3. Optimize DFlash verify / allocator behavior in the good regime.
4. Investigate adaptive block-size selection. `failfast` is now cloned at `/workspace/failfast` for that follow-up work.
