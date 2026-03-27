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

The current best long-budget reference regime on the local 3-problem harness is:

- target `page_size=1`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`

This is a greedy benchmark setup:

- `temperature=0.0`
- `top_p=1.0`
- `top_k=1`
- `min_p=0.0`
- `ignore_eos=True`

The corrected long-decode benchmark on this branch is now `showtime.py`-faithful in the
important sense:

- it keeps the original reference prompts unchanged
- it does **not** pad prompts out to `65k`
- it greedily decodes to the remaining context budget
- so the long-run measurements below are about long output continuation, not fake long input

## Current Status

- original GPT-OSS GQA works on `fa3` + `triton_kernel`
- GPT-OSS DFlash page-size handling is fixed
- paged target + inherited draft paging is still bad
- paged target + non-paged draft (`draft_page_size=1`) is the good regime
- `block_size=8` is the best current default on the short reference benchmark
- on the long-budget local reference harness, `page_size=1 / draft_page_size=1 / share_pools=False` is currently better than the paged-target setup
- DFlash overlap scheduling is still not production-ready on this branch
- on the corrected long-decode harness, the key acceptance variable is local continuation predictability, not simply "question difficulty"

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

## Long-Budget Reference Harness

Important: these `65k` / `131k` runs match the `showtime.py` serving budget style, not a filled-context stress test.

- the harness raises the server `context_length`
- it does **not** pad the prompt itself out to `65k` or `131k` tokens
- so these numbers are valid for the same budget regime, but not for a true full-context occupancy benchmark

Artifacts:

- `/workspace/dflash_longctx_20260327_metrics_v4/ctx65536_dec8192.json`
- `/workspace/dflash_longctx_20260327_metrics_v4/ctx131072_dec8192.json`
- `/workspace/dflash_longctx_20260327_controls/ctx65536_dec8192_page1_noshare.json`
- `/workspace/dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json`

### Paged Target, Non-Paged Draft

Regime:

- target `page_size=256`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`
- greedy decode

Results:

| run | baseline tok/s | dflash tok/s | accept | verify sum | verify avg | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `ctx65536_dec8192` | 149.581 | 209.309 | 3.056 | 8353 | 2784.333 | 1.399x |
| `ctx131072_dec8192` | 150.882 | 211.339 | 3.056 | 8353 | 2784.333 | 1.401x |

Interpretation:

- the nearly identical `65k` and `131k` rows confirm this harness is budget-matched, not prompt-filled
- the good short-context regime remains good here
- but it is not the best local long-budget regime we tested

### Non-Paged Target, Non-Paged Draft, No Sharing

Regime:

- target `page_size=1`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`
- greedy decode

Result:

| run | baseline tok/s | dflash tok/s | accept | verify sum | verify avg | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `ctx65536_dec8192_page1_noshare` | 149.299 | 226.235 | 3.212 | 7831 | 2610.333 | 1.515x |

Interpretation:

- on this local long-budget harness, disabling paging on the target as well is currently better
- acceptance improves from `3.056` to `3.212`
- verify count drops from `8353` to `7831`
- the speedup rises from about `1.40x` to about `1.52x`

So the current best-known regimes are now split:

- short-context throughput: `page=256 / draft_page=1 / block=8`
- local long-budget reference harness: `page=1 / draft_page=1 / share_pools=False / block=8`

## Corrected Showtime-Style Long Decode

This is the important regime for the reference problems:

- source prompts: `/root/reference.csv`
- prompt text unchanged
- no prompt padding
- greedy decode to remaining context budget
- `context_length=65536`
- `buffer_tokens=512`
- target `page_size=1`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`

Artifact:

- `/workspace/dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json`

Aggregate result:

| run | baseline tok/s | dflash tok/s | accept | verify sum | verify avg | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `ctx65536_page1_noshare_decodefill` | 183.648 | 310.940 | 3.388 | 57385 | 19128.333 | 1.693x |

Requested output budget per prompt:

- min: `64786`
- max: `64839`
- avg: `64813.0`

This is a real long-output decode-fill run, not the earlier fixed-`8192` decode budget.

## Per-Problem Acceptance Split

These are all plain no-tool `/generate` runs on the corrected long-decode harness. So the
long-accept behavior below is **not** a tool-calling effect.

Artifacts:

- `/workspace/dflash_showtime_decodefill_20260327/per_problem/92ba6a_block8.json`
- `/workspace/dflash_showtime_decodefill_20260327/per_problem/9c1c5f_block8.json`
- `/workspace/dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json`

Per-problem summary:

| problem | rough level | accept | verify sum | wall tok/s | notes |
|---|---|---:|---:|---:|---|
| `92ba6a` | easy | 7.086 | 9125 | 642.082 | spends long stretches at `7.9-8.0` acceptance |
| `9c1c5f` | medium | 5.914 | 10947 | 540.101 | starts verify-heavy, later climbs into near-cap acceptance |
| `a295e9` | medium-hard | 1.738 | 37313 | 160.321 | remains verify-heavy overall |

Notes:

- `a295e9` is inferred exactly from the completed 3-problem aggregate minus the two direct
  per-problem runs. It does not need a redundant second full rerun.
- `92ba6a` and `9c1c5f` both eventually enter a near-cap acceptance regime during forced
  long decode.
- `a295e9` does not do so enough to matter overall.

## What Actually Drives Long Acceptance

The best explanation is **local predictability of the continuation**.

That is more precise than just saying "easy question" or "hard question".

The empirical pattern is:

- early or mid active reasoning: lower acceptance, more verifies
- late stable tail: higher acceptance, sometimes near perfect

So the right interpretation is:

- high acceptance happens when the continuation has fallen into a low-entropy, high-confidence,
  easy-to-draft regime
- that can happen even after a genuinely hard problem, once the model is deep in a predictable tail

## Block Size 16 Investigation

The earlier `block_size=16` regression result was not trustworthy.

Two separate code issues were contaminating that conclusion:

1. Greedy `target_only` verify was ignoring `max_steps_per_req`.
   - That meant `physical_block=16 / logical_cap=8` was not actually capped in greedy mode.
   - The fix is in:
     - `python/sglang/srt/speculative/dflash_utils.py`
     - `python/sglang/srt/speculative/dflash_info.py`
2. The eager control path was also wrong.
   - `--disable-cuda-graph` still allowed piecewise CUDA graph capture to initialize.
   - That fix is in:
     - `python/sglang/srt/model_executor/model_runner.py`

So the corrected conclusion is:

- `block_size=16` is **not** collapsing because it is "unsupported"
- the earlier `16/8` measurements were invalid because the logical cap was dead on the greedy path
- after fixing that, `block=16` behaves normally on both the easy and hard controls below

### Corrected Easy Prefix Probe: `92ba6a`, decode `2048`

Artifacts:

- `/workspace/dflash_block_investigate_20260327/92ba6a_block8_eager2048.json`
- `/workspace/dflash_block_investigate_20260327/92ba6a_block16_eager2048.json`
- `/workspace/dflash_block_investigate_20260327/92ba6a_block16_force8_eager2048.json`

Results:

| run | physical block | logical cap | wall tok/s | accept len | step max | verify sum |
|---|---:|---:|---:|---:|---:|---:|
| `92ba6a_block8` | 8 | 8 | 269.906 | 3.537 | 7 | 579 |
| `92ba6a_block16` | 16 | 16 | 230.305 | 3.850 | 15 | 532 |
| `92ba6a_block16_force8` | 16 | 8 | 225.239 | 3.772 | 8 | 543 |

Interpretation:

- On the easy prefix, `block=16` accepts slightly more than `block=8`.
- But it is slower overall, so the extra physical width is mostly overhead here.
- The corrected `physical16/logical8` row now really is capped:
  - `spec_dflash_max_steps_mean = 8`
  - `spec_accept_length_step_max = 8`
- So the cap bug is fixed.

### Corrected Hard Prefix Probe: `a295e9`, decode `2048`

Artifacts:

- `/workspace/dflash_block_investigate_20260327_hard_v2/a295e9_block8_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_hard_v2/a295e9_block16_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_hard_v2/a295e9_block16_force8_eager2048.json`

Results:

| run | physical block | logical cap | wall tok/s | accept len | step max | verify sum |
|---|---:|---:|---:|---:|---:|---:|
| `a295e9_block8` | 8 | 8 | 21.687 | 1.159 | 7 | 1767 |
| `a295e9_block16` | 16 | 16 | 22.713 | 1.172 | 15 | 1747 |
| `a295e9_block16_force8` | 16 | 8 | 21.877 | 1.159 | 8 | 1767 |

Interpretation:

- On the hard prefix, `block=16` also does **not** collapse.
- `block=16` is slightly better than `block=8` on this probe:
  - slightly higher accept length
  - slightly fewer verifies
  - slightly better throughput
- The capped `physical16/logical8` row matches the `block=8` acceptance/verify behavior almost exactly.
- That means the corrected cap path is behaving as expected.

### Final Read

The right read now is:

- `block=16` itself is not broken
- the earlier bad `block16` conclusion came from:
  - dead greedy logical-cap plumbing
  - a bad eager benchmark path
- on easy/predictable continuations, larger physical blocks can raise acceptance but also add enough overhead that `block=8` still wins
- on hard/unpredictable continuations, both `block=8` and `block=16` are fundamentally draft-limited, so acceptance stays close to `1`

So the remaining work is not "make block 16 work at all." That part is now demonstrated.
The remaining work is:

- reduce verify overhead when acceptance is already good
- adapt the logical speculative depth when acceptance is bad
- export entropy/confidence signals and use them to drive that cap

This lines up with the same general signal family emphasized by EAFT:

- low entropy
- high confidence
- locally stable token distribution

Reference:

- https://github.com/PRIS-CV/EAFT

## Entropy / Confidence Hooks Already In Tree

The branch already computes the right DFlash-side difficulty signals for EAFT- or FailFast-style
policies.

Verify-side stats in:

- `python/sglang/srt/speculative/dflash_info.py`

Available scalar diagnostics:

- `accept_ratio_mean`
- `p_y_mean`
- `q_y_mean`
- `frac_p_y_zero`
- `frac_q_y_zero`
- `tv_mean`
- `p_entropy_mean`
- `q_entropy_mean`
- `p_max_mean`
- `q_max_mean`

Available step-wise diagnostics:

- `accept_ratio_mean_by_step`
- `tv_mean_by_step`
- `p_entropy_mean_by_step`
- `q_entropy_mean_by_step`
- `p_max_mean_by_step`
- `q_max_mean_by_step`

Draft-side confidence debug in:

- `python/sglang/srt/speculative/dflash_worker.py`

Available draft debug signals:

- `q_max_mean_first`
- `q_ent_mean_first`

Relevant env flags:

- `SGLANG_DFLASH_PQ_SCALAR_STATS=1`
- `SGLANG_DFLASH_PQ_DIAG_STATS=1`
- `SGLANG_DFLASH_PQ_STEP_STATS=1`
- `SGLANG_DFLASH_DRAFT_CONF_DEBUG=1`

So the entropy/probability instrumentation already exists. The next productization step is to
surface these per-request signals into benchmark JSON directly instead of only keeping them in
verify-side debug structures or logs.

## Adaptive Design Choice

Best next design:

- keep a fixed physical max block
- vary a logical effective length per request / per round

Why this is the right first move:

1. `DFlashWorker` is built around a fixed physical `block_size`.
2. `block_size` is threaded through:
   - buffer allocation
   - KV-slot accounting
   - proposal tensor shapes
   - CUDA-graph capture shapes
3. The verifier already supports exact early stop through `max_steps_per_req`.
4. The worker already contains DAWN/FailFast-style cap plumbing.
5. Pre-capturing multiple physical block widths would increase:
   - graph capture time
   - memory pressure
   - switching complexity
   - duplicated runtime paths

That means the fastest safe path is:

1. fixed physical block
2. adaptive logical cap
3. only later, if needed, multi-width capture

## How FailFast Maps to GPT-OSS DFlash

FailFast-style logic is most useful on the hard regime, not the easy one.

For `a295e9`-like segments:

- low acceptance
- many verifies
- high speculative overhead

That is where we should shrink the effective speculative length aggressively.

For `92ba6a`-like late tails:

- acceptance is already near the block cap
- shrinking does not help much
- larger physical block may help more than additional caution

So the policy should be asymmetric:

- hard regime: shrink logical speculative length
- easy regime: allow the full block
- easy late tails are also the best candidates for testing larger physical blocks

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

For GPT-OSS DFlash on this branch, prefer one of two regimes.

Short-context / throughput-oriented:

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

Long-budget local reference harness:

```bash
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=0

sglang serve \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 1 \
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
- `/workspace/dflash_longctx_20260327_metrics_v4/ctx65536_dec8192.json`
- `/workspace/dflash_longctx_20260327_metrics_v4/ctx131072_dec8192.json`
- `/workspace/dflash_longctx_20260327_controls/ctx65536_dec8192_page1_noshare.json`

## Immediate Next Work

1. Build a true filled-context stress harness instead of only changing `context_length`.
2. Compare `page=1 / draft=1 / no-share` against `page=256 / draft=1` on that filled-context harness.
3. Optimize DFlash verify / allocator behavior in the good regime.
4. Investigate adaptive block-size selection. `failfast` is now cloned at `/workspace/failfast` for that follow-up work.
