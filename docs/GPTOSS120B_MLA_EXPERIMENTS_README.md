# GPT-OSS-120B MLA Experiments README

This document records what we actually did for the GPT-OSS-120B CARE/CARE-E/MLA work, what changed in the runtime, and which results are trustworthy.

## Executive summary

The project started as a paper-faithful CARE reproduction and ended up as a GPT-OSS-120B MLA correctness and serving effort:

- We reproduced the CARE-style covariance pipeline and compared against the paper-style `Alpaca`/`C4` calibration shape.
- We extended the calibration and conversion path to GPT-OSS-120B with larger, mixed corpora and much larger coverage than the paper's smallest setup.
- We fixed several runtime issues that were invalidating the low-rank MLA path:
  - RoPE / sink / sliding-window handling
  - MXFP4 load-time swizzle behavior
  - MoE top-k format selection in compile warmup
  - BF16 KV cache selection
  - memory pool sizing for a single H100
- We validated the corrected serving path with a one-token FlashMLA smoke on an absorbed `r128` checkpoint.
- We validated deterministic `lm-eval` logprob comparisons for `r1024` and `r512`.

The main takeaway is simple: the earlier bad behavior was caused by engineering/runtime mismatches, not by MLA being impossible.

## What we were trying to reproduce

The original CARE paper is a zero-shot conversion method:

1. Collect covariance statistics.
2. Derive a rank schedule.
3. Convert the checkpoint.
4. Benchmark before any healing or SFT recovery.

We kept that structure, but adapted it to GPT-OSS-120B:

- GPT-OSS-120B is an MXFP4 MoE model, not a small dense Llama.
- We care about native MLA behavior, dynamic per-layer rank, sinks, and sliding-window semantics.
- We also want a serving path that can later support DSA/NSA and DFlash-style production infrastructure.

## Experiment phases

| Phase | What we did | Status |
| --- | --- | --- |
| CARE reproduction baseline | Built paper-faithful covariance runs using Alpaca-like calibration and a small `128x2048`-style setup | Done |
| Extended covariance / conversion | Ran larger multi-corpus covariance and conversion passes for GPT-OSS-120B | Done |
| Runtime correctness | Fixed sink handling, sliding-window handling, BF16 KV cache, and the preswizzled MXFP4 MoE path | Done |
| FlashMLA smoke | Served an absorbed `r128` checkpoint with `flashmla` on a single H100 and verified a real completion | Done |
| Deterministic eval | Ran `lm-eval` logprob on a fixed subset for `r1024` and `r512` | Done |
| Low-rank audit | Investigated `r256` / `r128` failures and found truncated / incomplete absorbed exports in the earlier path | Done |

## What changed in the code

The important runtime fixes landed in these places:

- `python/sglang/srt/layers/quantization/mxfp4.py`
  - Hopper now auto-selects the `triton_kernel` MoE backend for GPT-OSS MXFP4 so the weights stay swizzled instead of being upcast to dense BF16 during load.
- `python/sglang/srt/managers/scheduler.py`
  - The scheduler now switches GPT-OSS MXFP4 to `triton_kernel` before `initialize_moe_config()`, so the MoE runner and top-k format agree from startup.
- `python/sglang/srt/layers/utils/multi_platform.py`
  - `TopK` no longer silently falls back to the wrong format during compile warmup when the preswizzled backend is already active.
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
  - `max_running_requests` is now stored early enough for routed experts capture and memory sizing.
- `python/sglang/srt/server_args.py`
  - GPT-OSS MXFP4 defaults now prefer the triton-kernel path on Hopper and use a safer static memory fraction.
- `python/sglang/srt/layers/attention/flashmla_backend.py`
  - FlashMLA sink/sliding-window handling was corrected for GPT-OSS MLA use.

## What we validated

### 1. Manual regression checks

The local manual regression file passed:

- `test/manual/test_gpt_oss_care_cpu_checks.py`
- Result: `23 passed`

That file now checks:

- GPT-OSS FlashMLA smoke metadata
- GPT-OSS backend selection rules
- Hopper MXFP4 preswizzle behavior
- top-k format preservation during compile warmup
- MLA sink/sliding wiring
- KV-cache dtype behavior

### 2. Serving smoke

The current validated smoke run is:

- `smoke_preswizzle_r128_v7`

The smoke:

- loaded `/workspace/r128_absorbed`
- used `flashmla`
- used `bfloat16` KV cache
- allocated the KV pool successfully
- served `/v1/models`
- returned a valid one-token completion

The final result JSON on this host is:

- `/workspace/smoke_preswizzle_r128_v7/result.json`

### 3. Deterministic benchmark subset

For the logprob path we intentionally used `lm-eval` completions with `max_tokens=1`, not sampling.

The key subset run compared `r1024` and `r512` and produced a near-lossless result for the corrected path:

| Task | `r1024` | `r512` |
| --- | ---: | ---: |
| ARC-Challenge `acc_norm` | 0.53125 | 0.53125 |
| ARC-Easy `acc` | 0.75 | 0.75 |
| HellaSwag `acc_norm` | 0.65625 | 0.65625 |
| PIQA `acc_norm` | 0.8125 | 0.8125 |
| MMLU `acc` | 0.741776 | 0.734649 |
| OpenBookQA `acc_norm` | 0.5 | 0.5 |
| RACE `acc` | 0.34375 | 0.34375 |
| WinoGrande `acc` | 0.8125 | 0.8125 |

The matrix file for that run lives at:

- `/workspace/gptoss120b_lmeval_logprob_subset_20260320_081507/matrix_r1024_vs_r512.md`

## Important caveats

There are three separate things that can look like “MLA is broken” when they are not:

1. **Loader / format mismatch**
   - This happened when the absorbed checkpoint did not have the correct rank-specific attention shard or when the runtime expected the wrong top-k format.
2. **Runtime / kernel mismatch**
   - This happened when the backend was selected too late or compile warmup silently changed the top-k format.
3. **Incomplete low-rank exports**
   - Earlier `r256` / `r128` paths were found to be incomplete, so older low-rank numbers from those exports are not trustworthy.

The current state of the project is that the serving path is corrected and the low-rank comparison machinery is now reliable enough to rerun once complete exports are regenerated.

## Current interpretation

What we can say confidently:

- `r1024` is our near-original anchor.
- `r512` is close to `r1024` on the corrected deterministic subset.
- The preswizzled FlashMLA serving path now starts and answers requests on the single-H100 host.
- The earlier engineering blockers were fixed in the runtime, not by changing the conversion method itself.

What we cannot claim yet:

- A fully finalized, full-suite paper-style parity table across every rank and every benchmark on complete exports.
- That the earlier `r256` / `r128` evaluation numbers were valid, because the exports were later found to be incomplete.

## Where to look next

If you want the operational reproduction details, see:

- `docs/GPTOSS120B_MLA_REPRODUCTION_README.md`

If you want the broader conversion / evaluation tracking, the companion planning docs in `docs/` are:

- `docs/GPTOSS120B_CARE_FIXEDR512_BIG_RUN_PLAN.md`
- `docs/GPTOSS120B_ZERO_SHOT_MLA_FORMALIZATION.md`
- `docs/GPTOSS120B_DSA_NSA_GAP_AND_IMPLEMENTATION.md`

