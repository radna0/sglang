# DFlash Production Verify Roadmap

This note records the current implementation state and the next development
steps for production DFlash on this branch.

It is intentionally limited to production-relevant paths:

- greedy target-only verify
- sampled target-only verify
- FP8 target KV cache
- BF16 draft KV cache
- overlap/spec-v2 scheduling

`pq` verify is explicitly removed from the active roadmap. It is not the
production path we want to optimize.

## Current Confirmed State

### 1. Overlap/spec-v2 is disabled for DFlash

Current DFlash runs do **not** use overlap scheduling.

Code:
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/speculative/spec_info.py`

Current behavior:
- `server_args.py` forces `disable_overlap_schedule = True` for DFlash
- `spec_info.py` rejects overlap for DFlash with:
  - `"DFLASH does not support overlap scheduling (spec v2)."`

So today:
- no single-batch overlap
- no two-batch overlap
- no DFlash-specific spec-v2 future payload

### 2. Current production verify mode is `target_only`

Code:
- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_worker.py`

Current behavior:
- `verify_mode` defaults to `target_only`
- recent benchmark runs record `spec_dflash_verify_mode_last = "target_only"`

This is the path to optimize first.

### 3. Greedy and sampled target-only paths already exist

Code:
- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/layers/sampler.py`

#### Greedy target-only

When `sampling_info is None` or `sampling_info.is_all_greedy`:
- target verify uses `argmax` over target logits for each verify position
- acceptance is exact-match against those target predictions

#### Sampled target-only

When sampling is enabled:
- target logits are temperature-scaled
- target-only verify applies target-side sampling semantics using:
  - `temperature`
  - `top_k`
  - `top_p`
  - `min_p`
- target tokens are sampled from the target distribution for each verify position
- draft tokens are accepted only while they match sampled target tokens

So sampled target-only is already a real code path. It is the correct
production path to benchmark against baseline sampling.

### 4. What is fused today is KV materialization, not verify

Code:
- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`

Current fused path:
- batched KV projection
- RMSNorm
- RoPE
- draft KV pool writes

This is used when appending projected target hidden states into the draft KV
cache.

This is **not** a fused target verification kernel.

### 5. FP8 target KV and BF16 draft KV are separate today

Code:
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/speculative/dflash_worker.py`

Current behavior:
- target KV cache can be `fp8_e4m3`
- draft KV cache can be `bfloat16`
- they are handled in separate paths / separate pools

There is no current mixed-precision fused verify kernel that directly combines:
- target verify over FP8 target KV
- draft append over BF16 draft KV

## Roadmap: What We Actually Need

## A. DFlash-specific overlap/spec-v2 support

We need a DFlash-native spec-v2 payload instead of trying to reuse Eagle-only
plumbing.

Required design work:

1. Define a DFlash verify payload for overlap scheduling.
   It needs at least:
   - accepted lengths
   - accepted token ids
   - bonus token ids
   - request-to-sequence mapping
   - logical step count / block cap metadata

2. Thread that payload through:
   - scheduler
   - output processor
   - CUDA graph replay path
   - two-batch overlap / future-index handling

3. Keep DFlash flat-block semantics intact.
   This should not pretend DFlash is an Eagle tree.

Implementation target:
- add DFlash-specific overlap glue rather than trying to coerce Eagle
  `VerifyInput` types into DFlash.

## B. Fused verify/commit for target-only

The production target is **not** PQ.
It is target-only verify for:
- greedy target
- sampled target

The right breakdown is:

1. Target verify forward
   - target model reads FP8 target KV
   - target model produces verify logits

2. On-device verify/postprocess
   - greedy target-only:
     - argmax target tokens
     - exact-match accept mask
     - accept length computation
     - bonus token selection
   - sampled target-only:
     - temperature/top-k/top-p/min-p filtering
     - target token sampling
     - exact-match accept mask
     - accept length computation
     - bonus token selection

3. Commit / append path
   - accepted target hidden states are projected into draft-space
   - draft KV append uses BF16 path
   - this is where current fused KV materialization already helps

So the next fusion target is **not** “fuse FP8 target KV and BF16 draft KV into
one giant kernel.”

The more realistic target is:
- keep FP8 target attention/verify where it is
- fuse target-only verify postprocessing and commit bookkeeping on device
- then feed accepted target hidden into the existing draft BF16 fused append path

That is the mixed-precision boundary that matters in practice.

## C. Sampled production path to optimize first

For production benchmarking, we should split two regimes:

1. sampled target + greedy draft
2. sampled target + sampled draft

Order of implementation:

1. sampled target + greedy draft
   - production-first
   - closest to deployed speculative decoding practice

2. sampled target + sampled draft
   - research lane
   - benchmark separately

We should not mix these into one bucket.

## D. What we are explicitly *not* doing now

- not optimizing `pq` verify
- not treating PQ as the production path
- not claiming current fused KV append means verify is fused
- not claiming DFlash overlap/spec-v2 already works

## Immediate Engineering Plan

1. Keep benchmark lanes running and recorded as-is.
2. Add DFlash overlap/spec-v2 design notes to the scheduler side.
3. Map exact target-only verify hot path for:
   - greedy
   - sampled target
4. Design fused verify/postprocess kernel boundaries.
5. Reuse current fused BF16 draft KV append path after verify/commit.
6. Benchmark:
   - baseline sampled target
   - DFlash sampled target + greedy draft
   - later DFlash sampled target + sampled draft

## Success Criteria

We should consider the roadmap successful only when all of the following are true:

1. DFlash can run with overlap/spec-v2 scheduling enabled.
2. DFlash target-only verify uses a fused on-device postprocess/commit path.
3. FP8 target KV + BF16 draft KV remains the standard production regime.
4. Sampled target behavior matches baseline quality closely enough to be a real
   serving option.
