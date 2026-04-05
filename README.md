# DFlash on Top of SGLang v0.5.9

This branch is the clean design and implementation plan for bringing DFlash to the
official SGLang `v0.5.9` baseline, which is also `radna0/main`.

The goal is narrow:

- keep the official `v0.5.9` target-serving behavior intact
- add only the minimum DFlash serving changes
- avoid target-side drift from `showtime` / `dflash-pagesize-fix`
- prove correctness and speedup on a fixed reference problem set

This document replaces the old benchmark-ledger style README with a first-principles
design document.

## Scope

This clean branch is for:

- DFlash linear speculative decoding only
- GPT-OSS-120B serving
- FA3 target attention
- `triton_kernels` MoE backend
- BF16 bring-up first
- FP8_E4M3 target KV + BF16 draft KV second
- single-request and batched serving
- fixed-width block verification

This clean branch is not for:

- overlap/spec-v2
- SSD / eager-next proposal cache
- FailFast / DAWN-style adaptive verify caps
- DFLASH_TREE
- unrelated FP8 / MXFP4 target-model refactors
- general observability migrations
- unrelated scheduler/runtime/quantization rewrites

## Baseline: How Official v0.5.9 Speculative Decoding Works

Official `v0.5.9` already defines the serving contract we must preserve.

Supported speculative families in baseline:

- EAGLE
- EAGLE3
- STANDALONE
- NGRAM

Important baseline properties:

1. The scheduler and model runner are target-centric.
   The target worker owns the canonical request state, target KV ownership, decode loop,
   and target-side CUDA graph capture.

2. Draft workers share request indexing state, not target KV tensors.
   Official EAGLE / STANDALONE workers share:

   - `req_to_token_pool`
   - token-slot allocator semantics

   but they keep their own draft-model KV cache and draft-model forward path.

3. EAGLE / STANDALONE use separate draft CUDA-graph runners.
   Their draft proposal path is not captured through target TARGET_VERIFY graphs.

4. TARGET_VERIFY is already a first-class target-side concept.
   This is the critical insertion point for DFlash.

The main invariant to preserve is:

- DFlash may extend speculative decoding, but it must not rewrite the target serving
  model into a different runtime architecture.

## Upstream DFlash Intent

The upstream DFlash PR lineage (`#16818`, later `#22077`) makes the intended design
clear:

- DFlash uses a fixed-size block proposal
- DFlash verify reuses the target `ForwardMode.TARGET_VERIFY` path (and therefore the target-side capture machinery)
- EAGLE / STANDALONE / NGRAM draft workers should keep their own draft graph paths

The key upstream comment is:

> EAGLE/standalone/ngram draft workers use separate cuda-graph runners; do not
> capture TARGET_VERIFY graphs here. DFLASH draft uses a fixed-size block and
> reuses the TARGET_VERIFY mode for performance.

That is the design center of this branch.

## Production Contract We Must Satisfy

The current production launcher in `cuda-rl/kaggle/showtime.py` requires the serving
stack to support this contract:

- target model: GPT-OSS-120B
- target attention backend: `fa3`
- target MoE backend: `triton_kernels`
- target KV dtype: `fp8_e4m3`
- draft KV dtype: `bfloat16`
- DFlash enabled
- `cuda_graph_mode=full+piecewise`
- no overlap schedule
- large context serving
- batched serving

However, this README intentionally separates:

- the clean DFlash serving design
- the later production tuning choices

The clean design therefore starts with BF16 correctness and then extends to mixed
precision.

## Training-Time Facts vs Inference-Time Facts

The draft checkpoints were trained in the `cuda-dflash` training repo with:

- Flex Attention
- block size `16`

That does not force inference to use the same attention backend or the same block size.

Training-time facts:

- training used Flex Attention
- training default block size was `16`

Inference-time design for this branch:

- serving uses SGLang, not the training runtime
- target attention is pinned to `fa3`
- target MoE is pinned to `triton_kernels`
- inference block size is a runtime knob, not a training invariant

Inference block sizes that should be explicitly benchmarked:

- `4`
- `8`
- `12`
- `16`

The clean correctness bring-up should start at training-match block size `16`, then
benchmark smaller block sizes for throughput.

## Target Layer Capture Semantics

This is a correctness-critical detail for GPT-OSS DFlash and must not be guessed.

The draft checkpoints store semantic target layer ids such as:

- `target_layer_ids = [1, 9, 17, 25, 33]`

These ids mean:

- capture the hidden states **after** target layers `1, 9, 17, 25, 33`

This is how the training / export stack interprets them.

Why this is true:

- the draft checkpoint config preserves the raw semantic ids
- the SpecForge target path treats `hidden_states[0]` as embeddings and
  `hidden_states[i + 1]` as the output of target layer `i`
- the draft-side context-feature extraction also reads `layer_id + 1`

So the training/export meaning is unequivocally:

- layer id `k` means the output tensor **after layer `k`**

### SGLang Capture Mapping

SGLang GPT-OSS model implementations capture aux hidden states at a different
boundary:

- inside the layer loop, SGLang appends `hidden_states + residual`
- then it executes layer `i`

That means SGLang capture index `i` corresponds to the output of layer `i - 1`.

Therefore, to capture the semantic layer outputs:

- after layer `1`
- after layer `9`
- after layer `17`
- after layer `25`
- after layer `33`

SGLang must internally request:

- `[2, 10, 18, 26, 34]`

### Clean Rule

For GPT-OSS DFlash in SGLang:

- checkpoint / config layer ids stay as `[1, 9, 17, 25, 33]`
- `set_dflash_layers_to_capture(layer_ids)` must map them to
  `self.model.layers_to_capture = [val + 1 for val in layer_ids]`

This is not an optional compatibility shim. It is the correct semantic mapping
between:

- SpecForge training/export meaning
- SGLang serving-time capture meaning

Using the raw ids without `+1` would capture the wrong hidden states and silently
break DFlash target-context features.

## Clean DFlash Architecture

### 1. Draft Model

The draft model is a separate lightweight model that proposes a fixed-width block.

For each live request, the draft model produces:

- `draft_token_num` proposed tokens
- optional proposal-side metadata needed by the verify rule

The clean baseline assumes a linear block, not a tree.

### 2. Target Verify

The target model verifies the proposed block with `ForwardMode.TARGET_VERIFY`.

For a block size `B`:

- the target runs on the proposed block
- it produces one target decision per step
- acceptance is computed on the target side
- committed tokens are appended to target request state
- non-accepted speculative positions are discarded

This branch intentionally uses the target as the source of truth for acceptance.

### 3. Commit Semantics

For each request in a batch:

- accept a prefix of the proposed block
- stop at the first rejected step
- commit the accepted draft tokens
- commit the target-chosen continuation token ("bonus token")
- update request state and KV ownership

This is the linear DFlash contract.

### 4. CUDA Graph Design

Official `v0.5.9` already captures target decode / verify graphs.

DFlash should use that structure as follows:

- target verify graphs are reused for fixed-width DFlash verify
- DFlash does not require a separate EAGLE-style draft graph runner in the baseline
- full / piecewise graph policy remains a target runtime choice, not a DFlash-specific
  architecture change

The clean rule is:

- reuse TARGET_VERIFY graphs for DFlash verify
- do not introduce new target-side graph modes unless absolutely required

## Memory Design

### Shared vs Separate Pools

There are two different questions that must not be conflated:

1. shared request/token allocator state
2. shared KV cache tensors

Clean design decision:

- share request indexing / allocator ownership patterns with the target worker
- do not share target and draft KV tensors

Why:

- this matches official EAGLE / STANDALONE worker structure
- it avoids dtype coupling between target and draft KV
- it keeps target memory ownership simple
- it makes mixed precision practical

### Recommended Clean Baseline

Use:

- shared request/token allocator semantics
- separate target KV pool
- separate draft KV pool

This is the safest design for both BF16 and FP8 target serving.

### Why Not a Shared KV Pool

A shared KV pool would couple:

- target KV dtype
- draft KV dtype
- page layout
- allocator capacity
- commit / free semantics

That is exactly the wrong direction for the mixed-precision target FP8 + draft BF16
design we want.

## Precision Plan

## Phase A: BF16 First

The first proof point is:

- target attention backend `fa3`
- target KV `bfloat16`
- draft KV `bfloat16`

Why start here:

- easiest correctness debugging
- simplest page mapping behavior
- simplest acceptance debugging
- establishes DFlash semantics before quantized target KV enters the picture

BF16-first questions to answer:

- does the draft proposal shape match verify expectations?
- are accept-length and commit semantics correct?
- do page-size and commit/free rules behave correctly?
- do single-request and batch modes both work?

## Phase B: FP8_E4M3 Target + BF16 Draft KV

After BF16 correctness is proven, extend to:

- target KV `fp8_e4m3`
- draft KV `bfloat16`

Clean design rule:

- target-side KV storage may be FP8
- draft-side KV storage stays BF16
- target verification still uses the target runtime and target kernels
- draft proposal remains isolated from target KV dtype

This is the primary mixed-precision DFlash serving target for GPT-OSS-120B.

## GPT-OSS-120B Specific Assumptions

This branch assumes:

- target model is GPT-OSS-120B
- target weights are MXFP4
- target serving uses SGLang's GPT-OSS path
- target attention backend is FA3
- target MoE backend is `triton_kernels`

Clean-branch consequence:

- do not port unrelated GPT-OSS target refactors unless they are strictly required to
  make DFlash serve on top of `v0.5.9`
- do not drag in unrelated MXFP4 / FP8 target-side rewrites from `showtime`

## Page Size Strategy

Page size is a separate dimension from DFlash itself.

Clean bring-up order:

1. `page_size = 1`
2. prove BF16 DFlash correctness
3. prove FP8 target + BF16 draft KV correctness
4. only then extend to paged verify (`page_size > 1`)

Why:

- paged verify introduces temporary uncommitted verify-token mapping
- commit/free behavior becomes more complex
- incorrect page mapping can silently destroy acceptance

The `dflash-pagesize-fix` branch is useful as reference, but the clean README treats
page-size support as a second-stage extension, not part of the first correctness proof.

## Linear Verify vs Tree Verify

This clean branch keeps:

- DFlash linear verify

This clean branch excludes:

- DFLASH_TREE

Reason:

- tree verify is a separate algorithmic expansion
- it changes mask construction, graph reuse, and commit semantics
- it is not required to prove DFlash works on top of `v0.5.9`

If tree verify returns later, it should be added after linear DFlash is already stable.

## Sampling Modes

The design must preserve three runtime modes.

### 1. Draft Greedy + Target Greedy

- draft proposes argmax block
- target verifies with argmax decisions
- accept while tokens match

This is the simplest correctness mode and the first one to benchmark.

### 2. Draft Greedy + Target Sampled

- draft proposes greedily
- target samples from its own distribution at each verify step
- accept only exact matches against sampled target tokens

This preserves the target distribution while keeping the draft path simple.

### 3. Draft Sampled + Target Sampled

- draft proposes sampled candidates
- target still owns the true sampling decision
- accept only exact matches against sampled target tokens

This also preserves the target distribution, but proposal quality and acceptance will
depend more heavily on draft calibration.

Clean design rule:

- the target remains the source of truth in all three modes
- DFlash should not change the target output distribution

## Single Request vs Batch

The clean branch must support both:

- single-request decode
- batched decode / generation

The algorithm is the same in both cases:

- each request has its own proposed block
- verify is vectorized across the batch
- acceptance and commit remain per-request

What changes under batching:

- capture widths and batch sizes for TARGET_VERIFY graphs
- allocator pressure
- page mapping pressure
- average acceptance versus graph amortization tradeoff

The design target is not just "works for one request"; it must work under the same
batched serving regime that production uses.

## Minimal File Set for a Clean DFlash Port

The minimal expected DFlash-serving change set on top of `v0.5.9` is:

- `python/sglang/srt/models/dflash.py`
- `python/sglang/srt/speculative/spec_info.py`
- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_utils.py`
- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- `python/sglang/srt/managers/scheduler.py`

Possibly required, but only if strictly necessary:

- `python/sglang/srt/layers/sampler.py`
- `python/sglang/srt/models/registry.py`
- `python/sglang/srt/models/utils.py`
- `python/sglang/srt/compilation/cuda_piecewise_backend.py`

These "possibly required" files must be justified individually. They are not blanket
permission to import unrelated branch history.

## Changes Explicitly Excluded from the Clean Port

Do not pull these in unless a specific DFlash-serving dependency proves unavoidable:

- `gpt_oss.py` target-model drift unrelated to DFlash
- `mxfp4.py` changes
- generic FP8 quantization refactors
- Flex Attention serving backends
- observability package migration
- session-aware cache additions
- overlap / spec-v2-only pathways
- SSD / eager-next proposal reuse
- failfast / adaptive verify heuristics
- tree worker implementation
- unrelated model-registry filtering

If one of these turns out to be truly required, it must be documented as a dependency,
not smuggled in as part of a large branch diff.

## Validation Matrix

The branch is not considered "working" until it is proven on the fixed reference set:

- hardest: `86e8e5`
- harder: `dd7f5e`
- decently hard: `a295e9`
- medium: `9c1c5f`
- easiest: `92ba6a`

Each problem must be run in:

### BF16 correctness matrix

- target BF16, draft BF16
- `page_size = 1`
- single-request
- batched
- block sizes `16`, `8`, `4`

### Mixed-precision serving matrix

- target KV `fp8_e4m3`, draft KV `bfloat16`
- FA3 target attention
- `triton_kernels` MoE
- single-request
- batched
- production-style prompt path

## Metrics to Report

Every benchmark table should report:

- correctness / final answer
- total throughput (tok/s)
- wall time
- accepted token count
- mean acceptance length
- verify count
- average accepted tokens per verify
- baseline target-only comparison

Minimum proof requirement:

- show DFlash serving works
- show DFlash acceptance is non-trivial
- show DFlash throughput beats the target-only baseline

## Acceptance Criteria for This Clean Branch

This clean DFlash branch is done only when all of the following are true:

1. The implementation is based on official `v0.5.9` target behavior.
2. DFlash works in BF16 without extra target-side drift.
3. DFlash works in mixed precision with target KV `fp8_e4m3` and draft KV `bfloat16`.
4. The five reference problems above run successfully.
5. DFlash shows measurable speedup over target-only baseline.
6. The branch does not require overlap, SSD, FailFast, tree verify, or unrelated
   target-model rewrites.

## Recommended Implementation Order

1. Start from `v0.5.9`.
2. Port only the core DFlash-serving files.
3. Remove `DFLASH_TREE`.
4. Prove BF16 correctness at `page_size = 1`.
5. Prove batched BF16 correctness.
6. Add mixed precision: target KV `fp8_e4m3`, draft KV `bfloat16`.
7. Benchmark block sizes `16`, `8`, `4`.
8. Extend to paged verify only after the above is stable.
9. Re-run the five reference problems and compare against the old benchmark ledger.

## Status of This README

This README is a design and implementation contract.

It is intentionally not:

- a benchmark diary
- a list of ad hoc bring-up patches
- a ledger of every experiment from `showtime`

Those experiments remain useful references, but this document defines the clean branch
we actually want to maintain and eventually PR.
