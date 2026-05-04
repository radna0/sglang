# DFlash Production Verify Roadmap

This note is the code-anchored engineering plan for production DFlash on this
branch.

It is intentionally limited to production-relevant paths:

- `target_only` verify
- greedy target verify
- sampled target verify
- FP8 target KV cache
- BF16 draft KV cache
- overlap/spec-v2 scheduling

`pq` verify is explicitly out of scope for the production plan on this branch.

## Scope And Non-Goals

This roadmap is about shipping the production path that actually matters:

1. baseline-faithful target behavior
2. DFlash target-only speculative decode
3. FP8 target KV + BF16 draft KV
4. overlap/spec-v2 support for DFlash

It is **not** about:

- reviving `pq`
- treating `pq` as a production candidate
- claiming fused draft KV append means target verify is already fused
- pretending DFlash can reuse Eagle tree abstractions directly

## Current Code-Anchored State

### 0. Current benchmark checkpoint before more overlap work

The branch-level benchmark state now supports a narrow engineering conclusion:

- no-DFlash sampled baseline is still the strongest confirmed quality regime
- mixed-pool `explore32 -> route8` is not yet producing a true green-zone route set
- overlap-v2 is helping on short sampled A/B, but only modestly so far

That means the right immediate engineering target is not more routing work. It is:

1. overlap-v2 completion
2. fused / CUDA-graph / mixed-precision verify on the DFlash family path
3. linear + tree parity under that contract

The route study is therefore treated as documented and paused for now.

### 1. DFlash overlap/spec-v2 was hard-disabled, but the local worktree now has the gate wiring

Code:

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/speculative/spec_info.py`
- `python/sglang/srt/managers/scheduler.py`

Old behavior:

- `server_args.py` forces `disable_overlap_schedule = True` for DFlash
- `spec_info.py` raises if overlap is enabled for DFlash

Local worktree state now:

- `SpeculativeAlgorithm.supports_spec_v2()` includes DFlash
- `create_worker(...)` selects `DFlashWorkerV2` when overlap is enabled
- `server_args.py` no longer force-disables overlap for DFlash
- `scheduler.init_disaggregation()` no longer assumes Eagle-only nested draft runners for the DFlash overlap path
- `DFlashWorkerV2` now owns an explicit:
  - overlap prefill entrypoint
  - overlap decode/verify entrypoint
  instead of relying on the base worker's mixed v1/v2 public method

What still must be proven by benchmark before commit:

- live overlap-v2 scheduler replay works end to end on the actual DFlash serving path
- the overlap path is stable under CUDA graph replay
- the overlap path improves throughput enough to justify the gate flip
- DFLASH_TREE overlap-v2 now has internal worker/future/scheduler scaffolding in the
  local worktree, and `server_args.py` now exposes it behind the explicit
  experimental env gate:
  - `SGLANG_ENABLE_DFLASH_TREE_OVERLAP_EXPERIMENTAL=1`
  It is still not production-ready until benchmarked and validated under CUDA graph replay.

### 1a. DFLASH_TREE is no longer structurally blocked in the local worktree

Code:

- `python/sglang/srt/speculative/spec_info.py`
- `python/sglang/srt/speculative/dflash_tree_worker.py`
- `python/sglang/srt/speculative/dflash_tree_worker_v2.py`
- `python/sglang/srt/managers/overlap_utils.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`

Current local worktree state:

- `DFLASH_TREE` is a real `SpeculativeAlgorithm` enum member
- `supports_spec_v2()` now includes the DFlash family
- `create_worker(...)` can now select:
  - `DFlashTreeWorker` when overlap is disabled
  - `DFlashTreeWorkerV2` when overlap is enabled
- `FutureMap`, scheduler overlap state prep, and overlap output processing now key on
  the DFlash family rather than only `DFLASH`
- `DFlashTreeWorker` now has an internal overlap-v2-shaped execution path and emits:
  - `next_draft_input`
  - `verify_done`
  - `dflash_overlap_preprocessed`

What remains before we can claim tree overlap is really working:

- live scheduler replay on the tree path
- CUDA-graph validation
- real throughput validation vs tree non-overlap
- the new indexed cache plan no longer hard-requires `page_size == 1`
  - tree verify now computes page-aware evicted-page frees through the shared
    indexed cache-plan helpers instead of bailing out early

### 2. The real production verify path is `target_only`

Code:

- `python/sglang/srt/speculative/dflash_info.py`

Confirmed behavior:

- `verify_mode` defaults to `target_only`
- recent benchmark JSONs report `spec_dflash_verify_mode_last = "target_only"`

This is the path to optimize first for both greedy and sampled target decode.

### 3. Greedy target-only verify already exists and is exact

Code:

- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_utils.py`

Current path:

1. target forward produces `logits_output.next_token_logits`
2. greedy path computes `target_predict = argmax(logits)`
3. it calls `compute_dflash_accept_len_and_bonus(...)`
4. it then packs `[target_predict, accept_len]`
5. it copies that packed tensor to CPU
6. it performs per-request Python commit logic

Important implication:

- greedy verify math is already on device
- commit/pack/free/output mutation is still partly CPU-bound

### 4. Sampled target-only verify now has a production helper path, with fallback

Code:

- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_utils.py`

Current production sampled path in `dflash_info.py`:

1. prefer `compute_dflash_sampling_accept_len_and_bonus(...)` when available
2. preserve a strict inline fallback if the helper is unavailable or fails
3. preserve target-side:
   - `temperature`
   - `top_k`
   - `top_p`
   - `min_p`
4. preserve logical-cap behavior through `max_steps_per_req`
5. return the committed target-only prefix for CPU-side request mutation

Important detail:

- `dflash_utils.py` already contains
  `compute_dflash_sampling_accept_len_and_bonus(...)`
- that path uses `tree_speculative_sampling_target_only`
- it now supports:
  - logical cap
  - committed-prefix reconstruction
  - `min_p`
- `dflash_info.py` now records whether the helper path was used through:
  - `targetonly_sampled_helper`

So sampled target-only is no longer purely the old inline path. It now has a
production helper fast path, but still does not have a fully fused on-device
commit/update stage.

Local worktree hardening that is not committed yet:

- the sampled helper now clamps the kernel-reported accepted count into the legal
  DFlash range before compact target-only packing
- this is intended to stop the live sampled helper crash that previously surfaced
  as a device-side assert during `pack_dflash_target_only_commits(...)`
- the sampled helper no longer reconstructs the committed prefix from the wider
  kernel-side accept-index table
  - it now reconstructs the target-only committed proposal directly as:
    - accepted candidate prefix
    - plus the target-sampled bonus token
  - this is the correct target-only contract and removes a class of invalid-index
    failures that were not semantically necessary in the first place

### 4a. DFLASH_TREE now uses compact packed commits like the linear path

Code:

- `python/sglang/srt/speculative/dflash_tree_worker.py`
- `python/sglang/srt/speculative/dflash_utils.py`

Local worktree state now:

- tree verify no longer reconstructs proposals per request in a Python loop
- tree verify now packs accepted verify outputs into:
  - compact flat committed tokens
  - commit lengths
  - commit offsets
  - default `new_verified_id`
- CPU-side commit on the tree path now uses the same batched helper shape as the
  linear target-only path
- final accepted flat verify indices are now derived from:
  - original `accept_index`
  - final `commit_lens`
  instead of mutating `accept_index` row-by-row on CPU

Why this matters:

- this removes one of the biggest remaining structural differences between the
  linear path and the tree path
- it gives tree verify the same no-truncation fast-path metadata seam the linear
  path already had
- it is the correct staging point for later deeper fusion work

### 5. Accept/commit bookkeeping is still CPU-shaped, but the D2H boundary is smaller

Code:

- `python/sglang/srt/speculative/dflash_info.py`

Current path after accept lengths are computed:

1. build compact committed target-only prefixes on device
2. transfer compact committed prefixes plus per-request commit lengths to CPU
3. iterate requests in Python
4. append tokens to `req.output_ids`
5. update finish state / stop tokens / grammar
6. update:
   - `req.spec_verify_ct`
   - `req.spec_accepted_tokens`
   - acceptance histogram
7. free uncommitted KV slots/pages

This means the production hot path is currently split into:

- GPU verify math
- CPU commit / output / bookkeeping

What changed already:

- target-only no longer has to copy the full `draft_token_num` token row for each
  request
- it now copies only the committed prefix plus lengths
- the common plain-request `target_only` path now batches CPU finish-policy handling
  for:
  - `max_new_tokens`
  - EOS / stop-token detection
  - invalid-vocab handling
  while preserving strict fallback for:
  - grammar
  - stop strings
  - stop regex
- shared-pool `target_only` verify no longer eagerly materializes committed target
  hidden by default
  - `verify()` now returns structured commit metadata
  - the worker only gathers committed hidden on the staged-append fallback path
  - the direct shared-pool append path uses:
    - target verify hidden states
    - compact flat accepted indices
    - flat accepted positions from `verify_input.positions`
    - compact cache plan
    without pre-staging an accepted-hidden tensor on the verify side

What remains:

- CPU request mutation still exists
- finish/grammar/stop handling still lives in the Python loop
- KV free/compaction still happens after CPU-side appended-length decisions

What was improved on this branch:

- the CPU-side request mutation logic is now centralized in
  `commit_dflash_proposed_tokens_to_req(...)`
- both `dflash_info.py` and `dflash_tree_worker.py` use that shared helper
- target-only commit metadata now has a no-truncation fast path:
  - if the CPU commit loop did not shorten the proposed prefix for a request,
    we reuse the device-side `commit_lens` and `default_new_verified_id`
  - only requests that actually diverge from the device defaults are patched back
    from CPU outcomes
- this does not yet remove the CPU loop, but it gives us one stable seam for
  later:
  - device-generated commit metadata
  - device-generated keep/evict decisions
  - DFlash-specific spec-v2 payload work

So the CPU-shaped boundary is smaller now, but it is still the main fusion
target.

### 5a. What this means concretely for "fused verify"

The current fused-verify story on this branch is:

1. target verify math runs on device
2. committed target-only prefixes are compacted on device
3. the common no-truncation case now reuses device-side commit metadata
4. shared-pool regimes can append verified hidden directly into the draft KV path
5. overlap replay no longer re-copies / re-resolves accepted token payloads on CPU
6. shared-pool verify no longer stages committed hidden unless the direct append
   path falls back

What is still **not** fused:

- the Python request mutation loop itself
- finish / stop-string / grammar decisions
- any mixed FP8-target/BF16-draft single verify kernel

So the next production boundary is still host-side request semantics, not draft KV math.

### 5b. The next fused-verify seam is now explicit

After the current local changes, the remaining mixed-precision production seam is:

1. target verify produces target hidden on device
2. commit metadata and keep/evict plan are produced on device
3. the fused shared-pool lane now owns:
   - accepted-row selection
   - target->draft projection
   - target hidden norm
   - BF16 draft KV materialization
4. the remaining separation is that this is still a DFlash helper path, not yet a
   single dedicated selected-hidden append kernel

So the next undeniably best technical target is no longer vague. It is one of:

- a deeper DFlash-owned fused selected-hidden -> draft-KV append kernel
  - input:
    - target hidden buffer
    - compact accepted indices
    - accepted positions from the flat verify layout
    - compact cache locations
  - output:
    - BF16 draft KV written directly
- or a combined target+verify+draft CUDA graph on the DFlash overlap-v2 path that
  makes that boundary stable enough that the remaining gather cost is fully hidden

The current branch should pursue both in that order:

1. keep the shared-pool no-staging fast path as the baseline seam
2. replace the current DFlash-owned fused-helper path with a dedicated selected-hidden append kernel
3. then fold target verify + accepted-hidden append + next draft proposal into the
   DFlash-owned overlap-v2 graph path

Current local seam:

- `DFlashDraftModel.project_target_hidden_selected(...)`
- `DFlashWorker._project_and_write_verified_hidden_selected_to_draft_kv(...)`
- `FusedKVMaterializeHelper.materialize_from_target_hidden_selected(...)`

Parity note:

- the same selected-hidden fused-helper ownership now exists on:
  - `DFlashWorker`
  - `DFlashTreeWorker`

So the next kernel/graph step no longer needs separate main-vs-tree boundary work first.

This is now the one explicit replacement point for the deeper mixed-precision
fused append path.

### 6. Draft proposal generation already supports physical-vs-logical separation

Code:

- `python/sglang/srt/speculative/dflash_worker.py`

Current path in `_build_draft_proposal(...)`:

- `self.block_size` is the physical max block size
- `max_steps_per_req` is the logical cap
- `draft_token_num = effective_step_count + 1`

This is already the correct abstraction for:

- fixed physical max block
- adaptive logical effective length

So FailFast/EAFT-style control belongs on `max_steps_per_req`, not on ad hoc
physical-width recapture first.

### 7. What is fused today is draft KV materialization/append only

Code:

- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`

Current fused append path:

1. target hidden is projected into draft hidden
2. draft K/V projection runs in the helper
3. RMSNorm runs
4. RoPE runs
5. KV is written into the draft KV pool

Important constraint:

- this is draft-side fused append
- it does **not** fuse the target verification pass
- it does **not** fuse target FP8 KV reads with draft BF16 KV writes

### 8. There is also DFlash async shadow overlap, but it is not spec-v2

Code:

- `python/sglang/srt/speculative/dflash_worker.py`

Current path:

- `_prepare_ssd_fanout_branches_async(...)`
- `_ssd_overlap_executor`
- `_ssd_shadow_req_to_token_pool`

What it does:

- overlaps shadow draft fanout work for SSD-style branch preparation

What it does **not** do:

- integrate with scheduler overlap
- use `FutureMap`
- carry DFlash verify results as a spec-v2 payload
- overlap target verify batches with future decode batches

So this async shadow path is useful, but it is not a substitute for real
DFlash spec-v2 support.

### 8a. DFlash already has partial SSD scaffolding, but it is not yet the primary architecture

Code:

- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/managers/utils.py`

Current branch status:

- there is already a DFlash-specific SSD control surface behind env flags:
  - `SGLANG_DFLASH_SSD_ENABLE`
  - `SGLANG_DFLASH_SSD_PREPARE_NEXT`
  - `SGLANG_DFLASH_SSD_FANOUT`
  - `SGLANG_DFLASH_SSD_BRANCH_MODE`
  - `SGLANG_DFLASH_SSD_ASYNC_OVERLAP`
  - difficulty / miss-streak / alt-mass gates
- there is already a branch-candidate collection path:
  - `_collect_ssd_branch_candidates(...)`
- there is already an eager-next SSD cache path:
  - `_prepare_ssd_fanout_branches(...)`
- there is already an async SSD shadow-overlap path:
  - `_schedule_ssd_fanout_overlap(...)`
  - shadow req-to-token pool
  - shadow KV scratch
  - overlap executor / future
- request metrics already include SSD counters:
  - `spec_ssd_hit_ct`
  - `spec_ssd_prepare_ct`
  - `spec_ssd_prepare_failure_ct`
  - `spec_ssd_cache_pending`
  - `spec_ssd_overlap_launch_ct`
  - `spec_ssd_overlap_wait_ct`
- the current cached-proposal lookup is still keyed narrowly by:
  - `(expected_seq_len, expected_verified_id)`
  rather than a richer explicit verification-outcome object

What this means:

- we do **not** need to invent an SSD mode from scratch
- we already have a prototype of:
  - verification-outcome branch prediction
  - eager-next preparation
  - async overlap preparation
- but it is still attached to the current linear DFlash worker lifecycle
  rather than treated as a first-class overlap-v2 scheduler contract

So the real task is:

1. harden the existing SSD prototype
2. align it with the DFlash-native spec-v2 payload
3. strengthen the cache key / future payload so it represents the verification
   outcome explicitly, not just the current minimal key
4. then measure it against ordinary DFlash overlap, instead of building a
   separate unrelated system

### 8b. SSD is not the same thing as EAGLE tree verify

Reference material inspected:

- `https://arxiv.org/pdf/2603.03251`
- `/workspace/ssd_paper/ssd_2603.03251.txt`
- `/workspace/ssd_repo/README.md`
- `/workspace/ssd_repo/ssd/utils/verify.py`
- `/workspace/ssd_repo/ssd/engine/speculator_async.py`
- `/workspace/ssd_repo/ssd/engine/draft_runner.py`
- `/workspace/ssd_repo/ssd/engine/verifier.py`

Important distinction:

- EAGLE-style tree speculation increases verifier-side structure:
  - multiple branches are verified by the target in one tree-masked verify pass
  - verifier compute grows with the tree
- SSD-style async speculation does **not** add verifier compute:
  - the verifier still verifies one ordinary speculative rollout
  - the draft predicts likely future verification outcomes in parallel on
    separate hardware and caches them

This matters for DFlash:

- the current main DFlash worker is still fundamentally a linear block-speculation
  path
- DFlash tree behavior exists separately in `dflash_tree_worker.py`
- SSD belongs more naturally on top of:
  - DFlash overlap-v2 future payloads
  - outcome-cache / branch-preparation logic
  - draft-side overlap scheduling
- SSD does **not** imply reusing Eagle tree masks or Eagle tree payload semantics

So the right mental model is:

- EAGLE tree: more verifier structure
- SSD: more draft-side future-outcome preparation
- DFlash can and should eventually support both ideas, but they solve different
  bottlenecks

### 9. FP8 target KV and BF16 draft KV are separate and correctly configurable

Code:

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/speculative/dflash_worker.py`

## Current Dirty-Tree Benchmark State

Short sampled A/B, `92ba6a`, `ctx=65536`, `decode=2048`, `concurrency=4`,
`block_size=8`, target `fp8_e4m3`, draft `bfloat16`, CUDA graph on:

- current non-overlap:
  - `/workspace/dflash_overlap_compare_sampled_short_20260329/non_overlap_fused_current.json`
  - `561.643 tok/s`
  - `accept=2.923`
- current overlap:
  - `/workspace/dflash_overlap_compare_sampled_short_20260329/overlap_fused_current.json`
  - `581.786 tok/s`
  - `accept=3.042`
- current paired speedup:
  - about `1.036x`

Earlier hard sampled A/B, `86e8e5`, same short regime:

- current non-overlap:
  - `/workspace/dflash_overlap_compare_sampled_short_20260329/hard_non_overlap_fused_current.json`
  - `472.817 tok/s`
  - `accept=2.364`
- current overlap:
  - `/workspace/dflash_overlap_compare_sampled_short_20260329/hard_overlap_fused_current.json`
  - `473.963 tok/s`
  - `accept=2.448`
- current paired speedup:
  - about `1.002x`

There is still an older hard pair on disk with a larger overlap advantage:

- `/workspace/dflash_overlap_compare_sampled_short_20260329/hard_non_overlap_fused.json`
- `/workspace/dflash_overlap_compare_sampled_short_20260329/hard_overlap_fused.json`
- about `1.052x`

So the stable conclusion is:

- overlap-v2 + current fused-verify work is no longer losing on the hard case
- but the hard-case gain is still variance-sensitive and much smaller than on the easy case

This is enough to say the overlap-v2 + current fused-verify path is no longer
just an architectural bring-up. It is now a real throughput win on both the
easy and the harder sampled short workloads already measured on this branch.

Confirmed behavior:

- target worker can use `kv_cache_dtype = fp8_e4m3`
- draft worker can independently use `speculative_draft_kv_cache_dtype = bf16`
- `configure_kv_cache_dtype()` already branches differently for draft vs target

Important implication:

- the mixed-precision boundary is already a real production regime
- but it is still separated by worker/cache boundaries

### 10. CUDA graph capture already has a DFlash spec input, but not an overlap payload

Code:

- `python/sglang/srt/model_executor/cuda_graph_runner.py`

Current behavior:

- `CudaGraphRunner.get_spec_info()` creates a `DFlashVerifyInput`
- it captures enough metadata for non-overlap DFlash verify replay
- it does **not** carry Eagle-style overlap fields such as:
  - `future_indices`
  - `new_seq_lens`
  - `verify_done`

So DFlash already has a graph replay input object, but not a spec-v2 overlap
contract.

### 11. DFlash now has a native post-append future-state contract, but it is inert

Code:

- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/speculative/dflash_tree_worker.py`
- `python/sglang/srt/managers/overlap_utils.py`

Current branch status:

- `DFlashDraftInput` now carries DFlash-native future fields:
  - `future_indices`
  - `new_seq_lens`
  - `verify_done`
- DFlash KV append now makes `new_seq_lens` explicit after the append step
- `FutureMap` now has a DFlash-specific storage/resolve branch
- `FutureMap` now also preserves `verify_done` across DFlash future store/resolve,
  so the replayed draft state keeps the same verify-completion synchronization signal
- DFlash workers now emit `next_draft_input` as post-append draft state on both:
  - prefill completion
  - verify completion
- `ScheduleBatch.prepare_for_decode()` can now consume DFlash draft state through
  `DFlashDraftInput.prepare_for_decode(...)`
- scheduler overlap handoff now has an explicit DFlash/spec-v2 state-preparation
  helper instead of leaving DFlash-specific future-state mutation inline in `run_batch`
- `CudaGraphRunner.get_spec_info()` now constructs DFlash verify replay inputs via
  `DFlashVerifyInput.create_idle_input(...)`
- that branch stores only the stable post-append draft state:
  - `verified_id`
  - `draft_seq_lens`
  - `new_seq_lens`

Important design choice:

- the DFlash future payload does **not** carry transient `target_hidden`
  buffers
- it also rejects nonzero `ctx_lens` on store
- this is intentional: the overlap payload should represent the state *after*
  target hidden has already been appended into the draft KV cache

So the branch now has the correct DFlash-native payload seam, and the local
worktree has gone further than the original roadmap state:

- DFlash now reports `supports_spec_v2() == True`
- overlap scheduling is no longer force-disabled for DFlash on this branch
- the live overlap path now runs end to end with:
  - `FutureMap` DFlash payload storage/resolve
  - `DFlashWorkerV2` selection
  - DFlash-specific overlap token slicing in the scheduler output processor

What still remains true:

- the overlap path is still treated as experimental until it shows a repeatable
  throughput win, not just correctness/stability
- most of the remaining overhead is now at the host-side verify/commit boundary,
  not in missing payload plumbing

### 12. Current overlap/fused-verify benchmark state

Reference workload so far:

- sampled target decode
- target KV: `fp8_e4m3`
- draft KV: `bfloat16`
- `block_size=8`
- `page_size=1`
- shared pools
- `question_id=92ba6a`
- `num_prompts=4`
- `concurrency=4`
- `decode_len=2048`

Observed results on this branch:

- old non-overlap baseline:
  - `604.432 tok/s`
- old overlap before the new overlap/fused work:
  - `535.158 tok/s`
- overlap after direct shared-pool verify->draft append:
  - `601.787 tok/s`
- latest paired rerun after skipping redundant DFlash overlap token D2H:
  - non-overlap: `557.643 tok/s`
  - overlap: `563.073 tok/s`

Interpretation:

- overlap-v2 is now stable on the sampled DFlash short benchmark
- the overlap path is no longer clearly slower
- the direct shared-pool verify->draft append helped overlap much more than it
  helped non-overlap
- the latest DFlash-specific overlap CPU/D2H cut moved the paired rerun to a
  small overlap win (`~1.01x`)

This is enough to say the overlap path is real, but not enough yet to claim a
broad production win. The next validation step is a harder sampled workload, not
another easy-only claim.

## What The Next Production Design Must Look Like

## A. Fused target-only verify/postprocess, not “one giant kernel”

The right production breakdown is:

1. target verify forward stays where it is
   - target model reads FP8 target KV
   - target model emits verify logits and hidden states

2. on-device target-only postprocess becomes the new fusion target
   - greedy:
     - argmax target tokens
     - exact-match accept mask
     - accept length
     - bonus token
     - compact committed token span
   - sampled:
     - temperature/top-k/top-p/min-p
     - target-only sampling
     - exact-match accept mask
     - accept length
     - bonus token
     - compact committed token span

3. draft-side append remains a separate mixed-precision handoff
   - accepted target hidden is projected into draft space
   - existing fused BF16 draft KV append path is reused

So the practical fusion target is:

- fuse target-only verify postprocess and compact commit metadata on device
- then hand accepted hidden states to the existing BF16 append path

It is **not**:

- one monolithic kernel that simultaneously fuses target FP8 attention and draft
  BF16 KV writes

What is already landed locally:

- compact target-only commit packing on device
- compact cache-plan derivation for free/clear/keep
- direct shared-pool verify hidden -> draft append
- batched plain-request `target_only` CPU commit path for the common no-grammar,
  no-string-stop regime

So the remaining fused-verify gap is narrower now:

- reduce or batch the remaining host-side request/output mutation that still has
  to stay on CPU
- reduce overlap-specific D2H / CPU rematerialization for the cases that still
  diverge from device defaults
- only after that consider deeper kernel fusion

## B. The mixed-precision handoff should stay explicit

The production mixed-precision regime is:

- target verify forward on FP8 target KV
- accepted hidden states in target model dtype
- projection into draft hidden
- fused BF16 draft KV append

This boundary is already natural in the code:

- target runner owns target KV
- draft worker owns draft KV
- `project_target_hidden(...)` is the handoff boundary

## FluentLLM Takeaways

Reference repo inspected:

- `/workspace/SGLang-FluentLLM`
- commit `a9db4a83d2522e1bd1392cd54054a1965d5461f4`

Useful reference points:

- `python/sglang/srt/speculative/spec_decoding_cuda_graph_runner.py`
  really does capture a single speculative CUDA graph around the target verify +
  draft proposal path for their Eagle/PLD flow
- `python/sglang/srt/speculative/eagle_worker_overlap.py` uses device-resident
  future-token placeholders plus async D2D copies to feed overlap launches
- `python/sglang/srt/managers/scheduler_post_process_mixin.py` still performs a
  host-side per-request decode postprocess loop

What this means for DFlash:

- FluentLLM is a good reference for overlap producer/consumer ownership and
  graph-capture boundaries
- it is **not** evidence that the remaining DFlash host-side request commit
  boundary should be ignored or can be solved just by copying their overlap code
- their fork still keeps a scheduler-side request/output postprocess loop, so
  the remaining DFlash CPU commit/finish work is still a legitimate optimization
  target on this branch

So the current DFlash direction remains correct:

1. keep the DFlash-native overlap-v2 path
2. keep the explicit FP8-target -> BF16-draft handoff
3. batch the common plain-request `target_only` CPU commit path
4. only after that reassess whether any deeper mixed-precision fused-verify
   kernel is still worth the implementation cost

So the roadmap should improve that boundary, not erase it.

## C. DFlash needs its own spec-v2 payload contract

DFlash should not be forced into Eagle tree payload semantics.

The correct DFlash overlap payload is the **post-append** draft state, not the
pre-append verify bundle.

That means the payload should carry at least:

- `future_indices`
- `verified_id`
- `draft_seq_lens`
- `new_seq_lens`
- request/batch routing metadata needed by scheduler replay

And it should **not** carry:

- transient `target_hidden`
- nonzero `ctx_lens`
- Eagle tree `topk_p/topk_index` semantics

The right shape is:

- keep `DFlashVerifyInput` for verify replay
- keep `DFlashDraftInput` as the DFlash-native future payload carrier
- store only post-append state in `FutureMap`
- thread that payload through scheduler and graph replay without reusing Eagle
  tree buffers

## D. The unused sampled helper should be pulled into the production path

`compute_dflash_sampling_accept_len_and_bonus(...)` already exists and wraps
`tree_speculative_sampling_target_only`.

That should be evaluated as the candidate building block for:

- sampled target-only on-device accept/bonus computation
- reducing Python-side probability manipulation in `dflash_info.py`

But before promoting it, we still need:

- `max_steps_per_req` support or equivalent logical-cap support
- parity with current `temperature/top_k/top_p/min_p` semantics
- parity in emitted metrics and debug counters

## E. The current DFlash branch geometry is still split: linear main path, tree side path

Code:

- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/speculative/dflash_tree_worker.py`

Current state:

- `DFlashWorker` is still the primary linear block-speculation worker
- `DFlashTreeWorker` contains the tree / indexed verify behavior
- overlap-v2 and the new fused selected-hidden append work have been landing
  primarily on the main DFlash path first, with parity being added afterward

Implication:

- if we want "EAGLE-style branches" in the strong sense, that is a tree-verify
  question and belongs with the tree path
- if we want "SSD-style many likely future outcomes", that is an async branch
  preparation question and belongs with the main overlap-v2 draft scheduler path

So the best architecture is not to force them into one abstraction too early.
It is:

1. finish the linear DFlash overlap-v2 + fused verify path
2. finish the SSD-style future-outcome preparation path on top of that
3. only then decide how much tree-verify branching should be merged back into
   the same control plane

## Engineering Tasks

The plan below is split into:

- Tasks 1-10: build/design trace
- Tasks 11-18: implementation
- Tasks 19-20: reverse-trace verification

That final phase is deliberate. After implementation, we retrace the path that
got us there and verify each dependency in reverse.

### Build / Design Tasks

1. `[done]` Audit the current roadmap against code in:
   - `dflash_info.py`
   - `dflash_worker.py`
   - `model_runner.py`
   - `server_args.py`
   - `spec_info.py`

2. `[done]` Trace greedy `target_only` verify end-to-end:
   - target logits
   - argmax
   - `compute_dflash_accept_len_and_bonus(...)`
   - CPU pack
   - Python commit loop

3. `[done]` Trace sampled `target_only` verify end-to-end:
   - target-only sampling filters
   - sampled target token generation
   - accept computation
   - CPU pack
   - Python commit loop

4. `[done]` Trace accept/commit bookkeeping and identify the CPU boundary:
   - `packed = ... .cpu()`
   - per-request token append
   - finish/grammar handling
   - KV free logic

5. `[done]` Trace draft proposal generation and confirm physical-vs-logical split:
   - `self.block_size`
   - `max_steps_per_req`
   - `draft_token_num`

6. `[done]` Trace fused draft KV append and sequential fallback:
   - `_append_target_hidden_fused(...)`
   - `_append_target_hidden_sequential(...)`

7. `[done]` Trace target FP8 KV and draft BF16 KV configuration:
   - `configure_kv_cache_dtype()`
   - draft override behavior

8. `[done]` Trace the DFlash overlap-disable path and confirm spec-v2 is off:
   - `disable_overlap_schedule`
   - DFlash rejection in `spec_info.py`

9. `[done]` Trace the existing Eagle overlap contract for reference:
   - `future_indices`
   - `new_seq_lens`
   - `verify_done`
   - scheduler `FutureMap`

10. `[done]` Trace DFlash async shadow overlap and record that it is not spec-v2:
    - SSD shadow executor
    - shadow proposal precompute
    - no scheduler future payload

### Implementation Tasks

11. `[partial]` Add a DFlash-specific v2 payload contract.
    Target files:
    - `dflash_info.py`
    - new DFlash v2 helper/mixin if needed
    - `cuda_graph_runner.py`
    - `scheduler.py`
    Current branch status:
    - `DFlashDraftInput` now carries:
      - `future_indices`
      - `new_seq_lens`
      - `verify_done`
    - DFlash append paths now make `new_seq_lens` explicit
    - `FutureMap` now has a DFlash-native post-append storage/resolve branch
    - scheduler and graph-runner still do not produce or consume this payload in
      a live overlap path

12. `[pending]` Define the minimal device-resident verify result for greedy:
    - accept lengths
    - bonus tokens
    - compact committed tokens
    - new verified ids
    - commit lengths
    Current branch status:
    - target-only compact commit packing now also emits:
      - `commit_offsets`
      - `default_new_verified_id`
    - final `commit_lens` / `new_verified_id` are now materialized through a shared
      helper instead of ad hoc tensor construction in each callsite

13. `[partial]` Implement a fused greedy target-only postprocess path that
    eliminates the current full packed `[target_predict, accept_len].cpu()` flow.
    Current branch status:
    - full-row target-only pack is already replaced by a compact committed-prefix transfer
    - CPU request mutation is still present
    - the remaining CPU request mutation semantics are now shared in one helper:
      `commit_dflash_proposed_tokens_to_req(...)`
    - commit-offset and metadata shaping are now also centralized in helpers, so the
      next remaining extraction is KV free / mapping / hidden assembly rather than
      more target-only list plumbing

14. `[partial]` Promote sampled target-only toward a kernel-backed helper path,
    likely starting from `compute_dflash_sampling_accept_len_and_bonus(...)`,
    while preserving:
    - `temperature`
    - `top_k`
    - `top_p`
    - `min_p`
    - exact baseline-faithful target distribution
    Current branch status:
    - helper path is integrated
    - strict fallback remains
    - full commit/update fusion is still pending

15. `[done]` Add logical-cap support to the sampled helper path so
    `max_steps_per_req` remains first-class in both greedy and sampled target-only
    verify.

16. `[partial]` Keep the mixed-precision boundary explicit:
    - target verify forward on FP8 target KV
    - accepted hidden handoff
    - existing BF16 fused draft append reused after verify
    Current branch status:
    - no code on this branch tries to collapse target FP8 verify and draft BF16
      append into one monolithic mixed-precision kernel
    - the next optimization target remains target-only postprocess, not the
      target-to-draft ownership boundary

17. `[partial]` Add DFlash spec-v2 scheduler integration:
    - allocate/store/resolve DFlash future payloads
    - plumb replay metadata through graph runner
    - do not reuse Eagle tree semantics blindly
    Current branch status:
    - `FutureMap` can now allocate/store/resolve DFlash-native future state
    - `FutureMap` now preserves `verify_done` as part of that DFlash-native
      future state
    - DFlash workers now emit `next_draft_input` in the same result object shape
      the scheduler already expects
    - `ScheduleBatch` can now re-enter decode from DFlash draft state
    - `scheduler.py` now has an explicit spec-v2 overlap handoff helper for
      DFlash instead of mutating DFlash future state inline in the generic path
    - graph-runner verify-input construction is now explicit for DFlash
    - `ModelWorkerBatch` now carries the DFlash-owned cache references needed by
      verify/commit (`req_to_token_pool`, allocator, tree cache) and exposes the
      minimal batch/device helpers DFlash needs on the overlap path
    - scheduler output processing now has a DFlash-specific compact-overlap
      consumer, instead of assuming Eagle's fixed-stride accepted-token layout
    - target-only verify now records a compact overlap payload
      (`proposed_flat`, `commit_lens`) for future spec-v2 replay/output handling
    - `python/sglang/srt/speculative/dflash_worker_v2.py` now exists as the
      DFlash-native worker-v2 entry point, even though it is not selected yet
    - the overlap gate is still disabled
    - `scheduler.py` / `spec_info.py` still keep DFlash overlap disabled, so this
      is infrastructure only, not a live enabled path yet

18. `[pending]` Add DFlash-specific overlap safety checks:
    - grammar interaction
    - finish-state interaction
    - page-size interaction
    - mixed-precision cache ownership interaction

19. `[pending]` Reframe the current SSD prototype as a first-class DFlash overlap mode.
    Target files:
    - `dflash_worker.py`
    - `dflash_worker_v2.py`
    - `scheduler.py`
    - `overlap_utils.py`
    - `scheduler_output_processor_mixin.py`
    Goal:
    - stop treating SSD branch preparation as an auxiliary thread-side feature
    - make branch-prepared future outcomes visible to the overlap-v2 producer /
      consumer contract
    - keep verifier compute unchanged

20. `[pending]` Define the DFlash SSD future payload explicitly.
    It should carry, for each prepared outcome:
    - accepted-length hypothesis
    - bonus/recovery token hypothesis
    - prepared draft proposal identity
    - prepared draft KV ownership / validity metadata
    It should not reuse Eagle tree payload fields.

21. `[pending]` Decide the authoritative branch-selection signal for SSD mode.
    Candidate signals already present on this branch:
    - `accept_len_last`
    - `accept_len_ema`
    - `verify_ct_last`
    - `q_entropy_mean_last`
    - `q_max_mean_last`
    - miss streak
    - alt probability mass
    The implementation should choose one coherent policy rather than stacking
    ad hoc gates forever.

22. `[pending]` Benchmark the existing SSD prototype against the new overlap-v2 baseline.
    Compare:
    - overlap-v2 only
    - overlap-v2 + eager-next SSD cache
    - overlap-v2 + async shadow SSD overlap
    - later, overlap-v2 + hardened first-class SSD payload
    The benchmark must record:
    - cache hit rate
    - accepted suffix length on hit
    - accepted suffix length on miss
    - total throughput
    - draft-side overlap launch/wait counts

### Reverse-Trace Verification Tasks

23. `[pending]` Re-run the trace from the finished implementation backward:
    - fused verify result -> scheduler payload -> graph replay -> request commit
    and confirm every field is consumed by exactly one downstream step.

24. `[pending]` Re-run the original discovery path after implementation:
    - greedy target-only verify trace
    - sampled target-only verify trace
    - FP8/BF16 dtype trace
    - overlap/spec-v2 trace
    - SSD branch-preparation trace
    and confirm the new implementation actually replaced the old bottlenecks.

## Low-Risk Implementation Order

1. Land doc + trace first
2. Land greedy fused postprocess first
3. Keep sampled path functionally identical until greedy is stable
4. Promote sampled helper only after parity checks pass
5. Reuse existing fused BF16 draft append rather than redesigning it
6. Add DFlash spec-v2 overlap only after non-overlap fused verify is stable

## Best Next Implementation Point

Given the current branch state, the next undeniably best implementation target is:

1. finish the DFlash-owned fused selected-hidden append path on the overlap-v2
   serving route
2. make that path measurable under CUDA graph replay
3. only then promote SSD from a worker-side prototype into a first-class
   overlap-v2 future-outcome mode

Why this order is correct:

- SSD helps by eliminating draft wait time on predicted future outcomes
- but if the verify->append->next-draft boundary is still too expensive, SSD
  will be fighting the wrong bottleneck
- the current branch already has the right seam for this:
  - flat accepted indices
  - flat accepted positions
  - compact cache locations
  - DFlash-owned fused selected-hidden helper
- once that seam is stable under overlap-v2 graph replay, SSD can be layered on
  top of it without changing verifier semantics

So the architecture should be:

- step 1: stabilize fused verify/append on the overlap-v2 path
- step 2: expose SSD future-outcome preparation through the same payload system
- step 3: benchmark:
  - overlap-v2 only
  - overlap-v2 + SSD eager-next
  - overlap-v2 + SSD async overlap

This ordering keeps the risk surface small:

- first shrink the CPU verify/commit boundary
- then enable sampled parity
- then add overlap/spec-v2

## Remaining CPU-Owned Responsibilities After The Current Refactor

These are the exact pieces still sitting on the CPU side after compact D2H and
the shared request-mutation helper:

1. per-request token append / truncation decisions
   - max-new-token clipping
   - stop token / EOS clipping
   - grammar / regex / stop-string slow path
2. per-request stats mutation
   - `spec_verify_ct`
   - `spec_accepted_tokens`
   - acceptance histogram
3. conversion back to device tensors for downstream updates
   - `commit_lens`
   - `new_verified_id`
4. KV free / compaction policy selection
   - `page_size == 1`
   - paged aligned free path
5. req-level KV accounting
   - `kv_committed_len`
   - `kv_allocated_len`
6. req-to-token pool updates and tail clearing
7. next-step hidden-state segment assembly

That is the exact boundary the next implementation checkpoint should attack.

What is already extracted on this branch:

- `build_dflash_target_only_cache_plan(...)` now derives:
  - `keep_mask`
  - compacted `out_cache_loc`
  - page-aware `evicted_slots`
  - optional `evicted_pages`
  - `clear_start`
  - `clear_end`
- the helper includes a CPU fallback for page-alignment semantics so the logic can
  be unit-tested without CUDA
- `apply_dflash_target_only_cache_plan(...)` now owns allocator free/compact application
- `apply_dflash_target_only_mapping_updates(...)` now owns req-to-token mapping and
  clear-range updates
- `gather_dflash_committed_hidden(...)` now owns committed hidden assembly using the
  same `keep_mask`
- the same downstream helper surface is now reused by `dflash_tree_worker.py` for:
  - indexed cache compaction
  - req-level KV accounting
  - mapping updates
  - hidden gather by accepted flat indices

## Next Concrete Extraction Checkpoints

The next low-risk implementation sequence after the current branch state should
be:

1. move post-commit metadata shaping closer to the device boundary
   - offsets into the compact committed span
   - `new_verified_id`
   - per-request committed lengths
2. derive keep/evict and clear ranges from those compact lengths without
   rebuilding more CPU-side structure than necessary
3. only then introduce a DFlash-specific spec-v2 payload skeleton carrying:
   - `commit_len`
   - `new_verified_id`
   - compact committed-span metadata
   - next sequence lengths
   - any replay-safe KV ownership fields

This keeps the implementation order aligned with the real bottleneck instead of
jumping too early into overlap plumbing.

Checkpoint update:

- Step 1 above is now partially landed:
  - compact target-only commit offsets are emitted at pack time
  - final `commit_lens` / `new_verified_id` materialization is shared
- Step 2 is now partially landed:
  - `keep_mask`
  - paged `evicted_slots` / `evicted_pages`
  - clear-range metadata
- The next remaining part of step 2 is no longer the basic target-only path in
  `dflash_info.py`; it is:
  - any remaining inline req-level KV/accounting consumers
  - then spec-v2 payload design on top of the cleaner target-only path

Checkpoint update:

- parity extraction for the remaining `dflash_tree_worker.py` path is now landed
- the next major implementation frontier is no longer post-verify helperization;
  it is the DFlash-specific spec-v2 payload and overlap contract

## Benchmark / Validation Matrix

The benchmark matrix should stay separated by regime.

### Baselines

1. no-DFlash greedy baseline
2. no-DFlash sampled baseline

### Production DFlash

3. DFlash greedy target + greedy draft
4. DFlash sampled target + greedy draft

### Research Lane

5. DFlash sampled target + sampled draft

### For each lane, record

- final accuracy
- any-correct / pass@k
- majority/final selection accuracy
- total wall time
- per-question wall time
- verify count
- accept length
- accepted tokens
- entropy/confidence summaries when enabled

## Success Criteria

This roadmap is only successful when all of the following are true:

1. DFlash has a real spec-v2 overlap payload and can run with overlap enabled.
2. Greedy target-only verify no longer relies on the current full packed CPU
   transfer path.
3. Sampled target-only verify remains baseline-faithful and is materially closer
   to the optimized greedy postprocess path.
4. FP8 target KV + BF16 draft KV remains the standard production regime.
5. The final production path is still benchmarked separately for:
   - greedy target
   - sampled target
   - tool-calling vs no-tool
