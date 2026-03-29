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

### 1. DFlash overlap/spec-v2 is disabled today

Code:

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/speculative/spec_info.py`

Confirmed behavior:

- `server_args.py` forces `disable_overlap_schedule = True` for DFlash
- `spec_info.py` raises if overlap is enabled for DFlash

So current DFlash runs have:

- no overlap scheduler
- no spec-v2 future payload
- no DFlash participation in `FutureMap`
- no DFlash-specific overlap replay plumbing

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

What remains:

- CPU request mutation still exists
- finish/grammar/stop handling still lives in the Python loop
- KV free/compaction still happens after CPU-side appended-length decisions

What was improved on this branch:

- the CPU-side request mutation logic is now centralized in
  `commit_dflash_proposed_tokens_to_req(...)`
- both `dflash_info.py` and `dflash_tree_worker.py` use that shared helper
- this does not yet remove the CPU loop, but it gives us one stable seam for
  later:
  - device-generated commit metadata
  - device-generated keep/evict decisions
  - DFlash-specific spec-v2 payload work

So the CPU-shaped boundary is smaller now, but it is still the main fusion
target.

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

### 9. FP8 target KV and BF16 draft KV are separate and correctly configurable

Code:

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/speculative/dflash_worker.py`

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

So the branch now has the correct DFlash-native payload seam, but:

- DFlash still reports `supports_spec_v2() == False`
- scheduler overlap is still disabled
- no live overlap execution path uses the new payload yet

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
    - the overlap gate is still disabled
    - `cuda_graph_runner.py` and `scheduler.py` still need the live producer /
      consumer wiring for DFlash-specific overlap execution

18. `[pending]` Add DFlash-specific overlap safety checks:
    - grammar interaction
    - finish-state interaction
    - page-size interaction
    - mixed-precision cache ownership interaction

### Reverse-Trace Verification Tasks

19. `[pending]` Re-run the trace from the finished implementation backward:
    - fused verify result -> scheduler payload -> graph replay -> request commit
    and confirm every field is consumed by exactly one downstream step.

20. `[pending]` Re-run the original discovery path after implementation:
    - greedy target-only verify trace
    - sampled target-only verify trace
    - FP8/BF16 dtype trace
    - overlap/spec-v2 trace
    and confirm the new implementation actually replaced the old bottlenecks.

## Low-Risk Implementation Order

1. Land doc + trace first
2. Land greedy fused postprocess first
3. Keep sampled path functionally identical until greedy is stable
4. Promote sampled helper only after parity checks pass
5. Reuse existing fused BF16 draft append rather than redesigning it
6. Add DFlash spec-v2 overlap only after non-overlap fused verify is stable

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
