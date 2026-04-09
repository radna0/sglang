# DFlash Fused-Tree Drafting (SGLang) — Design Notes

This doc explains how to combine **DFlash** (parallel, non-causal draft blocks) with SGLang’s existing
**EAGLE/EAGLE3 fused-tree verification** to reduce verify overhead and improve throughput in
high-concurrency serving.

The concrete implementation target in this repo is `--speculative-algorithm DFLASH_TREE`.

## 1) Goals / Non-goals

**Goals**

- Increase effective speculative acceptance under batching (e.g. concurrency `c=8`) by letting the
  target pick from multiple draft candidates at each step.
- Reuse SGLang’s existing fused-tree attention mask + verification kernels (the same infra used by EAGLE).
- Keep memory bounded and CUDA-graph friendly (fixed shapes where possible).
- Maintain the DFlash “shifted” verify semantics and “bonus token” behavior.

**Non-goals (initially)**

- DP-attention / overlap scheduling (“spec v2”) support.
- Grammar-constrained decoding.
- Exact, distribution-preserving speculative sampling for arbitrary sampling params (see §7).

## 2) Background: what “fused-tree” means in SGLang

SGLang’s fused-tree speculative verification works by:

1. Building a **tree** of candidate tokens (nodes).
2. Running **one** target forward pass over all nodes with a **custom attention allow-mask**
   so each node only attends to its own prefix-path.
3. Running a GPU kernel that traverses the tree and returns:
   - accepted path indices (`accept_index`)
   - number of accepted draft tokens (`accept_token_num`)
   - `predict` token ids (accepted tokens + a “bonus” token).

The key semantics (greedy verify kernel) are in `sgl-kernel/csrc/speculative/eagle_utils.cu`:

- At each depth, compare:
  - candidate token at **child node**
  - vs target argmax at **parent node**
- If match: accept and descend; else try sibling; else stop.
- `predict[parent]` stores the accepted child token id (the “shift”).
- The final “bonus” token is stored at the last accepted node index: `predict[last] = target_predict[last]`.

This is the same “shifted” rule DFlash uses.

## 3) Existing building blocks used by DFlash

### 3.1 DFlash draft model in SGLang

`python/sglang/srt/models/dflash.py` defines `DFlashDraftModel`:

- **No lm_head**; it outputs hidden states only.
- Uses `RadixAttention(attn_type=ENCODER_ONLY)` → **non-causal** attention on the draft block.
- Requires `input_embeds` (we use the **target** model’s embedding for the mask/anchor tokens).
- Uses projected target hidden-state features as the draft KV cache context (materialized by the worker).

### 3.2 DFlash sequential worker (baseline behavior)

`python/sglang/srt/speculative/dflash_worker.py` (`DFLASH`) does:

1. Append newly committed target hidden-states into the draft KV cache.
2. Run one draft block forward (window = `block_size`).
3. Greedy-sample **one** candidate per position from the target’s vocab-parallel `lm_head`.
4. Verify with the target using a **causal** block mask (or backend-native causal masking).
5. Compute accept length with the shifted rule:
   `candidates[:, 1:] == target_predict[:, :-1]`.
6. Commit `verified_id + accepted draft tokens` into the target KV cache.
7. Return the new **bonus token** as the next `verified_id` (not yet committed; it becomes next iteration’s root).

## 4) Proposed algorithm: DFLASH_TREE (draft once + tree verify)

DFLASH_TREE keeps the DFlash draft model **one-shot** (fast) but upgrades verification from a
single linear chain to a **bounded tree**.

### Stage A — Draft once

- Build a draft block input of length `block_size`:
  - token 0 = current `verified_id` (the “anchor” token)
  - tokens 1..B-1 = `mask_token_id`
- Run `DFlashDraftModel` once to obtain hidden states `[bs, block_size, hidden]`.

### Stage B — Build a bounded proposal tree (per-step top-k + beam-style pruning)

For a configured `spec_steps = S` (`S < block_size`):

- For each step `t in [1..S]`, compute per-position `topk` token candidates and probs from the
  target `lm_head` applied to the draft hidden state at position `t`.
- Construct a bounded tree using the same beam-style utilities used by EAGLE:
  - `select_top_k_tokens()` to maintain a width-`topk` beam across steps
  - `organize_draft_results()` to select the top `num_verify_tokens - 1` nodes by score
- Then call `build_tree_kernel_efficient()` to produce:
  - `tree_mask` (custom attention allow-mask)
  - `positions` (RoPE positions for each node; root at `seq_len`, depth adds +1)
  - retrieval arrays (`retrive_index`, `retrive_next_token`, `retrive_next_sibling`)
  - flattened verify tokens `[root | selected nodes]`

This stage is *where tree capacity lives*:

- `num_verify_tokens` controls how many nodes the target must verify (cost).
- `topk` and `spec_steps` control how much branch diversity is available (benefit).

### Stage C — One target forward pass (tree verify)

Run the target model once with `ForwardMode.TARGET_VERIFY` over all nodes and the custom `tree_mask`.

### Stage D — Accept + commit + update draft KV

Greedy mode:

- Use `verify_tree_greedy` kernel to produce `accept_index`.
- Append the produced tokens to each request (respecting stop tokens / max_new_tokens).
- Commit KV for the accepted node indices (`accept_index_flat`) and free unaccepted slots.
- Extract target hidden states for the committed indices and append into the **draft** KV cache.
- Set the *new* `verified_id` to the last appended token (the “bonus” token).

## 5) Correctness invariants to preserve

### 5.1 “Shifted” verify rule

The tree verify kernel compares a **child** token id to the target argmax/probs at the **parent** node.
This matches the DFlash shifted rule used by the sequential accept-length computation.

### 5.2 Bonus token handling

The bonus token is returned to the user, but its KV is not materialized until the next iteration when it
becomes the next root `verified_id` and is processed by the target in the next verify forward.

### 5.3 Position ids

`build_tree_kernel_efficient` sets:

- root position = current `seq_len`
- node position = `seq_len + depth(node)`

This ensures that any accepted path corresponds to a contiguous causal sequence of positions.

## 6) Key knobs + expected tradeoffs

- `block_size` (`--speculative-dflash-block-size`):
  - size of the draft window; draft runs once per iteration over this many tokens.
- `spec_steps` (`--speculative-num-steps`):
  - maximum number of speculative steps (tree depth) we attempt to accept before emitting a bonus token.
  - must satisfy `spec_steps <= block_size - 1`.
- `topk` (`--speculative-eagle-topk` reused):
  - beam width / per-step branching factor.
- `num_verify_tokens` (`--speculative-num-draft-tokens`):
  - total nodes verified by the target per iteration.
  - **must be > block_size** if you want *extra* alternatives beyond the linear chain.

Rule of thumb:

- If acceptance drops under concurrency/sampling, you usually want to increase *tree capacity*
  (`num_verify_tokens`, possibly `topk`) rather than only `block_size`.

## 7) Production sampling (temperature/top-p/min-p): what’s needed

Greedy verification is not representative of production sampling (e.g. `temperature=1.0` + `min_p`).

To make acceptance and speedups reflect production, DFLASH_TREE should support a stochastic verify path.
SGLang already has a CUDA kernel for *target-only* tree sampling:

- `tree_speculative_sampling_target_only` (see `sgl-kernel/csrc/speculative/speculative_sampling.cuh`)

This requires:

1. Convert target logits → `target_probs` after applying the same sampling transforms used in normal decode
   (temperature, top-k/top-p renorm, logit processors, penalties, etc.).
2. Run the tree sampling kernel to obtain an accepted path and a final sampled token.

Important caveat:

- The current kernel is explicitly “target-only” (it does not fully leverage draft probabilities; see FIXME in code).
  It is *not* a full, exact speculative sampling implementation.

Quality-first roadmap:

- Phase 1: greedy-only (already).
- Phase 2: enable target-only stochastic tree sampling (fast, approximate).
- Phase 3: implement exact speculative sampling using draft probs (requires careful probability accounting + efficiency work).

## 8) Memory model (bounded and CUDA-graph friendly)

- Draft side:
  - allocate `block_size` ephemeral draft KV slots per request for the non-causal draft forward,
    then immediately restore allocator state (no draft growth from speculative blocks).
  - draft KV grows only with committed tokens whose target hidden states are appended each iteration.

- Target side:
  - allocate `num_verify_tokens` KV slots per request for the verify forward.
  - free unaccepted slots immediately after accept.

For CUDA graph:

- keep `(block_size, num_verify_tokens, topk, spec_steps)` static during capture.
- keep `page_size == 1` initially (paged + tree needs more work).

## 9) Implementation checklist (next steps)

- Add stochastic (non-greedy) verification path for DFLASH_TREE via `tree_speculative_sampling_target_only`.
- Support `page_size > 1` (paged KV + tree compaction).
- Support DP-attention + overlap scheduling (spec v2) if needed for large-scale serving.
- Add tests:
  - verify kernel “shift” + bonus behavior is respected end-to-end for DFLASH_TREE.
  - position-id and custom-mask correctness for simple toy trees.
- Add metrics/logging:
  - per-batch accept mean, p50/p90, verify count, and (for sampling) rejection stats.

## 10) Validation plan (H100/Kaggle-style)

Benchmark matrix (each with `c=1` and `c=8`, CUDA graph ON):

- Baseline (no speculative)
- DFLASH (sequential)
- DFLASH_TREE (vary `topk`, `num_verify_tokens`, `spec_steps`)

Measure:

- `output_tok_s`
- `spec_accept_length_mean` (and histogram)
- `spec_verify_ct` per request
- stability at long decode lengths (e.g. 8k / 16k / 65k) and large context length settings (65k/131k).

