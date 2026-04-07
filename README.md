# DFLASH on `showtime`

This `README.md` is the branch design contract for bringing GPT-OSS-120B DFLASH speculative decoding to SGLang `showtime`.

It replaces the old benchmark-heavy branch README with a code-anchored design and implementation plan. Raw benchmark history still lives in:

- `BENCHMARK_LEDGER.md`
- `DFLASH_PROD_VERIFY_ROADMAP.md`
- `docs/dflash_showtime_branch_inventory.md`
- `docs/dflash_target_drift_audit.md`
- `docs/dflash_fused_tree_design.md`

## 1. Scope

This branch is about one thing: making DFLASH correct first, then fast, for GPT-OSS-120B on SGLang.

The active serving target is:

- **target model**: GPT-OSS-120B
- **draft model**: DFLASH GPT-OSS draft checkpoint
- **official SGLang baseline**: `v0.5.9` lineage
- **serving branch**: `radna0/showtime`

The immediate product goal is not "every speculative mode at once". It is:

1. Exact target-faithful linear DFLASH
2. Batched request support
3. BF16 correctness lane
4. FP8 target KV + BF16 draft KV optimization lane
5. Explicit, opt-in tree verify after linear parity

Out of scope for the first production lane:

- Overlap/spec-v2 as the default path
- PQ verify
- Overlap-v2 correctness claims without fresh proof
- Adaptive route / PaCoRe as part of the first DFLASH proof
- Exact sampled speculative decoding beyond the current target-only support

## 2. Locked Decisions

| Area | Decision | Why |
| --- | --- | --- |
| First correctness lane | `DFLASH` linear verify | Smallest surface, easiest to prove exactness |
| First correctness precision | target KV `bf16`, draft KV `bf16` | Establish correctness before mixed precision |
| First optimization precision | target KV `fp8_e4m3`, draft KV `bf16` | Matches the intended H100 serving lane |
| First page-size lane | `page_size=1`, `speculative_draft_page_size=1` | Simplest cache-plan and eviction semantics |
| First pool strategy | `share_pools=False` | Avoid heterogeneous-pool and radix aliasing issues while proving correctness |
| First block size | `16` | Matches checkpoint training block size |
| Supported inference block sizes | `4`, `8`, `12`, `16` | Useful sweep knobs after the baseline passes |
| Target attention backend | `fa3` | Required serving contract for this branch |
| Draft attention backend | `fa3` | Keep DFLASH on the same verified attention family |
| Target MoE backend | `triton_kernel` | Locked serving contract for GPT-OSS-120B |
| Draft MoE backend | `triton_kernel` | Keep draft and target backend behavior aligned |
| Sampling backend | `pytorch` | Stable baseline for target-only sampled verify bring-up |
| Default overlap state | OFF | Do not mix correctness bring-up with overlap-v2 replay complexity |
| Default tree state | opt-in via `DFLASH_TREE` | Tree verify is valuable, but not the first proof target |

## 3. What Was Audited

The design below is based on five sources of truth:

1. Official SGLang `v0.5.9` speculative decoding structure
2. Upstream DFLASH PRs `#16818` and `#22077`
3. Local `showtime` plus `dflash-pagesize-fix`
4. The DFLASH training repository `radna0/cuda-dflash` (`gold-standard-multi-h100-single`)
5. The public draft checkpoint config on Hugging Face

### 3.1 Official SGLang `v0.5.9`

Audited files:

- `python/sglang/srt/speculative/eagle_worker.py`
- `python/sglang/srt/speculative/eagle_info.py`
- `python/sglang/srt/speculative/eagle_utils.py`
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/model_executor/model_runner.py`

### 3.2 Upstream DFLASH PRs

- **Closed precursor**: `sgl-project/sglang#16818`
- **Active successor**: `sgl-project/sglang#22077`

The architectural takeaway is the same in both PRs:

- Add a draft model class (`models/dflash.py`)
- Add DFLASH request/verify state (`dflash_info.py`)
- Add capture, sampling, and cache-plan helpers (`dflash_utils.py`)
- Add a linear DFLASH worker (`dflash_worker.py`)
- Plumb DFLASH through `spec_info.py`, `model_runner.py`, `server_args.py`, and `cuda_graph_runner.py`
- Reuse the `TARGET_VERIFY` path instead of adding an EAGLE-style separate draft graph family

`#22077` mainly rebases and cleans up the earlier work. It does not replace the core design.

### 3.3 Local branch evidence

- `showtime` already carries linear DFLASH, tree DFLASH, overlap experiments, fused KV helpers, capture-contract logging, and target/draft KV dtype separation.
- `dflash-pagesize-fix` is still the best source for old measured proof constraints.
- That older branch locked a narrower proof lane around `page_size=1`, `share_pools=False`, and a small block-size regime.

### 3.4 Training repo evidence

Audited repo: `workspace/cuda-dflash`

Key files:
- `SpecForge/scripts/train_dflash.py`
- `SpecForge/specforge/core/dflash.py`
- `SpecForge/specforge/data/preprocessing.py`
- `SpecForge/specforge/modeling/draft/dflash_gptoss.py`

### 3.5 Checkpoint evidence

Audited public config:
- `dflash_config.mask_token_id = 200019`
- `dflash_config.target_layer_ids = [1, 9, 17, 25, 33]`
- `block_size = 16`

## 4. Research Findings That Matter

### 4.1 EAGLE and DFLASH should not share the same worker design

Official EAGLE/EAGLE3 draft workers:
- Own separate draft graph runners.
- Own separate tree-building logic.
- Use EAGLE-specific retrieval and tree verify helpers.

DFLASH is different:
- The draft is a fixed-size non-causal block.
- Verify is still a target-model `TARGET_VERIFY` pass.
- The profitable seam is reusing the `TARGET_VERIFY` capture family, not cloning EAGLE's separate draft graph-runner model.

That means DFLASH should keep its own worker and state objects even when it borrows some tree infrastructure later.

### 4.2 The CUDA-graph contract is "reuse TARGET_VERIFY mode", not "copy EAGLE graph topology"

The important upstream comment is directionally correct:
> EAGLE/standalone/ngram draft workers use separate cuda-graph runners; DFLASH draft uses a fixed-size block and reuses TARGET_VERIFY graphs for performance.

The best reading of that comment, consistent with the current local code, is:
- DFLASH should stay inside the `TARGET_VERIFY` graph shape family.
- DFLASH should not add a second EAGLE-style draft graph capture stack.
- The draft block width must be fixed and graph-friendly.
- Target hidden capture requirements must be stable across prefill and verify.

### 4.3 The target-layer capture mapping is resolved: capture boundary is `-1`

I found the strongest local clue already: the SGLang target models that support DFLASH all use the `+1` capture mapping because the engine surfaces the hidden state from the previous layer boundary.

- config `target_layer_ids = [1, 9, 17, 25, 33]`
- draft trained to consume input to these layers.
- to get input to layer `L`, we capture output of layer `L-1`.
- so we capture after layers `[0, 8, 16, 24, 32]`.

In the code, this means:
`capture_layer_ids = [lid - 1 for lid in raw_target_layer_ids if lid > 0]`

### 4.4 Training and inference are related, but not identical

Training uses Flex Attention block masks and random anchor sampling. Inference uses the current verified token as the anchor and a fixed block of masked positions.
**Goal**: Preserve the target-hidden conditioning contract and block-wise draft semantics; do not try to literally replay the training-time random-anchor data pipeline online.

### 4.5 Mixed precision should be staged

- **Phase 1**: target KV `bf16`, draft KV `bf16` (Establish correctness first).
- **Phase 2**: target KV `fp8_e4m3`, draft KV `bf16` (optimization fast lane).

### 4.6 Shared pools are a later optimization

Initial recommendation: `share_pools=False`.
Avoids mixing target and draft KV pages with different dtypes and layouts while proving correctness. Uplift from separate pools is roughly 16.7% - 33% memory but significantly lowers engineering risk.

### 4.7 Tree verify is worth supporting, but not as the first default

Linear DFLASH is the simpler exactness target. `DFLASH_TREE` remains an explicit, opt-in algorithm.

### 4.8 Current benchmark evidence is uneven

The ladder:
1. `86e8e5` (Hardest) - Missing fresh evidence.
2. `dd7f5e` (Harder) - Missing fresh evidence.
3. `a295e9` (Decently hard) - Audited.
4. `9c1c5f` (Medium) - Audited.
5. `92ba6a` (Easy) - Audited.

The hardest two problems need re-runs under the finalized contract.

## 5. Detailed Design & Implementation Roadmap

(Follow the 50-task plan in `implementation_plan.md`)

... [Remaining 50 tasks from Task List] ...

---

That is the fastest path to something we can trust.
