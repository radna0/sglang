# GPT-OSS-120B CARE MLA Serving Status

## Scope

This document records the current serving/inference compatibility state for the
GPT-OSS-120B CARE/CARE-E converted MLA checkpoint in this SGLang fork:

- repo: `/root/sglang-gpt-oss-care-mla`
- branch: `gpt-oss-care-mla`
- commit: `49a0ad278`

The target serving requirement is stricter than "generic MLA support exists".
For GPT-OSS-120B CARE we need all of the following together:

1. native GPT-OSS MLA model loading
2. CARE-E dynamic per-layer `kv_lora_rank`
3. attention sinks
4. alternating sliding/full attention
5. deployable SGLang inference path

This document distinguishes:

- `implemented`: the code path exists
- `proven`: we have direct evidence that the path is correct enough to rely on
- `blocked`: an explicit missing piece prevents the full target path

## Current recommended serving lane

For the current GPT-OSS CARE checkpoint family, the only serving path that
matters is now:

- checkpoint:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210/conversion/converted_checkpoint`
- attention backend: `flashmla`
- page size: `64`
- model dtype: `bfloat16`
- KV cache dtype: `bfloat16`
- CUDA graph: disabled
- piecewise CUDA graph: disabled
- status: `proven for fixed-r512 smoke`

Proof artifact:

- smoke output:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210/conversion/sglang_flashmla_smoke_v14/smoke_flashmla.json`
- server log:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210/conversion/sglang_flashmla_smoke_v14/server_flashmla.log`

What this proves:

- GPT-OSS MLA checkpoint loading works in SGLang
- FlashMLA-only attention serving works for GPT-OSS MLA
- BF16 KV cache works
- alternating sliding/full attention runs
- GPT-OSS attention sinks are wired through the FlashMLA path
- the server accepts and completes an OpenAI-compatible generation request

Important limits:

- this proof is for the fixed-r512 CARE-U checkpoint family, not CARE-E
- dynamic per-layer `kv_lora_rank` remains unsupported on the FlashMLA path
- the current GPT-OSS FlashMLA path uses sparse BF16 decode for GPT-OSS-specific
  key geometry because `flash_mla_with_kvcache` is hard-limited to DeepSeek's
  `head_size_k = 576`
- non-attention kernels still exist elsewhere in the runtime logs
  (for example MoE / Mamba), but the MLA attention path itself is FlashMLA-only

Explicit non-goals for this serving proof:

- `flex_attention2`
- `flex_flash4`
- `flashinfer_mla`
- `triton` attention

Those are no longer the recommended path for GPT-OSS CARE serving in this repo.

## CPU verification status

We now have a CPU-side regression module for the GPT-OSS CARE serving wiring:

- `test/manual/test_gpt_oss_care_cpu_checks.py`

Current checks covered there:

- dynamic-rank helper parsing in `ModelConfig`
- production rounded CARE-E schedule logic
- GPT-OSS CARE backend auto-selection
- explicit rejection of the wrong backend for dynamic-rank + sliding GPT-OSS MLA
- `RadixAttention` sink / RoPE kwargs plumbing
- source-level checks that GPT-OSS MLA still preserves:
  - per-layer `kv_lora_rank_per_layer`
  - `sinks=self.sinks`
  - sliding-window wiring

The current CPU run is green:

- `python test/manual/test_gpt_oss_care_cpu_checks.py`
- result: `Ran 18 tests ... OK`

## Smoke-driven fixes already landed

Before the current successful FlashMLA smoke, the following serving blockers
were found and fixed:

1. GPT-OSS MLA attention was incorrectly passing the MXFP4 quant config into
   MLA-specific dense attention projections.
   - stock GPT-OSS attention does not do this
   - the result was an assertion failure in `ColumnParallelLinear` during model
     construction
   - fix: `GptOssMLADecoderLayer` now mirrors the stock GPT-OSS path and
     instantiates MLA attention projections with `quant_config=None`

2. `ModelRunner.configure_kv_cache_dtype()` was allowing the serving path to
   drift toward `float16` KV cache in `auto`.
   - that is not acceptable for this project
   - fix: the GPT-OSS CARE smoke path now explicitly launches with
     `--kv-cache-dtype bfloat16`
   - additionally, the runner now coerces `auto` from `torch.float16` to
     `torch.bfloat16` instead of trying to use fp16 KV cache

3. FlashMLA sparse sliding decode was incorrectly routing through a DeepSeek-only
   FP8 key-cache quantizer that hardcoded `512 + 64` key width.
   - fix: GPT-OSS sliding decode now stays in pure BF16 and uses the sparse
     FlashMLA kernel directly

4. FlashMLA `with_kvcache` decode itself is hard-limited to `head_size_k == 576`.
   - GPT-OSS fixed-r512 MLA uses `512 + 32`, not `512 + 64`
   - fix: GPT-OSS decode now uses sparse FlashMLA BF16 decode whenever the key
     width is not `576`

5. Several branch-regression compatibility issues blocked first-token streaming:
   - missing `ServerArgs.ssl_verify()`
   - missing `ServerArgs.disable_piecewise_cuda_graph`
   - stale `BatchTokenIDOutput` constructor fields
   - stale `SchedulerReqTimeStats` compatibility aliases
   - in-place overlapping `k_input[..., kv_lora_rank:] = k_pe` write in
     `gpt_oss.py`

These are no longer open blockers for the fixed-r512 FlashMLA smoke lane.

## Current CARE checkpoint state

The current Alpaca `128 x 2048` CARE reproduction produced a real converted
checkpoint. For serving and benchmarking, the canonical artifact is the
repaired clean checkpoint at:

- `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/converted_checkpoint_clean`

The original mixed artifact at:

- `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/converted_checkpoint`

is no longer the serving/benchmark target because its safetensors index still
advertises dense `k_proj` / `v_proj` tensors alongside MLA tensors.

The checkpoint is not uniform-rank. Its CARE-E schedule is:

- file:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/kv_lora_rank_schedule.json`
- layer count: `36`
- min rank: `308`
- max rank: `602`
- sum of ranks: `18432`
- first 10 ranks:
  `416, 308, 417, 431, 463, 456, 505, 449, 512, 531`

This matters because any serving claim must handle a true dynamic-rank MLA
checkpoint, not just a uniform-rank DeepSeek-style model, and it must do so on
the cleaned artifact we can benchmark consistently.

## What is already implemented

### 1. SGLang recognizes GPT-OSS MLA checkpoints as MLA models

Implemented.

Evidence:

- `python/sglang/srt/configs/model_config.py`
  - `GptOssMlaForCausalLM` is treated as `AttentionArch.MLA`
  - `head_dim = qk_nope_head_dim + qk_rope_head_dim`
  - MLA-specific config fields are loaded

Implication:

- SGLang will not misclassify a converted GPT-OSS MLA checkpoint as ordinary
  MHA when building runtime model metadata.

### 2. SGLang has a GPT-OSS MLA runtime model class

Implemented.

Evidence:

- `python/sglang/srt/models/gpt_oss.py`
  - `class GptOssMLAAttention`
  - `class GptOssMlaForCausalLM`

What this runtime class does:

- builds GPT-OSS MLA attention from:
  - `q_proj`
  - `kv_a_proj_with_mqa`
  - `o_proj`
- loads and folds absorbed weights:
  - `w_kc`
  - `w_vc`
- preserves GPT-OSS sinks
- preserves alternating sliding/full attention pattern through
  `sliding_window_size`

Implication:

- There is already a native SGLang-side GPT-OSS MLA model implementation.
- We are not limited to HF-only benchmarking.

### 3. Dynamic per-layer MLA rank exists in the general SGLang runtime

Implemented.

Evidence:

- `python/sglang/srt/configs/model_config.py`
  - `kv_lora_rank_per_layer`
  - `has_dynamic_mla_kv_lora_rank()`
  - `get_mla_kv_lora_rank(layer_id)`
  - `get_unique_mla_kv_lora_ranks()`
- `python/sglang/srt/mem_cache/memory_pool.py`
  - `MLATokenToKVPool` allocates per-layer KV cache dims
- `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
  - builds one wrapper per unique MLA rank
  - dispatches by per-layer `v_head_dim`
- `docs/dynamic_mla_rank.md`

Implication:

- CARE-E dynamic schedules are not a theoretical future feature; the core stack
  already has real support for them.

### 4. Attention sinks are implemented in the MLA runtime

Implemented, backend-dependent.

Evidence:

- `python/sglang/srt/models/gpt_oss.py`
  - `GptOssMLAAttention` owns `self.sinks`
  - passes `sinks=self.sinks` into `RadixAttention`
- `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
  - applies sink correction with `apply_attention_sinks(...)` in both prefill
    and decode
- `python/sglang/srt/layers/attention/triton_backend.py`
  - accepts `sinks=` in extend/decode
- `python/sglang/srt/layers/attention/torch_flex2_backend.py`
  - accepts `sinks=`
  - applies sink correction using `return_lse=True`

Implication:

- GPT-OSS-style sink attention is not missing from the runtime.
- The question is not "do sinks exist", but "which GPT-OSS MLA serving backend
  combines sinks with the rest of our requirements".

### 5. Sliding-window attention exists in the GPT-OSS MLA runtime

Implemented, backend-dependent.

Evidence:

- `python/sglang/srt/models/gpt_oss.py`
  - `GptOssMLAAttention` sets `sliding_window_size` on `RadixAttention`
  - uses `layer_type` to choose sliding vs full attention
- `python/sglang/srt/layers/attention/triton_backend.py`
  - explicit sliding-window logic in extend/decode
- `python/sglang/srt/layers/attention/torch_flex2_backend.py`
  - explicit `window_left = int(getattr(layer, "sliding_window_size", -1))`

Implication:

- The runtime has real support for sliding-window attention.
- The missing question is whether the exact MLA backend chosen for GPT-OSS CARE
  can combine sliding-window with dynamic rank and sinks.

## What is only partially implemented

### 6. GPT-OSS MLA absorbed runtime is now CARE-E-aware in code, but not yet inference-proven

Implemented in code, not yet proven end-to-end.

The original mismatch was that `GptOssMLAAttention` read only:

- `self.kv_lora_rank = int(getattr(config, "kv_lora_rank"))`

That is now patched to mirror the DeepSeek V2 pattern at decoder-layer
construction time:

- reading `kv_lora_rank_per_layer[layer_id]`

Evidence:

- `python/sglang/srt/models/deepseek_v2.py`
  - `DeepseekV2DecoderLayer` reads `kv_lora_rank_per_layer[layer_id]`
- `python/sglang/srt/models/gpt_oss.py`
  - `GptOssMLADecoderLayer` now reads `kv_lora_rank_per_layer[layer_id]`
  - and passes the layer-local rank into `GptOssMLAAttention(..., kv_lora_rank=...)`

Why this matters:

- the current CARE checkpoint is dynamic-rank
- GPT-OSS MLA tensors such as:
  - `kv_a_proj_with_mqa`
  - `w_kc`
  - `w_vc`
  are rank-dependent per layer
- a scalar-rank model class can silently become a mismatch point even if the
  lower runtime layers support dynamic MLA

Implication:

- the obvious GPT-OSS per-layer-rank construction bug is fixed
- the remaining serving gap is runtime proof on the real checkpoint

## What is explicitly blocked today

### 7. Non-FlashInfer MLA backends reject dynamic CARE-E schedules

Blocked by explicit guards.

Evidence:

- `python/sglang/srt/layers/attention/flashmla_backend.py`
- `python/sglang/srt/layers/attention/cutlass_mla_backend.py`
- `python/sglang/srt/layers/attention/trtllm_mla_backend.py`

Each of these backends raises `NotImplementedError` when a dynamic per-layer
`kv_lora_rank` schedule is detected.

Implication:

- dynamic-rank GPT-OSS CARE MLA cannot be deployed on:
  - `flashmla`
  - `cutlass_mla`
  - `trtllm_mla`

### 8. FlashInfer MLA supports dynamic rank and sinks, but not the full GPT-OSS sliding-window requirement

Blocked for the full target combination.

What FlashInfer MLA already has:

- dynamic per-layer rank support
- sink correction support

What is missing for GPT-OSS CARE serving:

- explicit sliding-window metadata/indexing path on the MLA backend
- a proven alternating sliding/full MLA runtime path for GPT-OSS hybrid layers

Evidence:

- `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
  has dynamic-rank wrapper selection and sink correction
- but it does not expose the Triton-style sliding-window metadata path used in
  `triton_backend.py`

Implication:

- FlashInfer MLA is currently the best generic dynamic-rank MLA backend
- it is **not yet enough** to claim full GPT-OSS CARE MLA serving readiness

## Serving compatibility matrix

| Feature | Core stack | GPT-OSS MLA model | Proven end-to-end today | Notes |
| --- | --- | --- | --- | --- |
| Native GPT-OSS MLA model loading | yes | yes | partial | model class exists |
| Absorbed `w_kc` / `w_vc` loading | yes | yes | partial | handled in `post_load_weights()` |
| CARE-E dynamic rank | yes | yes in code | no | GPT-OSS model class now reads per-layer ranks, but runtime proof is still pending |
| Attention sinks | yes | yes | partial | backend-dependent |
| Sliding-window MLA | yes | yes | partial | backend-dependent |
| Dynamic rank + sinks | yes on FlashInfer MLA | partial on GPT-OSS | no | blocked by GPT-OSS per-layer rank gap |
| Dynamic rank + sliding-window | partial | partial | no | no single proven path yet |
| Dynamic rank + sinks + sliding-window for GPT-OSS MLA | no | no | no | this is the exact missing serving milestone |

## Practical verdict

### What we can honestly claim now

1. SGLang already contains a real GPT-OSS MLA runtime model.
2. SGLang already contains real dynamic-rank MLA infrastructure.
3. SGLang already contains sink-aware and sliding-window-aware attention code.
4. The current GPT-OSS-120B CARE checkpoint is a real dynamic-rank MLA
   checkpoint, not a fake uniform-rank placeholder.

### What we cannot honestly claim yet

We cannot yet claim that this branch has a proven production serving path for:

- GPT-OSS-120B CARE MLA
- with dynamic per-layer rank
- with attention sinks
- with alternating sliding/full attention
- in one end-to-end SGLang inference runtime

The missing integration is real and specific, not vague:

1. one serving backend must support the full combination:
   - dynamic rank
   - sinks
   - sliding-window MLA

## Best implementation path

The most direct path is not "make every MLA backend dynamic-rank-capable".
The most direct path is:

1. keep the serving path on a dedicated MLA backend
2. prove inference first on that path
3. only then optimize for faster MLA kernels

Current backend direction:

- `flashinfer`

Reason:

- `flashmla` in this tree still rejects CARE-E dynamic rank outright
- `flashinfer_mla` already contains the dynamic-rank wrapper logic and sink
  correction logic
- the immediate runtime blocker on `flashinfer` was CUDA-graph capture, which
  we now disable for dynamic-rank GPT-OSS CARE smoke runs

Important caveat:

- `flashinfer` is still not a full final victory for GPT-OSS CARE serving
- the remaining proof target is still the exact combination:
  - dynamic rank
  - sinks
  - alternating sliding/full attention

## FluentLLM comparison

We inspected `meituan-longcat/SGLang-FluentLLM` as a possible `flashmla`
acceleration source.

What it clearly adds:

- FlashMLA SwapAB-oriented optimizations
- FlashMLA FP8 KV/cache compute optimizations
- FlashMLA-first speculative/decode integration for their stack

What it does **not** add for the current GPT-OSS CARE target:

- no CARE-E `kv_lora_rank_per_layer` dynamic-rank path
- no GPT-OSS MLA model path with per-layer MLA rank loading
- no proven GPT-OSS sink-aware FlashMLA serving path
- no proven GPT-OSS alternating sliding/full MLA serving path

Implication:

- FluentLLM is useful as a future optimization reference for a production
  `flashmla` lane
- it does **not** remove the current GPT-OSS CARE compatibility blocker
- if we later switch to a rounded or bucketized schedule that FlashMLA can
  support efficiently, it becomes much more relevant

## Smoke entrypoint prepared

To prove serving on the real converted checkpoint, this repo now has:

- `scripts/run_gpt_oss_care_sglang_smoke.py`
- `scripts/run_gpt_oss120b_care_sglang_smoke.sh`

Default shell entrypoint:

```bash
/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_sglang_smoke.sh
```

Current defaults:

- model path:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/converted_checkpoint_clean`
- backend:
  `auto -> flashinfer`
- tp size:
  `8`

This is intended to be the first end-to-end serving proof path after the
benchmark GPUs are free.

## Immediate implementation tasks

1. Prove an end-to-end SGLang inference smoke on the clean dynamic-rank GPT-OSS
   CARE checkpoint with `flashinfer`.
2. Prove sinks are numerically active on the SGLang path, not only present in
   code.
3. Prove alternating sliding/full attention works on the same runtime path.
4. Add a production-oriented rounded/bucketized schedule option so FlashMLA can
   be evaluated later without abandoning CARE-E completely.
5. Only after that, optimize backend choice and throughput.
