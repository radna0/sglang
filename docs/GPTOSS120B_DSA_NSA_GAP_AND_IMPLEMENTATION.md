# GPT-OSS-120B DSA / NSA Gap And Implementation

## Purpose

This document records what DeepSeek V3.2 DSA actually requires, what SGLang
already has, and what must be added for a GPT-OSS-120B MLA checkpoint to become
GPT-OSS-120B MLA + DSA rather than "just MLA".

The local source references used here are:

- `/root/2512.02556v1.clean.txt`
- `/root/DeepSeek_V3_2.clean.txt`

## Executive conclusion

The best approach is **not** to force GPT-OSS-120B into the existing
`GlmMoeDsaForCausalLM` path by pretending it is a DeepSeek checkpoint.

That would be the wrong abstraction for three reasons:

1. GPT-OSS has deployment semantics that DeepSeek V3.2 does not share:
   - attention sinks
   - alternating sliding-window and full-attention layers
   - GPT-OSS-specific checkpoint and serving conventions
2. DeepSeek V3.2 DSA is **not** just an inference backend switch.
   - it adds a learned lightning indexer
   - it changes which KV entries are attended to
   - it is introduced through continued training
3. The current GPT-OSS objective is quality first.
   - we need a GPT-OSS-native DSA model path that preserves GPT-OSS semantics
   - then we train DSA into that model

So the correct architecture target is:

- `GPT-OSS MLA` first
- then `GPT-OSS MLA + DSA`
- not "rename checkpoint to DeepSeek and hope"

## What the DeepSeek papers actually say

From the local clean text:

- DSA is the only architectural change relative to DeepSeek-V3.1-Terminus.
- DSA consists of:
  - a **lightning indexer**
  - a **fine-grained top-k token selection mechanism**
- DSA is instantiated **under MLA in MQA mode**.
- The model does **continued training**, not static post-hoc conversion.

Key details from the text:

- dense warm-up stage:
  - freeze all model parameters except the indexer
  - align indexer outputs to dense attention
  - KL objective on normalized dense attention scores
  - `1000` steps
  - `16 x 128K` sequences per step
  - total `2.1B` tokens
- sparse training stage:
  - enable top-k token selection
  - optimize all model parameters with language modeling loss
  - continue optimizing indexer with KL alignment
  - select `2048` KV tokens per query
  - `15000` steps
  - `480 x 128K` sequences per step
  - total `943.7B` tokens

That means:

- DSA is not a pure algebraic transform like CARE
- DSA is not "just FlashMLA with a sparse mask"
- DSA requires a learned indexer and a substantial sparse continued-training
  phase

## What SGLang already has for DeepSeek-style DSA

SGLang already contains a substantial DeepSeek-specific NSA/DSA stack.

### Model-side pieces already present

- `deepseek_v2.py`
  - MLA attention can instantiate an `Indexer`
  - when `use_nsa` is enabled, the model has a native sparse-attention path
- `forward_mla.py`
  - the indexer is called during MLA forward
  - `q_lora` is explicitly retained for indexer use

### Backend-side pieces already present

- `nsa_backend.py`
  - full NSA metadata structure
  - indexer metadata integration
  - top-k transform utilities
  - prefill/decode sparse backend selection
- `memory_pool.py`
  - `NSATokenToKVPool`
  - separate storage for indexer KV cache
- `server_args.py`
  - DeepSeek DSA backend selection and defaults

### Practical implication

There is already a real **DeepSeek-specific** DSA serving/training substrate in
this repo. The main GPT-OSS problem is not "invent sparse attention from zero".
It is:

- create a GPT-OSS-native DSA model path
- map GPT-OSS MLA into that path cleanly
- train the indexer and sparse model correctly

## What is incomplete or misleading in the current sparse code

There is also a more generic sparsity framework under
`mem_cache/sparsity/...`, but it is **not** the main production DeepSeek DSA
path today.

Important incomplete pieces there:

- `mem_cache/sparsity/backend/backend_adaptor.py`
  - `NSABackendAdaptor.adapt_for_attn_metadata()` is still TODO
- `mem_cache/sparsity/algorithms/deepseek_nsa.py`
  - representation pool construction/update methods are placeholders
- `mem_cache/sparsity/core/sparse_coordinator.py`
  - multiple request-lifecycle methods are still TODO

So the correct reading is:

- generic sparse-coordinator path: incomplete
- dedicated DeepSeek NSA model/backend path: materially implemented

For GPT-OSS-120B DSA, we should reuse the **dedicated model/backend path**, not
the unfinished generic coordinator as the mainline.

## What a GPT-OSS-120B DSA model must preserve

Our current converted GPT-OSS MLA checkpoints already encode GPT-OSS-specific
semantics:

- native GPT-OSS MLA structure
- fixed-r or dynamic-r latent KV
- attention sinks
- alternating sliding-window and full-attention behavior

Any DSA path that loses those semantics is the wrong target.

So GPT-OSS DSA must preserve:

1. GPT-OSS sink logic
2. GPT-OSS sliding/full layer pattern
3. GPT-OSS checkpoint conventions
4. GPT-OSS-compatible SGLang serving

This is the main reason not to simply reuse `GlmMoeDsaForCausalLM` as-is.

## Best architecture strategy

### Stage 1: finish strong GPT-OSS MLA checkpoints

This remains unchanged:

- finish the current extended fixed-r512 run
- benchmark it
- produce fixed-r1024 anchor
- determine whether covariance-only MLA is sufficient

### Stage 2: add a GPT-OSS-native DSA model family

Add a new GPT-OSS DSA model path rather than masquerading as DeepSeek:

- new model architecture, e.g.:
  - `GptOssDsaForCausalLM`
  - or `GptOssMlaDsaForCausalLM`
- GPT-OSS MLA attention extended with:
  - `Indexer`
  - NSA metadata plumbing
  - DSA backend handler

This path should reuse from DeepSeek:

- `Indexer`
- `NSATokenToKVPool`
- `nsa_backend.py`
- indexer metadata structures
- relevant top-k transform kernels

But it must remain GPT-OSS-native in:

- sinks
- sliding/full layer pattern
- config parsing
- checkpoint loading

### Stage 3: train DSA into the GPT-OSS MLA checkpoint

The correct training flow is:

1. Start from the best GPT-OSS MLA checkpoint.
2. Add randomly initialized or carefully initialized indexer weights.
3. Run dense warm-up:
   - freeze base model
   - train indexer only
   - KL match indexer output to teacher dense attention
4. Run sparse adaptation:
   - enable top-k retrieval
   - train the main model and indexer together
   - LM loss on the sparse model
   - indexer KL alignment retained
5. Run GPT-OSS-targeted post-training if needed:
   - reasoning
   - tool-calling
   - long-context

This is the closest faithful adaptation of DeepSeek V3.2 to GPT-OSS.

## What must be trained versus what can be reused

### Reusable without new training

- the best GPT-OSS MLA checkpoint
- MLA latent projections
- absorbed or non-absorbed serving weights
- current benchmark/eval harnesses
- existing DeepSeek NSA kernel/backend components

### Requires training

- lightning indexer weights
- sparse-adapted main model weights
- any GPT-OSS-specific DSA post-training

### Likely requires new code

- GPT-OSS-native DSA model class
- GPT-OSS-native config recognition
- GPT-OSS-native checkpoint conversion/export for DSA fields
- GPT-OSS-specific NSA serving validation

## Why "just convert to DeepSeek V3.2 checkpoint" is not enough

Even if we were to emit a checkpoint that superficially resembles DeepSeek V3.2:

- the indexer would still be missing or untrained
- sparse top-k behavior would still be unadapted
- GPT-OSS sinks/sliding semantics would still need to be preserved
- quality would still depend on actual sparse continued training

So the conversion target should be:

- **compatible DSA-capable GPT-OSS checkpoint**

not:

- "fake DeepSeek V3.2 checkpoint"

## Immediate implementation tasks

### A. MLA-first tasks

1. Finish the current extended fixed-r512 conversion.
2. Benchmark fixed-r512.
3. Run fixed-r1024 anchor.
4. Decide whether covariance-only MLA is close enough to keep as the DSA base.

### B. GPT-OSS DSA architecture tasks

5. Add a GPT-OSS-native DSA config/architecture.
6. Reuse `Indexer` inside GPT-OSS MLA attention.
7. Reuse NSA KV-pool and backend metadata plumbing.
8. Preserve sinks and sliding/full attention in the GPT-OSS path.
9. Define checkpoint fields for indexer parameters and DSA config.

### C. DSA training tasks

10. Implement dense warm-up for indexer-only training on GPT-OSS MLA.
11. Define the dense-attention teacher target used for KL alignment.
12. Implement sparse adaptation training with top-k retrieval.
13. Keep indexer KL loss separate from model LM loss, matching the paper.
14. Decide the first GPT-OSS DSA sequence length regime:
    - shorter pilot
    - then long-context
15. Decide the first top-k budget, likely starting from the DeepSeek `2048`
    anchor.

### D. Serving tasks

16. Prove GPT-OSS DSA load and construction in SGLang.
17. Prove GPT-OSS DSA decode.
18. Prove GPT-OSS DSA prefill.
19. Reconcile DSA with sinks.
20. Reconcile DSA with alternating sliding/full attention.

## Decision rule

If the extended fixed-r512 and fixed-r1024 MLA runs are both materially weak:

- do not jump straight to DSA training
- first fix the MLA conversion objective
- then build DSA on top of a healthy MLA base

If fixed-r1024 is healthy and fixed-r512 is the main gap:

- the MLA base is usable
- DSA becomes a meaningful next-stage efficiency architecture

## Bottom line

The right order is:

1. get GPT-OSS-120B MLA genuinely healthy
2. add a GPT-OSS-native DSA model path
3. train the indexer and sparse model
4. only then claim GPT-OSS-120B MLA + DSA / NSA

This is slower than pretending a checkpoint rewrite is enough, but it is the
correct path.

