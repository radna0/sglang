# GPT-OSS-120B Beyond CARE-E: Operator-Aware Fallbacks

## Why this document exists

The first paper-faithful GPT-OSS-120B CARE reproduction established an important
fact:

- structural native MLA conversion works
- zero-shot parity does not follow automatically

The full formal problem statement and objective hierarchy now live in:

- `docs/GPTOSS120B_ZERO_SHOT_MLA_FORMALIZATION.md`

This means the next big run should still prioritize zero-shot conversion, but we
also need a formal fallback path if extended CARE-E remains too weak.

The right fallback is not "blind healing later". It is to make the zero-shot
objective more faithful to the operator we actually care about.

## Current mainline

The mainline remains:

1. stronger multi-source covariance
2. CARE-E water-filling under a fixed KV budget
3. optional production rounding to multiples of 8 or 16
4. direct zero-shot evaluation

This is still the right next experiment because it keeps the conversion simple,
reproducible, and deployment-shaped.

## If extended CARE-E still fails

If the extended zero-shot run still collapses on the corrected benchmark path,
the next best innovations are:

### 1. Attention-logit projection objective

Instead of allocating rank only by covariance-weighted singular energy, estimate
how much error each extra latent channel removes in the actual attention
operator:

- sample windows from the deployment corpus
- compare teacher vs student `QK^T / sqrt(d)` on those windows
- allocate rank where an extra block reduces logit error most

This directly targets attention-map fidelity rather than only covariance
coverage.

### 2. Softmax/output operator objective

Go one step beyond logits and measure:

- attention-probability KL
- attention-output error `||A_teacher V_teacher - A_student V_student||`

This is slower than covariance-only CARE-E, but it is still zero-shot if we use
it only to choose subspaces and ranks rather than training parameters.

### 3. Final-logit preservation objective

Some layers may look acceptable in local KV space but still produce bad final
token distributions. We should add a diagnostic lane that measures:

- final-logit KL
- top-k agreement
- token rank displacement

on fixed samples from:

- AIMO3
- code
- math
- calib packs

This gives a direct answer to whether a proposed schedule is preserving actual
model behavior.

### 4. Hybrid rank allocation

The likely best practical extension is:

- start from CARE-E covariance gains
- then reweight difficult layers by operator error on sampled windows

This keeps CARE-E's efficiency while focusing extra budget on the layers that
actually break behavior.

### 5. Decoupled-RoPE stress test

If failure is concentrated on long-context retrieval or tool-calling prompts,
the next zero-shot diagnostic is not general healing. It is to test whether the
error localizes to RoPE-bearing channels:

- compare short vs long context operator error
- compare sliding vs full-attention layers
- quantify whether extra decoupled-RoPE channels fix the long-context slices

## Concrete next implementation steps

If extended CARE-E still misses parity, the next implementation lane should be:

1. add an operator-diagnostic script that compares original vs converted model
   on fixed prompts
2. record per-layer:
   - logit error
   - attention-map error on sampled windows
   - output-vector error
3. derive a hybrid allocation score:
   - covariance gain
   - plus operator-error reduction per extra rank block
4. rerun zero-shot conversion with:
   - multi-source covariance
   - rounded schedule
   - operator-aware reweighting

The first diagnostic building block is now implemented at:

- `scripts/measure_gpt_oss_logit_projection.py`

It measures, on fixed JSONL packs:

- teacher -> student next-token KL
- top-1 agreement
- top-k overlap
- teacher-token NLL under the student

## Decision rule

Use vanilla extended CARE-E first.

Only move to operator-aware allocation if:

- the corrected benchmark table still shows catastrophic loss relative to the
  original model, or
- long-context/tool/code slices diverge much more than generic MCQ slices

That keeps the experimentation disciplined:

- first prove the strongest pure CARE-E run
- then escalate to operator-aware zero-shot conversion only if needed
