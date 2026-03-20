# GPT-OSS-120B MLA Geometry Mismatch

## Current Finding

The current GPT-OSS MLA conversion path is not just underperforming. It is likely
wrong by construction for the `r=1024` anchor.

Original GPT-OSS attention:
- uses standard GQA
- has `num_attention_heads = 64`
- has `num_key_value_heads = 8`
- has `head_dim = 64`
- applies RoPE to the full `64`-dim query/key head

Current GPT-OSS MLA conversion/runtime:
- defaults to `qk_rope_head_dim = 32`
- defaults to `qk_nope_head_dim = 32`
- treats the rope slice as a shared path in the MLA decomposition

This creates two separate mismatches:

1. Some originally-RoPE channels are forced into a non-RoPE path.
2. Some originally head-specific key channels are forced into a shared rope path.

Either one can break parity. Together they make the current `r=1024` anchor
invalid as a quality signal.

## Why `r=1024` Still Fails

The important point is that the low-rank bottleneck is probably not the main
issue at `r=1024`.

For the current `32/32` split:
- unique factorized rows for `noPE + V` are `num_kv_heads * (qk_nope_head_dim + v_head_dim)`
- with GPT-OSS this is `8 * (32 + 64) = 768`

So for `r=1024`:
- the `noPE + V` low-rank part is already large enough to represent the unique
  rows exactly in principle
- remaining loss is therefore dominated by the geometry mismatch, especially the
  shared rope path

This means the current benchmark collapse at `r=1024` is not evidence that
CARE-U itself is hopeless. It is evidence that the current GPT-OSS MLA
architecture or conversion geometry is wrong.

## Geometry Audit Result

Audit artifact:
- [gptoss120b_r1024_geometry_audit.json](/root/sglang-gpt-oss-care-mla/docs/gptoss120b_r1024_geometry_audit.json)

Summary from the audit of original GPT-OSS K weights under the current
converted `32/32` split:

- `mean_rope_shared_rel_frob = 0.9349`
- `max_rope_shared_rel_frob = 0.9423`
- `mean_nope_head_variation_rel_frob = 0.9357`
- `unique_factorized_rows_at_r1024 = 768`
- `r1024_exact_for_nope_plus_value_factorization = true`

Interpretation:

- the current `r=1024` path is not primarily failing because the latent rank is
  too small
- the current path is failing because GPT-OSS head-specific key structure is
  being forced into an MLA geometry that shares the rope path across heads and
  treats some originally-RoPE channels as non-RoPE channels

Alternative full-RoPE audit:
- [gptoss120b_fullrope_geometry_audit.json](/root/sglang-gpt-oss-care-mla/docs/gptoss120b_fullrope_geometry_audit.json)

Result:
- `qk_rope_head_dim = 64`, `qk_nope_head_dim = 0`
- `mean_rope_shared_rel_frob = 0.9355`
- `unique_factorized_rows_at_r1024 = 512`
- `r1024_exact_for_nope_plus_value_factorization = true`

Meaning:

- simply changing the split to full-head RoPE does not solve the core problem
- the deeper problem is the shared rope path itself
- this pushes the likely best fix away from “correct one default” and toward a
  GPT-OSS-native / MHA-mode MLA design

## Trace Comparison Evidence

Forward-trace comparison artifact:
- [gptoss120b_r1024_trace_compare.json](/root/sglang-gpt-oss-care-mla/docs/gptoss120b_r1024_trace_compare.json)

Using a tiny prompt, original GPT-OSS and the current `r=1024` converted
checkpoint already diverge strongly:

- final-logit cosine: `0.5503`
- final-logit KL (teacher -> student): `4.5784`
- final-logit top-k overlap: `0.10`
- layer-0 attention output cosine: negative

This is important because it shows the failure is immediate and structural. It
is not a subtle long-range degradation that only appears in task benchmarks.

## DeepSeek V3.2 Relevance

DeepSeek V3.2 explicitly distinguishes:
- MHA mode of MLA
- MQA mode of MLA

The paper states that DeepSeek-V3.1-Terminus uses:
- MHA mode for training and prefilling
- MQA mode for decoding

That matters here because our current evaluation path is dominated by
full-sequence prefilling and loglikelihood evaluation, not decode-only serving.

So there are two plausible root causes:

1. Wrong RoPE/key geometry inside the current GPT-OSS MLA conversion.
2. Wrong MLA mode for the evaluation setting, meaning GPT-OSS should use an
   MHA-mode MLA design for dense/prefill correctness rather than only the
   current shared-latent style.

## Best Current Architecture Direction

The strongest current direction is:

1. GPT-OSS-native MLA in an MHA-mode representation for zero-shot conversion,
   dense evaluation, and any training-style or prefilling-style correctness
   checks.
2. MQA-mode MLA only as a later serving/decode-oriented form if we can prove a
   faithful transformation from the MHA-mode checkpoint.

This is stricter than the current path because the current path implicitly
evaluates GPT-OSS through a shared-latent MLA shape that is closer to MQA-mode,
while our benchmark suite is mostly full-sequence prefilling / loglikelihood
 evaluation.

So the most likely correct next implementation path is not:
- keep benchmarking the current family

It is:
- define GPT-OSS MLA MHA-mode formally
- convert into that form
- benchmark `r=1024` there first
- only later think about serving-oriented MQA-mode transforms

## Immediate Consequence

Current CARE-U sweep results from this family should be treated as invalid until
the anchor is fixed.

In particular:
- do not draw quality conclusions from current `r=1024`
- do not compare `512` or lower ranks against an invalid anchor

## Next Correct Steps

1. Define the correct GPT-OSS MLA attention geometry formally.
2. Decide whether a corrected split is enough, or whether GPT-OSS needs a
   GPT-OSS-native / MHA-mode MLA variant.
3. Patch converter, HF runtime, and SGLang runtime together.
4. Regenerate a corrected `r=1024` checkpoint from the existing covariance
   artifact.
5. Re-run only the `r=1024` anchor first.
6. Resume the full rank sweep only if `r=1024` is healthy.
