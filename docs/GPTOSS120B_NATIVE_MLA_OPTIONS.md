# GPT-OSS-120B Native MLA Options

## Problem

GPT-OSS attention is not naturally aligned with the current DeepSeek-style MLA
assumption we used so far.

Original GPT-OSS:
- GQA with `64` query heads and `8` KV heads
- full-head RoPE on both Q and K
- alternating sliding/full attention
- attention sinks

Current converted GPT-OSS MLA:
- assumes a DeepSeek-style split into `qk_nope_head_dim` and `qk_rope_head_dim`
- uses a shared rope key path
- behaves like a shared-latent MLA family

The current `r=1024` failure indicates this is not a small tuning mistake.

## Option A: Keep the Current DeepSeek-Style GPT-OSS MLA Path

Description:
- retain the current shared rope path
- tune `qk_rope_head_dim`
- tune `qk_nope_head_dim`
- hope a corrected split is enough

Verdict:
- rejected as the mainline path

Reason:
- the geometry audit shows that even full-head RoPE with `qk_rope_head_dim=64`
  still leaves very high shared-rope loss
- this means the core mismatch is not only the `32/32` split

## Option B: GPT-OSS-Native MHA-Mode MLA

Description:
- build a GPT-OSS-native MLA variant that matches dense/prefill semantics
- keep head-specific structure in the key path instead of forcing a single
  shared rope key
- target correctness first for full-sequence loglikelihood and dense evaluation

Why it is attractive:
- DeepSeek V3.2 explicitly distinguishes MHA-mode MLA from MQA-mode MLA
- DeepSeek itself uses MHA-mode for training and prefilling
- our benchmark suite is dominated by dense/prefill behavior, not decode-only

Implication:
- benchmark correctness should be established on MHA-mode MLA first
- decode/serving optimizations can come later

## Option C: Hybrid Anchor Path (Preserve K Semantics, Compress V Aggressively)

Description:
- keep GPT-OSS key geometry much closer to original GQA
- compress V first and only compress K where we can prove correctness
- use this as a near-lossless anchor family

Why it is attractive:
- quality-first
- easier to establish whether the problem is really K-side geometry
- may yield a trustworthy near-original checkpoint earlier than a fully native
  GPT-OSS MLA redesign

Limitation:
- weaker alignment with the long-term “fully native MLA + DSA” objective

## Option D: Jump Straight to MQA-Mode / DSA-Oriented MLA

Description:
- optimize directly for the eventual shared-entry sparse-attention regime
- force GPT-OSS into a DSA-ready shared-latent representation

Verdict:
- rejected for the current stage

Reason:
- quality anchor is not established
- this path is farther from original GPT-OSS semantics
- it is the wrong place to debug correctness

## Best Current Approach

The best current approach is:

1. establish a trustworthy GPT-OSS-native dense/prefill MLA path first
2. prefer an MHA-mode MLA family for that anchor
3. only after the anchor is healthy, study the transformation to a serving or
   DSA-oriented MQA-mode family

In practical terms:

- do not keep benchmarking the current shared-rope family
- do not return to CARE-E yet
- do not use healing to cover up a broken anchor

## Immediate Implementation Direction

1. Define the GPT-OSS-native MHA-mode MLA key/value cache layout.
2. Define how RoPE is represented in that layout without forcing a shared rope
   key path across all heads.
3. Patch converter, HF runtime, and SGLang runtime together.
4. Regenerate `r=1024`.
5. Benchmark only the corrected `r=1024` anchor first.
