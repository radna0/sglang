# GPT-OSS-120B CARE / CARE-E Paper Gap Matrix

## Why this exists

The current fixed-r512 extended conversion run is the strongest pure
covariance-based zero-shot GPT-OSS-120B MLA conversion we have attempted so far.

Before deciding what to do next, we need a precise answer to five questions:

1. What does the CARE paper actually assume?
2. Which of those assumptions are we still violating?
3. Which parts of our implementation are paper-faithful?
4. Which parts are deliberate extensions beyond the paper?
5. If the current run fails, does that falsify CARE, or only our current
   covariance-only variant?

This document answers those questions.

## Executive summary

### What the CARE paper most strongly supports

- activation-aware factorization is better than weight-only SVD
- adaptive rank allocation matters
- small calibration sets can already produce useful zero-shot results
- long-context behavior improves when calibration sequence length increases
- healing is presented as the practical way to close the residual gap

### What our current mainline run is

- GPT-OSS-120B
- fixed `r = 512`
- zero-shot only
- no healing in the primary path
- multi-domain extended calibration corpus
- much larger than the paper-faithful baseline

So the current run is:

- faithful to the CARE philosophy of covariance-aware zero-shot conversion
- **not** a strict paper-faithful CARE-E reproduction

It is a stronger GPT-OSS-targeted zero-shot experiment, but it intentionally
deviates from the paper in several important ways.

## Appendix assumptions and hyperparameters from the paper

From Appendix D / supplementary text:

- covariance samples: `128`
- covariance sequence length: `2048`
- calibration datasets listed: `C4 / PTB / WikiText / Alpaca`
- default MLA rank setting shown in appendix block: `384`
- min rank: `64`
- max rank: `1024`
- uniform allocation is shown in the appendix configuration block
- precision in the appendix block: `float16`
- healing framework: `axolotl`
- healing LR chosen: `2e-6`
- healing warmup: `100` steps
- healing max sequence length: `512`
- healing steps: `10000`
- healing precision: `bfloat16`

Important rebuttal clarifications:

- `alpaca-256-32` appears repeatedly in the rebuttal tables as the practical
  calibration label
- the rebuttal also reports long-context improvements from longer calibration
  sequence length, e.g. `Alpaca-2048` vs `Alpaca-256`
- healing budgets are discussed as roughly `1B–3B` tokens, with comparisons up
  to `6B`

## Discussion / rebuttal caveats most relevant to GPT-OSS-120B

The rebuttal contains several caveats that matter directly for our setting.

### 1. Domain shift is real

The authors explicitly acknowledge that calibration-set mismatch can hurt
zero-shot performance:

- text-only calibration for code-heavy deployment can degrade results
- covariance-based rank allocation remains relatively stable, but performance can
  still fluctuate

This matters for GPT-OSS-120B because our target distribution is not generic
prose. It is:

- code
- math
- tool calling
- long-context reasoning

### 2. Task-weighted covariance is a legitimate extension

The rebuttal explicitly says task-weighted calibration mixtures are a valid
extension, for example mixing code and math corpora with explicit weights.

This directly supports our multi-source weighted calibration direction.

### 3. Long-context quality depends on calibration sequence length

The rebuttal explicitly states that longer calibration sequences produce richer
covariance statistics and improve long-context behavior.

This means:

- `2048` is safer than very short calibration windows
- but it also means our future `AIMO3 65k` lane is not conceptually crazy; it is
  an extrapolation of a paper-supported direction

### 4. The paper still treats healing as the practical closure step

The authors are explicit that:

- zero-shot conversion is lossy
- for practical deployment they recommend offline healing with modest
  domain-representative data

This is important because our current mainline is intentionally stricter than
the paper:

- we want the strongest zero-shot checkpoint before accepting healing

### 5. The paper is not fully validated on GPT-OSS-like deployment semantics

Even after rebuttal expansion, CARE is still not validated on:

- GPT-OSS attention sinks
- alternating sliding-window + full-attention semantics
- GPT-OSS-120B specifically
- GPT-OSS-style serving constraints in SGLang

So a failure on GPT-OSS-120B would not mean CARE is false. It would mean CARE is
not yet proven sufficient for this specific deployment target.

### 6. Bigger calibration is not automatically better

The rebuttal explicitly says two things that matter for our current big run:

- increasing sample count yields only marginal gains after a point
- excessively long or over-specialized calibration can overfit the calibration
  distribution

That matters because our current run is not just "larger than the paper." It is
orders of magnitude larger. So if this run underperforms, one possible
interpretation is not "we still need even more data." It may instead mean:

- covariance-only fitting has saturated, or
- the calibration mixture is now too deployment-specific in the wrong way, or
- zero-shot needs a stronger objective than covariance alone

### 7. CARE itself does not claim zero-shot parity is sufficient for deployment

The rebuttal is unambiguous on this point:

- CARE without healing is lossy
- offline healing is the recommended deployment closure step

Our current mainline is therefore stricter than the paper:

- we want a strong zero-shot checkpoint before accepting healing

This is a valid research target, but it should not be confused with "strict
paper compliance." If pure zero-shot fails on GPT-OSS-120B, that is not a
contradiction of the paper's own claims.

## Zero-shot assumptions we may still be violating

These are the most plausible paper-level mismatches still in play.

| Assumption / paper tendency | Current state | Likely impact |
| --- | --- | --- |
| small calibration sets are sufficient | current run is 5468.76x larger than the Alpaca-128x2048 reproduction | may help domain coverage, but may also reduce faithfulness to the paper and can overfit the calibration mix |
| single-corpus or lightly varied calibration is the canonical baseline | current run is a five-source weighted mixture | stronger target alignment, but it changes the experimental question |
| zero-shot is evaluated before healing, but healing is still the deployment story | we are intentionally trying to make zero-shot itself good enough | stricter than the paper; failure here does not falsify CARE |
| dense / simpler attention semantics dominate the evidence base | GPT-OSS-120B is MoE with sinks and alternating sliding/full attention | paper evidence is directionally relevant but not sufficient |
| K/V allocation can be reasoned over multiple spectra | current GPT-OSS path still uses a shared per-layer `kv_lora_rank` in the mainline | may hide real K/V asymmetry and leave quality on the table |
| positional fixes are orthogonal and optional | current fixed-r512 mainline has no zero-shot positional correction | may disproportionately hurt long-context slices |

## Gap table: paper vs our current implementation

| Area | CARE / CARE-E paper | Our current state | Status |
| --- | --- | --- | --- |
| Activation-aware decomposition | Core method; covariance-aware factorization | Implemented | `implemented` |
| Adaptive rank allocation | Core CARE-E method via water-filling | Implemented in code, but current mainline run is fixed-r512 and intentionally bypasses it | `implemented but not active in current run` |
| Fixed-r covariance conversion | Used in ablations / comparison lanes | Implemented and active | `implemented` |
| Small paper-faithful calibration | `128 x 2048` style single-corpus setup | Implemented and already run on Alpaca | `implemented` |
| Large multi-domain zero-shot calibration | Suggested by rebuttal as valid extension, not the canonical paper setup | Implemented and active in current run | `extension` |
| Weighted per-source covariance fusion | Suggested in rebuttal as a promising extension | Implemented | `extension` |
| Separate K/V-aware allocation | Paper/rebuttal language suggests K and V are not fully symmetric and water-filling can be viewed across multiple spectral lists | Our current GPT-OSS allocator produces one shared per-layer `kv_lora_rank`, not separate K-rank and V-rank schedules | `missing / simplified` |
| Dynamic-rank serving | Paper discusses non-uniform rank as meaningful | Partially implemented in SGLang; not fully proven for GPT-OSS dynamic-rank + sinks + sliding window | `partial` |
| RoPE / decoupled-RoPE | Paper adds a decoupled RoPE channel during healing | Optional converter/export/healing support exists, but current fixed-r512 mainline does not use it | `implemented but inactive in current run` |
| RoRoPE-style positional alignment | Paper says CARE is orthogonal to RoRoPE and does not require it | We do not have a true zero-shot RoRoPE-style alignment path in the mainline | `missing relative to TransMLA-style positional handling` |
| Absorb / 100% MLA compatibility | Rebuttal claims CARE can be extended to Absorb-like full MLA restoration | We have absorbed export paths and partial serving work, but not yet end-to-end proven on GPT-OSS production semantics | `partial` |
| Healing | Paper treats brief healing as the practical closure step | Implemented, but deliberately not in the current primary path | `implemented but intentionally deferred` |
| Long-context evaluation | Rebuttal adds Needle-style results to 32K | We have passkey/fixed-pack/AIMO evaluation lanes, but not a final parity result yet | `partial` |
| MoE validation | Rebuttal adds Qwen3-30B-A3B | We are on GPT-OSS-120B MoE, which is materially larger and more serving-constrained | `paper partially relevant, not fully validated for our case` |

## Where CARE still implicitly assumes an easier world than GPT-OSS-120B

The paper becomes broader in the rebuttal, but several implicit assumptions
still remain easier than our target.

### 1. No sinks / sliding-window hybrid semantics

CARE discusses MLA conversion under KV parity, but not:

- attention sinks
- alternating sliding-window and full-attention patterns

GPT-OSS depends on both.

### 2. Simpler serving story

The paper is mostly about:

- conversion quality
- recovery quality
- conversion-time practicality

It is not about proving that the resulting checkpoint can run through the exact
GPT-OSS serving semantics we need in SGLang.

### 3. MoE scale mismatch

The rebuttal includes Qwen3-30B-A3B and 70B dense. That is useful, but still
not the same as:

- GPT-OSS-120B
- larger MoE
- tool-calling oriented
- long-context plus sinks/sliding serving constraints

So MoE evidence exists, but not at our full target difficulty.

## K versus V asymmetry: are we missing something?

Yes, at least partially.

The paper’s conceptual story already implies asymmetry:

- K errors damage routing / logits
- V errors damage transported content / outputs

The rebuttal and appendix material also speak in ways that suggest K and V can
be treated separately:

- separate K/V projection ranks are mentioned in appendix-style settings
- water-filling is described over multiple spectral lists

Our current GPT-OSS implementation is simpler:

- one shared latent rank per layer (`kv_lora_rank`)
- one shared per-layer schedule in CARE-E mode

That is a meaningful simplification.

It is not necessarily wrong, but it means our current implementation is not the
strongest version of the CARE idea if K/V asymmetry matters materially for
GPT-OSS-120B.

The practical implication is:

- if this fixed-r512 extended run still fails badly, one of the strongest
  remaining pure-CARE upgrades is not "more data"
- it is a more faithful K/V-aware allocation or objective

## RoPE / decoupled-RoPE / zero-shot positional handling

Important distinction:

- CARE basic decomposition is not a RoRoPE method
- CARE’s rebuttal says RoRoPE is orthogonal
- CARE adds the decoupled-RoPE channel in the healing story, not as the main
  zero-shot conversion core

Our current state:

- decoupled-RoPE support exists in the converter/export/healing path
- the current fixed-r512 extended run does **not** use it

That means if the current run fails mainly on long-context positional tasks, the
current failure may reflect:

- covariance-only conversion limits
- or missing positional correction
- or both

So a bad result on AIMO3 long-context would not automatically prove that
covariance-based zero-shot is fundamentally wrong. It may mean we need a
zero-shot positional-aware extension.

## Calibration corpus: paper-faithful vs our current setup

### Paper-faithful baseline

What we already ran:

- Alpaca single-corpus reproduction
- `128 x 2048`

This is the cleanest paper-faithful baseline.

### Current big run

Current run:

- `general_c4`
- `calib_packs`
- `aimo3`
- `livecodebench`
- `math500`

With:

- `310,808` source rows
- `700,001` packed sequences
- `2048` tokens each
- about `1.4336B` calibration tokens

Compared to the Alpaca `128 x 2048` reproduction, this is:

- `5468.76x` larger in packed sequences
- `5468.76x` larger in token volume

So the current run is **not** paper-faithful calibration. It is a deliberate
production-oriented extension.

The most important interpretation point is:

- the current run tests whether a much stronger and more deployment-shaped
  covariance estimate rescues zero-shot quality
- it does not test whether we reproduced the paper exactly

## Formalization doc: paper-faithful vs extension

The document:

- [GPTOSS120B_ZERO_SHOT_MLA_FORMALIZATION.md](./GPTOSS120B_ZERO_SHOT_MLA_FORMALIZATION.md)

contains both paper-faithful and extended material.

### Paper-faithful portions

- the hierarchy from weight-space to activation-space
- the claim that activation-aware compression is better than weight-only SVD
- the recognition that preserving behavior rather than only weights is the real
  objective

### Our extensions beyond CARE

- explicit attention-logit objective as the next zero-shot target
- explicit attention-output objective as the next zero-shot target
- token-logit diagnostics as the truth criterion for zero-shot quality
- query-aware key weighting
- output-aware value weighting
- Hutchinson / sketch-based operator preservation as a non-brute-force path

Those are not CARE as published. They are the proposed next step if CARE-style
covariance-only conversion remains insufficient.

## What this current run is actually testing

The current run tests this question:

> If we keep the MLA target architecture fixed at `r=512`, and dramatically
> strengthen the calibration distribution while remaining zero-shot and
> covariance-based, can GPT-OSS-120B produce a materially better MLA checkpoint
> than the paper-faithful Alpaca-only runs?

That is a strong and worthwhile question.

But it is **not** the final question.

If this run fails, the conclusion is:

- fixed-r covariance-only zero-shot on GPT-OSS-120B is not enough

not:

- native GPT-OSS-120B MLA is impossible

## Decision rule after the current run

If the current fixed-r512 extended run:

- materially improves generic MCQ and PPL retention
- and materially improves AIMO3 / long-context pack behavior

then the next pure-CARE step should be:

- compare fixed-r512 vs rounded fixed-r512
- then decide whether returning to CARE-E is worthwhile

If the current run still misses parity materially, then the next best step is
**not** healing first.

It is:

1. operator-aware zero-shot diagnostics
2. query-aware key weighting
3. output-aware value weighting
4. token-logit-grounded evaluation

## Provisional recommendation for the next pure-CARE experiment

If we stay strictly inside the CARE family after this run, the strongest next
experiment should be:

1. add fixed `r = 1024` as the next near-original anchor
2. keep the extended corpus
3. compare `r = 1024` vs `r = 512` before changing the objective family
4. add deployment-friendly rank rounding only if it does not alter total budget
   materially
5. do **not** jump back to dynamic CARE-E first unless the fixed-r branch is at
   least directionally healthy

Reason:

- without `r = 1024`, a poor `r = 512` result is underidentified
- a bad fixed-r1024 and fixed-r512 pair means we still have not established a decent
  covariance-only checkpoint
- jumping back to CARE-E too early would change two things at once: objective
  quality and rank schedule heterogeneity
- rounded fixed-r512 is the cleanest next pure-CARE serving-shaped comparison

So the priority order is:

1. current fixed-r512 extended run
2. fixed-r1024 on the same extended corpus
3. if close to usable, rounded fixed-r512
4. only then reconsider CARE-E as the next pure-CARE branch
5. if not close to usable, leave pure CARE and move to operator-aware zero-shot

In other words:

- first exhaust the strongest zero-shot mathematical path
- only then reopen healing as a practical fallback

## Bottom line

The paper is highly relevant, but our current GPT-OSS-120B project is already
beyond a strict CARE reproduction.

The current state is:

- CARE philosophy: yes
- paper-faithful baseline: already done
- strongest covariance-only zero-shot extension: currently running
- full GPT-OSS-120B operator-aware zero-shot method: not yet implemented

That means the current run is the decisive test of the covariance-only branch,
not the decisive test of MLA conversion as a whole.
