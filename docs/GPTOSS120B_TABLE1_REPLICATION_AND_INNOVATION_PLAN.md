# GPT-OSS-120B Table-1 Replication and Innovation Plan

## Why this document exists

The CARE paper's Llama-3.1-8B Table 1 is useful, but it is not our end goal.

For GPT-OSS-120B in 2026, the real target is stronger:

- `r = 1024` should behave as a near-lossless MLA anchor
- `r = 512` should be close to `r = 1024` and close to the original model
- if CARE/CARE-E is still not enough, our next method should push
  `r = 256` high enough to beat vanilla CARE/CARE-E at `r = 512`

That is the concrete quality target for this project.

## The core quality target

We are not trying to win only on "better than TransMLA on an old 8B table."

We are trying to establish, on GPT-OSS-120B:

1. a near-lossless MLA anchor at `r = 1024`
2. a production-worthy MLA checkpoint at `r = 512`
3. an innovation path where `r = 256` is competitive with or better than vanilla
   CARE/CARE-E at `r = 512`

That means the rank ladder is not decorative. It is the central diagnostic:

- if `r = 1024` is already poor, the objective family is wrong
- if `r = 1024` is good but `r = 512` is poor, the main issue is compression
- if `r = 256` with a stronger method beats CARE/CARE-E `r = 512`, we have a
  real methodological advance

## GPT-OSS-120B Table-1 equivalent

The paper-style comparison table we want to maintain for GPT-OSS-120B is:

| Rank | KV Save | Method | Wiki PPL | ARC | ARC-E | HellaSwag | PIQA | MMLU | OBQA | RACE | WinoGrande | AVG |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.00 | Original GPT-OSS-120B | target | target | target | target | target | target | target | target | target | target |
| 1024 | 0.00 | CARE-U | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 1024 | 0.00 | CARE-E | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 512 | 50.00 | CARE-U | partial | partial | partial | partial | pending | partial | pending | pending | pending | partial |
| 512 | 50.00 | CARE-E | partial | partial | partial | partial | pending | partial | pending | pending | pending | partial |
| 256 | 75.00 | CARE-U | future | future | future | future | future | future | future | future | future | future |
| 256 | 75.00 | CARE-E | future | future | future | future | future | future | future | future | future | future |
| 256 | 75.00 | Operator-aware zero-shot (future) | future | future | future | future | future | future | future | future | future | future |

Notes:

- `CARE-U` here means covariance-aware conversion with a uniform per-layer rank.
- `CARE-E` here means covariance-aware conversion with non-uniform rank allocation
  under a fixed total budget.
- the `Operator-aware zero-shot` row is the planned next family if covariance-only
  CARE still misses materially.
- if we later run TransMLA on GPT-OSS-120B, it can be inserted into the same table,
  but it is not required for the logic of the current project.

## Current known GPT-OSS-120B signals

From the Alpaca `128 x 2048` reproduction family:

| Rank | Method | Task | Metric | Value |
| --- | --- | --- | --- | ---: |
| 512 | CARE-E | ARC-Easy | acc_norm | `0.258838` |
| 512 | CARE-U | ARC-Easy | acc_norm | `0.259680` |
| 512 | CARE-E | HellaSwag | acc_norm | `0.260108` |
| 512 | CARE-U | HellaSwag | acc_norm | `0.262796` |
| 512 | CARE-E | MMLU | acc | `0.256231` |
| 512 | CARE-U | MMLU | acc | `0.248398` |

Those numbers already establish one hard fact:

- Alpaca-only `r = 512` CARE-U / CARE-E is structurally valid
- but it is nowhere near the original GPT-OSS-120B baseline

That is why the current big extended-data `r = 512` run matters.

## Why CARE healing is still not good enough for our goal

Two things need to be stated plainly.

### 1. The strongest healed story is still not our target story

The CARE paper's healing narrative is:

- better zero-shot initialization than older baselines
- then recover with a modest post-conversion SFT budget

That is an improvement over weaker baselines, but it is still not the target we
want for GPT-OSS-120B production MLA conversion.

Our target is stricter:

- zero-shot should already be strong
- `r = 1024` should be near-lossless
- `r = 512` should be close to `r = 1024`
- healing should be a fallback, not the main proof that the conversion works

### 2. "Tiny" is being used in a relative, not operational, sense

The paper text uses the phrase:

> "Using a TINY SFT corpus of 2.5B tokens"

And the rebuttal later reframes this as:

> "extremely small data budgets (e.g., 1B-3B tokens)"

For an actual production conversion pipeline, those budgets are not tiny in any
operational sense. They are only "tiny" relative to full pretraining.

For an 8B model, `1B-3B` healing tokens is already substantial.

For GPT-OSS-120B, a method that still depends on a multi-billion-token healing
budget to become usable is not meeting the standard we want, even if it is
better than prior MLA baselines.

### What the paper actually supports

To be precise and fair:

- the rebuttal healing tables are not only Llama-3.1-8B; they also include
  Qwen3-4B
- the broader rebuttal claims additional evaluation on larger models, including
  Qwen3-30B-A3B and Llama-70B

But that still does not change the central criticism:

- the healed evidence is still built around billion-token-scale repair budgets
- the main quality story is still "zero-shot is lossy, healing closes the gap"

That is better than TransMLA-era weight fitting, but still not enough for the
GPT-OSS-120B objective here.

### Design consequence for this project

So the project rule stays:

- do not let healing define success
- first demand a strong zero-shot checkpoint
- if fixed-r512 extended CARE still misses badly, move to a stronger zero-shot
  objective rather than accepting large healing budgets as the answer

That is the difference between:

- "better than old MLA conversion baselines"
- and "good enough for a serious GPT-OSS-120B production conversion pipeline"

## What the current big run is testing

The current live run is:

- GPT-OSS-120B
- fixed `r = 512`
- zero-shot only
- covariance-aware CARE-U
- extended corpus, about `1.4336B` calibration tokens

This is the strongest test so far of the proposition:

> Can a much stronger covariance estimate rescue zero-shot `r = 512` MLA quality
> for GPT-OSS-120B without healing?

If it fails, the conclusion is not "MLA is impossible."

The conclusion is:

- covariance-only CARE-U at `r = 512` is not enough for GPT-OSS-120B

## What "success" means now

### Minimum success

- `r = 1024` must be near-lossless relative to the original GPT-OSS-120B
- `r = 512` must materially improve over the current Alpaca-only checkpoints
- long-context AIMO3 slices must not collapse

### Real success

- `r = 512` is close to `r = 1024`
- `r = 512` is operationally close to the original model on generic and target tasks
- SGLang serving for the checkpoint family is realistic

### Stretch success

- a new operator-aware zero-shot method at `r = 256` beats vanilla CARE/CARE-E
  at `r = 512`

That would be a genuine methodological win, not just a benchmark cleanup.

## Why `r = 1024` is mandatory

`r = 1024` is not optional. It is the first anchor we need after the current run.

Reason:

- it is the closest MLA-converted approximation to the original model within the
  same conversion family
- it tells us whether the problem is:
  - the rank bottleneck, or
  - the objective family itself

Interpretation:

- if `r = 1024` is bad, then even the "easy" compression case is failing, so we
  need a stronger objective, not just more rank tuning
- if `r = 1024` is good and `r = 512` is bad, the main problem is compression
  under the current objective family

## The innovation target beyond CARE / CARE-E

The next method should still be zero-shot first.

We do not want to jump immediately to blind healing. We want a stronger
mathematical objective.

The leading candidate is:

- keep CARE-style covariance initialization
- add operator-aware statistics:
  - query-aware key weighting
  - attention-output-aware value weighting
  - token-logit diagnostics as the truth criterion

In other words:

- CARE fixes weight-space thinking by moving to activation-space
- the next step is to move from activation-space toward operator-space

This is the most credible way to make:

- `r = 512` approach `r = 1024`
- and possibly make `r = 256` beat vanilla CARE/CARE-E `r = 512`

## Next concrete comparison order

1. Finish the current extended-data fixed `r = 512` run.
2. Benchmark it cleanly.
3. Run the same extended-data lane at `r = 1024`.
4. Compare `1024` vs `512`.
5. If `1024` is healthy but `512` is still weak, then try:
   - rounded fixed `r = 512`
   - then operator-aware zero-shot
6. Only after that decide whether dynamic CARE-E should re-enter as the main branch.

## Current live conversion status

Live run root:

- `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210`

Live log:

- `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210/conversion/pipeline.log`

This run should be treated as the decisive covariance-only `r = 512` test before we
move to the `r = 1024` anchor and then to operator-aware zero-shot.
