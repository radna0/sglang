# GPT-OSS-120B CARE-U Rank Sweep Benchmark Plan

## Sweep scope

This sweep is for **CARE-U / fixed-rank MLA only**.

Current conversion sweep launcher:

- `/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_u_convert_only_sweep.sh`

Current benchmark sweep launcher:

- `/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_u_benchmark_sweep.sh`

Current target rank ladder:

- `1024`
- `512`
- `448`
- `384`
- `320`
- `256`
- `128`

## Why this order

The first goal is to separate three possibilities:

1. the conversion objective itself is wrong
2. the conversion objective is directionally right, but `r=512` is too small
3. the conversion objective is right and `r=512` is already usable

The `r=1024` checkpoint is therefore the mandatory near-lossless anchor.

## Benchmark policy

All checkpoints must be benchmarked on the **same** tasks and the **same fixed
packs**.

The evaluation path must use the corrected GPT-OSS MLA HF loader, not stock HF
GPT-OSS loading.

## Standard benchmark sweep

Run on every CARE-U checkpoint:

- `arc_easy`
- `arc_challenge`
- `hellaswag`
- `piqa`
- `mmlu`
- `openbookqa`
- `race`
- `winogrande`

These are the current minimum zero-shot comparability tasks.

## Long-context and domain benchmark sweep

Run on every CARE-U checkpoint:

- passkey retrieval:
  - `2048`
  - `4096`
- fixed combined pack PPL:
  - `2048`
  - `8192`
- fixed AIMO3-long pack PPL:
  - `2048`
  - `8192`

The fixed pack roots already exist locally under:

- `/workspace/sglang_gpt_oss_care_runs/fixed_domain_eval_packs_v1`

## Comparison rows

The final side-by-side table must include:

1. original GPT-OSS-120B
2. Alpaca-only CARE-E dynamic-rank
3. Alpaca-only CARE-U fixed-r512
4. extended-data CARE-U fixed-r1024
5. extended-data CARE-U fixed-r512
6. extended-data CARE-U fixed-r448
7. extended-data CARE-U fixed-r384
8. extended-data CARE-U fixed-r320
9. extended-data CARE-U fixed-r256
10. extended-data CARE-U fixed-r128

## Decision rules

### If `r=1024` is close to original

Interpretation:

- the conversion family is viable
- the remaining problem is compression quality at lower ranks

Next move:

- focus on improving `r=512` and `r=256`
- this justifies operator-aware zero-shot if covariance-only CARE-U saturates

### If `r=1024` is still weak

Interpretation:

- the current zero-shot objective is not preserving GPT-OSS behavior well enough

Next move:

- do not keep sweeping lower ranks as if the objective were healthy
- move to the operator-aware zero-shot lane

### Stretch target

The real success condition is:

- `r=512` is close to `r=1024` / original

The research target beyond CARE is:

- `r=256` with the next method should beat vanilla CARE-U / CARE-E at `r=512`

## Output structure

Each rank should have:

- converted checkpoint
- standard task JSON outputs
- passkey JSON
- fixed-pack JSON
- final combined matrix row

## Non-goals for this sweep

- no healing in the mainline
- no CARE-E dynamic-rank serving dependency
- no DSA/NSA training dependency

This sweep exists to answer one clean question:

- how far can fixed-r covariance-based CARE-U get on GPT-OSS-120B before we
  need a stronger zero-shot objective

