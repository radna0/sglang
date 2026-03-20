# GPT-OSS-120B CARE Reproduction Run Matrix

This is the paper-faithful first track for GPT-OSS-120B CARE reproduction.

## Principle

Do the smallest faithful reproduction first:

1. single-corpus calibration
2. small covariance set
3. CARE-E rank allocation under a fixed total KV budget
4. export a native MLA checkpoint
5. zero-shot eval
6. long-context sanity

Only after that do we move to the fused-domain extension for production quality.

## Primary run

- calibration corpus: `tatsu-lab/alpaca`
- covariance target: `128` packed sequences
- covariance sequence length: `2048`
- rank policy: `CARE-E`
- nominal rank target: `512`
- global total rank budget: `36 * 512 = 18432`
- minimum per-layer rank: `128`
- rope head dim: `32`
- q/nope head dim: inferred as `32`

Dataset spec:

- [gpt_oss120b_care_repro_alpaca_128x2048.json](/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_repro_alpaca_128x2048.json)

## Secondary run

- calibration corpus: `allenai/c4` (`en`, `train`)
- covariance target: `128` packed sequences
- covariance sequence length: `2048`
- all other settings identical to the primary run

Dataset spec:

- [gpt_oss120b_care_repro_c4_128x2048.json](/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_repro_c4_128x2048.json)

## Why this differs from the fused-domain plan

The paper/rebuttal mostly evaluates single-corpus calibration and then checks robustness across corpora. That is cleaner for first proof. The fused-domain extension remains the better production strategy, but it should come after we establish that GPT-OSS-120B can be converted into a real native MLA checkpoint with CARE-E under a paper-like setup.

## Important note on CARE-E rank shapes

The current reproduction track uses the raw CARE-E water-filling schedule, which can produce
arbitrary integer per-layer ranks. That is acceptable for paper-faithful reproduction, but it is
not yet the production-efficient shape policy.

The schedule rounding pass now exists in the allocator via `--round-multiple`.
Use it when the goal is deployment efficiency rather than raw paper-faithful
reproduction.

Recommended comparison after reproduction:

- raw CARE-E schedule
- rounded CARE-E schedule with `ROUND_MULTIPLE=8`
- rounded CARE-E schedule with `ROUND_MULTIPLE=16`

## Rank ladders to keep after reproduction

For the next systematic rank comparisons, keep both ladders explicit and anchored at
`r = 1024`.

CARE-U / fixed-r ladder:

- `1024`
- `512`
- `448`
- `384`
- `320`
- `256`
- `128`

CARE-E nominal-budget ladder:

- `1024`
- `512`
- `448`
- `384`
- `320`
- `256`
- `128`

Why `1024` stays in both lists:

- it is the cleanest near-original anchor within the MLA-converted family
- it tells us whether the remaining loss is primarily from rank compression or from the
  conversion objective itself
- without the `1024` anchor, `512` can look bad without telling us whether the family is
  salvageable

## Eval targets

Zero-shot eval:

- Wikitext2 PPL
- ARC
- ARC-Easy
- HellaSwag
- PIQA
- MMLU
- OpenBookQA
- RACE
- WinoGrande

Long-context sanity:

- passkey retrieval

## Launcher

Primary or secondary runs use:

- [run_gpt_oss120b_care_repro_zero_shot.sh](/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_repro_zero_shot.sh)
- [run_gpt_oss120b_care_rank_sweep.sh](/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_rank_sweep.sh) for the post-reproduction rank ladders

Examples:

```bash
bash scripts/run_gpt_oss120b_care_repro_zero_shot.sh \
  /workspace/offload_root/gpt-oss-120b \
  /workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca
```

```bash
ROUND_MULTIPLE=8 \
bash scripts/run_gpt_oss120b_care_repro_zero_shot.sh \
  /workspace/offload_root/gpt-oss-120b \
  /workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_r8
```

```bash
DATASET_SPEC_JSON=docs/gpt_oss120b_care_repro_c4_128x2048.json \
bash scripts/run_gpt_oss120b_care_repro_zero_shot.sh \
  /workspace/offload_root/gpt-oss-120b \
  /workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_c4
```

## Next step after reproduction

After Alpaca and C4 reproduction runs:

1. compare zero-shot tables
2. choose the stronger initialization for GPT-OSS-120B
3. move to the fused-domain extension:
   - general
   - Harmony calib packs
   - AIMO3
