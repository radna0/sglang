# GPT-OSS-120B CARE Benchmark Status (2026-03-14)

## Scope

This document records the current benchmark state for:

- Original `gpt-oss-120b`
- CARE-converted GPT-OSS-120B checkpoint from the Alpaca `128 x 2048` zero-shot reproduction run

The paper-style GPT-OSS-120B comparison target and innovation ladder now live in:

- `/root/sglang-gpt-oss-care-mla/docs/GPTOSS120B_TABLE1_REPLICATION_AND_INNOVATION_PLAN.md`

## Critical Caveat

The converted-checkpoint HF loader bug is fixed in code, and the converted checkpoint itself has now been repaired into a clean benchmark artifact with the dense `k_proj` / `v_proj` index entries removed.

What changed:

- converted checkpoints now install a local HF MLA model class at `modeling_gpt_oss_mla.py`
- the benchmark scripts automatically detect converted GPT-OSS MLA checkpoints, install the local loader package, and enable `trust_remote_code=True`
- the corrected loader resolves the converted checkpoint to `GptOssMlaForCausalLM` with `GptOssMlaAttention`, including per-layer `kv_lora_rank_per_layer`
- the HF loader now **fails fast** if a GPT-OSS MLA checkpoint still advertises dense `k_proj` / `v_proj` tensors in its safetensors index
- the repaired converted checkpoint path is:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/converted_checkpoint_clean`
- future converted HF benchmark runs must use the repaired checkpoint, not the original mixed artifact at `.../conversion/converted_checkpoint`

What this means:

- the **original-model HF baseline numbers below are valid**
- any converted-model number produced against the old mixed checkpoint is **stale for comparison**
- converted-model numbers produced against `converted_checkpoint_clean` are the new source of truth
- the converted-model side now needs clean reruns, not another loader patch

Additional current blocker:

- the current GPT-OSS MLA conversion family is now also known to have an
  attention-geometry problem, not just a historical loader problem
- specifically, original GPT-OSS uses full-head RoPE, while the current GPT-OSS
  MLA conversion defaults to a DeepSeek-style `32 rope / 32 no-PE` split
- this invalidates the current CARE-U `r=1024` anchor as a quality signal
- see:
  `/root/sglang-gpt-oss-care-mla/docs/GPTOSS120B_MLA_GEOMETRY_MISMATCH.md`

Operational consequence:

- the current CARE-U sweep results should be treated as **invalid pending
  geometry correction**
- do not compare `512` or lower ranks against the current `1024` row

## Current Active Runs

Fixed-r CARE comparison lane:

- fixed-r512 checkpoint:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/converted_checkpoint_fixed_r512`
- next fixed-r anchor to add:
  - `CARE-U / fixed-r1024`
  - reason: `r = 1024` is the near-original anchor needed to interpret the `512` gap
- comparison doc:
  `/root/sglang-gpt-oss-care-mla/docs/GPTOSS120B_CARE_FIXEDR512_VS_CAREE.md`
- first trustworthy fixed-r512 rerun completed on true `DP=8`:
- fixed-r512 comparison summary doc:
  `/root/sglang-gpt-oss-care-mla/docs/GPTOSS120B_CARE_FIXEDR512_VS_CAREE.md`
- fixed-r512 trustworthy reruns completed on true `DP=8`:
  - log:
    `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/arc_easy_fixedr512_clean_dp8.log`
  - json:
    `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/arc_easy_fixedr512_clean_dp8.json`
  - result:
    - `acc = 0.24621212121212122`
    - `acc_norm = 0.2596801346801347`
  - additional comparison results:
    - `hellaswag acc_norm = 0.262796255725951`
    - `mmlu acc = 0.24839766415040593`
- takeaway:
  - fixed-r512 is only marginally above CARE-E on `arc_easy` and `hellaswag`, and
    below CARE-E on `mmlu`
  - that means uniform `r=512` on Alpaca alone does not materially improve parity over
    the current CARE-E checkpoint family

Original-model long tasks are being relaunched on true `DP=8` under `accelerate`:

- wrapper log:
  `/root/sglang-gpt-oss-care-mla/logs/hf_eval/original_baseline_dp8_sharded_20260314_122022/tasks/dp8_hellaswag_mmlu.log`
- task outputs:
  - `/root/sglang-gpt-oss-care-mla/logs/hf_eval/original_baseline_dp8_sharded_20260314_122022/tasks/hellaswag_dp8.json`
  - `/root/sglang-gpt-oss-care-mla/logs/hf_eval/original_baseline_dp8_sharded_20260314_122022/tasks/mmlu_dp8.json`

Loader-fix verification run completed on the converted checkpoint before the checkpoint-index repair:

- log:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/arc_easy_loaderfix_smoke.log`
- json:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/arc_easy_loaderfix_smoke.json`
- result:
  - `acc = 0.2`
  - `acc_norm = 0.2`
  - `limit = 5`

This is only a smoke, not a reportable benchmark number, but it confirmed the corrected HF MLA path could load and score the converted checkpoint end to end.

First full corrected converted-task rerun completed on true `DP=8` before the checkpoint-index repair:

- launcher:
  `/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_hf_lm_eval_accelerate.sh`
- log:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/arc_easy_loaderfix_dp8_v2.log`
- json:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/arc_easy_loaderfix_dp8_v2.json`
- result:
  - `acc = 0.23947811447811448`
  - `acc_norm = 0.2588383838383838`

Current repaired-checkpoint reruns on true `DP=8`:

- log:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/arc_easy_loaderfix_clean_dp8.log`
- json:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/arc_easy_loaderfix_clean_dp8.json`
- status:
  - completed
  - uses `converted_checkpoint_clean`
  - result matches the mixed-checkpoint rerun exactly:
    - `acc = 0.23947811447811448`
    - `acc_norm = 0.2588383838383838`

Second full corrected converted-task rerun completed on true `DP=8`:

- log:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/hellaswag_loaderfix_clean_dp8.log`
- json:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/hellaswag_loaderfix_clean_dp8.json`
- result:
  - `acc = 0.2568213503286198`
  - `acc_norm = 0.26010754829715194`

Third full corrected converted-task rerun completed on true `DP=8`:

- log:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/mmlu_loaderfix_clean_dp8.log`
- json:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/tasks/mmlu_loaderfix_clean_dp8.json`
- status:
  - completed
  - uses `converted_checkpoint_clean`
- result:
  - `acc = 0.25623130608175476`

## Table 1: Original GPT-OSS-120B HF Baseline

| Benchmark | Metric | Value | Status | Source |
| --- | --- | ---: | --- | --- |
| WikiText2 | word_perplexity | 44.6211 | valid | `.../ppl/wikitext.json` |
| WikiText2 | byte_perplexity | 2.0346 | valid | `.../ppl/wikitext.json` |
| C4 | word_perplexity | NaN | invalid, rerun needed | `.../ppl/c4.json` |
| C4 | byte_perplexity | NaN | invalid, rerun needed | `.../ppl/c4.json` |
| ARC-Easy | acc | 0.7929 | valid | `.../tasks/arc_easy.json` |
| ARC-Easy | acc_norm | 0.7660 | valid | `.../tasks/arc_easy.json` |
| ARC-Challenge | acc | 0.4915 | valid | `.../tasks/arc_challenge.json` |
| ARC-Challenge | acc_norm | 0.5290 | valid | `.../tasks/arc_challenge.json` |
| PIQA | acc | 0.7748 | valid | `.../tasks/piqa.json` |
| PIQA | acc_norm | 0.7824 | valid | `.../tasks/piqa.json` |
| OpenBookQA | acc | 0.2640 | valid | `.../tasks/openbookqa.json` |
| OpenBookQA | acc_norm | 0.4020 | valid | `.../tasks/openbookqa.json` |
| RACE | acc | 0.2823 | valid | `.../tasks/race.json` |
| WinoGrande | acc | 0.6835 | valid | `.../tasks/winogrande.json` |
| HellaSwag | acc / acc_norm | pending | running on DP=8 | `.../tasks/hellaswag_dp8.json` |
| MMLU | acc / acc_norm | pending | running on DP=8 | `.../tasks/mmlu_dp8.json` |

## Table 2: Converted CARE Checkpoint HF Results

These numbers are recorded for traceability, but **must not be treated as real MLA results** because they were produced before the corrected HF MLA loader was installed.

| Benchmark | Metric | Value | Status | Source |
| --- | --- | ---: | --- | --- |
| WikiText2 | word_perplexity | 44.5240 | stale pre-fix, rerun required | `.../zero_shot_eval/ppl/wikitext.json` |
| WikiText2 | byte_perplexity | 2.0337 | stale pre-fix, rerun required | `.../zero_shot_eval/ppl/wikitext.json` |
| C4 | word_perplexity | 317.8960 | stale pre-fix, rerun required | `.../zero_shot_eval/ppl/c4.json` |
| C4 | byte_perplexity | 2.6186 | stale pre-fix, rerun required | `.../zero_shot_eval/ppl/c4.json` |
| ARC-Easy | acc | 0.2395 | valid on clean checkpoint, DP=8 | `.../zero_shot_eval/tasks/arc_easy_loaderfix_clean_dp8.json` |
| ARC-Easy | acc_norm | 0.2588 | valid on clean checkpoint, DP=8 | `.../zero_shot_eval/tasks/arc_easy_loaderfix_clean_dp8.json` |
| ARC-Challenge | acc | 0.4915 | stale pre-fix, rerun required | `.../zero_shot_eval/tasks/arc_challenge.json` |
| ARC-Challenge | acc_norm | 0.5290 | stale pre-fix, rerun required | `.../zero_shot_eval/tasks/arc_challenge.json` |
| PIQA | acc | 0.7748 | stale pre-fix, rerun required | `.../zero_shot_eval/tasks/piqa.json` |
| PIQA | acc_norm | 0.7824 | stale pre-fix, rerun required | `.../zero_shot_eval/tasks/piqa.json` |
| OpenBookQA | acc | 0.2640 | stale pre-fix, rerun required | `.../zero_shot_eval/tasks/openbookqa.json` |
| OpenBookQA | acc_norm | 0.4020 | stale pre-fix, rerun required | `.../zero_shot_eval/tasks/openbookqa.json` |
| RACE | acc | 0.2823 | stale pre-fix, rerun required | `.../zero_shot_eval/tasks/race.json` |
| WinoGrande | acc | 0.6835 | stale pre-fix, rerun required | `.../zero_shot_eval/tasks/winogrande.json` |
| HellaSwag | acc | 0.2568 | valid on clean checkpoint, DP=8 | `.../zero_shot_eval/tasks/hellaswag_loaderfix_clean_dp8.json` |
| HellaSwag | acc_norm | 0.2601 | valid on clean checkpoint, DP=8 | `.../zero_shot_eval/tasks/hellaswag_loaderfix_clean_dp8.json` |
| MMLU | acc | 0.2562 | valid on clean checkpoint, DP=8 | `.../zero_shot_eval/tasks/mmlu_loaderfix_clean_dp8.json` |
| MMLU | acc_norm | N/A | task config only emitted `acc` | `.../zero_shot_eval/tasks/mmlu_loaderfix_clean_dp8.json` |

## Table 3: Long-Context / Fixed-Pack Status

| Item | Result | Status | Notes |
| --- | --- | --- | --- |
| Converted passkey @ 2048 start | 0.75 | stale pre-fix, rerun required | `passkey_2048_4096.json` |
| Converted passkey @ 2048 middle | 0.75 | stale pre-fix, rerun required | `passkey_2048_4096.json` |
| Converted passkey @ 2048 end | 0.25 | stale pre-fix, rerun required | `passkey_2048_4096.json` |
| Converted passkey @ 4096 start | 0.25 | stale pre-fix, rerun required | `passkey_2048_4096.json` |
| Converted passkey @ 4096 middle | 1.00 | stale pre-fix, rerun required | `passkey_2048_4096.json` |
| Converted passkey @ 4096 end | 0.6875 | stale pre-fix, rerun required | `passkey_2048_4096.json` |
| Converted fixed-pack AIMO3 @ 2048 | finite byte-PPL, word-PPL inf | metric path patched, rerun still needed | previous run used bad word counting |
| Converted fixed-pack AIMO3 @ 8192 | NaN | all docs OOM | not meaningful |
| Original fixed-pack AIMO3 @ 2048 | finite byte-PPL, word-PPL inf | metric path patched, rerun still needed | previous run used bad word counting |
| Original fixed-pack AIMO3 @ 8192 | NaN | all docs OOM | not meaningful |
| Combined fixed-pack @ 2048 | JSON written | rerun needed after metric patch | old outputs used broken word counting |
| Combined fixed-pack @ 8192 | partial / NaN mix | rerun needed after metric patch | AIMO3 long samples OOM |

## Table 4: Side-by-Side Benchmark Matrix

This is the single combined matrix in the style of the CARE paper discussion.

Important:

- `Original` means the stock GPT-OSS-120B HF baseline.
- `Converted` means the CARE-converted checkpoint on the current HF eval path.
- `Delta` is `Converted - Original` where both values exist.
- `Status` explicitly marks whether the row is valid, provisional, pending, or invalid.

| Benchmark | Metric | Original GPT-OSS-120B | CARE-Converted 120B | Delta | Status | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| WikiText2 | word_ppl | 44.6211 | 44.5240 | -0.0971 | original valid, converted stale | converted value came from the pre-fix HF loader path |
| WikiText2 | byte_ppl | 2.0346 | 2.0337 | -0.0008 | original valid, converted stale | same caveat |
| C4 | word_ppl | NaN | 317.8960 | N/A | original invalid, converted stale | original run hit NaN bug; converted rerun required |
| C4 | byte_ppl | NaN | 2.6186 | N/A | original invalid, converted stale | same |
| ARC-Easy | acc | 0.7929 | 0.2395 | -0.5534 | original valid, converted valid | clean-checkpoint DP=8 rerun matches the earlier post-fix value exactly |
| ARC-Easy | acc_norm | 0.7660 | 0.2588 | -0.5072 | original valid, converted valid | same |
| ARC-Challenge | acc | 0.4915 | 0.4915 | 0.0000 | original valid, converted stale | same |
| ARC-Challenge | acc_norm | 0.5290 | 0.5290 | 0.0000 | original valid, converted stale | same |
| HellaSwag | acc | pending | 0.2568 | N/A | converted valid, original pending | converted uses the clean checkpoint and corrected MLA loader |
| HellaSwag | acc_norm | pending | 0.2601 | N/A | converted valid, original pending | same |
| PIQA | acc | 0.7748 | 0.7748 | 0.0000 | original valid, converted stale | same |
| PIQA | acc_norm | 0.7824 | 0.7824 | 0.0000 | original valid, converted stale | same |
| MMLU | acc | pending | 0.2562 | N/A | converted valid, original pending | converted uses the clean checkpoint and corrected MLA loader |
| MMLU | acc_norm | pending | N/A | N/A | original pending | converted task config only emitted `acc` |
| OpenBookQA | acc | 0.2640 | 0.2640 | 0.0000 | original valid, converted stale | same |
| OpenBookQA | acc_norm | 0.4020 | 0.4020 | 0.0000 | original valid, converted stale | same |
| RACE | acc | 0.2823 | 0.2823 | 0.0000 | original valid, converted stale | same |
| WinoGrande | acc | 0.6835 | 0.6835 | 0.0000 | original valid, converted stale | same |

## Table 5: Side-by-Side Long-Context / Domain Pack Matrix

| Evaluation | Original GPT-OSS-120B | CARE-Converted 120B | Status | Notes |
| --- | --- | --- | --- | --- |
| Passkey @ 2048 start | pending | 0.75 | converted stale | converted measured before loader fix; rerun required |
| Passkey @ 2048 middle | pending | 0.75 | converted stale | same |
| Passkey @ 2048 end | pending | 0.25 | converted stale | same |
| Passkey @ 4096 start | pending | 0.25 | converted stale | same |
| Passkey @ 4096 middle | pending | 1.00 | converted stale | same |
| Passkey @ 4096 end | pending | 0.6875 | converted stale | same |
| Fixed-pack AIMO3 @ 2048 | JSON written | JSON written | rerun needed | metric path patched after these runs |
| Fixed-pack AIMO3 @ 8192 | NaN | NaN | invalid | all long AIMO3 samples OOM |
| Fixed-pack combined @ 2048 | JSON written | JSON written | rerun needed | earlier outputs used broken word counting and HF converted loader caveat |
| Fixed-pack combined @ 8192 | partial / NaN mix | partial / NaN mix | invalid / rerun | AIMO3 long samples OOM |

## Immediate Next Steps

1. Finish original-model `DP=8` `hellaswag` and `mmlu`.
2. Render a clean original-model benchmark table from the completed JSONs.
3. Continue converted task reruns on the corrected HF MLA loader using the new `DP=8` launcher for the remaining stale tasks.
4. Rerun converted PPL / passkey / fixed-pack comparisons on the corrected HF MLA loader.
5. Replace the remaining stale pre-fix converted entries in this document with rerun results.
