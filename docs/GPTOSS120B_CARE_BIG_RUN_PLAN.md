# GPT-OSS-120B CARE Big Run Plan

This is the first non-smoke production run plan for GPT-OSS-120B CARE conversion and healing.

## Goal

Produce a real GPT-OSS-120B CARE MLA checkpoint and move it through the first full healing phases on 8xH100 with frozen GPT-OSS MXFP4 experts preserved in quantized form.

Current project progress:

- 92/100 toward the first production-grade GPT-OSS-120B CARE run.
- The remaining gap is no longer "can the runtime step?".
- The remaining gap is "run the full corpus-conditioned conversion and then scale healing from smoke to production datasets".

## Production datasets

Conversion corpus mix:

- `/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_corpus_mix.full.json`

Healing phases:

- general: `/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_general_phase.full.json`
- calib: `/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_calib_phase.full.json`
- aimo: `/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_aimo_phase.full.json`

## Run stages

1. Full CARE conversion

- target rows: `400000`
- seq len: `2048`
- target rank: `512`
- min rank: `128`
- qk rope head dim: `32`
- rank source fusion: enabled
- output: converted MLA checkpoint plus covariance and rank schedule

2. FSDP healing on 8xH100

- runtime: `fsdp`
- quantized expert layout: `replicated`
- teacher signal: cached top-k targets
- general seq len: `512`
- calib seq len: `2048`
- aimo seq len: `32768`
- general subset: `all_mla`
- calib subset: `rope_only`
- aimo subset: `all_mla_plus_o`

Why the AIMO phase is longer:

- the broad conversion stage should stay short enough to cover the full 400k-row corpus efficiently
- AIMO3 is the long-context, high-reasoning, tool-calling stage, so it gets a dedicated long-context healing pass instead of forcing the entire pipeline to 32k-131k
- current AIMO3 GPT-OSS tokenization stats on this box are roughly:
  - p50: `13240`
  - p90: `42968`
  - p95: `52958`
  - p99: `63150`
  - max: `67016`
- that means `32768` is a strong first production long-context target, while `65536` is the natural follow-up if we want to cover nearly the entire observed AIMO3 tail

3. Benchmark sweep

- PPL-first
- benchmark pack eval
- passkey retrieval
- LongBench-v2

## Acceptance gates

Conversion stage must produce:

- `covariance/`
- `kv_lora_rank_schedule.json`
- `converted_checkpoint/`
- `pipeline_manifest.json`

Healing stage must produce:

- non-empty `healing_fsdp_rank*.jsonl`
- `healing_fsdp_summary.json`
- `healed_absorbed_checkpoint/`

Evaluation stage must produce:

- PPL outputs on Wikitext and benchmark tasks
- benchmark suite manifest

## Immediate launch defaults

Full conversion:

- output root: `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_full_run`

Healing root:

- output root: `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_full_run_healing`

## Notes

- The current `run_gpt_oss_care_healing_fsdp.py` entrypoint is functioning through a source wrapper around the preserved compiled module. It is operational, but it should still be restored to plain source once the production run is no longer at risk.
- The shared MXFP4 preswizzle cache to reuse is:
  - `/workspace/gptoss_mxfp4_preswizzle_shared_v2`
