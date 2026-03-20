# CARE GPT-OSS-120B Conversion Pipeline

This repo now carries the GPT-OSS-120B CARE mainline directly.

Current status:

- We already have a **full offline 36-layer export** path, not just a one-layer smoke.
- What was still missing was the **dataset-conditioned CARE run**:
  - collect activation-space covariance on the real corpus mix
  - derive a CARE-E rank schedule under a fixed KV budget
  - convert with those stats
  - evaluate with PPL-first reporting
- SGLang serving compatibility for the converted GPT-OSS CARE MLA checkpoint is
  tracked separately in:
  - `docs/GPTOSS120B_CARE_SGLANG_SERVING_STATUS.md`
- Operator-aware fallback ideas beyond vanilla CARE-E are tracked in:
  - `docs/GPTOSS120B_OPERATOR_AWARE_FALLBACKS.md`
- The current next mainline run is the extended-data **fixed-r512** zero-shot conversion:
  - `docs/GPTOSS120B_CARE_FIXEDR512_BIG_RUN_PLAN.md`
  - `scripts/run_gpt_oss120b_care_fixedr512_big_8xh100.sh`
- The next planned rank ladders now explicitly keep `r = 1024` as the first anchor for
  both fixed-r CARE-U and CARE-E:
  - `1024, 512, 448, 384, 320, 256, 128`
  - sweep wrapper: `scripts/run_gpt_oss120b_care_rank_sweep.sh`

The design source remains:

- `/root/CARE-MLA/docs/CARE_GPTOSS120B_CONVERSION_PIPELINE.md`
- `/root/CARE-MLA/docs/dynamic_mla_rank.md`

## Current implementation in this repo

Runtime:

- `python/sglang/srt/models/gpt_oss.py`
  - `GptOssMlaForCausalLM`
  - `GptOssMLADecoderLayer`
  - `GptOssMLAAttention`
- `python/sglang/srt/configs/model_config.py`
  - `GptOssMlaForCausalLM` is treated as MLA architecture
  - hybrid sliding-window layer metadata works the same way as GPT-OSS MHA

Converter:

- `scripts/convert_gpt_oss_to_care_mla.py`

New pipeline pieces:

- `scripts/collect_gpt_oss_kv_covariance.py`
- `scripts/derive_gpt_oss_care_rank_schedule.py`
- `scripts/run_gpt_oss120b_care_pipeline.py`
- `scripts/run_gpt_oss120b_care_healing_pipeline.py`
- `scripts/run_gpt_oss120b_care_healing_8xh100.sh`
- `scripts/export_gpt_oss_care_absorbed.py`
- `scripts/run_gpt_oss_care_benchmark_suite.py`
- `scripts/run_gpt_oss_care_lm_eval.py`
- `docs/gpt_oss120b_care_corpus_mix.example.json`

Current CARE-E support in code:

- covariance-aware joint KV factorization in the converter
- per-layer dynamic rank allocation under a fixed total rank budget
- per-source covariance export for multi-domain calibration
- weighted covariance fusion at rank-allocation time via repeated `--covariance-dir`
- optional production rounding of the final CARE-E schedule via `--round-multiple`
- optional decoupled-RoPE channel export in the converter via `--decoupled-rope-dim`
- HF healing entrypoint in `scripts/run_gpt_oss_care_healing.py`
- multi-phase healing orchestration in `scripts/run_gpt_oss120b_care_healing_pipeline.py`
- rope-only and decoupled-rope-only healing subsets for targeted positional correction
- explicit absorbed export in `scripts/export_gpt_oss_care_absorbed.py`
- benchmark-suite automation in `scripts/run_gpt_oss_care_benchmark_suite.py`

## Converter contract

Input:

- standard GPT-OSS HF checkpoint directory with `config.json` and `model.safetensors.index.json`

Output:

- a new checkpoint directory with:
  - symlinked or copied original shards
  - a new MLA shard `model-care-mla-attention.safetensors`
  - updated `config.json`
  - updated `model.safetensors.index.json`
  - `care_mla_manifest.json`

## Current conversion scheme

Per layer:

1. load original `k_proj` and `v_proj`
2. expand GQA KV heads to per-query-head targets
3. split K channels into:
   - `qk_nope_head_dim`
   - `qk_rope_head_dim`
4. factorize `[K_nope ; V]` with rank `kv_lora_rank[layer]`
5. build shared rope projection from the rope slice
6. emit:
   - `kv_a_proj_with_mqa.weight`
   - `kv_a_proj_with_mqa.bias` when `attention_bias=true`
   - `kv_b_proj.weight`

Unchanged tensors are normally reused from the original checkpoint:

- embeddings / lm head / norms
- MoE weights
- `sinks`

When `--decoupled-rope-dim > 0`, the converter also rewrites:

- `q_proj.weight`
- `q_proj.bias` when present

This is how the checkpoint carries extra RoPE capacity instead of assuming the original query head width is fixed forever.

## Important limitation

This is a deployable CARE-oriented baseline, not the final word on GPT-OSS-to-DeepSeek parity.

The current baseline mapping assumes:

- `qk_nope_head_dim + base_qk_rope_head_dim == original head_dim`
- any extra decoupled rope capacity is appended as new `q_proj` / shared-K-rope rows and must be healed
- shared rope key projection is derived from the original K rope slice

That gives us a runnable MLA checkpoint while removing the old “RoPE healing is only a note” gap. There are now two healing runtimes:

- HF `device_map=auto` for quick bring-up
- torchrun/FSDP for gradient-sharded 120B healing on 8xH100

## Explicit absorbed export

If you want a checkpoint artifact that materializes `w_kc` and `w_vc` instead of relying on runtime folding:

```bash
python scripts/export_gpt_oss_care_absorbed.py \
  --model-path /root/out/gpt-oss-120b-care-mla-r512-r32 \
  --output-dir /root/out/gpt-oss-120b-care-mla-r512-r32-absorbed \
  --overwrite
```

The SGLang GPT-OSS MLA loader now accepts absorbed tensors directly and skips `kv_b_proj` folding when `w_kc` / `w_vc` are present.

## Healing pipeline

The practical 120B healing entrypoint is now the phase orchestrator:

```bash
python scripts/run_gpt_oss120b_care_healing_pipeline.py \
  --student-model-path /root/out/gpt-oss-120b-care-full-r512/converted_checkpoint \
  --general-dataset-spec-json /path/to/general_mix.json \
  --calib-dataset-spec-json /path/to/calib_packs.json \
  --aimo-dataset-spec-json /path/to/aimo3.json \
  --output-root /root/out/gpt-oss-120b-care-full-r512-healing \
  --device cuda \
  --device-map auto \
  --gradient-checkpointing
```

For a one-command 8xH100 HF-model-parallel launch:

```bash
scripts/run_gpt_oss120b_care_healing_8xh100.sh \
  /root/out/gpt-oss-120b-care-full-r512/converted_checkpoint \
  /path/to/general_mix.json \
  /path/to/calib_packs.json \
  /path/to/aimo3.json \
  --output-root /root/out/gpt-oss-120b-care-full-r512-healing
```

If you want the sharded path instead:

```bash
scripts/run_gpt_oss120b_care_healing_fsdp_8xh100.sh \
  /root/out/gpt-oss-120b-care-full-r512/converted_checkpoint \
  /path/to/general_mix.json \
  /path/to/calib_packs.json \
  /path/to/aimo3.json \
  --output-root /root/out/gpt-oss-120b-care-full-r512-healing-fsdp \
  --save-every 250
```

The phase pipeline also supports `--runtime fsdp`, which swaps the per-phase launcher from the HF healer to `torchrun ... run_gpt_oss_care_healing_fsdp.py`.

For FSDP distillation on 120B, the pipeline now also supports a two-phase teacher-cache flow:

```bash
python scripts/run_gpt_oss120b_care_healing_pipeline.py \
  --student-model-path /root/out/gpt-oss-120b-care-full-r512/converted_checkpoint \
  --general-dataset-spec-json /path/to/general_mix.json \
  --calib-dataset-spec-json /path/to/calib_packs.json \
  --aimo-dataset-spec-json /path/to/aimo3.json \
  --output-root /root/out/gpt-oss-120b-care-full-r512-healing-fsdp \
  --runtime fsdp \
  --fsdp-use-teacher-topk-cache \
  --teacher-topk-cache-device-map auto \
  --distill-topk 64 \
  --kl-weight 0.1
```

That mode runs `build_gpt_oss_teacher_topk_cache.py` first for each phase, then launches the FSDP healer with `--teacher-topk-cache-dir` instead of keeping a live 120B teacher resident beside the student.

Current caveat: the teacher-cache path removes the live-teacher OOM, and the replicated quantized-expert FSDP path is now operational with a warm shared preswizzle cache on 8xH100. In practice:

- a one-step 120B rope-only FSDP healing smoke completed end to end with cached teacher top-k targets
- frozen GPT-OSS expert weights stayed in HF MXFP4 form during loading/inference instead of being dequantized to dense bf16
- the shared preswizzle cache at `/workspace/gptoss_mxfp4_preswizzle_shared_v2` eliminated the earlier startup rebuild cost

The remaining runtime limitation is no longer "it cannot run"; it is efficiency. Replicated quantized experts still occupy about 56-57 GiB per GPU on 120B, so larger seq/batch/full-subset runs remain memory-heavy. The next optimization target is a stable sharded quantized-expert layout (or equivalent expert placement strategy), not a return to dense bf16 weights.

Operational note:

- `scripts/run_gpt_oss120b_care_healing_pipeline.py` now accepts `--mxfp4-preswizzle-dir` and `--quantized-expert-layout`
- `scripts/run_gpt_oss120b_care_healing_fsdp_8xh100.sh` now defaults `GPTOSS_MXFP4_PRESWIZZLE_DIR=/workspace/gptoss_mxfp4_preswizzle_shared_v2`
- setting `TORCH_DISABLE_ADDR2LINE=1` is recommended to avoid extremely noisy C++ symbolization output on distributed failures

## Full CARE run

The intended 120B run is now:

1. collect covariance on a weighted corpus mix
2. derive a CARE-E schedule under the target KV budget
3. convert into a deployable `GptOssMlaForCausalLM` checkpoint
4. serve and evaluate it

### Restart / resume semantics

The long covariance stage is now exact-resume capable across instance restarts.

What is saved at each partial checkpoint:

- per-rank partial covariance tensors
- processed batch and sequence counts
- active source progress
- packed-source shard / row cursor
- raw-source valid-row cursor plus pending token buffer

What happens on restart of the same run root:

- if only partial covariance exists, the pipeline relaunches the covariance collector in
  resume mode
- each rank restores its saved cursor and continues from the saved source position instead
  of restarting that source from zero
- once merged covariance exists, the pipeline skips covariance entirely
- once rank schedule JSON exists, the pipeline skips schedule derivation
- once the converted checkpoint has a completed `config.json`, the pipeline skips
  conversion

Current limitation:

- exact restart continuation is implemented for the covariance stage, which is by far the
  dominant runtime cost
- the final conversion/export stage is currently stage-resumable rather than mid-write
  resumable; if an export is interrupted before the completed checkpoint lands, that stage
  reruns on restart

### Recommended corpus mix

Use a weighted blend of:

- a general corpus like C4 or Alpaca
- the GPT-OSS-derived calibration packs
- the GPT-OSS-derived AIMO3 tool-calling corpus
- code-heavy calibration such as LiveCodeBench
- math-heavy calibration such as MATH-500

Example spec:

- [gpt_oss120b_care_corpus_mix.example.json](/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_corpus_mix.example.json)

This matches the paper discussion better than using only one domain. It also aligns with the target deployment profile here: general language + code/math + tool calling.

The example uses the already-materialized local JSONL filelists for the two ModelScope corpora because that path is more reliable in this environment than direct `MsDataset` loading.

If you want CARE-E to derive ranks from separate domain covariances instead of one mixed covariance dump, use `--rank-source-fusion`. That makes the collector emit `covariance/by_source/*` and the allocator fuse those per-source covariances using the effective source row-budget weights from the dataset spec.

If you want a deployment-friendly schedule instead of the raw scientific
water-filling output, pass `--round-multiple 8` or `--round-multiple 16`. The
allocator will keep the raw schedule in the JSON payload and emit a rebalanced
final schedule whose per-layer ranks are constrained to that multiple.

### One-command pipeline

```bash
python scripts/run_gpt_oss120b_care_pipeline.py \
  --model-path /root/gpt-oss-120b \
  --dataset-spec-json docs/gpt_oss120b_care_corpus_mix.example.json \
  --out-root /root/out/gpt-oss-120b-care-full-r512 \
  --target-total-rows 400000 \
  --seq-len 2048 \
  --batch-size 1 \
  --dtype bfloat16 \
  --device-map auto \
  --target-rank 512 \
  --min-rank 128 \
  --round-multiple 8 \
  --qk-rope-head-dim 32 \
  --rank-source-fusion
```

Outputs:

- `covariance/`
- `kv_lora_rank_schedule.json`
- `converted_checkpoint/`
- `pipeline_manifest.json`

### If you want uniform-r512 CARE without CARE-E scheduling

```bash
python scripts/run_gpt_oss120b_care_pipeline.py \
  --model-path /root/gpt-oss-120b \
  --dataset-spec-json docs/gpt_oss120b_care_corpus_mix.example.json \
  --out-root /root/out/gpt-oss-120b-care-uniform-r512 \
  --target-total-rows 400000 \
  --seq-len 2048 \
  --batch-size 1 \
  --dtype bfloat16 \
  --device-map auto \
  --target-rank 512 \
  --uniform-rank \
  --qk-rope-head-dim 32
```

### Rank-sweep policy

When comparing rank sensitivity after the current big run, keep two separate families:

- CARE-U:
  - covariance-aware conversion with `--uniform-rank`
- CARE-E:
  - covariance-aware conversion with water-filling under the corresponding total budget

Both families should keep `r = 1024` as the first anchor point before comparing `512`
and lower ranks. That gives the minimum useful gap test:

- `r = 1024` vs `r = 512`

If `r = 1024` still collapses, the problem is not just the compression ratio.

The wrapper that encodes these default rank lists is:

```bash
bash scripts/run_gpt_oss120b_care_rank_sweep.sh \
  /workspace/offload_root/gpt-oss-120b \
  /workspace/sglang_gpt_oss_care_runs/gptoss120b_rank_sweep \
  both
```

## Evaluation

The first evaluation pass should be PPL-first and benchmark-second.

Recommended first task set:

- `wikitext`
- `arc_easy`
- `arc_challenge`
- `hellaswag`
- `piqa`
- `mmlu`
- `openbookqa`
- `race`
- `winogrande`

Example:

```bash
python scripts/run_gpt_oss_care_lm_eval.py \
  --model-path /root/out/gpt-oss-120b-care-full-r512/converted_checkpoint \
  --tp-size 8 \
  --attention-backend triton \
  --tasks wikitext,arc_easy,arc_challenge,hellaswag,piqa,mmlu,openbookqa,race,winogrande \
  --out-json /root/out/gpt-oss-120b-care-full-r512/lm_eval.json
```

This wrapper expects `lm_eval` to be installed. If it is not installed yet, install `lm-eval-harness` first.

For an automated PPL/context sweep + benchmark pack:

```bash
python scripts/run_gpt_oss_care_benchmark_suite.py \
  --hf-model-path /root/out/gpt-oss-120b-care-full-r512/converted_checkpoint \
  --sglang-model-path /root/out/gpt-oss-120b-care-full-r512/converted_checkpoint \
  --out-root /root/out/gpt-oss-120b-care-full-r512/benchmarks \
  --contexts 2048,4096,8192,16384,32768 \
  --run-passkey \
  --passkey-contexts 4096,8192,16384,32768 \
  --run-longbench-v2 \
  --longbench-v2-num-examples 100 \
  --run-sglang
```

This writes:

- `hf_long_context/*.json` for the context-length PPL sweep
- `hf_benchmarks.json`
- `hf_passkey.json` when `--run-passkey` is enabled
- `hf_longbench_v2.json` when `--run-longbench-v2` is enabled
- `sglang_benchmarks.json` when `--run-sglang` is enabled
- `benchmark_suite_summary.json`

`hf_passkey.json` is a pure-HF synthetic retrieval eval. It hides a numeric key near the start, middle, or end of long contexts and measures exact recovery accuracy, which gives us a more direct long-context signal than perplexity alone.

`hf_longbench_v2.json` runs the official LongBench-v2 prompt format and answer extraction directly through HF generation, so the benchmark suite now covers both synthetic retrieval and real long-context multitask QA.

## Direct converter commands

### Recommended first run

```bash
python scripts/convert_gpt_oss_to_care_mla.py \
  --model-path /root/gpt-oss-120b \
  --output-dir /root/out/gpt-oss-120b-care-mla-r512 \
  --device cuda \
  --kv-lora-rank 512 \
  --qk-rope-head-dim 32
```

For a dynamic schedule:

```bash
python scripts/convert_gpt_oss_to_care_mla.py \
  --model-path /root/gpt-oss-120b \
  --output-dir /root/out/gpt-oss-120b-care-mla-caree \
  --device cuda \
  --rank-schedule-json /path/to/kv_lora_rank_schedule.json \
  --qk-rope-head-dim 32
```
