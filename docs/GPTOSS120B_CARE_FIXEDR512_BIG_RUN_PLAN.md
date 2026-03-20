# GPT-OSS-120B CARE Fixed-r512 Big Run Plan

This is the next mainline conversion plan after the paper-faithful Alpaca `128 x 2048`
reproduction and the current fixed-r512 vs CARE-E comparison sweep.

The goal is not speed. The goal is a materially stronger zero-shot CARE checkpoint built
from the extended corpus mix we actually care about.

## Decision

For the next big conversion run:

- keep `r = 512` fixed
- do **not** use CARE-E dynamic rank
- do **not** use healing in the primary path
- do **not** mix in serving/backend work
- do one serious zero-shot conversion on the extended corpus first

Reason:

- the current paper-faithful checkpoint proves the pipeline but not the quality
- the first fixed-r512 comparison result shows that simply replacing CARE-E with uniform
  rank on Alpaca alone does not recover parity
- the next variable to change should be **data**, not rank logic or healing

## Primary run

Checkpoint target:

- GPT-OSS-120B native MLA
- fixed `kv_lora_rank = 512`
- `qk_rope_head_dim = 32`
- `qk_nope_head_dim = 32`
- no decoupled-RoPE expansion in the first extended run

Corpus target:

- use the existing extended corpus mix:
  `/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_corpus_mix.full.json`
- target total rows: `400000`
- sequence length: `2048`
- covariance batch size: `2`
- preferred path: reuse a prepacked token cache when available instead of retokenizing raw
  rows every run

Prepack artifacts:

- launcher:
  `/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_prepack_full.sh`
- builder:
  `/root/sglang-gpt-oss-care-mla/scripts/prepack_gpt_oss_corpus.py`
- hybrid spec builder for incremental restarts:
  `/root/sglang-gpt-oss-care-mla/scripts/build_hybrid_prepacked_spec.py`
- packed-source kind consumed by the collector:
  `packed_torch_filelist`

Operational policy:

- do not wait for the full prepack cache before benefiting from it
- as soon as one source finishes prepacking, build a hybrid spec and restart the main
  conversion against that partially packed corpus
- this is especially valuable for `general_c4` and `calib_packs`, because they dominate
  the row budget and otherwise waste time in Python tokenization/packing

Crash/restart policy:

- covariance collection now checkpoints every `save_every_batches` interval per rank
- each checkpoint persists:
  - processed batch / sequence counts
  - active source progress
  - packed shard cursor for packed sources
  - raw-source valid-row cursor and pending token buffer for raw sources
  - per-rank partial covariance tensors
- restarting the same run root resumes the covariance stage from the saved cursor instead
  of starting that source from zero
- if covariance is already merged, the pipeline skips directly to the next unfinished
  stage
- if rank schedule JSON already exists, the pipeline skips schedule derivation
- if the converted checkpoint already exists, the pipeline skips conversion

Current limitation:

- exact mid-write resume is implemented for the covariance stage, which is the dominant
  long-running stage
- the final MLA conversion stage is restart-safe at the stage level, not mid-file: if the
  final checkpoint write is interrupted before `config.json` lands cleanly, that stage
  reruns on restart

Why this is the right first big run:

- it keeps the architecture fixed while changing only the calibration corpus
- it is much closer to the real deployment distribution than Alpaca-only
- it preserves the ability to compare directly against:
  - paper-faithful Alpaca CARE-E
  - paper-faithful Alpaca fixed-r512
  - future fixed-r checkpoints on the same benchmark lane

## AIMO3 long-context follow-up lane

This is **not** the first extended conversion run. It is the first follow-up if the broad
`seq_len=2048` run is still too weak on long-context tool-calling/reasoning.

Follow-up target:

- dataset:
  `/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_aimo65k_conversion.full.json`
- fixed `kv_lora_rank = 512`
- sequence length: `65536`
- source: AIMO3 only

Why this lane exists:

- AIMO3 is the highest-value reasoning/tool-calling domain in the current target stack
- long-context behavior may require calibration at long sequence lengths instead of only
  evaluating at long sequence lengths after a short-context conversion
- this lane isolates that question cleanly without mixing it into the first broad run

Decision rule:

- run the broad extended fixed-r512 conversion first
- if AIMO3 long-context eval is still unacceptable, launch the dedicated AIMO3 `65k`
  fixed-r512 conversion next
- before any full AIMO3 `65k` run, use the small backend-probe lane:
  - spec:
    `/root/sglang-gpt-oss-care-mla/docs/gpt_oss120b_care_aimo65k_backend_probe.small.json`
  - launcher:
    `/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_aimo65k_backend_probe.sh`
- the backend probe is for backend viability only and must not replace the mainline
  quality run

## Corpus contents

The current full mix is:

- `general_c4`
- `calib_packs`
- `aimo3`
- `livecodebench`
- `math500`

Important note on AIMO3:

- AIMO3 is included in the conversion mix because long-context reasoning and tool-calling
  are real target behaviors, not afterthoughts
- however, the primary big conversion still uses `seq_len=2048`
- therefore AIMO3 long-context quality must be treated as an **evaluation gate**
  immediately after conversion, not assumed from the conversion alone

## Acceptance gates

The fixed-r512 extended conversion is only considered interesting if it improves on the
current Alpaca-only fixed-r/CARE-E checkpoints on the corrected evaluation path.

Minimum comparison set:

- WikiText2 PPL
- ARC-Easy
- HellaSwag
- MMLU
- fixed-pack AIMO3 PPL at short and long context
- fixed-pack combined long-context/domain PPL

Success means:

- generic benchmark retention is materially above the current Alpaca-only fixed-r/CARE-E
  checkpoints
- AIMO3/fixed-pack long-context behavior is not worse than the current checkpoint family
- the checkpoint is structurally clean and benchmarked only through the corrected MLA HF
  loader path

## What comes after this run

Only after this fixed-r512 extended run is benchmarked do we decide whether to:

1. stay on fixed-r512 and iterate on corpus/sequence length
2. introduce an AIMO-emphasis conversion manifest
3. move back to CARE-E
4. reopen healing as a fallback

That means CARE-E is explicitly demoted until we have at least one decent fixed-r512 CARE
checkpoint from the extended corpus.

## Fixed-r / CARE-U comparison ladder

The next fixed-r ladder is now explicitly:

- `r = 1024`
- `r = 512`
- `r = 448`
- `r = 384`
- `r = 320`
- `r = 256`
- `r = 128`

Why `r = 1024` is required:

- it is the cleanest near-original anchor inside the same MLA conversion family
- it tells us how much of the `r = 512` gap is coming from the low-rank bottleneck itself
- it is the right first sanity comparison before spending time on lower-rank sweeps

Interpretation rule:

- if `r = 1024` is still poor, the problem is not just "rank too small"
- if `r = 1024` is much better while `r = 512` is poor, the problem is primarily the
  compression budget rather than the conversion family

For clarity in these docs, `CARE-U` means:

- covariance-aware conversion with a uniform per-layer rank
- i.e. the fixed-r path implemented by `--uniform-rank`

## Launcher

The dedicated launcher for this plan is:

- `/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_fixedr512_big_8xh100.sh`

The rank-sweep wrapper that now includes `r = 1024` first is:

- `/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss120b_care_rank_sweep.sh`

Default output root:

- `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_<timestamp>`

Relaunching the same run root now rotates the old `pipeline.log` automatically to
`pipeline.restart_<timestamp>.log` before writing the new live log.
