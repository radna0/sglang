# Fixed Domain Eval Packs

These packs are small, deterministic local sample sets meant for comparing multiple
GPT-OSS checkpoints on the exact same texts.

Current pack root:

- `/workspace/sglang_gpt_oss_care_runs/fixed_domain_eval_packs_v1`

Files:

- `aimo3_long.jsonl`
  - 12 deterministic longest AIMO3 tool-calling samples
  - intended for short-vs-long context PPL checks
- `calib_tool_seeded.jsonl`
  - 12 seeded Harmony calib-pack tool-bearing samples
- `livecodebench_seeded.jsonl`
  - 12 seeded LiveCodeBench code-generation prompts
- `math500_seeded.jsonl`
  - 12 seeded MATH-500 problems
- `combined_fixed_eval_pack.jsonl`
  - concatenation of all four packs above
- `manifest.json`
  - sample counts and char-length ranges

Builder:

- `/root/sglang-gpt-oss-care-mla/scripts/build_fixed_domain_eval_packs.py`

PPL runner:

- `/root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_hf_fixed_pack_ppl.py`

Harmony handling:

- AIMO3 and Harmony calib-pack rows are GPT-OSS Harmony transcripts, not plain text.
- The fixed-pack PPL runner now uses `openai_harmony` in `auto` mode to:
  - parse Harmony transcripts into structured messages for visible-text word/byte metrics
  - optionally round-trip the transcript through canonical Harmony rendering before tokenization
- canonicalization is performed with `auto_drop_analysis=False`, so long reasoning traces are preserved rather than silently dropped
- older fixed-pack outputs created before this Harmony-aware path should be treated as stale for AIMO3 / Harmony-heavy comparisons

Example:

```bash
CUDA_VISIBLE_DEVICES=6 /venv/main/bin/python \
  /root/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_hf_fixed_pack_ppl.py \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --pack-jsonl /workspace/sglang_gpt_oss_care_runs/fixed_domain_eval_packs_v1/aimo3_long.jsonl \
  --contexts 2048,8192 \
  --dtype bfloat16 \
  --device cuda \
  --output-path /workspace/sglang_gpt_oss_care_runs/original_gptoss120b_fixed_pack_eval/aimo3_long_ppl_2048_8192.json
```

Optional flags:

- `--harmony-visible-text auto|on|off`
- `--harmony-normalize-input auto|on|off`

The default `auto` mode is the recommended path for AIMO3 and Harmony calib-pack comparisons.

Recommended comparison policy:

- use the exact same pack files across original / converted / healed checkpoints
- keep the context ladder fixed per comparison
- for AIMO3, always include at least one long-context setting such as `8192`
- after the Harmony-aware runner change, rerun old AIMO3 / combined pack outputs before using them in side-by-side tables
