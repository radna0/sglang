# GPT-OSS-120B MLA Reproduction README

This README gives the practical commands and constraints for reproducing the GPT-OSS-120B MLA work on a single H100 host.

## Environment constraints

These runs are sensitive to memory pressure and backend selection. The important defaults on this host are:

- `kv_cache_dtype=bfloat16`
- `mem_fraction_static=0.95`
- `page_size=64`
- `attention_backend=flashmla`
- `tp_size=1` for the correctness smoke
- `lm-eval` logprob path for deterministic benchmark comparison

Before starting, verify disk headroom:

```bash
df -h / /root /workspace
```

The working repo is:

```bash
/workspace/sglang-gpt-oss-care-mla
```

## What is considered a valid smoke

A valid FlashMLA smoke must:

1. load the absorbed checkpoint without falling back to a dense BF16 upcast path,
2. allocate the KV pool successfully,
3. answer `/v1/models`,
4. complete a single completion request,
5. return a real text result rather than crashing or hanging.

On this host, the validated smoke output is stored at:

```bash
/workspace/smoke_preswizzle_r128_v7/result.json
```

## Validated smoke command

Use this exact shape when you want to reproduce the serving check:

```bash
PYTHONPATH=/workspace/sglang-gpt-oss-care-mla/python \
LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/torch/lib:/venv/main/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/venv/main/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/venv/main/lib/python3.12/site-packages/nvidia/nvshmem/lib:/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib \
/venv/main/bin/python /workspace/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_care_sglang_smoke.py \
  --model-path /workspace/r128_absorbed \
  --tokenizer-path openai/gpt-oss-120b \
  --tokenizer-mode auto \
  --attention-backend flashmla \
  --tp-size 1 \
  --page-size 64 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.95 \
  --disable-piecewise-cuda-graph \
  --server-timeout-s 1800 \
  --server-log /workspace/smoke_preswizzle_r128_v7/server.log \
  --out-json /workspace/smoke_preswizzle_r128_v7/result.json \
  --max-tokens 1 \
  --prompt hi
```

Notes:

- `--attention-backend flashmla` is the correctness target on this branch.
- `--disable-piecewise-cuda-graph` avoids the current unsupported graph path for FlashMLA sliding-window GPT-OSS MLA.
- `--mem-fraction-static 0.95` was required on this host to keep the KV pool allocation positive.
- If you change the absorbed checkpoint, make sure the rank-specific attention shard is complete.

## Validated deterministic benchmark command

The trusted comparison path is `lm-eval` logprob, not sampling.

The command shape is:

```bash
PYTHONPATH=/workspace/sglang-gpt-oss-care-mla/python \
LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/torch/lib:/venv/main/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/venv/main/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/venv/main/lib/python3.12/site-packages/nvidia/nvshmem/lib:/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib \
/venv/main/bin/python /workspace/sglang-gpt-oss-care-mla/scripts/run_gpt_oss_care_lm_eval.py \
  --model-path /path/to/your_absorbed_checkpoint \
  --tokenizer-path openai/gpt-oss-120b \
  --tokenizer-mode auto \
  --attention-backend flashmla \
  --tp-size 1 \
  --mem-fraction-static 0.95 \
  --batch-size auto \
  --num-concurrent 1 \
  --tasks arc_easy,arc_challenge,hellaswag,piqa,mmlu,openbookqa,race,winogrande \
  --limit 32 \
  --num-fewshot 0 \
  --out-json /workspace/gptoss120b_lmeval_logprob_subset_20260320_081507/r1024_limit32/combined.json
```

To compare another absorbed checkpoint, keep the same flags and only change `--model-path`.

Important:

- Do not use sampling for the comparison matrix.
- Keep the task list fixed when comparing ranks.
- `wikitext` was removed from this benchmark lane because it belongs in a dedicated rolling-PPL runner, not the generic completion-based MCQ subset.

## Sanity checks

Before rerunning the smoke, validate the repo-level regression checks:

```bash
PYTHONPATH=/workspace/sglang-gpt-oss-care-mla/python \
LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/torch/lib:/venv/main/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/venv/main/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/venv/main/lib/python3.12/site-packages/nvidia/nvshmem/lib:/venv/main/lib/python3.12/site-packages/nvidia/cu13/lib \
/venv/main/bin/pytest -q /workspace/sglang-gpt-oss-care-mla/test/manual/test_gpt_oss_care_cpu_checks.py
```

The validated result on this host is:

```bash
23 passed
```

## Checkpoint hygiene

For these experiments, treat the absorbed directories as the source of truth:

- `/workspace/r128_absorbed`
- `/workspace/r256_absorbed`

If you regenerate them, verify that the rank-specific attention shard is complete and that the runtime can load it before trusting any benchmark output.

The older incomplete low-rank exports were useful for debugging, but they were not valid evaluation checkpoints.

## Recommended workflow

1. Check disk space.
2. Run the manual regression checks.
3. Run the smoke against the absorbed checkpoint you want to validate.
4. Run the deterministic `lm-eval` logprob lane with a fixed task list and `limit`.
5. Only after that, expand to broader benchmark tables or new rank sweeps.
