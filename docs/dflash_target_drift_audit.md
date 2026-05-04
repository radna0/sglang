# DFlash Target Drift Audit

This file tracks the explicit boundary between DFlash production work and unrelated target-side drift.

## Keep

These changes are part of the intended DFlash production surface:

- speculative algorithm plumbing for `DFLASH`, `DFLASH_TREE`, and their overlap workers
- GPT-OSS aux-hidden capture for DFlash draft conditioning
- target verify CUDA-graph reuse for linear DFlash
- tree draft CUDA-graph runner and fused selected-verify append helpers
- independent target/draft KV dtype handling
- page-size related DFlash fixes that affect correctness or measured throughput
- benchmark and audit code that records DFlash contract and performance

## Quarantine

These areas are allowed only when directly justified by DFlash or GPT-OSS serving correctness:

- `server_args.py` speculative decoding config handling
- `model_runner.py`, `model_runner_kv_cache_mixin.py`, and `cuda_graph_runner.py`
- `scheduler.py` request validation and overlap/spec-v2 routing
- GPT-OSS model hooks required for aux-hidden capture
- attention backend compatibility checks for DFlash verify paths

## Do Not Expand

The following are not part of the DFlash production surface unless a later audit explicitly says otherwise:

- unrelated quantization refactors
- broad MXFP4 behavior changes outside GPT-OSS serving needs
- unrelated scheduler policy changes
- unrelated memory pool redesigns
- unrelated multimodal or tool parser changes
- speculative algorithms other than DFlash family, except where shared codepaths require compatibility

## Production Defaults Locked by Audit

- baseline proof lane stays on the same Showtime Harmony/tool-calling contract used in production
- first DFlash proof lane uses `DFLASH_LINEAR`, overlap off, `page_size=1`, `draft_page_size=1`
- first mixed-precision proof lane uses target KV `fp8_e4m3` and draft KV `bfloat16`
- tree mode and overlap mode are explicit follow-up lanes, not silent defaults

## Evidence Sources

- `docs/dflash_fused_tree_design.md`
- `docs/gpt_oss_dflash_kaggle.md`
- benchmark ledgers already committed on `showtime`
- the integrated runtime contract emitted at server startup by the DFlash capture audit logging
