# Showtime DFlash Branch Inventory

This document records the intended production composition of `radna0/showtime`.

## Integration Rule

- `showtime` is the only integration branch for new production DFlash work.
- `modal-dflash` is the runtime base for speculative worker, CUDA-graph, and scheduler plumbing.
- `dflash-pagesize-fix` is a source of measured GPT-OSS operating points, not a merge target.
- `tree-dflash-stable` is a source of selective tree-verify ideas and benchmark evidence, not a merge target.

## Runtime Features Currently Carried on `showtime`

| Feature | Source | Status |
| --- | --- | --- |
| Linear DFlash worker (`DFLASH`) | `modal-dflash` | Integrated |
| DFlash overlap worker (`DFlashWorkerV2`) | `modal-dflash` | Integrated, experimental |
| DFlash tree worker (`DFlashTreeWorker`) | `modal-dflash` + selective showtime work | Integrated |
| DFlash tree overlap worker (`DFlashTreeWorkerV2`) | `modal-dflash` + selective showtime work | Integrated, experimental |
| DFlash fused KV materialization helpers | `modal-dflash` / showtime | Integrated |
| DFlash target verify CUDA-graph reuse | `modal-dflash` | Integrated |
| Separate target/draft KV dtype plumbing | showtime | Integrated |
| GPT-OSS aux-hidden capture (`+1` capture mapping) | showtime | Integrated |
| Page-size tuning findings | `dflash-pagesize-fix` | Documented, selectively consumed |

## Production Contract

- Target attention backend: `fa3`
- Draft attention backend: `fa3`
- MoE backend: `triton_kernel`
- Sampling backend: `pytorch`
- First correctness lane:
  - `DFLASH_LINEAR`
  - fixed block size `16`
  - target KV `bf16`
  - draft KV `bf16`
  - `page_size=1`
  - `speculative_draft_page_size=1`
  - overlap disabled
  - decode CUDA graph enabled
- First mixed-precision lane:
  - target KV `fp8_e4m3`
  - draft KV `bfloat16`
  - separate target/draft KV pools
  - overlap disabled

## Tree Mode Contract

- Tree verify is a separate mode, not a replacement for linear DFlash.
- Initial production-facing tree contract:
  - `--speculative-algorithm DFLASH_TREE`
  - explicit `block_size`
  - explicit `speculative_num_steps`
  - explicit `speculative_eagle_topk`
  - explicit `speculative_num_draft_tokens` as verify-node budget
- Tree overlap remains experimental and must not be the default serving path.

## Deferred Items

- Shared-pool FP8/BF16 mixed-precision verify
- Adaptive logical block-size selection
- Route / PaCoRe integration into the first production proof lane
- Piecewise CUDA graph as a default requirement
