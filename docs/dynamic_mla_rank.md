# Dynamic per-layer MLA rank (CARE-E)

CARE-E's adjusted-rank allocation ("water-filling") can produce a non-uniform
`kv_lora_rank` across layers. The CARE OpenReview discussion calls out that most
existing MLA kernels do not support this yet ("Weakness 6 part II"), so serving
converted checkpoints requires system support for per-layer MLA latent rank.

This fork adds runtime support in SGLang for dynamic per-layer `kv_lora_rank`
on the FlashInfer MLA path.

## Config contract

Provide the per-layer schedule in the HF config as:

- `kv_lora_rank_per_layer`: `list[int]` of length `num_hidden_layers`

Fallback supported:

- `kv_lora_rank`: `list[int]` of length `num_hidden_layers`

SGLang keeps `kv_lora_rank` as a scalar (the `max(schedule)`) for compatibility
with older call sites.

## How it works (SGLang)

- KV cache allocation (`MLATokenToKVPool`) supports a per-layer buffer with:
  `kv_cache_dim[layer] = kv_lora_rank[layer] + qk_rope_head_dim`
- FlashInfer MLA plans one wrapper per unique rank in the current PP worker's
  layer range, then dispatches per layer via `layer.v_head_dim`.

## Usage

Use FlashInfer for MLA attention:

- `--attention-backend flashinfer`

## Current limitations

- Dynamic per-layer ranks are not supported in:
  - `flashmla`, `cutlass_mla`, `trtllm_mla`
  - CUDA-graph capture/replay in FlashInfer MLA
  - Host KV-cache pool / disaggregation offload for MLA KV cache
  - FP4 KV cache for dynamic schedules
- Ragged prefill and chunked-prefix runner are disabled automatically when
  dynamic ranks are detected.
