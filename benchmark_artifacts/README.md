# Benchmark Artifacts

This directory mirrors the benchmark JSON artifacts that were previously only present in
local `/workspace/...` paths referenced by the docs.

Examples:

- [dflash_ctxfill_concurrency_matrix_share_20260327/ctx65536_decfill_c4.json](/workspace/sglang-dflash-line/benchmark_artifacts/dflash_ctxfill_concurrency_matrix_share_20260327/ctx65536_decfill_c4.json)
- [dflash_ctxfill_concurrency_matrix_share_20260327/ctx131072_decfill_c4.json](/workspace/sglang-dflash-line/benchmark_artifacts/dflash_ctxfill_concurrency_matrix_share_20260327/ctx131072_decfill_c4.json)
- [dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json](/workspace/sglang-dflash-line/benchmark_artifacts/dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json)
- [dflash_reference_bench_20260327_draftpage1_mem090_nostrict/result.json](/workspace/sglang-dflash-line/benchmark_artifacts/dflash_reference_bench_20260327_draftpage1_mem090_nostrict/result.json)

Layout rule:

- local `/workspace/foo/bar.json`
- repo mirror `benchmark_artifacts/foo/bar.json`

Current note:

- completed historical benchmark JSONs referenced in `README.md` and `BENCHMARK_LEDGER.md`
  are mirrored here
- live running benchmark outputs are not copied until they are stable enough to keep
