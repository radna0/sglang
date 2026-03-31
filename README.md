# GPT-OSS DFlash Status on `dflash`

This branch is focused on making GPT-OSS DFlash correct and fast on the stable:

- target attention backend: `fa3`
- target MoE backend: `triton_kernel`
- target KV cache: `fp8_e4m3`
- draft KV cache: `bfloat16`

The current best short-context regime is:

- target `page_size=256`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`

The current best long-budget reference regime on the local 3-problem harness is:

- target `page_size=1`
- draft `page_size=1`
- `share_pools=True`
- `block_size=16`
- `mem_fraction_static=0.90` for most rows, `0.85` for `ctx=65536, concurrency=8`

This is a greedy benchmark setup:

- `temperature=0.0`
- `top_p=1.0`
- `top_k=1`
- `min_p=0.0`
- `ignore_eos=True`

Important:

- the current reference-harness numbers in this README are **no-tool** runs
- they use plain `/generate`
- they are **greedy** unless a section explicitly says otherwise
- they do **not** yet tell us the DFlash behavior for tool-calling or for sampled decoding

## Benchmark Ledger

Use [BENCHMARK_LEDGER.md](/workspace/sglang-dflash-line/BENCHMARK_LEDGER.md) as the
canonical run registry for:

- baseline `showtime.py` Harmony tool-calling
- DFlash `showtime.py` Harmony tool-calling
- PaCoRe synthesis rounds
- `explore32 -> route8`
- later combined route + PaCoRe lanes

Every future sweep should add a row there with:

- exact regime
- quality target
- total / per-question wall time
- early-stop vs full-round policy
- and DFlash acceptance / verify metrics when speculative decoding is enabled

Documented benchmark JSON outputs are mirrored into:

- [benchmark_artifacts/README.md](/workspace/sglang-dflash-line/benchmark_artifacts/README.md)

## Production Verify Roadmap

The current implementation roadmap for production DFlash verify is tracked in:

- [DFLASH_PROD_VERIFY_ROADMAP.md](/workspace/sglang-dflash-line/DFLASH_PROD_VERIFY_ROADMAP.md)

This is the active design note for:

- target-only greedy verify
- target-only sampled verify
- FP8 target KV + BF16 draft KV boundaries
- DFlash overlap/spec-v2 support

`pq` verify is explicitly out of the active production plan.

## Focused Route Study

The focused mixed-pool route study is now complete enough for the current branch checkpoint.

Problem ladder used for the route-focused runs:

- hardest: `86e8e5`
- harder: `dd7f5e`
- decently hard: `a295e9`
- medium: `9c1c5f`
- easiest: `92ba6a`

Original launcher:

- [run_route5_explore32_route8_block8.sh](/workspace/sglang-dflash-line/scripts/playground/run_route5_explore32_route8_block8.sh)

Mixed-pool follow-up results:

- old mixed-pool greedy, `8 -> 8` cap:
  - [result.json](/workspace/route5_explore32_route8_block8_20260328/result.json)
  - exploration `3146.926 tok/s`, accept `2.928`
  - continuation `791.732 tok/s`, accept `3.974`
  - routed boxed-correct rate `0.75`
- mixed-pool greedy, `4 -> 8` cap:
  - [result.json](/workspace/route5_explore32_route8_block4to8_greedy_20260329/result.json)
  - exploration `3439.704 tok/s`, accept `2.474`
  - continuation `803.360 tok/s`, accept `4.208`
  - routed boxed-correct rate `0.75`
- mixed-pool sampled, `4 -> 8` cap:
  - [result.json](/workspace/route5_explore32_route8_block4to8_sampled_20260329/result.json)
  - exploration `2275.360 tok/s`, accept `2.225`
  - continuation `557.241 tok/s`, accept `3.002`
  - routed boxed-correct rate `0.75`

What these three runs mean:

- lowering the exploration cap from `8` to `4` improved exploration throughput
- the greedy `4 -> 8` rerun slightly improved continuation throughput and accept length
- the sampled `4 -> 8` rerun was materially slower and did not improve routed quality
- all promoted branches in both `4 -> 8` runs were still labeled `hard_tail`
- current mixed-pool routing is therefore not producing a real green-zone route set yet

So the route study is complete enough for now. Work should return to:

- overlap-v2
- fused / CUDA-graph / mixed-precision verify
- linear + tree-verify path completion
- FailFast
- SSD
- and only after that PaCoRe

## Tree Config Sweep

The next prepared benchmark, before we promote tree overlap further, is a single-request
tree-config sweep:

- linear DFlash baseline per block size
- tree configs over `block_size = 4 / 8 / 16`
- compact tree grids over `spec_steps` and `topk`
- rank by `wall tok/s` while requiring correct boxed output

Prepared launcher:

- [run_dflash_tree_config_sweep.sh](/workspace/sglang-dflash-line/scripts/playground/run_dflash_tree_config_sweep.sh)
- [run_dflash_tree_overlap_single_request_best.sh](/workspace/sglang-dflash-line/scripts/playground/run_dflash_tree_overlap_single_request_best.sh)
- [run_dflash_tree_overlap_config_sweep.sh](/workspace/sglang-dflash-line/scripts/playground/run_dflash_tree_overlap_config_sweep.sh)

Default sweep intent:

- single request first
- overlap disabled
- graph-backed tree verify still on
- compare tree configs against the matching linear DFlash baseline for the same block size

Initial sweep signal:

- `block_size=4` best so far: `spec_steps=3`, `topk=4`, `num_verify_tokens=4`
  - tree tok/s: `271.792`
  - linear tok/s: `255.166`
  - speedup: `1.065x`
  - correct boxed rate: `1.0`
- `block_size=8` best so far: `spec_steps=4`, `topk=2`, `num_verify_tokens=6`
  - tree tok/s: `299.849`
  - linear tok/s: `279.095`
  - speedup: `1.074x`
  - correct boxed rate: `1.0`
- `block_size=16` best so far: `spec_steps=8`, `topk=1`, `num_verify_tokens=9`
  - tree tok/s: `332.333`
  - linear tok/s: `294.920`
  - speedup: `1.127x`
  - correct boxed rate: `1.0`

Interpretation:

- single-request tree verify is already clean
- the best tree settings beat the matched linear baseline on throughput
- the overlap follow-up should use the best tree settings first
- overlap-enabled tree sweep across the same config grid is now prepared

## Locked Single-Request Tree vs Linear

The earlier single-request mismatch is now resolved. Two benchmark issues were fixed first:

- non-stream benchmark JSON handling / nested sampling-param merge in
  [bench_serving.py](/workspace/sglang-dflash-line/python/sglang/bench_serving.py)
- local benchmark scripts now import `DatasetRow` from the local benchmark module, not
  another checkout on `PYTHONPATH`

After those fixes, the clean non-overlap, graph-backed, single-request comparisons for
`92ba6a` are:

| Block | Linear tok/s | Tree tok/s | Speedup | Tree config | Correct |
|---|---:|---:|---:|---|---:|
| `4` | `274.047` | `304.201` | `1.110x` | `steps=3, topk=4, vt=4` | `1.0` |
| `8` | `301.705` | `343.534` | `1.139x` | `steps=4, topk=2, vt=6` | `1.0` |
| `16` | `315.758` | `385.327` | `1.221x` | `steps=8, topk=1, vt=9` | `1.0` |

Artifacts:

- block `4` linear: [block4_linear_current_postfix.json](/workspace/tree_vs_linear_apples_20260330/block4_linear_current_postfix.json)
- block `4` tree: [block4_tree_current_postfix.json](/workspace/tree_vs_linear_apples_20260330/block4_tree_current_postfix.json)
- block `8` linear: [block8_linear_current_postfix.json](/workspace/tree_vs_linear_apples_20260330/block8_linear_current_postfix.json)
- block `8` tree: [block8_tree_current_postfix.json](/workspace/tree_vs_linear_apples_20260330/block8_tree_current_postfix.json)
- block `16` linear: [block16_linear_current_postfix.json](/workspace/tree_vs_linear_apples_20260330/block16_linear_current_postfix.json)
- block `16` tree: [block16_tree_current_postfix.json](/workspace/tree_vs_linear_apples_20260330/block16_tree_current_postfix.json)

Interpretation now:

- single-request `DFLASH_TREE` is clean
- tree is faster than matched linear at `block=4/8/16`
- the best current win is `block=16`, about `1.22x`
- the next correct benchmark is batched non-overlap tree-vs-linear using these locked configs

## Batched Non-Overlap Tree vs Linear

The next step after the locked single-request comparison is the matched batched
non-overlap sweep on the same best-per-block tree configs.

Completed rows so far on the current tree code:

| Block | Concurrency | Linear tok/s | Tree tok/s | Speedup | Tree accept len | Correct |
|---|---:|---:|---:|---:|---:|---:|
| `4` | `1` | `274.120` | `306.603` | `1.1185x` | `3.208` | `1.0` |
| `4` | `4` | `1179.283` | `1194.003` | `1.0125x` | `3.282` | `1.0` |

Artifacts:

- `block=4, c=1` linear: [linear_report.json](/workspace/dflash_tree_batch_sweep_20260330_postfix/linear/block_4/c_1/linear_report.json)
- `block=4, c=1` tree: [tree_report.json](/workspace/dflash_tree_batch_sweep_20260330_postfix/tree/block_4/c_1/tree_report.json)
- `block=4, c=4` linear: [linear_report.json](/workspace/dflash_tree_batch_sweep_20260330_postfix/linear/block_4/c_4/linear_report.json)
- `block=4, c=4` tree: [tree_report.json](/workspace/dflash_tree_batch_sweep_20260330_postfix/tree/block_4/c_4/tree_report.json)
- manifest: [manifest.jsonl](/workspace/dflash_tree_batch_sweep_20260330_postfix/manifest.jsonl)

Important interpretation:

- batched `DFLASH_TREE` is still genuinely faster than matched linear on the first two
  completed rows, but the gain is much smaller at higher concurrency than in the
  single-request case
- this tree path is **not optimized enough yet**
- current `DFLASH_TREE` batching is still leaving substantial speed on the table
- higher-concurrency tree runs are still unstable on the current worktree, so the batch
  sweep is not finished yet

So the current state is:

- correctness on the completed batched rows is good
- some speedup is real
- but `DFLASH_TREE` is definitely **not** “blazingly fast” or fully optimized yet
- more tree-specific development is still required before we should judge the ceiling

## Important Baseline Finding

This branch now has a very important baseline result for showtime-faithful Harmony tool-calling:

No-DFLASH greedy baseline:

- run: `/workspace/showtime_baseline10_20260328_earlystop_nodflash`
- `7/10` final selected correct
- `8/10` any-correct / pass@8
- wall time: `5227s` = `1h27m07s`

No-DFLASH sampled baseline:

- run: `/workspace/showtime_baseline10_20260328_earlystop_nodflash_sampled`
- `10/10` final selected correct
- wall time: `5627s` = `1h33m47s`
- delta vs greedy: `+400s` = `+6m40s` = about `+7.7%`

Interpretation:

- moving from greedy to sampled baseline produced a major quality gain for a relatively modest
  wall-time increase
- this must become the reference point when we later judge:
  - DFLASH
  - PaCoRe
  - DFLASH + PaCoRe
  - sampled-target DFLASH variants

Important future speculative-design split:

- sampled target with greedy draft
- sampled target with sampled draft

Those are different speculative regimes and must be benchmarked separately.

## Current Headline Matrix

This is the current branch-level benchmark checkpoint that should be treated as the
reference before more overlap-v2 / fused-verify work lands.

| Family | Regime | Quality | Speed / accept | Artifact |
|---|---|---|---|---|
| baseline | no DFlash, greedy, `early_stop=4` | `7/10` final, `8/10` any-correct | `5227s` total | [summary.json](/workspace/showtime_baseline10_20260328_earlystop_nodflash/summary.json) |
| baseline | no DFlash, sampled, `early_stop=4` | `10/10` final, `10/10` any-correct | `5627s` total | [summary.json](/workspace/showtime_baseline10_20260328_earlystop_nodflash_sampled/summary.json) |
| baseline | no DFlash, sampled, full-round `8/8` | `9/10` final, `10/10` any-correct | full-round worse than sampled early-stop on `86e8e5` | [summary.json](/workspace/showtime_baseline10_20260328_fullround_nodflash_sampled/summary.json) |
| route | mixed-pool greedy, `8 -> 8` cap | continuation boxed-correct `0.75` | explore `3146.926 tok/s`, accept `2.928`; continue `791.732 tok/s`, accept `3.974` | [result.json](/workspace/route5_explore32_route8_block8_20260328/result.json) |
| route | mixed-pool greedy, `4 -> 8` cap | continuation boxed-correct `0.75` | explore `3439.704 tok/s`, accept `2.474`; continue `803.360 tok/s`, accept `4.208` | [result.json](/workspace/route5_explore32_route8_block4to8_greedy_20260329/result.json) |
| route | mixed-pool sampled, `4 -> 8` cap | continuation boxed-correct `0.75` | explore `2275.360 tok/s`, accept `2.225`; continue `557.241 tok/s`, accept `3.002` | [result.json](/workspace/route5_explore32_route8_block4to8_sampled_20260329/result.json) |
| overlap microbench | sampled easy `92ba6a`, non-overlap | microbench only | `561.643 tok/s`, accept `2.923` | [json](/workspace/dflash_overlap_compare_sampled_short_20260329/non_overlap_fused_current.json) |
| overlap microbench | sampled easy `92ba6a`, overlap | microbench only | `581.786 tok/s`, accept `3.042` | [json](/workspace/dflash_overlap_compare_sampled_short_20260329/overlap_fused_current.json) |
| overlap microbench | sampled hard `86e8e5`, non-overlap | microbench only | `472.817 tok/s`, accept `2.364` | [json](/workspace/dflash_overlap_compare_sampled_short_20260329/hard_non_overlap_fused_current.json) |
| overlap microbench | sampled hard `86e8e5`, overlap | microbench only | `473.963 tok/s`, accept `2.448` | [json](/workspace/dflash_overlap_compare_sampled_short_20260329/hard_overlap_fused_current.json) |

What this matrix currently says:

- the strongest confirmed quality lane is still the no-DFlash sampled baseline
- sampled `early_stop=4` beat sampled full-round `8/8` on the unstable hard case
- mixed-pool `explore32 -> route8` is not yet finding a real green-zone subset
- reducing exploration cap from `8` to `4` helped exploration overhead
- sampled mixed-pool routing did not improve routed quality and was slower
- overlap-v2 is helping, but only modestly so far on the sampled short A/B

The branch should therefore treat the route study as paused and put engineering focus back on:

- overlap-v2
- fused / CUDA-graph / mixed-precision verify
- linear + tree-verify path completion

## Share-Enabled Decode-Fill Concurrency Matrix

The completed long-decode matrix on this branch is now:

- target `page_size=1`
- draft `page_size=1`
- `share_pools=True`
- target KV `fp8_e4m3`
- draft KV `bfloat16`
- target attention `fa3`
- target MoE `triton_kernel`
- DFlash `block_size=16`
- CUDA graph enabled
- piecewise CUDA graph enabled
- decode-to-context-limit, no prompt padding

Artifacts:

- [matrix.md](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/matrix.md)
- [ctx65536_decfill_c1.json](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/ctx65536_decfill_c1.json)
- [ctx65536_decfill_c4.json](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/ctx65536_decfill_c4.json)
- [ctx65536_decfill_c8.json](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/ctx65536_decfill_c8.json)
- [ctx65536_decfill_c16.json](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/ctx65536_decfill_c16.json)
- [ctx131072_decfill_c1.json](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/ctx131072_decfill_c1.json)
- [ctx131072_decfill_c4.json](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/ctx131072_decfill_c4.json)
- [ctx131072_decfill_c8.json](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/ctx131072_decfill_c8.json)
- [ctx131072_decfill_c16.json](/workspace/dflash_ctxfill_concurrency_matrix_share_20260327/ctx131072_decfill_c16.json)

| ctx | requested out min/max | conc | mem frac | share pools | dflash tok/s | accept | verify sum |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 65536 | `64814/64814` | `1` | `0.90` | `True` | `387.298` | `5.031` | `12867` |
| 65536 | `64786/64839` | `4` | `0.90` | `True` | `1056.599` | `7.142` | `36281` |
| 65536 | `64786/64839` | `8` | `0.85` | `True` | `613.464` | `3.178` | `163143` |
| 65536 | `64786/64839` | `16` | `0.90` | `True` | `1474.585` | `5.010` | `206959` |
| 131072 | `130350/130350` | `1` | `0.90` | `True` | `362.775` | `4.937` | `26389` |
| 131072 | `130322/130375` | `4` | `0.90` | `True` | `965.801` | `7.678` | `67888` |
| 131072 | `130322/130375` | `8` | `0.90` | `True` | `632.854` | `3.929` | `265361` |
| 131072 | `130322/130375` | `16` | `0.90` | `True` | `958.972` | `5.182` | `402484` |

What this matrix means:

- the scalable long-decode regime is the shared-pool `page_size=1` path, not the earlier no-share path
- concurrency scaling is **not monotonic**
- `c4` is the cleanest throughput point on these three problems
- `c8` is the most pathological point here because it accumulates large verify burden and tail stalls
- `c16` can still win throughput on `ctx=65536`, but it does so while paying a much larger verify bill

This matches the live behavior observed during the run:

- while several requests are in easy low-entropy continuation, throughput climbs sharply
- once the batch drains into a few hard tails, aggregate throughput collapses
- DFlash can still be fast overall, but the hardest tails dominate wall time

## Correctness-Conditioned Findings From The Share Matrix

The harness now exports per-request:

- boxed answer
- fallback answer
- exact correctness flag
- `spec_accept_length`
- `spec_verify_ct`
- `spec_dflash_q_entropy_mean`
- `spec_dflash_q_max_mean`

On the completed share-enabled matrix:

- total requests: `58`
- correct boxed answers: `48 / 58 = 82.76%`

Per-problem:

- `92ba6a`: `18 / 24` correct
- `9c1c5f`: `18 / 18` correct
- `a295e9`: `12 / 16` correct

Per row:

- `ctx=65536, c1`: `1 / 1` correct
- `ctx=65536, c4`: `4 / 4` correct
- `ctx=65536, c8`: `8 / 8` correct
- `ctx=65536, c16`: `10 / 16` correct
- `ctx=131072, c1`: `1 / 1` correct
- `ctx=131072, c4`: `4 / 4` correct
- `ctx=131072, c8`: `7 / 8` correct
- `ctx=131072, c16`: `13 / 16` correct

This is the important result:

- high acceptance is useful, but it is **not** a correctness oracle
- some wrong branches are exactly the EAFT-style `confident conflict` regime:
  - relatively high acceptance
  - low entropy
  - high `q_max`
  - wrong final boxed answer

Examples from the current matrix:

- `92ba6a`, `ctx=131072`, `c16`
  - `accept=5.467`
  - `q_entropy=0.308`
  - `q_max=0.929`
  - wrong answer `100` instead of `50`
- `92ba6a`, `ctx=65536`, `c16`
  - several repeats at `accept≈4.8-5.3`
  - `q_entropy≈0.52-0.65`
  - `q_max≈0.86-0.88`
  - wrong answer `100` instead of `50`
- `a295e9`, `ctx=65536`, `c16`
  - `accept≈3.9-4.15`
  - `q_entropy≈0.49-0.51`
  - `q_max≈0.876-0.880`
  - wrong answer `580` instead of `520`

So the current evidence is:

- `accept>=6` looked very strong on this sample
  - `21 / 21` correct
- but the mid-high band `accept in [4, 6)` was **not** safe
  - `20 / 28` correct
- very low acceptance was not automatically wrong either
  - tiny sample, `3 / 3` correct for `accept < 2`

That means the current branch should **not** use acceptance alone for branch dropping or quality routing.

The safe interpretation is narrower:

- very high sustained acceptance is promising
- mid-high acceptance still needs a conflict detector
- low acceptance means expensive / unstable generation, but not automatically wrong

The current data therefore supports a stricter EAFT-style claim:

- use acceptance together with entropy/confidence
- specifically guard against confident-conflict branches
- do not assume low entropy + high `q_max` is sufficient

Current branch-level routing implication from this sample:

- the only clearly safe green zone so far is **very high** acceptance
  - `accept >= 6`: `21 / 21` correct on this sample
- a naive hard-tail drop policy is **not** supported
  - `accept < 3`: `4 / 5` correct on this sample
- the real ambiguous band is the mid-high confident region
  - `3.5 <= accept < 6`, `q_entropy < 0.7`, `q_max > 0.85`
  - `24` requests in this band
  - `16` correct, `8` wrong

So the current evidence does **not** support:

- dropping every low-accept tail
- trusting every low-entropy / high-`q_max` branch

It does support:

- promoting clearly green high-accept branches
- keeping hard tails alive unless another stronger signal says otherwise
- treating the mid-high confident band as the main place where branch-selection needs a better conflict detector

In practice, the next branch-selection policy should score with at least:

- acceptance length
- verify count
- `q_entropy`
- `q_max`
- eventual correctness outcome on held-out runs

Not just:

- acceptance length alone

## Regime Separation: No-Tool vs Tool-Calling, Greedy vs Sampled

There are at least four materially different serving regimes here:

1. no-tool + greedy
2. no-tool + sampled
3. tool-calling + greedy-ish control
4. tool-calling + sampled

The current benchmark matrix on this branch is only measuring the first one:

- no-tool
- greedy
- long decode to remaining context budget
- DFlash on the target model directly

That matters because DFlash acceptance is a **draft-target agreement** signal, not a correctness oracle.

### No-Tool, Greedy

This is the cleanest regime for interpreting acceptance.

Why:

- the continuation is produced directly by the model
- there is no external tool response perturbing the token stream
- `temperature=0.0`, `top_p=1.0`, `top_k=1` means the target path is deterministic
- high acceptance here is the strongest evidence that the local continuation is low-entropy and easy for the draft to predict

This is why the current EAFT-style interpretation is meaningful on the existing runs:

- low entropy
- high `q_max`
- high acceptance

But even here, high acceptance still means:

- high local predictability

It does **not** automatically mean:

- globally correct reasoning
- correct final answer

A wrong continuation can also become low-entropy later if the model has already locked itself into a confident but wrong branch.

### No-Tool, Sampled

This is a different regime and should not be mixed with the greedy numbers.

Why:

- temperature / top-p sampling widens the active support of the next-token distribution
- that generally lowers local predictability for the draft
- acceptance should usually drop compared with greedy
- variance across runs matters much more

This regime is relevant if the goal is self-consistency or majority-vote accuracy, but the serving interpretation changes:

- lower acceptance does not necessarily mean worse quality
- it may just mean the sampler is exploring a wider candidate space

So for sampled evaluation we should record separately:

- `temperature`
- `top_p`
- number of samples / branches
- best-of / majority-vote policy

### Tool-Calling

Tool-calling is also a separate regime and should not be inferred from the current no-tool numbers.

Why:

- tool call boundaries introduce structured control tokens
- tool argument spans often behave differently from ordinary free-form text
- tool outputs feed new information back into the next target continuation
- the continuation becomes partly a function of tool observations, not just prior model text

Expected effect on quality:

- often **better** task accuracy / final correctness, especially for math, retrieval, search, code execution, or calculator-style tasks

Expected effect on DFlash drafting:

- potentially **worse** draft-target agreement around:
  - tool selection tokens
  - argument serialization
  - response-boundary tokens
  - immediately after tool results are injected

So it is completely plausible that:

- tool-calling improves correctness
- while also reducing average acceptance length

That would not contradict the current no-tool DFlash results. It would just mean the serving regime changed.

### What Must Be Reported Separately

For every future DFlash study, do not mix these settings in one table without labeling them:

- tool usage: `no-tool` vs `tool-calling`
- decode policy: `greedy` vs `sampled`
- sampler settings: `temperature`, `top_p`, `top_k`, `min_p`
- whether quality is:
  - single-run exact correctness
  - best-of-k
  - majority vote
  - tool-augmented correctness

If we fail to separate those axes, acceptance statistics become ambiguous.

### Practical Interpretation for This Branch

Current branch numbers mean:

- DFlash is working on the **no-tool, greedy** regime
- those acceptance numbers are most useful as a proxy for:
  - local predictability
  - serving efficiency
  - draft-target alignment

Current branch numbers do **not** yet mean:

- the same acceptance behavior will hold under tool-calling
- the same throughput/acceptance tradeoff will hold under sampling
- high acceptance automatically implies final-answer correctness

### Recommended Follow-Up Matrix

The next quality-conditioned matrix should be split explicitly into:

1. no-tool + greedy
2. no-tool + sampled (`temperature > 0`)
3. tool-calling + greedy-style control
4. tool-calling + sampled

For each quadrant, record:

- final-answer correctness
- acceptance length
- verify count
- `q_entropy`
- `q_max`
- whether the request entered a stable high-accept regime or collapsed to `accept≈1`

## Hypothesis: Acceptance as a Quality-And-Branching Signal

This is the working hypothesis for the next phase. It is **not proven yet**. It is the thing we want to test directly with the correctness-conditioned harness.

### Core Hypothesis

EAFT-style reasoning suggests that the continuation quality regime matters more than a single scalar like "difficulty".

The working claim is:

- high acceptance is usually a sign of:
  - low local entropy
  - high local next-token probability
  - strong draft-target agreement
  - stable continuation structure
- `accept≈1` is usually a sign of:
  - high entropy / low-confidence continuation
  - confident conflict regions
  - unstable reasoning trajectory
  - expensive verify-heavy generation

The stronger version we want to test is:

- branches that enter a sustained high-accept regime are **more likely** to end correct than branches that remain stuck near `accept≈1`

That is plausible, but it is not automatic.

Why it might be true:

- good branches may move from:
  - high-entropy / high-probability early exploration
  - into low-entropy / high-probability stable continuation
- bad branches may stay trapped in:
  - high-entropy / low-probability
  - or confident-conflict regions

Why it is not guaranteed:

- a wrong branch can also become low-entropy later if the model becomes confidently wrong
- so acceptance is a useful signal, but not a correctness oracle by itself

### What We Actually Need To Prove

The right empirical test is per-request, not aggregate:

- does sustained high acceptance correlate with final-answer correctness?
- does persistent `accept≈1` correlate with final-answer failure?
- do `q_entropy`, `q_max`, and `verify_ct` improve that prediction?
- is the relationship different for:
  - no-tool vs tool-calling
  - greedy vs sampled

This is why the harness now needs per-request:

- expected answer
- extracted boxed answer
- fallback integer answer
- correctness flag
- acceptance / entropy / confidence metrics on the same row

### Operational Hypothesis

If the correlation holds, then DFlash acceptance is not just a speed signal. It becomes a routing / search signal.

That would let us treat the first chunk of decode as an exploration phase:

- example: first `8192` decode tokens
- launch more branches than the final serving budget
- observe which branches fall into the good regime
- then allocate more long-decode budget to the branches that look structurally promising

### Case 1: One Tail Request Stalls The Whole Batch

Observed pattern:

- several requests climb into high-accept, high-throughput regimes
- one remaining request collapses to `accept≈1`
- the whole batch wall time then gets dominated by that last hard tail

If the quality correlation holds, then the hard tail is doubly bad:

- slower
- less likely to finish correct

That suggests a branch-management policy:

- do not always keep decoding the worst tail to the same depth
- instead, consider replacing it with:
  - another fresh branch
  - a restarted sample
  - a tool-augmented branch
  - a different search trajectory

This is only justified if the low-accept branch is empirically less likely to be correct.

### Case 2: Good Regime Requests Accelerate Over Time

The current long-decode traces already suggest a pattern:

- early decode: lower acceptance
- later decode: some requests climb into much higher acceptance

The working interpretation is:

- early reasoning is exploratory and locally harder to draft
- later stable continuation is easier to draft

If that pattern also predicts correctness, then entering the good regime means:

- more likely to be on a productive reasoning path
- more likely to stay fast under DFlash

That is exactly the kind of regime transition suggested by the EAFT framing:

- move from harder, higher-entropy regions
- into lower-entropy, higher-probability continuation

### Exploration-Phase Strategy

One concrete hypothesis to test:

- for the first `8192` decode tokens, use a larger exploration batch
- example:
  - sample `32` instead of `8`
  - or `16` instead of `4`
- score branches by:
  - acceptance
  - `q_entropy`
  - `q_max`
  - verify burden

Then transition into an exploitation phase.

### Exploitation Option 1: Hard Cap For Strict Latency

- explore with a large branch pool
- after the exploration window, keep only the top `K`
- discard the rest

This optimizes for latency / bounded serving cost.

Example:

- explore `32`
- keep best `8`
- continue long decode only on those `8`

### Exploitation Option 2: Soft Cap For Throughput

- explore with a large branch pool
- keep all branches that clear the "good regime" threshold
- do not force a hard cap if many are promising

This optimizes for total throughput / answer yield instead of strict latency.

Example:

- explore `32`
- if `17` look good, keep all `17`

### Special Case To Test

If no branch escapes the bad regime, then brute-force continuation may be the wrong policy.

That special-case policy should be tested explicitly:

- if the entire exploration set stays near `accept≈1`
- and entropy/confidence do not improve
- then escalate rather than continue the same path

Possible escalation routes:

- restart with new samples
- enable tool-calling
- raise search diversity
- switch to a different exploration budget
- hand the request to a non-speculative or different speculative policy

### FailFast / Adaptive Block-Size Implication

If the hypothesis is right, then FailFast-style control should be used asymmetrically:

- good regime:
  - allow larger speculative length
  - potentially scale toward much larger accepted blocks
- bad regime:
  - shrink speculative length aggressively
  - stop wasting draft/verify work on hopeless tails

That means the best architecture is probably:

- fixed physical max block
- adaptive logical speculative length
- branch selection based on acceptance + entropy + confidence

not:

- one static block size for all requests forever

### What This README Treats As Hypothesis, Not Result

The following are **hypotheses to test**, not established findings yet:

- high-accept branches are more likely to be correct
- low-accept tails are less likely to be worth continuing
- exploration-phase oversampling improves final quality or latency
- tool-calling changes the acceptance/correctness frontier in a useful way
- FailFast-style dynamic scaling can exploit the good regime without hurting quality

## Execution Mode Matrix

I used two different execution modes on this branch, and they answer different questions.

Production-style throughput runs, with CUDA graph enabled:

- `/workspace/dflash_reference_bench_20260327_draftpage1_mem090_nostrict/result.json`
- `/workspace/dflash_reference_bench_20260327_ctx65536_dec8192_page1_mem090_nostrict/result.json`
- `/workspace/dflash_longctx_20260327_metrics_v4/ctx65536_dec8192.json`
- `/workspace/dflash_longctx_20260327_metrics_v4/ctx131072_dec8192.json`
- `/workspace/dflash_longctx_20260327_controls/ctx65536_dec8192_page1_noshare.json`
- `/workspace/dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json`

For those runs:

- `cuda_graph = true`
- piecewise CUDA graph was enabled with `piecewise_cuda_graph_max_tokens = 8192`

Controller-isolation runs, with CUDA graph disabled on purpose:

- `/workspace/dflash_block_investigate_20260327_easy_v7/92ba6a_block16_adaptive_tuned_fastq_eager2048_ctx65536.json`
- `/workspace/dflash_block_investigate_20260327_easy_v8/92ba6a_block16_fixed_eager2048_ctx65536.json`
- `/workspace/dflash_block_investigate_20260327_hard_v7/a295e9_block16_adaptive_tuned_fastq_eager2048_ctx65536.json`
- `/workspace/dflash_block_investigate_20260327_hard_v8/a295e9_block16_fixed_eager2048_ctx65536.json`

For those runs:

- `cuda_graph = false`
- `--disable-cuda-graph` was used intentionally to isolate controller overhead and proposal-shaping logic without mixing in graph-capture effects

So the right read is:

- graph-on runs are the production throughput measurements
- eager runs are the controller-validation measurements
- do not compare eager controller rows against graph-on throughput rows without saying so explicitly

For the latest graph-on controller comparisons, I pinned:

- `PYTHONPATH=/workspace/sglang-dflash-line/python`
- `disable_stream=True` in the local harness

That matters because:

- the installed site-package copy of `sglang` can diverge from this repo checkout
- the local non-stream `/generate` path correctly reports `completion_tokens=2048`
- the earlier `128`-token graph artifact came from using the wrong import path during local replay, not from the DFlash server path itself

## Current Production Verdict On Adaptive Cap

The current adaptive logical-cap controller is **not** production-ready on the graph-on path.

Two production-path fixes were required first:

- DFlash CUDA-graph replay must use the actual logical token count on replay, not `bs * physical_block`
- manual DFlash draft `ForwardBatch` construction must carry `num_token_non_padded`

Code paths:

- `/workspace/sglang-dflash-line/python/sglang/srt/model_executor/cuda_graph_runner.py`
- `/workspace/sglang-dflash-line/python/sglang/srt/speculative/dflash_worker.py`
- `/workspace/sglang-dflash-line/scripts/playground/bench_reference_dflash.py`

### Graph-On Local Harness Results, `context_length=65536`, `decode_len=2048`, `page=1`, `draft_page=1`, `block=16`

Artifacts:

- `/workspace/dflash_block_investigate_20260327_graph_v1/92ba6a_block16_fixed_graph_ctx65536_local.json`
- `/workspace/dflash_block_investigate_20260327_graph_v1/92ba6a_block16_adaptive_graph_ctx65536.json`
- `/workspace/dflash_block_investigate_20260327_graph_v1/a295e9_block16_fixed_graph_ctx65536_local.json`
- `/workspace/dflash_block_investigate_20260327_graph_v1/a295e9_block16_adaptive_graph_ctx65536.json`

| prompt | regime | wall tok/s | accept | verify | max_steps_mean | q_entropy | q_max |
|---|---|---:|---:|---:|---:|---:|---:|
| `92ba6a` | fixed | 237.941 | 3.388 | 571 | 15.000 | 1.705 | 0.599 |
| `92ba6a` | adaptive | 102.182 | 2.102 | 1975 | 8.227 | 2.677 | 0.440 |
| `a295e9` | fixed | 186.482 | 3.022 | 783 | 15.000 | 2.219 | 0.484 |
| `a295e9` | adaptive | 183.043 | 2.595 | 777 | 9.712 | 2.234 | 0.482 |

Interpretation:

- the eager-tuned adaptive gate does **not** transfer to the graph-on production path
- on the easy prompt, it is decisively wrong:
  - throughput drops from `237.941` to `102.182 tok/s`
  - verify count jumps from `571` to `1975`
  - logical width collapses to about `8.2` even though fixed `16` is clearly better
- on the hard prompt, it is also not a win on the production path:
  - throughput slips from `186.482` to `183.043 tok/s`
  - acceptance drops from `3.022` to `2.595`

So the current production conclusion is:

- keep `block_size=16` fixed on the graph-on path for now
- do **not** enable the current adaptive logical-cap controller by default
- the eager gains were real for controller-isolation, but they do not survive the graph-on serving path yet

The corrected long-decode benchmark on this branch is now `showtime.py`-faithful in the
important sense:

- it keeps the original reference prompts unchanged
- it does **not** pad prompts out to `65k`
- it greedily decodes to the remaining context budget
- so the long-run measurements below are about long output continuation, not fake long input

## Current Status

- original GPT-OSS GQA works on `fa3` + `triton_kernel`
- GPT-OSS DFlash page-size handling is fixed
- paged target + inherited draft paging is still bad
- paged target + non-paged draft (`draft_page_size=1`) is the good regime
- `block_size=8` is the best current default on the short reference benchmark
- on the long-budget local reference harness, `page_size=1 / draft_page_size=1 / share_pools=False` is currently better than the paged-target setup
- DFlash overlap scheduling is still not production-ready on this branch
- on the corrected long-decode harness, the key acceptance variable is local continuation predictability, not simply "question difficulty"

## Why GPT-OSS DFlash Was Slow

The main GPT-OSS DFlash failure mode was not the draft checkpoint. It was page-size handling.

Bad regime:

- target paged KV
- draft inherits the same page size
- pool sharing stays enabled
- acceptance collapses to about `1.3`
- DFlash loses badly to baseline

Good regime:

- target stays paged
- draft runs non-paged with `draft_page_size=1`
- pool sharing is disabled
- acceptance returns to about `3.0+`
- DFlash beats baseline

## Reference Benchmark Setup

Reference benchmark:

- local problems from `/root/reference.csv`
- target model: `/workspace/offload_root/gpt-oss-120b`
- draft model: `/root/epoch_65_step_23760`
- attention backend: `fa3`
- MoE backend: `triton_kernel`
- target KV dtype: `fp8_e4m3`
- draft KV dtype: `bfloat16`

## Page Size / Block Size Matrix

| run | page | draft_page | block | accept | wall tok/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| target128_inherit | 128 | inherit=128 | 16 | 1.308 | 343.541 | 0.532x |
| target128_draft1 | 128 | 1 | 16 | 3.060 | 706.925 | 1.096x |
| target256_inherit | 256 | inherit=256 | 16 | 1.301 | 329.256 | 0.517x |
| target256_draft1 | 256 | 1 | 16 | 3.077 | 708.020 | 1.113x |
| target256_draft1_block4 | 256 | 1 | 4 | 2.581 | 778.190 | 1.223x |
| target256_draft1_block8 | 256 | 1 | 8 | 3.175 | 815.444 | 1.281x |
| target256_draft1_block16 | 256 | 1 | 16 | 3.077 | 719.152 | 1.130x |

Conclusion:

- `draft_page_size=1` is required for GPT-OSS DFlash on paged targets
- `block_size=8` is the best current short-context default

## Long-Budget Reference Harness

Important: these `65k` / `131k` runs match the `showtime.py` serving budget style, not a filled-context stress test.

- the harness raises the server `context_length`
- it does **not** pad the prompt itself out to `65k` or `131k` tokens
- so these numbers are valid for the same budget regime, but not for a true full-context occupancy benchmark

Artifacts:

- `/workspace/dflash_longctx_20260327_metrics_v4/ctx65536_dec8192.json`
- `/workspace/dflash_longctx_20260327_metrics_v4/ctx131072_dec8192.json`
- `/workspace/dflash_longctx_20260327_controls/ctx65536_dec8192_page1_noshare.json`
- `/workspace/dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json`

### Paged Target, Non-Paged Draft

Regime:

- target `page_size=256`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`
- greedy decode

Results:

| run | baseline tok/s | dflash tok/s | accept | verify sum | verify avg | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `ctx65536_dec8192` | 149.581 | 209.309 | 3.056 | 8353 | 2784.333 | 1.399x |
| `ctx131072_dec8192` | 150.882 | 211.339 | 3.056 | 8353 | 2784.333 | 1.401x |

Interpretation:

- the nearly identical `65k` and `131k` rows confirm this harness is budget-matched, not prompt-filled
- the good short-context regime remains good here
- but it is not the best local long-budget regime we tested

### Non-Paged Target, Non-Paged Draft, No Sharing

Regime:

- target `page_size=1`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`
- greedy decode

Result:

| run | baseline tok/s | dflash tok/s | accept | verify sum | verify avg | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `ctx65536_dec8192_page1_noshare` | 149.299 | 226.235 | 3.212 | 7831 | 2610.333 | 1.515x |

Interpretation:

- on this local long-budget harness, disabling paging on the target as well is currently better
- acceptance improves from `3.056` to `3.212`
- verify count drops from `8353` to `7831`
- the speedup rises from about `1.40x` to about `1.52x`

So the current best-known regimes are now split:

- short-context throughput: `page=256 / draft_page=1 / block=8`
- local long-budget reference harness: `page=1 / draft_page=1 / share_pools=False / block=8`

## Corrected Showtime-Style Long Decode

This is the important regime for the reference problems:

- source prompts: `/root/reference.csv`
- prompt text unchanged
- no prompt padding
- greedy decode to remaining context budget
- `context_length=65536`
- `buffer_tokens=512`
- target `page_size=1`
- draft `page_size=1`
- `share_pools=False`
- `block_size=8`

Artifact:

- `/workspace/dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json`

Aggregate result:

| run | baseline tok/s | dflash tok/s | accept | verify sum | verify avg | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `ctx65536_page1_noshare_decodefill` | 183.648 | 310.940 | 3.388 | 57385 | 19128.333 | 1.693x |

Requested output budget per prompt:

- min: `64786`
- max: `64839`
- avg: `64813.0`

This is a real long-output decode-fill run, not the earlier fixed-`8192` decode budget.

## Per-Problem Acceptance Split

These are all plain no-tool `/generate` runs on the corrected long-decode harness. So the
long-accept behavior below is **not** a tool-calling effect.

Artifacts:

- `/workspace/dflash_showtime_decodefill_20260327/per_problem/92ba6a_block8.json`
- `/workspace/dflash_showtime_decodefill_20260327/per_problem/9c1c5f_block8.json`
- `/workspace/dflash_showtime_decodefill_20260327/ctx65536_page1_noshare_decodefill.json`

Per-problem summary:

| problem | rough level | accept | verify sum | wall tok/s | notes |
|---|---|---:|---:|---:|---|
| `92ba6a` | easy | 7.086 | 9125 | 642.082 | spends long stretches at `7.9-8.0` acceptance |
| `9c1c5f` | medium | 5.914 | 10947 | 540.101 | starts verify-heavy, later climbs into near-cap acceptance |
| `a295e9` | medium-hard | 1.738 | 37313 | 160.321 | remains verify-heavy overall |

Notes:

- `a295e9` is inferred exactly from the completed 3-problem aggregate minus the two direct
  per-problem runs. It does not need a redundant second full rerun.
- `92ba6a` and `9c1c5f` both eventually enter a near-cap acceptance regime during forced
  long decode.
- `a295e9` does not do so enough to matter overall.

## What Actually Drives Long Acceptance

The best explanation is **local predictability of the continuation**.

That is more precise than just saying "easy question" or "hard question".

The empirical pattern is:

- early or mid active reasoning: lower acceptance, more verifies
- late stable tail: higher acceptance, sometimes near perfect

So the right interpretation is:

- high acceptance happens when the continuation has fallen into a low-entropy, high-confidence,
  easy-to-draft regime
- that can happen even after a genuinely hard problem, once the model is deep in a predictable tail

## Block Size 16 Investigation

The earlier `block_size=16` regression result was not trustworthy.

Two separate code issues were contaminating that conclusion:

1. Greedy `target_only` verify was ignoring `max_steps_per_req`.
   - That meant `physical_block=16 / logical_cap=8` was not actually capped in greedy mode.
   - The fix is in:
     - `python/sglang/srt/speculative/dflash_utils.py`
     - `python/sglang/srt/speculative/dflash_info.py`
2. The eager control path was also wrong.
   - `--disable-cuda-graph` still allowed piecewise CUDA graph capture to initialize.
   - That fix is in:
     - `python/sglang/srt/model_executor/model_runner.py`

So the corrected conclusion is:

- `block_size=16` is **not** collapsing because it is "unsupported"
- the earlier `16/8` measurements were invalid because the logical cap was dead on the greedy path
- after fixing that, `block=16` behaves normally on both the easy and hard controls below

### Corrected Easy Prefix Probe: `92ba6a`, decode `2048`

Important regime note:

- the natural `block=8` / `block=16` rows below were run at `context_length=65536`
- the newer proposal-cap validation rows were run at `context_length=8192`
- use the `65536` rows to compare natural physical block choices
- use the `8192` rows to validate proposal-side capping, entropy export, and adaptive policy behavior

Artifacts:

- `/workspace/dflash_block_investigate_20260327/92ba6a_block8_eager2048.json`
- `/workspace/dflash_block_investigate_20260327/92ba6a_block16_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_easy_v4/92ba6a_block16_force8_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_easy_v4/92ba6a_block16_adaptive_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_easy_v7/92ba6a_block16_adaptive_tuned_fastq_eager2048_ctx65536.json`
- `/workspace/dflash_block_investigate_20260327_easy_v8/92ba6a_block16_fixed_eager2048_ctx65536.json`

Natural physical block comparison (`context_length=65536`):

| run | physical block | logical cap | wall tok/s | accept len | step max | verify sum |
|---|---:|---:|---:|---:|---:|---:|
| `92ba6a_block8` | 8 | 8 | 269.906 | 3.537 | 7 | 579 |
| `92ba6a_block16` | 16 | 16 | 230.305 | 3.850 | 15 | 532 |

Interpretation:

- On the easy prefix, `block=16` accepts slightly more than `block=8`.
- But it is slower overall, so the extra physical width is mostly overhead here.

Proposal-side logical cap validation (`context_length=8192`):

| run | physical block | controller | effective step mean | total draft tokens | wall tok/s | accept len | q_entropy_mean | q_max_mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `92ba6a_block16_force8` | 16 | forced `8` | 8.000 | 4968 | 57.951 | 3.231 | 1.682 | 0.601 |
| `92ba6a_block16_adaptive` | 16 | adaptive | 11.440 | 6658 | 60.823 | 3.450 | 1.653 | 0.606 |

Interpretation:

- proposal-side capping now works for greedy `target_only`
- the draft path is no longer generating the full physical width when the logical cap is `8`
- on this easy prompt, the current adaptive thresholds still dip into the capped regime early, but recover:
  - `spec_dflash_max_steps_min = 8`
  - `spec_dflash_max_steps_last = 15`
- so the controller is active, but still too aggressive on easy prompts

Current-code apples-to-apples rerun (`context_length=65536`, eager on purpose):

| run | physical block | controller | effective step mean | total draft tokens | wall tok/s | accept len | q_entropy_mean | q_max_mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `92ba6a_block16_fixed_current` | 16 | fixed `16` | 15.000 | 8280 | 63.730 | 3.598 | 1.597 | 0.621 |
| `92ba6a_block16_adaptive_current` | 16 | adaptive | 12.345 | 6901 | 62.723 | 3.555 | 1.606 | 0.619 |

Interpretation:

- on the current code path, the tuned adaptive controller is effectively throughput-neutral on the easy prompt
- it trims draft work meaningfully:
  - `spec_dflash_total_draft_token_num`: `8280 -> 6901`
- while keeping easy-prompt confidence in the same band:
  - `q_entropy_mean`: `1.597 -> 1.606`
  - `q_max_mean`: `0.621 -> 0.619`
- so the tuned gate is no longer the source of the easy-path slowdown; it stays wide enough and barely changes throughput

### Corrected Hard Prefix Probe: `a295e9`, decode `2048`

Artifacts:

- `/workspace/dflash_block_investigate_20260327_hard_v2/a295e9_block8_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_hard_v2/a295e9_block16_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_hard_v4/a295e9_block16_force8_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_hard_v4/a295e9_block16_adaptive_eager2048.json`
- `/workspace/dflash_block_investigate_20260327_hard_v7/a295e9_block16_adaptive_tuned_fastq_eager2048_ctx65536.json`
- `/workspace/dflash_block_investigate_20260327_hard_v8/a295e9_block16_fixed_eager2048_ctx65536.json`

Natural physical block comparison (`context_length=65536`):

| run | physical block | logical cap | wall tok/s | accept len | step max | verify sum |
|---|---:|---:|---:|---:|---:|---:|
| `a295e9_block8` | 8 | 8 | 21.687 | 1.159 | 7 | 1767 |
| `a295e9_block16` | 16 | 16 | 22.713 | 1.172 | 15 | 1747 |

Interpretation:

- On the hard prefix, `block=16` also does **not** collapse.
- `block=16` is slightly better than `block=8` on this probe:
  - slightly higher accept length
  - slightly fewer verifies
  - slightly better throughput

Proposal-side logical cap validation (`context_length=8192`):

| run | physical block | controller | effective step mean | total draft tokens | wall tok/s | accept len | q_entropy_mean | q_max_mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `a295e9_block16_force8` | 16 | forced `8` | 8.000 | 5824 | 51.417 | 2.843 | 2.116 | 0.502 |
| `a295e9_block16_adaptive` | 16 | adaptive | 9.353 | 6725 | 52.016 | 2.864 | 2.133 | 0.498 |

Interpretation:

- proposal-side capping now works on the hard prompt
- compared with the older uncapped `block16` run, draft work is materially lower:
  - old uncapped `block16` at `65536`: `spec_draft_token_num = 26205`
  - forced logical `8` at `8192`: `spec_dflash_total_draft_token_num = 5824`
- the adaptive controller also triggers the cap without a forced override:
  - `spec_dflash_max_steps_min = 8`
  - `spec_dflash_max_steps_last = 8`
  - `spec_dflash_max_steps_mean = 9.353`
- so the controller is functional on hard prompts, but it still leaves some overhead on the table before it settles down

Current-code apples-to-apples rerun (`context_length=65536`, eager on purpose):

| run | physical block | controller | effective step mean | total draft tokens | wall tok/s | accept len | q_entropy_mean | q_max_mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `a295e9_block16_fixed_current` | 16 | fixed `16` | 15.000 | 11610 | 46.904 | 2.617 | 2.239 | 0.474 |
| `a295e9_block16_adaptive_current` | 16 | adaptive | 9.968 | 7127 | 52.998 | 2.875 | 2.122 | 0.500 |

Interpretation:

- on the current code path, the tuned adaptive controller is materially helpful on the hard prompt
- it reduces draft work and verify pressure:
  - `spec_dflash_total_draft_token_num`: `11610 -> 7127`
  - `verify_ct_sum`: `774 -> 715`
- and improves throughput:
  - `wall_tok_s`: `46.904 -> 52.998`
- the hard-prompt signal stays clearly separated from the easy prompt:
  - higher `q_entropy_mean`
  - lower `q_max_mean`
- so the current best read is:
  - easy prompts: keep the block wide
  - hard prompts: contract the logical cap

### Final Read

The right read now is:

- `block=16` itself is not broken
- the earlier bad `block16` conclusion came from:
  - dead greedy logical-cap plumbing
  - a bad eager benchmark path
- the proposal-side logical cap is now real for greedy `target_only`
- the branch now exports per-request entropy / confidence signals into the benchmark JSON
- on hard/unpredictable continuations, high `q_entropy` and low `q_max` are useful hard-prompt signals
- on easy/predictable continuations, the current thresholds are still too aggressive and can cap early before relaxing later

So the remaining work is not "make block 16 work at all." That part is now demonstrated.
The remaining work is:

- reduce verify overhead when acceptance is already good
- retune the adaptive controller so easy prompts stay wider
- adapt the logical speculative depth when acceptance is bad
- export entropy/confidence signals and use them to drive that cap

This lines up with the same general signal family emphasized by EAFT, with one important
GPT-OSS greedy distinction:

- low entropy / high confidence predicts the easy, locally stable continuation regime
- high entropy / low confidence is the useful hard-prompt gate for the current target-only greedy controller

Reference:

- https://github.com/PRIS-CV/EAFT

## Entropy / Confidence Hooks Already In Tree

The branch already computes the right DFlash-side difficulty signals for EAFT- or FailFast-style
policies.

Verify-side stats in:

- `python/sglang/srt/speculative/dflash_info.py`

Available scalar diagnostics:

- `accept_ratio_mean`
- `p_y_mean`
- `q_y_mean`
- `frac_p_y_zero`
- `frac_q_y_zero`
- `tv_mean`
- `p_entropy_mean`
- `q_entropy_mean`
- `p_max_mean`
- `q_max_mean`

Available step-wise diagnostics:

- `accept_ratio_mean_by_step`
- `tv_mean_by_step`
- `p_entropy_mean_by_step`
- `q_entropy_mean_by_step`
- `p_max_mean_by_step`
- `q_max_mean_by_step`

Draft-side confidence debug in:

- `python/sglang/srt/speculative/dflash_worker.py`

Available draft debug signals:

- `q_max_mean_first`
- `q_ent_mean_first`

Relevant env flags:

- `SGLANG_DFLASH_PQ_SCALAR_STATS=1`
- `SGLANG_DFLASH_PQ_DIAG_STATS=1`
- `SGLANG_DFLASH_PQ_STEP_STATS=1`
- `SGLANG_DFLASH_DRAFT_CONF_DEBUG=1`

So the entropy/probability instrumentation already exists. The next productization step is to
surface these per-request signals into benchmark JSON directly instead of only keeping them in
verify-side debug structures or logs.

## Adaptive Design Choice

Best next design:

- keep a fixed physical max block
- vary a logical effective length per request / per round

Why this is the right first move:

1. `DFlashWorker` is built around a fixed physical `block_size`.
2. `block_size` is threaded through:
   - buffer allocation
   - KV-slot accounting
   - proposal tensor shapes
   - CUDA-graph capture shapes
3. The verifier already supports exact early stop through `max_steps_per_req`.
4. The worker already contains DAWN/FailFast-style cap plumbing.
5. Pre-capturing multiple physical block widths would increase:
   - graph capture time
   - memory pressure
   - switching complexity
   - duplicated runtime paths

That means the fastest safe path is:

1. fixed physical block
2. adaptive logical cap
3. only later, if needed, multi-width capture

## How FailFast Maps to GPT-OSS DFlash

FailFast-style logic is most useful on the hard regime, not the easy one.

For `a295e9`-like segments:

- low acceptance
- many verifies
- high speculative overhead

That is where we should shrink the effective speculative length aggressively.

For `92ba6a`-like late tails:

- acceptance is already near the block cap
- shrinking does not help much
- larger physical block may help more than additional caution

So the policy should be asymmetric:

- hard regime: shrink logical speculative length
- easy regime: allow the full block
- easy late tails are also the best candidates for testing larger physical blocks

## Fused vs Unfused KV Materialization

Short reference benchmark:

- `context_length=8192`
- `decode_len=2048`
- `concurrency=8`
- `num_prompts=8`
- target `page_size=256`
- draft `page_size=1`
- `block_size=8`

### Fused KV Enabled

Artifact:

- `/workspace/dflash_timing_ab_20260327/short_ctx8192_dec2048_fused.json`

Result:

- `wall_tok_s = 832.947`
- `speedup = 1.3088x`
- `accept_length = 3.150`
- `accept min/max step = 1 / 8`
- `accept draft-token min/max = 0 / 7`
- `verify_ct_sum = 5277`
- `verify_ct_avg = 659.625`
- `verify_ct_min/max = 577 / 798`
- `accept_token_sum = 11099`
- `step_time_p20_s = 0.016213`
- `output_tok_s_p20 = 194.274`

One-shot timing:

- `verify wall = 0.027609s`
- `draft_kv_append wall = 0.003420s`

### Fused KV Disabled

Artifact:

- `/workspace/dflash_timing_ab_20260327/short_ctx8192_dec2048_nofused.json`

Result:

- `wall_tok_s = 816.565`
- `speedup = 1.2830x`
- `accept_length = 3.175`
- `accept min/max step = 1 / 8`
- `accept draft-token min/max = 0 / 7`
- `verify_ct_sum = 5237`
- `verify_ct_avg = 654.625`
- `verify_ct_min/max = 558 / 798`
- `accept_token_sum = 11139`
- `step_time_p20_s = 0.016491`
- `output_tok_s_p20 = 192.543`

One-shot timing:

- `verify wall = 0.027493s`
- `draft_kv_append wall = 0.001636s`

### Interpretation

Fused KV is not the main bottleneck.

- fused is only about `2.0%` faster on wall tok/s than unfused
- fused slightly worsens acceptance
- fused slightly increases total verify count

So the current ceiling is still dominated by verify-side overhead, not by CPU commit or by the fused/unfused append path alone.

## Overhead Diagnosis

Why `accept_length ~= 3.1` does not become a `3x` throughput win:

- baseline `step_time_p20_s` is about `0.006838`
- DFlash `step_time_p20_s` is about `0.0162 - 0.0165`
- that means a speculative step still costs about `2.4x` a baseline step

So even with `accept_length ~= 3.1`, the achievable gain is much smaller than `3x`.

The current evidence says:

- CPU pack/commit is not the main issue
- fused KV materialization is not the main issue
- the real remaining cost is still in:
  - verify path
  - allocator / KV-free behavior
  - DFlash-specific scheduling overhead

## Overlap Scheduling Status

Overlap scheduling is still disabled for DFlash on the main path.

Reason:

- the current overlap scheduler is still Eagle/spec-v2 shaped
- it expects `next_draft_input`, `topk_p`, `topk_index`, and future-token payloads
- DFlash does not return that payload contract yet

So this is not just a hidden flag. It still needs real DFlash-specific overlap plumbing before it is safe to enable by default.

## Code Changes on This Branch

Relevant files:

- `python/sglang/srt/speculative/dflash_worker.py`
- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_utils.py`
- `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- `python/sglang/bench_serving.py`
- `scripts/playground/bench_reference_dflash.py`

Implemented:

- explicit draft-page handling for GPT-OSS DFlash
- GPT-OSS fused KV path now supports QKV bias
- DFlash verify now records acceptance histograms correctly
- benchmark helper now preserves per-request `meta_info`
- reference benchmark now reports:
  - mean accept length
  - min/max step accept length
  - min/max accepted draft tokens
  - total/avg/min/max `spec_verify_ct`

## Recommended Launch Regime

For GPT-OSS DFlash on this branch, prefer one of two regimes.

Short-context / throughput-oriented:

```bash
sglang serve \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 256 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /root/epoch_65_step_23760 \
  --speculative-draft-attention-backend fa3 \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --speculative-draft-page-size 1 \
  --speculative-dflash-block-size 8 \
  --speculative-moe-runner-backend triton_kernel
```

Long-budget local reference harness:

```bash
export SGLANG_DFLASH_DRAFT_SHARE_POOLS=0

sglang serve \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 1 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /root/epoch_65_step_23760 \
  --speculative-draft-attention-backend fa3 \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --speculative-draft-page-size 1 \
  --speculative-dflash-block-size 8 \
  --speculative-moe-runner-backend triton_kernel
```

## Benchmark Command

Short reference benchmark:

```bash
export PYTHONPATH=/workspace/sglang-dflash-line/python
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_DFLASH_TIMING=1

/venv/main/bin/python /workspace/sglang-dflash-line/scripts/playground/bench_reference_dflash.py \
  --model-path /workspace/offload_root/gpt-oss-120b \
  --draft-model-path /root/epoch_65_step_23760 \
  --reference-csv /root/reference.csv \
  --context-length 8192 \
  --decode-len 2048 \
  --concurrency 8 \
  --num-prompts 8 \
  --page-size 256 \
  --draft-page-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --draft-kv-cache-dtype bfloat16 \
  --attention-backend fa3 \
  --moe-runner-backend triton_kernel \
  --draft-attention-backend fa3 \
  --speculative-moe-runner-backend triton_kernel \
  --speculative-dflash-block-size 8 \
  --mem-fraction-static 0.90
```

## Artifact Paths

Key artifacts:

- `/workspace/dflash_pagesize_matrix_20260327/target128_auto.json`
- `/workspace/dflash_pagesize_matrix_20260327/target128_draft1.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_auto.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block4.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block8.json`
- `/workspace/dflash_pagesize_matrix_20260327/target256_draft1_block16_timed.json`
- `/workspace/dflash_timing_ab_20260327/short_ctx8192_dec2048_fused.json`
- `/workspace/dflash_timing_ab_20260327/short_ctx8192_dec2048_nofused.json`
- `/workspace/dflash_longctx_20260327_metrics_v4/ctx65536_dec8192.json`
- `/workspace/dflash_longctx_20260327_metrics_v4/ctx131072_dec8192.json`
- `/workspace/dflash_longctx_20260327_controls/ctx65536_dec8192_page1_noshare.json`

## Exploration Router Prototype

`scripts/playground/route_reference_dflash.py` is the current application-layer
prototype for EAFT/FailFast-style branch routing on GPT-OSS DFLASH.

Current policy shape:

1. Run an oversampled exploration pool (for example `c32`) on the real reference
   prompts with greedy decode.
2. Stop exploration early once the pool is no longer clearly improving and a
   promoted subset outperforms the full pool.
3. Promote the selected subset into a smaller continuation pool (for example `c8`).
4. Run continuation with DFLASH adaptive logical caps enabled so hard survivors
   shrink toward near-target-only behavior while easy survivors keep using the
   larger physical block ceiling.

Current implementation details:

- Physical block ceiling stays fixed at `16`.
- Exploration is chunked by `--exploration-round-len` instead of always spending
  the full exploration budget.
- Promotion is triggered from exploration-round acceptance / score deltas, not
  from the final `8192`-token exploration budget alone.
- Continuation can enable the built-in server-side adaptive cap controller via:
  - `--continuation-adaptive-cap-enable`
  - `--continuation-adaptive-cap-verify-ct-ge`
  - `--continuation-adaptive-cap-accept-ema-hard-le`
  - `--continuation-adaptive-cap-accept-ema-medium-le`
  - `--continuation-adaptive-cap-hard-steps`
  - `--continuation-adaptive-cap-medium-steps`

This is still an application-layer prototype:

- it does not preserve KV across exploration -> continuation
- it is for routing-policy validation, not final serving efficiency
- it is where we should test early-promotion and hard-tail downgrade policies
  before moving them into the server/scheduler path

## Immediate Next Work

1. Finish validating the chunked exploration router on the `65536` long-decode lane.
2. Use the router results to decide the early-promotion trigger before retract-heavy `c32` linger.
3. Tune per-request hard-tail shrink / near-target-only fallback inside the promoted continuation pool.
4. Only after that, compare `block=16` exploration against a smaller exploration block such as `8`.
