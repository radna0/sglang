# Benchmark Ledger

This file is the canonical run registry for GPT-OSS `showtime.py` / Harmony /
PaCoRe / DFlash experiments on this branch.

## Locked Current-Branch Proof Contract

The current proof lane for `reference.csv` on this branch is locked to:

- target `page_size=1`
- draft `page_size=1`
- `share_pools=False`
- `block_size=4`
- `speculative_num_draft_tokens=4`
- `mem_fraction_static=0.90`
- `speculative_draft_mem_fraction_static=0.97`
- FA3
- full decode CUDA graph + piecewise CUDA graph
- overlap disabled

This is the geometry that current-branch proof runs must use unless a row explicitly says
otherwise.

Use it to answer four questions for every run:

1. What exact regime was tested?
2. What quality metric was used?
3. What speed / cost metric was used?
4. What helped, and what did not?

## Required Dimensions

Every benchmark row should record these dimensions before we compare results:

| Field | Meaning |
|---|---|
| `run_id` | Short stable name for the run |
| `date` | UTC date |
| `status` | `running`, `finished`, `failed`, `superseded` |
| `model` | target model path / checkpoint |
| `serving_path` | `baseline`, `DFLASH`, `explore32->route8`, etc. |
| `tool_regime` | `no-tool`, `showtime-harmony-tool`, `native-responses`, etc. |
| `sampling_regime` | `greedy`, `sampled` |
| `pacore` | `off`, or widths such as `8->1`, `8,4->1` |
| `attempts` | number of attempts / branches |
| `early_stop` | integer threshold or `off/full-round` |
| `context_length` | serving context limit |
| `turns` | Harmony turn cap |
| `max_turn_output_tokens` | per-turn decode cap |
| `attention_backend` | `fa3`, `flashmla`, etc. |
| `moe_backend` | `triton_kernel`, etc. |
| `kv_cache_dtype` | target KV dtype |
| `speculative` | `off` or `DFLASH` |
| `draft_model` | draft checkpoint if speculative |
| `draft_kv_cache_dtype` | draft KV dtype |
| `block_size` | speculative physical block size |
| `page_size` | target page size |
| `draft_page_size` | draft page size |
| `share_pools` | `True/False` |
| `cuda_graph` | `on/off` |
| `piecewise_cuda_graph` | `on/off` |
| `mem_fraction_static` | static memory fraction |
| `out_dir` | artifact directory |
| `notes` | one-line caveat or intent |

## Quality Metrics

Record all applicable quality metrics. Do not collapse them into a single score.

| Metric | Meaning |
|---|---|
| `final_accuracy` | final selected answer accuracy over the problem set |
| `majority_vote_accuracy` | majority-vote answer accuracy over attempts |
| `pass_at_k` | whether any attempt in the set was correct |
| `boxed_accuracy` | boxed-answer extraction accuracy |
| `fallback_accuracy` | fallback integer extraction accuracy |
| `tool_success_rate` | fraction of tool calls that executed successfully |
| `python_calls` | total / per-question Python calls |
| `python_errors` | total / per-question Python tool errors |

## Speed / Cost Metrics

| Metric | Meaning |
|---|---|
| `wall_s_total` | total benchmark wall time |
| `wall_s_per_question` | total wall time per question |
| `wall_s_per_round` | per-round wall time when PaCoRe or routing is used |
| `cost_per_correct` | wall time divided by number of correct final answers |
| `attempts_completed` | number of attempts actually completed |
| `round_terminated_by` | `early_stop`, `full_width`, `timeout`, etc. |

## DFlash Metrics

These matter only when `speculative=DFLASH`, but they must be logged when present.

| Metric | Meaning |
|---|---|
| `spec_accept_length` | average accepted speculative length |
| `spec_verify_ct` | verify count |
| `spec_accept_token_num` | accepted draft token count |
| `spec_dflash_q_entropy_mean` | draft entropy proxy |
| `spec_dflash_q_max_mean` | draft max-prob proxy |
| `spec_dflash_p_entropy_mean` | target entropy proxy if available |
| `spec_dflash_p_max_mean` | target max-prob proxy if available |

## Regime Matrix

These are the main regime families we care about.

| Family | Description | Primary question |
|---|---|---|
| `baseline_showtime` | Original FA3 + `triton_kernel`, no speculative decoding | Best quality / speed baseline? |
| `dflash_showtime` | Same showtime Harmony tool-calling path, but with DFlash | Does DFlash preserve quality while improving speed? |
| `pacore_showtime` | Harmony + PaCoRe synthesis rounds | Does extra synthesis compute improve accuracy enough to justify time? |
| `route_showtime` | `explore32 -> route8` before long execution | Does routing improve cost-per-correct? |
| `route_plus_pacore` | exploration/routing combined with PaCoRe | Best combined quality + speed regime? |

## Current Known Results

These are the current branch-level conclusions we already trust.

| Finding | Status |
|---|---|
| Native `/v1/responses` Harmony loop is not yet faithful enough for the benchmark lane | confirmed |
| `showtime.py`-faithful `/generate` + local Harmony parsing works | confirmed |
| `92ba6a` smoke on showtime-faithful Harmony + tools succeeds | confirmed |
| DFlash long-decode throughput depends heavily on local continuation predictability | confirmed |
| High acceptance is useful but not a correctness oracle by itself | confirmed |
| Tool-calling vs no-tool and greedy vs sampled must be separated | confirmed |
| Baseline no-DFlash showtime 10-problem early-stop run | finished |
| Baseline no-DFlash showtime 10-problem sampled full-round run | finished |
| Mixed-pool `explore32 -> route8` route study is complete enough for now | confirmed |
| DFlash + PaCoRe 10-problem showtime run | running / partial |

## Run Table

Append new rows here. Keep failed and superseded runs; do not erase them.

| run_id | status | serving_path | tool_regime | pacore | early_stop | speculative | quality target | speed target | out_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `showtime_baseline10_nodflash_greedy_earlystop4` | finished | `baseline` | `showtime-harmony-tool` | `off` | `4` | `off` | `final_accuracy + pass@8` | `wall_s_total` | `/workspace/showtime_baseline10_20260328_earlystop_nodflash` | greedy baseline over all 10 |
| `showtime_baseline10_nodflash_sampled_earlystop4` | finished | `baseline` | `showtime-harmony-tool` | `off` | `4` | `off` | `final_accuracy + majority_vote + pass@8` | `wall_s_total` | `/workspace/showtime_baseline10_20260328_earlystop_nodflash_sampled` | sampled baseline: `temperature=1.0`, `top_p=1.0`, `top_k=50`, `min_p=0.02` |
| `showtime_baseline10_nodflash_sampled_fullround8` | finished | `baseline` | `showtime-harmony-tool` | `off` | `off/full-round` | `off` | `final_accuracy + majority_vote + pass@8` | `wall_s_total + wall_s_per_question` | `/workspace/showtime_baseline10_20260328_fullround_nodflash_sampled` | finished: `9/10` final correct, `10/10` any-correct |
| `showtime_route10_pacore8` | running/partial | `DFLASH` | `showtime-harmony-tool` | `8->1` | `4` | `DFLASH` | `final_accuracy` | `wall_s_total + DFlash metrics` | `/workspace/showtime_harmony_route10_20260328` | current optimized PaCoRe lane |
| `showtime_smoke_92_dflash` | finished | `DFLASH` | `showtime-harmony-tool` | `off` | `4` | `DFLASH` | `final_accuracy` | `wall_s_total + DFlash metrics` | `/workspace/showtime_harmony_smoke_20260328` | `92ba6a` correct, Python tools exercised |
| `route5_explore32_route8_block8` | finished | `explore32->route8` | `no-tool` | `off` | `router early-promotion` | `DFLASH` | `route selection quality + final routed quality` | `wall_s_total + route/explore/continue split` | `/workspace/route5_explore32_route8_block8_20260328` | finished: explore `3146.926 tok/s`, continue `791.732 tok/s`, boxed-correct `0.75` |
| `route5_explore32_route8_block4to8_greedy` | finished | `explore32->route8` | `no-tool` | `off` | `router early-promotion` | `DFLASH` | `route selection quality + final routed quality` | `wall_s_total + route/explore/continue split` | `/workspace/route5_explore32_route8_block4to8_greedy_20260329` | finished: exploration `3439.704 tok/s`, continuation `803.360 tok/s`, boxed-correct `0.75` |
| `route5_explore32_route8_block4to8_sampled` | finished | `explore32->route8` | `no-tool` | `off` | `router early-promotion` | `DFLASH` | `route selection quality + final routed quality` | `wall_s_total + route/explore/continue split` | `/workspace/route5_explore32_route8_block4to8_sampled_20260329` | finished: exploration `2275.360 tok/s`, continuation `557.241 tok/s`, boxed-correct `0.75` |
| `route_ckpt213_shared_adaptivecap_bad` | superseded | `explore32->route8` | `no-tool` | `off` | `router early-promotion + adaptive continuation cap` | `DFLASH` | `route selection quality + final routed quality` | `wall_s_total + route/explore/continue split` | `/workspace/route_ckpt213_shared_adaptivecap_20260404` | invalidated: controller bug self-locked continuation to `accept_len=1.00`, `accept_rate=0.06`, about `290-308 tok/s` by hard-capping to one step too early |
| `route_ckpt213_shared_adaptivecap_safe` | running/partial | `explore32->route8` | `no-tool` | `off` | `router early-promotion + adaptive continuation cap` | `DFLASH` | `route selection quality + final routed quality` | `wall_s_total + route/explore/continue split` | `/workspace/route_ckpt213_shared_adaptivecap_safe_20260404` | corrected controller defaults removed the immediate self-lock, but current long-context continuation is still weak: live `299-332 tok/s`, `accept_len ~1.00-1.01`, `accept_rate ~0.06`; continuation server is `ctx=65536`, `c=8`, `block=16`, shared-pool |
| `dflash_tree_config_sweep_single_req` | finished | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH_TREE` | `tree verify correctness + wall tok/s` | `wall_s_total + accept_length + correct_boxed_rate` | `/workspace/dflash_tree_config_sweep_20260330` | sweep finished: block `4` best is `steps=3 topk=4 vt=4` (`271.792 tok/s`, `1.065x`), block `8` best is `steps=4 topk=2 vt=6` (`299.849 tok/s`, `1.074x`), block `16` best is `steps=8 topk=1 vt=9` (`332.333 tok/s`, `1.127x`) |
| `tree_vs_linear_single_req_locked_postfix` | finished | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH/DFLASH_TREE` | `single-request apples-to-apples tree vs linear` | `wall_tok_s + accept_length + correct_boxed_rate` | `/workspace/tree_vs_linear_apples_20260330` | locked after benchmark-driver fixes: block `4` `304.201/274.047=1.110x`, block `8` `343.534/301.705=1.139x`, block `16` `385.327/315.758=1.221x`, all boxed-correct |
| `dflash_tree_batch_sweep_postfix` | running/partial | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH/DFLASH_TREE` | `batched non-overlap tree vs linear using locked best configs` | `wall_tok_s + accept_length + correct_boxed_rate + batch speedup` | `/workspace/dflash_tree_batch_sweep_20260330_postfix` | partial results locked: block `4` `c=1` `306.603/274.120=1.1185x`, block `4` `c=4` `1194.003/1179.283=1.0125x`, both boxed-correct; `c=8` currently unstable on tree verify and under active debug; overlap forced off |
| `dflash_tree_overlap_single_request_best` | prepared | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH_TREE` | `tree overlap correctness + wall tok/s` | `wall_s_total + accept_length + correct_boxed_rate + overlap speedup` | `/workspace/dflash_tree_overlap_single_request_best_20260330` | prepared follow-up: single-request overlap compare using the best tree configs first (`block=4` and `block=8`) |
| `dflash_tree_overlap_config_sweep_single_req` | prepared | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH_TREE` | `tree overlap speedup across config grid` | `wall_s_total + accept_length + correct_boxed_rate + overlap speedup` | `/workspace/dflash_tree_overlap_config_sweep_20260330` | overlap sweep reuses the linear baselines from `dflash_tree_config_sweep_20260330` and scans the same compact tree grid |

## 2026-04-04 Route Continuation Correction

The adaptive-cap route wave on `ckpt213` needs a clear correction because the early
continuation readout looked much better than the real long-context continuation state.

Locked facts for the current live route lane:

- run root: `/workspace/route_ckpt213_shared_adaptivecap_safe_20260404`
- serving contract:
  - shared-pool
  - target `page_size=1`
  - draft `page_size=1`
  - continuation `DFLASH block_size=16`
  - continuation server `context_length=65536`
  - continuation server `max_running_requests=8`
- problem set:
  - `86e8e5`
  - `dd7f5e`
  - `a295e9`
  - `9c1c5f`
  - `92ba6a`

Route-budget semantics from `scripts/playground/route_reference_dflash.py`:

- full per-request decode budgets for this 5-problem ladder are:
  - `64545`
  - `64687`
  - `64786`
  - `64814`
  - `64839`
- corresponding prompt lengths are only:
  - `185`
  - `210`
  - `238`
  - `337`
  - `479`
- exploration always gets an `8192` token cap per request
- exploration is chunked in `2048`-token rounds
- continuation receives the remaining budget after exploration
- because the router enforces `exploration_min_rounds=2`, continuation starts only after at least
  `4096` exploration tokens per surviving branch
- therefore the continuation output budget is at least in the range:
  - `60449..60743` if routing stops after 2 rounds
  - `56353..56647` if routing uses the full 8192-token exploration budget

Important correction:

- the earlier `1.7k-2.3k tok/s` continuation numbers were from the early window right after
  promotion, before the longer continuation matured
- the current real long-context continuation state is much worse
- live server-side continuation metrics are now:
  - GPU3 greedy `4->16`: about `331.5 tok/s`, `avg_spec_accept_length 1.011`
  - GPU4 greedy `8->16`: about `323.9 tok/s`, `avg_spec_accept_length 1.014`
  - GPU5 sampled `4->16`: about `298.9 tok/s`, `avg_spec_accept_length 1.000`
- live decode logs match that:
  - `accept_len ~1.00`
  - `accept_rate ~0.06`

Current live continuation token load, sampled during this bad tail:

- GPU3 greedy `4->16`: `67373` total live full tokens across 8 routed requests
  - about `8422` tokens/request on average
- GPU4 greedy `8->16`: `57133` total live full tokens across 8 routed requests
  - about `7142` tokens/request on average
- GPU5 sampled `4->16`: `47867` total live full tokens across 8 routed requests
  - about `5983` tokens/request on average

Conclusion:

- yes, continuation throughput is already falling sharply as the promoted continuation gets longer
- the route lane is not currently in a healthy long-context continuation regime
- do not compare the early continuation burst against the older locked route results
  (`803 tok/s`, accept `4.208`) without accounting for this long-context degradation

## 2026-04-04 Pure Target-Only Control Check

A direct target-only control was launched to test the user's claim that the current
shared-pool DFLASH route continuation is worse than plain target decode with
`kv_cache_dtype=fp8_e4m3`.

Control artifact target:

- `/workspace/target_only_fp8_ctx65536_c8_20260404/greedy_reference5_c8.json`

Control regime:

- target only, no DFLASH
- same long-decode serving contract:
  - `context_length=65536`
  - `concurrency=8`
  - `page_size=1`
  - `kv_cache_dtype=fp8_e4m3`
  - FA3
  - full CUDA graph + piecewise CUDA graph
  - overlap disabled
- prompt set:
  - the same 5-problem route ladder repeated to 8 requests
  - this is not the promoted continuation text from the route worker
  - it is still a valid same-contract long-decode control

Live result:

- target-only control entered real decode at about `650-720 tok/s`
- at the same time, the current DFLASH route continuation tail was only about `299-332 tok/s`

Immediate read:

- the user's claim is correct
- under the current bad continuation regime, plain target-only FP8 serving is roughly
  `1.96x` to `2.41x` faster than the DFLASH continuation tail
- this is a severe regression signal against the current route continuation path
- final benchmark JSON for the target-only control had not landed yet when this note was added,
  but the live decode comparison is already decisive enough to reject the current DFLASH
  continuation state as acceptable

## Important Baseline Finding

The branch now has a strong baseline result that must be preserved when evaluating DFlash,
PaCoRe, and sampled-target speculative variants.

### No-DFLASH Greedy Baseline

- run: `/workspace/showtime_baseline10_20260328_earlystop_nodflash`
- regime:
  - no DFLASH
  - no PaCoRe
  - `attempts=8`
  - `early_stop=4`
  - greedy:
    - `temperature=0.0`
    - `top_p=1.0`
    - `top_k=1`
- result:
  - final selected accuracy: `7/10`
  - any-correct / pass@8: `8/10`
  - wall time: `5227s` = `1h27m07s`

### No-DFLASH Sampled Baseline

- run: `/workspace/showtime_baseline10_20260328_earlystop_nodflash_sampled`
- regime:
  - no DFLASH
  - no PaCoRe
  - `attempts=8`
  - `early_stop=4`
  - sampled:
    - `temperature=1.0`
    - `top_p=1.0`
    - `top_k=50`
    - `min_p=0.02`
- result:
  - final selected accuracy: `10/10`
  - wall time: `5627s` = `1h33m47s`
  - delta vs greedy: `+400s` = `+6m40s` = about `+7.7%`

### Why This Matters

- sampled baseline achieved a large quality gain for a relatively modest wall-time increase
- this is the reference point for all later:
  - DFLASH runs
  - PaCoRe runs
  - DFLASH + PaCoRe runs
  - sampled-target speculative runs
- we should not evaluate a speculative path only against greedy quality if the non-speculative
  sampled baseline is already materially better

### Future DFLASH Sampling Note

For later sampled DFLASH experiments, the key design split is:

1. target samples as normal, draft remains greedy
2. target samples as normal, draft also samples

This distinction must be benchmarked explicitly. The first option matches the common
\"sampled target, greedy draft\" design used by systems such as vLLM; the second is possible
but must be treated as a separate regime.

## Comparison Protocol

Before claiming one regime is better than another, compare all of:

1. `final_accuracy`
2. `majority_vote_accuracy`
3. `pass_at_k`
4. `wall_s_total`
5. `wall_s_per_question`
6. `cost_per_correct`
7. `spec_accept_length` and `spec_verify_ct` if DFlash is enabled

## What Helps / What Does Not

Keep this section brutally concrete.

### Helps

- `showtime.py`-faithful `/generate` + local Harmony parser + local Python tool execution
- separating benchmark families by `tool/no-tool`, `greedy/sampled`, `PaCoRe on/off`, `DFLASH on/off`
- recording both quality and cost, not just throughput

### Does Not Yet Work

- native `/v1/responses` benchmark lane as a faithful replacement for showtime
- comparing PaCoRe vs non-PaCoRe without normalizing rounds, early-stop policy, and wall time
- interpreting high acceptance as guaranteed correctness

## Next Prepared Run

The next prepared route-focused study before the full 10-problem DFlash route sweep is:

- `explore=32`
- `explore_tokens=8192`
- `route=8`
- fixed physical `DFLASH block_size=8`
- adaptive / FailFast continuation disabled for this first focused pass

Problems:

- hardest: `86e8e5`
- harder: `dd7f5e`
- decently hard: `a295e9`
- medium: `9c1c5f`
- easiest: `92ba6a`

The purpose is to answer:

1. Is explore/route actually helping on this difficulty ladder?
2. Do we really get enough high-confidence / low-entropy routes to justify `route=8`?
3. Where do we already see hard tails inside the exploration pool?
4. Does fixed physical block `8` behave better than the current `16` ceiling before proper FailFast / adaptive scaling is enabled?

Prepared launcher:

- [run_route5_explore32_route8_block8.sh](/workspace/sglang-dflash-line/scripts/playground/run_route5_explore32_route8_block8.sh)

Finished mixed-pool follow-up:

1. greedy mixed-pool rerun with exploration cap `4` and continuation cap `8`
   - [run_route5_explore32_route8_block4to8_greedy.sh](/workspace/sglang-dflash-line/scripts/playground/run_route5_explore32_route8_block4to8_greedy.sh)
   - result: [result.json](/workspace/route5_explore32_route8_block4to8_greedy_20260329/result.json)
2. sampled mixed-pool rerun with the same `4 -> 8` block-cap split
   - sampled target settings: `temperature=1.0`, `top_p=1.0`, `top_k=50`, `min_p=0.02`
   - [run_route5_explore32_route8_block4to8_sampled.sh](/workspace/sglang-dflash-line/scripts/playground/run_route5_explore32_route8_block4to8_sampled.sh)
   - result: [result.json](/workspace/route5_explore32_route8_block4to8_sampled_20260329/result.json)

Current conclusion from these route runs:

- greedy `4 -> 8` was better than sampled `4 -> 8` on throughput
- sampled `4 -> 8` did not improve routed quality
- all promoted branches were still `hard_tail`
- so the route study is complete enough for now, and the next work item is again:
  - overlap-v2
  - fused / CUDA-graph / mixed-precision verify
  - linear + tree-verify completion

Current checkpoint:

- no-DFlash sampled baseline is still the strongest confirmed quality regime
- sampled full-round did not beat sampled early-stop
- mixed-pool route study is complete enough for now
- next engineering focus is:
  - overlap-v2
  - fused / CUDA-graph / mixed-precision verify
  - linear + tree-verify completion

## 2026-04-04 Pure DFLASH FP8 Draft-KV Investigation

This section supersedes the old "route study is complete enough for now" framing for the current branch.
The active engineering question is no longer routing policy first; it is whether pure DFLASH can inherit
the same kind of fast FP8 decode behavior as plain target-only GPT-OSS.

### Core architectural finding

- There is already a hidden decode-style DFLASH draft proposal path in:
  - [dflash_worker.py](/workspace/sglang-dflash-pagesize-fix-old/python/sglang/srt/speculative/dflash_worker.py)
  - env gate: `SGLANG_DFLASH_PROPOSAL_USE_DECODE=1`
- Before 2026-04-04, that path did not get matching draft decode CUDA graphs because the draft worker
  still captured `TARGET_VERIFY` graphs.
- Fixed in:
  - [cuda_graph_runner.py](/workspace/sglang-dflash-pagesize-fix-old/python/sglang/srt/model_executor/cuda_graph_runner.py)
- New behavior:
  - if `SGLANG_DFLASH_PROPOSAL_USE_DECODE=1` and the worker is the DFLASH draft worker,
    draft CUDA-graph capture now uses `ForwardMode.DECODE` with `num_tokens_per_bs=1`
  - this gives the draft worker its own fixed-shape decode graph path instead of reusing
    fixed-width `TARGET_VERIFY` graphs

### Immediate consequence

- no-scale `fp8_e4m3` draft-KV + shared pools + decode-style draft proposal now starts cleanly
- the previous blocker was draft CUDA-graph capture; that blocker is gone on the new architecture
- this was proven on the exact ckpt213 shared-pool contract with:
  - target KV `fp8_e4m3`
  - draft KV `fp8_e4m3`
  - page size `1`
  - shared pools `on`
  - no KV scales

### Exact route comparison now running

Root:

- [route_ckpt213_shared_decodeproposal_draftkv_20260404](/workspace/route_ckpt213_shared_decodeproposal_draftkv_20260404)

Contract:

- exact preserved route contract
- `explore32 -> route8 -> greedy -> 4 -> 16`
- ckpt213
- shared pools
- target KV `fp8_e4m3`
- decode-style draft proposal enabled

#### BF16 draft-KV control

- path:
  - [bf16_gpu3/result.json](/workspace/route_ckpt213_shared_decodeproposal_draftkv_20260404/bf16_gpu3/result.json)
- first real exploration decode window:
  - `accept_len ~2.04 - 2.36`
  - `accept_rate ~0.51 - 0.59`
  - `gen throughput ~2388 - 2733 tok/s`
- conclusion:
  - decode-style draft proposal + matching draft decode graphs is a real speedup path
  - this is the first healthy pure-DFLASH route lane on the new draft architecture

#### FP8 draft-KV, no scales

- path:
  - [fp8/result.json](/workspace/route_ckpt213_shared_decodeproposal_draftkv_20260404/fp8/result.json)
- first real exploration decode window:
  - `accept_len ~1.00`
  - `accept_rate ~0.25`
  - `gen throughput ~1140 - 1200 tok/s`
- conclusion:
  - the new architecture fixed startup and graph capture
  - but plain no-scale draft KV `fp8_e4m3` still collapses draft quality badly on the same route workload

### Current interpretation

- the graph/capture architecture was one blocker, and it is now largely fixed
- the remaining blocker is draft quality under no-scale `fp8_e4m3`
- because BF16 draft-KV on the same decode-style path is healthy while FP8 draft-KV is not, the
  current regression is now strongly isolated to draft numerical behavior rather than route policy
  or target verify architecture

### Active next isolate

- same exact route contract
- draft KV `fp8_e4m3`
- keep draft Q in bf16 while storing KV in fp8:
  - env: `SGLANG_DFLASH_DRAFT_FP8_KEEP_Q_BF16=1`
- live path:
  - [fp8_keepq/result.json](/workspace/route_ckpt213_shared_decodeproposal_draftkv_20260404/fp8_keepq/result.json)

Purpose:

1. If `fp8_keepq` recovers acceptance, the problem is likely the draft attention compute path
   around the FP8 Q cast.
2. If `fp8_keepq` still collapses, the problem is much more likely the no-scale KV storage path itself.

### 2026-04-04 BF16 draft fast path promoted to default policy

Code changes:

- [dflash_worker.py](/workspace/sglang-dflash-pagesize-fix-old/python/sglang/srt/speculative/dflash_worker.py)
  - decode-style draft proposal is now auto-enabled by default on the stable fast lane:
    - shared pools
    - `page_size=1`
    - greedy sampling
    - non-FP8 draft KV
- [cuda_graph_runner.py](/workspace/sglang-dflash-pagesize-fix-old/python/sglang/srt/model_executor/cuda_graph_runner.py)
  - matching DFLASH draft `DECODE` cuda-graph capture is now auto-enabled by default on the same stable lane

Meaning:

- the fast BF16 draft path no longer requires `SGLANG_DFLASH_PROPOSAL_USE_DECODE=1`
- the environment gate is still available for forcing behavior, but the stable BF16 path now opts into it automatically

Validation:

- `py_compile` clean for both files

Fresh proof status:

- a clean BF16 rerun on the new default path was launched
- it did not fail in DFLASH code
- it failed in target startup due GPU memory pressure from other live jobs on the machine

So the current blocker to the next apples-to-apples number is environment headroom, not the code-side default-path implementation.

### 2026-04-04 BF16 draft default-path proof run recovered cleanly

Environment was cleaned by killing a stale 8-GPU `train_dflash.py` job and clearing dead CUDA contexts.

Run:

- [route_ckpt213_shared_greedy_b4to16_bf16_default_20260404/result.json](/workspace/route_ckpt213_shared_greedy_b4to16_bf16_default_20260404/result.json)

Exact contract:

- ckpt213
- shared pools
- target KV `fp8_e4m3`
- draft KV `bfloat16`
- exact route contract `explore32 -> route8 -> greedy -> 4 -> 16`
- no `SGLANG_DFLASH_PROPOSAL_USE_DECODE` env override

Observed startup markers:

- `DFLASH draft cuda-graph mode overridden to DECODE to match decode-style draft proposal.`
- target server uses KV cache dtype `torch.float8_e4m3fn`
- draft server uses KV cache dtype `torch.bfloat16`

Locked exploration-phase result from the live benchmark:

- output token throughput: `2491.15 tok/s`
- total token throughput: `2852.64 tok/s`
- accept length: `2.29`

Representative exploration decode windows from the same run:

- `accept_len ~2.04 - 2.90`
- `accept_rate ~0.51 - 0.73`
- `gen throughput ~2355 - 3251 tok/s`

Meaning:

- the BF16 draft fast path is now proven healthy again on the new default policy, not only behind
  the old env-gated experiment
- the next remaining question is continuation throughput, not whether the default BF16 path works

### Active paired experiment: decode-style target verify

Run:

- [route_ckpt213_shared_greedy_b4to16_bf16_decodeverify_20260404/result.json](/workspace/route_ckpt213_shared_greedy_b4to16_bf16_decodeverify_20260404/result.json)

Change:

- same exact contract as above
- plus env `SGLANG_DFLASH_VERIFY_USE_DECODE_BATCH=1`

Purpose:

- test whether speculative target verify can directly reuse more of the plain target-only decode path
  instead of paying the current `TARGET_VERIFY` / `forward_extend()` cost structure
