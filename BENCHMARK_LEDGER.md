# Benchmark Ledger

This file is the canonical run registry for GPT-OSS `showtime.py` / Harmony /
PaCoRe / DFlash experiments on this branch.

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
| `dflash_tree_config_sweep_single_req` | finished | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH_TREE` | `tree verify correctness + wall tok/s` | `wall_s_total + accept_length + correct_boxed_rate` | `/workspace/dflash_tree_config_sweep_20260330` | sweep finished: block `4` best is `steps=3 topk=4 vt=4` (`271.792 tok/s`, `1.065x`), block `8` best is `steps=4 topk=2 vt=6` (`299.849 tok/s`, `1.074x`), block `16` best is `steps=8 topk=1 vt=9` (`332.333 tok/s`, `1.127x`) |
| `tree_vs_linear_single_req_locked_postfix` | finished | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH/DFLASH_TREE` | `single-request apples-to-apples tree vs linear` | `wall_tok_s + accept_length + correct_boxed_rate` | `/workspace/tree_vs_linear_apples_20260330` | locked after benchmark-driver fixes: block `4` `304.201/274.047=1.110x`, block `8` `343.534/301.705=1.139x`, block `16` `385.327/315.758=1.221x`, all boxed-correct |
| `dflash_tree_batch_sweep_postfix` | running/partial | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH/DFLASH_TREE` | `batched non-overlap tree vs linear using locked best configs` | `wall_tok_s + accept_length + correct_boxed_rate + batch speedup` | `/workspace/dflash_tree_batch_sweep_20260330_postfix` | partial results locked: block `4` `c=1` `306.603/274.120=1.1185x`, block `4` `c=4` `1194.003/1179.283=1.0125x`, both boxed-correct; `c=8` currently unstable on tree verify and under active debug; overlap forced off |
| `dflash_tree_overlap_single_request_best` | prepared | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH_TREE` | `tree overlap correctness + wall tok/s` | `wall_s_total + accept_length + correct_boxed_rate + overlap speedup` | `/workspace/dflash_tree_overlap_single_request_best_20260330` | prepared follow-up: single-request overlap compare using the best tree configs first (`block=4` and `block=8`) |
| `dflash_tree_overlap_config_sweep_single_req` | prepared | `DFLASH_TREE` | `no-tool` | `off` | `off/full-round` | `DFLASH_TREE` | `tree overlap speedup across config grid` | `wall_s_total + accept_length + correct_boxed_rate + overlap speedup` | `/workspace/dflash_tree_overlap_config_sweep_20260330` | overlap sweep reuses the linear baselines from `dflash_tree_config_sweep_20260330` and scans the same compact tree grid |

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
