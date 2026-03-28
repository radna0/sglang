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
| Baseline no-DFlash showtime 10-problem early-stop run | running |
| DFlash + PaCoRe 10-problem showtime run | running / partial |

## Run Table

Append new rows here. Keep failed and superseded runs; do not erase them.

| run_id | status | serving_path | tool_regime | pacore | early_stop | speculative | quality target | speed target | out_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `showtime_smoke_92_nodflash` | planned | `baseline` | `showtime-harmony-tool` | `off` | `4` | `off` | `final_accuracy` | `wall_s_total` | `/workspace/showtime_baseline10_20260328_earlystop_nodflash` | canonical baseline lane over all 10 |
| `showtime_route10_pacore8` | running/partial | `DFLASH` | `showtime-harmony-tool` | `8->1` | `4` | `DFLASH` | `final_accuracy` | `wall_s_total + DFlash metrics` | `/workspace/showtime_harmony_route10_20260328` | current optimized PaCoRe lane |
| `showtime_smoke_92_dflash` | finished | `DFLASH` | `showtime-harmony-tool` | `off` | `4` | `DFLASH` | `final_accuracy` | `wall_s_total + DFlash metrics` | `/workspace/showtime_harmony_smoke_20260328` | `92ba6a` correct, Python tools exercised |
| `route5_explore32_route8_block8` | prepared | `explore32->route8` | `no-tool` | `off` | `router early-promotion` | `DFLASH` | `route selection quality + final routed quality` | `wall_s_total + route/explore/continue split` | `/workspace/route5_explore32_route8_block8_20260328` | focused 5-problem difficulty ladder with fixed physical block `8`, adaptive cap off |

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
