# GPT-OSS-120B CARE: Fixed-r512 vs CARE-E

This note tracks the clean comparison lane for the paper-faithful Alpaca `128 x 2048`
reproduction family.

Inclusion rules:

- `Original` means the corrected HF baseline path for stock GPT-OSS-120B.
- `CARE-E` means the validated clean-checkpoint `dp=8` reruns from the dynamic-rank
  checkpoint only.
- `CARE-fixed-r512` means the new fixed-r512 checkpoint built from the same Alpaca
  covariance collection and benchmarked on the same corrected HF MLA path.

Artifacts:

- Fixed-r512 checkpoint:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/converted_checkpoint_fixed_r512`
- CARE-E checkpoint:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/converted_checkpoint_clean`
- Shared covariance source:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/conversion/covariance`
- Fixed-r512 benchmark dir:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/fixed_r512_compare`
- CARE-E benchmark dir:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_repro_alpaca_dp8_debug/zero_shot_eval/care_e_compare`
- Generated matrix:
  `/root/sglang-gpt-oss-care-mla/docs/GPTOSS120B_CARE_FIXEDR512_VS_CAREE_MATRIX.md`

Current takeaway:

- `arc_easy`: fixed-r512 is only marginally above CARE-E on `acc_norm`
  (`0.259680` vs `0.258838`) and both are far below the original model (`0.765993`).
- `hellaswag`: fixed-r512 is slightly above CARE-E on `acc_norm`
  (`0.262796` vs `0.260108`), but again the gain is small.
- `mmlu`: fixed-r512 is below CARE-E on `acc`
  (`0.248398` vs `0.256231`).
- So far, replacing CARE-E with uniform `r=512` on Alpaca alone does **not** materially
  fix zero-shot parity. It is slightly better on two rows and worse on another, but not
  enough to change the conclusion.
- The next required anchor is `CARE-fixed-r1024`. Without it, `r=512` can look bad
  without telling us whether the failure is mainly the compression budget or the
  conversion family itself.

Current matrix:

| Task | Metric | Original | CARE-E | CARE-fixed-r512 |
| --- | --- | --- | --- | --- |
| arc_easy | acc_norm,none | 0.765993 | 0.258838 | 0.259680 |
| arc_challenge | acc_norm,none | 0.529010 | pending | pending |
| hellaswag | acc_norm,none | pending | 0.260108 | 0.262796 |
| piqa | acc_norm,none | 0.782372 | pending | pending |
| mmlu | acc,none | pending | 0.256231 | 0.248398 |
| openbookqa | acc_norm,none | 0.402000 | pending | pending |
| race | acc,none | 0.282297 | pending | pending |
| winogrande | acc,none | 0.683504 | pending | pending |
