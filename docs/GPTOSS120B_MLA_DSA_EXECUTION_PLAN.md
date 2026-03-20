# GPT-OSS-120B MLA + DSA Execution Plan

## Scope

This document is the current execution plan while the fixed-r512 extended
covariance run continues in the background.

Live run:

- run root:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210`
- main log:
  `/workspace/sglang_gpt_oss_care_runs/gptoss120b_care_fixedr512_big_bs8prepacked_resumev1_20260314_191210/conversion/pipeline.log`

This plan has two major tracks:

1. finish GPT-OSS-120B MLA conversion and prove inference compatibility
2. prepare the post-MLA DeepSeek-V3.2-style sparse-attention track

The clean local references for DeepSeek V3.2 are:

- `/root/2512.02556v1.clean.txt`
- `/root/DeepSeek_V3_2.clean.txt`

## Ground rules

- Do not stop or perturb the live fixed-r512 extended conversion run.
- Use the converted fixed-r checkpoints as the first serving/correctness targets.
- Keep MLA quality work ahead of healing.
- Treat DSA/NSA as a second-stage training problem, not a free inference-only
  remap.

## 24-step execution plan

### A. Live conversion and evaluation track

1. Keep the current fixed-r512 extended covariance run alive, checkpoint-safe,
   and resumable until covariance completes.
2. When covariance completes, preserve the merged covariance directory, manifest,
   and full conversion manifest as immutable run artifacts.
3. Convert the checkpoint to the clean fixed-r512 MLA artifact and verify the
   safetensors index contains only the intended MLA tensors.
4. Benchmark the new fixed-r512 checkpoint on the corrected HF MLA loader path
   against the original GPT-OSS-120B baseline.
5. Render a single side-by-side table for:
   - Original GPT-OSS-120B
   - Alpaca-only CARE-U/CARE-E
   - extended-data fixed-r512 CARE-U
6. Run the fixed long-context/domain packs on the new checkpoint, including:
   - AIMO3 short-context PPL
   - AIMO3 long-context PPL
   - combined tool/code/math/calib pack PPL
7. Run the near-lossless anchor conversion at fixed `r=1024` on the same
   extended covariance so `r=512` has a real reference point.
8. Compare `r=1024` against `r=512` to determine whether the main gap is
   compression-budget loss or a deeper conversion-objective failure.
9. If `r=1024` is also materially weak, classify the failure as
   objective/architecture mismatch rather than rank compression.

### B. GPT-OSS MLA serving compatibility in SGLang

10. Re-audit the GPT-OSS MLA runtime in
    `python/sglang/srt/models/gpt_oss.py` using the converted fixed-r512
    checkpoint as the primary correctness target.
11. Prove CPU-side structural correctness for the fixed-r512 checkpoint:
    model config resolution, per-layer attention layout, sink wiring, and
    sliding/full attention pattern.
12. Re-check the real MLA backends, not flex paths, for GPT-OSS compatibility:
    - `flashmla`
    - `flashinfer_mla`
    - `cutlass_mla`
    - `trtllm_mla`
13. Document exactly why `flashmla` cannot yet serve the current dynamic-rank
    CARE-E checkpoints and what remains to make it viable for GPT-OSS MLA.
14. Prove the fixed-r512 checkpoint can at least load and construct cleanly on
    the intended SGLang MLA serving path with:
    - bf16 KV cache
    - attention sinks preserved
    - alternating sliding/full layers preserved
15. Identify whether sliding-window GPT-OSS MLA is blocked by model wiring,
    backend metadata, or kernel support, and write that down precisely.
16. Create a compatibility matrix for GPT-OSS MLA serving:
    - fixed-r vs dynamic-r
    - sinks on/off
    - sliding-window on/off
    - backend by backend
17. If fixed-r512 is the only currently serveable family, define that as the
    first production checkpoint family and keep CARE-E serving as a follow-on
    task.

### C. CARE / CARE-E paper alignment and next MLA research step

18. Keep the paper-gap matrix current with the new fixed-r512 extended results
    and explicitly mark what is paper-faithful, missing, or our extension.
19. Re-check whether the current GPT-OSS path still oversimplifies K/V asymmetry
    into one shared latent rank where CARE suggests a richer treatment.
20. Re-check whether any zero-shot positional handling remains missing beyond
    the current MLA conversion path, especially around RoPE / decoupled-RoPE.
21. Decide after the extended fixed-r512 and fixed-r1024 runs whether the next
    pure-CARE experiment should be:
    - rounded fixed-r
    - CARE-E again
    - or not another covariance-only run at all
22. If fixed-r512 still misses parity materially after the extended run, promote
    the operator-aware zero-shot lane:
    - query-aware key weighting
    - attention-output-aware value weighting
    - token-logit diagnostics
23. Treat operator-aware zero-shot as the main innovation lane if covariance-only
    CARE is shown to saturate on GPT-OSS-120B.

### D. DeepSeek-V3.2 / DSA / NSA follow-on track

24. Convert the DeepSeek references into a GPT-OSS-specific DSA/NSA work plan:
    - DSA is continued training, not an algebraic post-hoc conversion
    - lightning indexer and sparse retrieval must be trained and integrated
    - SGLang NSA support is incomplete today, including the unfinished
      `NSABackendAdaptor`
    - after MLA checkpoint quality is acceptable, define the GPT-OSS-120B DSA
      training and serving milestones separately from MLA conversion itself

## Immediate missing blockers already known

### MLA serving blockers

- `flashmla` does not currently support the GPT-OSS CARE-E dynamic-rank path.
- `flashinfer_mla` is the closest MLA backend for dynamic-rank checkpoints, but
  the full GPT-OSS combination of dynamic rank + sinks + sliding-window is not
  yet proven.
- GPT-OSS fixed-r512 is the most realistic first serveable checkpoint family.

### DSA / NSA blockers

- `DeepSeekNSAAlgorithm` contains placeholder methods for representation pool
  construction and updates.
- `NSABackendAdaptor.adapt_for_attn_metadata()` is still `TODO`.
- DeepSeek V3.2 usage docs assume a model that was already continued-trained for
  DSA; GPT-OSS-120B will need its own training path rather than only checkpoint
  rewriting.

## Decision rule after the current run

If the extended fixed-r512 run still misses parity badly:

- do not default back to healing
- do not assume more covariance data alone will solve it
- use the `r=1024` anchor to separate capacity loss from objective failure
- if objective failure is confirmed, move to operator-aware zero-shot

