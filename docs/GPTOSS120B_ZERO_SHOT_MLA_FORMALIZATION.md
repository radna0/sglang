# GPT-OSS-120B Zero-Shot MLA Conversion: Formal Problem Statement

## Why this document exists

We now have a structurally correct GPT-OSS-120B MLA conversion pipeline, but the
first trustworthy benchmark results showed a hard truth:

- native MLA conversion is possible
- zero-shot parity is not automatic

That means the real problem is not "can we rewrite the weights into MLA form?"
The real problem is:

- under a strict KV-cache budget,
- without brute-force finetuning,
- how do we convert a pretrained GPT-OSS-120B MoE attention stack into native
  MLA while preserving the teacher's actual behavior?

This document formalizes that problem and clarifies what is and is not a
dead-end.

## Notation

For one attention layer:

- `X in R^{T x d_model}`: hidden states over a sequence window
- `Q = X W_Q`
- `K = X W_K`
- `V = X W_V`
- `L = Q K^T / sqrt(d_h)`: attention logits
- `A = softmax(L + mask)`: attention probabilities
- `O = A V`: attention output

For the MLA student:

- `Z = X W_down`, where `W_down in R^{d_model x r}`
- `K_hat = Z W_up^K`
- `V_hat = Z W_up^V`

with rank `r << d_kv`.

For GPT-OSS there are extra details:

- RoPE / no-RoPE channel split
- attention sinks
- alternating sliding-window and full-attention layers
- MoE downstream blocks

Those are deployment constraints, but the core conversion question lives at the
attention-operator level.

## The hierarchy of objectives

There is a strict hierarchy here.

### 1. Weight-space objective

The weakest objective is:

`L_W = ||W_K - W_hat_K||_F^2 + ||W_V - W_hat_V||_F^2`

This is the classic low-rank fitting view.

It is a dead end for serious MLA conversion because:

- large weight-space error can be harmless if the corresponding directions are
  never used by real activations
- small weight-space error can still produce large attention errors after
  query interaction and softmax amplification

This is the fundamental limitation of naive SVD-on-weights.

### 2. Activation-space objective

The next objective is:

`L_X = E[||X(W_K - W_hat_K)||_F^2 + ||X(W_V - W_hat_V)||_F^2]`

If `C_X = E[X^T X]`, then this becomes:

`L_X = tr((W_K - W_hat_K)^T C_X (W_K - W_hat_K))`
`    + tr((W_V - W_hat_V)^T C_X (W_V - W_hat_V))`

This is the core improvement in CARE over weight-only fitting.

This is already much better than TransMLA-style brute-force weight fitting
because it says:

- preserve directions the teacher actually uses
- not directions that merely have weight energy

But it is still indirect.

### 3. Attention-logit objective

What really matters for routing is:

`L_logit = E[||Q K^T - Q K_hat^T||_F^2]`

Equivalently:

`L_logit = E[||Q (K - K_hat)^T||_F^2]`

This is stronger than `L_X` because it measures the quantity that actually
drives the attention map.

### 4. Attention-output objective

What really matters for content transport is:

`L_out = E[||A V - A_hat V_hat||_F^2]`

Even if K/V projections look locally good, the output can drift because:

- small logit errors become large probability errors after softmax
- probability errors change value mixing
- the residual stream changes

### 5. Token-logit objective

The final objective is:

`L_tok = E[KL(p_teacher(. | x) || p_student(. | x))]`

This is the ultimate behavior-level objective, but it is also the most
expensive one because it includes:

- attention drift
- residual accumulation
- MLP/MoE downstream amplification
- final LM head effects

## What was wrong with older MLA conversion thinking

The brute-force dead end was:

- fit `W`
- if that fails, train more

That is not a principled formulation. It ignores the operator the model
actually computes.

CARE improved that substantially by moving from:

- weight-space fitting

to:

- activation-aware fitting via `C_X`

That is already a major conceptual advance.

But the user's key intuition is correct:

- even `XW` is not the full target
- the teacher and student will not share the exact same hidden trajectories
- what matters is preserving the attention operator and, ultimately, token
  behavior

So the formal target is not merely:

- reconstruct `W`
- or reconstruct `XW`

It is:

- preserve `QK^T`
- preserve `softmax(QK^T)`
- preserve `A V`
- preserve token logits

## Why the papers often stop before that

Because exact operator-aware conversion is much more expensive.

### Exact attention-map matching is quadratic

For sequence length `T`, the full attention map is `T x T`.

If we try to directly preserve:

- logits
- probabilities
- outputs

across large windows, we immediately pay:

- `O(T^2)` memory
- `O(T^2)` statistics collection
- high cost across many layers and many samples

For GPT-OSS-120B, this is worse because:

- there are many layers
- the model is large
- MoE amplifies downstream drift
- long-context behavior matters

So the researchers were not being irrational. They were trading exactness for a
closed-form, scalable method.

## The real non-brute-force path

The correct next step is not "go back to SGD." It is:

- keep zero-shot closed-form structure
- move one level closer to the real operator
- do it with second-order or sketched statistics, not full optimization

That gives a new family of methods.

## A stronger zero-shot formulation

The right target is a zero-shot operator-aware objective:

`L_total = lambda_X L_X + lambda_K L_logit + lambda_V L_out + lambda_T L_tok_diag`

where:

- `L_X` preserves teacher-used activation directions
- `L_logit` preserves routing geometry
- `L_out` preserves value transport
- `L_tok_diag` is not used to train weights directly, but to diagnose whether a
  zero-shot conversion is behavior-preserving

The key design constraint is:

- do not optimize model weights with brute force
- collect reusable statistics once
- solve low-rank problems in closed form or near-closed form

## Closed-form operator-aware extensions

### A. Query-aware key objective

For keys, a better objective than plain `C_X` is:

`L_K = E[||Q (K - K_hat)^T||_F^2]`

Under a second-order approximation, this induces a stronger weighting than
plain `C_X`. A practical surrogate is:

`C_K^Q ~= C_X W_Q W_Q^T C_X`

and the key-side approximation becomes:

`L_K_sur ~= tr((W_K - W_hat_K)^T C_K^Q (W_K - W_hat_K))`

Interpretation:

- plain CARE asks which input directions have high variance
- query-aware CARE asks which input directions are actually amplified by the
  query geometry

This is already closer to the attention operator while staying second-order and
closed-form.

### B. Attention-output-aware value objective

For values, the relevant quantity is not query geometry but transport under the
teacher attention operator.

A natural target is:

`L_V = E[||A (V - V_hat)||_F^2]`

This induces a value weighting matrix of the form:

`C_V^A = E[X^T A^T A X]`

so a value-side surrogate becomes:

`L_V_sur ~= tr((W_V - W_hat_V)^T C_V^A (W_V - W_hat_V))`

This is harder than the key case because `A` depends on `softmax(QK^T)`.

But it is still zero-shot if we:

- sample teacher windows
- compute teacher attention only on those windows
- accumulate `X^T A^T A X`
- use that as a weighting matrix

No SGD is required.

### C. Separate key and value objectives

This matters because K and V are not symmetric.

Keys determine:

- routing
- attention support
- probability mass

Values determine:

- transported content
- residual perturbation

So the correct formulation is not one shared objective for both.

It is:

- key rank / subspace chosen for routing fidelity
- value rank / subspace chosen for transport fidelity

That is a more principled version of the intuition behind CARE-E's separate K/V
allocation.

## The "mathematical genius" bypass to full quadratic maps

The cleanest bypass is to use operator sketches instead of full maps.

### A. Hutchinson / random probe idea

For any matrix `M`:

`||M||_F^2 = E_omega ||M omega||_2^2`

when `omega` is a random probe with identity covariance.

This means we do not need to materialize the full operator to estimate its
energy.

Applied to attention logits:

`M = Q (K - K_hat)^T`

Then:

`||M||_F^2 = E_omega ||Q (K - K_hat)^T omega||_2^2`

So instead of storing the full `T x T` attention-logit matrix, we can estimate
its distortion with a small number of random sequence probes.

Applied to outputs:

`M = A (V - V_hat)`

Then:

`||M||_F^2 = E_eta ||A (V - V_hat) eta||_2^2`

with random probes `eta` over value channels.

This is the key non-brute-force idea:

- preserve the operator
- without storing or matching the full operator exactly
- by using random sketches

That is mathematically principled and much cheaper than full attention-map
distillation.

## A practical zero-shot research program

The disciplined path is:

### Stage 0. Base CARE initialization

Collect `C_X` and build the current fixed-r or CARE-E checkpoint.

### Stage 1. Query-aware key weighting

Augment the key-side weighting with:

- `C_X`
- `W_Q`
- optionally sampled query-window statistics

This is still cheap and near-closed-form.

### Stage 2. Attention-output value weighting

Collect sampled-window teacher attention output statistics and build
`C_V^A`.

This is more expensive than plain covariance, but still zero-shot and still
much cheaper than full training.

### Stage 3. Token-logit diagnostics

Do not optimize on token logits yet. Use them as the acceptance criterion.

Measure on fixed packs:

- next-token KL
- top-1 agreement
- top-k overlap
- teacher-token NLL under the student

That tells us whether the new zero-shot objective is actually improving the
right thing.

## Why this is better than brute-force optimization

Brute force says:

- choose a target
- backprop through the model
- hope the optimizer repairs the mismatch

The operator-aware zero-shot path says:

- define the right invariants
- collect reusable second-order or sketched operator statistics
- solve the compression problem in closed form or almost-closed form
- use token-level behavior only as validation, not as the primary training loop

This is more mathematically disciplined because:

- it preserves structure
- it separates calibration from optimization
- it produces reusable artifacts across many target ranks
- it scales better to repeated rank sweeps

## Why GPT-OSS-120B is harder than the paper baselines

The papers often work on smaller or simpler models.

GPT-OSS-120B is harder because:

- it is large
- it is MoE
- long-context behavior matters
- tool-calling/code/math tails matter
- sinks and sliding-window semantics matter

That means average-prose covariance alone is unlikely to be enough.

But it does not mean zero-shot is impossible. It means zero-shot must become
more operator-aware.

## The current project state

Right now:

- the active run collects only activation covariance `C_X`
- that is already far better than weight-only fitting
- but it is still indirect

So the next escalation, if the fixed-r512 extended run still misses parity,
should not be "healing first."

It should be:

1. keep the current covariance-based initialization
2. add query-aware key weighting
3. add sampled-window value/output weighting
4. keep token-logit evaluation as the truth criterion

## Bottom line

The user's core intuition is correct.

The real object to preserve is not:

- the weights
- nor only the projected activations

It is:

- attention routing
- attention transport
- token behavior

The mathematically disciplined way to get there is not brute-force finetuning.
It is:

- operator-aware zero-shot conversion via second-order and sketched statistics

That is the strongest non-brute-force extension of CARE for GPT-OSS-120B.
