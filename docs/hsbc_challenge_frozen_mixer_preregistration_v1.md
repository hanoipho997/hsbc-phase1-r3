# HSBC stacked-LCU frozen-mixer control v1 — preregistration

Status at freeze: **PREREGISTERED; OUTCOME NOT YET RUN**  
Freeze date: 2026-08-31 (Europe/Brussels)  
Parent evidence: `runs/hsbc_challenge/audit_v1/priority_attacks_v1.json`

## Question and estimands

The v1 audit found a ten-seed mean increment for the trained one-layer mixer
over the analytic no-mixer scorer. This control asks whether optimizing the
mixer angle is necessary, or whether a randomly initialized but frozen mixer
already produces the effect.

For each original seed 500--509, initialize the four one-layer parameters
exactly as in `audit_priority_attacks_v1.py`, then run two controls. In the
**frozen-random** arm, hold parameter 3, the global `RX(phi)` mixer angle, at
that initialized value. In the **frozen-zero** arm, set `phi=0` before the
first update and hold it there. In both arms train parameters 0--2 (two
softmax logits and the cost angle) for the same 300 Adam updates, using the
same batch RNG stream, fraud rows, legitimate batch size, validation subset,
checkpoint cadence, and earliest strict-best checkpoint rule as the original
replication. The full four-vector gradient is evaluated and the `phi`
component discarded, so the statevector/gradient evaluation budget is
identical across all three L=1 arms; no saved work is reassigned.

Co-primary estimands:

`mean_seed AUPRC(trained mixer) - mean_seed AUPRC(frozen random mixer)`

and

`mean_seed AUPRC(frozen random mixer) - mean_seed AUPRC(frozen-zero mixer)`.

Descriptive diagnostic:

`mean_seed AUPRC(frozen-zero mixer) - AUPRC(S1 analytic filter)`.

All AUPRCs use the untouched full 56,962-row ULB test partition (99 frauds).
The trained and S1 score arrays are reused byte-for-byte from
`priority_scores_v1.npz`; no model or seed is selected using test outcomes.

## Uncertainty and decisions

Use 2,000 paired Poisson(1) test-row bootstrap draws, seed 20260901, with the
same row weights applied to every trained, frozen-random, frozen-zero, and S1
score array. Average the ten seed-specific AUPRCs inside each draw before
forming a difference. Report two-sided 97.5% percentile intervals for the two
co-primary contrasts (Bonferroni-equivalent familywise alpha 0.05), plus an
explicitly descriptive two-sided 95% interval for frozen-zero versus S1.

- **Mixer-angle training necessary at this resolution:** trained minus
  frozen-random is positive and its 97.5% interval is strictly above zero.
- **Mixer-angle training not identified:** otherwise. This licenses only that
  optimization of `phi` was not resolved; it does not by itself establish an
  architecture effect.
- **Frozen mixer architecture effect identified:** frozen-random minus
  frozen-zero is positive and its 97.5% interval is strictly above zero.
- If that architecture contrast contains zero, the proposal may not attribute
  the parent increment to merely inserting a random coherent mixer.
- Frozen-zero versus S1 is reported only to diagnose the contribution of
  optimizing the filter logits/cost angle; it carries no confirmatory label.

No `0.01` practical-tie override is used for this mechanistic ablation because
the parent effect is only about `0.0072`; the adjusted interval decisions are
primary and all point effects are printed without favorable-branch selection.

## Integrity and kill criteria

- Refuse to run if the preregistration or frozen parent score artifact hash
  differs from its source lock.
- Refuse to overwrite an output. Publish a JSON summary and compressed aligned
  score bundle atomically enough that the JSON records the score-bundle hash.
- Retain every seed. A non-finite seed makes the primary result unavailable;
  it is never replaced.
- Never balance or resample validation/test rows. Training keeps the original
  fraud/legitimate batch construction.
- A frozen-random arm is invalid if its final or selected `phi` differs from
  its initialized `phi` by more than `1e-15`; a frozen-zero arm is invalid if
  `|phi|>1e-15`. Either arm is invalid if test-row identity differs from the
  parent bundle or the full-gradient evaluation path is not used.

This protocol was written before any frozen-mixer outcome was computed.
