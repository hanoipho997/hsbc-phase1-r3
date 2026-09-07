# Engine-A unblinding report — v1 (FINAL)

**Run:** 2026-08-31, single execution of
`scripts/hsbc_challenge/engine_a_unblind_v1.py --real` against preregistration
`docs/hsbc_engine_a_preregistration_v1.md` (SHA-256
`0d9825231ca590baf7d9cb12b9d28958a7ac3b4eff42fac5f6af609f746102a8`,
committed as `953bebf` before the seal was opened). Artifact:
`runs/hsbc_challenge/engine_a_v1/unblind_real_v1.json`. The sealed test
partition was read once, by this run only. No re-runs, no post-hoc changes.

## Verdict

**FAIL** — G1 ✗, G2 ✗, G3 ✗, G4 ✗. Per the preregistration's pre-drafted
wording: *"The preregistered routed-syndrome experiment did not meet its
gates. We report it in full."* The submission leads with Engine B
(conditions map + hardware-validated mechanism and training-economics
results) and this null as evidence of protocol integrity.

## Test-set facts (chronological final 20%; first and only read)

- 118,108 rows, 4,064 frauds (3.441% — drifted down from validation's
  3.904%).
- Frozen routing under drift: auto-flag captured 167 rows (val projection
  237) at **100% fraud precision** (167/167); the review pool thinned to
  896 rows / 790 frauds (88.2%; val projection 1,181/1,084); 3,107 frauds
  remained below the band.
- Backbone full-test AUPRC **0.52939** vs 0.65195 on validation — the tuned
  classical backbone itself degrades substantially under one further period
  of drift.

## Endpoints (paired bootstrap, 10,000 resamples, seed 1234)

| arm | AUPRC | ΔAUPRC vs backbone [95% CI] | recall@budget | ΔRecall [95% CI] |
|---|---|---|---|---|
| backbone | 0.52939 | — | 0.13066 | — |
| **Q (syndrome)** | 0.52709 | **−0.00230 [−0.00409, −0.00068]** | 0.13066 | +0.00032 [−0.00099, +0.00174] |
| C-a logistic-12 | 0.52835 | −0.00103 [−0.00291, +0.00064] | 0.12992 | −0.00068 [−0.00244, +0.00099] |
| C-b MLP(8) | 0.52266 | −0.00660 [−0.01027, −0.00336] | 0.12869 | −0.00167 [−0.00349, +0.00000] |
| C-c PCA-residual | 0.52557 | −0.00382 [−0.00577, −0.00204] | 0.12894 | −0.00149 [−0.00347, +0.00050] |
| C-f noise control | 0.52218 | **−0.00717 [−0.01046, −0.00406]** | 0.13066 | −0.00007 [−0.00147, +0.00121] |
| Q − C-a (G3 contrast) | — | −0.00126 [−0.00314, +0.00048] | — | +0.00100 [−0.00049, +0.00256] |

Gate readings: G1 (CI excludes 0 favorably): no. G2 (ΔAUPRC ≥ +0.005 or
ΔRecall ≥ +2.0 pp): no — as the A4 reachability analysis anticipated for
the co-primary. G3 (beats every control with resolved CI): no — Q is
statistically inseparable from the matched classical residual (−0.0013
[−0.0031, +0.0005] AUPRC; +0.10 pp [−0.05, +0.26] recall). G4 (no-harm
floor −0.001): **no — Q's −0.0023 is a resolved harm**, and this failure is
shared by the design, not the circuit (see F2).

## Findings

**F1 — The classical frontier holds under drift.** Nothing — quantum or
classical — recovered operationally meaningful headroom in the backbone's
own review band on the drifted test period. Within-pool reranking
(co-primary) is null for every arm; the validation-period hint of a small
positive band effect (+0.16 pp, both Q and C-a) did not survive the
temporal shift. Combined with the A4 headroom bound (< 1 pp capturable at
K=400 even for a perfect reranker), this is a quantitative
classical-frontier statement: this backbone's review band is not where
residual models — of any kind — add value on this dataset.

**F2 — Score-replacement hybridization is the design failure, isolated by
the noise control.** Every residual arm *reduced* full-test AUPRC, and the
sharpest reduction (−0.0072, resolved) belongs to C-f, whose features are
pure noise: its head can only be a validation-pool-calibrated monotone
recalibration of the backbone logit, so its harm must come entirely from
the §6 score-replacement mechanics — pool scores re-mapped through
val-period calibration interleave incorrectly with out-of-pool backbone
scores under test-period drift. Lesson for any future design (recorded for
future work, not re-analysis): hybrid scores must preserve cross-region
calibration (e.g., rank-preserving within-band blends), and G4-style
no-harm gates should be evaluated under a drift rehearsal, not only
in-period.

**F3 — No quantum-specific effect, in either direction.** The syndrome arm
is indistinguishable from its matched classical residual on both endpoints
(G3 contrast CIs straddle zero). The measured chain across both datasets is
now consistent: the cross-view syndrome is a competitive, circuit-realizable
residual family — never separable from a matched classical one at these
scales. The proposal's H4 status updates to: **FAIL, preregistered, on
IEEE-CIS temporal split** (ULB: null; IEEE-CIS: null-to-harmful via the
replacement design).

## What ships

Per the preregistration's FAIL branch: the Phase-1 submission leads with
Engine B — the measured conditions map (ULB + IEEE-CIS, now including F1's
drift result and the A4 headroom bound), the hardware-validated mechanism
package (failure-sector gradients, flag-NG, syndrome observables — all
machine-precision verified), and this preregistered null as the
integrity demonstration. The proposal's §5 hypothesis table and §6
deployment text must be updated to cite this report (P0 before submission);
no claims of application lift survive anywhere in the document.

## Deviations

None. Amendments A1–A6 were all logged pre-unblinding; the test partition
was read once; every number above regenerates from the committed scripts
and `unblind_real_v1.json`.
