# HSBC partial-dephasing witness — analysis amendment v2

Status: **POST-OUTCOME CORRECTION; NOT A PREREGISTRATION**  
Date: 2026-09-04 (Europe/Brussels)  
Supersedes: the uncertainty intervals and gate interpretations in
`docs/hsbc_challenge_dephasing_witness_preregistration_v1.md` and
`docs/hsbc_challenge_claude_novelty_dossier_v1.md`; it does not alter the v1
raw artifacts.

## 1. Why an amendment is required

The frozen v1 protocol specified a 32-row non-parametric bootstrap with
replacement and said that uncertainty combined row and shot noise. The v1
runner instead collapsed duplicate bootstrap indices with `numpy.unique` and
used one already-realized count table inside every bootstrap replicate.
Therefore the printed v1 intervals are not the specified bootstrap and do not
propagate multinomial shot uncertainty.

The multiplicity statement was also incomplete. It adjusted two named
contrasts but then required the ideal result to pass independently for five
model seeds. The intended decision family is conservatively treated here as
two contrasts times five seeds, or ten decisions.

The v1 low-noise calculation used only model seed 900. That choice was present
in the runner but absent from the frozen protocol, and exact noisy
probabilities were not saved. It cannot establish the declared five-seed
noise-family gate. The reduced eight-row target was selected after the
full-width result and is exploratory.

## 2. Corrected analysis fixed before rerun

Script:
`scripts/hsbc_challenge/audit_dephasing_uncertainty_v2.py`  
Input:
`runs/hsbc_challenge/novelty_v1/dephasing_witness_curves_v1.npz` and
`runs/hsbc_challenge/novelty_v1/braket_reduced_instance_v1.npz`  
Output:
`runs/hsbc_challenge/novelty_v2/dephasing_uncertainty_v2.json`

For each bootstrap replicate:

1. draw 32 row indices with replacement and retain duplicates;
2. independently draw 2,000 multinomial shots for each selected row at
   `q=0` and `q=1` from the saved exact probability arrays;
3. recompute the frozen unbiased U-statistics; and
4. record `W_hat(0)-W_hat(1)`.

There are 10,000 replicates with root RNG seed 20260904. To control the
familywise error rate at 0.05 over the intended ten decisions, each ideal-seed
interval is a two-sided Bonferroni 99.5% percentile interval. A separate
fixed-row 97.5% interval shows shot variation only and is descriptive.

For saved noise counts, the v2 script draws from each row's observed
frequencies. Those intervals are explicitly **conditional plug-in
sensitivity analyses**, not confirmatory noise-model intervals. They cannot
replace the missing exact noisy probabilities or four missing model seeds.

## 3. Corrected results

### 3.1 Full-width ideal simulator

| model seed | exact `W(0)-W(1)` | corrected nested row+shot 99.5% interval | fixed-row shot-only 97.5% interval |
|---:|---:|---:|---:|
| 900 | 0.073956 | [0.020622, 0.116585] | [0.070110, 0.077862] |
| 901 | 0.031739 | [0.013184, 0.046139] | [0.029642, 0.033888] |
| 902 | 0.053077 | [0.015832, 0.081643] | [0.050306, 0.055960] |
| 903 | 0.055267 | [0.014847, 0.087280] | [0.051971, 0.058585] |
| 904 | 0.044891 | [0.014129, 0.068904] | [0.042205, 0.047487] |

**Corrected verdict:** `CONFIRMED` for ideal-simulator resolvability. All five
family-adjusted lower bounds remain above zero. This confirms a simulated,
finite-shot contrast in between-row syndrome variation; it is not device or
QPU evidence.

### 3.2 Saved noise-count sensitivity

| stand-in | observed contrast | conditional plug-in nested 97.5% interval | verdict |
|---|---:|---:|---|
| low, seed 900 only | 0.015051 | [0.005957, 0.022122] | positive sensitivity result; confirmatory family gate unresolved |
| high, seed 900 only | 0.000140 | [-0.000230, 0.000493] | unresolved / no resolvable signal |

**Corrected verdict:** the claim that the declared low-noise primary gate
passed is `REFUTED AS A FAMILY CLAIM`. Only a one-seed, plug-in sensitivity
survives. The earlier positive high-noise interval was an artifact of the
incorrect resampling routine and does not survive.

### 3.3 Reduced instance

For the eight-row, six-system-qubit random-parameter instance, the exact
contrast is 0.019828 and the exploratory nested 95% interval is
[0.003373, 0.035859]. This is a useful prospective design candidate, not a
selected confirmatory target: it uses validation-derived rows, one random
configuration, and was chosen after the full-width result.

## 4. Claim boundary and proposal-safe wording

The verified endpoint theorem and density-matrix referees remain intact:
complete within-layer ancilla dephasing makes the fresh-ancilla pattern law
input independent at the declared insertion point, with numerical agreement
to `8.3e-16` in the small-width referee. The appropriate empirical term is
**between-row syndrome variation**, not “information.”

The proposal may say:

> For the fresh-ancilla binary S-LCU implementation, complete path dephasing
> at the declared insertion point makes the ancilla-pattern law input
> independent. Exact and density-matrix simulations verify that endpoint,
> and a corrected nested row-and-shot analysis on 32 validation rows resolves
> the coherent-to-dephased contrast at 2,000 shots for all five model seeds
> after a conservative ten-decision correction (smallest 99.5% lower bound
> 0.0132). This is simulator evidence for a prospective hardware mechanism
> test, not evidence of fraud lift, runtime advantage, classical hardness, or
> QPU performance.

It may not say that the witness works on hardware, passes a device-noise
family gate, or is ready for a QPU batch.

## 5. Prospective QPU protocol required before execution

A real QPU cannot execute the simulator-only `phase_flip` noise instruction.
Implement `D_q` by randomized physical `I/Z` interventions with fixed
allocation `q/2` to `Z` and `1-q/2` to `I`, randomized job order, and
aggregation specified before data collection. The next protocol must freeze:

- a public or governance-cleared input set;
- one device, native gate set, real connectivity, compiler version, and
  calibration snapshot;
- identical-branch and phase-twirl references;
- five-model or explicitly single-model scope;
- a device-calibrated power analysis and familywise uncertainty rule;
- a shot/task budget including reference circuits; and
- abort rules for drift in calibration or reference-circuit failure.

The base reduced design alone is `8 rows x 5 q values x 2,000 shots = 80,000`
shots, before references, repeats, and calibration. Cost and schedule must be
estimated from the selected device's current Braket pricing before approval.

## 6. Data governance

The v1 curve NPZ contains 32 transformed IEEE-CIS validation rows and labels;
the reduced NPZ contains eight transformed validation rows. Both are
`PRIVATE_LICENSED_IEEE_CIS_DERIVED`. They must not enter a public release or
VM handoff unless the competition licence and the intended processing permit
it. A public package should contain only aggregate JSON results, theorem and
synthetic-referee fixtures. Sending derived rows to AWS also requires the
project's data-governance approval.

