# HSBC revision-3 novelty dossier v2 — corrected evidence package

**Date:** 2026-09-04 (Europe/Brussels)  
**Status:** replaces Claude dossier v1 for proposal decisions; v1 is retained
as an immutable audit input  
**Scope:** post-audit novelty and Phase-2 positioning; no proposal source is
edited here  
**Top-level verdict:** **GO for revision 3 only with the narrow simulator-to-
hardware mechanism story below. No quantum-advantage or application-lift
claim is supported.**

## Executive decision

Claude's v1 dossier found useful assets, but it promoted three conclusions
beyond their evidence. The corrected package makes one addition proposal-
eligible:

1. **Selected:** the complete-dephasing endpoint theorem plus a partial-
   dephasing simulator pilot, framed as a prospective hardware interference
   test. The corrected nested row-and-shot analysis remains positive for all
   five model seeds after a conservative ten-decision correction. It is not
   hardware evidence.
2. **Demoted:** the splice ablation is a deployment diagnostic. Both locally
   frozen in-period causal contrasts are unresolved; a secondary
   rolling-origin result is consistent with score interleaving but cannot
   identify the cause of the sealed Engine-A loss.
3. **Appendix only:** the compile counts are generic Qiskit basis/topology
   transpilation envelopes, not IonQ- or IQM-native compilations.

The X-readout derivative identity, dephased endpoint, exact classical path
twin, mixer replication, frozen-mixer control, rival bracket, and Engine-A
FAIL remain the scientific core. Classical OCC and supervised baselines remain
as good as or better than the quantum scorer everywhere measured. Phase 2 is
fundable only as a bounded hardware-mechanism and model-risk study with stop
rules—not as a promised fraud-performance or FTQC-advantage program.

## 1. Source lock and evidence classes

This document audits rather than overwrites:

- `docs/hsbc_challenge_claude_novelty_dossier_v1.md`;
- `docs/hsbc_challenge_audit_report_v2.md`;
- `runs/hsbc_challenge/novelty_v1/`;
- `runs/hsbc_challenge/engine_a_v1/`; and
- the locally frozen pre-outcome protocol files for the dephasing and splice
  experiments.

The protocol files were created locally before their corresponding outcomes,
but were not externally timestamped or registered. The accurate provenance
term is **locally frozen pre-outcome protocol**, not formal preregistration.

Evidence labels used below:

| label | meaning |
|---|---|
| `MECHANICAL` | exact identity or deterministic source/artifact check |
| `SIMULATOR` | exact or sampled simulator result |
| `EXPLORATORY` | post-outcome, underpowered, incomplete-family, or secondary analysis |
| `HARDWARE` | managed simulator or QPU data; none exist in this package |
| `PRIVATE_LICENSED` | contains IEEE-CIS-derived rows or labels; not public-release safe by default |

## 2. Claim-by-claim corrections to dossier v1

### 2.1 Partial-dephasing witness

**Verdict: CORRECTED; ideal-simulator signal survives.**

The theorem is unchanged. For fresh ancillas, real binary PREP, unitary
branches, and within-layer dephasing inserted after SELECT and before
`PREP^dagger`, complete dephasing removes cross-path terms. The measured
ancilla-pattern law becomes the product distribution determined by the PREP
weights and is independent of the input. Exact Gram-path and density-matrix
referees agree to at most `8.3e-16` in the tested small-width cases.

The v1 uncertainty code did not implement its frozen protocol: it collapsed
duplicate bootstrap rows with `numpy.unique` and did not redraw shot counts.
The corrected post-outcome audit retains duplicates, redraws independent
multinomial counts inside every row replicate, and uses two-sided 99.5%
Bonferroni intervals for the intended two-contrast by five-seed family.

| seed | exact `W(0)-W(1)` | corrected nested row+shot 99.5% interval |
|---:|---:|---:|
| 900 | 0.073956 | [0.020622, 0.116585] |
| 901 | 0.031739 | [0.013184, 0.046139] |
| 902 | 0.053077 | [0.015832, 0.081643] |
| 903 | 0.055267 | [0.014847, 0.087280] |
| 904 | 0.044891 | [0.014129, 0.068904] |

All five ideal-simulator lower bounds remain positive. The result supports
finite-shot resolvability of **between-row syndrome variation** in this
simulator study. “Information” is avoided because no predictive-information
estimand was tested.

The noise conclusion is narrower. Only seed 900 was run, exact noisy
probabilities were not saved, and the restriction was not in the protocol.
Conditional plug-in nested 97.5% intervals from the saved counts are:

| stand-in | observed contrast | corrected interval | status |
|---|---:|---:|---|
| low | 0.015051 | [0.005957, 0.022122] | exploratory seed-900 sensitivity |
| high | 0.000140 | [-0.000230, 0.000493] | unresolved; no signal |

Therefore “the low-noise primary family gate passed” is withdrawn. The
eight-row reduced target has exact contrast 0.019828 and exploratory nested
95% interval [0.003373, 0.035859], but was chosen after the full-width result
and uses one random parameter configuration. It is a design candidate, not a
confirmatory hardware target.

Full method and claim boundary:
`docs/hsbc_challenge_dephasing_witness_analysis_amendment_v2.md`.

### 2.2 Engine-A splice ablation

**Verdict: CORRECTED; causal identification refuted.**

The load-bearing in-period contrasts are:

| contrast | point | simultaneous / Bonferroni 97.5% interval | decision |
|---|---:|---:|---|
| keyed-noise `C_f`, M1 minus M2 | -0.003828 | [-0.007808, 0.000149] | unresolved |
| circuit `Q`, M1 minus M2 | -0.000618 | [-0.003278, 0.002087] | unresolved |

The secondary rolling-origin `C_f` contrast is -0.011851
[-0.016976, -0.006807]. That analysis is consistent with harmful
cross-region score interleaving, but its heads and hyperparameters had already
used the full validation period. It is not a confirmatory causal explanation
for the sealed Engine-A loss.

Three v1 sentences are withdrawn:

- “the mechanism is identified”;
- “every M2 arm's harm disappears”—rolling-origin `C_b|M2` is -0.001472 with
  95% interval [-0.003067, -0.000031]; and
- “no arm improves internal order resolvably”—in-period `Q|M2` is +0.001182
  [0.000463, 0.001941] and `C_a|M2` is +0.00115 [0.00021, 0.00210] at the
  reported unadjusted 95% level.

Proposal-safe replacement:

> A secondary rolling-origin validation analysis is consistent with
> cross-region score interleaving, while both locally frozen in-period
> contrasts were unresolved. This does not identify the causal mechanism of
> the sealed Engine-A loss. Rank-preserving score integration remains a
> deployment safeguard to test prospectively.

The splice result is a deployment correction and diagnostic, not one of the
two selected novelty additions.

### 2.3 Compilation artifact

**Verdict: CORRECTED; counts confirmed, target-native label refuted.**

The ten-seed Qiskit optimization-level-3 distributions are mechanically
reproducible:

| abstract target | two-qubit-gate count |
|---|---:|
| all-to-all `{rx,ry,rz,rxx}` | 155 RXX |
| all-to-all `{cz,rz,sx,x}` | 238 CZ |
| synthetic full 4x5 square grid `{r,cz}` | 354–374 CZ |
| synthetic 15-qubit line `{cz,rz,sx,x}` | 486–513 CZ |

These are **generic basis/topology transpilation envelopes**. They are not
vendor-native compilations. In particular, the synthetic 4x5 grid has 31
undirected edges and must not be called “IQM-Garnet-like.” Current Braket
documentation lists IonQ's verbatim native set as `gpi/gpi2/zz` and IQM's as
`prx/cz`; a real compile must use the selected device topology and vendor path.
See the [Braket device documentation](https://docs.aws.amazon.com/braket/latest/developerguide/braket-submit-tasks.html).

Sze et al. ([arXiv:2501.18515](https://arxiv.org/abs/2501.18515)) is useful
prior art for multiplexer compilation and trapped-ion hardware experiments.
It does not validate these counts or this model.

Proposal-safe replacement:

> The exact 15-qubit circuit has a seed-swept generic transpilation envelope
> of 155 RXX on an abstract all-to-all RXX basis, 238 CZ on abstract all-to-
> all CZ, 354–374 CZ on a synthetic full 4x5 square grid, and 486–513 CZ on a
> line. Vendor-native compilation, device topology, calibrated fidelity, and
> latency remain Phase-2 gates.

### 2.4 Failure-sector derivative / gradient readout

**Verdict: identity CONFIRMED; novelty and general shot-efficiency demoted.**

The tested X-basis failure/success interference readout reproduces the
success-probability derivative to numerical precision for the tested layers.
But its reported shot ratios assume equal allocation across shifted circuits,
not an oracle- or Neyman-optimal allocation. They apply only to the declared
per-row success derivative and must not become a general statement that
failure-sector readout is shot-efficient.

Daskin's all-ancilla-outcomes work ([arXiv:2605.02986](https://arxiv.org/abs/2605.02986))
and standard LCU differentiation make broad novelty unsafe. The surviving
object is a fresh-ancilla stacked implementation and measurement identity,
not a new differentiation principle. Use “exact X-readout derivative
identity,” not “gradient recycling.”

### 2.5 Syndrome drift detector

**Verdict: EXPLORATORY; not proposal-ready.**

The v1 runner used a per-window alpha of 0.01 and 200 permutations. It did not
control sequential familywise false alarms or demonstrate a target average
run length. Because syndrome features are a deterministic transform of their
input rows at fixed parameters, they cannot contain more drift information
than the raw input representation. The result is a preliminary
interpretability comparison, not a statistically controlled monitoring
capability.

Any future claim requires an anytime-valid or explicitly calibrated sequential
test, a frozen false-alarm/ARL target, and matched raw-feature and backbone-
score controls. Kuang and Xia ([arXiv:2609.00536](https://arxiv.org/abs/2609.00536))
is relevant recent prior art for anytime-valid change detection.

## 3. Direct S-LCU prior-art boundary

Khatri, Zohren, and Matos ([arXiv:2607.24686](https://arxiv.org/abs/2607.24686))
directly use “S-LCU” for sequential composition of independent LCU blocks and
analyze its expressivity, trainability, classical simulation scaling, and
optimization. Architecture, naming, generic trainability, and the complexity
dial are therefore not ours to claim.

| object | closest direct prior art | contribution boundary here |
|---|---|---|
| sequential independent LCU layers | Khatri–Zohren–Matos | known; no architecture/name claim |
| trainability and classical-simulation scaling | Khatri–Zohren–Matos | known; cite, do not rebrand |
| optimizing component unitaries / LCU parameters | Khatri–Zohren–Matos | potentially overlapping; no “coefficients not trained” distinction |
| binary fresh-ancilla circuit with full pass/fail patterns | this implementation, adjacent to Daskin | new-for-setting at most |
| complete-dephasing input-independent endpoint at the declared insertion point | elementary theorem specialized here | novel-for-setting candidate, not publication-cleared |
| partial-dephasing per-row witness with exact classical path twin and references | this package | novel-for-setting candidate pending independent literature/referee clearance |
| fraud-monitoring application | this package | novel-for-setting only; no performance advantage |

No “first” claim is made. The literature sweep did not establish global
priority, and absence from a bounded search is not evidence of firstness.

## 4. Corrected addition ranking

| rank | direction | novelty x cost x challenge fit | decision and matched control |
|---:|---|---|---|
| 1 | partial-dephasing hardware witness | strongest surviving mechanism; moderate hardware cost | **SELECT**, after prospective protocol; exact `2^L` path twin, full-dephasing endpoint, phase twirl, identical-branch circuits |
| 2 | trainable-vs-frozen mixer control | already completed, directly hardens the only positive application mechanism | **FOLD IN EXISTING RESULT**; prespecified random-freeze distribution and `phi=0` controls |
| 3 | headroom bound as governance tool | useful and cheap, but classical | **DEPLOYMENT APPENDIX**; oracle in-band reranker and backbone |
| 4 | generic compile envelope | useful resource disclosure, low novelty | **RESOURCE APPENDIX**; same operator across bases/topologies and seeds |
| 5 | splice/interleaving diagnostic | operationally useful but causality unresolved | **DEPLOYMENT SAFEGUARD**, not novelty; M1/M2 with `Q`, feature heads, noise, and logit-only arms |
| 6 | failure-sector X readout | exact identity but prior-art and allocation caveats | **MECHANISM APPENDIX**; optimal-allocation parameter-shift and Hadamard-test estimators |
| 7 | syndrome drift detector | sequential error control missing and raw inputs dominate | **DEFER**; raw features and backbone score at matched ARL |
| 8 | shot-adaptive ranking | known allocation problem and exact path oracle is cheap here | **DO NOT ADD** |

Only rank 1 is a new experimental addition. Rank 2 is already part of the
audited evidence package and should harden the mixer sentence, not be sold as
a second new invention.

## 5. Prospective hardware gate

The Braket-local circuit uses a simulator `phase_flip` instruction. A QPU
cannot execute that noise channel. Materialize the intervention as randomized
physical `I/Z` blocks, with `Z` allocation `q/2`, fixed before execution and
randomized in job order. The minimum reduced experiment is 80,000 base shots
(`8 rows x 5 q values x 2,000`) before identical-branch, phase-twirl,
calibration, and repeat circuits.

Before any managed-simulator or QPU run, freeze:

- a public synthetic input set or approved data-processing plan;
- the exact Braket device, native gates, topology, compiler and calibration;
- the randomized-channel construction and job allocation;
- five-model versus explicitly single-model inference scope;
- corrected nested uncertainty and multiplicity;
- device-calibrated power, shot/task budget and current price; and
- abort criteria for reference failure or calibration movement.

Use current [Amazon Braket pricing](https://aws.amazon.com/braket/pricing/)
only at approval time. A dated cost estimate is not a performance result.

## 6. Data-governance boundary

The files below contain IEEE-CIS-derived validation data and are
`PRIVATE_LICENSED`:

- `runs/hsbc_challenge/novelty_v1/dephasing_witness_curves_v1.npz` contains
  32 transformed rows and labels;
- `runs/hsbc_challenge/novelty_v1/braket_reduced_instance_v1.npz` contains
  eight transformed rows.

Do not publish them, attach them to a public release, or send them to a VM or
AWS service without licence and governance clearance. A public artifact may
include aggregate JSON, theorem code, and synthetic referee fixtures. The v2
uncertainty JSON deliberately emits no row features, labels, identifiers, or
probability tables.

## 7. Revision-3 replacement paragraph

> We do not claim quantum advantage or application lift on the measured fraud
> datasets. Classical one-class and supervised baselines remain as good as or
> better wherever tested, and the prospectively frozen IEEE-CIS application stack
> failed its sealed gates. The surviving contribution is a circuit-realizable
> measurement package: fresh-ancilla pass/fail patterns, an exact X-readout
> success-derivative identity, and a complete-dephasing endpoint whose pattern
> law is input independent. Exact and density-matrix referees validate the
> mechanism; a corrected 32-row simulator analysis resolves the coherent-to-
> dephased between-row contrast for all five model seeds at 2,000 shots after
> conservative multiplicity correction. Phase 2 is a capped Braket mechanism
> study against exact-path, dephased, phase-twirled, identical-branch, and
> classical controls, with native compilation and device-calibrated stop
> rules. The circuit remains monitoring-only and never overwrites the
> classical fraud score unless a prospectively registered benefit gate passes.

Keep the Engine-A FAIL table in the main body as integrity and deployment
evidence. Keep the FTQC LCU/QSVT roadmap separate and explicitly conditional;
the present experiments neither approach nor predict an end-to-end FTQC
advantage.

## 8. P0 / P1 actions

### P0 before revision 3

1. Replace every dephasing v1 interval and gate interpretation with the v2
   corrected analysis; call noise results exploratory.
2. Replace the splice causal sentence with the unresolved-primary wording in
   §2.2 and demote it from selected novelty.
3. Rename compilation results “generic basis/topology transpilation
   envelopes”; remove “target-native,” “Ion-like,” and “Garnet-like.”
4. Add the Khatri side-by-side boundary; remove architecture/name/trainability
   novelty and every “first” claim.
5. Use “locally frozen pre-outcome protocol,” not formal preregistration, for
   these new unarchived experiments.
6. Mark both data-bearing NPZ files `PRIVATE_LICENSED` and exclude them from
   public/VM/AWS transfer pending approval.
7. Describe Braket execution accurately: local density-matrix simulation only;
   no DM1, managed simulator, or QPU run.

### P1 before hardware approval

1. Write and externally timestamp the prospective QPU protocol in §5.
2. Produce a real vendor/device-native compilation and calibrated power/cost
   calculation.
3. Replace simulator noise instructions by randomized physical `I/Z`
   interventions.
4. Add optimally allocated parameter-shift as the gradient-cost control.
5. If drift monitoring remains, calibrate sequential false alarms/ARL and
   retain raw-feature and backbone-score controls.

## 9. Reproduction

Corrected uncertainty audit:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/hsbc_challenge/audit_dephasing_uncertainty_v2.py \
  --outdir runs/hsbc_challenge/novelty_v2_reproduction
```

Historical v1 runners can be executed unchanged without moving or overwriting
the published directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/hsbc_challenge/reproduce_claude_novelty_isolated_v2.py \
  --runner gradient \
  --stage /private/tmp/hsbc-gradient-reproduction-v2
```

For the Braket local referee, create an isolated Python 3.12 environment from
`scripts/hsbc_challenge/requirements_braket_novelty_v2.txt`, first reproduce
or supply the reduced NPZ in the isolated stage, and select `braket-local`.
This remains a local simulator run.

## 10. Machine-readable artifacts

| path | role |
|---|---|
| `scripts/hsbc_challenge/audit_dephasing_uncertainty_v2.py` | corrected, isolated-output uncertainty analysis |
| `runs/hsbc_challenge/novelty_v2/dephasing_uncertainty_v2.json` | aggregate corrected numbers and evidence labels |
| `docs/hsbc_challenge_dephasing_witness_analysis_amendment_v2.md` | transparent post-outcome amendment |
| `scripts/hsbc_challenge/reproduce_claude_novelty_isolated_v2.py` | non-overwriting launcher for unchanged v1 runners |
| `scripts/hsbc_challenge/requirements_braket_novelty_v2.txt` | exact top-level Braket environment pins |
| `runs/hsbc_challenge/novelty_v2/isolated_gradient_reproduction_v2.json` | durable clean-room reproduction record; exact semantic match after excluding wall time |
| `runs/hsbc_challenge/novelty_v2/manifest_v2.json` | hashes, classifications, and validation ledger |
