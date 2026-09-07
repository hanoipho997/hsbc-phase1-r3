# Engine-A kill-gate preregistration — v1 (FROZEN)

**Scope.** The single confirmatory experiment behind the pivoted HSBC thesis
(*coherent cross-view syndromes for routed fraud reranking*): does a routed
stacked-LCU **syndrome residual** improve fraud ranking on IEEE-CIS over (i)
a tuned classical backbone and (ii) every matched classical residual, on a
frozen chronological split, at preregistered thresholds?

**Freeze status.** Written 2026-08-30, **before any contact with IEEE-CIS
data** (not yet downloaded; no statistics of it have been inspected beyond
the challenge statement's published summary table). Everything in §2–§9 is
frozen. **Amendment policy:** any change requires a new numbered version of
this file with a dated amendment log entry (§10), made *before* test-set
unblinding; post-unblinding changes are prohibited — results then report
against this version as written. This file, the analysis code, and all
artifacts must be **git-committed before the test-set unblinding run**
(audit C11 corrective).

**Prerequisite verification (done).** The ancilla-syndrome machinery is
implemented in `scripts/hsbc_challenge/hsbc_common.py`
(`BatchedSLCU.syndrome_sector_states/_distribution/_features`,
`dephased_syndrome_distribution`) and verified in
`scripts/hsbc_challenge/verify_syndrome_v1.py` →
`runs/hsbc_challenge/engine_a_v1/verify_syndrome_v1.json`:
completeness 4.4e-16; flag consistency exact vs the repo-verified success
path; **circuit-level ground truth** (full Qiskit statevector with per-layer
ancillas) block vectors 1.2e-13 / pattern probabilities 4.1e-14;
failure-sector = derivative identity in implementation (logit) coordinates
8.6e-17; **dephased-twin theorem** (ancilla dephasing ⇒ pattern distribution
exactly data-independent, product of Bernoulli(2a₀a₁)) 1.7e-16 with zero
spread — the syndrome signal is purely interference-borne; 48 µs/tx on real
rows. Simulability disclosure: at these widths the same features are exactly
computable classically via the 2^L-path overlap expansion; the claimed
contribution is the feature family, its circuit realization, and its
gradient/measurement economics — never classical inaccessibility.

---

## 1. Hypothesis under test

**H-A.** On a chronological IEEE-CIS split, appending the 2^L
ancilla-syndrome log-probability features of a trained cross-view stacked-LCU
circuit to the backbone's logit — **only for transactions routed to a
preregistered uncertainty band** — improves fraud ranking by at least the
practical thresholds in §7, and by more than every matched classical
residual. Mechanistic rationale (declared, testable): routed pools are
small-N; a ~18-parameter model with a declared cross-view-disagreement prior
should generalize where higher-capacity classical residuals overfit.

## 2. Data, split, hygiene

- **Dataset:** IEEE-CIS `train_transaction.csv` left-joined
  `train_identity.csv` on `TransactionID` (590,540 labeled rows; verified).
  The unlabeled competition files are not used. Downloaded under Kaggle
  competition terms. SHA-256 (recorded at M2, 2026-08-30):
  `train_transaction.csv` = `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642`;
  `train_identity.csv` = `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c`;
  sealed test partition parquet =
  `f09bcaefe7ed4a4f…` (full hash in
  `runs/hsbc_challenge/engine_a_v1/ingest_manifest_v1.json`).
- **Split:** sort by `TransactionDT`; train = first 60% of rows, validation
  = next 20%, test = final 20%. No shuffling. Row-index boundaries recorded.
- **Blinding:** the test partition is written to a separate file at ingest
  and not read by any code path until the single unblinding run (§9). All
  selection, tuning, thresholds, calibration: train/validation only.
- **Non-confirmatory dry run:** the identical pipeline is exercised once on
  ULB (row-order split) purely to validate code paths; its numbers are
  labeled PIPELINE-VALIDATION and carry no evidentiary weight.

## 3. Classical backbone (frozen tuning protocol)

XGBoost and LightGBM on the full feature set with standard public-kernel
preprocessing (frozen recipe: label-encode categoricals; frequency-encode
`card1`, `addr1`, `P_emaildomain`; `TransactionAmt` decimal-part feature;
day/hour from `TransactionDT`; no target encoding). Search space (36 configs
each): depth ∈ {6, 8, 10} × lr ∈ {0.02, 0.05, 0.1} × subsample ∈ {0.7, 0.9}
× colsample ∈ {0.7, 0.9}; n_estimators ≤ 4000 with early-stopping 100 on
validation AUPRC; scale_pos_weight ∈ {1, neg/pos}. **Backbone p_C :=** the
single model or the two-model mean with the best validation AUPRC; frozen
thereafter. Seed 0 throughout.

## 4. Routing rule (frozen)

Score thresholds are **validation-quantile score values** applied to test
scores (no test-quantile peeking):

- **Auto-flag region:** p_C above the validation 99.8th-percentile score —
  left in backbone order, never re-ranked.
- **Review pool (the routed set):** p_C in (validation 98.8th percentile,
  validation 99.8th percentile] — expected ≈ 1.0% of rows (≈ 1,181 test
  rows). Only these rows are re-scored.
- Everything below: backbone order, untouched (no-harm by construction).

For residual *training*, the train-side pool analog uses **out-of-fold
backbone scores** (5-fold refits of the frozen backbone config on train,
fold seed 0) with the same validation-derived thresholds.

## 5. Quantum component (frozen spec; implementation milestone M1)

- **Features:** d = 12 = 3 semantic views × 4 features. Views by schema:
  A = transaction (from `TransactionAmt`, `ProductCD`, `TransactionDT`-hour,
  `dist1`, `C1..C14`), B = card/address (`card1..card6`, `addr1`, `addr2`),
  C = email/device/identity (`P_emaildomain`, `R_emaildomain`, `DeviceType`,
  `DeviceInfo`, `id_01..id_38`). Within each view: top-4 by mutual
  information with the label on **train only** (supervised screening,
  declared as such — audit C5 lesson: no "label-free" wording anywhere).
  Quantile-angle encoding per feature (train-CDF), one qubit each.
- **Circuit:** L = 3 cross-view layers on pairs (A,B), (B,C), (A,C); layer
  j branches U_{j0} = e^{−iθ_{j0} H_V}·RX_V(φ_{j0}) and
  U_{j1} = e^{−iθ_{j1} H_W}·RX_W(φ_{j1}), where H_V is the per-view Ising
  model fitted by pseudo-likelihood on **train legitimate rows** of view V's
  bits, and RX_V acts on view V's qubits only. 6 parameters/layer, 18 total.
  One ancilla per layer; outputs = the 8 syndrome log-probabilities
  log(P(a|x) + 10⁻⁶).
- **Training:** pairwise ranking loss within the train OOF pool (all pool
  frauds × 256 sampled pool legit per iteration), Adam lr 0.05, 300
  iterations, seeds **900–904**; model selection by validation-pool AUPRC
  only; the deployed score is the **mean over the five seeds' residual
  outputs** (no seed selection).
- **M1 gate:** the cross-view layer class must pass the same verification
  suite (V1–V5 of `verify_syndrome_v1.py`, adapted) before any IEEE-CIS
  training.

## 6. Residual model and matched controls (all frozen)

**Quantum arm (Q):** L2-logistic on [logit p_C, 8 syndrome features]
(10 parameters); fit on the train OOF pool; ridge C ∈ {0.1, 1, 10} and Platt
calibration chosen on the validation pool. Final test score: backbone
everywhere; on pool rows, the calibrated residual model's output.

**Matched classical residual controls** — identical routing, identical pool
rows, identical 12 features (quantile-transformed), identical tuning
protocol and seeds:
- **C-a:** L2-logistic on [logit p_C, 12 raw features].
- **C-b:** MLP, one hidden layer of 8 (≈ 113 params), same inputs as C-a.
- **C-c:** PCA residual — PCA (k = 4) fit on train-legit 12 features;
  reconstruction error appended: logistic on [logit p_C, PCA-error]. (The
  audit's OCC-parity winner, given its shot at the same job.)
- **C-f:** noise floor — logistic on [logit p_C, 8 iid N(0,1) features].
- **Disclosure C-e:** the exact classical 2^L-path overlap expansion
  reproduces the syndrome features identically in simulation; it is listed
  as the simulability disclosure, not raced as a rival.
- **Mechanism report (not a gate):** dephased-twin syndrome is
  data-independent (verified theorem) — reported alongside, with the
  coherent-vs-exact-classical equivalence, in the mechanism section.

## 7. Endpoints, gates, and outcome labeling (frozen)

Computed once, on the single unblinding run; paired bootstrap 10,000
resamples, seed 1234, identical test rows for every arm.

- **Primary endpoint:** ΔAUPRC = AUPRC(Q-hybrid) − AUPRC(backbone), full
  chronological test set.
- **Co-primary (operational):** ΔRecall@budget, budget = auto-flag region +
  top **K = 400** pool rows by the arm's pool ranking; recall over all test
  frauds. (Secondary, exploratory: K ∈ {100, 200, 800}.)
- **Gates — ALL must pass for a PASS verdict:**
  - **G1 (statistical):** 95% paired CI excludes 0 on the primary or the
    co-primary.
  - **G2 (practical):** ΔAUPRC ≥ **+0.005** or ΔRecall@budget ≥ **+2.0 pp**.
  - **G3 (fairness):** Q-hybrid point estimate strictly exceeds **every**
    control (C-a, C-b, C-c, C-f) on the gate-passing endpoint, **and** the
    paired 95% CI of (Q − best control) excludes 0.
  - **G4 (no-harm):** off-pool test scores bit-identical to backbone
    (asserted), and full-test AUPRC(Q-hybrid) ≥ AUPRC(backbone) − 0.001.
- **Power pre-check (before unblinding, documented):** predicted test-pool
  fraud count from validation-pool prevalence; if < 40, the primary endpoint
  is dropped and G1/G2 apply to the co-primary alone (thresholds unchanged).
- **Outcome labeling (pre-drafted, verbatim):**
  - **PASS →** "On a preregistered chronological IEEE-CIS split, routed
    syndrome reranking improved [endpoint, value, CI] over a tuned backbone
    and over every matched classical residual (best control: [name,
    value])." Engine A becomes the submission headline.
  - **FAIL →** "The preregistered routed-syndrome experiment did not meet
    its gates ([which gate], [values, CIs]). We report it in full." The
    submission leads with Engine B (conditions map + hardware-validated
    mechanism and training-economics results) and this null as evidence of
    protocol integrity. No re-runs, no post-hoc routing or threshold
    changes, no alternative datasets presented as confirmatory.

## 8. Multiplicity and integrity constraints

One routing rule; one quantum arm; one primary + one co-primary endpoint;
five seeds ensembled with no selection; secondary K values and all ULB
numbers labeled exploratory/pipeline-validation. No metric, threshold,
feature list, or control may be revised after the first read of any test-set
quantity. Negative or null results are publishable deliverables of equal
standing (audit precedent).

## 9. Execution order

M1 cross-view simulator extension + verification gate → M2 IEEE-CIS ingest
(hashes recorded; test partition sealed) → backbone tuning (train/val) →
freeze backbone + routing thresholds → OOF pools → train Q arm + all
controls (train/val only) → power pre-check documented → **commit code,
this file, and manifests** → single unblinding run executing every arm and
endpoint in one script (`engine_a_unblind_v1.py`, to be written at M1, must
print this file's SHA-256 in its output) → report
`docs/hsbc_engine_a_report_v1.md` with PASS/FAIL verdict and all numbers.

## 10. Amendment log

- **A1 (2026-08-30, clarification, pre-M2, no test-data contact):** M1 is
  complete. Implementation details fixed by the M1 code, recorded here for
  the avoidance of post-hoc freedom: (i) per-view Hamiltonian diagonals are
  span-normalized to [0,1] per view; (ii) branch operator order is
  U = e^{−iθH_V}·RX_V(φ) — mixer first, phase second; (iii) per-layer
  parameter block ordering [logit0, logit1, θ_V, φ_V, θ_W, φ_W]; (iv)
  syndrome features use ε = 10⁻⁶ (as in §5). M1 verification
  (`scripts/hsbc_challenge/verify_crossview_v1.py` →
  `runs/hsbc_challenge/engine_a_v1/verify_crossview_v1.json`): completeness
  6.7e-16; repo-dense twin forward 1.1e-16 and all 18 parameter derivatives
  2.1e-16; Qiskit circuit-level sector blocks 4.1e-13; sector=derivative
  identity 6.9e-17; dephased-twin theorem 1.1e-16 with zero spread;
  training-loss gradient vs finite differences 1.9e-9 relative; full-spec
  (n=12, L=3, 18 params) throughput ≈2.0 ms/row syndrome and ≈4.5 ms/row
  forward+derivatives (projected training cost ≈12 min/seed, ≈1 h for the
  five-seed ensemble — within budget).
- **A2 (2026-08-30, M2 ingest + backbone-protocol clarifications, recorded
  BEFORE any backbone fit and with the test partition sealed unread):**
  (i) source hashes and library versions filled in §2/§11; ingest manifest
  at `runs/hsbc_challenge/engine_a_v1/ingest_manifest_v1.json`; test rows
  written to `test_SEALED.parquet` (chmod 400) with no statistics computed.
  Train/val (permitted): 354,324 rows / 11,988 frauds (3.383%);
  118,108 / 4,611 (3.904%). (ii) §3 grid ambiguity resolved maximally
  inclusively: 36 hyperparameter cells × scale_pos_weight ∈ {1, neg/pos} =
  **72 fits per library**; selection unchanged (validation AUPRC, sklearn
  `average_precision_score` on validation predictions for both libraries).
  (iii) Feature-matrix clarifications: `TransactionID`, raw `TransactionDT`,
  `isFraud`, and the split column are excluded from features (derived
  hour-of-day and day-of-week retained per the frozen recipe); label
  encoders and frequency maps are fit on train+validation only, with unseen
  test-time categories mapped to −1; encoded categoricals are passed as
  numeric to both libraries identically; NaNs pass through natively (no
  imputation); numerics downcast to float32; LightGBM `num_leaves` =
  min(2^depth, 255) alongside `max_depth`.
- **A3 (2026-08-31, backbone-compute relocation, recorded BEFORE any VM
  result existed):** the §3 backbone grid is executed on the team's
  255-core Ubuntu 24.04 VM (x86_64, Python 3.12.3) with the identical
  frozen grid, recipe, seeds, and pinned package versions
  (xgboost 3.4.1, lightgbm 4.7.0, scikit-learn 1.9.0, pandas 2.3.3,
  numpy 2.4.6, pyarrow 25.0.1), 12 concurrent fits × 20 threads per fit
  (thread count fixed for determinism). The **VM run is canonical** for
  backbone selection; after the grid, the best configuration per library is
  refit once to produce the serialized models (booster prediction is
  platform-independent, so unblinding scoring may run on either machine).
  The partially completed macOS run of the same grid continues as a
  non-canonical cross-check and its overlapping cells are reported for
  consistency. **Only `trainval.parquet` and code are transferred; the
  sealed test partition never leaves the primary machine.** §11's
  "single-machine, macOS arm64" line is superseded accordingly for the
  backbone-fitting step only.
- **A4 (2026-08-31, post-backbone freeze; recorded BEFORE OOF/quantum
  training and before any test contact):** (i) Backbone frozen per §3: the
  two-model mean (val AUPRC 0.65195; xgb d10/lr0.05/s0.9/c0.9/balanced
  0.63860 at 3,932 trees; lgb d10/lr0.02/s0.7/c0.9/w1 0.62676 at 534
  trees). A3 cross-check: 15 overlapping macOS cells, max |Δ AUPRC|
  0.0079, mixed signs, structure reproduced. (ii) Routing thresholds frozen
  from validation quantiles: t_lo = 0.878636, t_hi = 0.994177
  (`runs/hsbc_challenge/engine_a_v1/routing_freeze_v1.json`). Power
  pre-check: predicted test-pool frauds 1,084 ≥ 40 → primary endpoint
  retained. (iii) **Recorded observation, gates unchanged:** the validation
  pool is 91.8% fraud (1,084/1,181), so the maximum achievable
  ΔRecall@budget over the backbone ordering is +0.87 pp at K=400 (+1.26 pp
  at K=800) — the co-primary branch of G2 (+2.0 pp) is unreachable on
  val-projected composition, and G2 must realistically pass via
  ΔAUPRC ≥ +0.005. This is reported as a classical-frontier finding either
  way. (iv) OOF mechanics: KFold(5, shuffle=True, random_state=0) over
  train rows; per-fold refits use the frozen configurations with
  n_estimators fixed at the canonical best iterations (3,932 / 534), no
  fold-level early stopping; scale_pos_weight at the train-global value for
  the balanced config. (v) View-selection pragmatics: within each §5
  candidate list, object-dtype candidates are frequency-encoded (train
  counts) before the train-CDF quantile transform; mutual information uses
  `mutual_info_classif(random_state=0)` on a 120k-row seed-0 stratified
  train subsample with NaN→−1 sentinel and discrete-feature flags for
  encoded categoricals; top-4 per view. View qubit assignment: A→0–3,
  B→4–7, C→8–11.
- **A5 (2026-08-31, pre-arms-training clarifications; no test contact):**
  (i) §5's per-iteration batch ("all pool frauds × 256 sampled pool legit")
  was written for a legit-majority pool; the frozen routing band is
  fraud-majority, so the intent is preserved as: **all of the pool's
  minority class + 256 sampled (without replacement where possible) of the
  majority class per iteration**. (ii) Circuit-level model selection metric
  = validation-pool AUPRC of the flag score (1 − p_s as fraud score).
  (iii) Residual heads: features standardized on the train pool; ridge
  C ∈ {0.1, 1, 10} selected by validation-pool AUPRC (the logistic head is
  itself the calibrated probability map). (iv) C-c's PCA is fit on all
  train-legitimate rows' 12 quantile features (not pool-restricted), k = 4.
  (v) Validation-pool metrics of every arm are computed and recorded
  pre-unblinding (validation quantities are non-test by design); the test
  set remains sealed.
- **A6 (2026-08-31, pre-unblinding; found by the rehearsal referee, no test
  contact):** the C-f noise-control features originally came from a shared
  RNG stream whose position depended on batch composition — every other arm
  reproduced at 0.0e+00 in rehearsal, C-f deviated by 2.1e-3. Fixed to
  order-invariant per-row noise keyed by (seed 7, TransactionID)
  (`hsbc_common.keyed_noise`); the C-f head was regenerated accordingly
  (val-pool AUPRC 0.91076, C=10). No other arm touched.

## 11. Seeds and environment (frozen)

Data/split seed: none (deterministic chronological). Backbone + folds:
seed 0. Circuit training: 900–904. Bootstrap: 1234. Environment: repo
`.venv` (Qiskit 2.5, scikit-learn 1.9, xgboost 3.4.1 + libomp, lightgbm
4.7.0, pyarrow 25.0.1 — recorded at M2), macOS arm64; all runs
single-machine.
