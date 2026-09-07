# Adversarial audit of the HSBC Phase-1 proposal

**Audit version:** v1. **Protocol freeze:** 2026-08-30, before running any
new audit experiment. **Status at protocol freeze:** `PREREGISTERED; OUTCOMES
NOT YET READ`. **Auditor:** Codex, instructed to treat falsification as a
success outcome. **Final status:** `COMPLETE; NO-GO AS WRITTEN`.

## 0. Scope, source lock, and missing inputs

The objects under audit are the 2026-08-29 versions of the proposal
(`.md`, `.tex`, and the seven-page compiled PDF), the companion theory note,
the fit assessment, the seven HSBC experiment scripts, and the six `fit_v0`
JSON artifacts named in the handoff. The baseline SHA-256 inventory and the
fieldwise reproduction comparison will be reported in Appendix A.

The handoff's section 0 does **not** contain the promised seeded external
references; it contains the literal placeholder `[TO ADD: Ha drops links/papers
here before launch]`. This is an evidence limitation, not a reason to skip the
audit. The local authoritative challenge statement
`external/HSBC-Challenge-Statement-vFinalRevised.pdf` and the
primary papers named in C9 will be checked. The challenge statement permits a
focus on one or two datasets; this audit nevertheless attempts ULB and Sparkov
because the handoff imposes that stronger condition. IEEE-CIS is not assumed
available without accepting its competition license.

The worktree is a dirty shared tree. Existing modifications and untracked
files are out of scope. This audit writes only
`docs/hsbc_challenge_audit_report_v1.md`, new scripts under
`scripts/hsbc_challenge/`, and new artifacts under
`runs/hsbc_challenge/audit_v1/`. It does not overwrite `fit_v0`, stage, reset,
stash, or clean anything.

## 1. Preregistered audit protocols

This section was written before running or reading any new audit outcome.
All choices below are frozen. If a method cannot run, the failure and reason
remain in the artifact; no substitute inherits its name.

### 1.1 Shared data, splits, metrics, and uncertainty

- ULB source: `runs/hsbc_challenge/data/ulb_creditcard.npz`, SHA-256 to be
  recorded. The original stratified seed-0 60/20/20 row indices are reused.
  Feature selection, quantile transforms, and all fitting remain train/validation
  only. Test labels are used only for final metrics and paired uncertainty.
- Primary metric: AUPRC on the complete imbalanced held-out test evaluation
  pool defined by each protocol. No balanced-subsample metric is allowed.
  Secondary metrics: AUC-ROC and recall at FPR `1e-3` where defined.
- Confidence intervals: 2,000 paired row bootstraps with audit seed 20260831.
  Resamples without both classes are rejected. For C1, the primary training-seed
  estimand is averaged across all ten frozen seeds before taking the row-bootstrap
  quantiles; no seed is selected by test or validation performance.
- Quantum-score rows are reported both exact and, when relevant, after 128
  Bernoulli shots per transaction. Classical rows have `shots/tx = N/A` and are
  not artificially noised. Parameters and fit/evaluation row caps are reported.
- A result is called resolved only when its preregistered paired 95% interval
  excludes zero. AUPRC remains prevalence-dependent; every restricted pool
  reports both class counts and prevalence.

### 1.2 C5 clean fraud-mode holdout (highest-priority attack)

For each `k in {3,4,6}`, fit `KMeans(n_clusters=k, n_init=20,
random_state=20260830)` **only** on train+validation frauds in the original
eight quantile-transformed features. Assign train, validation, and test frauds
by those frozen centroids. For cluster `c`:

1. remove cluster-c frauds from XGBoost train and validation rows;
2. fit the original fixed-tuning XGBoost-full baseline on the remaining rows;
3. evaluate strictly on all 56,863 test legitimate rows plus cluster-c **test**
   frauds--never train or validation frauds;
4. evaluate the S1 score with the already-declared `tau=4` and original feature
   screen, explicitly labelled “legit-fitted conditional on a supervised
   feature screen”, not fully label-free;
5. report cluster size in train/validation/test, AUPRC CI, AUC, and
   recall-at-FPR for every cluster, with the largest train+validation cluster
   designated the dominant mode before test metrics are read.

The 0.984-vs-0.263 headline survives only if the corrected dominant cluster has
the same qualitative ordering for all three k values and the paired AUPRC
interval excludes zero. The exact old headline is corrected whenever the
strict-test point estimate differs, even if the ordering survives. A separate
unsupervised-screen sensitivity uses the eight highest-variance standardized
features chosen from legitimate training rows only; it is not allowed to
replace the primary original-feature result.

### 1.3 C1 preregistered L=1 mixer replication

Reuse the original n=8 split, fraud/legitimate batches, 300 Adam iterations,
20-iteration validation checkpoints, and L=1 architecture. Freeze new training
seeds `500..509`. Each seed keeps its own best validation checkpoint, but **all
ten** seed outcomes enter the primary mean delta versus the fixed analytic S1
score; there is no best-seed selection. Report the per-seed test deltas, their
sign count, the mean delta, and a paired row-bootstrap CI for the mean across
seeds. A positive original sign is confirmed only if this interval excludes
zero; otherwise C1's mixer increment is protocol-dependent/corrected even if
the rest of the ablation ladder reproduces.

### 1.4 C6 competent hybrid integration

Refit the original XGBoost-full and reconstruct the frozen S2(L=2) score.
Fit meta-models on validation only, evaluate once on test, and report every
frozen candidate rather than only the favorable branch:

- regularized logistic stacking on `[logit(xgb), standardized_s2]`, `C=1`;
- the same plus their product interaction;
- separate isotonic validation calibrators combined by the arithmetic mean;
- an isotonic product `1-(1-p_xgb)(1-p_s2)`;
- a band-conditional logistic stack used only in the validation-defined
  `[P99.0,P99.9]` XGBoost band, preserving XGBoost order outside it.

All transforms are fitted on validation only. The primary C6 comparison is the
first logistic stack versus XGBoost-full; the others are multiplicity-exposed
stress tests. H4 changes only if a prespecified candidate has a positive paired
CI and also beats the original noise-control delta of +0.008 in point estimate.

### 1.5 OCC-parity gate and matched-budget tournament

Run n=8 and n=12 with the identical original features and test rows. Every OCC
is fitted on legitimate training rows only. Frozen classical OCC bracket:

- IsolationForest: 300 trees, `max_samples=8192`, seed 20260830;
- One-Class SVM: RBF, `nu=0.001727`, `gamma="scale"`, deterministic 20,000-row
  legitimate cap;
- GMM: diagonal covariance, components selected from `{1,4,8}` by legitimate
  validation log-likelihood only;
- Gaussian KDE: 5,000-row legitimate cap, bandwidth selected from
  `{0.2,0.5,1.0}` by legitimate validation likelihood only;
- LocalOutlierFactor novelty mode: 35 neighbors, 40,000-row legitimate cap;
- empirical-copula tail score: independent two-sided empirical tail log-score,
  labelled `ECOD-like` rather than ECOD if `pyod` is unavailable;
- PCA reconstruction: components retaining 90% legitimate-train variance;
- deep-SVDD: bias-free two-layer ReLU network `(32,16)`, center fixed after
  initialization, 30 epochs, batch 1024, seed 20260830, legitimate-only
  squared-radius objective with `1e-6` weight decay. PyTorch is unavailable,
  so the frozen architecture is implemented directly in NumPy with explicit
  Adam gradients; it is not replaced by an autoencoder.

The OCC-parity gate fails for the proposal if any plain classical OCC has a
point estimate at least S1's and its paired delta versus S1 is not resolved
negative. “Matches” means `|delta| <= 0.01` or a paired CI containing zero.

Frozen NISQ/quantum-inspired rows, each with exact-simulation or explicitly
labelled proxy semantics: product-state fidelity kernel SVC; projected
`Z/ZZ` kernel SVC; a shallow re-uploading circuit classifier at an eight-parameter
budget; a shallow quantum-autoencoder anomaly score fitted legitimate-only;
128-shot randomized-Pauli/classical-shadow features; the dephased LCU control;
the exact `2^L` coherent path expansion; an MPS/TT rank-4 legitimate density;
and a legitimate-fitted Bernoulli RBM. Dense or kernel training caps and all
parameter counts are reported. If a faithful implementation is unavailable,
the named row is marked `NOT RUN`; a classical proxy is not relabelled quantum.

Tournament table columns are method, AUPRC [CI], AUC, paired delta versus
S2(L=1), paired delta versus XGB-subset, parameter count, shots/transaction,
training rows/cost, and scope notes. The complete imbalanced test set is used.

### 1.6 Two frozen improvement directions

1. **Cross-view LCU:** split the original ordered eight-feature screen into
   alternating four-feature views, fit four-component CCA on legitimate
   training rows, and use equal-weight branch overlap as the LCU success flag.
   CCA signs and scaling are fixed without fraud labels. This is reported as a
   first circuit-realizable cross-view implementation, not as the proposal's
   final learned unitary design.
2. **Flag natural gradient versus Adam:** train the same n=8 L=1 success-score
   model on identical batches and seeds `610..614`. Flag-NG uses exact
   per-sample success Jacobians, damping `1e-2`, learning rate `0.05`, and an
   L2 trust cap of `0.1`; Adam keeps the original hyperparameters. All five
   seeds are reported without best-seed selection. Divergence or stalling is a
   valid result.

Calibration (Platt versus isotonic on validation) is a secondary third
direction and cannot rescue a ranking loss because monotone calibration leaves
AUPRC unchanged.

### 1.7 C10 finite-shot replication

Using the frozen S2 score and seeds `700..729`, draw 30 independent Bernoulli
readouts at each of 128, 1,024, and 8,192 shots per test transaction. Report
the AUPRC distribution (mean, SD, median, 2.5/97.5 percentiles) and paired
delta distribution versus exact. The claim “unchanged at 128 shots” is retained
only as a distributional statement; a single favorable noise draw is never
reported as evidence.

### 1.8 Regime hunt

- **Label scarcity (pre-outcome protocol correction):** the initial text
  mistakenly left all 98 validation fraud labels available, which would not
  measure total supervision scarcity. Before any run, this was corrected so
  XGBoost on the matched eight features and XGBoost-full receive exactly
  `{5,20,50,200,all}` fraud labels **in total across training and validation**.
  Finite budgets use a frozen 80/20 train/validation allocation (at least one
  validation fraud); all unallocated fraud rows are removed, all legitimate
  rows remain, and the five frozen subset seeds are reported. The test set is
  unchanged and never used for selection.
- **Clean drift/novel modes:** C5 is the novel-mode result. The temporal ULB
  result is reported only after verifying row order against the local OpenML
  ARFF `Time` column.
- **Prevalence:** target prevalence `{0.02,0.05,0.10,0.1727,0.50}%` is imposed
  by exact class weights on the complete test set, not by balanced resampling.
- **Feature poverty:** repeat matched S1, a frozen IsolationForest OCC, and
  supervised XGBoost-subset at `n in {2,4,6,8,12}` with train-only selection.
  Tournament-selected “best OCC” rows are quoted separately only at n=8/12;
  they are not retroactively substituted into this curve.
- **Bounded shift:** move each test-fraud quantile vector toward the legitimate
  median by `epsilon in {0,0.05,0.10,0.20}` using
  `u'=(1-epsilon)u+0.5 epsilon`; legitimate test rows stay fixed. Compare
  methods trained on the same quantile features. This is a normalized-space
  stress test, not a claim about attack feasibility.
- **Sparkov:** if the documented open dataset is acquired, preserve its
  supplied train/test temporal split, evaluate the full imbalanced test file,
  and use eight frozen, semantically defined train-only features: log amount,
  hour sine/cosine, customer age, customer-to-merchant distance, log population,
  legitimate-train category frequency, and legitimate-train card amount
  deviation. Training caps are allowed and reported; evaluation is never
  balanced. ULB-to-Sparkov transfer is restricted to genuinely shared
  amount/time-derived features and labelled weak/partial because ULB's PCA
  coordinates have no Sparkov counterpart.

The conditions map will report `axis | threshold | winning family | evidence
pointer`. If classical supervised/OCC methods cover every measured regime,
the final verdict will say plainly that no quantum component is needed on
these datasets and will include demoted proposal wording.

### 1.9 Validation stability and low-replication checks

Before the priority script is run, freeze a 2,000-draw paired Poisson row
bootstrap of the original n=8 validation pool across the already-declared S1
tau grid. Report how often each tau is selected and the AUPRC separation from
tau=4; this audits selection stability with only 98 validation frauds without
changing the model. The n=12 three-seed conclusion is not promoted to a
population claim: the exact reproduction and the n=12 OCC/tournament rows are
reported, but no new seed grid may be introduced post hoc.

The proposal's `1.9--60 microseconds/transaction` latency range has no timing
field in a `fit_v0` artifact. Freeze a replacement benchmark at n=8: one
warm-up plus five complete-test repetitions for analytic S1 and structured
S2(L=2), starting from the already-transformed feature angles and including
product-state construction. Report median/min/max, interpreter/library
versions, machine, batch size, and the fact that raw feature transformation is
excluded. It is a local CPU benchmark, not an authorization-latency guarantee.

### 1.10 Post-run audit-of-audit correction: label scarcity

After the first regime script had run, but **before reading any corrected
scarcity outcome**, code review found that its XGBoost rows respected the
fraud-label budgets while the comparison S1 row still used the original
all-label feature screen and validation-selected tau. That is not a matched
label budget. The first scarcity curve is excluded from every verdict and
conditions map.

The corrected rerun is frozen as follows. For each budget/replicate, select
the eight features using all legitimate training rows plus only the allotted
training-fraud labels; fit the empirical quantile transform on those same
accessible training rows; fit S1's Ising model and IsolationForest on the
legitimate training rows in that representation; select S1 tau from the grid
`{0.5,1,2,4,6,8,12}` using all validation legitimate rows plus only the
allotted validation-fraud labels; and fit XGBoost-subset on the identical
eight raw features and accessible labelled rows. XGBoost-full receives the
same labelled rows. Test rows remain untouched and complete. Five frozen
replicates and the original budget subsets are retained; no choice is changed
in response to the discarded outcomes.

## 2. Executive verdict

**NO-GO as written; GO only after the P0 demotion below.** The isolated
baseline gate passes: every non-timing field in all six seeded `fit_v0` JSONs
reproduces at `rtol=1e-12, atol=1e-14`. Falsification then succeeds on two
load-bearing claims and forces corrections to five others: **4 CONFIRMED, 5
CORRECTED, 2 REFUTED**.

The OCC-parity gate fails the proposal. At n=8, legitimate-only PCA error is
0.724 AUPRC [0.635,0.810], above analytic S1 at 0.696 and the preregistered L=1
stack at 0.706; its paired delta over L=1 is +0.018 [+0.008,+0.030]. At n=12,
IsolationForest is 0.697 [0.605,0.785], versus S1 0.657 and S2(L=2) 0.659; its
paired delta over S2 is +0.037 [+0.015,+0.064]. XGBoost-subset remains stronger
at 0.749/0.752. No tested NISQ or quantum-inspired rival establishes a lift
over the best classical row.

C5's exact `0.984 vs 0.263`, “label-free,” general novel-mode headline is
**REFUTED**. Under train+validation-fitted clusters and test-only evaluation,
the supervised-screened S1 ordering does survive for the one dominant mode:
0.944 versus XGBoost 0.228/0.413/0.394 for k=3/4/6, with paired intervals
excluding zero. It does not generalize across modes: minority-mode S1 AUPRCs
range from 0.0001 to 0.246 and usually trail XGBoost. More decisively, replacing
the label-using feature screen by a legitimate-only variance screen yields
0.0050 on the full test set and 0.00039 on its k=4 dominant held-out mode.

C1 survives its hardest attack. Ten unseen L=1 seeds are all positive versus
S1; mean delta is +0.00724 with paired 95% interval
[+0.00137,+0.01337]. This confirms the narrow fixed L=1 sign, not a generic
depth advantage: the original L=2 branch is -0.0445 and the alternate depth
sweep is flat. C6's headroom null also survives: the primary competent
validation-only logistic stack adds +0.000227 AUPRC
[-0.000706,+0.001258], and every stress-test integration is unresolved or
worse.

The conditions map contains no measured regime that requires a quantum
component. Classical supervised plus classical OCC covers every label budget,
feature count, temporal split, and Sparkov result. S1 wins selected reweighted
prevalence and synthetic inward-shift rows, and the supervised-screened
dominant ULB mode, but S1 is itself exactly classically evaluable here; the
four-path L=2 classical expansion matches the structured simulator to
`7.8e-16` in score. The honest contribution is therefore a
**quantum-circuit-realizable, classically tractable OCC mechanism and hardware
mechanism-validation plan**, plus this negative conditions map--not a needed
quantum fraud detector.

## 3. Claim-by-claim rulings (C1-C11)

| claim | verdict | evidence and required boundary |
|---|---|---|
| **C1 ablation** | **CONFIRMED** | The full hard 0.264 -> soft 0.688 -> S1 0.696 -> diagonal 0.691 ladder reproduces. The preregistered unseen-seed L=1 replication gives 10/10 positive deltas, mean +0.00724 [+0.00137,+0.01337]. Keep the explicit L=2 reversal (-0.0445) and flat alternate depth sweep; do not generalize beyond the fixed L=1 protocol. Evidence: `priority_attacks_v1.json:C1_mixer_replication`. |
| **C2 dequantization** | **CORRECTED** | The variance proof, slope -0.49243, and rejection-rate inequality are correct for product encodings plus commuting diagonal filters. The measured shot/MC RMSE ratios are 5.95--7.67, not “3--7”. The measured cross-view proposal has no evidence of classical hardness, and L=2 still has a four-path exact classical expansion. |
| **C3 no-free-sampling** | **CONFIRMED** | The identity is correct for a target basis state and extends by summation to degenerate target sets and nonuniform inputs. “Mixers required” must mean raw repeated sampling from this diagonal family: amplitude amplification supplies a quadratic query-level escape, and changing/warm-starting the input changes the bound's right-hand side. |
| **C4 alignment** | **CORRECTED** | All EXP-D outcomes reproduce, including 2/5 E.ON QNG collapses and 822 floor hits. The causal phrase “objective-selected, not optimizer-selected” is unsupported: HSBC success ranking is stable under Adam but highly unstable under QNG, so the result is an objective-by-optimizer interaction. Published `1e16--1e21` full-metric conditions include two exact softmax gauge nulls; the six-coordinate reduced condition in the audit diagnostic is `1.26e6`. |
| **C5 robustness** | **REFUTED** | The old mixed-pool/leaky result reproduces, including 0.984/0.263, but is not the clean estimand. Correct dominant-mode test-only AUPRC is S1/XGB = 0.944/0.228, 0.945/0.413, 0.945/0.394 for k=3/4/6; minority modes usually reverse. The legitimate-only feature-screen sensitivity (still retaining the predeclared tau=4) collapses to 0.0050 full-test AUPRC. Replace the headline with the conditional, supervised-screened result and disclose every mode. Killing artifact: `priority_attacks_v1.json:C5_clean_mode_holdout`. |
| **C6 headroom null** | **CONFIRMED** | Primary stack +0.000227 [-0.000706,+0.001258]; interaction +0.000307 [-0.00348,+0.00461]; band conditional -0.000210; isotonic variants -0.038 to -0.040. None beats the +0.008 noise-control point threshold, and no positive interval resolves. H4 remains null. |
| **C7 flag-NG** | **CONFIRMED** | The coarse-grained classical-quantum Fisher decomposition, factor-four convention, and 5-versus-21 setting arithmetic at L=4 are correct. Clipping must be labelled regularization, and the flag block is also the classical Bernoulli generalized-Gauss--Newton metric. The first matched implementation modestly beats Adam on all five seeds; ensemble delta +0.00188 [+0.00024,+0.00406]. |
| **C8 alias rule** | **CORRECTED** | The period `pi(M-1)/K` and measured 0.001-to-0.697 crossing reproduce. Period exceeding the observed span is necessary, not sufficient, for rank faithfulness or small approximation error. The code's “trapezoid” weights do not half-weight endpoints. The E.ON schedule tends toward fixed `K/(M-1)~0.8`, explaining non-refinement. |
| **C9 gradient/novelty** | **CORRECTED** | The theory-coordinate identity `dK/dbeta=2F` is algebraically correct. The E.ON referee uses `cos^2(beta/2)` and therefore validates the identity only after a factor-two coordinate bridge. The implementation's raw logits require a chain rule and contain a common-logit gauge. Direct all-ancilla-outcome and nonunitary-LCU-QML prior art narrows this to a candidate new-for-setting failure-sector readout; novelty is not cleared. |
| **C10 resources/deployment** | **CORRECTED** | Abstract 10-qubit/280-CZ/2q-depth-242 counts reproduce, but CZ is not a native Forte compile. At 128 shots, 30-draw AUPRC is mean 0.6436, SD 0.0234, 2.5/97.5% 0.6025/0.6838 versus exact 0.6485; “unchanged” is not an equivalence result. Local transformed-feature timing is 0.505 microseconds/tx S1 and 14.125 microseconds/tx S2 median, not an end-to-end authorization benchmark. Require native compile, noise model, and hardware mechanism referee. |
| **C11 text integrity** | **REFUTED** | The source PDF builds and matches its source-locked extracted text, and score/binary/attribution scaffolds exist. But the inspected scripts/data/artifacts are untracked despite “committed” wording; submission placeholders remain; the 0.70/eight-parameter sentence mixes protocols; “label-free” is false; account-takeover detection is explicitly out of challenge scope; and the proposal lacks the OCC gate and corrected conditions map. |

## 4. OCC-parity gate and method tournament

**Gate verdict: FAIL at both n=8 and n=12.** Every AUPRC is on the same full
56,962-row test set (99 frauds; 0.1738%). Brackets are 2,000 paired Poisson
row-bootstrap intervals. `Delta stack` means versus the preregistered n=8
S2(L=1, seed 500), and versus frozen S2(L=2) at n=12 because no n=12 L=1
replication was preregistered. Parameter counts for nonparametric models are
model-size proxies, not trainable scalar counts. Kernel/circuit compute caps
are explicit; therefore those rows are matched in features/splits/evaluation
but not claimed as exhaustive hyperparameter searches.

| n | method | AUPRC [95% CI] | AUC | Delta stack [95% CI] | Delta XGB-sub [95% CI] | params/proxy | shots/tx | train cost | notes |
|---:|---|---|---:|---|---|---:|---|---|---|
| 8 | IsolationForest | 0.644 [0.541,0.750] | 0.953 | -0.062 [-0.119,-0.013] | -0.105 [-0.177,-0.031] | 1,144,626 | N/A | 170,589 / 2.37 s | legit-only |
| 8 | OC-SVM | 0.073 [0.043,0.113] | 0.928 | -0.633 [-0.714,-0.548] | -0.677 [-0.757,-0.582] | 1,206 | N/A | 20,000 / 0.30 s | legit-only cap |
| 8 | GMM-diag | 0.425 [0.326,0.549] | 0.930 | -0.280 [-0.353,-0.187] | -0.324 [-0.405,-0.218] | 135 | N/A | 170,589 / 0.99 s | 8 comps by legit val likelihood |
| 8 | KDE | 0.002 [0.001,0.005] | 0.384 | -0.704 [-0.792,-0.613] | -0.748 [-0.834,-0.652] | 40,000 | N/A | 5,000 / 28.37 s | legit-only cap |
| 8 | LOF | 0.025 [0.019,0.034] | 0.881 | -0.680 [-0.764,-0.594] | -0.724 [-0.805,-0.634] | 320,000 | N/A | 40,000 / 1.00 s | novelty mode |
| 8 | ECOD-like | 0.613 [0.518,0.706] | 0.942 | -0.093 [-0.145,-0.049] | -0.136 [-0.209,-0.069] | 1,364,712 | N/A | 170,589 / 0.15 s | independent tail score, not PyOD ECOD |
| 8 | **PCA error** | **0.724 [0.635,0.810]** | 0.969 | **+0.018 [+0.008,+0.030]** | -0.026 [-0.071,+0.019] | 64 | N/A | 170,589 / 0.02 s | **best mandatory OCC** |
| 8 | Mahalanobis control | 0.716 [0.625,0.803] | 0.956 | +0.011 [+0.001,+0.023] | -0.033 [-0.080,+0.015] | 72 | N/A | 170,589 / <0.01 s | extra control |
| 8 | deep-SVDD | 0.010 [0.007,0.013] | 0.831 | -0.696 [-0.783,-0.608] | -0.740 [-0.825,-0.647] | 768 | N/A | 170,589 / 0.93 s | bias-free 32/16, 30 epochs |
| 8 | S1 analytic | 0.696 [0.604,0.786] | 0.926 | -0.010 [-0.017,-0.004] | -0.054 [-0.106,-0.005] | 36 | exact | 170,589 | label-screened OCC |
| 8 | S2 L=1 seed 500 | 0.706 [0.615,0.794] | 0.936 | 0 | -0.044 [-0.093,+0.004] | 4 raw / 3 eff | exact | 300 iters | preregistered seed |
| 8 | S2 L=2 frozen | 0.648 [0.549,0.750] | 0.914 | -0.057 [-0.107,-0.011] | -0.101 [-0.154,-0.048] | 8 raw / 6 eff | exact | frozen | original EXP-B branch |
| 8 | dephased LCU | 0.002 [0.001,0.002] | 0.500 | -0.704 [-0.792,-0.613] | -0.748 [-0.835,-0.653] | 8 | exact | frozen | input-independent control |
| 8 | exact four-path | 0.648 [0.549,0.750] | 0.914 | -0.057 [-0.107,-0.011] | -0.101 [-0.154,-0.048] | 8 | classical exact | 4 paths | max score dev `7.8e-16` |
| 8 | XGBoost-subset | 0.749 [0.661,0.835] | 0.971 | +0.044 [-0.006,+0.092] | 0 | 6,014 | N/A | 170,884 | same eight raw features |
| 8 | fidelity QSVC | 0.187 [0.145,0.247] | 0.965 | -0.519 [-0.599,-0.424] | -0.562 [-0.635,-0.474] | 2,984 | exact kernel | 2,295 | 2,000-legit cap + all fraud |
| 8 | projected Z/ZZ QSVC | 0.189 [0.149,0.242] | 0.971 | -0.517 [-0.590,-0.438] | -0.560 [-0.630,-0.481] | 14,328 | exact kernel | 2,295 | 36 projected features |
| 8 | re-upload VQC | 0.014 [0.010,0.019] | 0.898 | -0.692 [-0.779,-0.601] | -0.736 [-0.820,-0.643] | 8 | exact | 679/step x200 | faithful shallow control, not SOTA |
| 8 | QAE anomaly | 0.003 [0.002,0.004] | 0.709 | -0.703 [-0.791,-0.612] | -0.746 [-0.833,-0.652] | 8 | exact | 384/step x200 | shallow trash-qubit objective |
| 8 | shadow features + IF | 0.068 [0.042,0.109] | 0.889 | -0.638 [-0.714,-0.551] | -0.681 [-0.760,-0.592] | 1,771,274 | 128 | 40,000 | 64 Pauli products, 2 shots each |
| 8 | MPS/TT density | 0.080 [0.055,0.115] | 0.867 | -0.626 [-0.704,-0.544] | -0.669 [-0.747,-0.583] | 168 | N/A | 170,589 | rank 4 legit density |
| 8 | RBM legit | 0.002 [0.002,0.003] | 0.444 | -0.704 [-0.792,-0.613] | -0.747 [-0.834,-0.653] | 44 | N/A | 170,589 | 4 hidden units |
| 12 | **IsolationForest** | **0.697 [0.605,0.785]** | 0.948 | **+0.037 [+0.015,+0.064]** | -0.055 [-0.117,+0.009] | 1,110,962 | N/A | 170,589 / 1.68 s | **best mandatory OCC** |
| 12 | OC-SVM | 0.102 [0.060,0.161] | 0.924 | -0.558 [-0.648,-0.456] | -0.650 [-0.736,-0.548] | 2,249 | N/A | 20,000 / 0.27 s | legit-only cap |
| 12 | GMM-diag | 0.421 [0.321,0.534] | 0.918 | -0.239 [-0.316,-0.149] | -0.331 [-0.426,-0.225] | 199 | N/A | 170,589 / 1.16 s | 8 comps |
| 12 | KDE | 0.001 [0.001,0.002] | 0.400 | -0.658 [-0.755,-0.559] | -0.750 [-0.839,-0.649] | 60,000 | N/A | 5,000 / 20.95 s | legit-only cap |
| 12 | LOF | 0.118 [0.088,0.159] | 0.927 | -0.541 [-0.618,-0.455] | -0.634 [-0.707,-0.546] | 480,000 | N/A | 40,000 / 2.63 s | novelty mode |
| 12 | ECOD-like | 0.564 [0.469,0.658] | 0.942 | -0.096 [-0.162,-0.038] | -0.188 [-0.273,-0.099] | 2,047,068 | N/A | 170,589 / 0.20 s | independent tail score |
| 12 | PCA error | 0.639 [0.538,0.736] | 0.913 | -0.021 [-0.040,-0.006] | -0.113 [-0.189,-0.036] | 120 | N/A | 170,589 / 0.02 s | 9 comps |
| 12 | Mahalanobis control | 0.686 [0.586,0.779] | 0.948 | +0.026 [+0.008,+0.049] | -0.066 [-0.131,+0.000] | 156 | N/A | 170,589 / <0.01 s | extra control |
| 12 | deep-SVDD | 0.045 [0.017,0.095] | 0.841 | -0.614 [-0.703,-0.507] | -0.707 [-0.797,-0.606] | 896 | N/A | 170,589 / 0.65 s | bias-free 32/16 |
| 12 | S1 analytic | 0.657 [0.555,0.754] | 0.906 | -0.003 [-0.009,+0.004] | -0.095 [-0.169,-0.022] | 78 | exact | 170,589 | label-screened OCC |
| 12 | S2 L=2 frozen | 0.659 [0.561,0.756] | 0.924 | 0 | -0.092 [-0.164,-0.022] | 8 raw / 6 eff | exact | frozen | three-seed source protocol |
| 12 | dephased LCU | 0.002 [0.001,0.002] | 0.500 | -0.658 [-0.755,-0.559] | -0.750 [-0.839,-0.648] | 8 | exact | frozen | input-independent control |
| 12 | exact four-path | 0.659 [0.561,0.756] | 0.924 | 0 | -0.092 [-0.164,-0.022] | 8 | classical exact | 4 paths | max score dev `7.8e-16` |
| 12 | XGBoost-subset | 0.752 [0.658,0.844] | 0.971 | +0.092 [+0.024,+0.165] | 0 | 8,268 | N/A | 170,884 | same 12 raw features |
| 12 | fidelity QSVC | 0.518 [0.421,0.640] | 0.958 | -0.141 [-0.236,-0.030] | -0.234 [-0.325,-0.131] | 6,576 | exact kernel | 2,295 | capped |
| 12 | projected Z/ZZ QSVC | 0.330 [0.253,0.438] | 0.969 | -0.329 [-0.421,-0.216] | -0.422 [-0.513,-0.311] | 31,824 | exact kernel | 2,295 | 78 projected features |
| 12 | re-upload VQC | **NOT RUN** | -- | -- | -- | 12 | -- | -- | dense parameter-shift budget exceeded |
| 12 | QAE anomaly | **NOT RUN** | -- | -- | -- | 12 | -- | -- | dense parameter-shift budget exceeded |
| 12 | shadow features + IF | 0.119 [0.068,0.195] | 0.883 | -0.541 [-0.636,-0.433] | -0.633 [-0.721,-0.523] | 1,806,222 | 128 | 40,000 | 64 Pauli products |
| 12 | MPS/TT density | 0.661 [0.566,0.754] | 0.877 | +0.001 [-0.020,+0.025] | -0.091 [-0.162,-0.023] | 296 | N/A | 170,589 | ties S2, trails XGB |
| 12 | RBM legit | 0.002 [0.002,0.003] | 0.491 | -0.657 [-0.754,-0.559] | -0.750 [-0.838,-0.648] | 64 | N/A | 170,589 | 4 hidden units |

This is a broad but bounded tournament, not a literature-wide SOTA claim.
In particular, deep-SVDD has the frozen small direct objective without
autoencoder pretraining, and the shallow VQC/QAE are faithful controls rather
than heavily tuned representatives. None of those limitations rescues the
proposal's OCC gate: two very plain, full-row classical models already beat
the scorer under paired evaluation.

## 5. Method improvements

Two prespecified directions were implemented; only one helps, and modestly.

| direction | result | verdict |
|---|---|---|
| Cross-view CCA branch overlap | AUPRC 0.561 [0.466,0.656], AUC 0.947; delta versus S1 -0.134 [-0.192,-0.082] | **WORSE.** This first circuit-realizable equal-weight overlap construction does not implement the theory note's learned branch unitaries and gives no reason to displace S1. |
| Flag-NG versus Adam | Five per-seed AUPRC pairs: 0.7038/0.6984, 0.7042/0.7018, 0.7052/0.7029, 0.7043/0.7029, 0.7045/0.7044 (flag-NG/Adam). Mean per-seed 0.7044/0.7021; ensemble delta +0.00188 [+0.00024,+0.00406]. | **MODEST IMPROVEMENT.** All signs agree, but effect is small and measured only at n=8 L=1. Promote to a controlled optimizer study, not an application advantage. |
| Validation-only calibration/stacking | Logistic +0.000227 unresolved; interaction +0.000307 unresolved; isotonic combinations lose 0.038--0.040. | **NO IMPROVEMENT.** Calibration cannot rescue ranking; H4 stays null. |

The flag-NG diagnostic reports zero clipping at the frozen `1e-4` boundary and
positive-subspace Fisher conditions from roughly `8e2` to `2e4`. It solves a
damped four-coordinate system that still contains one common-logit gauge;
the damping makes the solve finite, but the next implementation should use
the explicit three-coordinate layer parameterization from Appendix B.

## 6. Conditions map

The corrected scarcity curve obeys the total fraud-label budget in feature
selection, tau selection, and model fitting. Values are five-seed means; the
full ranges and per-seed feature lists are in
`improvements_regimes_v1.json:label_scarcity`. The discarded mismatched curve
described in section 1.10 is not used.

| axis | threshold / regime | winning method family | evidence |
|---|---|---|---|
| Standard ULB, n=8 | full labels, 0.1738% test fraud | **Classical supervised/OCC** | XGB-subset 0.749; PCA 0.724; L=1 0.706; S1 0.696. `occ_tournament_v1.json:sizes.8` |
| Standard ULB, n=12 | full labels | **Classical supervised/OCC** | XGB-subset 0.752; IsolationForest 0.697; S2 0.659; S1 0.657. `sizes.12` |
| Fraud-label scarcity | 5 total labels | **Classical OCC** | Corrected means: IsolationForest 0.587, XGB-full 0.546, XGB-subset 0.529, S1 0.483. Ranges are wide; IF is the point winner, not a resolved population theorem. |
| Fraud-label scarcity | 20 total labels | **Classical coverage / near tie** | XGB-subset 0.670 versus S1 0.667, XGB-full 0.630, IF 0.526. No quantum opening is established. |
| Fraud-label scarcity | 50 total labels | **Classical supervised/OCC** | XGB-full 0.719, IF 0.696, S1 0.688, XGB-subset 0.678. |
| Fraud-label scarcity | 200 / all | **Classical supervised** | XGB-full 0.786/0.799 versus S1 0.701/0.691. |
| ULB feature poverty | n=2,4,6,8,12 | **Classical supervised at every n** | XGB 0.608/0.733/0.773/0.761/0.750 versus S1 0.571/0.678/0.696/0.696/0.657. |
| Clean temporal ULB | later chronological 20% | **Classical supervised** | XGB-full 0.792, XGB-subset 0.786, S2 0.753, S1 0.750. The OpenML `Time` column is nondecreasing and labels/Amount match the NPZ exactly. |
| Clean dominant fraud mode | k=3/4/6, train+val centroids, test only | **Analytic one-class filter, conditionally** | Supervised-screened S1 0.944/0.945/0.945 versus retrained XGB 0.228/0.413/0.394. This is one large off-manifold mode, not label-free general robustness. |
| Minority fraud modes | remaining k=3/4/6 modes | **No useful unsupervised winner; XGB usually less bad** | S1 0.0001--0.246; paired differences are mostly negative or unresolved. Legitimate-only-screen S1 is 0.00039 on its k=4 dominant mode. |
| Prevalence stress | exact 0.02% class weighting | **S1 point winner, classically evaluable** | S1 0.498, XGB-subset 0.488, IF 0.336. No row subsampling and no hardware need. |
| Prevalence stress | 0.05--0.50% | **Classical supervised** | XGB 0.626--0.811 versus S1 0.605--0.738 and IF 0.488--0.722. |
| Bounded inward fraud shift | epsilon=0 | **Classical supervised** | XGB 0.749, S1 0.696, IF 0.644. |
| Bounded inward fraud shift | epsilon=0.05/0.10/0.20 | **S1 filter, classically evaluable** | S1 0.592/0.580/0.450 versus XGB 0.549/0.365/0.358 and IF 0.245/0.121/0.018. This normalized-space stress is not an attack-feasibility claim. |
| Sparkov supplied test split | 555,719 rows, 2,145 frauds | **Classical supervised** | XGB 0.8721, IF 0.0640, S1 0.0062. Eight semantic features, full imbalanced evaluation. `sparkov_v1.json` |
| Cross-dataset transfer | only amount/time semantic overlap | **Weak classical OCC; no useful transfer** | ULB->Sparkov IF 0.0537; Sparkov->ULB IF 0.00426. The PCA coordinate mismatch makes this a deliberately weak partial test. |

**Conditions verdict:** classical algorithms do not win every individual row,
but classical *coverage* suffices everywhere measured. The winning S1 rows are
computed exactly by a classical product-distribution expectation and do not
need a quantum processor. The low-label opening vanishes under matched feature
and tau budgets. The data therefore support no statement that a quantum
component is needed on ULB or Sparkov.

## 7. P0/P1 proposal corrections and replacement wording

### P0 -- must fix before submission

1. **Demote the central performance position.** Add the OCC tournament and say
   that PCA/IsolationForest meet or beat the scorer and that XGBoost is
   stronger. Remove any implication that a quantum component is needed on
   these datasets.
2. **Delete the `0.984 vs 0.263 label-free` headline and positive-H4 use.**
   Replace it with the clean k=3/4/6 dominant-mode numbers, every minority
   mode, and the legitimate-only-screen collapse. Call the surviving result
   “supervised-screened, protocol-dependent off-manifold sensitivity.”
3. **Repair parameter/result accounting.** State that L=1 has four raw/three
   identifiable parameters and gives 0.704 source / 0.706 new-seed AUPRC.
   State separately that the original eight-raw/six-identifiable L=2 branch is
   0.648 and that an alternate seed protocol gives 0.705. Do not pair “0.70”
   with “eight parameters” without naming that alternate protocol.
4. **Fix one-class language.** `H` is legitimate-fitted, but the primary
   feature screen and tau use fraud labels and the quantile transform sees the
   accessible train pool. Use “supervised-screened OCC.” Reserve “label-free”
   for a pipeline whose representation, model, and hyperparameters use no
   fraud labels.
5. **Correct theory-to-code notation.** Separate relative branch phase from
   RX mixer angle; expose one logit difference or beta per layer; include the
   raw-logit chain rule; state the E.ON `beta/2` convention bridge; remove
   floor-clipped full-space condition numbers as evidence.
6. **Narrow deployment claims.** Replace abstract CZ counts by a native
   Forte compile and noise/hardware referee before calling the circuit
   plausible. Report the 30-draw shot distributions, not one draw; identify
   transformed-feature timing scope. Hardware should validate flag and
   failure-sector mechanisms, not claim authorization-path utility.
7. **Repair submission integrity.** Publish/commit the scripts, data manifest,
   environment lock, and artifacts at immutable hashes; fill every team,
   contact, publication, and DOI placeholder. Remove account-takeover
   detection from the proposed scope because the challenge statement lists it
   as out of scope. Make the theory note build in the locked environment.

### P1 -- should fix

- Re-express flag-NG in the six-coordinate L=2 identifiable space, replicate
  beyond five n=8 seeds, and compare damping/trust-region choices without
  selecting on test.
- Treat the failed CCA overlap as a negative first implementation. Any learned
  cross-view branch study needs an equally tuned classical cross-view model,
  a dephased control, and exact-path accounting.
- Add nested uncertainty across label-subset seeds and test rows for the
  scarcity curve; the current means/ranges are a conditions map, not sharp
  population thresholds.
- Extend to a license-cleared IEEE-CIS study only after freezing semantic
  features and compute caps. Sparkov already shows that the present S1
  construction does not transfer.
- Rename the LCHS endpoint rule, state the alias inequality as necessary only,
  and add an approximation-error gate alongside ranking.
- Carry the prior-art table in the proposal and label the failure-sector
  identity “candidate new-for-setting, not novelty-cleared.”

### Replacement proposal wording

> We study a compact spectral one-class scorer that is realizable as a
> coherent stacked-LCU circuit and exactly classically evaluable at the tested
> depth. On the full imbalanced ULB test split, analytic S1 reaches AUPRC 0.696
> and a preregistered L=1 training replication reaches 0.706. Plain
> legitimate-only PCA reconstruction reaches 0.724 and XGBoost on the same
> eight features reaches 0.749; therefore we claim neither quantum advantage
> nor a need for a quantum component on ULB. At n=12, IsolationForest likewise
> exceeds the circuit score. The L=2 circuit has an exact four-path classical
> expansion at this depth.
>
> A clean fraud-mode stress test finds a protocol-dependent strength: using a
> fraud-label-selected representation, S1 scores 0.944--0.945 AUPRC on one
> dominant held-out mode while retrained XGBoost scores 0.228--0.413. This does
> not extend to minority modes, and a legitimate-only feature screen fails;
> the result is not label-free robustness. Across matched label scarcity,
> feature poverty, temporal drift, prevalence, bounded shift, and Sparkov,
> classical supervised plus classical OCC methods cover every measured
> regime. Our Phase-1 contribution is consequently the transparent
> circuit-realizable OCC mechanism, a negative conditions map, and a proposed
> hardware validation of success/failure-flag observables and natural-gradient
> geometry--not a production quantum fraud detector.

Replacement H4 wording:

> **H4 -- application (not supported on current data):** validation-only
> stacking of the circuit score with XGBoost gives no resolved AUPRC lift, and
> plain classical OCC models meet or beat the standalone score. H4 remains a
> preregistered future-dataset question; the current ULB/Sparkov evidence is a
> null.

## Appendix A. Reproduction and artifact manifest

### A.1 Source lock

| object | SHA-256 |
|---|---|
| audit handoff | `8f76acf3f036ab9e9c9eb6eabf61412e4c4a987d0a824753c69ca480c0df6779` |
| proposal Markdown | `185e48fa61f7b87e14720c12b41ed81c1fe703b8cc337290c8974648361756f1` |
| proposal TeX | `8d7ab72849ff62d29883ed928764787b6c8db8dd67d116b11b174efbbfb314dc` |
| proposal PDF | `0120eeef94ec8f64e3db1433a70d7c5dc3c3e42c6f5724849d80fe18d2d911e4` |
| theory TeX | `ede03ef4ddfc9b89478517e50d4d4580b281375527e7f4687585da5aa5a61ae5` |
| fit assessment | `bf67227516d8dbb7486633edd55f0741e80b3a8f1a9c7594524e3e9cd11010bd` |
| ULB NPZ | `40ad6e73b1ef5c2b42265b7a02e89caf9738f09520770451a6c8d4491f4d79d6` |
| HSBC challenge statement PDF | `0c20c8079ae100442eb9ab6159958f514fb9eb0b135ef3ffe0ff7fe05202b523` |

Original-script hashes are: `hsbc_common.py`
`66d087d1e10be0160c934c73e8b9c219a90ed9653045f9e3add551407a9ddada`;
EXP-B `bf003a99d4360e5d683545e33f7ad45468d7223af864511305c0886c3d1df79d`;
B2 `0b2059872055e2c1f9a3f306661e1e2d5b4e54f1bcddc6adb14baedf3b4b50d5`;
C `56d03e4a0254eb3b7c1a9dbf1ec306867f1f6f77485dff1da30e97cea303b7ac`;
D `11e2b923aafc1e70596ff333fcb0f996f605fa35d6d0fac4e4d086bb5e744979`;
E `c94d1eeca82eaea27553b3db6d8ad892da3a80af14778198db2093e5e7581496`;
and F `4407763bac8884603ba5b173ae1e1a989ecbdeee6fe85848c99612df3a88436b`.
The environment is macOS 26.4 arm64, Python 3.12.13, NumPy 2.4.6, SciPy
1.18.0, scikit-learn 1.9.0, and XGBoost 3.4.1.

The inspected checkout is branch `stacked-lcu-qng` at
`dbc632aa1ee35f94a547d6c63c3afcd697cf0d0e`. Every HSBC proposal/theory/fit
object, the ULB NPZ, `fit_v0`, and `scripts/hsbc_challenge/` appears as
untracked in this checkout. Consequently the proposal's statement that the
scripts are “committed to the repository” is false for the inspected state;
the hashes above source-lock the audit but do not substitute for a published
commit or archive.

### A.2 Exact reproduction gate

**PASS.** Each source script was executed unchanged with the repo `.venv` in
`/private/tmp/hsbc_audit_repro_20260830`, whose dataset path was a symlink to
the source-locked ULB NPZ. The recursive comparator excludes only
`wall_seconds`; every other field in every file matches at `rtol=1e-12,
atol=1e-14`.

| artifact | comparison |
|---|---|
| `exp_b_scorer.json` | MATCH |
| `exp_b2_addendum.json` | MATCH |
| `exp_c_dequantization.json` | MATCH |
| `exp_d_alignment.json` | MATCH |
| `exp_e_depth.json` | MATCH |
| `exp_f_bottlenecks.json` | MATCH |

The machine-readable comparison is
`runs/hsbc_challenge/audit_v1/baseline_reproduction_v1.json`, SHA-256
`66dc3c41fcc6d67e73cde50a32d934136b2412d2803d86015311e02756043f24`.
The comparator command is:

```bash
.venv/bin/python scripts/hsbc_challenge/audit_reproduction_compare_v1.py \
  --reproduced-root /private/tmp/hsbc_audit_repro_20260830/runs/hsbc_challenge/fit_v0
```

The adversarial runs are reproduced from the repository root with:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/hsbc_challenge/audit_priority_attacks_v1.py
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/hsbc_challenge/audit_occ_tournament_v1.py
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/hsbc_challenge/audit_improvements_regimes_v1.py
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/hsbc_challenge/audit_sparkov_v1.py \
  --sparkov-dir /private/tmp/hsbc_sparkov_20260830
```

### A.3 Document build and visual QA

The proposal rebuilds under `pdflatex` in two passes and its extracted text is
byte-identical to the source-locked PDF's extracted text. Poppler rendering of
all seven pages found no clipped text or unreadable table. Page 5 is almost
blank because of the forced appendix break. Submission-blocking placeholders
remain for team name, affiliation/contact, member biographies, public mirror,
and publications/repository DOI.

The companion theory note does **not** build in the declared local TeX
environment: `pdflatex -halt-on-error` stops at `File 'cleveref.sty' not
found`. This occurs before the handoff's anticipated `tcolorbox` dependency is
reached. Either vendor the dependency set or provide a locked build container.

### A.4 Audit artifact manifest

The authoritative manifest is
`runs/hsbc_challenge/audit_v1/audit_manifest_v1.json`. It contains full hashes
for all five audit scripts, five JSON result files, four NPZ score bundles,
the baseline comparator, ULB, and both Sparkov CSVs. Every embedded NPZ hash
was independently recomputed and matches; every JSON parses; every stored
array is finite. Sparkov uses the Kaggle listing
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection),
with the supplied test file evaluated in full.

## Appendix B. Theory and prior-art audit

### B.1 Measured implementation versus proposed parameterization

The theory uses

`K = cos^2(beta) U0 + sin^2(beta) exp(i phi_rel) U1`

and obtains `dK/dbeta = 2F`. The measured HSBC implementation instead uses
per-layer `[logit0, logit1, theta_cost, phi_RX]`, positive softmax weights, and
no explicit relative branch phase (`hsbc_common.py:233-345`). Adding a common
constant to both logits changes nothing, so the advertised eight raw L=2
parameters contain two exact gauge directions: there are six identifiable
parameters. The proposal also uses `phi` for both the relative branch phase
and the RX mixer angle. A beta coordinate can be introduced from the logit
difference, but raw-logit gradients then require its chain rule; the theory
identity is not literally the derivative implemented by either logit.

The headline “AUPRC 0.70 with 8 trainable parameters” also mixes protocols.
The preregistered ablation's L=1 result near 0.704 has four raw/three effective
parameters. The EXP-B L=2 eight-raw-parameter best-of-five result is 0.648,
whereas a different seed family in EXP-E gives L=2 near 0.705. Both are
disclosed later in the proposal, but the executive sentence presents only the
favorable parameter count/result pairing and must be made protocol-specific.

The proposed learned cross-view branch unitaries are theory only. The measured
branches are the Ising cost evolution and collective RX mixer. This does not
invalidate the abstract cross-view construction, but it prevents the measured
ULB numbers from serving as evidence for that mechanism.

Finally, `H` is fitted on legitimate rows, but the eight features are selected
by training-label AUC (`hsbc_common.py:91-96`) and tau is selected using
validation labels. The complete pipeline is therefore a supervised-screened
one-class scorer, not “label-free.” Only the separately reported legitimate-
only unsupervised-screen sensitivity earns that label.

### B.2 Commuting dequantization and no-free-sampling boundary

For product encoding and a commuting diagonal filter, `p_s(x)=E_q[g]` with
`0<=g<=1`. Therefore `Var_q(g)/N <= p_s(1-p_s)/N`, because `E[g^2]<=E[g]`.
Classical rejection sampling from `q` with acceptance `g/g_max` has rate
`p_s/g_max >= p_s`. These arguments and EXP-C are correct. They cover S1 and
diagonal stacks; they do not cover an entangling/noncommuting cross-view stack.
The mixer experiment only refutes this particular product-measure shortcut;
it is not a classical-hardness result, and the explicit four-path expansion
remains exact at L=2.

For a target state `z*`, `p_s P(z*|S)=q0(z*)g(z*)<=q0(z*)`; for a degenerate
target set replace both sides by the corresponding sum. The proof permits
nonuniform `q0`. Amplitude amplification is a genuine quadratic query-level
escape from raw rejection overhead, but it neither changes the conditional
target distribution nor creates a super-quadratic escape from small initial
target mass. Thus “mixers required” is correct only for raw repeated sampling
from this fixed-input diagonal-filter family, absent amplitude amplification
or a changed/warm-started input.

### B.3 Flag Fisher and metric arithmetic

For the classical-quantum state that retains the normalized success state and
lumps every failure into a fixed symbol, the Fubini--Study-normalized geometry
is

`G* = p_s g_cond + grad(p_s)grad(p_s)^T/[4 p_s(1-p_s)]`.

The factor four relative to the classical Bernoulli Fisher is correct. The
setting arithmetic is also correct: flag Fisher needs `1+L`; the stated full
mixing-angle block needs `1+2L+2*C(L,2)`, hence 5 versus 21 at L=4. This is a
geometry of the deliberately coarse-grained output; retaining parameter-
dependent failure states adds their conditional geometry.

Numerically clipping the denominator near zero or one changes the metric and
must be declared as regularization, not treated as the exact Fisher. The flag
Fisher is also a classical generalized-Gauss--Newton/outer-product metric for
the measured Bernoulli model; “quantum” is not what gives that matrix its
form. The preregistered flag-NG versus Adam result decides practical value.

### B.4 LCHS alias rule

Uniform nodes have spacing `Delta k=2K/(M-1)`, so the finite exponential sum
has period `2pi/Delta k=pi(M-1)/K` in `tau E`. A period exceeding the observed
`tau*span` is necessary to avoid a wrap across that interval, but it is not
sufficient for a small filter error or globally monotone ranking. The HSBC
crossing is therefore an empirical result plus a necessary alias diagnostic,
not a general rank-faithfulness theorem. The E.ON schedule has
`K/(M-1)={1.000,0.857,0.800,0.806,0.794}` for M=4..64: it approaches roughly
0.8 by keeping node spacing non-refining, consistent with the earlier E.ON
audit's nonconvergence finding.
The implementation labels its weights “trapezoid” while assigning full rather
than half weight to both endpoints (`hsbc_common.py:211-219`); the reported
finite-sum numbers remain reproducible, but the quadrature label is wrong.

### B.5 Gradient identity and novelty scope

The E.ON referee reruns: sector-versus-finite-difference residuals are
`8.44e-11` and `9.52e-11`; X-setting objective and success gradients match to
printed precision. Its code, however, parameterizes weights as
`cos^2(beta/2), sin^2(beta/2)`, so its stored failure sector equals the
derivative with respect to that script angle. The theory's `cos^2(beta)`
coordinate instead gives `dK/dbeta=2F`. The result is correct after the
factor-two coordinate bridge, not as an unqualified same-symbol validation.

| source | what it establishes | safe novelty boundary |
|---|---|---|
| [Faehrmann, Eisert, Kueng (2025)](https://arxiv.org/abs/2505.15913) | uses the Hadamard-test garbage/system register with classical shadows | not a paper about reusing discarded LCU ancilla outcomes; the proposal's grouping is inaccurate |
| [Daskin (2026)](https://arxiv.org/abs/2605.02986) | exploits all LCU ancilla outcomes for low-rank recovery/trapdoor structure | direct prior art for “use all LCU outcomes,” not for this gradient identity |
| [Heredge et al. (2024/2025)](https://arxiv.org/abs/2405.17388) | nonunitary/LCU QML classifiers and layers | close architectural prior art; fraud application and flag readout are new-for-setting at most |
| [Masta et al. (2026)](https://arxiv.org/abs/2603.27377) | large empirical comparison of nonunitary LCU-QML layers, unitary controls, and Fisher-efficiency regimes | newer direct prior art for empirical nonunitary-QML/Fisher positioning; not the same failure-sector identity |
| [Khatri, Zohren, Matos (2026)](https://arxiv.org/abs/2607.24686) | stacked-LCU trainability results for a specified free-fermion family | no theorem transfer to the measured HSBC Ising-plus-RX model |

A bounded search did not locate the exact fresh-ancilla failure-sector
derivative identity. It remains an elementary candidate new-for-setting
identity, **not novelty-cleared**. “Failure-sector gradient readout” is safer
than “gradient recycling,” which can imply a broader discarded-shot benefit.

### B.6 Resource and hardware boundary

The stored 10-qubit/280-CZ/two-qubit-depth-242 count is reproducible for an
unrouted abstract `cz/rz/sx/x` basis. Forte-class hardware is all-to-all and
large enough, but its native interface uses GPI/GPI2 and Forte ZZ gates; a CZ
count is not a native compiled gate count. [IonQ's native-gate documentation](https://docs.ionq.com/sdks/qiskit/native-gates-qiskit)
explicitly distinguishes Forte ZZ from the abstract gate set, and the current
[Forte Enterprise specification](https://www.ionq.com/quantum-systems/forte-enterprise)
reports 0.4% two-qubit randomized-benchmarking error. A naive independent-gate
survival proxy is `0.996^280 ~= 0.326`; it is not a circuit-fidelity estimate,
but it shows why qubit count/connectivity alone do not establish plausibility.
[Amazon Braket's IonQ mitigation documentation](https://docs.aws.amazon.com/braket/latest/developerguide/error-mitigation-ionq.html)
also imposes shot and gate-count constraints. C10 therefore requires a native
compile, noise/referee run, and mechanism validation before a hardware claim.

## Appendix C. Priority-attack detail

### C.1 Clean fraud-mode holdout

Every row below contains all 56,863 legitimate test rows and only the named
test-fraud cluster. Centroids use train+validation frauds only. `Delta` is S1
minus retrained XGBoost AUPRC under the paired bootstrap; `*` marks the
dominant mode chosen by train+validation count before test metrics.

| k / cluster | fit frauds | test frauds | test prevalence | S1 AUPRC | XGB AUPRC | Delta [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| 3 / 0* | 317 | 70 | 0.1230% | 0.9441 | 0.2278 | +0.7162 [+0.5973,+0.8122] |
| 3 / 1 | 36 | 14 | 0.0246% | 0.00053 | 0.00276 | -0.00223 [-0.00762,-0.00034] |
| 3 / 2 | 40 | 15 | 0.0264% | 0.01266 | 0.00910 | +0.00356 [-0.01621,+0.01942] |
| 4 / 0* | 296 | 66 | 0.1159% | 0.9447 | 0.4128 | +0.5319 [+0.3829,+0.6736] |
| 4 / 1 | 27 | 8 | 0.0141% | 0.00027 | 0.00108 | -0.00081 [-0.00584,+0.00000] |
| 4 / 2 | 48 | 20 | 0.0352% | 0.00990 | 0.01693 | -0.00703 [-0.05068,+0.00732] |
| 4 / 3 | 22 | 5 | 0.0088% | 0.24599 | 0.35556 | -0.10956 [-0.41075,+0.05556] |
| 6 / 0 | 32 | 9 | 0.0158% | 0.01921 | 0.04892 | -0.02971 [-0.19793,+0.01268] |
| 6 / 1* | 296 | 66 | 0.1159% | 0.9447 | 0.3940 | +0.5507 [+0.4152,+0.6863] |
| 6 / 2 | 13 | 5 | 0.0088% | 0.00027 | 0.00190 | -0.00163 [-0.00716,-0.00010] |
| 6 / 3 | 22 | 5 | 0.0088% | 0.24599 | 0.28032 | -0.03433 [-0.24399,+0.15093] |
| 6 / 4 | 16 | 3 | 0.0053% | 0.00006 | 0.00010 | -0.00004 [-0.00023,+0.00004] |
| 6 / 5 | 14 | 11 | 0.0193% | 0.00062 | 0.00909 | -0.00847 [-0.06157,-0.00007] |

The old fraud rows were dropped from XGBoost fitting, not relabelled as
legitimate. Its liberties were cluster leakage, mixed-split evaluation, and
post hoc k/mode emphasis. The clean dominant ordering is real but narrower.

### C.2 L=1 replication, hybrid attack, and shot distribution

| C1 seed | 500 | 501 | 502 | 503 | 504 | 505 | 506 | 507 | 508 | 509 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test AUPRC | .7058 | .7029 | .7052 | .7038 | .7017 | .6980 | .7033 | .6992 | .7038 | .7057 |
| delta vs S1 | +.0101 | +.0073 | +.0095 | +.0081 | +.0060 | +.0023 | +.0076 | +.0035 | +.0081 | +.0100 |

| C6 integration | AUPRC | delta vs XGB [95% CI] |
|---|---:|---|
| XGB-full | 0.79818 | 0 |
| logistic stack | 0.79841 | +0.000227 [-0.000706,+0.001258] |
| logistic + interaction | 0.79849 | +0.000307 [-0.003480,+0.004615] |
| isotonic mean | 0.75788 | -0.04030 [-0.08151,-0.00900] |
| isotonic product | 0.76009 | -0.03809 [-0.07536,-0.00933] |
| validation-band conditional | 0.79797 | -0.000210 [-0.001336,+0.000870] |

| shots/tx | mean AUPRC | SD | median | 2.5/97.5% | mean delta vs exact | delta 2.5/97.5% |
|---:|---:|---:|---:|---|---:|---|
| 128 | 0.64365 | 0.02344 | 0.64422 | [0.60252,0.68375] | -0.00483 | [-0.04596,+0.03528] |
| 1,024 | 0.66528 | 0.01632 | 0.66235 | [0.63873,0.69021] | +0.01681 | [-0.00975,+0.04174] |
| 8,192 | 0.66328 | 0.01783 | 0.66806 | [0.62693,0.68749] | +0.01481 | [-0.02154,+0.03901] |

Exact L=2 AUPRC is 0.64847. Finite shots perturb many closely spaced scores;
the exact array has 712 duplicate rows and minimum distinct gap `8.45e-12`.
No equivalence margin was preregistered, so the proper conclusion is a noisy,
unresolved distribution--not proof of unchanged ranking.

### C.3 Validation and latency diagnostics

With 98 validation frauds, tau=4 is selected in only 47.65% of the 2,000
paired Poisson draws; tau=0.5 is selected 28.35%, tau=1 10.95%, tau=2 10.20%,
and tau=6 2.85%. Differences of tau 0.5/1/2/6 versus tau=4 all have intervals
containing zero. Tau=8 and 12 are resolved worse. This is a stable exclusion
of large tau, not stable identification of tau=4.

The frozen n=8 local timing benchmark, after one warm-up and over five complete
test repetitions, gives:

| scorer | median | min | max | scope |
|---|---:|---:|---:|---|
| S1 analytic | 0.505 microseconds/tx | 0.492 | 0.507 | transformed angles through score |
| S2 L=2 structured | 14.125 microseconds/tx | 14.120 | 14.237 | includes product-state construction |

Raw feature-to-quantile transformation, service overhead, serialization, and
hardware queue time are excluded.
