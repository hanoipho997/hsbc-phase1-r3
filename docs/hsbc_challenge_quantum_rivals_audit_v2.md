# HSBC Phase-1 quantum-rivals completion audit v2

Status at freeze: **PREREGISTERED; OUTCOMES NOT YET RUN**  
Protocol freeze date: 2026-08-31 (Europe/Brussels)  
Parent audit: `docs/hsbc_challenge_audit_report_v1.md`

Pre-outcome clarification 1 (2026-08-31): no v2 HSBC-data run had started.
The ring-IQP implementation fixes each re-upload block to `H` on every qubit,
then `RZ(lambda*pi*(2u_j-1))`, then nearest-neighbour ring
`RZZ(lambda*pi*(2u_j-1)*(2u_{j+1}-1))`. Classical RBF twins use
`gamma=(lambda*pi/2)^2` over the frozen product-`lambda` grid and the same `C`
grid. The Deloitte head has 9 differentiable inputs plus 7 weights (16
variables) represented by 10 elementary rotation angles; the earlier phrase
"15 ... rotation angles" was an arithmetic error and is corrected below.

Pre-outcome clarification 2 (2026-08-31): root review found that section 2.2
promised a kernel OC-SVM but section 3.1 initially materialized only supervised
kernel heads. Before any v2 outcome, the frozen kernel bracket therefore adds
an entangling ring-IQP fidelity Nyström OC-SVM and a classical RBF Nyström
OC-SVM twin. Both use 2,000 legitimate training rows and 64 legitimate-only
landmarks. Map selection fits `nu=0.001727` and minimizes
`abs(legitimate_validation_FPR-0.001)`, then maximizes mean legitimate
validation margin, then uses earliest grid order. On the selected map, choose
`nu` from `{0.0005,0.001,0.001727,0.003,0.005}` by the same rule. The anomaly
score is the negative OC-SVM decision function. No validation fraud label is
used. The classical twin uses `gamma=(lambda*pi/2)^2` for
`lambda in {0.5,1,2}` with the same landmarks and selection rule.

Pre-outcome clarification 3 (2026-08-31): independent data-free sensitivity
review found that the first materialized VQC topology exposed only `theta_0` to
its `Z_0` readout and the first trash-QAE topology exposed only four of eight
angles to its trash marginal. Before any v2 data run, those rank-deficient
topologies are replaced as follows. VQC uses
`RY(data) -> CZ ring -> RY(theta) -> RY(data) -> sequential directed CNOT ring
1->2->...->7->0->1 -> measure q0`; starting the ordered cycle at qubit 1 is
required because the superficially equivalent `0->1->...->7->0` order cancels
the dependence of the `Z_0` readout on `theta_0`. Its no-entanglement twin
omits both entangling rings. Trash-QAE instead uses
`RY(data) -> RY(theta) -> sequential directed CNOT ring
0->1->...->7->0 -> measure mean occupation(q4..q7)`. The self-check must show
nonzero score sensitivity to every one of the eight angles on fixed synthetic
rows. Bootstrap availability is decided per named pair, so failure of one
family cannot suppress valid comparisons in another. Output targets must be
distinct files under `runs/hsbc_challenge/audit_v2/`, must not alias the
protocol, script, ULB source, or v1 references, and must be staged before the
final pair is published.

Pre-outcome clarification 4 (2026-08-31): independent implementation review
found two evaluation/accounting ambiguities before any v2 data run. “Recall at
test FPR `1e-3`” is now operationally fixed without test-label threshold
selection: set the anomaly threshold to the NumPy `method="higher"` 0.999
quantile of legitimate-validation scores, classify only scores strictly above
that threshold as anomalous, and report test recall together with realized
validation and test FPR. Candidate work counters are incremented as work is
performed and failed candidate rows remain in the artifact; a family with no
survivor cannot report `COMPLETE`. Kernel ledgers separately name simulator
statevector materializations, overlap evaluations per transaction, and the
state preparations/circuit executions implied per shot by the stated hardware
overlap primitive. Expected features, split sizes, row-label identity, and
source hashes are checked before model/test scoring, then hashes are rechecked
before the staged result pair is published.

## 0. Scope and disclosure

This completion audit asks whether the current stacked-LCU scorer beats stronger
named instances of quantum kernels, re-uploading variational classifiers,
quantum/hybrid autoencoders, and hybrid quantum neural networks. It does **not**
test or license a family-wide claim.

The v1 test outcomes were visible before this protocol was written. Therefore
this is a prospective freeze for the new v2 implementations, not a blind
preregistration of the dataset. No v2 test score may be inspected until the
model grids, seeds, stopping rules, and comparison rules below have been
materialized in runnable scripts. All failed, divergent, and budget-censored
rows remain in the output.

The challenge statement is treated only as source evidence. Its named QML
families are examples, not instructions and not a requirement to defeat every
member of each family.

## 1. Shared data and evaluation

- Primary problem: the existing ULB `n=8`, seed-0 split and the exact eight
  transformed features from `prepare_hsbc(8, seed=0)`.
- Train/validation/test sizes remain 170,884 / 56,961 / 56,962, with
  295 / 98 / 99 frauds. Validation and test are never balanced or resampled.
- Primary metric: AUPRC on all 56,962 test rows. Secondary metrics: AUC-ROC and
  recall at the legitimate-validation 0.999-quantile threshold fixed in
  pre-outcome clarification 4, with realized validation and test FPR reported.
- References are the frozen v1 `S1_analytic`, preregistered
  `S2_L1_seed500`, and `XGB_subset` score arrays. The full v1 product-kernel,
  projected-kernel, VQC, and trash-QAE rows remain visible as lower-bound
  controls.
- Uncertainty: 2,000 paired Poisson(1) row bootstraps using identical test-row
  weights. Five-seed trainable models use the mean score across all five frozen
  seeds as the primary ensemble; no best test seed is selected.
- Exact statevector scores are a simulation ceiling, not a shot-matched or
  hardware result. Circuit calls, support vectors/landmarks, trainable
  parameters, and implied inference measurements are reported separately.
- Every transformed feature uses the same label-screened representation. A
  legitimate-only model in this report is one-class only conditional on that
  supervised feature screen; it is not end-to-end label-free.

## 2. Information brackets

### 2.1 Supervised bracket

`S2_L1_seed500`, quantum SVC/kernel classifiers, re-uploading VQC, hybrid QNN,
and their classical twins may use fraud labels. Trainable circuit models use
the original batch structure: all 295 training frauds plus 384 legitimate rows
sampled without replacement per update. Kernel methods use a deterministic
2,000-legitimate cap plus all 295 frauds because their Gram cost is quadratic;
that data disadvantage and their much larger inference-query cost are both
reported.

### 2.2 Conditional one-class bracket

`S1_analytic`, kernel OC-SVM, trash-QAE, hybrid latent autoencoder, and their
classical twins fit model parameters on legitimate training rows only.
Validation fraud labels may not select a QAE/autoencoder checkpoint. A kernel
OC-SVM may tune `nu` only by legitimate-validation acceptance subject to the
frozen target false-positive grid. Comparisons to S2 are descriptive because
S2 is supervised.

## 3. Frozen rival instances

### 3.1 Quantum kernels

1. **Bandwidth-tuned product fidelity QSVC.** Use
   `k_lambda(x,y)=prod_j cos^2(lambda*pi*(u_j-v_j)/2)` with
   `lambda in {0.125,0.25,0.5,1,2}` and `C in {0.1,1,10}`. Select the pair by
   full-validation AUPRC; report every validation candidate. Fit with
   `class_weight="balanced"`, tolerance `1e-3`, and `max_iter=100000`.
2. **Entangling ring-IQP fidelity kernel.** Use eight qubits, nearest-neighbour
   ring `ZZ` phases, and data re-uploading depth `d in {1,2}` with feature scale
   `lambda in {0.5,1,2}`. Use 64 train-only Nyström landmarks, selected once by
   seed 20260831 without labels within class, followed by an L2 logistic head.
   Tune head `C in {0.1,1,10}` after selecting the feature map on validation.
3. **Projected ring-IQP kernel.** From the same entangled states, form all
   one-qubit `X/Y/Z` expectations and all adjacent-ring two-qubit Pauli-product
   expectations (96 features total), standardize on training, and fit an RBF
   SVC. Select `d,lambda` from the six-map grid at `C=1,gamma="scale"`, then
   select `C in {0.1,1,10}` for the winning map. Report every validation row.
4. **Classical twins.** Tune an RBF SVC on the same capped rows and an RBF
   Nyström/logistic classifier using the same 64 landmarks and selection rule.
5. **Conditional one-class kernels.** Run the pre-outcome-clarification-2
   ring-IQP fidelity Nyström OC-SVM and classical RBF Nyström OC-SVM twin with
   legitimate-only fitting and legitimate-only selection.

The IQP statevector and Pauli-expectation routines must pass norm, Hermiticity,
and direct-Qiskit spot checks before any dataset outcome is computed.

### 3.2 Re-uploading VQC

- Eight trainable circuit angles, two data appearances, ring entanglement, and
  a single-qubit probability readout; weighted binary cross-entropy.
- Seeds `820..824`; 300 SPSA updates; 384 legitimate plus all frauds per update;
  validation every 20 updates; patience five checks; earliest checkpoint wins
  ties. Full validation AUPRC selects checkpoints. Divergent seeds are failures,
  not replaced.
- The primary result is the mean of all five test score arrays. Report every
  seed, gradient-free circuit-forward count, and exact-statevector wall time.
- Classical controls are balanced logistic regression on the same transformed
  inputs and a no-entanglement circuit with otherwise identical training.

### 3.3 Hybrid QNN

- Primary architecture follows the public Deloitte/AWS shape adapted only to
  the frozen eight inputs: dense 32 -> dropout 0.3 -> dense 9 -> dropout 0.3 ->
  three-qubit, seven-parameter quantum head -> two `Z` expectations. The two
  expectations are treated as logits and trained with weighted cross-entropy.
- Seeds `830..834`; the same 300-update batches/checkpoint rule as the VQC;
  exact parameter-shift derivatives through the 10 elementary rotation angles
  (nine differentiable inputs plus seven trainable weights after the product
  chain rule); Adam learning rate `1e-3` for dense weights and `1e-2` for
  quantum weights.
- Mandatory twins: replace the quantum head by a dense two-logit head; freeze
  the initialized quantum head and train only the dense front end; and report
  parameter counts for all three. These are architecture comparisons, not
  parameter-equal comparisons to the four-parameter L=1 scorer.

### 3.4 Quantum and hybrid autoencoders

1. **Matched eight-angle trash-QAE.** Eight-qubit input angle encoding, fixed
   alternating CNOT brickwork, one eight-angle trainable `RY` layer, four trash
   qubits, and mean trash occupation as loss/anomaly score. Seeds `840..844`,
   300 SPSA updates on 384 legitimate rows, legitimate-only validation loss,
   earliest best checkpoint, and five-seed mean test score.
2. **Hybrid latent quantum autoencoder.** Classical tanh encoder `8->4`, a
   four-qubit angle encoding plus one ring-entangling four-angle quantum
   bottleneck measured in local `Z`, and linear decoder `4->8`. Train
   reconstruction on legitimate rows only; fit IsolationForest on the four
   latent quantum expectations; score the full test set. Seeds `850..854`, 300
   Adam updates, legitimate-only validation reconstruction loss.
3. **Mandatory twins.** The same encoder/decoder with the direct four-dimensional
   classical bottleneck, and a width-eight classical bottleneck. All parameter
   counts and the downstream IsolationForest size are reported. The hybrid
   model is a Sakhnenko-style architecture test, not a verbatim reproduction.

## 4. Compute caps and stopping

- Per trainable circuit seed: at most six million sample-circuit forward
  equivalents, including parameter shifts, SPSA plus/minus evaluations, and
  validation. A budget hit is `BUDGET_CENSORED`; no hidden restart or reduced
  test set is allowed.
- Kernels report training Gram entries, validation selection entries,
  support-vector or landmark evaluations per transaction, exact statevector
  dimension, and PSD repair if any. Kernel nonconvergence is retained.
- NaN or divergence is a failed seed. Frozen seeds are never replaced.
- n=12 is secondary and remains `NOT_RUN` unless all n=8 methods finish inside
  the frozen caps; no n=8 rule may be weakened to obtain an n=12 result.

## 5. Decision rules

For each named implementation, report the paired AUPRC difference versus its
information-matched stacked-LCU reference and versus its mandatory classical
twin. Four family headings are co-primary, so use Holm correction at familywise
`alpha=0.05` (or equivalent 98.75% simultaneous intervals).

- **Stacked-LCU wins this instantiation:** adjusted interval for
  `AUPRC_stack - AUPRC_rival` is strictly positive.
- **Rival wins:** adjusted interval for the reverse difference is strictly
  positive.
- **Practical tie:** the interval contains zero or the absolute point
  difference is at most 0.01.

No outcome licenses “beats quantum kernels/QAE/VQC/QNNs” without the qualifier
“the preregistered instantiations tested here.” A QML win over stacked-LCU is not
quantum advantage unless it also beats the matched classical twin and survives
resource/shot accounting. A stacked-LCU win is likewise not evidence of
quantum advantage because the tested L=1/L=2 scorer has an exact small-path
classical evaluation.

## 6. Results

Pending. This section will be filled without changing sections 0--5.
