# Coherent Cross-View Syndromes for Credit-Card Fraud: Mechanism Validation Under Classical Stop Gates

**2026 Global Quantum + AI Challenge — Phase 1 Concept Proposal**
**Revision 3, 2026-09-08.** This revision incorporates the two
adversarial audits, the prospectively frozen Engine-A result, the frozen-mixer
control, the bounded QML-rivals bracket, and the corrected novelty dossier.
Superseded claims remain visible in the audit trail.

**Problem statement addressed:** HSBC — *Quantum-Enhanced Credit Card Fraud
Detection for Digital Payment Ecosystems*

**Team:** Ha Cong Nguyen (single-member team).
**Lead and contact:** Ha Cong Nguyen, via the public project repository:
`https://github.com/hanoipho997/hsbc-phase1-r3`.
**Model-development and model-risk owner:** Ha Cong Nguyen—responsible for
evidence integrity, classical stop gates, calibration/drift monitoring, and
requiring independent validation before any production influence.

---

## 1. Decision problem and proposal thesis

Credit-card fraud detection needs a probability-like score, a binary decision,
per-transaction attribution, and evidence on unchanged-prevalence data. We use
AUPRC as the primary ranking metric because it is recommended for the extreme
imbalance in these benchmarks, alongside AUC-ROC, precision, recall, F1,
calibration, and recall at fixed review budget.

We propose a compact, circuit-realizable coherent anomaly and syndrome
mechanism, then subject it to unchanged-prevalence classical and QML twins.
The fixed-L=1 mixer increment replicates, and angle adaptation beats the
prespecified random-freeze distribution under a separate control. The scorer
is not statistically resolved from our strongest product quantum kernel or
hybrid QNN, while competent classical models remain as good or better. We
claim no quantum advantage. Phase 2 is therefore a capped Amazon Braket
mechanism-validation study with exact classical twins and stop gates, not a
promise to improve a production fraud backbone.

**Three claims, in order of strength:**

1. **Verified mechanism.** In a fresh-ancilla binary stacked-LCU
   implementation, the joint pass/fail pattern is an interpretable coherent
   measurement. Complete path dephasing at the declared insertion point makes
   its pattern law input independent. The endpoint, sector algebra, and an
   X-readout success-derivative identity have exact and circuit referees.
2. **Bounded named-instance competitiveness.** On the frozen ULB split,
   stacked-LCU's differences from the strongest tested product-fidelity QSVC
   and Deloitte/AWS-shaped hybrid QNN are unresolved; equivalence is not
   established, and classical point estimates are as good or better. This is
   a common-split predictive-quality bracket, not a matched-runtime or
   matched-hardware-resource tournament.
3. **Application result.** On the prospectively frozen IEEE-CIS
   chronological experiment, every application gate failed and the quantum
   residual harmed AUPRC. The classical backbone remains the detector; the
   circuit is monitoring-only unless a future prospectively registered
   no-harm and benefit gate passes.

Prior fraud-QML metrics use materially different, often balanced or reduced
protocols. We do not infer how external models would perform under ours. Our
own unchanged-prevalence bracket supplies the comparison we actually ran.

The deployed-size circuit and its exact `2^L` path twin are classically
tractable. No result below shows runtime, energy, sample-complexity, or
end-to-end fault-tolerant advantage.

## 2. Technical approach

### 2.1 Architecture and parameter coordinates

A transaction is reduced to 12 features in three semantic views A/B/C
(transaction; card/address; identity/device), then quantile-angle encoded on
12 qubits. Three binary LCU layers act with one fresh ancilla per layer; layer
`j` mixes two view-local unitaries behind `PREP–SELECT–PREP†`:

\[
K_j=\cos^2\!\beta_j\,U^{(V)}+
    \sin^2\!\beta_j e^{i\phi_j}U^{(W)},\qquad
F_j=\cos\beta_j\sin\beta_j
    \left(e^{i\phi_j}U^{(W)}-U^{(V)}\right),
\]

where `U^(V)=exp(-i theta H_V) RX_V(varphi)`. Thus
`partial K_j / partial beta_j = 2F_j`. This is an exact X-readout derivative
identity, not a claim of a new general differentiation method. In Engine A,
PREP weights use two softmax logits `(l_0,l_1)` per layer. The common-logit
direction is a gauge and

\[
\partial_{l_0}K=-\sqrt{a_0a_1}\,F,\qquad
\partial_{l_1}K=+\sqrt{a_0a_1}\,F.
\]

The three-layer model therefore has 18 raw but 15 identifiable parameters.
Khatri, Zohren and Matos already define S-LCU as sequential composition of
independent LCU blocks and study its trainability and simulation scaling. We
claim neither the architecture nor the name; our setting-specific object is
the fresh-ancilla syndrome/dephasing measurement package.

### 2.2 Syndrome measurement and exact twins

One Z-basis ancilla setting yields eight joint pattern bins—seven independent
values—for the three view-pair checks `(A,B)`, `(B,C)`, `(A,C)`. Finite shots
are still required. The complete-dephasing endpoint is the input-independent
product of `Bernoulli(2a_0a_1)` factors under these explicit assumptions:
fresh ancillas, unitary branches, exact within-layer dephasing after SELECT and
before `PREP†`, and no later operation on that layer's ancilla.

At current width, the full partial-dephasing curve is reproduced by an exact
`2^L` classical path expansion. It is the mandatory classical twin and makes
the experiment a hardware-coherence diagnostic, not a hardness claim.

### 2.3 Training and explainability

Mixing-angle success gradients use one X-basis ancilla setting per layer;
branch parameters use finite-Fourier shifts. The success-flag natural-gradient
metric is the Fisher metric of the measured Bernoulli flag. For the deployed
L=3 mixing-angle/flag block it uses `1+L=4` settings versus 13 for the full
conditional-state construction. This is a settings comparison, not a uniform
shot advantage and not an accounting of all 18 Engine-A parameters. The
measured-output Fisher is directly measurable and noise-compatible; it is not
the only well-defined noisy metric.

Explainability has three layers: the legit-fitted per-view Ising tables;
named view-pair syndrome failures; and local parameter-shift sensitivity and
energy contributions. The classical backbone receives SHAP explanations.
The quantum rail never replaces the backbone explanation or probability.

## 3. Evidence, strongest first

### 3.1 Verified mechanism and simulator-to-hardware bridge

Dense algebra and circuit referees reproduce states, sectors and derivatives
at `10^-13–10^-16`. The dephased endpoint agrees with the exact formula to
`8.3e-16` in the small-width density-matrix referee. A Braket-SDK **local**
density-matrix run of a reduced 6-system+3-ancilla instance agrees with its
exact path twin to `3.3e-15`; it is not SV1, DM1, a managed simulator, or QPU
evidence.

The original uncertainty analysis was corrected post-outcome because it
deduplicated bootstrap rows and failed to redraw shots. The corrected nested
row-and-shot analysis retains duplicate rows, redraws 2,000 multinomial shots
per row and endpoint, and uses 99.5% intervals for the intended ten-decision
family. On 32 IEEE-CIS validation rows, all five ideal-simulator contrasts
`W(0)-W(1)` remain positive:

| model seed | exact contrast | corrected nested 99.5% interval |
|---:|---:|---:|
| 900 | 0.07396 | [0.02062, 0.11659] |
| 901 | 0.03174 | [0.01318, 0.04614] |
| 902 | 0.05308 | [0.01583, 0.08164] |
| 903 | 0.05527 | [0.01485, 0.08728] |
| 904 | 0.04489 | [0.01413, 0.06890] |

Only a seed-900 plug-in noise sensitivity is available: low-noise interval
`[0.00596,0.02212]`; high-noise `[-0.00023,0.00049]`. This does not pass the
intended five-seed device-noise family gate. It motivates a prospective
hardware experiment; it does not show that the witness works on hardware.

### 3.2 ULB: ablation, OCC parity, and named QML rivals

Define **S1** as the analytic single-layer LCU/imaginary-time-inspired anomaly
score without a trained mixer; **S2-L1** as its fixed-depth, trained-mixer
version; and **OCC** as a detector fitted only on legitimate training rows.
On the unchanged-prevalence ULB holdout (56,962 rows, 99 frauds), the ablation
is hard-bit energy 0.264, soft quantile encoding 0.688, analytic filter 0.696,
and S2-L1 0.704–0.706. The dominant gain is classical encoding.

The fixed-L=1 trained-mixer increment independently replicates at
`+0.007242 [+0.001366,+0.013372]` AUPRC. A separately frozen control shows
trained minus frozen-random `+0.120265 [+0.084777,+0.153449]`, while
frozen-random minus matched `phi=0` is
`-0.106891 [-0.137744,-0.072477]`. All ten trained-minus-random point signs
are positive, but these row-bootstrap intervals condition on the ten fitted
seeds and do not measure seed-population uncertainty. The supported statement
is a trained fixed-L=1 protocol effect relative to the prespecified random
freeze—not a generic coherence, depth, architecture, or advantage claim.

The missing classical OCC gate changes the application conclusion: PCA
reconstruction reaches 0.7238 at eight features and IsolationForest 0.6967 at
12, matching or beating the standalone scorer. Classical supervised point
leaders are higher still.

The QML bracket uses the same split, features and 56,962 evaluation rows with
family-specific caps, but exact statevectors and no common runtime, shot,
overlap-query, or hardware currency. Stack reference S2-L1 is 0.705758.
Seven-comparison simultaneous intervals are shown where applicable:

| named implementation | AUPRC | stack minus arm | classical/ablation twin |
|---|---:|---:|---:|
| product-fidelity QSVC | 0.701587 | +0.004171 [-0.050248,+0.068656] | RBF-SVC 0.705807 |
| Deloitte/AWS-shaped hybrid QNN | 0.707566 | -0.001807 [-0.035944,+0.033507] | dense head 0.721783 |
| ring-IQP Nyström | 0.542580 | +0.163179 [+0.048194,+0.270152] | RBF-Nyström 0.730616 |
| projected ring-IQP SVC | 0.632009 | +0.073749 [-0.026860,+0.169265] | RBF-SVC 0.705807 |

The product-QSVC and hybrid-QNN differences are not resolved and neither meets
`±0.01` equivalence. Local 80%-power MDEs are 0.0748 and 0.0435 AUPRC,
respectively, conditional on the paired score distributions. Reduced-effort
VQC/QAE/autoencoder implementations are retained in the audit as named stress
tests but omitted here because their tuning, polarity, or architecture makes a
family conclusion unsafe.

The supervised-screened dominant-mode result is protocol-dependent but worth
reporting accurately: S1 versus retrained XGBoost is 0.944/0.228 for k=3,
0.945/0.413 for k=4, and 0.945/0.394 for k=6, with paired differences
`+0.7162 [+0.5973,+0.8122]`, `+0.5319 [+0.3829,+0.6736]`, and
`+0.5507 [+0.4152,+0.6863]`. Fraud labels from train+validation determine the
screen, centroids and dominant mode; minority modes mostly reverse; a
legitimate-only screen collapses. This is off-manifold sensitivity, not
label-free novel-fraud advantage. At 128 shots the mean AUPRC delta is
`-0.00483`, but the 30-draw range `[-0.04596,+0.03528]` leaves direction
unresolved.

### 3.3 IEEE-CIS Engine A: prominent FAIL and deployment constraint

The core split, routing, endpoints, gates and both outcome wordings were
frozen before sealed-test access. Feature/view and arm design used
train/validation data; six amendments were logged before unblinding. The
chronological split contains 590,540 labeled rows. A 144-fit XGBoost+LightGBM
backbone reached validation AUPRC 0.65195, then fell to 0.52939 on the sealed
test period.

> **ENGINE-A VERDICT: FAIL G1–G4.** Quantum minus backbone AUPRC is
> `-0.002298 [-0.004087,-0.000682]`. Quantum minus the strongest matched
> classical residual is `-0.001265 [-0.003144,+0.000476]`, unresolved—not an
> equivalence result. The keyed-noise arm is `-0.007168
> [-0.010463,-0.004060]`. No residual is deployed.

At review budget K=400, the frozen validation-band oracle headroom was only
0.8675 recall percentage points. A secondary rolling-origin validation
analysis is consistent with cross-region score interleaving, while both
locally frozen in-period splice contrasts were unresolved. The noise arm
excludes a quantum-specific explanation but does not separate finite-sample
overfit, reranking, boundary calibration, and temporal drift. It therefore
does not identify the cause of the sealed loss.

### 3.4 Conditions map

| axis / threshold | point winner | evidence and boundary |
|---|---|---|
| five total fraud labels, including screening/tuning | classical OCC | IsolationForest 0.587; XGB-full 0.546; S1 0.483; wide seed ranges |
| clean temporal ULB, later 20% | classical supervised | XGB-full 0.792; XGB-8 0.786; S2 0.753 |
| supervised-screened dominant mode, k=3/4/6 | S1, classically evaluable | 0.944/0.945/0.945; minority modes reverse |
| prevalence reweighted to 0.02%, no row subsampling | S1, classically evaluable | S1 0.498; XGB-8 0.488; IsolationForest 0.336 |
| bounded inward normalized shift, epsilon .05/.10/.20 | S1, classically evaluable | S1 .592/.580/.450 vs XGB .549/.365/.358; not an attack-feasibility claim |
| Sparkov supplied test | classical supervised | XGB 0.8721; IsolationForest 0.0640; S1 0.0062 |
| ULB↔Sparkov amount/time-only transfer | no useful transfer | best listed weak OCC 0.0537; feature coordinates do not transfer |
| IEEE-CIS temporal holdout | backbone / no residual | backbone 0.52939; Q residual -0.00230 |

**Conditions verdict:** classical coverage suffices everywhere measured.
Some rows favor S1, but S1 is exactly classically evaluable at the tested
depth. No measured condition requires a quantum processor.

## 4. Outputs, explainability, and expected impact

The deployable output is the classical backbone probability in `[0,1]`, plus
a validation-chosen binary decision. On the visible IEEE-CIS validation set,
the frozen backbone has Brier 0.02135 versus 0.03752 for the constant-
prevalence predictor and ten-bin equal-frequency ECE 0.01270; no post-hoc
recalibration was fitted. At the frozen auto-flag threshold 0.994177,
validation gives TP/FP/FN/TN = 223/14/4388/113483 (precision 0.9409, recall
0.0484, F1 0.0920). From the existing aggregate-only sealed result—without a
new test read—the corresponding row is 167/0/3897/114044 (precision 1.000,
recall 0.0411, F1 0.0789). Extreme imbalance makes the low recall explicit.

An artifacted validation illustration, selected without its fraud label,
shows the investigator-facing format. For row token `cb7b9f51cf78481f`, the
backbone probability is 0.9071 (review band). Mean syndrome failure
probabilities across five frozen circuit seeds are 0.347 for A–B, 0.136 for
B–C, and 0.547 for A–C; `P(any failure)=0.781`, and the most likely nonzero
pattern is A–C-only at probability 0.347. This is a deliberately extreme
model-selected example, not a representative-case or accuracy claim.

Expected Phase-2 impact is bounded and auditable: a Braket mechanism
demonstration; a no-harm monitoring/attribution rail; a public full-prevalence
evaluation template; and a conditions map telling HSBC when to stop. It is
not a claim that the quantum component improves the fraud score.

## 5. Phase-2 Braket work package and stop gates

1. **Partial-dephasing witness.** Implement each `D_q` as randomized physical
   I/Z blocks, assigning Z with probability `q/2`; simulator-only
   `phase_flip` instructions cannot be sent to a QPU. Freeze one device,
   native compile, topology, calibration, job randomization, data approval,
   familywise analysis and power before execution. The reduced base design is
   80,000 shots before phase-twirl, identical-branch and calibration controls.
2. **X-readout estimator.** Compare the exact failure/success X readout with
   optimally allocated parameter shift, SPSA and a Hadamard-test construction
   at matched state preparations and total shots. Report gradient cosine,
   bias, variance and shots-to-target; equal-shift allocation is not the
   strongest control.
3. **Flag geometry.** Replicate flag-NG versus Adam under calibrated noise,
   counting shots and state preparations rather than settings alone. The
   existing `+0.001882 [+0.000239,+0.004057]` five-seed result is a small
   simulator optimizer effect, not a hardware claim.
4. **Native-resource gate.** Compile against the selected Braket device's
   actual native gates and connectivity, then power and price the complete
   controlled experiment. Generic Qiskit basis counts do not clear this gate.
5. **Application gate remains closed.** Reopening requires a new externally
   timestamped protocol with temporal rehearsal, rank-preserving integration,
   calibrated probability output, and a no-harm gate. The IEEE-CIS sealed
   partition receives no further confirmatory read.

Every quantum experiment retains exact-path, full-dephasing, phase-twirl,
identical-branch and classical controls. Stop if reference circuits fail, the
predeclared contrast is underpowered, calibrated noise erases the signal, or
the classical twin matches the proposed operational benefit.

## 6. Deployment posture and data governance

The backbone ranking is never overwritten. Syndrome patterns run as an
asynchronous monitoring and attribution rail on an approved review subset;
any future influence on order requires the new application gate above.
Classical exact-path evaluation remains the reproducible fallback.

The dephasing source NPZ files contain transformed IEEE-CIS validation rows
and are `PRIVATE_LICENSED`. They are excluded from a public release, VM
handoff, or AWS transfer unless the competition licence and HSBC governance
approve that processing. Public artifacts contain aggregate JSON, theorem
code and synthetic referees only.

## 7. Team capability and submission status

Ha Cong Nguyen is the single-member team, lead/contact, model developer and
model-risk owner. The completed work demonstrates adversarial self-audit, OCC
parity, sealed-test discipline, unchanged-prevalence QML comparisons,
exact-versus-circuit referees, negative-result publication, and versioned
post-outcome correction. Independent model validation is still required
before any production influence. Affiliation, short biography and publication
list must be supplied in the submission form if that form requires them.

**Release record:** public repository
`https://github.com/hanoipho997/hsbc-phase1-r3`; immutable scientific-content
commit `0bb260f21814a2d2aed9c1b3b5cf53b0d6e87cf0`; DOI
`10.5281/zenodo.22650874`.

---

# Appendix A — Evidence and scope ledger

### A.1 Parameter and measurement ledger

| instance | raw / identifiable parameters | measurement statement |
|---|---:|---|
| ULB L=1 | 4 / 3 | source/new-seed AUPRC about 0.704/0.706 |
| original ULB L=2 | 8 / 6 | selected result 0.648; depth did not help |
| Engine-A cross-view L=3 | 18 / 15 | eight bins/seven independent syndrome values from one Z setting |

### A.2 Implementation-qualified rival stress tests

Our prespecified re-uploading VQC (AUPRC 0.00648), one-layer trash-occupation
surrogate (0.00103), kernel OCC (0.00451), and small hybrid latent autoencoder
(0.06435) performed poorly. These values do **not** bound their families: the
VQC lacked an architecture/optimizer/restart grid; the trash score was not the
canonical joint fidelity and had polarity risk; the kernel OCC missed its
native validation-FPR target; and the hybrid AE was capacity-limited. They are
reproducible stress tests, not headline competitors.

### A.3 Theory floor

- For the tested commuting core and product encodings, classical estimators
  match or improve the reported shot variance; this rules out a present
  sampling advantage at deployed width.
- For an unamplified diagonal filter,
  `p_s P(z*|success)=q_0(z*)g(E(z*))`; this is a raw-hit-yield identity, not a
  universal no-go theorem.
- The LCHS period/span rule is a necessary alias-free precheck. Falling below
  it permits aliasing but need not invert every finite set; exceeding it
  guarantees neither monotonicity nor approximation accuracy.
- The complete-dephasing theorem is scoped to the fresh-ancilla, unitary-
  branch, exact-insertion assumptions in §2.2. Its partial-dephasing curve is
  classically exact at current `L`.
- Flag-NG geometry is known; the surviving research question is matched-shot
  behavior under calibrated device noise.

### A.4 Generic compilation envelope

The exact 15-qubit operator was transpiled with Qiskit optimization level 3
over ten seeds. These are abstract basis/topology envelopes, not vendor-native
or equal-fidelity results:

| target abstraction | two-qubit gates |
|---|---:|
| all-to-all `{rx,ry,rz,rxx}` | 155 RXX |
| all-to-all `{cz,rz,sx,x}` | 238 CZ |
| synthetic full 4×5 square grid `{r,cz}` | 354–374 CZ |
| synthetic 15-qubit line `{cz,rz,sx,x}` | 486–513 CZ |

The synthetic grid is not an IQM Garnet topology. IonQ- or IQM-native
compilation, fidelity and latency remain Phase-2 gates. Sze et al. provide
useful trapped-ion multiplexer precedent, not validation of these counts.

### A.5 FTQC roadmap boundary

Repository modules for coherent polynomial optimization and linear solving
support a separate fault-tolerant research program. They do not turn this
fraud result into quantum advantage or parameter-space QNG. Any FTQC claim
requires end-to-end data/oracle construction, condition number, precision,
success amplification, non-tomographic readout, and a best-classical
comparison under the same access model. It is not a Phase-2 performance
promise.

# Appendix B — References

1. HSBC, *Quantum-Enhanced Credit Card Fraud Detection for Digital Payment
   Ecosystems*, 2026 Global Quantum + AI Challenge statement.
2. H. Nguyen, Engine-A protocol and report,
   `docs/hsbc_engine_a_preregistration_v1.md` and
   `docs/hsbc_engine_a_report_v1.md`, 2026.
3. H. Nguyen, adversarial audit v1 and revision-2 hostile audit/novelty
   dossier, `docs/hsbc_challenge_audit_report_v1.md` and
   `docs/hsbc_challenge_audit_report_v2.md`, 2026.
4. H. Nguyen, corrected novelty dossier and dephasing amendment,
   `docs/hsbc_challenge_claude_novelty_dossier_v2.md` and
   `docs/hsbc_challenge_dephasing_witness_analysis_amendment_v2.md`, 2026.
5. N. Khatri, S. Zohren and G. Matos, arXiv:2607.24686; arXiv:2506.22310
   (S-LCU architecture, trainability and simulation scaling).
6. A. Daskin, arXiv:2605.02986; P. Faehrmann, J. Eisert and R. Kueng,
   arXiv:2505.15913; J. Heredge et al., arXiv:2405.17388 (LCU outcomes and
   non-unitary QML).
7. V. Havlíček et al., *Nature* 567, 209–212 (2019); R. Kübler,
   M. Buchholz and B. Schölkopf, NeurIPS (2021); H.-Y. Huang et al.,
   *Nature Communications* 12, 2631 (2021) (quantum kernels).
8. A. El Alami, N. Innan, M. Shafique and M. Bennai, arXiv:2412.19441;
   N. Innan, M. A. Khan and M. Bennai, *International Journal of Quantum
   Information* 22(2), 2350044; A. Dinuț et al., *Electronics* 15, 2489
   (2026) (fraud-QML comparisons).
9. A. Sakhnenko et al., *Quantum Machine Intelligence* (2022); Deloitte and
   AWS, Braket hybrid-QNN fraud case study (2024).
10. J. Bowles, S. Ahmed and M. Schuld, controlled QML benchmarking; H. Sze et
    al., arXiv:2501.18515 (multiplexer compilation and trapped-ion experiment).
11. J. Stokes et al., *Quantum* 4, 269 (2020); B. Koczor and S. C. Benjamin,
    *Physical Review A* 106, 062416 (2022); D. Wierichs et al., *Quantum* 6,
    677 (2022).
12. Amazon Web Services, Amazon Braket device, native-gate and pricing
    documentation, accessed 2026-09-04.

Source lock: Engine-A implementation/evidence commit `5242b2d`; Revision 2
base commit `2fe355f`; sanitized Revision-3 scientific-content commit
`0bb260f21814a2d2aed9c1b3b5cf53b0d6e87cf0`; DOI
`10.5281/zenodo.22650874`.

# Appendix C — Reproducibility and release status

The committed Engine-A sealed result is read once and cited from
`runs/hsbc_challenge/engine_a_v1/unblind_real_v1.json`; “one read” applies
only to that IEEE-CIS sealed partition. ULB test rows have been reused across
the declared audits. Baseline and mechanism scripts are seeded, and isolated
reproductions preserve wall-time and cross-platform differences rather than
relaxing gates.

Revision-3 additions form a fail-closed sanitized release package. Its
scientific-content commit is frozen before the final metadata-only commit and
Zenodo archive. The output-completeness protocol and runner are
`docs/hsbc_challenge_revision3_output_protocol_v1.md` and
`scripts/hsbc_challenge/report_revision3_outputs_v1.py`; their aggregate
artifact is `runs/hsbc_challenge/revision3_v1/output_completeness_v1.json`.
No sealed row or score was reread. IEEE-CIS-derived row arrays remain private
under the competition licence.
