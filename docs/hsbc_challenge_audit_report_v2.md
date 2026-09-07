# HSBC Phase-1 proposal revision-2 hostile audit and novelty dossier

**Audit version:** v2  
**Audit date:** 2026-08-31 (Europe/Brussels)  
**Source proposal:** commit `2fe355fc860d8e209bd74faa20b090ed408d2134`  
**Mandate:** falsification is a success outcome; reproduce before criticizing  
**Final recommendation:** **NO-GO AS REVISION 2; GO FOR A REVISION 3 ONLY AFTER THE P0 REWEIGHTING AND CLAIM REPAIRS BELOW.**  
**IEEE-CIS rivals extension:** **NOT RUN — Q20 budget gate invoked.**

## Executive verdict

Revision 2 is substantially more honest than revision 1: it adopts the OCC
failure, removes quantum-advantage language, publishes the Engine-A all-gates
FAIL, and treats the classical backbone as the detector. Its load-bearing
Engine-A numbers match the one-read artifact. It is nevertheless not ready to
submit.

The central strategic error is weighting, not a request to reclaim a win. The
challenge's primary ask is a Braket-enabled model evaluated for measurable
gains; leading with an application experiment that harmed AUPRC, failed every
gate, and has no Braket execution invites a low score before a judge reaches
the strongest material. The proposal's own preregistered FAIL branch said to
lead with Engine B. Revision 3 should lead with the verified measurement and
hardware-metrology package, carry Engine A prominently as its constraint and
integrity evidence, and make no application-lift promise.

Several revisions are scientifically mandatory. The Markdown derivative
formula is false as printed; the 128-shot direction is unresolved rather than
a measured degradation; the F2 noise head is not a monotone recalibrator, so
the claimed causal isolation is too strong; “none [of the published rivals]
would survive” is an untested counterfactual; the claimed n=12/L=3 gate range
has no compile artifact; and the document-wide one-read statement is false for
ULB. Submission placeholders and a non-building theory note remain.

Two new hardening results sharpen the honest positive story:

1. The preregistered frozen-mixer control **shows that angle adaptation matters
   relative to the prespecified random-freeze distribution**. Mean-seed
   trained minus frozen-random AUPRC is `+0.120265`
   with Bonferroni-equivalent simultaneous 97.5% interval
   `[+0.084777,+0.153449]`. These row-bootstrap intervals are conditional on
   the ten fitted seeds; all ten trained-minus-random point signs are positive,
   but seed-population uncertainty is not included. Frozen-random minus matched `phi=0` is
   `-0.106891 [-0.137744,-0.072477]`: arbitrary nonzero mixers are often
   harmful. The original `+0.007242 [+0.001366,+0.013372]` survives as a
   **trained fixed-L=1 protocol effect**, not a generic “coherence helps” or
   depth claim.
2. Across seven primary named rivals, the strongest product-QSVC and hybrid-QNN
   differences remain unresolved after correct multiplicity handling, but are
   not equivalence results. Stack minus product-QSVC is
   `+0.004171 [-0.050248,+0.068656]`; stack minus hybrid-QNN is
   `-0.001807 [-0.035944,+0.033507]`. Neither 90% interval lies inside
   `±0.01`. With only 99 test frauds, the observed local 80%-power MDE is
   `0.074779` and `0.043497` AUPRC, respectively. Say **“not resolved,” not
   “ties” or “matches.”** Classical point leaders remain as good or better.

The novelty sweep also narrows the contribution. The failure-sector algebra
is latent in prior binary-LCU block formulas; the new-for-setting object is the
fresh-ancilla stacked readout. The dephased endpoint plus an architecture-
specific, per-sample partial-dephasing curve is the strongest novelty seed.
Flag-NG's geometry, the no-free-sampling identity, and LCHS spectral aliasing
are known. Broad “first preregistered QML” and “first full-prevalence fraud-QML
benchmark” claims are refuted. No measured regime in ULB, Sparkov, or Engine A
requires a quantum component.

## 0. Source lock, reproduction, and evidence boundary

### 0.1 Objects inspected

- `docs/hsbc_challenge_phase1_proposal.md`, `.tex`, and `.pdf` at `2fe355f`;
- `docs/hsbc_lcu_qng_theory_note.tex`;
- `docs/hsbc_challenge_audit_report_v1.md` and all v1 audit artifacts;
- `docs/hsbc_engine_a_preregistration_v1.md`, the Engine-A report, verification
  JSONs, routing freeze, heads, and `unblind_real_v1.json`;
- the frozen v2 rivals protocol, its three runners, score bundles, combined
  artifact, and results memo;
- the supplied 16-page HSBC challenge statement at
  `external/HSBC-Challenge-Statement-vFinalRevised.pdf`.

The handoff still contains the literal placeholder `[TO ADD: Ha's refs here]`.
That absence is recorded rather than silently treated as a completed seed
list. This audit used the papers explicitly named in Q5--Q17 and an independent
2019--2026 primary-source sweep.

### 0.2 Source hashes and repository state

| object | SHA-256 / state |
|---|---|
| proposal Markdown | `2d8344887d56d5a0a52b15ee0af8f8d50d33c5d82ec70b5985ff9693f0860c88` |
| proposal TeX | `6514aff5865fda89cb691402105833dd39de7f115c7ebd1cba1509416fc8c12b` |
| proposal PDF | `eee98447ef7775d324320a6ae3d7edbb45f38816bfff007b005061f21eedb690` |
| v1 audit | `8786c37f5b7d369aa1ce275235ff5465ef6c2b0114d0817e44373d81e2699c99` |
| theory note TeX | `ede03ef4ddfc9b89478517e50d4d4580b281375527e7f4687585da5aa5a61ae5` |
| supplied challenge PDF | `0c20c8079ae100442eb9ab6159958f514fb9eb0b135ef3ffe0ff7fe05202b523` |
| Engine-A preregistration | `0d9825231ca590baf7d9cb12b9d28958a7ac3b4eff42fac5f6af609f746102a8` |
| Engine-A one-read result | `8caaa07b282bbf7b4e0282748530233b56a6916b02f270ff1d8beba1e51215e0` |
| rivals protocol | `de10a4640e310986ea94ed52a35a7edba2bb32fceac7e84116dc2c0bc5ebc115` |
| rivals combined result | `8166d606d5635aa2fe81e9fdb2fb309cc16595afd2ddae5a1a5f0c9545b90ebb` |

The three revision-2 proposal sources have no scoped diff from `2fe355f`.
Engine-A and revision 2 are committed. In contrast,
`docs/hsbc_challenge_quantum_rivals_results_v2.md`, the v2 rivals protocol,
runners, and `runs/hsbc_challenge/audit_v2/` were untracked at audit start.
Thus “everything is committed” is true for revision 2's existing evidence, not
for the proposed revision-3 rivals fold-in. The shared worktree was preserved;
no reset, stash, clean, or broad staging was used.

### 0.3 Reproduction before criticism

| reproduction | result |
|---|---|
| Six `fit_v0` artifacts, isolated exact-script rerun | **MATCH** recursively after excluding `wall_seconds`, `rtol=1e-12`, `atol=1e-14`; no field differences. Durable comparison: `runs/hsbc_challenge/audit_report_v2/baseline_reproduction_v2.json`. |
| Cross-view verification, local locked environment | All non-timing fields exactly match the committed JSON. State/derivative residuals up to `2.08e-16`, failure identity `6.94e-17`, Qiskit blocks `4.11e-13`, dephased formula `1.11e-16`, zero data spread. |
| Syndrome verification, local locked environment | All non-timing fields exactly match. Circuit blocks `1.20e-13`, probabilities `4.12e-14`, derivative identity `8.58e-17`, dephased formula `1.67e-16`. |
| Linux VM mechanism smoke | All scientific assertions pass and syndrome non-timing fields match. Strict cross-platform comparison is **MISMATCH** for three cross-view residuals: block vector `5.23e-13` vs `4.11e-13`, probability `2.04e-13` vs `1.70e-13`, and finite-difference relative deviation `2.57e-9` vs `1.89e-9`. Tolerances were not relaxed. |
| Rivals finalizer | Fresh deterministic combination is byte-identical to the published combined JSON (`8166d606...`). |
| Engine-A endpoints | Checked directly against `unblind_real_v1.json`; the sealed test runner was **not** rerun. |

Timing fields changed by roughly a factor of two across fresh local runs and
are machine/load observations, not deterministic verification quantities.
That matters because revision 2 presents timings as if they belonged to the
new n=12 cross-view model; they do not.

The Engine-A one-read boundary was respected. Only that IEEE-CIS sealed test
partition has a one-read claim. ULB's test partition has been inspected and
reused by the fit, audit, and rivals studies.

### 0.4 Challenge-scoring boundary

The challenge statement's primary objective is to develop a quantum or
quantum-inspired fraud model and evaluate it against established classical
baselines; its executive summary asks for measurable gains. Its secondary
objectives value conditions of differential performance, robustness under
shift, explainability, and hardware-specific documentation. Full end-to-end
hardware execution is optional; component-level, stratified hardware tests are
explicitly acceptable. No page cap was found in the supplied statement; any
portal-specific rule remains unverified.

## Part A — Hostile review of revision 2

### Q1 — Is FAIL-as-centerpiece a winning strategy?

**Verdict: CORRECTED. Re-weight; do not re-claim.**

The argument for the current choice is real. Publishing an all-gates FAIL
demonstrates uncommon protocol integrity, answers the secondary conditions
question, prevents balanced-subsample theatre, and produces a no-harm
deployment lesson. It should remain prominent and verbatim.

The argument against it is decisive for this competition. A judge first sees
an application arm that failed, harmed AUPRC, is exactly classically tractable
at the tested scale, and has not run on Braket. That is weak against the primary
measurable-gain criterion. Worse, the Engine-A preregistration and final report
both froze the FAIL branch as “lead with Engine B ... and this null as evidence
of protocol integrity.” Calling Engine A “the centerpiece” in revision 2
contradicts that predetermined branch.

**Recommendation:** lead with the verified syndrome/dephasing/gradient and
matched-estimator hardware package. Rename §3.2 to **“Preregistered application
stress test and deployment constraint.”** Keep the FAIL table in the main
body, but make it evidence that Phase 2 has a bounded stop/go contract—not the
product headline.

The one sentence a rival should attack first is: “none would survive this
protocol.” Those external models were not rerun under this protocol. Replace
it with:

> The cited metrics arise from materially different, often balanced or
> reduced-feature protocols and are not evidence of unchanged-prevalence
> performance; our named-instance bracket below supplies the controlled
> comparison we actually ran.

### Q2 — Why should HSBC fund Phase 2?

**Verdict: CORRECTED.** As a production fraud detector or expected metric-lift
project, the evidence says **do not fund it**. The strongest honest funding
case is instead a capped hardware-metrology and model-risk package:

> Demonstrate the 15-qubit syndrome and failure-sector observables on Braket
> against exact-path, dephased, randomized-branch, and readout-reference
> controls; compare failure-sector, parameter-shift, and SPSA estimators at
> matched executions; replicate flag-NG versus Adam under calibrated noise;
> and stop if preregistered gradient-cosine, shots-to-target, or hardware-delta
> gates fail. The exact classical path twin remains the deployable monitoring
> and attribution rail and never overwrites the backbone.

That package is supported by the `1e-13`--`1e-16` verification chain and the
Engine-A no-harm lesson. It is not a promise of quantum advantage. Sze et al.
([arXiv:2501.18515](https://arxiv.org/abs/2501.18515)) is useful precedent for
ion-trap LCU multiplexor compilation and hardware overlap measurements, but it
does not validate this proposal's n=12/L=3 gate count or fraud performance.

### Q3 — Claim and P0 incorporation audit

The statement that every revision-1 P0 was incorporated is **REFUTED**.

| v1 P0 | revision-2 verdict | required repair |
|---|---|---|
| Demote central performance claim; add OCC parity | **CONFIRMED**, but misweighted | Classical coverage and no-advantage wording are present. Restore Engine-B-first ordering. |
| Remove leaky `0.984 vs 0.263` label-free headline | **CORRECTED only partially** | Clean points are present, but “excluded from all supervision” is false and minority reversals/CIs are absent from the main text. |
| Raw/effective parameter and protocol accounting | **REFUTED** | Print L=1 `4 raw/3 identifiable`, source/new-seed `0.704/0.706`; original L=2 `8 raw/6 identifiable`, `0.648`; Engine A `18 raw/15 identifiable`. |
| Correct one-class language | **CORRECTED only partially** | Use “supervised-screened, protocol-dependent dominant-mode/off-manifold sensitivity.” |
| Align theory and implementation notation | **REFUTED in Markdown; partial in TeX** | Put the relative phase in both `K` and `F`; state the raw-logit chain rule and gauge. |
| Narrow resources/deployment | **CORRECTED only partially** | Remove “hardware-cheap,” unsupported `300–700` gates, and cross-model microsecond inference claim. |
| Submission integrity | **REFUTED** | Fill identity/DOI/bio fields, cite all supporting commits, publish the new bracket, and make the theory note build. |

#### Numerically confirmed claims

- Engine A: backbone `0.5293927`; Q `0.5270923`; Q-minus-backbone
  `-0.0022984 [-0.0040865,-0.0006819]`; noise control
  `-0.0071678 [-0.0104627,-0.0040600]`; Q-minus-C-a
  `-0.0012648 [-0.0031442,+0.0004761]`. G1--G4 are all false.
- The K=400 frozen review-band oracle headroom is `0.8675` recall percentage
  points, so “<1 pp” is correct only with that band and budget attached.
- Mixer replication is `10/10` positive, mean `+0.007242` with paired 95%
  interval `[+0.001366,+0.013372]`; Q17 now establishes that random freezing
  does not reproduce it.
- Flag-NG's small n=8/L=1 result is `+0.001882
  [+0.000239,+0.004057]`, all five point signs positive.
- PCA OCC `0.723832` at n=8 and IsolationForest `0.6967` at n=12 confirm that
  classical OCC meets or beats the scorer.

#### Residual overclaims and underclaims

| proposal claim | verdict | correction |
|---|---|---|
| “AUPRC ... mandated” | **CORRECTED** | The challenge and ULB authors recommend it; they do not mandate it. |
| “Frozen before any data contact” | **REFUTED** | The core was frozen before IEEE-CIS contact; A1--A6 followed train/validation contact but preceded sealed-test unblinding. |
| `K=c^2U_V+s^2e^{iϕ}U_W`, `F=cs(U_W-U_V)`, `dK/dβ=2F` | **REFUTED as printed** | `F` must contain `e^{iϕ}U_W`. Engine A instead uses softmax logits, with `dK/dl_0=-sqrt(a_0a_1)F`. |
| “8 features at zero extra measurement cost” | **CORRECTED** | Eight bins, seven independent degrees of freedom, from one Z setting; fewer settings is not zero shot cost. |
| `1+L` versus 21 | **CORRECTED** | Correct for the declared mixing-angle/flag block at L=4, not the full Engine-A 18-raw-parameter model. |
| measured-output Fisher is the “only well-defined” noisy metric | **REFUTED** | It is directly measurable and noise-compatible; mixed-state QFI and other output-likelihood metrics are also defined. |
| `0.944` robustness | **Underclaimed numerically, overclaimed semantically** | Report deltas: k=3 `+0.7162 [+0.5973,+0.8122]`; k=4 `+0.5319 [+0.3829,+0.6736]`; k=6 `+0.5507 [+0.4152,+0.6863]`. Call it supervised-screened dominant-mode sensitivity, not drift or label-free novelty. |
| 128-shot “measurable degradation” | **REFUTED** | Mean delta `-0.00483`, but 30-draw 2.5/97.5% range `[-0.04596,+0.03528]`; dispersion is measurable, direction unresolved. |
| F2 harm “is the splice ... not the circuit” | **REFUTED causally** | The keyed-noise head perturbs rankings. It excludes quantum specificity but does not separate finite-sample noise overfit, reranking, boundary calibration, and drift. |
| F3 “across both datasets” | **REFUTED** | IEEE Q/C-a is unresolved; ULB's first cross-view branch scored `0.561`, `-0.134 [-0.192,-0.082]` versus S1, not a tie. |
| “hardware-cheap,” `300–700` two-qubit gates | **UNSUPPORTED** | The only artifact is n=8/L=2: 10 qubits, 280 abstract CZ, depth 695, two-qubit depth 242. No n=12/L=3 native compile exists. |
| `0.5–14 us/tx` for the proposal model | **REFUTED scope** | Those are old n=8 transformed-feature paths. Fresh cross-view timings are milliseconds and machine-sensitive; neither is end-to-end. |
| no test quantity read more than once | **REFUTED document-wide** | Narrow to the IEEE-CIS sealed test. |
| “first preregistered ... QML fraud” | **UNCLEAR** | Q9 rejects the broad first and does not clear the exact conjunction. |
| “balanced-subsample metrics do not survive” | **UNSUPPORTED family-wide** | Say the reported protocols are incomparable; print only this audit's named-instance measurements. |
| no claim depends on unpublished data | **UNSUPPORTED until release** | Local paths, an empty DOI, untracked rival assets, and IEEE-CIS licensing remain. |

F2 needs an especially explicit repair. The fitted noise head is logistic on
the backbone logit plus eight keyed-noise variables, not a monotone transform
of the backbone alone. Its standardized coefficients are backbone `1.0712`
and noise `[-0.2824,0.0227,0.0482,-0.1196,0.1763,-0.0498,0.3169,-0.2620]`;
the noise block has L2 norm `0.5472`. Proposal-safe wording is:

> On the chronological holdout, keyed-noise residual replacement caused the
> largest resolved loss, `-0.00717 [-0.01046,-0.00406]` AUPRC. This excludes a
> quantum-specific explanation but does not yet separate noise-head overfit,
> within-band reranking, cross-boundary score mismatch, and temporal drift.

These structural numbers are reproduced without loading any IEEE-CIS row by
`scripts/hsbc_challenge/diagnose_engine_a_noise_head_v2.py`; its immutable
output is
`runs/hsbc_challenge/audit_report_v2/engine_a_noise_head_diagnostic_v2.json`.

### Q4 — Page, format, and standalone coherence

**Mechanical PDF: CONFIRMED with caveats. Submission readiness: REFUTED.**

The committed PDF is five A4 pages at 10 pt: three main pages and two appendix
pages. All five pages were rendered and visually inspected. There is no
clipping, overlap, or unreadable glyph. A clean two-pass build preserves the
extracted text. One bibliography path produces a 7.75 pt overfull box.

No Phase-1 page limit was found in the supplied challenge statement. The
proposal's own appendix-maximum rule is met, subject to an unverified portal
rule. The standalone document nevertheless fails:

- team name, affiliation, contact, member bios, publications, and DOI remain
  `[TO FILL]`;
- the PDF's visible appendix headings skip “Appendix B”;
- page 5 is roughly half blank while the promised conditions map is absent;
- S1, S2, OCC parity, and the “audit winners” are not defined for a fresh
  judge;
- outputs promise calibrated probabilities, binary decisions, SHAP, and
  per-transaction explanation, but no Brier/calibration table, confusion
  result, or representative explanation artifact is shown;
- Braket is only future work, so current Braket-use scoring is low;
- Appendix C omits commit `5242b2d`, which carries ULB audit evidence;
- Markdown and TeX disagree on the phase in `F`;
- `pdflatex` on `docs/hsbc_lcu_qng_theory_note.tex` fails in the locked local
  environment at missing `cleveref.sty`.

Use the spare page for a compact conditions/evidence table and one concrete
explanation example rather than more narrative.

## Part B — Novelty dossier, 2019--2026

Novelty labels are claim-level, not judgments of value. Absence searches do
not prove priority; “first” is withheld unless a public, time-stamped record
supports it.

### Q5 — Failure-sector equals derivative

**Verdict: NOVEL-FOR-SETTING. The algebra itself is not novel.**

Closest works:

1. Akhalwaya et al., [A Modular Engine for Quantum Monte Carlo Integration](https://arxiv.org/abs/2308.06081),
   Eq. 5.17, prints the same binary PREP--SELECT--PREP-dagger success and
   failure blocks; the derivative follows immediately after a coordinate
   change.
2. Daskin, [Exploiting all ancilla outcomes in linear combinations of unitaries](https://arxiv.org/abs/2605.02986),
   reuses structured LCU outcomes, but not as derivatives.
3. Heredge et al., [Non-Unitary Quantum Machine Learning](https://arxiv.org/abs/2405.17388),
   studies parameterized LCU QML and success probabilities, but discards the
   failure sectors.

Faehrmann--Eisert--Kueng recycle work-register information via shadows;
Abbas et al. establish general backpropagation scaling; Bowles--Wierichs--Park,
shadow-gradient work, and general LCU derivative circuits reduce or reorganize
gradient measurements. None found uses this exact fresh-ancilla,
single-failure stacked readout. The identity remains elementary and latent in
Akhalwaya's block formula.

**Safe non-priority claim:**

> In our fresh-ancilla binary stacked-LCU construction, a single-failure
> branch is one half of the all-success-stack derivative, and its interference
> with the success branch supplies an exact in-circuit X-basis gradient
> readout.

Never say “new derivative identity.” Preserve the relative phase, coordinate
factor, raw-logit chain rule, and common-logit gauge. Z-basis failure records
are not gradients, and L incompatible layer settings remain.

### Q6 — Dephased syndrome and partial-dephasing witness

**Exact endpoint: NOVEL-FOR-SETTING. Generic dephasing sweep: KNOWN. The
architecture-specific per-sample information curve: NOVEL-FOR-SETTING.**

Closest works:

1. Liao et al., [Decohering Tensor Network Quantum Machine Learning Models](https://arxiv.org/abs/2209.01195),
   sweep continuous dephasing and connect the fully dephased model to a
   classical Bayesian network.
2. Miller, Roeder, and Bradley, [Probabilistic Graphical Models and Tensor Networks](https://arxiv.org/abs/2106.15666),
   treat fully and partially decohered Born/tensor-network models.
3. Daskin's all-outcomes LCU paper is the nearest outcome-distribution prior,
   but has no data-independent endpoint.

No located work proves for this ancilla placement that complete dephasing
between SELECT and uncomputation makes the full pattern law the input-
independent product `prod_j Bernoulli(2 c_j^2 s_j^2)`. Local verification is
`1.11e-16`/`1.67e-16` with zero data spread.

The code and numerical referees currently contain the derivation, but the
companion theory note does not yet state a theorem with the required fresh-
ancilla, unitary-branch, and insertion-point assumptions. Therefore the
following is **proposal-safe only after that formal proof is added**:

> For fresh-ancilla stacked binary LCU, complete ancilla-path
> dephasing between SELECT and uncomputation makes the pass/fail-pattern law
> an input-independent Bernoulli product, and use the calibrated partial-
> dephasing response of `P_q(a|x)` as a per-sample witness of information
> carried by path interference.

This witnesses interference relative to a declared channel/insertion point,
not advantage, contextuality, classical hardness, or entanglement.

### Q7 — Flag-NG and setting economics

**Verdict: KNOWN — cite.**

Meyer reviews Fisher information in NISQ applications
([arXiv:2103.15191](https://arxiv.org/abs/2103.15191)); Abbas et al. use the
Fisher of measurement-derived QNN likelihoods for effective dimension
([arXiv:2011.00027](https://arxiv.org/abs/2011.00027)); Koczor and Benjamin
generalize QNG to noisy/nonunitary circuits
([arXiv:1912.08660](https://arxiv.org/abs/1912.08660)). Crowley et al., PRA 89,
023845 (2014), already give the orthogonal-block QFI direct sum
`F_Q[oplus_i p_i rho_i] = F_C(p) + sum_i p_i F_Q(rho_i)`, from which the
proposal's coarse-grained flag formula follows up to convention.

No novelty sentence is defensible. Use an implementation statement:

> For the stacked-LCU coefficient block, we instantiate the standard Bernoulli
> natural gradient using one Z setting and one success/failure-interference
> setting per layer, versus `1+2L+2*C(L,2)` settings for our specified
> ungrouped conditional-state reconstruction—5 versus 21 at L=4.

Call `G*` the metric of the deliberately coarse-grained flagged model, not the
QFI of the complete physical output. Settings are not shots or wall time.

### Q8 — No-free-sampling and LCHS aliasing

**No-free-sampling verdict: KNOWN / rejection-sampling identity.** Closest
works are Ozols--Roetteler--Roland, [Quantum Rejection Sampling](https://arxiv.org/abs/1103.2774),
probabilistic imaginary-time filtering
([arXiv:2111.12471](https://arxiv.org/abs/2111.12471)), and amplitude-amplified
PITE ([arXiv:2212.13816](https://arxiv.org/abs/2212.13816)). The equation is
the joint-probability identity
`P(S,z*)=P(S)P(z*|S)=q0(z*)g(E(z*))`.

Safe non-novel statement:

> For a contractive diagonal filter, conditional concentration cannot improve
> unamplified raw hit yield because
> `p_s P(z*|S)=q0(z*)g(E(z*)) <= q0(z*)`.

The proposal's “only escapes” claim is false. A warm-start changes `q0`;
amplitude estimation changes the task/cost measure; target sets follow by
summation; the result does not constrain score estimation.

**LCHS alias verdict: KNOWN; at most NOVEL-FOR-SETTING as a fraud-ranking
diagnostic.** An--Liu--Lin introduce LCHS
([arXiv:2303.01029](https://arxiv.org/abs/2303.01029)); Wang et al. explicitly
interpret discretization as spectral folding via Poisson summation
([arXiv:2604.02874](https://arxiv.org/abs/2604.02874)); Aftab--An--Trivisa
analyze LCHS quadrature/cost
([arXiv:2606.11475](https://arxiv.org/abs/2606.11475)). For a uniform grid,
the magnitude-squared score period `pi(M-1)/K` is an elementary Fourier-grid
corollary.

> Requiring that period to exceed a declared continuous `tau E` interval is a
> necessary, not sufficient, alias-free ranking precheck.

Below threshold, a finite observed set need not invert; above it, monotonicity
and approximation accuracy still do not follow. Keep an independent error
gate.

### Q9 — Preregistration novelty

**Broad “first preregistered QML”: KNOWN / REFUTED. Exact fraud + sealed-test +
noise-control conjunction: UNCLEAR, at most NOVEL-FOR-SETTING.**

Lockwood and Si's negative hybrid quantum-classical Atari study was presented
at the NeurIPS 2020 Preregistration Workshop and published in PMLR 148
([paper](https://proceedings.mlr.press/v148/lockwood21a.html)). Bowles, Ahmed,
and Schuld benchmark 12 QML models over 160 datasets with classical and
disentangled controls
([arXiv:2403.07059](https://arxiv.org/abs/2403.07059)), but not under a
preregistered fraud protocol.

The search was logged on 2026-08-31 rather than inferred from memory:

| surface | exact query families | result / boundary |
|---|---|---|
| OSF/OSF-indexed web and Registered Reports indexes | `"quantum machine learning" preregistration`, `quantum machine learning fraud preregistered`, `"registered report" "quantum machine learning"` | no public exact-conjunction registration located; this is a search null, not proof, because indexing can miss embargoed or poorly tagged records; [OSF defines a preregistration as a time-stamped read-only pre-analysis plan](https://help.osf.io/article/330-welcome-to-registrations) |
| NeurIPS 2020/2021 Preregistration Workshop and PMLR 148/181 | `quantum`, `hybrid quantum-classical`, `QML` | Lockwood--Si is a positive prior hit and refutes broad priority; [the workshop record](https://preregister.science/) confirms the preregistration workflow |
| arXiv/journal QML benchmarking and fraud intersections | `preregistered QML benchmark`, `sealed test quantum machine learning`, `full prevalence quantum fraud benchmark` | Bowles--Ahmed--Schuld is the closest broad controlled benchmark but not preregistered; Dinuț et al. is the closest full-prevalence fraud-QML bracket but not located as preregistered |

The local Git chain is not an externally time-stamped registration. Commit
`953bebf` precedes `a4b0825` by local commit time, but the branch has no
upstream and no cached remote contains either commit. Unless a pre-outcome
public push or third-party archive is produced, use:

> The repository records endpoints, controls, gates, and both outcome
> wordings as frozen in commit `953bebf` before the single chronological test
> read; the resulting FAIL was then published without outcome-driven retuning.

Do not call this a Registered Report or claim first priority. Archive the full
chain and search log on OSF/Zenodo now; that improves provenance but cannot
retroactively create an external preregistration.

### Q10 — F2 calibration/routing failure

**General phenomenon: KNOWN. Exact observation: NOVEL-FOR-SETTING. Causal
isolation: UNCLEAR.**

Pampari and Ermon show calibration can fail under covariate shift
([arXiv:2006.16405](https://arxiv.org/abs/2006.16405)); calibrated mixture-of-
experts work documents routing/aggregate-calibration failures under shift
([arXiv:2606.20544](https://arxiv.org/abs/2606.20544)); drift-aware fraud work
is established by Mai et al.
([arXiv:2109.14155](https://arxiv.org/abs/2109.14155)).

The precise Engine-A result is worth a short setting-specific note only after
a rolling-origin, train/validation-only causal ablation compares:

1. identity/backbone;
2. backbone-logit-only calibration;
3. keyed-noise residual replacement;
4. rank-preserving within-band blending; and
5. a boundary-safe mapping.

Do not reread the sealed test. Until that study, print the corrected F2
sentence from Q3 and no stronger causal claim.

## Part C — Cheap, honest novelty additions

### Q11/Q17 — Ranking by novelty x cheapness x challenge fit

Scores are ordinal `1..5` and include submission-information novelty, not only
theorem novelty.

| rank | addition | novelty label | score | matched control / gate |
|---:|---|---|---:|---|
| 1 | Partial-dephasing interference-witness curve on Braket | NOVEL-FOR-SETTING | `4x3x5=60` | exact path, analytic endpoint, randomized branch, readout references |
| 2 | Frozen-mixer control on our circuit | known control; mandatory evidence | `2x5x5=50` | trained, frozen-random, matched `phi=0`, S1; **completed here** |
| 3 | Causal splice/boundary-safe ablation | NOVEL-FOR-SETTING | `2x5x5=50` | identity, logit-only, keyed noise, rank-preserving mapping |
| 4 | Headroom-bound audit tool | known/elementary governance | `2x5x4=40` | global oracle and policy-constrained in-band oracle |
| 5 | Syndrome-distribution drift detector | NOVEL-FOR-SETTING | `3x3x4=36` | raw-feature MMD/C2ST, backbone-score drift, exact/dephased syndrome |
| 6 | Shot-adaptive review-band ranking | known allocation idea; setting-specific objective | `2x3x4=24` | uniform shots, static variance allocation, exact-score oracle |
| 7 | Rivals bracket on IEEE-CIS Engine-A views | NOVEL-FOR-SETTING replication | `2x2x4=16` | full requested classical/twin bracket |

The frozen-mixer control outranked its novelty score operationally because it
governed the only positive model effect. It is complete; see Q17 below.

### Top experiment 1 — partial-dephasing hardware witness

Define per-ancilla
`D_q(rho)=(1-q)rho+q Delta_Z(rho)`, implemented by a stochastic Z with
probability `q/2`, so `q=1` fully kills off-diagonal coherence. Freeze
`q in {0,0.25,0.5,0.75,1}` before data-bearing execution.

Protocol:

1. Select 24--32 IEEE-CIS **validation-only** rows before outcome inspection,
   stratified by time and backbone score; disclose any label stratification.
2. Run the exact simulator first, then a Braket DM1/device model, then one
   randomized QPU batch with identical mapping and randomized q order.
3. Allocate 2,000--4,000 shots per row/q in independent job blocks.
4. Primary statistic is split-sample or shot-noise-subtracted between-row
   diversity
   `W(q)=2/[m(m-1)] sum_{i<j} ||P_q(a|x_i)-P_q(a|x_j)||_2^2`.
5. Secondary outcomes are area under `W(q)` and explicitly exploratory task
   informativeness.

Kill if simulator q=1 deviation exceeds `1e-12`; simultaneous intervals do not
separate `W(0)` from `W(1)`; the QPU curve exits its preregistered device-noise
band; or reference circuits explain the apparent data dependence.

Claim if it survives:

> We operationalize the dephased-syndrome theorem as a tunable hardware
> interference witness: between-sample information in the ancilla-pattern law
> is tracked continuously to its provably data-independent dephased endpoint.

### Q12 — OCC-parity-style control for the top addition

The exact `2^L` path twin must reproduce the curve and is cheaper at current
scale. The quantum circuit is the natural implementation only of the **physical
dephasing intervention**, not of a computationally hard statistic. The addition
passes parity as hardware mechanism validation and fails it as a fraud-detection
or runtime-advantage claim. If the classical or randomized-branch controls
explain the curve, retain no proposal addition.

For the drift-detector idea, the mandatory controls are raw-feature MMD/C2ST
and backbone-score drift at a fixed false-alarm average run length. If either
ties or wins on alarm delay, the syndrome monitor remains only an
interpretability rail. For shot allocation, compare against uniform, static
variance-optimal allocation, and the exact-score oracle; kill unless it reduces
shots at a fixed top-K ranking-error guarantee.

## Part D — Quantum-rivals bracket: hardening and revision-3 fold-in

### Q13 — Strawman defense and tuning ledger

**Overall verdict: CORRECTED; REFUTED AS A LITERAL MATCHED-RESOURCE
TOURNAMENT.** Product-QSVC and the hybrid QNN are useful, competent named-
instance comparisons. The catastrophic VQC, trash-QAE, kernel-OCC, and
hybrid-QAE values do not bound their method families. Revision 3 must say
**“our prespecified implementation of X”** and must not turn those scores into
a family-ranking claim.

The protocol used exact statevectors and no finite hardware shots. It shares
features, splits, and evaluation rows and freezes family-specific training and
circuit-forward caps, but it does **not** impose a common runtime, state-
preparation, shot, overlap-query, or hardware-execution currency. It is a
bounded named-instance predictive-quality bracket, not the literal matched-
budget tournament requested in the work order. Revision 3 must not call it
“matched budget” unless a common resource ledger and cap are prospectively
enforced.

| named arm | frozen search / selection | budget, seeds, stopping | nearest published configuration and delta | strawman ruling |
|---|---|---|---|---|
| Product-fidelity QSVC | exact analytic `prod_j cos^2(lambda*pi*(u_j-v_j)/2)`; `lambda={.125,.25,.5,1,2}` x `C={.1,1,10}`, 15 exhaustive candidates; full-validation AUPRC | 2,000 legitimate + all 295 fraud train rows; no stochastic model seed or early stopping; selected `lambda=.5,C=.1`, 436 support vectors/overlaps per transaction if implemented literally | [Kübler--Buchholz--Schölkopf (NeurIPS 2021)](https://papers.nips.cc/paper/2021/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html) product rotation kernel and [Schuld--Killoran (PRL 2019)](https://doi.org/10.1103/PhysRevLett.122.040504) fidelity-kernel SVM framework; classically tractable by construction | **Defensible named control.** It reaches `0.701587`; the difference from stack is unresolved. It is not advantage evidence. |
| Ring-IQP fidelity Nyström | six maps: `depth={1,2}` x `lambda={.5,1,2}` at `C=1`, then `C={.1,1,10}` | seed `20260831` selects 64 class-stratified landmarks; selected `d=2,lambda=.5,C=.1`; 355,536 map-selection and 56,962 test statevectors; 64 overlaps/transaction/shot | [Havlíček et al. (Nature 2019)](https://doi.org/10.1038/s41586-019-0980-2)-inspired two-block IQP map, but nearest-neighbor ring rather than all-to-all, 64-landmark logistic head rather than full Gram SVM, and locally linear scaling of both one-/two-body phases | **Named instance only.** Stack is multiplicity-resolved higher for this instance; the RBF-Nyström twin is higher still. |
| Projected ring-IQP SVC | same six maps at `C=1,gamma=scale`, then `C={.1,1,10}`; 96 features (24 singles + 72 adjacent-pair Pauli expectations) | selected `d=2,lambda=2,C=10`, 664 support vectors; naive 96 observable circuits/transaction/shot; no stochastic model seed or early stopping | [Huang et al. (Nature Communications 2021)](https://doi.org/10.1038/s41467-021-22539-9) local-observable projected-kernel construction, extended here with adjacent-pair observables | **Defensible named projected-kernel baseline.** Stack-minus-rival `+0.07375` remains unresolved after seven-way correction. No geometric-difference or hardness claim. |
| Legit-only ring-IQP Nyström OC-SVM | six maps at fixed `nu=.001727`, then `nu={.0005,.001,.001727,.003,.005}` | 2,000 legitimate rows, 64 legitimate landmarks; selected `d=2,lambda=.5,nu=.005`, 181 support vectors; no fraud labels | Kyriienko--Magnusson [arXiv:2208.01203](https://arxiv.org/abs/2208.01203) use a fuller fidelity Gram, all-to-all repeated IQP map, `nu=.1`, and enriched ULB subset; [Kölle et al. (ICAART 2024)](https://doi.org/10.5220/0012381200003636) document approximation instability | **Failed reduced instance.** Native legitimate-validation FPR is `0.32172`, not `0.001`; a later legitimate quantile validly calibrates the reported threshold but does not rescue native boundary selection. Inputs also came from a supervised feature screen. Do not use `0.004506` as family evidence. |
| Re-uploading VQC | one custom eight-angle, two-upload topology and one `q0` probability readout; no architecture/depth/optimizer/readout/restart grid | seeds `820..824`; SPSA, nominal 300 updates, all stopped at 120 and selected update 20; 561,688 forwards/seed, far below the six-million cap | [Dinuț--Constantinescu--Alexandrescu (Electronics 2026)](https://doi.org/10.3390/electronics15112489) use a 32-parameter, reps-3 EfficientSU2 and about 400 COBYLA iterations on full-prevalence ULB; [El Alami--Innan--Shafique--Bennai](https://arxiv.org/abs/2412.19441) sweep maps, ansätze, depth, and optimizers on balanced ULB | **High strawman risk; our implementation only.** A competent implementation could plausibly do much better at the same forward budget. Official AP stays `0.006482`; posthoc reversed AP `0.064799` only diagnoses score-direction sensitivity. Omit from the main bracket unless rerun. |
| Trash-occupation QAE surrogate | one eight-angle RY layer with a sequential directed CNOT ring; mean occupation of four trash qubits used as loss and score; no depth/entangler/optimizer/fidelity/restart grid | seeds `840..844`; 300 SPSA updates, all selected update 300; 1,140,307 forwards/seed | [Ngairangbam--Spannowsky--Takeuchi (PRD 2022)](https://doi.org/10.1103/PhysRevD.105.095004) use joint trash/reference fidelity, all-pairs CNOTs, 50 epochs, 5,000 shots, and a classical-AE twin; [Huot et al. (IEEE Access 2024)](https://doi.org/10.1109/ACCESS.2024.3496901) report one fraud-QAE layer inadequate while deeper circuits improve | **Decisive strawman/polarity risk.** Official AP `0.001025`; posthoc reversed AP `0.424031`. Mean marginal occupation is not canonical joint trash-register fidelity, and no standalone classical-AE twin was run. Relegate as “our one-layer trash-occupation surrogate” unless rerun. |
| Deloitte/AWS-shaped hybrid QNN | exact public dense `8->32->9` front end and three-qubit seven-weight head, adapted to eight frozen inputs; Adam with dense LR `.001`, quantum LR `.01` | seeds `830..834`; best updates `160,160,180,300,180`; 4.50--5.19 million forwards/seed; mandatory dense and frozen-quantum-head twins | [Deloitte/AWS public architecture](https://aws.amazon.com/blogs/machine-learning/how-deloitte-italy-built-a-digital-payments-fraud-detection-solution-using-quantum-machine-learning-and-amazon-braket/); this is an architecture adaptation, not a reproduction of its reported data result | **Competent and strongest fidelity match.** AP `0.707566`; stack difference unresolved. Dense head `0.721783` is higher under the original four-family interval, and the trainable head does not beat its frozen twin. |
| Hybrid quantum-latent AE | tanh `8->4`, one RY/ring four-angle PQC, four local-Z latents, linear `4->8` decoder + 300-tree IF | seeds `850..854`; all best/final at update 300; 3,095,857 forwards/seed | [Sakhnenko et al. (Quantum Machine Intelligence 2022)](https://doi.org/10.1007/s42484-022-00075-z) use `input->56->4`, a mirrored decoder, and a sweep of more than 30 PQCs | **Our small fixed bottleneck only.** AP `0.064345` versus width-8 classical `0.524426` shows that capacity/bottleneck choice dominates. Omit or restore the published-scale encoder/decoder and a parameter-matched twin. |

The polarity values above are a **post-test diagnostic**, not alternate primary
results. They regenerate from
`scripts/hsbc_challenge/diagnose_rival_score_polarity_v2.py` and
`runs/hsbc_challenge/audit_report_v2/rival_score_polarity_diagnostic_v2.json`.
They show why a catastrophic prespecified score must be downgraded, not
silently flipped or replaced.

Two frozen-protocol errata are material: §3.4 says “alternating CNOT
brickwork,” while its clarification and JSON implement a sequential directed
ring; and arXiv:2412.19441 is by El Alami, Innan, Shafique, and Bennai, not
Karimi et al. The actual Karimi et al. 2023 work is a DQC1 normalized-trace
kernel paper, not a fraud VQC. Innan--Khan--Bennai is *International Journal
of Quantum Information* **22(2)**, 2350044, not volume 21. These are recorded
as post-freeze corrections rather than silently changing the protocol.

Required proposal qualifier:

> On the frozen ULB split, stacked-LCU's differences from the product-fidelity
> QSVC and Deloitte/AWS-shaped hybrid QNN were not statistically resolved;
> reduced-effort VQC/QAE/HAE implementations are reported separately as named
> stress tests and do not bound those model families.

### Q14 — Statistics of the strongest unresolved comparisons

**Verdict: CORRECTED.** The original 98.75% intervals were a conservative
equivalent for four family headings, not a seven-rival simultaneous analysis.
The hardened artifact uses 10,000 paired Poisson(1) full-test-row draws,
Bonferroni 99.2857% percentile intervals across seven declared stack-versus-
rival comparisons, and Holm-adjusted centered-bootstrap tests. The hybrid
latent AE is treated as a secondary QAE implementation, not an eighth primary.

| comparison (stack minus rival) | point | seven-comparison simultaneous interval | local 80%-power MDE | ruling |
|---|---:|---:|---:|---|
| product-QSVC | `+0.004171` | `[-0.050248,+0.068656]` | `0.074779` | difference not resolved; not equivalent within `±0.01` |
| hybrid QNN | `-0.001807` | `[-0.035944,+0.033507]` | `0.043497` | difference not resolved; not equivalent within `±0.01` |
| ring-IQP Nyström | `+0.163179` | `[+0.048194,+0.270152]` | `0.150497` | stack higher for this instance |
| projected-IQP SVC | `+0.073749` | `[-0.026860,+0.169265]` | `0.131432` | difference not resolved |
| kernel OC-SVM | `+0.691182` | `[+0.559166,+0.813644]` | `0.167460` | stack higher for failed named implementation |
| VQC | `+0.699277` | `[+0.566552,+0.817211]` | `0.166432` | stack higher for our implementation |
| trash-QAE | `+0.694664` | `[+0.562183,+0.818161]` | `0.167961` | stack higher for our implementation |

An AUPRC MDE is not determined by “99 frauds” alone; it depends on the paired
rank distributions. The printed MDE is the local normal approximation
`(z_(1-alpha/(2m))+z_.8)*paired_bootstrap_SE`, conditional on these scores.
It is a power diagnostic, not a universal sample-size theorem.

**Exact proposal sentence:**

> On the unchanged-prevalence 56,962-row ULB holdout (99 frauds), stacked-LCU
> minus product-QSVC was `+0.004171` with seven-comparison simultaneous
> interval `[-0.050248,+0.068656]`, and stacked-LCU minus hybrid-QNN was
> `-0.001807 [-0.035944,+0.033507]`; neither difference was resolved, neither
> met `±0.01` equivalence, and competent classical point estimates remained
> as good or better.

Artifact:
`runs/hsbc_challenge/audit_report_v2/quantum_rivals_stats_hardened_v2.json`,
generated by `scripts/hsbc_challenge/harden_quantum_rivals_stats_v2.py`.

### Q15 — Revision-3 fold-in diff

This is a drafting diff, not authorization to overwrite revision 2.

1. **Title/thesis:** keep claim 1 as a verified mechanism package after
   deleting “hardware-cheap.” Replace thesis claim 2 with:

   > Under a prospectively frozen, unchanged-prevalence ULB bracket, the
   > scorer's difference from the strongest product-QSVC and hybrid-QNN
   > implementations was not statistically resolved; no equivalence was
   > established, and classical point estimates were as good or better.

2. **Opening evidence order:** replace “Engine-A ... centerpiece” by a
   mechanism-first paragraph, followed immediately by a boxed Engine-A FAIL
   and the deployment constraint. Preserve all negative endpoints.
3. **Add §3.1b, “Bounded named QML rivals at unchanged prevalence.”** Print a compact
   main table with product-QSVC, hybrid-QNN, ring-IQP, and their available
   classical/ablation twins. Move VQC/QAE/HAE and failed kernel-OCC stress
   tests to an implementation-qualified appendix table unless they are
   competently rerun. Use the Q14 sentence verbatim.
4. **Upgrade the balanced-subsample caveat from assertion to measurement:**

   > Prior fraud-QML metrics use materially different protocols. In our
   > unchanged-prevalence named-instance bracket, product-QSVC and hybrid-QNN
   > were unresolved from stacked-LCU, several shallow implementations failed,
   > and classical point leaders remained as good or better.

5. **Update the mixer line:** retain the replicated `+0.007242` result and add
   the frozen control: trained-minus-frozen-random `+0.120265
   [+0.084777,+0.153449]`; arbitrary frozen mixers were harmful versus
   matched `phi=0` under the prespecified random distribution. Call it a
   trained fixed-L=1 protocol effect and disclose that intervals condition on
   the ten fitted seeds.
6. **Correct F2, the derivative display, parameter identifiability, finite-shot
   wording, and resource/timing scope** exactly as Part A specifies.
7. **Use page 5:** insert a six-row conditions map and one explanation example.
   Trim repeated Engine-A prose, the known Q7/Q8 derivations, and duplicative
   integrity claims. Keep one link/QR to the full audit.
8. **References:** correct arXiv:2412.19441's authors; add Havlicek-style
   kernels, the kernel-OCC source, Bowles et al. benchmarking, Lockwood--Si,
   Akhalwaya, Liao, Dinuț et al. 2026, and Sze et al. 2025. Do not claim first.
9. **FTQC and “classical struggles” separation:** revision 2 contains no FTQC
   promise, which is good, but its `0.944` “excluded from all supervision” and
   “drift-conditional one-class” framing still implies a classical-struggle
   regime that the clean protocol does not establish. Apply Part A's
   supervised-screened dominant-mode wording. If `fqe.py`, `poly_opt.py`, or
   `linear_solve.py` is mentioned, put it in a separately scoped roadmap:
   no Phase-1 fraud evidence, no parameter-QNG identification, and no advantage
   claim without oracle construction, precision/condition/success costs, a
   non-tomographic readout task, and a best-classical comparison.

Recommended pivot paragraph:

> We propose a compact, circuit-realizable coherent anomaly and syndrome
> mechanism, then subject it to unchanged-prevalence classical and QML twins.
> The fixed-L=1 mixer increment replicates, and angle adaptation beats the
> prespecified random-freeze distribution under a new control. The scorer is
> not resolved from our strongest product
> quantum kernel or hybrid QNN, while competent classical models remain as good
> or better; we claim no quantum advantage. Phase 2 is therefore a capped
> Braket mechanism-validation study with exact classical twins and stop gates,
> not a promise to improve the production fraud backbone.

### Q16 — Is the bracket itself first?

**Broad first full-prevalence controlled fraud-QML benchmark: REFUTED. Exact
preregistered seven-family conjunction: UNCLEAR / at most NOVEL-FOR-SETTING.**

Closest studies:

1. Dinuț, Constantinescu, and Alexandrescu,
   [*Electronics* 15, 2489 (2026)](https://doi.org/10.3390/electronics15112489),
   evaluate full unchanged-prevalence test folds over the 284,807-row ULB
   dataset (training is capped/balanced at 656 rows per fit), repeated over
   three folds and five seeds, with AUPRC, bootstrap intervals, QSVM/VQC versus
   tuned tree baselines, an RBF twin, and depolarizing-noise sweeps; quantum
   models lose.
2. Innan et al. compare QSVC, VQC, EstimatorQNN, and SamplerQNN on a balanced
   200-row sample ([arXiv:2308.05237](https://arxiv.org/abs/2308.05237)).
3. FD4QC compares QSVM, VQC, and HQNN on AML data, but with reduction and
   undersampling ([arXiv:2507.19402](https://arxiv.org/abs/2507.19402)).
4. Bowles--Ahmed--Schuld is the broader controlled QML benchmarking precedent.

Safe sentence:

> Under a repository-local prospective freeze for the new implementations on
> the unchanged-prevalence ULB holdout, seven named QML implementations and their available classical or
> ablation twins show no QML advantage: the strongest product-QSVC and hybrid-
> QNN differences from stacked-LCU are unresolved, while competent classical
> point estimates are as good or better.

### Q17 — Frozen-mixer control: completed result

**Verdict: ANGLE ADAPTATION NECESSARY RELATIVE TO THE PRESPECIFIED
RANDOM-FREEZE DISTRIBUTION; RANDOM FROZEN MIXERS HARMFUL.**

The protocol was frozen before outcomes in
`docs/hsbc_challenge_frozen_mixer_preregistration_v1.md` (SHA-256
`b51ebbe2...`). Seeds `500..509`, batches, 300 updates, validation subset,
checkpoint cadence, and full four-component gradient evaluation were matched.
The controls discarded the computed `phi` gradient; no work was reassigned.
All `phi` displacements are exactly zero.

| arm/contrast | result |
|---|---:|
| trained mean of seed AUPRCs | `0.702931` |
| frozen-random mean | `0.582666` |
| matched frozen-`phi=0` mean | `0.689557` |
| analytic S1 | `0.695689` |
| trained minus frozen-random | `+0.120265 [+0.084777,+0.153449]` simultaneous 97.5%, conditional on fitted seeds |
| frozen-random minus frozen-zero | `-0.106891 [-0.137744,-0.072477]` simultaneous 97.5%, conditional on fitted seeds |
| frozen-zero minus S1 | `-0.006132 [-0.018238,+0.003926]` descriptive 95% |

One random seed produced frozen AP `0.015625`; it was retained as
preregistered. All ten trained-minus-random signs are positive. The confidence
intervals resample rows while holding the ten fitted seeds fixed, so they do
not quantify seed-population uncertainty. The outcome says adaptation matters
relative to this frozen-random distribution and that these arbitrary frozen
mixers do not reproduce the effect. It does not rule out a fixed or
validation-selected nonzero angle, or license a depth, hardware, advantage,
architecture-wide, or universal-coherence claim.

Artifacts:

- runner: `scripts/hsbc_challenge/audit_frozen_mixer_v1.py`;
- JSON: `runs/hsbc_challenge/audit_report_v2/frozen_mixer_v1.json`
  (`90b4b9d3...`);
- aligned scores: `runs/hsbc_challenge/audit_report_v2/frozen_mixer_scores_v1.npz`
  (`22e4f8d0...`).

## Part E — IEEE-CIS rivals extension (conditional)

### Q18 — Prospective design skeleton; not yet a preregistration

**Status: DESIGNED; NOT RUN; NOT FROZEN.** The sealed Engine-A test must not be read again.
The target is the already-visible 118,108-row Engine-A validation partition,
which contains 4,611 frauds. This can be a prospectively frozen **new-arm
validation study**, but not a new blind test: Engine-A validation-period
backbone/routing outcomes are already known and must be disclosed.

1. **Rows and features.** Reuse the exact original chronological 60/20/20 row
   indices. Read only train plus validation. Load the twelve transformed view
   features and quantile transforms from `views_v1.joblib` for every arm,
   quantum and classical. Assert row hashes, feature order, fit scope, counts,
   and no sealed-test access before scoring.
2. **No validation tuning.** Split the original 60% train partition
   chronologically into an internal fit prefix (first 80% of train) and tune
   suffix (last 20% of train). All maps, hyperparameters, checkpoint counts,
   calibration, thresholds, and early stopping use only those two pieces.
   After selecting a configuration, refit once on all original train rows with
   its selected update count; evaluate once on Engine-A validation. The frozen
   `views_v1.joblib` was already fitted/selected using the full original train
   partition, including that suffix. Thus the suffix is validation-clean and
   fair across arms but is **not** a pristine feature-construction holdout;
   rebuilding views on the prefix would answer a different protocol.
3. **Primary arms.** Product-fidelity QSVC; hybrid QNN, dense-head twin, and
   frozen-quantum-head twin; stacked-LCU success flag and syndrome; RBF-SVC or
   Nyström twin; class-weighted logistic; legitimate-only PCA error; XGBoost
   on the same twelve features. Use the exact-path expansion as a classical
   twin for each stack output.
4. **Reduced-effort families.** Exclude the ULB VQC and QAE implementations
   from the primary table because Q13 exposes polarity/tuning and architecture
   limitations. If included for continuity, freeze one corrected-polarity-
   diagnostic run and label every row **“our reduced-effort implementation;
   not family evidence.”** It cannot enter a primary winner claim.
5. **Budgets.** Before execution, a separate preregistration must freeze exact
   search grids, seeds, shot counts, train-row caps, support/landmark caps,
   update/checkpoint rules, maximum sample-circuit forwards, and exact
   statevector/finite-shot status. Do not pretend a support-
   vector overlap and a local-readout circuit have equal hardware query cost;
   report executions, state preparations, parameters, memory, and wall time
   separately.
6. **Metrics.** Primary AUPRC on all validation rows, unchanged prevalence.
   Secondary AUC, calibration/Brier, and recall at a threshold fixed from the
   internal tune suffix at FPR `1e-3` and at a fixed review budget. No balanced
   metric.
7. **Uncertainty.** One set of at least 10,000 paired Poisson(1) validation-row
   weights across every arm. Predeclare the primary comparisons and use Holm
   or Bonferroni simultaneous intervals. Five-seed trainable arms are averaged
   inside each draw; no best validation seed.
8. **Decision boundary.** “Quantum-specific gain” requires a QML arm to beat
   both stacked-LCU and its matched classical/ablation twin with multiplicity-
   corrected intervals, survive a practical margin fixed before outcomes, and
   pass resource accounting. Otherwise report a named-instance ordering only.

A v3 confirmatory test read would require a genuinely new sealed partition or
external dataset. The current IEEE test has been read and characterized; a
second read cannot be made confirmatory by calling it one. No request for such
approval is recommended.

### Q19 — Outcome-contingent proposal sentences frozen in advance

If the ULB ordering transfers:

> On the full Engine-A validation period using the same twelve frozen views,
> stacked-LCU again was not resolved from product-QSVC or hybrid-QNN, and no
> quantum arm beat its matched classical twin; the ULB named-instance ordering
> transfers, with no quantum-advantage claim.

If rankings flip:

> On the full Engine-A validation period using the same twelve frozen views,
> the ULB ordering did not transfer; the accompanying table reports every
> predeclared arm and simultaneous interval. We report the flip as dataset/
> period sensitivity and do not select the favorable bracket.

Revision 3's named-rivals sentence changes only from “ordering transfers” to
“ordering is dataset-sensitive.” The no-advantage sentence changes only if a
quantum arm beats its matched classical twin under the frozen statistical and
resource gates—not merely if it beats stacked-LCU.

### Q20 — Budget gate and VM decision

**Verdict: STOP AFTER PARTS A--D.** The ULB bracket itself took roughly ten
minutes of completed runs, but IEEE-CIS requires new leakage-safe data plumbing,
12-feature arm adaptations, tuning-ledger review, memory profiling, and paired
full-validation scoring. Honest estimate: several VM-hours after 1--2
engineer-days of implementation/referee work. A rushed bracket would be less
valuable than the now-hardened ULB result, mandatory frozen-mixer control, and
proposal diff.

The VM is now prepared without changing its original clean `main` clone at
`b762a48`. An incremental Git bundle created a separate worktree
`<VM_REPRODUCTION_ROOT>` on VM-local branch
`codex/hsbc-audit-v2`, exact base `2fe355f`. An explicit overlay transferred
the audit report, new runners/artifacts, rivals bundle, and public ULB NPZ;
the report and data hashes were verified after transfer. No Git remote was
pushed and no IEEE-CIS raw/test table was transferred. The exact base commit
already contains `engine_a_unblind_v1.py`; it was not executed and has no
sealed table available in the VM worktree.

The new worktree has its own `.venv`, Python 3.12.3 on Linux, with the ten key
pins installed and the repository editable. The local audit used Python
3.12.13 on macOS 26.4 arm64. These pins are not a complete transitive or cross-
platform lock. The VM mechanism smoke passed every scientific assertion but,
correctly, failed the deliberately strict numeric reproduction comparator on
three platform-sensitive cross-view residuals. The mismatch is preserved in
`vm_mechanism_smoke_v2.json`; no tolerance was changed. The IEEE-CIS rivals
bracket remains unrun under the Q20 gate.

Before an approved IEEE launch, add a separately frozen revision-3 protocol
and only the Engine-A train/validation inputs required by Q18. Keep the sealed
test data excluded. The completed environment/smoke preparation does not
authorize the heavy bracket.

## Conditions map to print in revision 3

This is a compact selection from the audit-v1 map plus Engine A. “Winner” is a
measured point/family statement for the specified protocol, not a population
theorem.

| axis | threshold / regime | winning family | evidence and boundary |
|---|---|---|---|
| Standard ULB, 8 features | full labels; 0.1738% test fraud | classical supervised/OCC | XGB-subset `0.749`, PCA `0.724`, trained L=1 `0.706`, S1 `0.696` |
| Standard ULB, 12 features | full labels | classical supervised/OCC | XGB-subset `0.752`, IsolationForest `0.697`, S2 `0.659` |
| Fraud-label scarcity | 5 labels total, including screen/tuning | classical OCC point winner | IsolationForest `0.587`, XGB-full `0.546`, S1 `0.483`; wide seed ranges |
| Clean temporal ULB | later chronological 20% | classical supervised | XGB-full `0.792`, XGB-subset `0.786`, S2 `0.753`, S1 `0.750` |
| Dominant fraud mode | k=3/4/6; centroids and dominant designation use train+validation fraud labels, test only for evaluation | S1 conditionally, but classically evaluable | S1 `0.944/0.945/0.945` vs XGB `0.228/0.413/0.394`; supervised screen; minority modes reverse |
| Extreme prevalence reweighting | exact 0.02% | S1 point winner, no quantum need | S1 `0.498`, XGB-subset `0.488`, IF `0.336`; no row subsampling |
| Bounded inward normalized shift | epsilon `.05/.10/.20` | S1 point winner, no quantum need | S1 `.592/.580/.450` vs XGB `.549/.365/.358`; not an attack-feasibility claim |
| Sparkov supplied test | 555,719 rows / 2,145 frauds | classical supervised | XGB `0.8721`, IF `0.0640`, S1 `0.0062` |
| ULB↔Sparkov transfer | amount/time semantic overlap only | weak classical OCC; no useful transfer | ULB→Sparkov IF `0.0537`; Sparkov→ULB IF `0.00426`; PCA coordinates do not transfer, so this is deliberately weak |
| IEEE-CIS temporal holdout | Engine-A review-band residuals | backbone / no residual | backbone `0.52939`; every residual harms or is unresolved; Q `-0.00230` |

**Conditions verdict:** classical coverage suffices everywhere measured. Rows
won by S1 are exactly classically evaluable at the tested depth. The map
contains conditions under which the mechanism behaves differently, but none
under which a quantum processor is needed.

## Final P0/P1 list for proposal revision 3

### P0 — must fix before submission

1. **Restore mechanism-first weighting.** Keep the Engine-A FAIL prominent but
   stop calling it the centerpiece; follow the frozen Engine-B-first branch.
2. **Remove unsupported external and novelty counterfactuals.** Delete “none
   would survive,” broad balanced-subsample survival claims, and every “first”
   claim not backed by an external pre-outcome timestamp and completed search.
3. **Repair the displayed mathematics and coordinate bridge.** Put the branch
   phase in `F`; state the beta/logit chain rule, common-logit gauge, and
   `18 raw/15 identifiable` Engine-A parameters. Synchronize Markdown and TeX.
4. **Correct application causality.** Replace F2 with the keyed-noise/reranking/
   calibration/drift boundary; delete “not the circuit” and the unsupported
   cross-dataset F3 sentence.
5. **Correct the robustness and shot claims.** Use supervised-screened
   dominant-mode wording, print every k delta/CI and minority reversal, and
   describe 128 shots as unresolved directional dispersion.
6. **Fold in the rivals bracket without strawmen or budget laundering.** Use
   Q14's simultaneous intervals and “not resolved” wording. Label
   VQC/QAE/OC-kernel/HAE rows as our implementations; never use their
   catastrophic scores as family proof. Call the existing bracket bounded,
   not matched-budget, until a common resource currency is enforced.
7. **Use the frozen-mixer result correctly.** Call the positive result a trained
   fixed-L=1 protocol effect relative to the prespecified random-freeze
   distribution; state the conditional-on-fitted-seeds interval boundary. No
   depth, advantage, architecture-wide, or generic-coherence extrapolation.
8. **Delete unsupported resource claims.** Remove “hardware-cheap,” the
   unverified n=12/L=3 `300–700` range, and the old microsecond latency as a
   claim about the proposal model. Add a target-native compile artifact before
   restoring a gate count.
9. **Make the proposal standalone and output-complete.** Add the conditions
   table, calibrated probability/Brier evidence, a thresholded confusion row,
   and one actual per-transaction explanation; define S1/S2/OCC at first use.
10. **Repair submission integrity.** Fill all team/contact/bio/publication/DOI
    fields; make the theory note build; cite commit `5242b2d`; commit/publish
    rivals scripts and artifacts at immutable hashes; narrow the one-read claim
    to IEEE-CIS; disclose IEEE-CIS redistribution limits; correct the El Alami,
    Innan--Khan--Bennai, ring-entangler, and QAE-surrogate metadata in Q13.
11. **Correct measurement and freeze scope.** Replace “8 features at zero
    extra measurement cost” by eight bins/seven independent values from one Z
    setting, with finite shots still required. Scope `1+L` versus 21 to the
    declared L=4 mixing-angle/flag block, not all 18 Engine-A parameters. Call
    measured-output Fisher directly measurable and noise-compatible, not the
    only well-defined noisy metric. Replace “frozen before any data contact”
    by the exact core/A1--A6/train-validation/sealed-test chronology, and call
    AUPRC recommended rather than mandated.
12. **Repair the no-free and LCHS appendix claims.** Delete “only escapes”;
    scope the identity to unamplified raw hit yield. Call the period/span rule
    a necessary alias-free precheck: falling below it permits aliasing but need
    not invert every finite set, while exceeding it guarantees neither
    monotonicity nor approximation accuracy.
13. **Formalize Q6 before claiming a theorem.** Add the fresh-ancilla,
    unitary-branch, exact-dephasing-insertion assumptions and the layerwise
    independence/product proof to the theory note. Numeric `1e-16`
    verification and a code comment are not a publication-ready proof.

### P1 — should fix or explicitly defer

- Run the partial-dephasing simulator/DM1/QPU witness with the controls and
  kill criteria in Q11; without it, Braket use remains only a plan.
- Native-compile the exact n=12/L=3 circuit for the selected Braket target and
  report distributions over mappings, two-qubit gates, depth, and noise—not
  one favorable compile.
- Replicate flag-NG beyond five n=8/L=1 seeds in identifiable coordinates;
  compare damping/trust choices without test selection.
- Run the rolling-origin F2 causal ablation on train/validation periods only.
- Archive the preregistration/amendment/outcome chain and literature-search log
  externally; do not imply that later archiving proves earlier public priority.
- Repair or drop the kernel-OCC arm whose native validation FPR missed its
  target by two orders of magnitude.
- Rerun or omit the shallow VQC/QAE/HAE stress tests: prospectively select VQC
  polarity/readout, use joint trash-register fidelity plus a classical-AE twin,
  and restore a published-scale hybrid-AE bottleneck before main-table use.
- Run IEEE-CIS rivals only after a standalone frozen protocol, clean VM smoke,
  and explicit compute authorization. Never reread the Engine-A sealed test.
- Keep the fully coherent FTQC/QSVT gradient/linear-solve program separate from
  the NISQ fraud evidence until an end-to-end access-model advantage theorem
  and non-tomographic task are supplied.

## Audit artifact ledger and reproduction

### New v2 hardening artifacts

| artifact | SHA-256 | status |
|---|---|---|
| `docs/hsbc_challenge_frozen_mixer_preregistration_v1.md` | `b51ebbe2bd98e7e96258bcb414284b69bf7a8557108a023aca10382761ff8f16` | frozen before outcome |
| `scripts/hsbc_challenge/audit_frozen_mixer_v1.py` | `cd77218a9b3dedfd8d1e95d46c8139f3905a30cb2c071d9f6b8430acff418393` | runnable |
| `runs/hsbc_challenge/audit_report_v2/frozen_mixer_v1.json` | `90b4b9d3d745f02b58b67fe0d35bc7c21f881c34cff75d8077f8a42c4c7ba45d` | COMPLETE |
| `runs/hsbc_challenge/audit_report_v2/frozen_mixer_scores_v1.npz` | `22e4f8d0306e3c26d3dde253612a56f77d00e87bd0fc794e4ba1ac526267bcec` | aligned scores |
| `scripts/hsbc_challenge/harden_quantum_rivals_stats_v2.py` | `510309303a29c790e54c1d859cf8388eae5277e7d6998fcb563d25ad77d11fbe` | runnable post-outcome hardening |
| `runs/hsbc_challenge/audit_report_v2/quantum_rivals_stats_hardened_v2.json` | `b9d4dfb8dae528d2d1b21e5dfd08740e5253ef212a714fc8f0d5930243822ded` | COMPLETE |
| `scripts/hsbc_challenge/diagnose_rival_score_polarity_v2.py` | `3913d5c79afa1ee38598f0cb52f3f442b61c13990898abb15353c3a4f1445f3e` | runnable posthoc diagnostic |
| `runs/hsbc_challenge/audit_report_v2/rival_score_polarity_diagnostic_v2.json` | `f893d7ded8c273a50097b738939c84ef5b3add90bb0dcc970d90942c1e0ecbc9` | COMPLETE_POSTHOC_DIAGNOSTIC |
| `scripts/hsbc_challenge/diagnose_engine_a_noise_head_v2.py` | `18ed91eda3443e8be82eb3da0c029449f48192e4401483347e399c708461c43f` | runnable, read-only, no sealed rows |
| `runs/hsbc_challenge/audit_report_v2/engine_a_noise_head_diagnostic_v2.json` | `ba30703039ca1ddaa14ffbc6a0ca81cebcec2540b3791d4d6adc33d5bada4347` | COMPLETE_READ_ONLY_ARTIFACT_DIAGNOSTIC |
| `scripts/hsbc_challenge/reproduce_audit_inputs_isolated_v2.py` | `4541c95b42a50416f7ccaa446d54085c794cda9f80a174303faff1a2085aee33` | safe isolated orchestrator; mechanism scope executed; full scope supplied but not rerun; never invokes unblind |
| `runs/hsbc_challenge/audit_report_v2/mechanism_reproduction_v2.json` | `6222b09ef7bde2f18ba5cdd6751d75f0cc73762974ca83af8de071ab3e225111` | two-of-two non-timing MATCH |
| `runs/hsbc_challenge/audit_report_v2/vm_mechanism_smoke_v2.json` | `3935481cd69dfa8405c7cd924c9ae0ed8fa10d4e66e180f861b038016a54ba5f` | assertions pass; strict cross-platform MISMATCH preserved |
| `runs/hsbc_challenge/audit_report_v2/baseline_reproduction_v2.json` | `66dc3c41fcc6d67e73cde50a32d934136b2412d2803d86015311e02756043f24` | six-of-six MATCH |

The key-package version snapshot (not a full transitive lock) is
`scripts/hsbc_challenge/requirements_audit_v2.txt`
(`4e5f66bf8d6a934ba48eacf553a332783ea6fb0e12f44f1c253b51a2a3730d6d`).
All new score statistics use the complete imbalanced test rows; no balanced
subsample metric appears.

### Commands

From the repository root with the locked local environment:

```bash
# All six fit-v0 runners plus both mechanism referees. Legacy hard-coded
# writers are contained under a new child of a unique temporary directory.
audit_parent="$(mktemp -d /private/tmp/hsbc-audit-v2.XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/hsbc_challenge/reproduce_audit_inputs_isolated_v2.py \
  --scope all --output-root "$audit_parent/reproduction"

# The orchestrator emits fieldwise comparison JSON at:
# $audit_parent/reproduction/audit_reproduction_comparison_v2.json

# Frozen-mixer reproduction to fresh, non-overwriting names
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/hsbc_challenge/audit_frozen_mixer_v1.py \
  --output runs/hsbc_challenge/audit_report_v2/frozen_mixer_repro_v1.json \
  --scores-output runs/hsbc_challenge/audit_report_v2/frozen_mixer_scores_repro_v1.npz

# Deterministic post-outcome statistics to fresh names
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/hsbc_challenge/harden_quantum_rivals_stats_v2.py \
  --output runs/hsbc_challenge/audit_report_v2/quantum_rivals_stats_repro_v2.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/hsbc_challenge/diagnose_rival_score_polarity_v2.py \
  --output runs/hsbc_challenge/audit_report_v2/rival_score_polarity_repro_v2.json

# Read-only fitted-head diagnostic; opens no IEEE-CIS rows or test scores
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/hsbc_challenge/diagnose_engine_a_noise_head_v2.py \
  --output runs/hsbc_challenge/audit_report_v2/engine_a_noise_head_repro_v2.json
```

**Do not run** `engine_a_unblind_v1.py --real`; its one-read budget is spent.
Reproduce the report by checking its immutable JSON, not by reopening the
sealed test.

## Honesty ledger

- **Application advantage:** REFUTED on ULB/Sparkov coverage and FAILED on the
  preregistered IEEE-CIS experiment.
- **Competitive circuit-realizable scorer:** CONFIRMED only as named-instance,
  exact-simulation evidence; the strongest quantum-rival differences are
  unresolved and classical point leaders remain higher.
- **Fixed-L=1 trained mixer effect:** CONFIRMED and strengthened relative to
  the prespecified random-freeze distribution; those random frozen mixers are
  harmful, while fixed/selected nonzero-angle architecture effects remain open.
- **Failure-sector algebra novelty:** REFUTED broadly; NOVEL-FOR-SETTING only
  for the fresh-ancilla stacked readout.
- **Dephased endpoint / per-sample curve:** NOVEL-FOR-SETTING and the best
  Phase-2 hardware contribution.
- **Flag-NG geometry, no-free lemma, LCHS alias theorem:** KNOWN; retain as
  transparent implementation/accounting results, not novelty.
- **Preregistration/bracket priority:** broad first claims REFUTED; exact
  conjunctions UNCLEAR.
- **Recommended scientific story:** competitive, classically tractable
  quantum-circuit mechanism + measured failure map + bounded Braket mechanism
  validation. No quantum component is needed on the datasets measured so far.
