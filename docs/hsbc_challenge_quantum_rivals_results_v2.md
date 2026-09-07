# HSBC Phase-1 quantum-rivals audit v2 — results and proposal pivot

Post-outcome companion to the immutable preregistered protocol
`docs/hsbc_challenge_quantum_rivals_audit_v2.md`. Sections 0--5 of that
protocol remain byte-for-byte at the pre-outcome SHA-256 recorded below.

## 6. Results

### 6.1 Execution integrity

The pre-outcome source lock is
`runs/hsbc_challenge/audit_v2/preoutcome_source_lock_v2.json` (SHA-256
`79e87c64e0cf09819d9be72b7213fb6e0636787bffddbecba6c20d7f23a765e4`).
At that lock, this protocol had SHA-256
`de10a4640e310986ea94ed52a35a7edba2bb32fceac7e84116dc2c0bc5ebc115`
and none of the six outcome targets existed. The frozen runner hashes were:

- quantum kernels:
  `51606793c259b59a025dac903267542c584cbcd35631302e93c1fca0080ffcf7`;
- VQC/QAE:
  `3c9776d25c9d1c0496ea91645d3b3ab56b9d696c6cb4404148c2291eb941dd58`;
- hybrid QNN/autoencoder:
  `a3b2fce85e80e64457fbdd17511cb7b8ae70eacdfd312508f06dd102ab62885e`.

All three explicit runs completed. Every result uses the same 56,962 test
transactions containing 99 frauds; validation and test were not resampled or
balanced. The combined deterministic reporter
`scripts/hsbc_challenge/finalize_quantum_rivals_v2.py` verifies the three test
label arrays and artifact hashes, then applies one set of 2,000 paired
Poisson(1) row weights. Its output is
`runs/hsbc_challenge/audit_v2/quantum_rivals_v2_combined.json` (SHA-256
`8166d606d5635aa2fe81e9fdb2fb309cc16595afd2ddae5a1a5f0c9545b90ebb`).

### 6.2 Co-primary paired result

Intervals below are the frozen 98.75% simultaneous intervals. “Tie” also
applies whenever the absolute point difference is at most 0.01. A positive
stack-minus-rival value favours the stack; a positive quantum-minus-twin value
favours the named quantum implementation.

| Family / named implementation | Quantum AUPRC | Stack reference | Stack minus quantum, point [interval] | Classical/control twin AUPRC | Quantum minus twin, point [interval] | Frozen verdict |
|---|---:|---:|---:|---:|---:|---|
| Tuned product-fidelity QSVC | 0.701587 | S2 L=1 0.705758 | +0.004171 [-0.047895, +0.066731] | RBF SVC 0.705807 | -0.004220 [-0.030404, +0.013518] | tie with stack and classical twin |
| Ring-IQP fidelity Nyström | 0.542580 | S2 L=1 0.705758 | +0.163179 [+0.054966, +0.265336] | RBF Nyström 0.730616 | -0.188036 [-0.283855, -0.078603] | stack and classical twin win |
| Projected ring-IQP SVC | 0.632009 | S2 L=1 0.705758 | +0.073749 [-0.018073, +0.158634] | RBF SVC 0.705807 | -0.073798 [-0.160010, +0.017963] | tie under the frozen simultaneous rule; large adverse point gaps |
| Legit-only ring-IQP Nyström OC-SVM | 0.004506 | S1 0.695689 | +0.691182 [+0.564128, +0.796504] | RBF Nyström OC-SVM 0.000934 | +0.003572 [+0.001871, +0.007264] | stack wins; quantum/classical OC kernels are a practical tie and both fail |
| Re-uploading VQC, five-seed mean | 0.006482 | S2 L=1 0.705758 | +0.699277 [+0.569729, +0.804563] | balanced logistic 0.729685 | -0.723203 [-0.816956, -0.605489] | stack and logistic win |
| Eight-angle trash-QAE, five-seed mean | 0.001025 | S1 0.695689 | +0.694664 [+0.567457, +0.799248] | standalone classical AE not in this runner | n/a | stack wins; no family-wide QAE claim |
| Deloitte/AWS-shaped hybrid QNN, five-seed mean | 0.707566 | S2 L=1 0.705758 | -0.001807 [-0.031686, +0.031186] | dense head 0.721783 | -0.014217 [-0.031857, -0.000211] | tie with stack; dense head wins |
| Hybrid quantum-latent AE, five-seed mean | 0.064345 | S1 0.695689 | +0.631344 [+0.506224, +0.727934] | direct-4 / width-8: 0.039663 / 0.524426 | +0.024683 [+0.011042,+0.049688] / -0.460080 [-0.596858,-0.335067] | beats underspecified direct-4 twin, loses decisively to width-8 twin and stack |

Two additional controls sharpen the interpretation. The trainable hybrid-QNN
head ties its frozen-quantum-head control: quantum-minus-frozen is -0.005608
[-0.028629,+0.014370]. The entangling VQC ties its deliberately
effective-one-dimensional no-entanglement readout: -0.011053
[-0.020272,+0.022721]. Neither experiment identifies a benefit from training
the quantum subcircuit.

The point-AUPRC leaders remain classical: XGB-subset 0.749410, classical RBF
Nyström 0.730616, balanced logistic 0.729685, and the dense QNN head 0.721783,
versus S2 L=1 at 0.705758. With only 99 test frauds, their paired 98.75%
intervals versus S2 remain wide and are classified as practical ties; that is
not evidence that S2 beats them. Together with the v1 PCA OCC result 0.723832,
the audit still finds no measurable quantum gain over a competent classical
baseline.

### 6.3 Answer to the family question

**CORRECTED:** the present scorer beats several *tested instantiations*, not
the families. It decisively beats the tested re-uploading VQC, trash-QAE,
hybrid autoencoder, legitimate-only kernel OC-SVM, and ring-IQP Nyström model.
It does **not** beat the strongest tested product-fidelity kernel or hybrid
QNN: both are practical ties with S2 L=1. This completion audit therefore does
not license “better than quantum kernels, quantum autoencoders, VQCs, and
hybrid QNNs.” It licenses only the complete named-instance table above.

The result also rejects the blanket premise that variational/hybrid models are
uncompetitive merely because they use a classical optimizer. The re-uploading
VQC fails badly here, but the hybrid QNN reaches 0.707566 and ties stacked-LCU.
The relevant questions are representation, trainability, query/shot cost, and
matched classical controls. Stacked-LCU itself uses classical Adam or flag-NG,
so “uses classical optimization” cannot distinguish it from the rivals. QAOA
is not a natural primary fraud-classification baseline unless a specific QUBO
decision formulation is declared.

### 6.4 Resource boundary

These are exact-statevector simulation ceilings, not shot-matched hardware
results. The selected product QSVC needs 436 support-state overlaps per new
transaction per shot; ring-IQP Nyström uses 64 landmark overlaps; and the
projected-IQP path materializes 96 Pauli expectations per transaction before
its classical SVC. The VQC/QAE and QNN/HAE scores use compact local readouts,
but their training ledgers include every shifted/SPSA sample-circuit forward.
No finite-shot hardware comparison was run. Consequently none of the v2
results is evidence of runtime, energy, or sample-complexity advantage.

### 6.5 Proposal pivot

The smart Phase-1 story is:

> We propose a compact, quantum-circuit-realizable coherent anomaly scorer and
> test its mechanism under full-prevalence fraud evaluation. A preregistered
> L=1 replication finds a small mixer increment, mean AUPRC +0.00724 with
> paired 95% interval [+0.00137,+0.01337]. In the strengthened rival audit, the
> scorer reaches 0.705758 AUPRC, ties the strongest tested product quantum
> kernel and hybrid QNN, and outperforms several named VQC/QAE/IQP
> instantiations. Competent classical models remain as good or better at point
> estimate, so we claim no quantum advantage. Phase 1 contributes a
> reproducible coherent-interference mechanism, a hardware-validation plan,
> full score/binary/attribution outputs, and an explicit conditions map showing
> where any future quantum component must earn its place.

Delete any claim that the 0.984-vs-0.263 fraud-mode result establishes
label-free novel-mode advantage; the v1 audit refuted that interpretation.
Do not say that stacked-LCU is “reaching the regime where quantum becomes
useful when classical struggles” on ULB: the OCC gate, classical point leaders,
and conditions map do not support it.

### 6.6 Fault-tolerant coherent-gradient boundary

The repository's `fqe.py`, `poly_opt.py`, `linear_solve.py`, and
`coherent_step.py` support a legitimate **FTQC research roadmap**: coherent
state updates, polynomial-objective gradients, QSVT linear solving, and a
coherent preconditioned step. They do not turn the Phase-1 fraud experiment
into a quantum-advantage result. `docs/qlss_natural_gradient_scope.md` already
records that the QSVT inverse is a coherent second-order/preconditioned state
update, not parameter-space QNG; the parameter-space QFIM is a small classical
solve and the ambient metric is trivial. A defensible FTQC advantage claim
would additionally need end-to-end data/oracle construction, condition-number
and success-amplification bounds, total precision dependence, and a readout
task that does not require reconstructing all transaction scores. Until those
are proved against the best classical algorithm for the same access model,
keep FTQC as a separate theory workstream, not the promised explanation for
Phase-1 competitiveness.

### 6.7 Reproduction

From the repository root with the locked `.venv`:

```bash
.venv/bin/python scripts/hsbc_challenge/audit_quantum_kernels_v2.py --run
.venv/bin/python scripts/hsbc_challenge/audit_vqa_qae_v2.py --run
.venv/bin/python scripts/hsbc_challenge/audit_hybrid_qml_v2.py --run
.venv/bin/python scripts/hsbc_challenge/finalize_quantum_rivals_v2.py
```

The runners refuse to overwrite the published result pairs. Reproduction must
therefore use a fresh copy/run directory or deliberately preserve and move the
existing immutable artifacts first; do not delete them as part of this audit.
