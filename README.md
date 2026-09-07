# HSBC Phase-1 Revision 3 — public evidence package

This repository contains the sanitized public package for **Coherent
Cross-View Syndromes for Credit-Card Fraud: Mechanism Validation Under
Classical Stop Gates**, prepared for the 2026 Global Quantum + AI Challenge.

The proposal makes **no quantum-advantage claim**. Its measured IEEE-CIS
application experiment failed the prospectively frozen benefit gates. The
surviving contribution is a circuit-realizable monitoring mechanism, exact
mechanism referees, a corrected full-prevalence audit, and a bounded Amazon
Braket validation plan with classical twins and stop rules.

## Contents

- `docs/hsbc_challenge_phase1_proposal.{md,tex,pdf}` — Revision-3 proposal;
- `docs/hsbc_lcu_qng_theory_note.{tex,pdf}` — companion theory note;
- audit, rivals, Engine-A and novelty reports under `docs/`;
- reproducibility code under `scripts/hsbc_challenge/`;
- aggregate-only evidence JSON under `runs/hsbc_challenge/`; and
- `PUBLIC_RELEASE_MANIFEST.json` plus `verify_public_release_v1.py`.

## Data boundary

No IEEE-CIS row, label vector, transformed row array, prediction array,
serialized model/preprocessor, NPZ, NPY, Joblib, Parquet, CSV, or pickle is
included. Private source hashes, row tokens, private classifications, and
references to absent row/model payloads are also removed from released JSON;
reported scientific values are unchanged. IEEE-CIS must be obtained and used
under its own terms. Some scripts therefore require private/local inputs that
are intentionally absent; the reports and aggregate artifacts label those
boundaries.

Verify the bundle from its root:

```bash
python verify_public_release_v1.py
```

## Team and contact

Ha Cong Nguyen is the single-member team, lead/contact, model developer and
model-risk owner. Project contact is through this repository.

## Citation

Use the DOI and immutable scientific-content commit printed in the final
proposal and `CITATION.cff`.
