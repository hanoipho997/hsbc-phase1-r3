# HSBC revision-3 output-completeness protocol v1

Status at freeze: **LOCALLY FROZEN; OUTCOMES NOT YET COMPUTED**  
Freeze date: 2026-09-04 (Europe/Brussels)  
Purpose: supply the calibration, binary-output, and illustrative-explanation
rows required for proposal revision 3 without reopening the IEEE-CIS sealed
test partition.

## Inputs and hard boundary

- Visible validation rows only from
  `runs/hsbc_challenge/data/ieee/trainval.parquet` with `split == "val"`.
- Frozen validation predictions
  `best_xgb_val_pred.npy` and `best_lgb_val_pred.npy`; the declared backbone
  probability is their arithmetic mean.
- Frozen routing thresholds and five Q-arm parameter vectors.
- Existing aggregate-only `unblind_real_v1.json` and `ingest_manifest_v1.json`
  for the already-published test confusion row.
- `test_SEALED.parquet` is forbidden and must not be opened.

## Estimands

On all 118,108 visible validation rows:

1. Brier score of the frozen backbone probability;
2. Brier score of the constant validation-prevalence predictor;
3. ten equal-frequency-bin expected calibration error (ECE), with bins fixed
   by stable sorting and consecutive equal-count slices;
4. confusion counts, precision, recall, and F1 at the frozen upper routing
   threshold `t_hi`.

For the sealed-test row, compute no new predictions. Derive TP/FP/FN/TN only
from the existing aggregate unblinding report: `above` rows are the published
100%-precision auto-flag set, total fraud is `above + pool_frauds +
below_frauds`, and total rows comes from the pre-unblinding ingest manifest.

## Explanation example

Within the visible validation review band only, calculate the exact syndrome
pattern distributions for the five frozen Q seeds. Select the row with the
largest five-seed-mean probability of at least one ancilla failure; ties break
by earliest `TransactionDT`, then smallest `TransactionID`. Fraud label is not
used for selection and is not written.

Report only:

- a one-way hash of the transaction identifier, salted by the literal public
  string `hsbc-r3-example-v1`;
- backbone probability and routing region;
- the three layer-pair names;
- each layer's marginal failure probability;
- joint all-pass probability and most-likely failure pattern; and
- the five-seed range of probability of any failure.

This is a deliberately extreme, model-selected illustration and is not a
representative-case or predictive-accuracy claim.

## Integrity and output

Runner: `scripts/hsbc_challenge/report_revision3_outputs_v1.py`  
Output: `runs/hsbc_challenge/revision3_v1/output_completeness_v1.json`

The runner must print this protocol's SHA-256, hash every input, refuse to
overwrite the output, and emit no raw features, labels, transaction IDs, or
sealed-test predictions.

