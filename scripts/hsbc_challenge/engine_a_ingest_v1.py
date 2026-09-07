"""M2: IEEE-CIS ingest under the Engine-A preregistration (v1, section 2/9).

- train_transaction LEFT JOIN train_identity on TransactionID (labeled rows
  only; the unlabeled competition test files are never read).
- Chronological sort by TransactionDT (stable mergesort; ties keep CSV order).
- Split boundaries at 60% / 80% of rows. The final 20% is written to a
  SEALED parquet file whose contents are not inspected here: no label, score,
  or feature statistic of the test partition is computed or printed by this
  script — only its row count, byte hash, and the last pre-test
  TransactionDT (a train/val quantity) are recorded.
- Artifacts: runs/hsbc_challenge/data/ieee/trainval.parquet,
  runs/hsbc_challenge/data/ieee/test_SEALED.parquet (chmod 400), and
  runs/hsbc_challenge/engine_a_v1/ingest_manifest_v1.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SRC = "external/ieee-fraud-detection"
OUTD = "runs/hsbc_challenge/data/ieee"
MAN = "runs/hsbc_challenge/engine_a_v1/ingest_manifest_v1.json"
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


os.makedirs(OUTD, exist_ok=True)
tt_path = os.path.join(SRC, "train_transaction.csv")
ti_path = os.path.join(SRC, "train_identity.csv")
hash_tt, hash_ti = sha256(tt_path), sha256(ti_path)
log(f"hashed sources: train_transaction {hash_tt[:16]}..., "
    f"train_identity {hash_ti[:16]}...")

tt = pd.read_csv(tt_path)
ti = pd.read_csv(ti_path)
log(f"loaded: transaction {tt.shape}, identity {ti.shape}")
df = tt.merge(ti, how="left", on="TransactionID")
assert len(df) == 590540, f"expected 590540 labeled rows, got {len(df)}"
assert df["TransactionID"].is_unique

df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
n = len(df)
b1, b2 = int(0.6 * n), int(0.8 * n)

trainval = df.iloc[:b2].copy()
trainval["split"] = np.where(np.arange(b2) < b1, "train", "val")
test_sealed = df.iloc[b2:]

tv_path = os.path.join(OUTD, "trainval.parquet")
te_path = os.path.join(OUTD, "test_SEALED.parquet")
trainval.to_parquet(tv_path, index=False)
test_sealed.to_parquet(te_path, index=False)
os.chmod(te_path, 0o400)
del test_sealed  # sealed: nothing further computed on it

tr = trainval[trainval["split"] == "train"]
va = trainval[trainval["split"] == "val"]
manifest = {
    "prereg": "docs/hsbc_engine_a_preregistration_v1.md (v1 + A1)",
    "source_sha256": {"train_transaction.csv": hash_tt,
                      "train_identity.csv": hash_ti},
    "unlabeled_competition_files_used": False,
    "rows_total": int(n),
    "boundaries": {"b1_train_end": b1, "b2_val_end": b2,
                   "test_rows": int(n - b2)},
    "last_preteset_TransactionDT": int(va["TransactionDT"].max()),
    "train": {"rows": int(len(tr)), "frauds": int(tr["isFraud"].sum()),
              "fraud_rate": float(tr["isFraud"].mean()),
              "dt_range": [int(tr["TransactionDT"].min()),
                           int(tr["TransactionDT"].max())]},
    "val": {"rows": int(len(va)), "frauds": int(va["isFraud"].sum()),
            "fraud_rate": float(va["isFraud"].mean()),
            "dt_range": [int(va["TransactionDT"].min()),
                         int(va["TransactionDT"].max())]},
    "test": {"rows": int(n - b2),
             "note": "SEALED - no statistics computed at ingest"},
    "artifacts_sha256": {"trainval.parquet": sha256(tv_path),
                         "test_SEALED.parquet": sha256(te_path)},
    "columns": int(df.shape[1]),
    "versions": {"pandas": pd.__version__, "numpy": np.__version__},
}
with open(MAN, "w") as fh:
    json.dump(manifest, fh, indent=2)

log(f"train: {manifest['train']['rows']} rows / {manifest['train']['frauds']} frauds "
    f"({100*manifest['train']['fraud_rate']:.3f}%)")
log(f"val:   {manifest['val']['rows']} rows / {manifest['val']['frauds']} frauds "
    f"({100*manifest['val']['fraud_rate']:.3f}%)")
log(f"test:  {manifest['test']['rows']} rows SEALED "
    f"(chmod 400, sha {manifest['artifacts_sha256']['test_SEALED.parquet'][:16]}...)")
log(f"M2 ingest complete -> {MAN}")
