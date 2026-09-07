"""Candidate C referee: does the syndrome distribution P(a|x) add drift
signal beyond its classical controls?  VALIDATION rows only (IEEE-CIS
chronological validation partition split into 8 windows); no labels used
for detection; the sealed test parquet is never opened.

Representations compared at matched false-alarm level (permutation
two-sample energy-distance tests, alpha = 0.01, 2,000-row subsamples,
200 permutations, identical subsample indices across representations):
  R1  raw 12 quantile features U (the syndrome's own input)
  R2  backbone score logit (black-box shift detection, Lipton 2018 /
      Rabanser 2019)
  R3  exact syndrome P(a|x), seed-900 Engine-A parameters (8-dim)
  R4  fully dephased syndrome (constant by theorem -> zero statistic)
Data-processing note: R3 = f(R1) for a fixed f, so any test on R3 is a test
on R1 at the same false-alarm rate; R3 can only win by finite-sample
projection effects, never by information content.

Artifacts -> runs/hsbc_challenge/novelty_v1/syndrome_drift_v1.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import (
    CrossViewConfig, CrossViewSLCU, broadcast_view_diagonal, product_state_batch,
    dephased_syndrome_distribution,
)
from novelty_common_v1 import save_json_no_overwrite, sha256_file

T0 = time.time()
BASE = "runs/hsbc_challenge/engine_a_v1"
OUTD = "runs/hsbc_challenge/novelty_v1"
os.makedirs(OUTD, exist_ok=True)
OUT_PATH = os.path.join(OUTD, "syndrome_drift_v1.json")
if os.path.exists(OUT_PATH):
    raise FileExistsError(OUT_PATH)
OUT = {"schema": "hsbc-novelty-syndrome-drift-v1", "sources": {}}
N_SUB, N_PERM, ALPHA, N_WIN = 2000, 200, 0.01, 8


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


views = joblib.load(os.path.join(BASE, "views_v1.joblib"))
arms = json.load(open(os.path.join(BASE, "arms_v1", "arms_summary_v1.json")))
OUT["sources"]["views_v1.joblib"] = sha256_file(os.path.join(BASE, "views_v1.joblib"))
OUT["sources"]["arms_summary_v1.json"] = sha256_file(os.path.join(BASE, "arms_v1", "arms_summary_v1.json"))
cols = list(dict.fromkeys(["TransactionID", "TransactionDT", "isFraud", "split"] + views["features"]))
TRAINVAL = "runs/hsbc_challenge/data/ieee/trainval.parquet"
assert "SEALED" not in TRAINVAL
df = pd.read_parquet(TRAINVAL, columns=cols)
va = df[df["split"] == "val"].reset_index(drop=True)
del df
pval = 0.5 * (np.load(os.path.join(BASE, "backbone_vm", "best_xgb_val_pred.npy")).astype(np.float64)
              + np.load(os.path.join(BASE, "backbone_vm", "best_lgb_val_pred.npy")).astype(np.float64))
order = np.argsort(va["TransactionDT"].to_numpy(), kind="mergesort")
va = va.iloc[order].reset_index(drop=True)
pval = pval[order]
y = va["isFraud"].to_numpy()


def views_u(fr):
    out = []
    for c in views["features"]:
        s = fr[c]
        if c in views["freq_maps"]:
            s = s.astype(str).map({k: float(v) for k, v in views["freq_maps"][c].items()}) \
                if s.dtype == object else s.astype(float)
        x = np.nan_to_num(s.to_numpy(dtype=float), nan=-1.0)
        col = views["sorted_cols"][c]
        lo = np.searchsorted(col, x, side="left")
        hi = np.searchsorted(col, x, side="right")
        out.append(np.clip(0.5 * (lo + hi) / len(col), 0.0, 1.0))
    return np.column_stack(out)


U = views_u(va)
VQ = views["view_qubits"]
name2idx = {"A": 0, "B": 1, "C": 2}
cfg = CrossViewConfig(
    n_qubits=12, views=tuple(tuple(VQ[v]) for v in ("A", "B", "C")),
    layer_pairs=tuple((name2idx[a], name2idx[b]) for a, b in views["layer_pairs"]),
    hp_views=tuple(broadcast_view_diagonal(np.array(views["hp_tables"][v]), VQ[v], 12)
                   for v in ("A", "B", "C")))
sim = CrossViewSLCU(cfg)
params = np.array(arms["Q"]["seeds"]["900"]["params"])
t0 = time.time()
P = np.concatenate([sim.syndrome_distribution(params, product_state_batch(U[i:i + 4096]).astype(complex))
                    for i in range(0, len(U), 4096)])
log(f"syndrome for {len(U)} rows in {time.time()-t0:.1f}s")
Pdeph = dephased_syndrome_distribution(params, 3, 1, stride=6)[0]
logit_b = np.log(np.clip(pval, 1e-7, 1 - 1e-7) / (1 - np.clip(pval, 1e-7, 1 - 1e-7)))


def energy_distance(X, Y):
    """Squared energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'| (unbiased-ish V-stat)."""
    def mean_pdist(A, B):
        d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0))
        return d.mean()
    return 2 * mean_pdist(X, Y) - mean_pdist(X, X) - mean_pdist(Y, Y)


def perm_test(X, Y, rng, n_perm=N_PERM):
    stat = energy_distance(X, Y)
    Z = np.concatenate([X, Y])
    n = len(X)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(Z))
        if energy_distance(Z[p[:n]], Z[p[n:]]) >= stat:
            cnt += 1
    return float(stat), float((cnt + 1) / (n_perm + 1))


REPS = {"R1_raw_features": U, "R2_backbone_logit": logit_b[:, None], "R3_syndrome": P,
        "R4_dephased_syndrome": np.tile(Pdeph, (len(U), 1))}
wins = np.array_split(np.arange(len(U)), N_WIN)
rng = np.random.default_rng(20260904)
ref_idx = rng.choice(wins[0], N_SUB, replace=False)
results = {name: [] for name in REPS}
fraud_rate = [float(y[w].mean()) for w in wins]
for k in range(1, N_WIN):
    idx = rng.choice(wins[k], N_SUB, replace=False)
    prng_seed = 100 + k
    for name, R in REPS.items():
        prng = np.random.default_rng(prng_seed)            # identical permutations across reps
        X, Y = R[ref_idx], R[idx]
        if name == "R4_dephased_syndrome":
            results[name].append({"window": k, "stat": 0.0, "p_value": 1.0})
            continue
        stat, pv = perm_test(X, Y, prng)
        results[name].append({"window": k, "stat": stat, "p_value": pv})
    log(f"window {k}: " + ", ".join(f"{n}: stat {r[-1]['stat']:.4g} p {r[-1]['p_value']:.3f}"
                                   for n, r in results.items()))

first_detect = {}
for name, rows in results.items():
    det = [r["window"] for r in rows if r["p_value"] < ALPHA]
    first_detect[name] = int(det[0]) if det else None
OUT["design"] = {"windows": N_WIN, "subsample": N_SUB, "permutations": N_PERM, "alpha": ALPHA,
                 "reference_window": 0, "rows_per_window": [int(len(w)) for w in wins],
                 "fraud_rate_per_window_disclosed": fraud_rate}
OUT["results"] = results
OUT["first_detection_window_at_alpha"] = first_detect
OUT["data_processing_note"] = ("R3 is a deterministic function of R1 (same 12 features); any level-alpha "
                               "test on R3 is a level-alpha test on R1, so R1's optimal power dominates "
                               "R3's; observed differences are finite-sample projection effects only.")
OUT["wall_seconds"] = time.time() - T0
save_json_no_overwrite(OUT_PATH, OUT)
log(f"DONE -> {OUT_PATH}; first detection windows {first_detect}")
