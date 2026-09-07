"""Sparkov regime and ULB<->Sparkov transfer audit for HSBC v1.

The open dataset source, feature contract, temporal split, and training caps
were frozen in docs/hsbc_challenge_audit_report_v1.md section 1.8.

Example:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
    scripts/hsbc_challenge/audit_sparkov_v1.py \
    --sparkov-dir /private/tmp/hsbc_sparkov_20260830
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_priority_attacks_v1 import fit_xgb, probs_matmul  # noqa: E402
from hsbc_common import (  # noqa: E402
    QuantileTransform,
    fit_ising_pseudolikelihood,
    hard_bits,
    load_ulb,
    product_probs_batch,
    ranking_metrics,
    stratified_split,
)


SEED = 20260830
CHUNK = 8192
T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


USECOLS = [
    "trans_date_trans_time",
    "cc_num",
    "category",
    "amt",
    "lat",
    "long",
    "city_pop",
    "dob",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


def load_frame(path: Path):
    import pandas as pd

    return pd.read_csv(
        path,
        usecols=USECOLS,
        parse_dates=["trans_date_trans_time", "dob"],
    )


def fit_legit_maps(frame):
    legit = frame[frame["is_fraud"] == 0]
    category_frequency = legit["category"].value_counts(normalize=True).to_dict()
    log_amt = np.log1p(legit["amt"].to_numpy(float))
    import pandas as pd

    tmp = pd.DataFrame({"cc_num": legit["cc_num"].to_numpy(), "log_amt": log_amt})
    stats = tmp.groupby("cc_num")["log_amt"].agg(["mean", "std"])
    card_mean = stats["mean"].to_dict()
    card_std = stats["std"].fillna(0.0).to_dict()
    global_mean = float(np.mean(log_amt))
    global_std = float(np.std(log_amt))
    return category_frequency, card_mean, card_std, global_mean, global_std


def engineer(frame, maps):
    category_frequency, card_mean, card_std, global_mean, global_std = maps
    dt = frame["trans_date_trans_time"]
    hour = dt.dt.hour.to_numpy(float) + dt.dt.minute.to_numpy(float) / 60.0
    age = (dt - frame["dob"]).dt.total_seconds().to_numpy(float) / (365.25 * 86400.0)
    log_amt = np.log1p(frame["amt"].to_numpy(float))
    means = frame["cc_num"].map(card_mean).fillna(global_mean).to_numpy(float)
    stds = frame["cc_num"].map(card_std).fillna(global_std).to_numpy(float)
    stds = np.maximum(stds, 0.05)
    card_dev = np.clip((log_amt - means) / stds, -20, 20)
    cat_freq = frame["category"].map(category_frequency).fillna(0.0).to_numpy(float)
    distance = haversine_km(
        frame["lat"].to_numpy(float),
        frame["long"].to_numpy(float),
        frame["merch_lat"].to_numpy(float),
        frame["merch_long"].to_numpy(float),
    )
    features = np.column_stack(
        [
            log_amt,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            age,
            np.log1p(distance),
            np.log1p(frame["city_pop"].to_numpy(float)),
            cat_freq,
            card_dev,
        ]
    )
    names = [
        "log_amount",
        "hour_sin",
        "hour_cos",
        "age_years",
        "log_distance_km",
        "log_city_population",
        "legit_category_frequency",
        "legit_card_amount_deviation",
    ]
    return features, frame["is_fraud"].to_numpy(int), names


def chunked_spectral_score(U, model, tau):
    g = np.exp(-2.0 * tau * model.hp)
    rows = []
    for start in range(0, len(U), CHUNK):
        rows.append(product_probs_batch(U[start : start + CHUNK]) @ g)
    return -np.concatenate(rows)


def read_ulb_times():
    path = Path(
        "runs/hsbc_challenge/data/openml/openml.org/data/v1/download/1673544/creditcard.arff.gz"
    )
    times = []
    in_data = False
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if not in_data:
                if line.strip().lower() == "@data":
                    in_data = True
                continue
            if line.strip():
                times.append(float(line.split(",", 1)[0]))
    return np.asarray(times)


def shared_time_amount_features(amount, seconds):
    hour = (seconds % 86400.0) / 3600.0
    return np.column_stack(
        [np.log1p(np.asarray(amount, float)), np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    )


def transfer_audit(X_fit, y_fit, X_test, y_test, spark_fit_frame, spark_test_frame):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X_ulb, y_ulb, _ = load_ulb()
    times = read_ulb_times()
    idx_tr, _, idx_te = stratified_split(y_ulb, seed=0)
    shared_ulb = shared_time_amount_features(X_ulb[:, -1], times)
    spark_fit_shared = X_fit[:, :3]
    spark_test_shared = X_test[:, :3]

    # ULB -> Sparkov
    scaler_u = StandardScaler().fit(shared_ulb[idx_tr][y_ulb[idx_tr] == 0])
    iso_u = IsolationForest(
        n_estimators=300,
        max_samples=8192,
        random_state=SEED,
        n_jobs=4,
    ).fit(scaler_u.transform(shared_ulb[idx_tr][y_ulb[idx_tr] == 0]))
    u_to_s = -iso_u.score_samples(scaler_u.transform(spark_test_shared))

    # Sparkov -> ULB
    rng = np.random.default_rng(SEED)
    legit_s = np.flatnonzero(y_fit == 0)
    cap = np.sort(rng.choice(legit_s, min(80000, len(legit_s)), replace=False))
    scaler_s = StandardScaler().fit(spark_fit_shared[cap])
    iso_s = IsolationForest(
        n_estimators=300,
        max_samples=8192,
        random_state=SEED,
        n_jobs=4,
    ).fit(scaler_s.transform(spark_fit_shared[cap]))
    s_to_u = -iso_s.score_samples(scaler_s.transform(shared_ulb[idx_te]))
    return {
        "shared_features": ["log_amount", "hour_sin", "hour_cos"],
        "ULB_to_Sparkov": ranking_metrics(y_test, u_to_s),
        "Sparkov_to_ULB": ranking_metrics(y_ulb[idx_te], s_to_u),
        "scope": "weak semantic overlap only; ULB PCA features have no Sparkov counterpart",
    }, u_to_s, s_to_u, y_ulb[idx_te]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparkov-dir", required=True)
    parser.add_argument(
        "--output",
        default="runs/hsbc_challenge/audit_v1/sparkov_v1.json",
    )
    parser.add_argument(
        "--scores-output",
        default="runs/hsbc_challenge/audit_v1/sparkov_scores_v1.npz",
    )
    args = parser.parse_args()
    data_dir = Path(args.sparkov_dir)
    train_path, test_path = data_dir / "fraudTrain.csv", data_dir / "fraudTest.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Sparkov fraudTrain.csv and fraudTest.csv are required")
    log("loading Sparkov CSVs")
    train_frame, test_frame = load_frame(train_path), load_frame(test_path)
    cut = int(0.8 * len(train_frame))
    fit_frame, val_frame = train_frame.iloc[:cut].copy(), train_frame.iloc[cut:].copy()
    maps = fit_legit_maps(fit_frame)
    X_fit, y_fit, names = engineer(fit_frame, maps)
    X_val, y_val, _ = engineer(val_frame, maps)
    X_test, y_test, _ = engineer(test_frame, maps)
    log(f"engineered fit/val/test {len(y_fit)}/{len(y_val)}/{len(y_test)}")

    # Legit-fitted S1 scorer.
    qt = QuantileTransform(X_fit)
    U_fit, U_val, U_test = qt(X_fit), qt(X_val), qt(X_test)
    model = fit_ising_pseudolikelihood(hard_bits(U_fit)[y_fit == 0], seed=0)
    tau_grid = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
    val_scores = {tau: chunked_spectral_score(U_val, model, tau) for tau in tau_grid}
    from sklearn.metrics import average_precision_score

    tau = max(tau_grid, key=lambda value: average_precision_score(y_val, val_scores[value]))
    s1 = chunked_spectral_score(U_test, model, tau)
    log(f"Sparkov S1 tau={tau}, AUPRC={average_precision_score(y_test, s1):.4f}")

    # Classical OCC on the identical eight features.
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    legit = np.flatnonzero(y_fit == 0)
    occ_cap = np.sort(rng.choice(legit, min(80000, len(legit)), replace=False))
    scaler = StandardScaler().fit(X_fit[occ_cap])
    iso = IsolationForest(
        n_estimators=300,
        max_samples=8192,
        random_state=SEED,
        n_jobs=4,
    ).fit(scaler.transform(X_fit[occ_cap]))
    occ = -iso.score_samples(scaler.transform(X_test))
    log(f"Sparkov IsolationForest AUPRC={average_precision_score(y_test, occ):.4f}")

    # Supervised XGB, capped only on legitimate training rows; every fraud and
    # the complete validation/test distribution are retained.
    fraud_fit = np.flatnonzero(y_fit == 1)
    legit_cap = np.sort(rng.choice(legit, min(300000, len(legit)), replace=False))
    xgb_rows = np.sort(np.concatenate([legit_cap, fraud_fit]))
    xgb = fit_xgb(X_fit[xgb_rows], y_fit[xgb_rows], X_val, y_val, seed=SEED)
    xgb_score = xgb.predict_proba(X_test)[:, 1]
    log(f"Sparkov XGB AUPRC={average_precision_score(y_test, xgb_score):.4f}")

    transfer, u_to_s, s_to_u, y_ulb_test = transfer_audit(
        X_fit, y_fit, X_test, y_test, fit_frame, test_frame
    )
    output = Path(args.output)
    scores_output = Path(args.scores_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scores_output,
        y_test=y_test,
        S1=s1,
        IsolationForest=occ,
        XGB_subset=xgb_score,
        ULB_to_Sparkov=u_to_s,
        y_ULB_test=y_ulb_test,
        Sparkov_to_ULB=s_to_u,
    )
    result = {
        "schema": "hsbc-sparkov-v1",
        "protocol_freeze": "docs/hsbc_challenge_audit_report_v1.md section 1.8",
        "source": {
            "canonical_listing": "https://www.kaggle.com/datasets/kartik2112/fraud-detection",
            "download_endpoint": "https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection",
            "train_sha256": sha256(train_path),
            "test_sha256": sha256(test_path),
        },
        "temporal_contract": {
            "fit_rows_frauds": [int(len(y_fit)), int(np.sum(y_fit))],
            "validation_rows_frauds": [int(len(y_val)), int(np.sum(y_val))],
            "test_rows_frauds": [int(len(y_test)), int(np.sum(y_test))],
            "fit_time_min_max": [str(fit_frame["trans_date_trans_time"].min()), str(fit_frame["trans_date_trans_time"].max())],
            "validation_time_min_max": [str(val_frame["trans_date_trans_time"].min()), str(val_frame["trans_date_trans_time"].max())],
            "test_time_min_max": [str(test_frame["trans_date_trans_time"].min()), str(test_frame["trans_date_trans_time"].max())],
            "train_file_nondecreasing": bool(train_frame["trans_date_trans_time"].is_monotonic_increasing),
            "test_file_nondecreasing": bool(test_frame["trans_date_trans_time"].is_monotonic_increasing),
        },
        "features": names,
        "tau_selected_on_validation": tau,
        "methods": {
            "S1": ranking_metrics(y_test, s1),
            "IsolationForest": ranking_metrics(y_test, occ),
            "XGB_subset": ranking_metrics(y_test, xgb_score),
        },
        "train_caps": {
            "OCC_legitimate_rows": int(len(occ_cap)),
            "XGB_legitimate_rows": int(len(legit_cap)),
            "XGB_fraud_rows": int(len(fraud_fit)),
            "test_evaluation_rows": int(len(y_test)),
        },
        "cross_dataset_transfer": transfer,
        "scores_artifact": str(scores_output),
        "scores_sha256": sha256(scores_output),
        "wall_seconds": time.time() - T0,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()

