"""Revision-3 output-completeness report, validation-only plus sealed aggregates.

Protocol: docs/hsbc_challenge_revision3_output_protocol_v1.md
This runner never opens the IEEE-CIS sealed parquet.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import (  # noqa: E402
    CrossViewConfig,
    CrossViewSLCU,
    broadcast_view_diagonal,
    product_state_batch,
)


PROTOCOL = Path("docs/hsbc_challenge_revision3_output_protocol_v1.md")
BASE = Path("runs/hsbc_challenge/engine_a_v1")
OUT = Path("runs/hsbc_challenge/revision3_v1/output_completeness_v1.json")
SALT = "hsbc-r3-example-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def equal_frequency_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    order = np.argsort(p, kind="mergesort")
    rows = []
    weighted = 0.0
    for idx in np.array_split(order, bins):
        mean_p = float(p[idx].mean())
        observed = float(y[idx].mean())
        weighted += len(idx) * abs(mean_p - observed)
        rows.append({
            "n": int(len(idx)),
            "mean_probability": mean_p,
            "observed_fraud_rate": observed,
            "absolute_gap": abs(mean_p - observed),
        })
    return {"bins": bins, "ece": weighted / len(y), "bin_rows": rows}


def view_features(frame: pd.DataFrame, views: dict) -> np.ndarray:
    columns = []
    for name in views["features"]:
        series = frame[name]
        if name in views["freq_maps"]:
            if series.dtype == object:
                mapping = {key: float(value) for key, value in views["freq_maps"][name].items()}
                series = series.astype(str).map(mapping)
            else:
                series = series.astype(float)
        values = np.nan_to_num(series.to_numpy(dtype=float), nan=-1.0)
        sorted_train = views["sorted_cols"][name]
        left = np.searchsorted(sorted_train, values, side="left")
        right = np.searchsorted(sorted_train, values, side="right")
        columns.append(np.clip(0.5 * (left + right) / len(sorted_train), 0.0, 1.0))
    return np.column_stack(columns)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = sha256_file(PROTOCOL)
    print(f"PROTOCOL {PROTOCOL} SHA-256 = {protocol_hash}", flush=True)

    views_path = BASE / "views_v1.joblib"
    routing_path = BASE / "routing_freeze_v1.json"
    arms_path = BASE / "arms_v1/arms_summary_v1.json"
    ingest_path = BASE / "ingest_manifest_v1.json"
    unblind_path = BASE / "unblind_real_v1.json"
    xgb_path = BASE / "backbone_vm/best_xgb_val_pred.npy"
    lgb_path = BASE / "backbone_vm/best_lgb_val_pred.npy"
    trainval_path = Path("runs/hsbc_challenge/data/ieee/trainval.parquet")
    assert "SEALED" not in str(trainval_path)

    views = joblib.load(views_path)
    routing = json.loads(routing_path.read_text())
    arms = json.loads(arms_path.read_text())
    ingest = json.loads(ingest_path.read_text())
    unblind = json.loads(unblind_path.read_text())
    p_val = 0.5 * (
        np.load(xgb_path).astype(np.float64) + np.load(lgb_path).astype(np.float64)
    )

    columns = list(dict.fromkeys(
        ["TransactionID", "TransactionDT", "isFraud", "split", *views["features"]]
    ))
    frame = pd.read_parquet(trainval_path, columns=columns)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    del frame
    y_val = val["isFraud"].to_numpy(dtype=int)
    if len(val) != len(p_val):
        raise AssertionError(f"validation alignment mismatch: {len(val)} != {len(p_val)}")
    if len(val) != ingest["val"]["rows"] or int(y_val.sum()) != ingest["val"]["frauds"]:
        raise AssertionError("validation counts disagree with the frozen ingest manifest")

    prevalence = float(y_val.mean())
    t_lo = float(routing["routing_thresholds"]["t_lo_val_q988"])
    t_hi = float(routing["routing_thresholds"]["t_hi_val_q998"])
    validation = {
        "rows": int(len(val)),
        "frauds": int(y_val.sum()),
        "prevalence": prevalence,
        "backbone_brier": float(np.mean((p_val - y_val) ** 2)),
        "constant_prevalence_brier": float(np.mean((prevalence - y_val) ** 2)),
        "equal_frequency_calibration": equal_frequency_ece(y_val, p_val),
        "binary_at_frozen_t_hi": {
            "threshold": t_hi,
            **binary_metrics(y_val, p_val > t_hi),
        },
        "interpretation": (
            "Descriptive validation evidence only. Brier/ECE assess the frozen "
            "backbone probability output; no posthoc recalibration was fitted."
        ),
    }

    regions = unblind["regions"]
    test_rows = int(ingest["boundaries"]["test_rows"])
    test_tp = int(regions["above"])
    test_frauds = test_tp + int(regions["pool_frauds"]) + int(regions["below_frauds"])
    test_binary = binary_metrics(
        np.r_[np.ones(test_frauds, dtype=int), np.zeros(test_rows - test_frauds, dtype=int)],
        np.r_[np.ones(test_tp, dtype=int), np.zeros(test_rows - test_tp, dtype=int)],
    )
    # The constructed vectors above encode only the already-published aggregate
    # counts: all auto-flag rows were frauds. No test row or score is loaded.
    test_binary.update({
        "threshold": t_hi,
        "rows": test_rows,
        "frauds": test_frauds,
        "source": "derived only from ingest_manifest_v1.json + unblind_real_v1.json",
        "sealed_test_reopened": False,
    })

    pool = (p_val > t_lo) & (p_val <= t_hi)
    pool_frame = val.loc[pool].reset_index(drop=True)
    pool_p = p_val[pool]
    u_pool = view_features(pool_frame, views)
    states = product_state_batch(u_pool).astype(complex)
    view_qubits = views["view_qubits"]
    name_to_index = {"A": 0, "B": 1, "C": 2}
    cfg = CrossViewConfig(
        n_qubits=12,
        views=tuple(tuple(view_qubits[name]) for name in ("A", "B", "C")),
        layer_pairs=tuple(
            (name_to_index[a], name_to_index[b]) for a, b in views["layer_pairs"]
        ),
        hp_views=tuple(
            broadcast_view_diagonal(np.asarray(views["hp_tables"][name]), view_qubits[name], 12)
            for name in ("A", "B", "C")
        ),
    )
    simulator = CrossViewSLCU(cfg)
    seeds = sorted(arms["Q"]["seeds"])
    p0_sum = np.zeros(len(pool_frame), dtype=float)
    for seed in seeds:
        params = np.asarray(arms["Q"]["seeds"][seed]["params"])
        p0_sum += simulator.syndrome_distribution(params, states)[:, 0]
    mean_any_failure = 1.0 - p0_sum / len(seeds)
    candidates = np.flatnonzero(mean_any_failure == mean_any_failure.max())
    if len(candidates) > 1:
        order = np.lexsort((
            pool_frame.loc[candidates, "TransactionID"].to_numpy(),
            pool_frame.loc[candidates, "TransactionDT"].to_numpy(),
        ))
        selected = int(candidates[order[0]])
    else:
        selected = int(candidates[0])

    state = states[selected:selected + 1]
    distributions = []
    for seed in seeds:
        params = np.asarray(arms["Q"]["seeds"][seed]["params"])
        distributions.append(simulator.syndrome_distribution(params, state)[0])
    distributions = np.asarray(distributions)
    mean_distribution = distributions.mean(axis=0)
    patterns = np.arange(2 ** cfg.n_layers)
    marginals = [
        float(mean_distribution[((patterns >> layer) & 1) == 1].sum())
        for layer in range(cfg.n_layers)
    ]
    failure_patterns = patterns[patterns != 0]
    most_likely_failure = int(
        failure_patterns[np.argmax(mean_distribution[failure_patterns])]
    )
    transaction_id = str(int(pool_frame.loc[selected, "TransactionID"]))
    row_hash = hashlib.sha256(f"{SALT}:{transaction_id}".encode()).hexdigest()[:16]
    explanation = {
        "selection": "largest five-seed-mean P(any ancilla failure) within validation review band",
        "representative_case_claim": False,
        "row_token_sha256_prefix": row_hash,
        "backbone_probability": float(pool_p[selected]),
        "routing_region": "review_band",
        "layer_pairs": [list(pair) for pair in views["layer_pairs"]],
        "mean_layer_failure_marginals": marginals,
        "mean_joint_all_pass_probability": float(mean_distribution[0]),
        "mean_probability_any_failure": float(1.0 - mean_distribution[0]),
        "five_seed_probability_any_failure_range": [
            float((1.0 - distributions[:, 0]).min()),
            float((1.0 - distributions[:, 0]).max()),
        ],
        "most_likely_nonzero_pattern_little_endian": format(
            most_likely_failure, f"0{cfg.n_layers}b"
        )[::-1],
        "most_likely_nonzero_pattern_probability": float(
            mean_distribution[most_likely_failure]
        ),
        "pattern_bit_order": "layer 0, layer 1, layer 2; 1 means that ancilla failed",
        "fraud_label_used_for_selection": False,
        "fraud_label_emitted": False,
    }

    sources = {
        str(path): sha256_file(path)
        for path in (
            PROTOCOL,
            views_path,
            routing_path,
            arms_path,
            ingest_path,
            unblind_path,
            xgb_path,
            lgb_path,
            trainval_path,
        )
    }
    payload = {
        "schema": "hsbc-revision3-output-completeness-v1",
        "status": "VALIDATION_ONLY_PLUS_EXISTING_SEALED_AGGREGATES",
        "protocol": {"path": str(PROTOCOL), "sha256": protocol_hash},
        "sources": sources,
        "validation_probability_output": validation,
        "sealed_test_binary_from_existing_aggregates": test_binary,
        "illustrative_validation_explanation": explanation,
        "data_release_classification": "PRIVATE_LICENSED_INPUTS_AGGREGATE_OUTPUT",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "output": str(OUT),
        "validation_brier": validation["backbone_brier"],
        "validation_ece": validation["equal_frequency_calibration"]["ece"],
        "validation_binary": validation["binary_at_frozen_t_hi"],
        "sealed_test_binary": test_binary,
        "example": explanation,
    }, indent=2))


if __name__ == "__main__":
    main()
