#!/usr/bin/env python3
"""Run the preregistered HSBC L=1 frozen-random-mixer control.

The outcome protocol is frozen in
``docs/hsbc_challenge_frozen_mixer_preregistration_v1.md``.  This runner keeps
the original seed/batch/checkpoint path and computes, then discards, the mixer
gradient so its evaluation budget matches the trained L=1 replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(SCRIPT.parent))

import audit_priority_attacks_v1 as parent  # noqa: E402


PROTOCOL = REPO_ROOT / "docs/hsbc_challenge_frozen_mixer_preregistration_v1.md"
PARENT_JSON = REPO_ROOT / "runs/hsbc_challenge/audit_v1/priority_attacks_v1.json"
PARENT_SCORES = REPO_ROOT / "runs/hsbc_challenge/audit_v1/priority_scores_v1.npz"
DATA = REPO_ROOT / "runs/hsbc_challenge/data/ulb_creditcard.npz"
EXP_B = REPO_ROOT / "runs/hsbc_challenge/fit_v0/exp_b_scorer.json"
COMMON = REPO_ROOT / "scripts/hsbc_challenge/hsbc_common.py"
PARENT_SCRIPT = REPO_ROOT / "scripts/hsbc_challenge/audit_priority_attacks_v1.py"

EXPECTED_HASHES = {
    PROTOCOL: "b51ebbe2bd98e7e96258bcb414284b69bf7a8557108a023aca10382761ff8f16",
    PARENT_JSON: "8292f89f3108d2758a46cd80db8a922d8f63bd8b2640774cd1116f398f7ff4ee",
    PARENT_SCORES: "64a9bbfbd06d79eb917e39a59d9951237dccc522b3b149beb1e724ea4728e547",
    DATA: "40ad6e73b1ef5c2b42265b7a02e89caf9738f09520770451a6c8d4491f4d79d6",
    EXP_B: "c735713d6f6a897ebe4b6aa1d622dddbcf7c7a2476c27650e956a53a14d507ee",
    COMMON: "8623715b486de24a6e143411249679bede20a22c3282ba518c096b1f11e1d2ec",
    PARENT_SCRIPT: "f7e82da19395e16b488ab7d4b046f41fd58add28f26fc81b5819c46a9c0c29bd",
}

SEEDS = tuple(range(500, 510))
N_BOOT = 2000
BOOT_SEED = 20260901
UPDATES = 300
CHECK_EVERY = 20
BATCH_LEGIT = 384
PHI_INDEX = 3
PHI_ATOL = 1e-15
T0 = time.time()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def check_source_lock() -> None:
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from repository root {REPO_ROOT}")
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"source-lock mismatch for {path.relative_to(REPO_ROOT)}: "
                f"expected {expected}, observed {observed}"
            )
    parent_payload = json.loads(PARENT_JSON.read_text())
    if parent_payload.get("scores_sha256") != EXPECTED_HASHES[PARENT_SCORES]:
        raise RuntimeError("parent JSON does not bind the frozen score bundle")


def validation_subset(y_validation: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sort(
        np.concatenate(
            [
                np.flatnonzero(y_validation == 1),
                rng.choice(np.flatnonzero(y_validation == 0), 8000, replace=False),
            ]
        )
    )


def train_frozen_seed(pipe: dict, seed: int, mode: str) -> tuple[np.ndarray, dict]:
    from sklearn.metrics import average_precision_score

    U_train, U_validation, U_test = (
        pipe["U"][key] for key in ("tr", "va", "te")
    )
    y_train, y_validation = pipe["ys"]["tr"], pipe["ys"]["va"]
    simulator = parent.BatchedSLCU(
        parent.SLCUStackConfig(n_qubits=8, n_layers=1, hp=pipe["model"].hp)
    )
    fraud_states = parent.product_state_batch(U_train[y_train == 1]).astype(complex)
    legitimate = np.flatnonzero(y_train == 0)
    selected_validation = validation_subset(y_validation)

    rng = np.random.default_rng(seed)
    params = np.array(
        [*rng.normal(scale=0.15, size=2), rng.uniform(0.5, 2.5), rng.normal(scale=0.15)]
    )
    drawn_initial = params.copy()
    if mode == "frozen_zero":
        params[PHI_INDEX] = 0.0
    elif mode != "frozen_random":
        raise ValueError(mode)
    initial = params.copy()
    first_moment = np.zeros(4)
    second_moment = np.zeros(4)
    best = (-np.inf, params.copy(), -1)
    history = []

    for update in range(1, UPDATES + 1):
        batch = rng.choice(legitimate, BATCH_LEGIT, replace=False)
        legitimate_states = parent.product_state_batch(U_train[batch]).astype(complex)
        loss, gradient, mean_legitimate, mean_fraud = parent.pairwise_loss_and_grad(
            simulator, params, legitimate_states, fraud_states
        )
        if gradient.shape != (4,) or not np.all(np.isfinite(gradient)):
            raise RuntimeError(
                f"non-finite/full-gradient failure for seed {seed}, mode {mode}"
            )
        # The complete four-component gradient has been evaluated. Discarding
        # only this component preserves the original evaluation budget.
        gradient = gradient.copy()
        gradient[PHI_INDEX] = 0.0
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        params = params - 0.05 * (first_moment / (1.0 - 0.9**update)) / (
            np.sqrt(second_moment / (1.0 - 0.999**update)) + 1e-8
        )
        if abs(params[PHI_INDEX] - initial[PHI_INDEX]) > PHI_ATOL:
            raise RuntimeError(f"frozen phi moved for seed {seed}, mode {mode}")
        if update % CHECK_EVERY == 0:
            validation_score = -parent.ps_chunked(
                simulator, params, U_validation[selected_validation]
            )
            validation_ap = float(
                average_precision_score(
                    y_validation[selected_validation], validation_score
                )
            )
            history.append(
                {
                    "update": update,
                    "validation_auprc": validation_ap,
                    "training_loss": float(loss),
                    "mean_legitimate_ps": float(mean_legitimate),
                    "mean_fraud_ps": float(mean_fraud),
                }
            )
            if validation_ap > best[0]:
                best = (validation_ap, params.copy(), update)

    validation_ap, selected_params, checkpoint = best
    if checkpoint < 0 or abs(selected_params[PHI_INDEX] - initial[PHI_INDEX]) > PHI_ATOL:
        raise RuntimeError(f"no valid frozen checkpoint for seed {seed}, mode {mode}")
    test_score = -parent.ps_chunked(simulator, selected_params, U_test)
    if not np.all(np.isfinite(test_score)):
        raise RuntimeError(f"non-finite test score for seed {seed}, mode {mode}")
    return test_score, {
        "seed": seed,
        "mode": mode,
        "drawn_initial_params": drawn_initial.tolist(),
        "initial_params": initial.tolist(),
        "selected_params": selected_params.tolist(),
        "frozen_phi": float(initial[PHI_INDEX]),
        "phi_displacement": float(selected_params[PHI_INDEX] - initial[PHI_INDEX]),
        "best_checkpoint": int(checkpoint),
        "validation_auprc": float(validation_ap),
        "test_metrics": parent.ranking_metrics(pipe["ys"]["te"], test_score),
        "updates": UPDATES,
        "full_four_component_gradient_evaluated_each_update": True,
        "discarded_gradient_index": PHI_INDEX,
        "history": history,
    }


def _method_draws(
    y: np.ndarray, score_map: dict[str, np.ndarray], n_boot: int, seed: int
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    from sklearn.metrics import average_precision_score

    orders = {
        name: np.argsort(-np.asarray(score), kind="mergesort")
        for name, score in score_map.items()
    }
    groups = {}
    for name, order in orders.items():
        sorted_score = np.asarray(score_map[name])[order]
        groups[name] = np.flatnonzero(
            np.r_[sorted_score[1:] != sorted_score[:-1], True]
        )
    points = {
        name: float(average_precision_score(y, score))
        for name, score in score_map.items()
    }
    values: dict[str, list[float]] = {name: [] for name in score_map}
    rng = np.random.default_rng(seed)
    made = 0
    while made < n_boot:
        take = min(16, n_boot - made)
        weights = rng.poisson(1.0, size=(take, len(y))).astype(np.float32)
        valid = (weights[:, y == 1].sum(axis=1) > 0) & (
            weights[:, y == 0].sum(axis=1) > 0
        )
        weights = weights[valid]
        if len(weights) == 0:
            continue
        for name, order in orders.items():
            draws = parent._weighted_ap_sorted(y[order], weights[:, order], groups[name])
            values[name].extend(draws[np.isfinite(draws)].tolist())
        made = min(len(draws) for draws in values.values())
    return points, {
        name: np.asarray(draws[:n_boot], float) for name, draws in values.items()
    }


def summarize(
    y: np.ndarray,
    trained_scores: list[np.ndarray],
    frozen_random_scores: list[np.ndarray],
    frozen_zero_scores: list[np.ndarray],
    s1_score: np.ndarray,
) -> dict:
    score_map = {"S1": s1_score}
    score_map.update(
        {f"trained_{seed}": score for seed, score in zip(SEEDS, trained_scores)}
    )
    score_map.update(
        {
            f"frozen_random_{seed}": score
            for seed, score in zip(SEEDS, frozen_random_scores)
        }
    )
    score_map.update(
        {
            f"frozen_zero_{seed}": score
            for seed, score in zip(SEEDS, frozen_zero_scores)
        }
    )
    points, draws = _method_draws(y, score_map, N_BOOT, BOOT_SEED)
    trained_point = float(np.mean([points[f"trained_{seed}"] for seed in SEEDS]))
    frozen_random_point = float(
        np.mean([points[f"frozen_random_{seed}"] for seed in SEEDS])
    )
    frozen_zero_point = float(
        np.mean([points[f"frozen_zero_{seed}"] for seed in SEEDS])
    )
    trained_draw = np.mean(
        np.vstack([draws[f"trained_{seed}"] for seed in SEEDS]), axis=0
    )
    frozen_random_draw = np.mean(
        np.vstack([draws[f"frozen_random_{seed}"] for seed in SEEDS]), axis=0
    )
    frozen_zero_draw = np.mean(
        np.vstack([draws[f"frozen_zero_{seed}"] for seed in SEEDS]), axis=0
    )
    s1_draw = draws["S1"]
    training_delta = trained_draw - frozen_random_draw
    architecture_delta = frozen_random_draw - frozen_zero_draw
    filter_delta = frozen_zero_draw - s1_draw

    training_point = trained_point - frozen_random_point
    architecture_point = frozen_random_point - frozen_zero_point
    filter_point = frozen_zero_point - points["S1"]
    adjusted_quantiles = [0.0125, 0.9875]
    training_interval = [
        float(v) for v in np.quantile(training_delta, adjusted_quantiles)
    ]
    architecture_interval = [
        float(v) for v in np.quantile(architecture_delta, adjusted_quantiles)
    ]
    filter_interval = [float(v) for v in np.quantile(filter_delta, [0.025, 0.975])]
    training_identified = training_point > 0.0 and training_interval[0] > 0.0
    architecture_identified = (
        architecture_point > 0.0 and architecture_interval[0] > 0.0
    )

    return {
        "bootstrap": "paired Poisson(1) full-test-row weights",
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "co_primary_interval": (
            "two-sided 97.5% percentile, Bonferroni-equivalent familywise alpha 0.05"
        ),
        "descriptive_interval": "unadjusted two-sided 95% percentile",
        "test_rows": int(len(y)),
        "test_frauds": int(np.sum(y)),
        "point_auprc": {
            "S1": points["S1"],
            "trained_mixer_mean_of_seed_auprcs": trained_point,
            "frozen_random_mixer_mean_of_seed_auprcs": frozen_random_point,
            "frozen_zero_mixer_mean_of_seed_auprcs": frozen_zero_point,
        },
        "co_primary_trained_minus_frozen_random": {
            "point": training_point,
            "ci97_5": training_interval,
            "decision": (
                "MIXER_ANGLE_TRAINING_NECESSARY_AT_THIS_RESOLUTION"
                if training_identified
                else "MIXER_ANGLE_TRAINING_NOT_IDENTIFIED"
            ),
        },
        "co_primary_frozen_random_minus_frozen_zero": {
            "point": architecture_point,
            "ci97_5": architecture_interval,
            "decision": (
                "FROZEN_MIXER_ARCHITECTURE_EFFECT_IDENTIFIED"
                if architecture_identified
                else "FROZEN_MIXER_ARCHITECTURE_EFFECT_NOT_IDENTIFIED"
            ),
        },
        "descriptive_frozen_zero_minus_S1": {
            "point": filter_point,
            "ci95": filter_interval,
            "decision": "DESCRIPTIVE_ONLY",
        },
        "per_seed_point_auprc": {
            str(seed): {
                "trained": points[f"trained_{seed}"],
                "frozen_random": points[f"frozen_random_{seed}"],
                "frozen_zero": points[f"frozen_zero_{seed}"],
                "trained_minus_frozen_random": points[f"trained_{seed}"]
                - points[f"frozen_random_{seed}"],
                "frozen_random_minus_frozen_zero": points[f"frozen_random_{seed}"]
                - points[f"frozen_zero_{seed}"],
            }
            for seed in SEEDS
        },
    }


def publish_pair(
    output: Path, scores_output: Path, payload: dict, scores: dict[str, np.ndarray]
) -> None:
    output = output.resolve(strict=False)
    scores_output = scores_output.resolve(strict=False)
    allowed = (REPO_ROOT / "runs/hsbc_challenge/audit_report_v2").resolve()
    if output.parent != allowed or scores_output.parent != allowed:
        raise ValueError(f"outputs must be direct children of {allowed}")
    if output.exists() or scores_output.exists():
        raise FileExistsError("refusing to overwrite frozen-mixer output")
    allowed.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.time_ns()}"
    staged_npz = scores_output.with_name(f".{scores_output.stem}.{nonce}.tmp.npz")
    staged_json = output.with_name(f".{output.stem}.{nonce}.tmp.json")
    published_npz = False
    try:
        np.savez_compressed(staged_npz, **scores)
        with np.load(staged_npz, allow_pickle=False) as check:
            if set(check.files) != set(scores):
                raise RuntimeError("staged score key mismatch")
            for key, expected in scores.items():
                if not np.array_equal(check[key], np.asarray(expected), equal_nan=True):
                    raise RuntimeError(f"staged score mismatch: {key}")
        payload["scores_sha256"] = sha256(staged_npz)
        payload["scores_artifact"] = str(scores_output.relative_to(REPO_ROOT))
        staged_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(staged_json.read_text())["scores_sha256"] != sha256(staged_npz):
            raise RuntimeError("staged JSON/NPZ link mismatch")
        os.replace(staged_npz, scores_output)
        published_npz = True
        os.replace(staged_json, output)
    except Exception:
        staged_npz.unlink(missing_ok=True)
        staged_json.unlink(missing_ok=True)
        if published_npz and scores_output.exists() and not output.exists():
            scores_output.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/hsbc_challenge/audit_report_v2/frozen_mixer_v1.json",
    )
    parser.add_argument(
        "--scores-output",
        default="runs/hsbc_challenge/audit_report_v2/frozen_mixer_scores_v1.npz",
    )
    args = parser.parse_args()
    check_source_lock()

    output = (REPO_ROOT / args.output).resolve(strict=False)
    scores_output = (REPO_ROOT / args.scores_output).resolve(strict=False)
    pipe = parent.original_pipeline()
    y_test = np.asarray(pipe["ys"]["te"], int)

    with np.load(PARENT_SCORES, allow_pickle=False) as source:
        parent_y = np.asarray(source["c1_y"], int)
        if not np.array_equal(y_test, parent_y):
            raise RuntimeError("current full-test labels differ from parent score rows")
        s1_score = np.asarray(source["c1_s1"], float).copy()
        trained_scores = [
            np.asarray(source[f"c1_seed_{seed}"], float).copy() for seed in SEEDS
        ]

    frozen_random_scores = []
    frozen_zero_scores = []
    seed_rows = []
    for seed in SEEDS:
        rows_for_seed = {}
        for mode, collector in (
            ("frozen_random", frozen_random_scores),
            ("frozen_zero", frozen_zero_scores),
        ):
            score, row = train_frozen_seed(pipe, seed, mode)
            collector.append(score)
            rows_for_seed[mode] = row
            log(
                f"seed={seed} mode={mode} frozen_phi={row['frozen_phi']:+.6f} "
                f"checkpoint={row['best_checkpoint']} "
                f"test_AP={row['test_metrics']['auprc']:.6f}"
            )
        seed_rows.append({"seed": seed, "arms": rows_for_seed})

    summary = summarize(
        y_test, trained_scores, frozen_random_scores, frozen_zero_scores, s1_score
    )
    score_bundle = {"y_test": y_test, "S1": s1_score}
    score_bundle.update(
        {f"trained_{seed}": score for seed, score in zip(SEEDS, trained_scores)}
    )
    score_bundle.update(
        {
            f"frozen_random_{seed}": score
            for seed, score in zip(SEEDS, frozen_random_scores)
        }
    )
    score_bundle.update(
        {
            f"frozen_zero_{seed}": score
            for seed, score in zip(SEEDS, frozen_zero_scores)
        }
    )
    payload = {
        "schema": "hsbc-frozen-mixer-control-v1",
        "status": "COMPLETE",
        "protocol": str(PROTOCOL.relative_to(REPO_ROOT)),
        "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "runner_sha256": sha256(SCRIPT),
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): digest
            for path, digest in EXPECTED_HASHES.items()
        },
        "seed_rows": seed_rows,
        "summary": summary,
        "budget_match": {
            "updates_per_seed_both_arms": UPDATES,
            "full_four_component_gradient_evaluated_per_control_update": True,
            "discarded_phi_gradient": True,
            "validation_subset_and_checkpoint_cadence_match": True,
            "spare_evaluations_reassigned": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "wall_seconds": time.time() - T0,
    }
    publish_pair(output, scores_output, payload, score_bundle)
    print(json.dumps({"status": "COMPLETE", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
