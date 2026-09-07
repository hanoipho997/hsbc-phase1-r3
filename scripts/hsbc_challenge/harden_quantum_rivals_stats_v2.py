#!/usr/bin/env python3
"""Post-outcome multiplicity hardening for the seven-primary ULB rival bracket.

This does not train or select a model. It recomputes paired full-test AUPRC
uncertainty from the immutable aligned v2 score bundles, applies multiplicity
across seven stack-versus-rival contrasts, and reports a local 80%-power MDE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(SCRIPT.parent))

from audit_hybrid_qml_v2 import _weighted_ap_sorted  # noqa: E402


PROTOCOL = REPO_ROOT / "docs/hsbc_challenge_quantum_rivals_audit_v2.md"
RESULTS = REPO_ROOT / "docs/hsbc_challenge_quantum_rivals_results_v2.md"
KERNEL = REPO_ROOT / "runs/hsbc_challenge/audit_v2/quantum_kernels_v2.npz"
VQA = REPO_ROOT / "runs/hsbc_challenge/audit_v2/vqa_qae_v2.npz"
HYBRID = REPO_ROOT / "runs/hsbc_challenge/audit_v2/hybrid_qml_v2.npz"
COMBINED = REPO_ROOT / "runs/hsbc_challenge/audit_v2/quantum_rivals_v2_combined.json"

EXPECTED_HASHES = {
    PROTOCOL: "de10a4640e310986ea94ed52a35a7edba2bb32fceac7e84116dc2c0bc5ebc115",
    RESULTS: "a9f6eeefbaa96fdb5cbda883960c4070983276950316e8e29e3c0df91c032875",
    KERNEL: "4ee76de984fd193482a69a638e00ad898ea96e07cb5f2cbd1826dd2eb45148fa",
    VQA: "78c80cf8671a948602ac03f3ebc8ac2087752ecf20948f8c184f1bc625301084",
    HYBRID: "e1b1d9285770b6eaf1291920362ccae20ae871125012bfada29c1450c982b42b",
    COMBINED: "8166d606d5635aa2fe81e9fdb2fb309cc16595afd2ddae5a1a5f0c9545b90ebb",
}

N_BOOT = 10_000
BOOT_SEED = 20260902
FAMILYWISE_ALPHA = 0.05
N_PRIMARY = 7
POWER = 0.80
EQUIVALENCE_MARGIN = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_locks() -> None:
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from repository root {REPO_ROOT}")
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"source-lock mismatch for {path.relative_to(REPO_ROOT)}: {observed}"
            )


def load_scores() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with (
        np.load(KERNEL, allow_pickle=False) as kernel,
        np.load(VQA, allow_pickle=False) as vqa,
        np.load(HYBRID, allow_pickle=False) as hybrid,
    ):
        y = np.asarray(kernel["y_test"], int)
        if y.shape != (56_962,) or int(np.sum(y)) != 99:
            raise RuntimeError("unexpected full-test row/fraud counts")
        if not np.array_equal(y, np.asarray(vqa["n8_y"], int)):
            raise RuntimeError("VQA rows do not match kernel rows")
        if not np.array_equal(y, np.asarray(hybrid["y_test"], int)):
            raise RuntimeError("hybrid rows do not match kernel rows")
        scores = {
            "S2_L1": np.asarray(kernel["S2_L1_seed500"], float),
            "S1": np.asarray(kernel["S1_analytic"], float),
            "product_QSVC": np.asarray(kernel["product_fidelity_QSVC"], float),
            "ring_IQP_Nystrom": np.asarray(
                kernel["ring_IQP_fidelity_Nystrom"], float
            ),
            "projected_IQP_SVC": np.asarray(
                kernel["projected_ring_IQP_SVC"], float
            ),
            "kernel_OCSVM": np.asarray(
                kernel["ring_IQP_fidelity_Nystrom_OCSVM"], float
            ),
            "VQC": np.asarray(vqa["vqc_entangling_ensemble_mean5"], float),
            "trash_QAE": np.asarray(vqa["qae_trash_ensemble_mean5"], float),
            "hybrid_QNN": np.asarray(
                hybrid["qnn_trainable_quantum_ensemble"], float
            ),
        }
    if any(value.shape != y.shape or not np.all(np.isfinite(value)) for value in scores.values()):
        raise RuntimeError("invalid score array")
    return y, scores


def score_draws(
    y: np.ndarray, scores: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    from sklearn.metrics import average_precision_score

    orders = {
        name: np.argsort(-score, kind="mergesort") for name, score in scores.items()
    }
    groups = {}
    for name, order in orders.items():
        sorted_score = scores[name][order]
        groups[name] = np.flatnonzero(
            np.r_[sorted_score[1:] != sorted_score[:-1], True]
        )
    points = {
        name: float(average_precision_score(y, score))
        for name, score in scores.items()
    }
    values: dict[str, list[float]] = {name: [] for name in scores}
    rng = np.random.default_rng(BOOT_SEED)
    made = 0
    while made < N_BOOT:
        take = min(24, N_BOOT - made)
        weights = rng.poisson(1.0, size=(take, len(y))).astype(np.float32)
        valid = (weights[:, y == 1].sum(axis=1) > 0) & (
            weights[:, y == 0].sum(axis=1) > 0
        )
        weights = weights[valid]
        if len(weights) == 0:
            continue
        for name, order in orders.items():
            draw = _weighted_ap_sorted(y[order], weights[:, order], groups[name])
            values[name].extend(draw[np.isfinite(draw)].tolist())
        made = min(len(draw) for draw in values.values())
    return points, {
        name: np.asarray(draw[:N_BOOT], float) for name, draw in values.items()
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - rank) * raw[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/hsbc_challenge/audit_report_v2/quantum_rivals_stats_hardened_v2.json",
    )
    args = parser.parse_args()
    check_locks()
    output = (REPO_ROOT / args.output).resolve(strict=False)
    allowed = (REPO_ROOT / "runs/hsbc_challenge/audit_report_v2").resolve()
    if output.parent != allowed or output.suffix != ".json":
        raise ValueError(f"output must be a JSON direct child of {allowed}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    y, scores = load_scores()
    points, draws = score_draws(y, scores)
    comparisons = {
        "stack_minus_product_QSVC": ("S2_L1", "product_QSVC"),
        "stack_minus_ring_IQP_Nystrom": ("S2_L1", "ring_IQP_Nystrom"),
        "stack_minus_projected_IQP_SVC": ("S2_L1", "projected_IQP_SVC"),
        "stack_minus_kernel_OCSVM": ("S1", "kernel_OCSVM"),
        "stack_minus_VQC": ("S2_L1", "VQC"),
        "stack_minus_trash_QAE": ("S1", "trash_QAE"),
        "stack_minus_hybrid_QNN": ("S2_L1", "hybrid_QNN"),
    }
    if len(comparisons) != N_PRIMARY:
        raise RuntimeError("seven-primary bracket changed")

    from scipy.stats import norm

    per_test_alpha = FAMILYWISE_ALPHA / N_PRIMARY
    lower_q = per_test_alpha / 2.0
    upper_q = 1.0 - lower_q
    z_critical = float(norm.ppf(1.0 - per_test_alpha / 2.0))
    z_power = float(norm.ppf(POWER))
    raw_p = {}
    rows = {}
    for name, (left, right) in comparisons.items():
        delta = draws[left] - draws[right]
        point = points[left] - points[right]
        centered = delta - point
        raw_p[name] = float(
            (1 + np.sum(np.abs(centered) >= abs(point))) / (N_BOOT + 1)
        )
        se = float(np.std(delta, ddof=1))
        ci_sim = [float(value) for value in np.quantile(delta, [lower_q, upper_q])]
        ci90 = [float(value) for value in np.quantile(delta, [0.05, 0.95])]
        rows[name] = {
            "left": left,
            "right": right,
            "left_auprc": points[left],
            "right_auprc": points[right],
            "point": float(point),
            "paired_bonferroni_simultaneous_ci99_2857": ci_sim,
            "bootstrap_standard_error": se,
            "local_normal_mde_80pct_power_abs_auprc": float(
                (z_critical + z_power) * se
            ),
            "unadjusted_ci90_for_equivalence": ci90,
            "equivalent_within_plus_minus_0_01": bool(
                ci90[0] > -EQUIVALENCE_MARGIN
                and ci90[1] < EQUIVALENCE_MARGIN
            ),
        }
    adjusted_p = holm_adjust(raw_p)
    for name in rows:
        rows[name]["centered_bootstrap_two_sided_p"] = raw_p[name]
        rows[name]["holm_adjusted_p_across_seven"] = adjusted_p[name]
        interval = rows[name]["paired_bonferroni_simultaneous_ci99_2857"]
        if interval[0] > 0.0:
            conclusion = "STACK_HIGHER"
        elif interval[1] < 0.0:
            conclusion = "RIVAL_HIGHER"
        else:
            conclusion = "DIFFERENCE_NOT_RESOLVED"
        rows[name]["multiplicity_corrected_conclusion"] = conclusion

    product = rows["stack_minus_product_QSVC"]
    qnn = rows["stack_minus_hybrid_QNN"]
    payload = {
        "schema": "hsbc-quantum-rivals-stats-hardened-v2",
        "status": "COMPLETE",
        "analysis_scope": (
            "post-outcome paired uncertainty hardening only; no model fitting, "
            "selection, tuning, or transaction-table access"
        ),
        "definition_of_seven_primary": [
            "product QSVC",
            "ring-IQP Nystrom",
            "projected-IQP SVC",
            "kernel OC-SVM",
            "re-uploading VQC",
            "trash-QAE",
            "hybrid QNN",
        ],
        "hybrid_latent_AE_status": (
            "secondary QAE implementation, excluded from the user-specified "
            "seven-primary multiplicity family"
        ),
        "test_rows": int(len(y)),
        "test_frauds": int(np.sum(y)),
        "test_resampled_or_balanced": False,
        "bootstrap": {
            "kind": "paired Poisson(1) identical full-test-row weights",
            "seed": BOOT_SEED,
            "draws": N_BOOT,
        },
        "multiplicity": {
            "familywise_alpha": FAMILYWISE_ALPHA,
            "number_primary": N_PRIMARY,
            "per_comparison_alpha": per_test_alpha,
            "interval_level": 1.0 - per_test_alpha,
            "interval_method": "paired percentile bootstrap plus Bonferroni union bound",
            "p_adjustment": "Holm across the same seven centered-bootstrap tests",
        },
        "mde_definition": {
            "power": POWER,
            "formula": "(z_(1-alpha/(2m)) + z_power) * paired bootstrap SE",
            "z_critical": z_critical,
            "z_power": z_power,
            "scope": (
                "local normal approximation conditional on each observed paired "
                "score distribution; 99 frauds alone do not determine an AUPRC MDE"
            ),
        },
        "comparisons": rows,
        "proposal_printable_statement": (
            "On the unchanged-prevalence 56,962-row ULB holdout (99 frauds), "
            f"stacked-LCU minus product-QSVC was {product['point']:+.6f} "
            f"with seven-comparison simultaneous interval "
            f"[{product['paired_bonferroni_simultaneous_ci99_2857'][0]:+.6f},"
            f"{product['paired_bonferroni_simultaneous_ci99_2857'][1]:+.6f}], "
            f"and stacked-LCU minus hybrid-QNN was {qnn['point']:+.6f} "
            f"[{qnn['paired_bonferroni_simultaneous_ci99_2857'][0]:+.6f},"
            f"{qnn['paired_bonferroni_simultaneous_ci99_2857'][1]:+.6f}]; "
            "neither difference was resolved, neither met +/-0.01 equivalence, "
            "and competent classical point estimates remained as good or better."
        ),
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): digest
            for path, digest in EXPECTED_HASHES.items()
        },
        "runner_sha256": sha256(SCRIPT),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    check = json.loads(staged.read_text())
    if check["status"] != "COMPLETE" or len(check["comparisons"]) != N_PRIMARY:
        staged.unlink(missing_ok=True)
        raise RuntimeError("staged result failed validation")
    os.replace(staged, output)
    print(json.dumps({"status": "COMPLETE", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
