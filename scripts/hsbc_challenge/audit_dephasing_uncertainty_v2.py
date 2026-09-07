"""Post-outcome uncertainty audit for the HSBC partial-dephasing pilot.

This script does not alter the v1 experiment or call the result a new
preregistration.  It reads the frozen v1 probability/count artifacts and:

* keeps duplicate rows in the non-parametric row bootstrap;
* redraws multinomial shot counts inside every row-bootstrap replicate;
* uses 99.5% two-sided intervals for the intended 2 contrasts x 5 seeds
  family (Bonferroni familywise alpha 0.05);
* treats the saved low/high-noise counts as a plug-in sensitivity analysis,
  because exact noisy probabilities and four of five noise seeds were not
  saved; and
* reports the eight-row reduced instance as exploratory only.

No transaction identifiers, features, or labels are written to the output.
The source NPZ remains PRIVATE_LICENSED under the IEEE-CIS data terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    total = p.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("encountered an empty probability row")
    return p / total


def sample_counts(p: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    p = normalize_rows(p)
    return np.stack([rng.multinomial(shots, row) for row in p])


def unbiased_w(counts: np.ndarray) -> float:
    """Unbiased between-row diversity estimator used by the v1 protocol."""
    counts = np.asarray(counts, dtype=float)
    if counts.shape[0] < 2:
        raise ValueError("W requires at least two rows")
    shots = counts.sum(axis=1)
    if np.any(shots <= 1):
        raise ValueError("each row requires at least two shots")
    phat = counts / shots[:, None]
    qhat = (counts * (counts - 1.0)).sum(axis=1) / (shots * (shots - 1.0))
    distance = ((phat[:, None, :] - phat[None, :, :]) ** 2).sum(axis=-1)
    correction = (1.0 - qhat) / shots
    estimate = distance - correction[:, None] - correction[None, :]
    upper = np.triu_indices(counts.shape[0], 1)
    return float(estimate[upper].mean())


def exact_w(p: np.ndarray) -> float:
    p = normalize_rows(p)
    distance = ((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=-1)
    upper = np.triu_indices(p.shape[0], 1)
    return float(distance[upper].mean())


def percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    tail = 50.0 * (1.0 - confidence)
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))]


def nested_bootstrap_contrast(
    p0: np.ndarray,
    p1: np.ndarray,
    shots: int,
    replicates: int,
    rng: np.random.Generator,
    *,
    resample_rows: bool,
) -> np.ndarray:
    """Bootstrap W_hat(0)-W_hat(1), with independent shots in both arms."""
    p0, p1 = normalize_rows(p0), normalize_rows(p1)
    if p0.shape != p1.shape:
        raise ValueError(f"endpoint shapes differ: {p0.shape} != {p1.shape}")
    m = p0.shape[0]
    draws = np.empty(replicates, dtype=float)
    fixed = np.arange(m)
    for b in range(replicates):
        # Duplicates are intentionally retained: this is the ordinary
        # empirical row bootstrap specified in the frozen protocol.
        idx = rng.integers(0, m, size=m) if resample_rows else fixed
        c0 = sample_counts(p0[idx], shots, rng)
        c1 = sample_counts(p1[idx], shots, rng)
        draws[b] = unbiased_w(c0) - unbiased_w(c1)
    return draws


def analyze_exact_pair(
    p0: np.ndarray,
    p1: np.ndarray,
    shots: int,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict:
    rng = np.random.default_rng(seed)
    nested = nested_bootstrap_contrast(p0, p1, shots, replicates, rng, resample_rows=True)
    shot_only = nested_bootstrap_contrast(
        p0, p1, shots, replicates, np.random.default_rng(seed + 1_000_000),
        resample_rows=False,
    )
    point = exact_w(p0) - exact_w(p1)
    return {
        "estimand_exact_W0_minus_W1": point,
        "nested_row_and_shot_bootstrap": {
            "replicates": replicates,
            "confidence": confidence,
            "interval": percentile_interval(nested, confidence),
            "mean": float(nested.mean()),
            "duplicates_retained": True,
        },
        "shot_only_descriptive_97_5pct": {
            "interval": percentile_interval(shot_only, 0.975),
            "mean": float(shot_only.mean()),
        },
    }


def analyze_plugin_counts(
    counts0: np.ndarray,
    counts1: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    counts0, counts1 = np.asarray(counts0), np.asarray(counts1)
    shots0 = np.unique(counts0.sum(axis=1))
    shots1 = np.unique(counts1.sum(axis=1))
    if len(shots0) != 1 or len(shots1) != 1 or shots0[0] != shots1[0]:
        raise ValueError("saved endpoint rows do not have one common shot count")
    shots = int(shots0[0])
    phat0 = normalize_rows(counts0)
    phat1 = normalize_rows(counts1)
    draws = nested_bootstrap_contrast(
        phat0, phat1, shots, replicates, np.random.default_rng(seed),
        resample_rows=True,
    )
    return {
        "observed_W_hat0_minus_W_hat1": unbiased_w(counts0) - unbiased_w(counts1),
        "conditional_plugin_nested_bootstrap_97_5pct": {
            "replicates": replicates,
            "interval": percentile_interval(draws, 0.975),
            "mean": float(draws.mean()),
            "duplicates_retained": True,
        },
        "shots_per_row_per_endpoint": shots,
        "status": "EXPLORATORY_PLUGIN_SENSITIVITY_ONLY",
        "limitation": (
            "The v1 artifact saved one realized count table, not exact noisy "
            "probabilities. Multinomial draws from the observed frequencies are "
            "conditional plug-in sensitivity and cannot repair the missing four "
            "noise seeds or establish a confirmatory device-noise result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curves",
        default="runs/hsbc_challenge/novelty_v1/dephasing_witness_curves_v1.npz",
    )
    parser.add_argument(
        "--reduced",
        default="runs/hsbc_challenge/novelty_v1/braket_reduced_instance_v1.npz",
    )
    parser.add_argument("--outdir", default="runs/hsbc_challenge/novelty_v2")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--shots", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves_path = Path(args.curves)
    reduced_path = Path(args.reduced)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "dephasing_uncertainty_v2.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if args.replicates < 1_000:
        raise ValueError("use at least 1,000 bootstrap replicates")

    curves = np.load(curves_path, allow_pickle=False)
    reduced = np.load(reduced_path, allow_pickle=False)
    seeds = sorted(
        int(k.removeprefix("P_q_seed"))
        for k in curves.files if k.startswith("P_q_seed")
    )
    if seeds != [900, 901, 902, 903, 904]:
        raise ValueError(f"unexpected model seed family: {seeds}")

    # Frozen v1 intended two primary contrasts across five separately analyzed
    # seeds. Bonferroni alpha=0.05/10 gives 99.5% marginal intervals.
    family_size = 10
    familywise_alpha = 0.05
    confidence = 1.0 - familywise_alpha / family_size
    ideal = {}
    for offset, model_seed in enumerate(seeds):
        p = curves[f"P_q_seed{model_seed}"]
        ideal[str(model_seed)] = analyze_exact_pair(
            p[0], p[-1], args.shots, args.replicates,
            args.seed + 10_000 * offset, confidence,
        )

    noise = {}
    for offset, level in enumerate(("low", "high")):
        key = f"noise_{level}_counts"
        if key not in curves.files:
            noise[level] = {"status": "NOT_SAVED_IN_V1_ARTIFACT"}
            continue
        counts = curves[key]
        noise[level] = analyze_plugin_counts(
            counts[0], counts[-1], args.replicates,
            args.seed + 200_000 + offset * 10_000,
        )

    p_reduced = reduced["P_exact"]
    reduced_result = analyze_exact_pair(
        p_reduced[0], p_reduced[-1], args.shots, args.replicates,
        args.seed + 400_000, 0.95,
    )
    reduced_result["status"] = "EXPLORATORY_POST_OUTCOME_DESIGN_CANDIDATE"
    reduced_result["limitations"] = [
        "Eight validation-derived rows were selected after observing the full-width run.",
        "The circuit parameters are a single random seed, not a trained Engine-A model.",
        "No QPU, managed-simulator, native-device compile, or device-calibrated power analysis was run.",
    ]

    all_ideal_pass = all(
        rec["nested_row_and_shot_bootstrap"]["interval"][0] > 0
        for rec in ideal.values()
    )
    payload = {
        "schema": "hsbc-dephasing-uncertainty-audit-v2",
        "status": "POST_OUTCOME_CORRECTION_NOT_PREREGISTRATION",
        "inputs": {
            "curves": {"path": str(curves_path), "sha256": sha256_file(curves_path)},
            "reduced": {"path": str(reduced_path), "sha256": sha256_file(reduced_path)},
            "classification": "PRIVATE_LICENSED_IEEE_CIS_DERIVED",
        },
        "analysis": {
            "shots_per_row_per_endpoint": args.shots,
            "bootstrap_replicates": args.replicates,
            "rng_seed": args.seed,
            "row_bootstrap": "m rows sampled with replacement; duplicates retained",
            "shot_bootstrap": "independent multinomial redraw inside every endpoint and row replicate",
            "multiplicity": {
                "intended_family": "2 primary contrasts x 5 model seeds",
                "family_size": family_size,
                "familywise_alpha": familywise_alpha,
                "method": "Bonferroni",
                "per_interval_confidence": confidence,
            },
        },
        "ideal_full_width": {
            "seeds": ideal,
            "all_five_lower_bounds_above_zero_at_corrected_confidence": all_ideal_pass,
            "verdict": "RESOLVABLE_IN_IDEAL_SIMULATION" if all_ideal_pass else "NOT_RESOLVABLE",
        },
        "saved_noise_count_sensitivity_seed900_only": noise,
        "reduced_instance": reduced_result,
        "claim_boundary": (
            "The corrected analysis supports ideal-simulator resolvability of "
            "between-row syndrome variation. It does not support a QPU result, "
            "a device-noise family claim, fraud lift, runtime advantage, or "
            "classical hardness."
        ),
    }
    with output_path.open("x") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "all_ideal_pass": all_ideal_pass,
        "ideal_lower_bounds": {
            seed: rec["nested_row_and_shot_bootstrap"]["interval"][0]
            for seed, rec in ideal.items()
        },
        "noise_plugin_intervals": {
            k: v.get("conditional_plugin_nested_bootstrap_97_5pct", {}).get("interval")
            for k, v in noise.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
