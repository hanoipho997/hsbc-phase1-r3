"""Fail-closed hash and claim-boundary verifier for novelty dossier v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="runs/hsbc_challenge/novelty_v2/manifest_v2.json",
    )
    parser.add_argument("--output", help="optional fresh path for the verification JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    if manifest.get("schema") != "hsbc-claude-novelty-package-manifest-v2":
        raise AssertionError("unexpected manifest schema")

    rows = []
    for item in manifest["files"]:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != item["sha256"]:
            raise AssertionError(
                f"hash mismatch for {path}: {observed} != {item['sha256']}"
            )
        rows.append({"path": str(path), "sha256": observed, "status": "MATCH"})

    result_path = Path("runs/hsbc_challenge/novelty_v2/dephasing_uncertainty_v2.json")
    with result_path.open() as fh:
        result = json.load(fh)
    assert result["status"] == "POST_OUTCOME_CORRECTION_NOT_PREREGISTRATION"
    assert result["ideal_full_width"]["all_five_lower_bounds_above_zero_at_corrected_confidence"]
    assert result["analysis"]["multiplicity"]["family_size"] == 10
    assert result["analysis"]["row_bootstrap"].endswith("duplicates retained")
    assert result["saved_noise_count_sensitivity_seed900_only"]["low"]["status"] \
        == "EXPLORATORY_PLUGIN_SENSITIVITY_ONLY"
    high_interval = result["saved_noise_count_sensitivity_seed900_only"]["high"] \
        ["conditional_plugin_nested_bootstrap_97_5pct"]["interval"]
    assert high_interval[0] <= 0.0 <= high_interval[1]

    # The aggregate v2 artifact must not leak the data-bearing arrays or row
    # identifiers from the PRIVATE_LICENSED source NPZ.
    serialized = json.dumps(result)
    for forbidden in ("transaction_ids", "fraud_labels", '"U"', '"labels"'):
        if forbidden in serialized:
            raise AssertionError(f"aggregate artifact contains forbidden field: {forbidden}")

    summary = {
        "schema": "hsbc-claude-novelty-package-verification-v2",
        "manifest": str(manifest_path),
        "hash_rows": rows,
        "claim_boundary_checks": "PASS",
        "all_checks": "PASS",
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
