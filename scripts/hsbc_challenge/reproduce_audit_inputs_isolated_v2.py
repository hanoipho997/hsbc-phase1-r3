#!/usr/bin/env python3
"""Reproduce HSBC fit-v0 and mechanism inputs in a fresh isolated directory.

The legacy runners write to repository-relative paths.  This orchestrator
gives them a fresh working directory, links only immutable inputs, and refuses
to touch an existing output root.  It never invokes the Engine-A unblind
runner or loads the sealed IEEE-CIS test table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
RUNNER_ROOT = REPO_ROOT / "scripts/hsbc_challenge"
SOURCE_RUNS = REPO_ROOT / "runs/hsbc_challenge"

FIT_RUNNERS = (
    "exp_b_scorer.py",
    "exp_b2_addendum.py",
    "exp_c_dequantization.py",
    "exp_d_alignment.py",
    "exp_e_depth.py",
    "exp_f_bottlenecks.py",
)
FIT_OUTPUTS = tuple(name.replace(".py", ".json") for name in FIT_RUNNERS)
MECHANISM_RUNNERS = ("verify_crossview_v1.py", "verify_syndrome_v1.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_timing_key(key: str) -> bool:
    return key == "wall_seconds" or "_us_per_" in key


def compare(left, right, path="$", differences=None):
    if differences is None:
        differences = []
    if type(left) is not type(right):
        differences.append({"path": path, "reason": "type"})
    elif isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if is_timing_key(key):
                continue
            if key not in left or key not in right:
                differences.append({"path": f"{path}.{key}", "reason": "missing"})
            else:
                compare(left[key], right[key], f"{path}.{key}", differences)
    elif isinstance(left, list):
        if len(left) != len(right):
            differences.append({"path": path, "reason": "length"})
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]", differences)
    elif isinstance(left, float):
        if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-14):
            differences.append(
                {"path": path, "reason": "float", "source": left, "reproduced": right}
            )
    elif left != right:
        differences.append(
            {"path": path, "reason": "value", "source": left, "reproduced": right}
        )
    return differences


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source.resolve(), destination, target_is_directory=source.is_dir())


def execute(runner: str, output_root: Path) -> None:
    subprocess.run(
        [sys.executable, str(RUNNER_ROOT / runner)],
        cwd=output_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("fit", "mechanisms", "all"), default="all"
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError(f"refusing existing output root: {output_root}")
    output_root.mkdir(parents=True)

    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from repository root {REPO_ROOT}")
    link(
        SOURCE_RUNS / "data/ulb_creditcard.npz",
        output_root / "runs/hsbc_challenge/data/ulb_creditcard.npz",
    )

    run_fit = args.scope in ("fit", "all")
    run_mechanisms = args.scope in ("mechanisms", "all")
    if run_fit:
        for runner in FIT_RUNNERS:
            execute(runner, output_root)
    elif run_mechanisms:
        link(SOURCE_RUNS / "fit_v0", output_root / "runs/hsbc_challenge/fit_v0")

    if run_mechanisms:
        (output_root / "runs/hsbc_challenge/engine_a_v1").mkdir(
            parents=True, exist_ok=True
        )
        for runner in MECHANISM_RUNNERS:
            execute(runner, output_root)

    rows = {}
    if run_fit:
        for name in FIT_OUTPUTS:
            source = SOURCE_RUNS / "fit_v0" / name
            reproduced = output_root / "runs/hsbc_challenge/fit_v0" / name
            differences = compare(json.loads(source.read_text()), json.loads(reproduced.read_text()))
            rows[f"fit_v0/{name}"] = {
                "status": "MATCH" if not differences else "MISMATCH",
                "source_sha256": sha256(source),
                "reproduced_sha256": sha256(reproduced),
                "differences": differences,
            }

    if run_mechanisms:
        for name in ("verify_crossview_v1.json", "verify_syndrome_v1.json"):
            source = SOURCE_RUNS / "engine_a_v1" / name
            reproduced = output_root / "runs/hsbc_challenge/engine_a_v1" / name
            differences = compare(json.loads(source.read_text()), json.loads(reproduced.read_text()))
            rows[f"engine_a_v1/{name}"] = {
                "status": "MATCH" if not differences else "MISMATCH",
                "comparison": "recursive; wall-clock and per-row timing fields excluded",
                "source_sha256": sha256(source),
                "reproduced_sha256": sha256(reproduced),
                "differences": differences,
            }

    result = {
        "schema": "hsbc-audit-isolated-reproduction-v2",
        "status": "MATCH" if all(row["status"] == "MATCH" for row in rows.values()) else "MISMATCH",
        "scope": args.scope,
        "repo_root": str(REPO_ROOT),
        "isolated_output_root": str(output_root),
        "sealed_engine_a_test_loaded": False,
        "runner_hashes": {
            name: sha256(RUNNER_ROOT / name)
            for name in (*FIT_RUNNERS, *MECHANISM_RUNNERS)
            if (run_fit and name in FIT_RUNNERS) or (run_mechanisms and name in MECHANISM_RUNNERS)
        },
        "rows": rows,
    }
    report = output_root / "audit_reproduction_comparison_v2.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "MATCH":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
