"""Run an unchanged Claude novelty-v1 runner in an isolated filesystem stage.

The v1 runners intentionally hard-code runs/hsbc_challenge/novelty_v1 and
refuse to overwrite it.  This launcher preserves those historical sources and
published artifacts.  It creates a fresh working directory, symlinks only the
read-only repository inputs required by the runners, and captures new outputs
under the stage's own runs/hsbc_challenge/novelty_v1 directory.

Example:
  .venv/bin/python scripts/hsbc_challenge/reproduce_claude_novelty_isolated_v2.py \
      --runner gradient --stage /private/tmp/hsbc-gradient-reproduction-v2

Arguments after ``--`` are forwarded to the selected runner.  For example,
``--runner dephasing ... -- --skip-noise`` performs the fast exact-stage run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


RUNNERS = {
    "dephasing": "novelty_dephasing_witness_v1.py",
    "gradient": "novelty_gradient_recycling_v1.py",
    "compile": "novelty_native_compile_v1.py",
    "compile-b": "novelty_native_compile_v1b.py",
    "splice": "novelty_splice_ablation_v1.py",
    "drift": "novelty_syndrome_drift_v1.py",
    "braket-local": "novelty_braket_local_dm_v1.py",
}

EXPECTED_OUTPUTS = {
    "dephasing": [
        "dephasing_witness_v1.json",
        "dephasing_witness_curves_v1.npz",
        "braket_reduced_instance_v1.npz",
    ],
    "gradient": ["gradient_recycling_v1.json"],
    "compile": ["native_compile_v1.json"],
    "compile-b": ["native_compile_v1b.json"],
    "splice": ["splice_ablation_v1.json"],
    "drift": ["syndrome_drift_v1.json"],
    "braket-local": ["braket_local_dm_v1.json"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strip_runtime_fields(value):
    if isinstance(value, dict):
        return {
            key: strip_runtime_fields(item)
            for key, item in value.items()
            if key != "wall_seconds"
        }
    if isinstance(value, list):
        return [strip_runtime_fields(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--stage", required=True, help="fresh output/staging directory")
    parser.add_argument(
        "--braket-input",
        default="runs/hsbc_challenge/novelty_v1/braket_reduced_instance_v1.npz",
        help="input NPZ used only by --runner braket-local",
    )
    parser.add_argument(
        "--record-copy",
        help="optional second path for the aggregate reproduction record",
    )
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.runner_args[:1] == ["--"]:
        args.runner_args = args.runner_args[1:]
    return args


def link(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    stage = Path(args.stage).expanduser().resolve()
    if stage == repo or repo in stage.parents:
        raise ValueError("the isolated stage must be outside the repository")
    if stage.exists() and any(stage.iterdir()):
        raise FileExistsError(f"refusing to use non-empty stage: {stage}")
    stage.mkdir(parents=True, exist_ok=True)

    link(repo / "docs", stage / "docs")
    staged_runs = stage / "runs" / "hsbc_challenge"
    staged_runs.mkdir(parents=True, exist_ok=True)
    for name in ("data", "engine_a_v1", "fit_v0", "audit_v1", "audit_v2"):
        link(repo / "runs" / "hsbc_challenge" / name, staged_runs / name)

    fresh_output = staged_runs / "novelty_v1"
    fresh_output.mkdir(parents=True, exist_ok=False)
    if args.runner == "braket-local":
        source = (repo / args.braket_input).resolve() if not Path(args.braket_input).is_absolute() else Path(args.braket_input)
        link(source, fresh_output / "braket_reduced_instance_v1.npz")

    runner = repo / "scripts" / "hsbc_challenge" / RUNNERS[args.runner]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(repo / "scripts" / "hsbc_challenge"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command = [sys.executable, str(runner), *args.runner_args]
    completed = subprocess.run(command, cwd=stage, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    missing = [name for name in EXPECTED_OUTPUTS[args.runner] if not (fresh_output / name).exists()]
    if missing:
        raise RuntimeError(f"runner completed but expected outputs are missing: {missing}")
    comparisons = {}
    published = repo / "runs" / "hsbc_challenge" / "novelty_v1"
    for name in EXPECTED_OUTPUTS[args.runner]:
        reproduced = fresh_output / name
        reference = published / name
        rec = {"reproduced_sha256": sha256_file(reproduced)}
        if reference.exists():
            rec["reference_sha256"] = sha256_file(reference)
            rec["byte_identical"] = rec["reproduced_sha256"] == rec["reference_sha256"]
            if reproduced.suffix == ".json":
                with reproduced.open() as fh:
                    reproduced_json = strip_runtime_fields(json.load(fh))
                with reference.open() as fh:
                    reference_json = strip_runtime_fields(json.load(fh))
                rec["semantic_exact_equal_excluding_wall_seconds"] = reproduced_json == reference_json
        comparisons[name] = rec

    record = {
        "schema": "hsbc-claude-novelty-isolated-reproduction-v2",
        "runner": args.runner,
        "source": str(runner),
        "source_unchanged": True,
        "command": command,
        "stage": str(stage),
        "outputs": [str(fresh_output / name) for name in EXPECTED_OUTPUTS[args.runner]],
        "comparisons_to_published_v1": comparisons,
    }
    record_path = stage / "reproduction_record_v2.json"
    with record_path.open("x") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    if args.record_copy:
        record_copy = Path(args.record_copy).expanduser().resolve()
        record_copy.parent.mkdir(parents=True, exist_ok=True)
        with record_copy.open("x") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
