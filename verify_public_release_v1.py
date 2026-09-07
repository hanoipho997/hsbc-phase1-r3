"""Verify a built HSBC public release from the release-tree root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "PUBLIC_RELEASE_MANIFEST.json"
FORBIDDEN_SUFFIXES = {
    ".npz", ".npy", ".joblib", ".parquet", ".csv", ".pkl", ".pickle",
    ".model", ".aux", ".log", ".out", ".toc",
}
PRIVATE_CLASSIFICATIONS = {
    "PRIVATE_LICENSED_IEEE_CIS_DERIVED",
    "PRIVATE_LICENSED_INPUTS_AGGREGATE_OUTPUT",
}
PRIVATE_PROVENANCE_KEYS = {
    "sources",
    "source_data",
    "source_sha256",
    "artifacts_sha256",
    "row_token_sha256_prefix",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_json(value: object, *, location: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            audit_json(item, location=f"{location}[{index}]")
        return
    if not isinstance(value, dict):
        if isinstance(value, str) and value in PRIVATE_CLASSIFICATIONS:
            raise AssertionError(f"private classification in {location}")
        return
    for key, item in value.items():
        if key in PRIVATE_PROVENANCE_KEYS:
            raise AssertionError(f"private provenance key in {location}: {key}")
        candidates = [key]
        if isinstance(item, str):
            candidates.append(item)
        if any(Path(candidate).suffix.lower() in FORBIDDEN_SUFFIXES for candidate in candidates):
            raise AssertionError(f"forbidden payload reference in {location}: {key}")
        audit_json(item, location=f"{location}.{key}")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["schema"] != "hsbc-revision3-public-release-manifest-v1":
        raise AssertionError("unexpected manifest schema")
    if manifest["private_monorepo_history_included"]:
        raise AssertionError("private monorepo history marked included")
    if manifest["private_or_row_level_files_included"]:
        raise AssertionError("private/row-level files marked included")

    expected = {row["path"]: row for row in manifest["files"]}
    actual = {
        str(path.relative_to(ROOT)): path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
        and path.name != "PUBLIC_RELEASE_MANIFEST.json"
    }
    if set(actual) != set(expected):
        raise AssertionError({
            "missing": sorted(set(expected) - set(actual)),
            "unexpected": sorted(set(actual) - set(expected)),
        })
    for relative, path in actual.items():
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise AssertionError(f"forbidden suffix: {relative}")
        row = expected[relative]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise AssertionError(f"hash/size mismatch: {relative}")
        if relative.startswith("runs/") and path.suffix.lower() == ".json":
            audit_json(json.loads(path.read_text()), location=relative)
    print(json.dumps({
        "status": "PASS",
        "verified_files": len(actual),
        "manifest_sha256": sha256(MANIFEST),
        "private_or_row_level_files_included": False,
    }))


if __name__ == "__main__":
    main()
