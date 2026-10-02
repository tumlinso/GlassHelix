#!/usr/bin/env python3
"""Check original-source identities and immutable GH seed adoption; no compute."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root: Path) -> dict:
    original = root / "planning/moonshot-v1"
    experiment = root / "experiments/moonshot-parallel-v1"
    manifest = json.loads((original / "MANIFEST.json").read_text())["files"]
    # Validate every package file to retain the original intent and provenance.
    for relative, expected in manifest.items():
        if sha256(original / relative) != expected:
            raise ValueError(f"original package hash mismatch: {relative}")
    seeds = experiment / "seeds"
    record = json.loads((seeds / "manifest.json").read_text())
    expected_sources = {f"payload/glasshelix/{name}" for name in
                        ("actor_fields.py", "learn_toy.py", "smoke.py")}
    if {entry["source"] for entry in record["files"].values()} != expected_sources:
        raise ValueError("incomplete or unexpected GH seed source inventory")
    for relative, entry in record["files"].items():
        path = seeds / relative
        if not path.resolve().is_relative_to(seeds.resolve()):
            raise ValueError(f"seed escapes inventory scope: {relative}")
        digest = sha256(path)
        if digest != entry["sha256"] or digest != manifest[entry["source"]]:
            raise ValueError(f"immutable seed mismatch: {relative}")
        ast.parse(path.read_text(), filename=str(path))
    contract = experiment / "interfaces/model-contract.md"
    if not contract.is_file():
        raise ValueError("missing shared model contract")
    agreement = contract.read_text()
    for required in ("MatrixPatchField", "SylvesterFlow", "ActorPortField",
                     "RegrowthBranch", "GH-ML2-REPAIR", "GH-ML2-ACCEPT",
                     "Whole joint", "supplied", "private"):
        if required not in agreement:
            raise ValueError(f"missing adoption agreement term: {required}")
    return {"record_kind": "gh_moonshot_adoption_validation", "schema_version": 1,
            "status": "passed", "original_files_verified": len(manifest),
            "seed_files_verified": len(record["files"]),
            "source_commit_at_adoption": record["glasshelix_source_commit"],
            "contract_sha256": sha256(contract),
            "compute": "not_run", "native_integration": "not_run"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.root.resolve())
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
