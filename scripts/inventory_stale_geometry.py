#!/usr/bin/env python3
"""Hash and classify encountered stale geometry/scientific artifacts.

The scanner is intentionally allowlisted by root.  It never traverses the
authoritative WISTELL-D source root or any Prompt-01-v3 lane artifact root,
and it performs no move or deletion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


AUTHORITATIVE_HASHES = {
    "9231969001203a8133255ee0a275bf552b114cc12524dda0608ab2f12047f7ac",
    "7748369407d28a70f35b5c4a7c0ab860495a08fd0030002112ea933fe570159b",
    "56baa090d61b67273efba61213849b7516beabb2a57fc2ad4751a6f3a32b2db4",
    "fdb85b2c0c8cd72f5d000302e0b67349ebf72679f98f9c4d7739e5d8484cdde3",
    "3579e5d8fe97dd74c8700e5676964159f00f07989ca6436528f60462889f05bd",
    "65264e15669d09c43f107c3b43c2af24ffbd15173e3bbd0e990b527bfa0b5322",
    "0ed18ab58bcc1e9884bf1b5c8bf19a7b7558ce7afe1869f1a2b01710148af6df",
}

KNOWN_GENERIC_EXAMPLE_HASHES = {
    "1cebb8d46e60d77df4a6904662a9c9f943137a9fb59f7290e5309af15fa04797",
    "69f508b216f0b674368ca8731d390c9d514736ff092f0ebecd854e8772ae04ab",
    "83902138eccefc266df638cf8662e00eb8a11cb7c66848d80c0d9e19805383e5",
}

PROTECTED_ROOTS = (
    Path(r"D:\Scratch\stellarator_optimization\wistell-d_data"),
    Path(
        r"D:\parastell-artifacts\prompt01-v3-wistell-20260827\lane-01a-geometry"
    ),
    Path(
        r"D:\parastell-artifacts\prompt01-v3-wistell-20260827\lane-01b-source-statistics"
    ),
    Path(
        r"D:\parastell-artifacts\prompt01-v3-wistell-20260827\lane-01c-transport"
    ),
    Path(
        r"D:\parastell-artifacts\prompt01-v3-wistell-20260827\lane-01c-boundary-banks"
    ),
    Path(
        r"D:\2026_DPA\artifacts\prompt01-v3-wistell-20260827\lane-01d-consumer-fixtures"
    ),
)

ROOTS = (
    {
        "path": Path(r"D:\Scratch\parastell_multi_config_20260827_01"),
        "owner": "prior ParaStell multi-configuration task",
        "basis": "known generic/example ParaStell model campaign",
        "classification": "EXCLUDE",
    },
    {
        "path": Path(
            r"D:\parastell-artifacts\prompt-a-surface-field-visual-production-gate-20260826"
        ),
        "owner": "prior Prompt-A task",
        "basis": "receipt identifies wrong 18-magnet/18-winding example geometry",
        "classification": "EXCLUDE",
    },
    {
        "path": Path(
            r"D:\parastell-artifacts\prompt1b-medium-qualification-20260825"
        ),
        "owner": "prior Prompt-1B task",
        "basis": "receipt marks artifacts as stale inventory seeds only",
        "classification": "EXCLUDE",
    },
    {
        "path": Path(r"D:\parastell-artifacts\geometry-recovery-20260827"),
        "owner": "prior geometry-recovery task",
        "basis": "receipt rejects prior example-derived geometry and scientific results",
        "classification": "EXCLUDE",
    },
    {
        "path": Path(r"D:\2026_DPA\DPA_workflow\examples\wistell_d"),
        "owner": "DPA_workflow example tree",
        "basis": "pre-v3 example workflow; not a new accepted lane artifact",
        "classification": "EXCLUDE",
    },
    {
        "path": Path(r"D:\2026_DPA\wistell-d-parastell-90deg"),
        "owner": "user-provided WISTELL-D evidence repository",
        "basis": (
            "authoritative-hash input copies are protected; stored 31x61 and "
            "historical explicit-coil models are rejected as scientific geometry"
        ),
        "classification": "EXCLUDE",
    },
    {
        "path": Path(r"D:\Scratch\parastell"),
        "owner": "preserved scratch ParaStell repository",
        "basis": "receipt permits code inspection only and rejects example artifacts",
        "classification": "EXCLUDE",
    },
    {
        "path": Path(
            r"D:\2026_DPA\worktrees\explicit-magnet-runtime-reference-closure-20260827"
        ),
        "owner": "preserved dirty DPA worktree",
        "basis": "receipt permits detached software/contracts only; no scientific result reuse",
        "classification": "EXCLUDE",
    },
    {
        "path": Path(__file__).resolve().parents[1] / "examples",
        "owner": "ParaStell repository example tree",
        "basis": "generic ParaStell example inputs are prohibited for WISTELL-D science",
        "classification": "EXCLUDE",
    },
    {
        "path": Path(__file__).resolve().parents[1]
        / "tests"
        / "files_for_tests",
        "owner": "ParaStell repository software-test fixtures",
        "basis": "fixture bytes require example-bound versus synthetic classification",
        "classification": "AMBIGUOUS",
    },
)

CANDIDATE_SUFFIXES = {
    ".step",
    ".stp",
    ".stl",
    ".h5m",
    ".h5",
    ".hdf5",
    ".msh",
    ".vtk",
    ".vtu",
    ".exo",
    ".nc",
    ".npy",
    ".npz",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".csv",
    ".tsv",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".pkl",
    ".pickle",
    ".bin",
    ".mcpl",
    ".wwinp",
}

CANDIDATE_NAME_MARKERS = (
    "statepoint",
    "source",
    "bank",
    "tally",
    "flux",
    "spectrum",
    "activation",
    "pka",
    "dpa",
    "dagmc",
    "geometry",
    "mesh",
    "manifest",
    "receipt",
    "audit",
    "handoff",
    "material",
    "boundary",
    "weight_window",
    "plot",
    "figure",
)

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
}

SYNTHETIC_FIXTURE_NAMES = {
    "one_cube.h5m",
    "two_cubes.h5m",
    "three_blocks.step",
}

HEX64 = re.compile(r"(?i)\b[0-9a-f]{64}\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def is_candidate(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in CANDIDATE_SUFFIXES or any(
        marker in name for marker in CANDIDATE_NAME_MARKERS
    )


def artifact_class(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".step", ".stp", ".stl"}:
        return "CAD_GEOMETRY"
    if suffix == ".h5m":
        return "DAGMC_OR_MOAB_GEOMETRY_OR_SOURCE_MESH"
    if suffix in {".msh", ".vtk", ".vtu", ".exo"}:
        return "MESH"
    if "statepoint" in name:
        return "OPENMC_STATEPOINT"
    if "bank" in name or suffix == ".mcpl":
        return "BOUNDARY_OR_PHASE_SPACE_BANK"
    if "weight" in name and "window" in name:
        return "WEIGHT_WINDOW"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "FIGURE_OR_REPORT"
    if suffix in {".json", ".yaml", ".yml", ".xml"}:
        return "MANIFEST_OR_CONFIGURATION"
    if suffix in {".npy", ".npz", ".h5", ".hdf5", ".bin", ".pkl", ".pickle"}:
        return "SCIENTIFIC_ARRAY_OR_BUNDLE"
    if suffix == ".nc":
        return "VMEC_OR_SCIENTIFIC_DATA"
    return "SCIENTIFIC_OR_GEOMETRY_ASSOCIATED_ARTIFACT"


def iter_candidates(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in relative_parts):
            continue
        resolved = path.resolve()
        if any(
            is_within(resolved, protected) for protected in PROTECTED_ROOTS
        ):
            raise RuntimeError(
                f"protected root entered candidate traversal: {resolved}"
            )
        if is_candidate(path):
            yield resolved


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from flatten_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_strings(item)
    elif isinstance(value, str):
        yield value


def read_reference_strings(path: Path) -> tuple[list[str], set[str]]:
    if path.stat().st_size > 32 * 1024 * 1024:
        return [], set()
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return [], set()
    strings = []
    if path.suffix.lower() == ".json":
        try:
            strings.extend(flatten_strings(json.loads(text)))
        except json.JSONDecodeError:
            strings.append(text)
    else:
        strings.append(text)
    return strings, {match.lower() for match in HEX64.findall(text)}


def classify(
    path: Path, digest: str, root_row: dict[str, Any]
) -> dict[str, Any]:
    if digest in AUTHORITATIVE_HASHES:
        return {
            "classification": "PROTECTED",
            "reason": "byte-identical copy of an authoritative WISTELL-D input",
            "evidence": [f"authoritative_source_sha256:{digest}"],
            "fixture_kind": "SCIENTIFIC_INPUT_COPY",
            "artifact_bytes_may_survive": True,
            "safe_to_remove": False,
            "proposed_action": "NO_ACTION_PROTECTED_BY_CONTENT_HASH",
        }
    relative_lower = str(path.relative_to(root_row["path"].resolve())).lower()
    if (
        root_row["owner"] == "ParaStell repository software-test fixtures"
        and path.name.lower() in SYNTHETIC_FIXTURE_NAMES
    ):
        return {
            "classification": "PROTECTED",
            "reason": "geometry-neutral synthetic software fixture",
            "evidence": ["synthetic_fixture_allowlist", relative_lower],
            "fixture_kind": "DETACHED_SYNTHETIC_SOFTWARE_TEST",
            "artifact_bytes_may_survive": True,
            "safe_to_remove": False,
            "proposed_action": "RETAIN_ONLY_IN_ISOLATED_SOFTWARE_TEST_SCOPE",
        }
    generic_fingerprint = digest in KNOWN_GENERIC_EXAMPLE_HASHES
    classification = root_row["classification"]
    if generic_fingerprint:
        classification = "EXCLUDE"
    user_owned = root_row["owner"].startswith("user-provided")
    generated_root = "artifacts" in str(root_row["path"]).lower()
    evidence = [root_row["basis"]]
    if generic_fingerprint:
        evidence.append(f"known_generic_example_sha256:{digest}")
    if (
        "31x61" in relative_lower
        or "validated_45deg_reference" in relative_lower
    ):
        evidence.append(
            "31x61 source CAD rejected by complete 36-pair overlap audit"
        )
    if "explicit_coil" in relative_lower:
        evidence.append(
            "explicit-coil geometry conflicts with continuous-envelope model"
        )
    if classification == "AMBIGUOUS":
        action = "QUARANTINE_PENDING_COORDINATOR_PROVENANCE_REVIEW"
    elif user_owned:
        action = "PROPOSE_RECOVERABLE_QUARANTINE; REQUIRE_USER_AND_COORDINATOR_REVIEW"
    else:
        action = "MOVE_TO_RECOVERABLE_QUARANTINE_AFTER_COORDINATOR_HASH_REVIEW"
    return {
        "classification": classification,
        "reason": root_row["basis"],
        "evidence": evidence,
        "fixture_kind": (
            "EXAMPLE_BOUND_SOFTWARE_FIXTURE"
            if "test" in relative_lower or "example" in relative_lower
            else "SCIENTIFIC_ARTIFACT"
        ),
        "artifact_bytes_may_survive": False,
        "safe_to_remove": bool(generated_root and not user_owned),
        "proposed_action": action,
        "negative_evidence_to_retain": (
            "SHA-256, classification, provenance reason, and rejection metrics only; "
            "do not retain artifact bytes as a scientific or software fixture"
        ),
    }


def build_inventory(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    root_rows = []
    entries = []
    seen_paths = set()
    quarantine_root = Path(
        r"D:\parastell-quarantine\prompt01-v3-wistell-20260827\coordinator-reviewed"
    )
    for root_index, configured in enumerate(ROOTS):
        root = configured["path"].resolve()
        if any(
            is_within(root, protected) or is_within(protected, root)
            for protected in PROTECTED_ROOTS
        ):
            raise RuntimeError(
                f"configured scan root overlaps protected root: {root}"
            )
        candidates = sorted(
            iter_candidates(root), key=lambda item: str(item).lower()
        )
        root_rows.append(
            {
                "path": str(root),
                "root_id": f"root-{root_index:02d}",
                "exists": root.is_dir(),
                "owner": configured["owner"],
                "classification_basis": configured["basis"],
                "candidate_file_count": len(candidates),
            }
        )
        for path in candidates:
            normalized = str(path).lower()
            if normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            digest = sha256_file(path)
            disposition = classify(path, digest, {**configured, "path": root})
            quarantine_path = (
                None
                if disposition["classification"] == "PROTECTED"
                else str(
                    quarantine_root
                    / f"root-{root_index:02d}"
                    / path.relative_to(root)
                )
            )
            entries.append(
                {
                    "absolute_path": str(path),
                    "artifact_class": artifact_class(path),
                    "byte_size": path.stat().st_size,
                    "sha256": digest,
                    "owner": configured["owner"],
                    **disposition,
                    "proposed_quarantine_path": quarantine_path,
                    "geometry_source_fingerprints": [],
                    "transitive_parent_hashes_when_available": [],
                    "known_manifests_or_bundles_referencing": [],
                    "referenced_artifact_hashes": [],
                }
            )

    by_path = {entry["absolute_path"].lower(): entry for entry in entries}
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_hash.setdefault(entry["sha256"], []).append(entry)
    for entry in entries:
        path = Path(entry["absolute_path"])
        strings, referenced_hashes = read_reference_strings(path)
        recognized = sorted(
            referenced_hashes
            & (AUTHORITATIVE_HASHES | KNOWN_GENERIC_EXAMPLE_HASHES)
        )
        entry["geometry_source_fingerprints"] = recognized
        entry["referenced_artifact_hashes"] = sorted(
            referenced_hashes & set(by_hash)
        )
        referenced_paths = set()
        for value in strings:
            candidate_value = value.strip().strip("\"'")
            possibilities = []
            try:
                value_path = Path(candidate_value)
                if value_path.is_absolute():
                    possibilities.append(value_path.resolve())
                else:
                    possibilities.append((path.parent / value_path).resolve())
            except (OSError, ValueError):
                pass
            for possibility in possibilities:
                key = str(possibility).lower()
                if key in by_path and key != entry["absolute_path"].lower():
                    referenced_paths.add(key)
            normalized_value = candidate_value.replace("/", "\\").lower()
            if (
                normalized_value in by_path
                and normalized_value != entry["absolute_path"].lower()
            ):
                referenced_paths.add(normalized_value)
        for digest in entry["referenced_artifact_hashes"]:
            for target in by_hash[digest]:
                if target is not entry:
                    referenced_paths.add(target["absolute_path"].lower())
        for target_key in sorted(referenced_paths):
            target = by_path[target_key]
            target["known_manifests_or_bundles_referencing"].append(
                {
                    "path": entry["absolute_path"],
                    "sha256": entry["sha256"],
                }
            )
            target["transitive_parent_hashes_when_available"].append(
                entry["sha256"]
            )

    for entry in entries:
        entry["known_manifests_or_bundles_referencing"] = sorted(
            entry["known_manifests_or_bundles_referencing"],
            key=lambda row: row["path"].lower(),
        )
        entry["transitive_parent_hashes_when_available"] = sorted(
            set(entry["transitive_parent_hashes_when_available"])
        )

    counts = {
        classification: sum(
            entry["classification"] == classification for entry in entries
        )
        for classification in ("EXCLUDE", "AMBIGUOUS", "PROTECTED")
    }
    payload = {
        "schema": "wistell_d.stale_artifact_inventory/v1.0.0",
        "generated_utc": utc_now(),
        "mode": "READ_ONLY_HASH_AND_REFERENCE_SCAN",
        "cleanup_performed": False,
        "classification_counts": counts,
        "entry_count": len(entries),
        "scan_roots": root_rows,
        "hard_excluded_from_traversal": [
            str(path.resolve()) for path in PROTECTED_ROOTS
        ],
        "protected_source_hashes": sorted(AUTHORITATIVE_HASHES),
        "proof_of_cleanup_exclusion": {
            "authoritative_root_never_traversed": True,
            "new_lane_roots_never_traversed": True,
            "matching_authoritative_hash_forces_PROTECTED": True,
            "cleanup_target_set_contains_protected_entry": False,
        },
        "cleanup_proposal": {
            "status": "PROPOSED_NOT_EXECUTED",
            "coordinator_review_required": True,
            "recoverable_quarantine_root_not_created": str(quarantine_root),
            "procedure": [
                "re-hash each exact path immediately before action",
                "hard-stop on any hash/path now protected by an accepted lane bundle",
                "move EXCLUDE files additively into a path-preserving quarantine tree",
                "leave AMBIGUOUS files in place until coordinator and owner review",
                "retain only hash/rejection metadata for example-bound negative evidence",
                "record original path, quarantine path, move time, and rollback command",
                "remove quarantined bytes only after downstream-reference audit and explicit review",
            ],
        },
        "entries": entries,
    }
    if any(
        entry["classification"] == "PROTECTED" and entry["safe_to_remove"]
        for entry in entries
    ):
        raise RuntimeError("protected entry was marked safe to remove")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "entry_count": len(entries),
                "classification_counts": counts,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_inventory(Path(args.output).resolve())


if __name__ == "__main__":
    main()
