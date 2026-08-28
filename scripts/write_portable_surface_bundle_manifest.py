"""Rebase container paths and verify a surface evidence bundle on its host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_create_only(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _rebase(path_text: str, mappings: dict[str, Path]) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path.resolve()
    normalized = str(PurePosixPath(path_text.replace("\\", "/")))
    for prefix in sorted(mappings, key=len, reverse=True):
        normalized_prefix = str(PurePosixPath(prefix.replace("\\", "/")))
        if normalized == normalized_prefix or normalized.startswith(
            normalized_prefix + "/"
        ):
            suffix = normalized[len(normalized_prefix) :].lstrip("/")
            candidate = (
                mappings[prefix] / Path(*PurePosixPath(suffix).parts)
            ).resolve()
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no verified portable path for {path_text!r}")


def _collect_pairs(
    value: Any, pointer: str = "$"
) -> list[tuple[str, str, str]]:
    pairs = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(
            value.get("sha256"), str
        ):
            pairs.append((pointer, value["path"], value["sha256"].lower()))
        for key, child in value.items():
            pairs.extend(_collect_pairs(child, f"{pointer}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            pairs.extend(_collect_pairs(child, f"{pointer}[{index}]"))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("strict_audit", type=Path)
    parser.add_argument("evidence_receipt", type=Path)
    parser.add_argument("bundle_directory", type=Path)
    parser.add_argument("path_mapping_json", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()

    strict_path = args.strict_audit.resolve(strict=True)
    receipt_path = args.evidence_receipt.resolve(strict=True)
    bundle = args.bundle_directory.resolve(strict=True)
    mapping_path = args.path_mapping_json.resolve(strict=True)
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping_payload.get("schema") != "parastell.path_rebase_map/v1.0.0":
        raise ValueError("unsupported path-rebase mapping schema")
    raw_mappings = mapping_payload.get("prefix_mappings")
    if not isinstance(raw_mappings, dict) or not raw_mappings:
        raise ValueError("path-rebase map has no prefix mappings")
    mappings = {
        str(prefix): Path(str(root)).resolve()
        for prefix, root in raw_mappings.items()
    }
    if any(not root.is_dir() for root in mappings.values()):
        raise FileNotFoundError(
            "one or more portable mapping roots are absent"
        )

    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for source_name, payload in (
        ("strict_audit", strict),
        ("receipt", receipt),
    ):
        for pointer, execution_path, expected_hash in _collect_pairs(payload):
            if len(expected_hash) != 64:
                continue
            key = (execution_path, expected_hash)
            if key in seen:
                continue
            seen.add(key)
            is_execution_absolute = Path(execution_path).is_absolute() or (
                PurePosixPath(execution_path.replace("\\", "/")).is_absolute()
            )
            if not is_execution_absolute:
                candidate = bundle / "figures" / execution_path
                if not candidate.is_file():
                    continue
                portable_path = candidate.resolve()
            else:
                portable_path = _rebase(execution_path, mappings)
            actual_hash = _sha256(portable_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"portable artifact hash mismatch for {execution_path!r}"
                )
            rows.append(
                {
                    "source": source_name,
                    "json_pointer": pointer,
                    "execution_path": execution_path,
                    "portable_path": str(portable_path),
                    "sha256": actual_hash,
                    "bytes": portable_path.stat().st_size,
                }
            )
    if not rows:
        raise ValueError("no portable artifacts were verified")
    manifest = {
        "schema": "parastell.portable_surface_evidence_bundle/v1.0.0",
        "status": "PASS",
        "strict_audit_sha256": _sha256(strict_path),
        "evidence_receipt_sha256": _sha256(receipt_path),
        "path_rebase_map_sha256": _sha256(mapping_path),
        "verified_artifact_count": len(rows),
        "artifacts": rows,
    }
    _write_json_create_only(args.output_manifest.resolve(), manifest)
    print(args.output_manifest.resolve())


if __name__ == "__main__":
    main()
