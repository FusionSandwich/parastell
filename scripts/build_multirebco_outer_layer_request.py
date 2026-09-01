#!/usr/bin/env python3
"""Create a fail-closed multi-REBCO outer-layer producer request."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.source_conformal_interface import (
    build_source_conformal_multirebco_producer_request_v2,
    write_compact_json_create_only,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def build_request(
    master_manifest_path: Path, family_paths: list[Path]
) -> dict:
    manifest = _load_json(master_manifest_path)
    master = manifest.get("master_interface", {})
    packet = manifest.get("master_interface_packet", {})
    if not isinstance(master, dict) or not isinstance(packet, dict):
        raise TypeError(
            "master manifest omits immutable interface packet metadata"
        )
    families = [_load_json(path) for path in family_paths]
    family_ids = [str(value.get("family_id", "")) for value in families]
    tapes = {
        family_id: [str(value) for value in family.get("tape_ids", [])]
        for family_id, family in zip(family_ids, families, strict=True)
    }
    request = build_source_conformal_multirebco_producer_request_v2(
        master_interface_sha256=str(master.get("master_interface_sha256", "")),
        master_interface_packet=packet,
        requested_family_ids=family_ids,
        requested_tape_ids_by_family=tapes,
    )
    request["master_manifest"] = {
        "path": str(master_manifest_path.resolve()),
        "schema": manifest.get("schema"),
    }
    request["family_contracts"] = [
        {"path": str(path.resolve()), "family_id": family_id}
        for path, family_id in zip(family_paths, family_ids, strict=True)
    ]
    request.pop("request_sha256", None)
    request["request_sha256"] = hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument(
        "--family-contract",
        type=Path,
        action="append",
        required=True,
        help="repeat for each REBCO family; at least two are required",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = build_request(args.master_manifest, args.family_contract)
    identity = write_compact_json_create_only(args.output, request)
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
