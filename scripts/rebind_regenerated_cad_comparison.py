"""Rebind a passed CAD/H5M comparison to byte-identical regenerated BREPs.

This is a transitive evidence operation, not a new geometric comparison.  It
is valid only when the exact casing and winding-pack BREP bytes, source inputs,
transport IDs, fingerprint, and canonical H5M are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path) -> tuple[Path, dict]:
    resolved = Path(path).resolve()
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def _source_identity(manifest: dict) -> dict:
    source = deepcopy(manifest["source"])
    source.pop("exporter_implementation", None)
    return {
        "magnet_id": manifest["magnet_id"],
        "built_coil_index": manifest["built_coil_index"],
        "source_filament_index": manifest["source_filament_index"],
        "coordinate_contract": manifest["coordinate_contract"],
        "target_dagmc": manifest["target_dagmc"],
        "source": source,
    }


def rebind(
    *,
    prior_manifest_path: str | Path,
    prior_comparison_path: str | Path,
    new_manifest_path: str | Path,
    canonical_h5m_path: str | Path,
    output_path: str | Path,
) -> dict:
    prior_manifest_file, prior_manifest = _load(prior_manifest_path)
    prior_comparison_file, prior_comparison = _load(prior_comparison_path)
    new_manifest_file, new_manifest = _load(new_manifest_path)
    h5m = Path(canonical_h5m_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    if (
        prior_comparison.get("status") != "PASS"
        or prior_comparison.get("classification")
        != "PARASTELL_SOURCE_CAD_REGENERATED_EXACT_INPUTS"
    ):
        raise ValueError(
            "prior CAD/H5M comparison is not a passing qualification"
        )
    if prior_comparison["source_cad_manifest"]["sha256"] != file_hash(
        prior_manifest_file
    ):
        raise ValueError(
            "prior comparison does not bind the supplied prior manifest"
        )
    if prior_comparison["canonical_h5m"]["sha256"] != file_hash(h5m):
        raise ValueError(
            "canonical H5M hash differs from the prior comparison"
        )
    if _source_identity(prior_manifest) != _source_identity(new_manifest):
        raise ValueError(
            "regenerated source inputs or transport identity changed"
        )
    if "exporter_implementation" not in new_manifest["source"]:
        raise ValueError(
            "new manifest does not bind the exporter implementation"
        )

    roles = {}
    for role in ("casing", "winding_pack"):
        prior_record = prior_manifest["artifacts"][role]
        new_record = new_manifest["artifacts"][role]
        if prior_record["dagmc_volume_id"] != new_record["dagmc_volume_id"]:
            raise ValueError(f"{role} DAGMC volume identity changed")
        prior_brep = prior_manifest_file.parent / prior_record["brep"]["path"]
        new_brep = new_manifest_file.parent / new_record["brep"]["path"]
        prior_hash = file_hash(prior_brep)
        new_hash = file_hash(new_brep)
        if prior_hash != prior_record["brep"]["sha256"]:
            raise ValueError(f"prior {role} BREP hash mismatch")
        if new_hash != new_record["brep"]["sha256"]:
            raise ValueError(f"new {role} BREP hash mismatch")
        if prior_hash != new_hash:
            raise ValueError(f"{role} BREP is not byte-identical")
        roles[role] = {
            "dagmc_volume_id": new_record["dagmc_volume_id"],
            "brep_sha256": new_hash,
            "byte_identical_to_prior_compared_brep": True,
        }

    result = deepcopy(prior_comparison)
    result["source_cad_manifest"] = {
        "local_identifier": new_manifest_file.name,
        "sha256": file_hash(new_manifest_file),
    }
    result["target_geometry_fingerprint"] = new_manifest["target_dagmc"][
        "canonical_geometry_fingerprint"
    ]
    result["evidence_reuse"] = {
        "method": "TRANSITIVE_BYTE_IDENTICAL_BREP_REBIND",
        "geometric_metrics_recomputed": False,
        "prior_manifest": {
            "local_identifier": prior_manifest_file.name,
            "sha256": file_hash(prior_manifest_file),
        },
        "prior_comparison": {
            "local_identifier": prior_comparison_file.name,
            "sha256": file_hash(prior_comparison_file),
        },
        "canonical_h5m_rehashed": True,
        "canonical_h5m_sha256": file_hash(h5m),
        "source_inputs_identical": True,
        "roles": roles,
        "new_exporter_implementation": new_manifest["source"][
            "exporter_implementation"
        ],
    }
    result.setdefault("limitations", []).append(
        "metrics are reused transitively because both new BREP payloads are byte-identical to the previously compared payloads"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-manifest", required=True)
    parser.add_argument("--prior-comparison", required=True)
    parser.add_argument("--new-manifest", required=True)
    parser.add_argument("--canonical-h5m", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            rebind(
                prior_manifest_path=args.prior_manifest,
                prior_comparison_path=args.prior_comparison,
                new_manifest_path=args.new_manifest,
                canonical_h5m_path=args.canonical_h5m,
                output_path=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
