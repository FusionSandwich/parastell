"""Write a create-only source-domain qualification receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.reference_geometry import sha256_file
from parastell.source_domain import audit_source_domain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc", type=Path)
    parser.add_argument("source_mesh", type=Path)
    parser.add_argument("reference_source_mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--expected-source-mesh-sha256", required=True)
    parser.add_argument(
        "--expected-reference-source-mesh-sha256", required=True
    )
    parser.add_argument(
        "--expected-reference-source-fingerprint", required=True
    )
    parser.add_argument("--acceptance-criteria", required=True, type=Path)
    parser.add_argument("--expected-acceptance-criteria-sha256", required=True)
    parser.add_argument("--source-material", default="Vacuum")
    parser.add_argument("--source-component", default="chamber")
    parser.add_argument("--source-volume-id", required=True, type=int)
    arguments = parser.parse_args()

    dagmc = arguments.dagmc.resolve()
    source = arguments.source_mesh.resolve()
    reference = arguments.reference_source_mesh.resolve()
    criteria_path = arguments.acceptance_criteria.resolve()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    expected = {
        dagmc: arguments.expected_dagmc_sha256,
        source: arguments.expected_source_mesh_sha256,
        reference: arguments.expected_reference_source_mesh_sha256,
    }
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(
                f"hash mismatch for {path}: expected {digest}, got {actual}"
            )
    criteria_sha256 = sha256_file(criteria_path)
    if criteria_sha256 != arguments.expected_acceptance_criteria_sha256:
        raise ValueError("acceptance-criteria hash mismatch")
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
    result = audit_source_domain(
        dagmc,
        source,
        reference,
        source_volume_id=arguments.source_volume_id,
        source_component=arguments.source_component,
        source_material=arguments.source_material,
        source_strength_relative_tolerance=float(
            criteria["source_domain"]["source_strength_relative_tolerance"]
        ),
    )
    result["acceptance_criteria_path"] = str(criteria_path)
    result["acceptance_criteria_sha256"] = criteria_sha256
    result["expected_reference_source_mesh_sha256"] = (
        arguments.expected_reference_source_mesh_sha256
    )
    result["expected_reference_source_fingerprint"] = (
        arguments.expected_reference_source_fingerprint
    )
    result["reference_source_fingerprint_matches_expected"] = (
        result["reference_source_mesh_identity"]["canonical_fingerprint"]
        == arguments.expected_reference_source_fingerprint
    )
    result["post_audit_input_hashes"] = {
        "dagmc": sha256_file(dagmc),
        "source_mesh": sha256_file(source),
        "reference_source_mesh": sha256_file(reference),
        "acceptance_criteria": sha256_file(criteria_path),
    }
    result["input_immutability_pass"] = result["post_audit_input_hashes"] == {
        "dagmc": arguments.expected_dagmc_sha256,
        "source_mesh": arguments.expected_source_mesh_sha256,
        "reference_source_mesh": (
            arguments.expected_reference_source_mesh_sha256
        ),
        "acceptance_criteria": (arguments.expected_acceptance_criteria_sha256),
    }
    result["source_domain_gate_pass"] = bool(
        result["source_domain_gate_pass"]
        and result["reference_source_fingerprint_matches_expected"]
        and result["input_immutability_pass"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
