"""Collect a sealed, decision-free DAGMC faceting evidence packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parastell.faceting_evidence import collect_faceting_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect semantic topology, certified facet-to-source-CAD bounds, "
            "and former-overlap-zone statistics for one faceting level."
        )
    )
    parser.add_argument("--dagmc", type=Path, required=True)
    parser.add_argument(
        "--level", choices=("coarse", "refined"), required=True
    )
    parser.add_argument("--refaceting-manifest", type=Path, required=True)
    parser.add_argument("--expected-refaceting-manifest-sha256", required=True)
    parser.add_argument("--refaceting-seal", type=Path, required=True)
    parser.add_argument("--expected-refaceting-seal-sha256", required=True)
    parser.add_argument("--faceting-protocol", type=Path, required=True)
    parser.add_argument("--expected-faceting-protocol-sha256", required=True)
    parser.add_argument("--expected-h5m-sha256", required=True)
    parser.add_argument(
        "--expected-physical-input-identity-sha256", required=True
    )
    parser.add_argument("--native-qualification", type=Path, required=True)
    parser.add_argument(
        "--expected-native-qualification-sha256", required=True
    )
    parser.add_argument("--source-cad-audit", type=Path, required=True)
    parser.add_argument("--expected-source-cad-audit-sha256", required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--expected-source-map-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--public-overlap-report", type=Path, required=True)
    parser.add_argument(
        "--expected-public-overlap-report-sha256", required=True
    )
    parser.add_argument(
        "--maximum-projected-subtriangle-centroids", type=int, required=True
    )
    parser.add_argument(
        "--progress-every-triangle-incidences", type=int, default=100
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = collect_faceting_evidence(
        dagmc_path=arguments.dagmc,
        level=arguments.level,
        refaceting_manifest_path=arguments.refaceting_manifest,
        expected_refaceting_manifest_sha256=(
            arguments.expected_refaceting_manifest_sha256
        ),
        refaceting_seal_path=arguments.refaceting_seal,
        expected_refaceting_seal_sha256=(
            arguments.expected_refaceting_seal_sha256
        ),
        faceting_protocol_path=arguments.faceting_protocol,
        expected_faceting_protocol_sha256=(
            arguments.expected_faceting_protocol_sha256
        ),
        expected_h5m_sha256=arguments.expected_h5m_sha256,
        expected_physical_input_identity_sha256=(
            arguments.expected_physical_input_identity_sha256
        ),
        native_qualification_path=arguments.native_qualification,
        expected_native_qualification_sha256=(
            arguments.expected_native_qualification_sha256
        ),
        source_cad_audit_path=arguments.source_cad_audit,
        expected_source_cad_audit_sha256=(
            arguments.expected_source_cad_audit_sha256
        ),
        source_map_path=arguments.source_map,
        expected_source_map_sha256=arguments.expected_source_map_sha256,
        source_root=arguments.source_root,
        public_overlap_report_path=arguments.public_overlap_report,
        expected_public_overlap_report_sha256=(
            arguments.expected_public_overlap_report_sha256
        ),
        maximum_projected_subtriangle_centroids=(
            arguments.maximum_projected_subtriangle_centroids
        ),
        progress_every_triangle_incidences=(
            arguments.progress_every_triangle_incidences
        ),
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
