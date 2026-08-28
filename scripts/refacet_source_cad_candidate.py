"""CLI for byte-preserving source-CAD DAGMC refaceting."""

from __future__ import annotations

import argparse
from pathlib import Path

from parastell.source_cad_refaceting import refacet_source_cad_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("source_audit", type=Path)
    parser.add_argument("source_audit_seal", type=Path)
    parser.add_argument("physical_change_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--acceptance-criteria", type=Path, required=True)
    parser.add_argument("--faceting-protocol", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument(
        "--level", choices=("coarse", "refined"), required=True
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-source-audit-sha256", required=True)
    parser.add_argument("--expected-source-audit-seal-sha256", required=True)
    parser.add_argument(
        "--expected-physical-change-report-sha256", required=True
    )
    parser.add_argument("--expected-reference-manifest-sha256", required=True)
    parser.add_argument("--expected-acceptance-criteria-sha256", required=True)
    parser.add_argument("--expected-faceting-protocol-sha256", required=True)
    parser.add_argument("--expected-runtime-receipt-sha256", required=True)
    parser.add_argument("--expected-attempt-id", required=True)
    parser.add_argument("--expected-launch-nonce", required=True)
    parser.add_argument("--expected-host-alias", required=True)
    arguments = parser.parse_args()
    refacet_source_cad_candidate(
        arguments.source_dir,
        arguments.source_audit,
        arguments.source_audit_seal,
        arguments.physical_change_report,
        arguments.output_dir,
        acceptance_criteria_path=arguments.acceptance_criteria,
        faceting_protocol_path=arguments.faceting_protocol,
        runtime_receipt_path=arguments.runtime_receipt,
        level=arguments.level,
        threads=arguments.threads,
        expected_source_manifest_sha256=(
            arguments.expected_source_manifest_sha256
        ),
        expected_source_audit_sha256=arguments.expected_source_audit_sha256,
        expected_source_audit_seal_sha256=(
            arguments.expected_source_audit_seal_sha256
        ),
        expected_physical_change_report_sha256=(
            arguments.expected_physical_change_report_sha256
        ),
        expected_reference_manifest_sha256=(
            arguments.expected_reference_manifest_sha256
        ),
        expected_acceptance_criteria_sha256=(
            arguments.expected_acceptance_criteria_sha256
        ),
        expected_faceting_protocol_sha256=(
            arguments.expected_faceting_protocol_sha256
        ),
        expected_runtime_receipt_sha256=(
            arguments.expected_runtime_receipt_sha256
        ),
        expected_attempt_id=arguments.expected_attempt_id,
        expected_launch_nonce=arguments.expected_launch_nonce,
        expected_host_alias=arguments.expected_host_alias,
    )


if __name__ == "__main__":
    main()
