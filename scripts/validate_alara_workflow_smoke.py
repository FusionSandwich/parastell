"""Validate and seal one bounded ParaStell-owned ALARA smoke result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.alara_activation import validate_alara_result_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--remote-run-root", required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    receipt = validate_alara_result_files(
        output_path=args.result_directory / "alara.out",
        stderr_path=args.result_directory / "alara.stderr",
        delayed_photon_path=(
            args.result_directory / "delayed_photon_source.txt"
        ),
        input_manifest_sha256=args.input_manifest_sha256,
        remote_run_root=args.remote_run_root,
        provenance_origin="SYNTHETIC_WORKFLOW_FIXTURE",
    )
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
