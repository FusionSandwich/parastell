"""Extract every wired full-response tally from one bounded statepoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.openmc16_response_results import read_openmc16_response_set
from parastell.openmc16_response_results import validate_response_coverage
from parastell.parametric_openmc16_model import _canonical_sha
from parastell.transport_response_plan import validate_response_plan
from scripts.run_parametric_openmc16_geometry_debug import sha256_file


SCHEMA = "parastell.parametric_full_response_result_bundle/v1.0.0"
SMOKE_SCHEMA = "parastell.parametric_openmc16_full_response_smoke/v1.0.0"


def _load_smoke_receipt(path: Path, expected_sha256: str) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError("full-response smoke receipt hash mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    content_hash = receipt.pop("receipt_content_sha256", None)
    content_pass = content_hash == _canonical_sha(receipt)
    receipt["receipt_content_sha256"] = content_hash
    plan = receipt.get("full_response_wiring", {}).get("response_plan")
    if (
        receipt.get("schema") != SMOKE_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("claim") != "BOUNDED_INFRASTRUCTURE_SMOKE_ONLY"
        or receipt.get("inputs_immutable") is not True
        or receipt.get("statistics_qualified") is not False
        or receipt.get("production_run_authorized") is not False
        or not content_pass
        or not isinstance(plan, dict)
        or plan.get("proof_level") != "WIRED"
        or plan.get("surface_bank_configured") is not True
    ):
        raise ValueError("full-response smoke receipt is not accepted")
    validate_response_plan(plan)
    names = plan.get("wired_tally_names")
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(value, str) or not value.strip() for value in names
        )
        or len(names) != len(set(names))
    ):
        raise ValueError("wired tally inventory is invalid")
    return receipt


def extract(
    smoke_receipt_path: Path,
    statepoint_path: Path,
    *,
    expected_smoke_receipt_sha256: str,
    expected_statepoint_sha256: str,
) -> dict:
    receipt = _load_smoke_receipt(
        smoke_receipt_path, expected_smoke_receipt_sha256
    )
    actual_statepoint = sha256_file(statepoint_path)
    if actual_statepoint != expected_statepoint_sha256:
        raise ValueError("statepoint hash mismatch")
    declared = receipt.get("statepoints")
    if (
        not isinstance(declared, list)
        or len(declared) != 1
        or declared[0].get("sha256") != actual_statepoint
    ):
        raise ValueError(
            "statepoint is not the one bound by the smoke receipt"
        )
    plan = receipt["full_response_wiring"]["response_plan"]
    names = list(plan["wired_tally_names"])
    response_set = read_openmc16_response_set(statepoint_path, names)
    expected_run = {
        "statepoint_sha256": actual_statepoint,
        "particles_per_batch": receipt["particles_per_batch"],
        "batches": receipt["batches"],
        "source_histories": receipt["histories"],
        "seed": receipt["seed"],
    }
    if any(
        response_set.get(key) != value for key, value in expected_run.items()
    ):
        raise ValueError(
            "statepoint run metadata disagrees with smoke receipt"
        )
    coverage = validate_response_coverage(response_set, names)
    if coverage["status"] != "COMPLETE":
        raise ValueError("statepoint does not contain every wired tally")
    result = {
        "schema": SCHEMA,
        "status": "SMOKE_RESULT_EXTRACTED",
        "claim": "INFRASTRUCTURE_ONLY_NOT_STATISTICS_QUALIFIED",
        "smoke_receipt": {
            "path": str(smoke_receipt_path),
            "sha256": expected_smoke_receipt_sha256,
        },
        "statepoint": {
            "path": str(statepoint_path),
            "sha256": actual_statepoint,
        },
        "selected_magnet_ids": receipt["selected_magnet_ids"],
        "selected_magnet_cell_ids": receipt["selected_magnet_cell_ids"],
        "source_normalization": receipt["source_normalization"],
        "response_plan": plan,
        "coverage": coverage,
        "response_set": response_set,
        "missing_response_semantics": "MISSING_IS_NOT_ZERO",
        "covariance_status": "UNAVAILABLE_UNLESS_EXPLICITLY_COMPUTED",
        "statistics_qualified": False,
        "production_run_authorized": False,
    }
    result["bundle_content_sha256"] = _canonical_sha(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_receipt", type=Path)
    parser.add_argument("statepoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-smoke-receipt-sha256", required=True)
    parser.add_argument("--expected-statepoint-sha256", required=True)
    args = parser.parse_args()
    smoke = args.smoke_receipt.resolve(strict=True)
    statepoint = args.statepoint.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    value = extract(
        smoke,
        statepoint,
        expected_smoke_receipt_sha256=args.expected_smoke_receipt_sha256,
        expected_statepoint_sha256=args.expected_statepoint_sha256,
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
