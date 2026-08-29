"""Extract every wired full-response tally from one bounded statepoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.openmc16_response_results import read_openmc16_response_set
from parastell.openmc16_response_results import validate_response_coverage
from parastell.openmc16_response_handoff import (
    build_openmc16_radiation_consumer_handoff,
)
from parastell.radiation_consumer_handoff import (
    validate_radiation_consumer_handoff,
)
from parastell.parametric_openmc16_model import _canonical_sha
from parastell.transport_response_plan import validate_response_plan
from scripts.run_parametric_openmc16_geometry_debug import sha256_file


SCHEMA = "parastell.parametric_full_response_result_bundle/v1.0.0"
SMOKE_SCHEMA = "parastell.parametric_openmc16_full_response_smoke/v1.0.0"
TALLY_BINDING_SCHEMA = "parastell.openmc16_exact_tally_bindings/v1.0.0"
CONSUMER_INPUT_SCHEMA = "parastell.openmc16_consumer_handoff_input/v1.0.0"


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


def _load_json_binding(path: Path, expected_sha256: str, label: str) -> dict:
    path = path.resolve(strict=True)
    if sha256_file(path) != str(expected_sha256).lower():
        raise ValueError(f"{label} hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def build_consumer_handoff(
    extraction: dict,
    *,
    tally_bindings_path: Path,
    expected_tally_bindings_sha256: str,
    consumer_input_path: Path,
    expected_consumer_input_sha256: str,
) -> dict:
    """Convert the exact extracted statepoint into the validated handoff."""
    bindings = _load_json_binding(
        tally_bindings_path,
        expected_tally_bindings_sha256,
        "tally bindings",
    )
    required_tallies = set(extraction["response_plan"]["wired_tally_names"])
    binding_digest = bindings.get("bindings_sha256")
    unsigned_bindings = dict(bindings)
    unsigned_bindings.pop("bindings_sha256", None)
    if (
        bindings.get("schema") != TALLY_BINDING_SCHEMA
        or bindings.get("status") != "PASS"
        or bindings.get("statepoint_sha256")
        != extraction["statepoint"]["sha256"]
        or bindings.get("model_xml_sha256") is None
        or not isinstance(bindings.get("tallies"), dict)
        or set(bindings["tallies"]) != required_tallies
        or binding_digest != _canonical_sha(unsigned_bindings)
    ):
        raise ValueError(
            "exact tally bindings are incomplete or run-mismatched"
        )
    model_hash = str(bindings["model_xml_sha256"]).lower()
    if len(model_hash) != 64 or any(
        c not in "0123456789abcdef" for c in model_hash
    ):
        raise ValueError("tally-binding model XML hash is invalid")
    consumer = _load_json_binding(
        consumer_input_path,
        expected_consumer_input_sha256,
        "consumer input",
    )
    if (
        consumer.get("schema") != CONSUMER_INPUT_SCHEMA
        or consumer.get("status") != "PASS"
        or set(consumer)
        != {
            "schema",
            "status",
            "provenance",
            "materials",
            "boundary_phase_space",
            "activation_schedule_reference",
        }
    ):
        raise ValueError("consumer handoff input is incomplete")
    provenance = consumer["provenance"]
    if (
        not isinstance(provenance, dict)
        or provenance.get("statepoint_sha256")
        != extraction["statepoint"]["sha256"]
        or provenance.get("model_xml_sha256") != model_hash
        or int(provenance.get("source_histories", -1))
        != int(extraction["response_set"]["source_histories"])
    ):
        raise ValueError("consumer provenance is not extraction/tally bound")
    handoff = build_openmc16_radiation_consumer_handoff(
        response_set=extraction["response_set"],
        tally_bindings=bindings["tallies"],
        provenance=provenance,
        materials=consumer["materials"],
        boundary_phase_space=consumer["boundary_phase_space"],
        activation_schedule_reference=consumer[
            "activation_schedule_reference"
        ],
    )
    handoff["extraction_binding"] = {
        "result_bundle_sha256": extraction["bundle_content_sha256"],
        "tally_bindings_sha256": str(expected_tally_bindings_sha256).lower(),
        "consumer_input_sha256": str(expected_consumer_input_sha256).lower(),
    }
    unsigned = dict(handoff)
    unsigned.pop("handoff_content_sha256", None)
    handoff["handoff_content_sha256"] = _canonical_sha(unsigned)
    validate_radiation_consumer_handoff(handoff)
    return handoff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_receipt", type=Path)
    parser.add_argument("statepoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-smoke-receipt-sha256", required=True)
    parser.add_argument("--expected-statepoint-sha256", required=True)
    parser.add_argument("--tally-bindings", type=Path)
    parser.add_argument("--expected-tally-bindings-sha256")
    parser.add_argument("--consumer-input", type=Path)
    parser.add_argument("--expected-consumer-input-sha256")
    parser.add_argument("--consumer-handoff-output", type=Path)
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
    consumer_arguments = (
        args.tally_bindings,
        args.expected_tally_bindings_sha256,
        args.consumer_input,
        args.expected_consumer_input_sha256,
        args.consumer_handoff_output,
    )
    handoff = None
    handoff_output = None
    if any(value is not None for value in consumer_arguments):
        if any(value is None for value in consumer_arguments):
            raise ValueError("consumer handoff CLI arguments must be complete")
        handoff = build_consumer_handoff(
            value,
            tally_bindings_path=args.tally_bindings,
            expected_tally_bindings_sha256=args.expected_tally_bindings_sha256,
            consumer_input_path=args.consumer_input,
            expected_consumer_input_sha256=args.expected_consumer_input_sha256,
        )
        handoff_output = args.consumer_handoff_output.resolve()
        if handoff_output.exists():
            raise FileExistsError(
                f"create-only consumer handoff exists: {handoff_output}"
            )
    if output.exists():
        raise FileExistsError(
            f"create-only extraction output exists: {output}"
        )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if handoff is not None:
        handoff_output.parent.mkdir(parents=True, exist_ok=True)
        with handoff_output.open("x", encoding="utf-8") as stream:
            json.dump(
                handoff, stream, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
