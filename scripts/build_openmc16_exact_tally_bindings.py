"""Create exact, hash-bound tally definitions for consumer extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from parastell.openmc16_response_handoff import build_exact_tally_bindings


def _load(path: Path, expected_sha256: str, label: str):
    source = path.resolve(strict=True)
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != expected_sha256.lower()
    ):
        raise ValueError(f"{label} hash mismatch")
    return json.loads(source.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("response_bundle", type=Path)
    parser.add_argument("reaction_mt_control", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-response-bundle-sha256", required=True)
    parser.add_argument("--expected-reaction-mt-control-sha256", required=True)
    parser.add_argument("--model-xml-sha256", required=True)
    arguments = parser.parse_args()
    bundle = _load(
        arguments.response_bundle,
        arguments.expected_response_bundle_sha256,
        "response bundle",
    )
    if (
        bundle.get("schema")
        != "parastell.parametric_full_response_result_bundle/v1.0.0"
        or bundle.get("status") != "SMOKE_RESULT_EXTRACTED"
    ):
        raise ValueError("response bundle is not accepted")
    control = _load(
        arguments.reaction_mt_control,
        arguments.expected_reaction_mt_control_sha256,
        "reaction MT control",
    )
    if (
        set(control) != {"schema", "reaction_mt_by_tally"}
        or control.get("schema")
        != "parastell.reaction_mt_binding_control/v1.0.0"
    ):
        raise ValueError("reaction MT control schema is invalid")
    bindings = build_exact_tally_bindings(
        bundle["response_set"],
        model_xml_sha256=arguments.model_xml_sha256,
        reaction_mt_by_tally=control["reaction_mt_by_tally"],
    )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(bindings, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
