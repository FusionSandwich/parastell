"""Validate the pinned activation data in the qualified OpenMC runtime."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import openmc
import openmc.deplete

from parastell.activation_handoff import inspect_activation_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chain", type=Path)
    parser.add_argument("cross_sections", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")

    data = inspect_activation_data(
        chain_path=arguments.chain,
        cross_sections_path=arguments.cross_sections,
        hash_transport_payloads=True,
    )
    chain = openmc.deplete.Chain.from_xml(arguments.chain)
    library = openmc.data.DataLibrary.from_xml(arguments.cross_sections)
    gate = bool(
        openmc.__version__ == "0.16.0"
        and data["chain"]["qualified"]
        and data["transport_catalog"]["qualified"]
        and data["transport_catalog"]["payloads_all_present"]
        and data["transport_catalog"]["payloads_all_hashed"]
        and len(chain.nuclides) == 3820
        and len(library) == 728
    )
    receipt = {
        "schema": "parastell.activation_runtime_validation/v1.0.0",
        "status": "PASS" if gate else "FAIL",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "openmc_version": openmc.__version__,
        "depletion_chain_nuclide_count": len(chain.nuclides),
        "transport_library_count": len(library),
        "activation_data": data,
        "transport_or_depletion_executed": False,
        "gate_pass": gate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
