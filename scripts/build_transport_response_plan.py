"""Build a validated, geometry-neutral magnet transport response plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.transport_response_plan import (
    build_response_plan,
    estimate_response_cardinality,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    config = json.loads(
        arguments.config.resolve(strict=True).read_text(encoding="utf-8")
    )
    plan = build_response_plan(
        case_id=config["case_id"],
        magnet_ids=config["magnet_ids"],
        neutron_energy_edges_eV=config["energy_axes_eV"]["neutron"],
        photon_energy_edges_eV=config["energy_axes_eV"]["photon"],
        nuclide_mt_requests=config.get("nuclide_mt_requests"),
    )
    payload = {"response_plan": plan}
    if "surface_count" in config:
        payload["cardinality_estimate"] = estimate_response_cardinality(
            plan,
            surface_count=int(config["surface_count"]),
            local_mesh_bins_per_magnet=int(
                config.get("local_mesh_bins_per_magnet", 0)
            ),
            reaction_family_count=int(config.get("reaction_family_count", 6)),
        )
    target = arguments.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(target)


if __name__ == "__main__":
    main()
