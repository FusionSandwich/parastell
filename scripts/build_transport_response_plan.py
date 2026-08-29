"""Build a validated, geometry-neutral magnet transport response plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from parastell.reaction_identity import derive_material_nuclide_mt_requests
from parastell.transport_response_plan import (
    build_response_plan,
    estimate_response_cardinality,
)


def _material_derivation(config: dict, config_path: Path):
    control = config.get("material_mt_derivation")
    if control is None:
        return config.get("nuclide_mt_requests"), None
    if config.get("nuclide_mt_requests") is not None:
        raise ValueError(
            "explicit and material-derived MT requests cannot be combined"
        )
    if not isinstance(control, dict) or set(control) != {
        "materials",
        "default_mts",
        "nuclide_overrides",
    }:
        raise ValueError("material MT derivation control is incomplete")
    binding = control["materials"]
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("material inventory path/hash binding is incomplete")
    path = Path(str(binding["path"]))
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.resolve(strict=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != str(binding["sha256"]).lower():
        raise ValueError("material inventory hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    materials = (
        payload.get("materials") if isinstance(payload, dict) else payload
    )
    if not isinstance(materials, list):
        raise ValueError(
            "material inventory must be a list or contain materials"
        )
    derivation = derive_material_nuclide_mt_requests(
        materials,
        default_mts=control["default_mts"],
        nuclide_overrides=control["nuclide_overrides"],
    )
    derivation["material_inventory_sha256"] = actual
    unsigned = dict(derivation)
    unsigned.pop("derivation_sha256")
    derivation["derivation_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return derivation["nuclide_mt_requests"], derivation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    requests, derivation = _material_derivation(config, config_path)
    plan = build_response_plan(
        case_id=config["case_id"],
        magnet_ids=config["magnet_ids"],
        neutron_energy_edges_eV=config["energy_axes_eV"]["neutron"],
        photon_energy_edges_eV=config["energy_axes_eV"]["photon"],
        nuclide_mt_requests=requests,
        nuclide_mt_derivation=derivation,
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
