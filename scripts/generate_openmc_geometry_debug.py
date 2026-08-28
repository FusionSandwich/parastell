"""Create a bounded OpenMC geometry-debug model outside an immutable H5M."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from parastell.reference_geometry import ReferenceGeometry
from parastell.reference_geometry import native_dagmc_id_inventory
from parastell.reference_geometry import sha256_file


def _material(openmc, material_id, name, nuclide, density):
    material = openmc.Material(material_id=int(material_id), name=name)
    material.add_nuclide(nuclide, 1.0)
    material.set_density("g/cm3", density)
    return material


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _maximum_vertex_radius(dagmc_path: Path) -> float:
    from pymoab import core

    mesh = core.Core()
    mesh.load_file(str(dagmc_path))
    vertices = mesh.get_entities_by_dimension(mesh.get_root_set(), 0)
    if not len(vertices):
        raise ValueError("DAGMC H5M has no vertices")
    coordinates = mesh.get_coords(vertices).reshape((-1, 3))
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("DAGMC vertex coordinates are not finite")
    return float(np.linalg.norm(coordinates, axis=1).max())


def nuclear_data_manifest(
    cross_sections_path: Path,
    *,
    required_nuclides: tuple[str, ...] = ("H1", "Fe56", "Li6", "Cu63"),
) -> dict:
    """Hash the exact neutron libraries needed by the bounded model."""
    cross_sections_path = cross_sections_path.resolve()
    root = ET.parse(cross_sections_path).getroot()
    rows = []
    for nuclide in required_nuclides:
        matches = [
            node
            for node in root.findall("./library")
            if node.get("type") == "neutron"
            and nuclide in (node.get("materials") or "").split()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cross_sections.xml requires exactly one neutron library for {nuclide}"
            )
        library_path = Path(matches[0].get("path", ""))
        if not library_path.is_absolute():
            library_path = cross_sections_path.parent / library_path
        library_path = library_path.resolve()
        if not library_path.is_file():
            raise FileNotFoundError(library_path)
        rows.append(
            {
                "nuclide": nuclide,
                "path": str(library_path),
                "sha256": sha256_file(library_path),
                "size_bytes": library_path.stat().st_size,
            }
        )
    payload = {
        "cross_sections_path": str(cross_sections_path),
        "cross_sections_sha256": sha256_file(cross_sections_path),
        "libraries": rows,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def model_xml_contract(path: Path) -> dict:
    """Extract the exact transport-relevant contract from an OpenMC model."""
    root = ET.parse(path).getroot()

    def one(xpath: str) -> ET.Element:
        rows = root.findall(xpath)
        if len(rows) != 1:
            raise ValueError(f"model XML requires exactly one {xpath}")
        return rows[0]

    settings = one("./settings")
    source = one("./settings/source")
    space = one("./settings/source/space")
    angle = one("./settings/source/angle")
    energy = one("./settings/source/energy")
    dagmc = one("./geometry/dagmc_universe")
    wrapper_cell = one("./geometry/cell")
    wrapper_surface = one("./geometry/surface")
    materials = root.findall("./materials/material")
    cross_sections = one("./materials/cross_sections")
    point = [
        float(value) for value in (space.findtext("parameters") or "").split()
    ]
    energy_parameters = [
        float(value) for value in (energy.findtext("parameters") or "").split()
    ]
    if len(point) != 3 or not np.all(np.isfinite(point)):
        raise ValueError("model XML point source is invalid")
    if len(energy_parameters) != 2 or not np.all(
        np.isfinite(energy_parameters)
    ):
        raise ValueError("model XML discrete energy is invalid")
    normalized = copy.deepcopy(root)
    normalized_seed = normalized.find("./settings/seed")
    if normalized_seed is None:
        raise ValueError("model XML seed is missing")
    normalized_seed.text = "REGISTERED_SEED"
    normalized_bytes = ET.tostring(normalized, encoding="utf-8")
    auto_geom_ids = dagmc.get("auto_geom_ids")
    if auto_geom_ids is None:
        auto_geom_ids = "false"
    auto_geom_ids = auto_geom_ids.lower()
    if auto_geom_ids not in {"true", "false"}:
        raise ValueError("DAGMC auto_geom_ids value is invalid")

    material_contract = []
    for material in materials:
        density = material.find("./density")
        if density is None:
            raise ValueError("material density is missing")
        nuclides = material.findall("./nuclide")
        material_contract.append(
            {
                "id": int(material.get("id", "-1")),
                "name": material.get("name"),
                "density": {
                    "units": density.get("units"),
                    "value": float(density.get("value", "nan")),
                },
                "nuclides": sorted(
                    [
                        {
                            "name": nuclide.get("name"),
                            "percent_type": (
                                "ao" if nuclide.get("ao") is not None else "wo"
                            ),
                            "percent": float(
                                nuclide.get("ao")
                                if nuclide.get("ao") is not None
                                else nuclide.get("wo", "nan")
                            ),
                        }
                        for nuclide in nuclides
                    ],
                    key=lambda row: row["name"],
                ),
            }
        )
    surface_coefficients = [
        float(value) for value in (wrapper_surface.get("coeffs") or "").split()
    ]
    if len(surface_coefficients) != 4 or not np.all(
        np.isfinite(surface_coefficients)
    ):
        raise ValueError("wrapper sphere coefficients are invalid")
    return {
        "model_xml_sha256": sha256_file(path),
        "seed_normalized_fingerprint": hashlib.sha256(
            normalized_bytes
        ).hexdigest(),
        "run_mode": settings.findtext("run_mode"),
        "batches": int(settings.findtext("batches", "-1")),
        "particles": int(settings.findtext("particles", "-1")),
        "seed": int(settings.findtext("seed", "-1")),
        "source_particle": source.get("particle"),
        "source_space_type": space.get("type"),
        "source_point_cm": point,
        "source_angle_type": angle.get("type"),
        "source_energy_type": energy.get("type"),
        "source_energy_parameters_ev": energy_parameters,
        "dagmc_filename": dagmc.get("filename"),
        "dagmc_universe_id": int(dagmc.get("id", "-1")),
        "dagmc_auto_geom_ids": auto_geom_ids,
        "wrapper_surface": {
            "id": int(wrapper_surface.get("id", "-1")),
            "type": wrapper_surface.get("type"),
            "boundary": wrapper_surface.get("boundary"),
            "coefficients_cm": surface_coefficients,
        },
        "wrapper_cell": {
            "id": int(wrapper_cell.get("id", "-1")),
            "fill": int(wrapper_cell.get("fill", "-1")),
            "region": wrapper_cell.get("region"),
            "universe": int(wrapper_cell.get("universe", "-1")),
        },
        "cross_sections_path": cross_sections.text,
        "materials": sorted(material_contract, key=lambda row: row["id"]),
    }


def generate(
    dagmc_path: Path,
    source_mesh: Path,
    output: Path,
    cross_sections: str,
    *,
    expected_dagmc_sha256: str,
    expected_source_mesh_sha256: str,
    acceptance_criteria: Path,
    expected_acceptance_criteria_sha256: str,
    source_domain_audit: Path,
    expected_source_domain_audit_sha256: str,
    expected_reference_source_mesh_sha256: str,
    expected_reference_source_fingerprint: str,
    expected_nuclear_data_manifest_sha256: str,
    seed: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    cross_sections_path = Path(cross_sections).resolve()
    if not cross_sections_path.is_file():
        raise FileNotFoundError(cross_sections_path)
    data_manifest = nuclear_data_manifest(cross_sections_path)
    if (
        data_manifest["manifest_sha256"]
        != expected_nuclear_data_manifest_sha256
    ):
        raise ValueError("nuclear-data manifest hash mismatch")
    if sha256_file(source_mesh) != expected_source_mesh_sha256:
        raise ValueError("source mesh hash mismatch")
    if sha256_file(source_domain_audit) != expected_source_domain_audit_sha256:
        raise ValueError("source-domain receipt hash mismatch")
    criteria_sha256 = sha256_file(acceptance_criteria)
    if criteria_sha256 != expected_acceptance_criteria_sha256:
        raise ValueError("acceptance-criteria hash mismatch")
    criteria = _read_json(acceptance_criteria)
    registered = criteria["openmc_geometry_debug"]
    if int(seed) not in [int(value) for value in registered["seeds"]]:
        raise ValueError("seed is not preregistered")
    if int(registered["histories"]) != 4000:
        raise ValueError("geometry-debug history count is not 4000")
    domain = _read_json(source_domain_audit)
    if domain.get("schema") != "parastell.source_domain_audit/v1.0.0":
        raise ValueError("source-domain receipt schema mismatch")
    if domain.get("source_domain_gate_pass") is not True:
        raise ValueError("source-domain gate has not passed")
    if domain.get("raw_h5m_sha256") != expected_dagmc_sha256:
        raise ValueError("source-domain H5M binding mismatch")
    if domain.get("source_mesh_sha256") != expected_source_mesh_sha256:
        raise ValueError("source-domain mesh binding mismatch")
    if domain.get("acceptance_criteria_sha256") != criteria_sha256:
        raise ValueError("source-domain criteria binding mismatch")
    if (
        domain.get("reference_source_mesh_sha256")
        != expected_reference_source_mesh_sha256
    ):
        raise ValueError("source-domain reference-source hash mismatch")
    if (
        domain.get("expected_reference_source_mesh_sha256")
        != expected_reference_source_mesh_sha256
    ):
        raise ValueError("source-domain expected reference-source mismatch")
    if (
        domain.get("reference_source_mesh_identity", {}).get(
            "canonical_fingerprint"
        )
        != expected_reference_source_fingerprint
    ):
        raise ValueError("source-domain reference fingerprint mismatch")
    if (
        domain.get("expected_reference_source_fingerprint")
        != expected_reference_source_fingerprint
    ):
        raise ValueError(
            "source-domain expected reference fingerprint mismatch"
        )
    required_zero_fields = (
        "invalid_tetrahedron_count",
        "negative_source_strength_count",
        "invalid_vertex_sample_count",
        "invalid_quadrature_point_count",
        "invalid_sample_point_count",
    )
    if any(domain.get(field) != 0 for field in required_zero_fields):
        raise ValueError("source-domain receipt contains invalid source data")
    if domain.get("source_mesh_semantic_identity_pass") is not True:
        raise ValueError("source-domain semantic identity did not pass")
    if domain.get("reference_source_fingerprint_matches_expected") is not True:
        raise ValueError(
            "source-domain expected-reference binding did not pass"
        )
    if domain.get("input_immutability_pass") is not True:
        raise ValueError("source-domain input immutability did not pass")
    source_point = [float(item) for item in domain["selected_source_point_cm"]]
    if len(source_point) != 3 or not np.all(np.isfinite(source_point)):
        raise ValueError("source-domain audit has an invalid source point")

    import openmc

    if openmc.__version__ != "0.16.0":
        raise RuntimeError(
            f"OpenMC 0.16.0 is required, got {openmc.__version__}"
        )

    openmc.reset_auto_ids()
    reference = ReferenceGeometry.open(
        dagmc_path, expected_sha256=expected_dagmc_sha256
    )
    before = reference.accepted_sha256
    materials = openmc.Materials(
        [
            _material(openmc, 1, "Vacuum", "H1", 1.0e-20),
            _material(openmc, 2, "first_wall", "Fe56", 7.8),
            _material(openmc, 3, "breeder", "Li6", 1.0),
            _material(openmc, 4, "back_wall", "Fe56", 7.8),
            _material(openmc, 5, "shield", "Fe56", 7.8),
            _material(openmc, 6, "vac_vessel", "Fe56", 7.8),
            _material(openmc, 7, "magnets", "Cu63", 8.96),
        ]
    )
    materials.cross_sections = str(cross_sections_path)
    source = openmc.IndependentSource(
        space=openmc.stats.Point(source_point),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([14.1e6], [1.0]),
        particle="neutron",
    )
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = 2
    settings.particles = 2000
    settings.seed = int(seed)
    settings.source = source
    wrapper_radius_cm = 2500.0
    maximum_vertex_radius_cm = _maximum_vertex_radius(dagmc_path)
    wrapper_margin_cm = wrapper_radius_cm - maximum_vertex_radius_cm
    if not np.isfinite(wrapper_margin_cm) or wrapper_margin_cm <= 0.0:
        raise ValueError("external CSG sphere does not contain the DAGMC H5M")
    native_ids = native_dagmc_id_inventory(dagmc_path)
    first_wrapper_id = int(native_ids["maximum_native_id"]) + 1
    geometry = reference.openmc_geometry(
        external_vacuum_radius_cm=wrapper_radius_cm
    )
    model = openmc.Model(
        geometry=geometry, materials=materials, settings=settings
    )
    output.mkdir(parents=True, exist_ok=False)
    model.export_to_model_xml(output / "model.xml")
    contract = model_xml_contract(output / "model.xml")
    expected_contract = {
        "run_mode": "fixed source",
        "batches": 2,
        "particles": 2000,
        "seed": int(seed),
        "source_particle": "neutron",
        "source_space_type": "point",
        "source_point_cm": source_point,
        "source_angle_type": "isotropic",
        "source_energy_type": "discrete",
        "source_energy_parameters_ev": [14.1e6, 1.0],
        "dagmc_filename": str(dagmc_path.resolve()),
        "dagmc_universe_id": first_wrapper_id + 3,
        "dagmc_auto_geom_ids": "false",
        "wrapper_surface": {
            "id": first_wrapper_id,
            "type": "sphere",
            "boundary": "vacuum",
            "coefficients_cm": [0.0, 0.0, 0.0, wrapper_radius_cm],
        },
        "wrapper_cell": {
            "id": first_wrapper_id + 1,
            "fill": first_wrapper_id + 3,
            "region": f"-{first_wrapper_id}",
            "universe": first_wrapper_id + 2,
        },
        "cross_sections_path": str(cross_sections_path),
        "materials": [
            {
                "id": 1,
                "name": "Vacuum",
                "density": {"units": "g/cm3", "value": 1.0e-20},
                "nuclides": [
                    {"name": "H1", "percent_type": "ao", "percent": 1.0}
                ],
            },
            {
                "id": 2,
                "name": "first_wall",
                "density": {"units": "g/cm3", "value": 7.8},
                "nuclides": [
                    {"name": "Fe56", "percent_type": "ao", "percent": 1.0}
                ],
            },
            {
                "id": 3,
                "name": "breeder",
                "density": {"units": "g/cm3", "value": 1.0},
                "nuclides": [
                    {"name": "Li6", "percent_type": "ao", "percent": 1.0}
                ],
            },
            {
                "id": 4,
                "name": "back_wall",
                "density": {"units": "g/cm3", "value": 7.8},
                "nuclides": [
                    {"name": "Fe56", "percent_type": "ao", "percent": 1.0}
                ],
            },
            {
                "id": 5,
                "name": "shield",
                "density": {"units": "g/cm3", "value": 7.8},
                "nuclides": [
                    {"name": "Fe56", "percent_type": "ao", "percent": 1.0}
                ],
            },
            {
                "id": 6,
                "name": "vac_vessel",
                "density": {"units": "g/cm3", "value": 7.8},
                "nuclides": [
                    {"name": "Fe56", "percent_type": "ao", "percent": 1.0}
                ],
            },
            {
                "id": 7,
                "name": "magnets",
                "density": {"units": "g/cm3", "value": 8.96},
                "nuclides": [
                    {"name": "Cu63", "percent_type": "ao", "percent": 1.0}
                ],
            },
        ],
    }
    contract_gate = all(
        contract[key] == value for key, value in expected_contract.items()
    )
    if not contract_gate:
        raise ValueError("exported model XML differs from the frozen contract")
    reference.verify_unchanged()
    after = sha256_file(dagmc_path)
    receipt = {
        "schema": "parastell.openmc_geometry_debug_input/v1.2.0",
        "openmc_version": openmc.__version__,
        "geometry_mode": reference.geometry_mode,
        "dagmc_path": str(reference.path),
        "raw_h5m_sha256_before": before,
        "raw_h5m_sha256_after_model_export": after,
        "h5m_mutated": after != before,
        "source_mesh_path": str(source_mesh),
        "source_mesh_sha256": expected_source_mesh_sha256,
        "acceptance_criteria_path": str(acceptance_criteria),
        "acceptance_criteria_sha256": criteria_sha256,
        "source_domain_audit_path": str(source_domain_audit),
        "source_domain_audit_sha256": sha256_file(source_domain_audit),
        "expected_source_domain_audit_sha256": (
            expected_source_domain_audit_sha256
        ),
        "native_geometry_ids": native_ids,
        "external_csg_vacuum_boundary": {
            "kind": "sphere",
            "radius_cm": wrapper_radius_cm,
            "surface_id": first_wrapper_id,
            "wrapper_cell_id": first_wrapper_id + 1,
            "root_universe_id": first_wrapper_id + 2,
            "dagmc_universe_id": first_wrapper_id + 3,
            "maximum_dagmc_vertex_radius_cm": maximum_vertex_radius_cm,
            "minimum_containment_margin_cm": wrapper_margin_cm,
            "location": "OpenMC model.xml only; not written to H5M",
        },
        "source_point_cm": source_point,
        "batches": 2,
        "particles_per_batch": 2000,
        "histories": 4000,
        "seed": int(seed),
        "cross_sections": str(cross_sections_path),
        "cross_sections_sha256": sha256_file(cross_sections_path),
        "nuclear_data_manifest": data_manifest,
        "nuclear_data_manifest_sha256": data_manifest["manifest_sha256"],
        "expected_nuclear_data_manifest_sha256": (
            expected_nuclear_data_manifest_sha256
        ),
        "reference_source_mesh_sha256": (
            expected_reference_source_mesh_sha256
        ),
        "reference_source_fingerprint": expected_reference_source_fingerprint,
        "model_xml_contract": contract,
        "model_xml_contract_pass": contract_gate,
    }
    (output / "input_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc", type=Path)
    parser.add_argument("source_mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cross-sections", required=True)
    parser.add_argument("--expected-dagmc-sha256", required=True)
    parser.add_argument("--expected-source-mesh-sha256", required=True)
    parser.add_argument("--acceptance-criteria", required=True, type=Path)
    parser.add_argument("--expected-acceptance-criteria-sha256", required=True)
    parser.add_argument("--source-domain-audit", required=True, type=Path)
    parser.add_argument("--expected-source-domain-audit-sha256", required=True)
    parser.add_argument(
        "--expected-reference-source-mesh-sha256", required=True
    )
    parser.add_argument(
        "--expected-reference-source-fingerprint", required=True
    )
    parser.add_argument(
        "--expected-nuclear-data-manifest-sha256", required=True
    )
    parser.add_argument("--seed", required=True, type=int)
    arguments = parser.parse_args()
    generate(
        arguments.dagmc.resolve(),
        arguments.source_mesh.resolve(),
        arguments.output.resolve(),
        arguments.cross_sections,
        expected_dagmc_sha256=arguments.expected_dagmc_sha256,
        expected_source_mesh_sha256=arguments.expected_source_mesh_sha256,
        acceptance_criteria=arguments.acceptance_criteria.resolve(),
        expected_acceptance_criteria_sha256=(
            arguments.expected_acceptance_criteria_sha256
        ),
        source_domain_audit=arguments.source_domain_audit.resolve(),
        expected_source_domain_audit_sha256=(
            arguments.expected_source_domain_audit_sha256
        ),
        expected_reference_source_mesh_sha256=(
            arguments.expected_reference_source_mesh_sha256
        ),
        expected_reference_source_fingerprint=(
            arguments.expected_reference_source_fingerprint
        ),
        expected_nuclear_data_manifest_sha256=(
            arguments.expected_nuclear_data_manifest_sha256
        ),
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
