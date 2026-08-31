"""Hash-bound OpenMC 0.16 model construction for direct-period ParaStell CAD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np

from .reference_geometry import ReferenceGeometry, sha256_file
from .source_domain import _source_arrays, audit_source_tetrahedra_arrays
from .continuous_radial_contract import COMPONENT_ORDER


CONTROL_SCHEMA = "parastell.parametric_openmc16_build_control/v1.0.0"
RECEIPT_SCHEMA = "parastell.parametric_openmc16_model/v1.0.0"
MATERIAL_NAMES = {
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "magnets",
}
CONTINUOUS_MATERIAL_NAMES = (MATERIAL_NAMES - {"magnets"}) | {
    "homogenized_magnet"
}
SEMANTIC_ROLES = (
    "chamber",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "vacuum_gap",
    *(f"magnet-{index:04d}" for index in range(18)),
)
CONTINUOUS_GEOMETRY_MODE = "continuous_radial_envelope"
LEGACY_GEOMETRY_MODE = "swept_magnets_legacy"


def _geometry_mode(control: Mapping[str, Any]) -> str:
    mode = str(control.get("geometry_mode", LEGACY_GEOMETRY_MODE))
    if mode not in {CONTINUOUS_GEOMETRY_MODE, LEGACY_GEOMETRY_MODE}:
        raise ValueError(f"unsupported geometry_mode {mode!r}")
    return mode


def _geometry_contract(control: Mapping[str, Any]) -> dict[str, Any]:
    """Return physical cell identities without inventing engineering solids."""
    if _geometry_mode(control) == CONTINUOUS_GEOMETRY_MODE:
        return {
            "geometry_mode": CONTINUOUS_GEOMETRY_MODE,
            "semantic_roles": tuple(COMPONENT_ORDER),
            "component_cell_ids": {
                role: index + 1 for index, role in enumerate(COMPONENT_ORDER)
            },
            "magnet_cell_ids": {"continuous-magnet-layer": 9},
            "physical_volume_count": 9,
        }
    return {
        "geometry_mode": LEGACY_GEOMETRY_MODE,
        "semantic_roles": SEMANTIC_ROLES,
        "component_cell_ids": {
            role: index + 1 for index, role in enumerate(SEMANTIC_ROLES[:8])
        },
        "magnet_cell_ids": {
            f"magnet-{index:04d}": index + 9 for index in range(18)
        },
        "physical_volume_count": 26,
    }


def _expected_material_names(control: Mapping[str, Any]) -> set[str]:
    return (
        CONTINUOUS_MATERIAL_NAMES
        if _geometry_mode(control) == CONTINUOUS_GEOMETRY_MODE
        else MATERIAL_NAMES
    )


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _statepoint_batches(run: Mapping[str, Any]) -> list[int]:
    """Return the explicit, periodic statepoint schedule for one run.

    OpenMC always receives an explicit final checkpoint.  A declared interval
    additionally preserves intermediate tally moments for long poster/research
    runs without changing source histories or tally definitions.
    """
    batches = int(run["batches"])
    interval = run.get("statepoint_interval_batches")
    if interval is None:
        return [batches]
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
        or interval > batches
    ):
        raise ValueError(
            "statepoint_interval_batches must be a positive integer no "
            "larger than run.batches"
        )
    checkpoints = list(range(interval, batches + 1, interval))
    if not checkpoints or checkpoints[-1] != batches:
        checkpoints.append(batches)
    return checkpoints


def _bound_path(row: Mapping[str, Any], label: str) -> Path:
    if set(row) != {"path", "sha256"}:
        raise ValueError(f"{label} path/hash binding is incomplete")
    path = Path(str(row["path"])).resolve(strict=True)
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return path


def _load_control(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if sha256_file(path) != expected_sha256:
        raise ValueError("build-control hash mismatch")
    control = json.loads(path.read_text(encoding="utf-8"))
    if control.get("schema") != CONTROL_SCHEMA:
        raise ValueError("unsupported parametric OpenMC build-control schema")
    return control


def _validate_geometry_evidence(
    control: Mapping[str, Any],
    *,
    dagmc_hash: str,
    export_receipt: Mapping[str, Any],
    premesh: Mapping[str, Any],
    writeback: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _geometry_contract(control)
    if contract["geometry_mode"] == CONTINUOUS_GEOMETRY_MODE:
        geometry = export_receipt.get("geometry", {})
        topology = premesh.get("topology", {})
        if (
            export_receipt.get("schema")
            != "parastell.continuous_radial_direct90_dagmc/v1.0.0"
            or export_receipt.get("status")
            != "EXPORTED_NATIVE_DAGMC_AND_OPENMC_GATES_PENDING"
            or export_receipt.get("h5m", {}).get("sha256") != dagmc_hash
            or geometry.get("extent_degrees") != 90.0
            or geometry.get("direct_parastell_full_period") is not True
            or geometry.get("explicit_swept_coils") is not False
            or geometry.get("physical_h5m_mutation") is not False
            or geometry.get("volume_count") != 9
            or geometry.get("continuous_magnet_volume_global_id") != 9
            or tuple(geometry.get("component_order", ()))
            != tuple(COMPONENT_ORDER)
        ):
            raise ValueError(
                "DAGMC export receipt is not the continuous direct 90-degree model"
            )
        if (
            premesh.get("schema")
            != "parastell.continuous_radial_premesh_topology/v1.0.0"
            or premesh.get("status") != "PREMESH_9_VOLUME_TOPOLOGY_PASS"
            or premesh.get("magnet_representation")
            != "continuous_30_cm_radial_envelope"
            or tuple(premesh.get("component_order", ()))
            != tuple(COMPONENT_ORDER)
            or topology.get("status")
            != "PREMESH_OCC_INCIDENCE_AND_MANIFOLD_PASS"
            or topology.get("volume_count") != 9
            or topology.get("surface_count") != 27
        ):
            raise ValueError("continuous premesh topology proof is invalid")
        if (
            writeback.get("schema")
            != "parastell.continuous_radial_direct90_h5m_writeback/v1.0.0"
            or writeback.get("pass") is not True
            or writeback.get("h5m_sha256") != dagmc_hash
            or writeback.get("volume_count") != 9
            or writeback.get("continuous_magnet_volume_global_id") != 9
            or tuple(writeback.get("semantic_roles_by_global_volume_id", ()))
            != tuple(COMPONENT_ORDER)
        ):
            raise ValueError(
                "continuous H5M semantic writeback is not accepted"
            )
        return contract

    if (
        export_receipt.get("schema")
        != "parastell.parametric_direct90_dagmc_export/v1.0.0"
        or export_receipt.get("status")
        != "DAGMC_EXPORTED_NATIVE_GATES_PENDING"
        or export_receipt.get("h5m", {}).get("sha256") != dagmc_hash
        or export_receipt.get("geometry", {}).get("extent_degrees") != 90.0
        or export_receipt.get("geometry", {}).get(
            "combined_from_45_degree_models"
        )
        is not False
        or export_receipt.get("geometry", {}).get("radial_volume_count") != 8
        or export_receipt.get("geometry", {}).get("magnet_volume_count") != 18
        or export_receipt.get("geometry", {}).get("physical_volume_count")
        != 26
        or export_receipt.get("geometry", {}).get("ports") is not False
        or export_receipt.get("geometry", {}).get("casing_winding_split")
        is not False
        or export_receipt.get("premesh_topology", {}).get("status")
        != "PREMESH_26_VOLUME_TOPOLOGY_PASS"
    ):
        raise ValueError(
            "DAGMC export receipt is not the direct 90-degree model"
        )
    if (
        premesh.get("schema") != "parastell.parametric_direct90_premesh/v1.0.0"
        or premesh.get("status") != "PREMESH_TOPOLOGY_PASS_MESH_PENDING"
        or tuple(premesh.get("semantic_roles", ())) != SEMANTIC_ROLES
        or premesh.get("topology", {}).get("status")
        != "PREMESH_26_VOLUME_TOPOLOGY_PASS"
    ):
        raise ValueError("premesh periodic topology proof is invalid")
    identities = writeback.get("magnet_identity_by_global_volume_id")
    if (
        writeback.get("schema")
        != "parastell.parametric_direct90_h5m_writeback/v1.0.0"
        or writeback.get("pass") is not True
        or writeback.get("h5m_sha256") != dagmc_hash
        or tuple(writeback.get("semantic_roles_by_global_volume_id", ()))
        != SEMANTIC_ROLES
        or not isinstance(identities, list)
        or len(identities) != 18
    ):
        raise ValueError("H5M semantic writeback is not accepted")
    for index, row in enumerate(identities):
        if (
            row.get("magnet_id") != f"magnet-{index:04d}"
            or row.get("volume_global_id") != index + 9
            or row.get("unique_match") is not True
            or row.get("pass") is not True
        ):
            raise ValueError("H5M magnet identity mapping is invalid")
    return contract


def _validate_evidence(control: Mapping[str, Any]) -> dict[str, Path]:
    required = {
        "dagmc_h5m",
        "source_mesh_h5m",
        "materials_xml",
        "cross_sections_xml",
        "geometry_gate_receipt",
        "dagmc_export_receipt",
        "premesh_topology",
        "source_domain_receipt",
        "h5m_writeback",
        "source_physics_manifest",
    }
    if set(control.get("inputs", {})) != required:
        raise ValueError("OpenMC build-control inputs are incomplete")
    paths = {
        key: _bound_path(control["inputs"][key], key)
        for key in sorted(required)
    }
    dagmc_hash = control["inputs"]["dagmc_h5m"]["sha256"]
    source_hash = control["inputs"]["source_mesh_h5m"]["sha256"]

    export_receipt = json.loads(
        paths["dagmc_export_receipt"].read_text(encoding="utf-8")
    )
    premesh = json.loads(paths["premesh_topology"].read_text(encoding="utf-8"))
    writeback = json.loads(paths["h5m_writeback"].read_text(encoding="utf-8"))
    _validate_geometry_evidence(
        control,
        dagmc_hash=dagmc_hash,
        export_receipt=export_receipt,
        premesh=premesh,
        writeback=writeback,
    )

    gate = json.loads(
        paths["geometry_gate_receipt"].read_text(encoding="utf-8")
    )
    watertight = gate.get("check_watertight", {})
    overlaps = gate.get("overlap_checks")
    if (
        gate.get("schema") != "parastell.dagmc_native_qualification/v1.0.0"
        or gate.get("native_dagmc_gate_pass") is not True
        or gate.get("raw_h5m_sha256_before") != dagmc_hash
        or gate.get("raw_h5m_sha256_after") != dagmc_hash
        or gate.get("h5m_unchanged") is not True
        or gate.get("native_id_inventory", {}).get("native_id_gate_pass")
        is not True
        or watertight.get("pass") is not True
        or watertight.get("unmatched_edge_count") != 0
        or watertight.get("unsealed_surface_count") != 0
        or watertight.get("unsealed_volume_count") != 0
        or not isinstance(overlaps, list)
        or [row.get("points_per_edge") for row in overlaps] != [1, 2, 4]
        or any(
            row.get("pass") is not True
            or row.get("terminal_overlap_location_count") != 0
            or row.get("parsed_overlap_location_count") != 0
            for row in overlaps
        )
    ):
        raise ValueError("native geometry receipt does not accept the H5M")

    domain = json.loads(
        paths["source_domain_receipt"].read_text(encoding="utf-8")
    )
    if (
        domain.get("schema") != "parastell.source_domain_audit/v1.0.0"
        or domain.get("source_domain_gate_pass") is not True
        or domain.get("raw_h5m_sha256") != dagmc_hash
        or domain.get("source_mesh_sha256") != source_hash
        or domain.get("source_volume_id") != 1
        or domain.get("source_component") != "chamber"
        or domain.get("source_material") != "Vacuum"
        or domain.get("input_immutability_pass") is not True
    ):
        raise ValueError("source-domain receipt is not accepted")

    source = json.loads(
        paths["source_physics_manifest"].read_text(encoding="utf-8")
    )
    if (
        source.get("schema") != "parastell.source_physics_manifest/v1.0.0"
        or source.get("particle") != "neutron"
        or source.get("angle") != "isotropic"
        or source.get("energy_law") != "discrete"
        or source.get("energy_eV") != [14_100_000.0]
        or source.get("probability") != [1.0]
        or source.get("claim") != "BOUNDED_SMOKE_ONLY"
    ):
        raise ValueError(
            "source-physics manifest is not the bounded smoke law"
        )

    if control.get("n_field_periods") != 4:
        raise ValueError("direct WISTELL-D model requires four field periods")
    if control.get("modeled_extent_degrees") != 90.0:
        raise ValueError("direct WISTELL-D model must span exactly 90 degrees")
    run = control.get("run", {})
    if set(run) not in (
        {"particles", "batches", "seed"},
        {
            "particles",
            "batches",
            "seed",
            "statepoint_interval_batches",
        },
    ) or any(
        isinstance(run.get(key), bool)
        or not isinstance(run.get(key), int)
        or run.get(key) <= 0
        for key in ("particles", "batches", "seed")
    ):
        raise ValueError("bounded run controls are invalid")
    _statepoint_batches(run)
    radius = float(control.get("external_vacuum_radius_cm", np.nan))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("external vacuum radius is invalid")
    return paths


def _maximum_vertex_radius(path: Path) -> float:
    from pymoab import core

    mesh = core.Core()
    mesh.load_file(str(path))
    vertices = mesh.get_entities_by_dimension(mesh.get_root_set(), 0)
    coordinates = np.asarray(mesh.get_coords(vertices), dtype=float).reshape(
        (-1, 3)
    )
    if not len(coordinates) or np.any(~np.isfinite(coordinates)):
        raise ValueError("DAGMC H5M has no finite vertices")
    return float(np.linalg.norm(coordinates, axis=1).max())


def _material_names(materials, expected: set[str] | None = None) -> set[str]:
    names = [str(material.name).strip() for material in materials]
    folded = [name.casefold() for name in names]
    if any(not name for name in names) or len(folded) != len(set(folded)):
        raise ValueError("OpenMC material names must be nonempty and unique")
    observed = {name for name in names if name != "Vacuum"}
    expected_names = MATERIAL_NAMES if expected is None else set(expected)
    if observed != expected_names:
        raise ValueError("OpenMC material names do not match DAGMC tags")
    return set(names)


def _nuclear_data_manifest(
    materials_xml: Path, cross_sections_xml: Path
) -> dict:
    """Hash every neutron and photoatomic file required by the materials."""
    material_root = ET.parse(materials_xml).getroot()
    if material_root.findall("./material/element"):
        raise ValueError(
            "bounded smoke materials must expand elements to nuclides"
        )
    if material_root.findall("./material/sab"):
        raise ValueError(
            "bounded smoke materials do not support S(alpha,beta)"
        )
    nuclides = sorted(
        {
            str(node.get("name", "")).strip()
            for node in material_root.findall("./material/nuclide")
        }
    )
    if not nuclides or any(not value for value in nuclides):
        raise ValueError("materials XML has no complete nuclide inventory")
    catalog_root = ET.parse(cross_sections_xml).getroot()
    libraries = catalog_root.findall("./library")

    def select(kind: str, material: str) -> Path:
        matches = [
            row
            for row in libraries
            if row.get("type") == kind
            and material in str(row.get("materials", "")).split()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cross-section catalog requires one {kind} library for {material}"
            )
        path = Path(str(matches[0].get("path", "")))
        if not path.is_absolute():
            path = cross_sections_xml.parent / path
        return path.resolve(strict=True)

    rows = []
    seen = set()
    for nuclide in nuclides:
        match = re.match(r"^([A-Z][a-z]?)", nuclide)
        if match is None:
            raise ValueError(f"cannot infer element from nuclide {nuclide!r}")
        for kind, material in (
            ("neutron", nuclide),
            ("photon", match.group(1)),
        ):
            path = select(kind, material)
            key = (kind, material, str(path))
            if key not in seen:
                rows.append(
                    {
                        "type": kind,
                        "material": material,
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
                seen.add(key)
    payload = {
        "cross_sections_xml_sha256": sha256_file(cross_sections_xml),
        "required_nuclides": nuclides,
        "libraries": rows,
    }
    payload["manifest_sha256"] = _canonical_sha(payload)
    return payload


def build_model(
    control_path: Path, expected_control_sha256: str, output: Path
):
    """Build and export one immutable, periodic OpenMC 0.16 model."""
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    control = _load_control(control_path, expected_control_sha256)
    paths = _validate_evidence(control)
    geometry_contract = _geometry_contract(control)
    immutable_hashes_before = {
        name: sha256_file(path) for name, path in sorted(paths.items())
    }
    import openmc

    if openmc.__version__ != "0.16.0":
        raise RuntimeError(f"OpenMC 0.16.0 required, got {openmc.__version__}")
    dagmc_hash = control["inputs"]["dagmc_h5m"]["sha256"]
    source_hash = control["inputs"]["source_mesh_h5m"]["sha256"]
    reference = ReferenceGeometry.open(
        paths["dagmc_h5m"], expected_sha256=dagmc_hash
    )
    radius = float(control["external_vacuum_radius_cm"])
    maximum_radius = _maximum_vertex_radius(paths["dagmc_h5m"])
    if radius <= maximum_radius:
        raise ValueError("external vacuum sphere does not enclose the H5M")
    geometry = reference.openmc_one_period_geometry(
        n_field_periods=4, external_vacuum_radius_cm=radius
    )
    materials = openmc.Materials.from_xml(path=paths["materials_xml"])
    material_names = _material_names(
        materials, _expected_material_names(control)
    )
    materials.cross_sections = str(paths["cross_sections_xml"])
    nuclear_data = _nuclear_data_manifest(
        paths["materials_xml"], paths["cross_sections_xml"]
    )

    tetrahedra, volumes, strengths = _source_arrays(paths["source_mesh_h5m"])
    source_audit = audit_source_tetrahedra_arrays(
        tetrahedra, volumes, strengths
    )
    if source_audit["tetrahedron_data_gate_pass"] is not True:
        raise ValueError("source-mesh tetrahedron data are invalid")
    mesh = openmc.UnstructuredMesh(str(paths["source_mesh_h5m"]), "moab")
    source_manifest = json.loads(
        paths["source_physics_manifest"].read_text(encoding="utf-8")
    )
    source = openmc.IndependentSource(
        space=openmc.stats.MeshSpatial(
            mesh, strengths=strengths.tolist(), volume_normalized=False
        ),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete(
            source_manifest["energy_eV"], source_manifest["probability"]
        ),
        particle="neutron",
    )
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = int(control["run"]["particles"])
    settings.batches = int(control["run"]["batches"])
    settings.seed = int(control["run"]["seed"])
    statepoint_batches = _statepoint_batches(control["run"])
    settings.statepoint = {"batches": statepoint_batches}
    settings.source = source
    settings.photon_transport = True
    model = openmc.Model(
        geometry=geometry, materials=materials, settings=settings
    )
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "model.xml"
    model.export_to_model_xml(model_path)
    reference.verify_unchanged()
    immutable_hashes_after = {
        name: sha256_file(path) for name, path in sorted(paths.items())
    }
    if immutable_hashes_after != immutable_hashes_before:
        raise RuntimeError(
            "one or more model inputs changed during XML export"
        )
    library_hashes_after = {
        row["path"]: sha256_file(Path(row["path"]))
        for row in nuclear_data["libraries"]
    }
    if library_hashes_after != {
        row["path"]: row["sha256"] for row in nuclear_data["libraries"]
    }:
        raise RuntimeError(
            "selected nuclear-data file changed during XML export"
        )
    strength_bytes = np.asarray(strengths, dtype="<f8").tobytes()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "MODEL_EXPORTED_TRANSPORT_PENDING",
        "claim": "BOUNDED_SMOKE_ONLY",
        "openmc_version": openmc.__version__,
        "control_sha256": expected_control_sha256,
        "dagmc_sha256": dagmc_hash,
        "source_mesh_sha256": source_hash,
        "materials_xml_sha256": control["inputs"]["materials_xml"]["sha256"],
        "cross_sections_xml_sha256": control["inputs"]["cross_sections_xml"][
            "sha256"
        ],
        "nuclear_data_manifest": nuclear_data,
        "material_names": sorted(material_names),
        "geometry_mode": geometry_contract["geometry_mode"],
        "physical_volume_count": geometry_contract["physical_volume_count"],
        "component_cell_ids": geometry_contract["component_cell_ids"],
        "modeled_extent_degrees": 90.0,
        "n_field_periods": 4,
        "transport_periodic_wrapper": {
            "period_degrees": 90.0,
            "periodic_planes_paired": True,
            "external_vacuum_radius_cm": radius,
            "maximum_h5m_vertex_radius_cm": maximum_radius,
        },
        "source": {
            "particle": "neutron",
            "mesh_strength_semantics": "integrated_rate_per_tetrahedron",
            "volume_normalized": False,
            "strength_count": int(len(strengths)),
            "strengths_sha256": hashlib.sha256(strength_bytes).hexdigest(),
            "physical_rate_n_per_s_for_modeled_90_degrees": float(
                np.sum(strengths)
            ),
            "normalization_application": "apply exactly once downstream",
        },
        "magnet_cell_ids": geometry_contract["magnet_cell_ids"],
        "photon_transport": True,
        "prompt_photons": "transported when produced by nuclear data",
        "delayed_photons": "separate activation-derived source",
        "statepoint_policy": {
            "explicit": True,
            "batches": statepoint_batches,
            "interval_batches": control["run"].get(
                "statepoint_interval_batches"
            ),
            "final_batch_included": statepoint_batches[-1]
            == int(control["run"]["batches"]),
            "purpose": (
                "intermediate poster/research recovery plus final results"
            ),
        },
        "model_xml": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "physical_h5m_mutation": False,
        "source_mesh_mutation": False,
        "all_bound_inputs_immutable": True,
        "selected_nuclear_data_immutable": True,
        "production_run_authorized": False,
    }
    receipt["receipt_content_sha256"] = _canonical_sha(receipt)
    receipt_path = output / "PARAMETRIC_OPENMC16_MODEL_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return model, receipt
