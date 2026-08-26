"""Restartable OpenMC 0.16 model preparation for magnet-radiation fields."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dt_source import build_temperature_dependent_mesh_source
from .dt_source import read_tagged_source_mesh
from .energy_groups import get_structure
from .dagmc_envelope import canonical_geometry_policy
from .magnet_local_mesh import LocalMeshDefinition
from .magnet_radiation_field import MagnetRadiationFieldProducer
from .magnet_radiation_field import ProducerSelection
from .material_manifest import openmc_materials_from_manifest


SCHEMA = "parastell.magnet_openmc_model/v1.0.0"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_mesh(value: Mapping[str, Any]) -> LocalMeshDefinition:
    fields = LocalMeshDefinition.__dataclass_fields__
    return LocalMeshDefinition(
        **{name: value[name] for name in fields if name in value}
    )


def _material_binding(
    material_manifest: Mapping[str, Any],
    material_ids_by_tag: Mapping[str, int],
    dagmc_material_tag: str,
) -> tuple[str, int]:
    """Return the resolved record name and OpenMC ID for a DAGMC tag."""
    tag = str(dagmc_material_tag)
    try:
        material_name = str(material_manifest["material_tags"][tag])
    except KeyError as exc:
        raise ValueError(
            f"DAGMC material tag {tag!r} is not configured"
        ) from exc
    try:
        material_id = int(material_ids_by_tag[tag])
    except KeyError as exc:
        raise ValueError(
            f"OpenMC material for DAGMC tag {tag!r} was not constructed"
        ) from exc
    return material_name, material_id


def prepare_magnet_openmc_model(
    output_directory: str | Path,
    *,
    dagmc_path: str | Path,
    source_mesh_path: str | Path,
    material_manifest_path: str | Path,
    cross_sections_path: str | Path,
    associations_path: str | Path,
    magnet_selection: str | int | Sequence[str | int] = "all",
    tally_profile: str = "magnet_damage_and_handoff",
    neutron_edges_eV: Sequence[float],
    photon_edges_eV: Sequence[float],
    particles_per_batch: int,
    batches: int,
    seed: int,
    max_surface_particles: int,
    max_surface_files: int = 1,
    local_mesh_manifest_path: str | Path | None = None,
    supported_responses: Sequence[str] | None = None,
    temperature_method: str = "nearest",
    temperature_tolerance_K: float = 1000.0,
    temperature_policy_justification: str = "",
    coordinate_quantum_cm: float | None = None,
    faceting_tolerances: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare XML and contracts only; execution remains a separate stage."""
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required to prepare transport XML"
        ) from exc
    if tuple(
        int(value) for value in openmc.__version__.split("+")[0].split(".")[:2]
    ) < (0, 16):
        raise RuntimeError("OpenMC 0.16 or newer is required")
    if (
        min(
            particles_per_batch,
            batches,
            seed,
            max_surface_particles,
            max_surface_files,
        )
        <= 0
    ):
        raise ValueError(
            "OpenMC history, seed, and surface-bank controls must be positive"
        )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dagmc_path = Path(dagmc_path).resolve()
    source_mesh_path = Path(source_mesh_path).resolve()
    cross_sections_path = Path(cross_sections_path).resolve()
    associations = json.loads(
        Path(associations_path).read_text(encoding="utf-8")
    )
    artifact_policy = dict(associations.get("canonical_geometry_policy", {}))
    geometry_policy = canonical_geometry_policy(
        (
            coordinate_quantum_cm
            if coordinate_quantum_cm is not None
            else artifact_policy.get("coordinate_quantum_cm", 1.0e-6)
        ),
        (
            faceting_tolerances
            if faceting_tolerances is not None
            else artifact_policy.get("faceting_tolerances", {})
        ),
    )
    if artifact_policy:
        expected_policy = canonical_geometry_policy(
            artifact_policy.get("coordinate_quantum_cm", 1.0e-6),
            artifact_policy.get("faceting_tolerances", {}),
        )
        if geometry_policy != expected_policy:
            raise ValueError(
                "transport canonical geometry policy disagrees with the "
                "association artifact"
            )
    material_manifest = json.loads(
        Path(material_manifest_path).read_text(encoding="utf-8")
    )
    producer = MagnetRadiationFieldProducer(
        dagmc_path,
        selection=ProducerSelection(
            magnet_selection=magnet_selection,
            tally_profile=tally_profile,
        ),
        associations={
            int(key): value
            for key, value in associations["associations"].items()
        },
        centreline_points_by_coil=associations["centreline_points_by_coil"],
        **geometry_policy,
        expected_canonical_geometry_fingerprint=associations.get(
            "canonical_geometry_fingerprint"
        ),
    )
    producer.discover()
    producer.build_envelopes()
    dagmc = openmc.DAGMCUniverse(str(dagmc_path), auto_geom_ids=False)
    bounds = dagmc.bounding_box
    radius = float(
        np.max(np.abs(np.concatenate((bounds.lower_left, bounds.upper_right))))
        * 1.2
    )
    world = openmc.Sphere(
        surface_id=10_000_001, r=radius, boundary_type="vacuum"
    )
    geometry = openmc.Geometry(
        [openmc.Cell(cell_id=10_000_001, fill=dagmc, region=-world)]
    )
    source_state = read_tagged_source_mesh(source_mesh_path)
    mesh_source, source_audit = build_temperature_dependent_mesh_source(
        source_state, source_mesh_path
    )
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = int(particles_per_batch)
    settings.batches = int(batches)
    settings.seed = int(seed)
    settings.source = [mesh_source]
    settings.temperature = {
        "method": temperature_method,
        "tolerance": float(temperature_tolerance_K),
    }
    settings.output = {"tallies": False, "summary": True}
    surface_ids = sorted(
        {
            surface_id
            for envelope in producer.envelopes
            for surface_id in envelope.envelope.surface_ids
        }
    )
    settings.surf_source_write = {
        "surface_ids": surface_ids,
        "max_particles": int(max_surface_particles),
        "max_source_files": int(max_surface_files),
    }
    materials = openmc_materials_from_manifest(material_manifest)
    materials.cross_sections = str(cross_sections_path)
    material_ids_by_tag = {
        str(material.name): int(material.id) for material in materials
    }
    material_ids_by_name = {
        str(material_manifest["material_tags"][tag]): material_id
        for tag, material_id in material_ids_by_tag.items()
    }
    dagmc_openmc_cell_map = []
    for pair in producer.selected_pairs:
        for component in (pair.winding_pack, pair.casing):
            if component is None:
                continue
            material_name, openmc_material_id = _material_binding(
                material_manifest,
                material_ids_by_tag,
                component.material,
            )
            dagmc_openmc_cell_map.append(
                {
                    "magnet_id": pair.magnet_id,
                    "component_role": component.component_role,
                    "dagmc_volume_id": int(component.volume_id),
                    "openmc_cell_id": int(component.volume_id),
                    "material_tag": component.material,
                    "material_name": material_name,
                    "openmc_material_id": openmc_material_id,
                    "mapping_basis": (
                        "OpenMC DAGMCUniverse(auto_geom_ids=False) retains "
                        "DAGMC volume IDs as DAGMC cell IDs"
                    ),
                    "transport_statepoint_verified": False,
                }
            )
    model = openmc.Model(
        geometry=geometry, materials=materials, settings=settings
    )
    local_filters = {}
    local_mesh_manifest = None
    if local_mesh_manifest_path is not None:
        local_mesh_manifest = json.loads(
            Path(local_mesh_manifest_path).read_text(encoding="utf-8")
        )
        pair_by_magnet = {
            pair.magnet_id: pair for pair in producer.selected_pairs
        }
        for magnet_id, value in local_mesh_manifest["meshes"].items():
            if magnet_id not in pair_by_magnet:
                continue
            local_filters[pair_by_magnet[magnet_id].winding_pack.volume_id] = (
                _local_mesh(value).openmc_filter()
            )
    volume_axes = {
        "neutron_configured_fine": ("neutron", neutron_edges_eV),
        "neutron_ccfe_709": (
            "neutron",
            get_structure("CCFE-709", particle="neutron").edges_eV,
        ),
        "neutron_ukaea_1102": (
            "neutron",
            get_structure("UKAEA-1102", particle="neutron").edges_eV,
        ),
        "photon_configured": ("photon", photon_edges_eV),
    }
    tally_inventory = producer.attach_openmc(
        model,
        neutron_edges_eV=neutron_edges_eV,
        photon_edges_eV=photon_edges_eV,
        volume_flux_energy_axes=volume_axes,
        local_mesh_filters_by_cell=local_filters,
        supported_responses=supported_responses,
    )
    model.export_to_xml(directory=output)
    xml_files = {
        path.name: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.glob("*.xml"))
    }
    manifest = {
        "schema": SCHEMA,
        "openmc_version": openmc.__version__,
        "dagmc": {
            "path": str(dagmc_path),
            "raw_h5m_sha256": _sha256(dagmc_path),
            "canonical_geometry_fingerprint": producer.inventory.canonical_geometry_fingerprint,
            "canonical_geometry_policy": geometry_policy,
        },
        "source": asdict(source_audit),
        "source_mesh_path": str(source_mesh_path),
        "physical_source_rate_per_s": source_audit.source_rate_per_s,
        "material_manifest": {
            "path": str(Path(material_manifest_path).resolve()),
            "sha256": _sha256(material_manifest_path),
            "resolved_manifest_sha256": material_manifest[
                "resolved_manifest_sha256"
            ],
        },
        "openmc_material_ids_by_name": material_ids_by_name,
        "openmc_material_ids_by_tag": material_ids_by_tag,
        "dagmc_openmc_cell_map": dagmc_openmc_cell_map,
        "cross_sections": {
            "path": str(cross_sections_path),
            "sha256": _sha256(cross_sections_path),
        },
        "histories": int(particles_per_batch) * int(batches),
        "particles_per_batch": int(particles_per_batch),
        "batches": int(batches),
        "seed": int(seed),
        "temperature_policy": {
            "method": temperature_method,
            "tolerance_K": float(temperature_tolerance_K),
            "justification": temperature_policy_justification,
        },
        "openmc_output_policy": {
            "statepoint": True,
            "summary": True,
            "human_readable_tallies_dump": False,
        },
        "surface_source": settings.surf_source_write,
        "producer": producer.manifest(),
        "tallies": tally_inventory.to_dict(),
        "local_mesh_manifest": local_mesh_manifest,
        "xml_files": xml_files,
        "execution_performed": False,
    }
    manifest_path = output / "magnet_openmc_model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
