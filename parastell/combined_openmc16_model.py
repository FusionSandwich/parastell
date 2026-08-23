"""One-model ParaStell reactor, magnet, source, and handoff construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dagmc_envelope import DagmcEnvelope
from .dagmc_envelope import extract_closed_envelope
from .dagmc_envelope import extract_closed_envelopes
from .dt_source import DTSourceAudit
from .dt_source import build_temperature_dependent_mesh_source
from .openmc16 import TallyInventory
from .openmc16 import add_envelope_tallies
from .openmc16 import configure_transport
from .parastell import Stellarator
from .production_handoff import load_and_validate_no_port_configuration
from .utils import read_yaml_config


def _hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CombinedGeometryResult:
    dagmc_path: Path
    source_mesh_path: Path
    dagmc_sha256: str
    source_mesh_sha256: str
    material_tags: tuple[str, ...]
    source_mesh_shape: tuple[int, int, int]


@dataclass(frozen=True)
class CombinedModelResult:
    model: Any
    envelope: DagmcEnvelope
    source_audit: DTSourceAudit
    tally_inventory: TallyInventory
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CombinedMultiMagnetModelResult:
    model: Any
    envelopes: tuple[DagmcEnvelope, ...]
    source_audit: DTSourceAudit
    tally_inventory: TallyInventory
    metadata: Mapping[str, Any]


def build_combined_geometry(
    config_path: str | Path,
    *,
    vmec_path: str | Path,
    coils_path: str | Path,
    output_directory: str | Path,
    dagmc_filename: str = "combined_reactor_magnet.h5m",
    source_filename: str = "source_mesh.h5m",
    source_mesh_shape: tuple[int, int, int] = (11, 81, 61),
    casing_thickness_cm: float = 5.0,
    min_mesh_size_cm: float = 20.0,
    max_mesh_size_cm: float = 50.0,
) -> tuple[Stellarator, CombinedGeometryResult]:
    """Generate reactor structures and magnets in one CAD-to-DAGMC model."""
    load_and_validate_no_port_configuration(config_path)
    data = read_yaml_config(config_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stellarator = Stellarator(str(vmec_path))
    ivb = dict(data["invessel_build"])
    stellarator.construct_invessel_build(**ivb)
    magnet = dict(data["magnet_coils"])
    stellarator.construct_magnets_from_filaments(
        str(coils_path),
        float(magnet["width"]),
        float(magnet["thickness"]),
        float(magnet["toroidal_extent"]),
        case_thickness=float(
            magnet.get("case_thickness", casing_thickness_cm)
        ),
        sample_mod=int(magnet.get("sample_mod", 6)),
        mat_tag=("magnet_casing", "winding_pack"),
    )
    stellarator.build_cad_to_dagmc_model()
    dagmc_name = Path(dagmc_filename).stem
    stellarator.export_cad_to_dagmc(
        filename=dagmc_name,
        export_dir=output,
        min_mesh_size=min_mesh_size_cm,
        max_mesh_size=max_mesh_size_cm,
    )
    radial, poloidal, toroidal = source_mesh_shape
    extent = float(data["source_mesh"].get("toroidal_extent", 90.0))
    stellarator.construct_source_mesh(
        np.linspace(0.0, 1.0, radial),
        np.linspace(0.0, 360.0, poloidal),
        np.linspace(0.0, extent, toroidal),
    )
    source_name = Path(source_filename).stem
    stellarator.export_source_mesh(source_name, export_dir=output)
    dagmc_path = output / Path(dagmc_filename).with_suffix(".h5m")
    source_path = output / Path(source_filename).with_suffix(".h5m")
    result = CombinedGeometryResult(
        dagmc_path=dagmc_path,
        source_mesh_path=source_path,
        dagmc_sha256=_hash(dagmc_path),
        source_mesh_sha256=_hash(source_path),
        material_tags=tuple(stellarator._material_tags),
        source_mesh_shape=source_mesh_shape,
    )
    return stellarator, result


def representative_materials(cross_sections: str | Path):
    """Return documented homogenized reactor and winding-pack materials."""
    import openmc

    def material(name, density, nuclides):
        value = openmc.Material(name=name)
        value.set_density("g/cm3", density)
        for nuclide, fraction in nuclides.items():
            value.add_nuclide(nuclide, fraction, "ao")
        return value

    steel = {"Fe56": 0.68, "Cr52": 0.19, "Ni58": 0.11, "Mo98": 0.02}
    materials = [
        material("first_wall", 7.9, steel),
        material("back_wall", 7.9, steel),
        material("vac_vessel", 7.9, steel),
        material("magnet_casing", 7.9, steel),
        material(
            "shield",
            10.5,
            {
                "W184": 0.70,
                **{key: value * 0.30 for key, value in steel.items()},
            },
        ),
        material(
            "breeder",
            2.4,
            {"Li6": 0.12, "Li7": 0.28, "Si28": 0.12, "O16": 0.48},
        ),
        material(
            "winding_pack",
            7.6,
            {
                "Cu63": 0.30,
                "Cu65": 0.14,
                "Ni58": 0.20,
                "Cr52": 0.08,
                "Fe56": 0.05,
                "Mo98": 0.03,
                "Ag107": 0.01,
                "Y89": 0.01,
                "Ba138": 0.02,
                "O16": 0.06,
                "C0": 0.05,
                "H1": 0.05,
            },
        ),
    ]
    vacuum = openmc.Material(name="Vacuum")
    vacuum.set_density("g/cm3", 1.0e-12)
    vacuum.add_nuclide("H1", 1.0)
    materials.append(vacuum)
    result = openmc.Materials(materials)
    result.cross_sections = str(cross_sections)
    return result


def prepare_combined_model(
    stellarator: Stellarator,
    geometry: CombinedGeometryResult,
    *,
    cross_sections: str | Path,
    winding_pack_volume_id: int,
    neutron_edges_eV: Sequence[float],
    photon_edges_eV: Sequence[float],
    particles_per_batch: int,
    batches: int,
    plasma_direction_global: Sequence[float],
    toroidal_direction_global: Sequence[float],
    poloidal_direction_global: Sequence[float],
    threads: int = 1,
) -> CombinedModelResult:
    """Prepare one coupled OpenMC model around the combined DAGMC artifact."""
    import openmc

    envelope = extract_closed_envelope(
        geometry.dagmc_path,
        winding_pack_volume_id,
        envelope_id=f"winding-pack-{winding_pack_volume_id}",
        magnet_id=f"magnet-{winding_pack_volume_id}",
        plasma_direction_global=plasma_direction_global,
        toroidal_direction_global=toroidal_direction_global,
        poloidal_direction_global=poloidal_direction_global,
    )
    # Surface and volume IDs are part of the handoff contract.  Never allow
    # OpenMC to renumber them; reserve high IDs for the native CSG wrapper.
    dagmc = openmc.DAGMCUniverse(str(geometry.dagmc_path), auto_geom_ids=False)
    bounds = dagmc.bounding_box
    radius = (
        float(
            np.max(
                np.abs(np.concatenate((bounds.lower_left, bounds.upper_right)))
            )
        )
        * 1.2
    )
    world = openmc.Sphere(
        surface_id=10_000_001, r=radius, boundary_type="vacuum"
    )
    geometry_model = openmc.Geometry(
        [openmc.Cell(cell_id=10_000_001, fill=dagmc, region=-world)]
    )
    mesh_source, source_audit = build_temperature_dependent_mesh_source(
        stellarator.source_mesh, geometry.source_mesh_path
    )
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = int(particles_per_batch)
    settings.batches = int(batches)
    settings.source = [mesh_source]
    settings.surf_source_write = {
        "surface_ids": list(envelope.envelope.surface_ids),
        "max_particles": int(particles_per_batch) * int(batches),
    }
    configure_transport(
        settings,
        grazing_cutoff=1.0e-8,
        grazing_ratio=16.0,
        collision_track={
            "cell_ids": [int(winding_pack_volume_id)],
            "max_collisions": 10000,
            "max_collision_track_files": 8,
        },
    )
    model = openmc.Model(
        geometry=geometry_model,
        materials=representative_materials(cross_sections),
        settings=settings,
    )
    inventory = add_envelope_tallies(
        model,
        surface_ids=envelope.envelope.surface_ids,
        cell_ids=[winding_pack_volume_id],
        neutron_edges_eV=neutron_edges_eV,
        photon_edges_eV=photon_edges_eV,
    )
    metadata = {
        "geometry": asdict(geometry),
        "source": asdict(source_audit),
        "openmc_threads": int(threads),
        "grazing_cutoff": settings.surface_grazing_cutoff,
        "grazing_ratio": settings.surface_grazing_ratio,
        "photon_transport": settings.photon_transport,
        "tallies": inventory.to_dict(),
    }
    return CombinedModelResult(
        model, envelope, source_audit, inventory, metadata
    )


def prepare_combined_multimagnet_model(
    stellarator: Stellarator,
    geometry: CombinedGeometryResult,
    *,
    cross_sections: str | Path,
    winding_pack_volume_ids: Sequence[int],
    frames_by_volume: Mapping[int, Mapping[str, Sequence[float]]],
    neutron_edges_eV: Sequence[float],
    photon_edges_eV: Sequence[float],
    particles_per_batch: int,
    batches: int,
    threads: int = 1,
) -> CombinedMultiMagnetModelResult:
    """Prepare one coupled OpenMC run for all explicitly selected magnets."""
    import openmc

    volume_ids = tuple(int(item) for item in winding_pack_volume_ids)
    if not volume_ids or len(volume_ids) != len(set(volume_ids)):
        raise ValueError("winding-pack volume IDs must be nonempty and unique")
    envelopes = extract_closed_envelopes(
        geometry.dagmc_path,
        volume_ids,
        frames_by_volume=frames_by_volume,
    )
    first_frame = frames_by_volume[volume_ids[0]]
    prepared = prepare_combined_model(
        stellarator,
        geometry,
        cross_sections=cross_sections,
        winding_pack_volume_id=volume_ids[0],
        neutron_edges_eV=neutron_edges_eV,
        photon_edges_eV=photon_edges_eV,
        particles_per_batch=particles_per_batch,
        batches=batches,
        plasma_direction_global=first_frame["plasma_direction_global"],
        toroidal_direction_global=first_frame["toroidal_direction_global"],
        poloidal_direction_global=first_frame["poloidal_direction_global"],
        threads=threads,
    )
    surface_ids = tuple(
        sorted(
            {
                surface_id
                for envelope in envelopes
                for surface_id in envelope.envelope.surface_ids
            }
        )
    )
    prepared.model.settings.surf_source_write = {
        "surface_ids": list(surface_ids),
        "max_particles": int(particles_per_batch) * int(batches),
    }
    configure_transport(
        prepared.model.settings,
        grazing_cutoff=1.0e-8,
        grazing_ratio=16.0,
        collision_track={
            "cell_ids": list(volume_ids),
            "max_collisions": 10000,
            "max_collision_track_files": 8,
        },
    )
    prepared.model.tallies = openmc.Tallies()
    inventory = add_envelope_tallies(
        prepared.model,
        surface_ids=surface_ids,
        cell_ids=volume_ids,
        neutron_edges_eV=neutron_edges_eV,
        photon_edges_eV=photon_edges_eV,
    )
    metadata = {
        **dict(prepared.metadata),
        "winding_pack_volume_ids": list(volume_ids),
        "envelopes": [item.envelope.to_dict() for item in envelopes],
        "surface_ids": list(surface_ids),
        "tallies": inventory.to_dict(),
    }
    return CombinedMultiMagnetModelResult(
        prepared.model,
        envelopes,
        prepared.source_audit,
        inventory,
        metadata,
    )
