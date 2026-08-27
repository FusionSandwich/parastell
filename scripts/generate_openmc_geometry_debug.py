"""Create a bounded OpenMC geometry-debug model outside an immutable H5M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from parastell.reference_geometry import ReferenceGeometry
from parastell.reference_geometry import sha256_file


def _material(openmc, name, nuclide, density):
    material = openmc.Material(name=name)
    material.add_nuclide(nuclide, 1.0)
    material.set_density("g/cm3", density)
    return material


def _source_point(source_mesh: Path) -> list[float]:
    from pymoab import core, types

    mesh = core.Core()
    mesh.load_file(str(source_mesh))
    tetrahedra = mesh.get_entities_by_type(0, types.MBTET)
    if len(tetrahedra) == 0:
        raise ValueError("source mesh has no tetrahedra")
    connectivity = mesh.get_connectivity(tetrahedra[len(tetrahedra) // 2])
    vertices = mesh.get_coords(connectivity).reshape((-1, 3))
    return [float(item) for item in vertices.mean(axis=0)]


def generate(
    dagmc_path: Path,
    source_mesh: Path,
    output: Path,
    cross_sections: str,
    source_point_override: list[float] | None = None,
) -> dict:
    import openmc

    reference = ReferenceGeometry.open(dagmc_path)
    before = reference.accepted_sha256
    materials = openmc.Materials(
        [
            _material(openmc, "Vacuum", "H1", 1.0e-20),
            _material(openmc, "first_wall", "Fe56", 7.8),
            _material(openmc, "breeder", "Li6", 1.0),
            _material(openmc, "back_wall", "Fe56", 7.8),
            _material(openmc, "shield", "Fe56", 7.8),
            _material(openmc, "vac_vessel", "Fe56", 7.8),
            _material(openmc, "magnets", "Cu63", 8.96),
        ]
    )
    materials.cross_sections = cross_sections
    source_point = (
        [float(item) for item in source_point_override]
        if source_point_override is not None
        else _source_point(source_mesh)
    )
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
    settings.seed = 20260826
    settings.source = source
    geometry = reference.openmc_geometry(external_vacuum_radius_cm=2500.0)
    model = openmc.Model(
        geometry=geometry, materials=materials, settings=settings
    )
    output.mkdir(parents=True, exist_ok=True)
    model.export_to_model_xml(output / "model.xml")
    reference.verify_unchanged()
    receipt = {
        "schema": "parastell.openmc_geometry_debug_input/v1.0.0",
        "openmc_version": openmc.__version__,
        "geometry_mode": reference.geometry_mode,
        "dagmc_path": str(reference.path),
        "raw_h5m_sha256_before": before,
        "raw_h5m_sha256_after_model_export": sha256_file(dagmc_path),
        "h5m_mutated": False,
        "external_csg_vacuum_boundary": {
            "kind": "sphere",
            "radius_cm": 2500.0,
            "location": "OpenMC model.xml only; not written to H5M",
        },
        "source_point_cm": source_point,
        "batches": 2,
        "particles_per_batch": 2000,
        "histories": 4000,
        "seed": 20260826,
        "cross_sections": cross_sections,
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
    parser.add_argument("--source-point", nargs=3, type=float)
    arguments = parser.parse_args()
    generate(
        arguments.dagmc.resolve(),
        arguments.source_mesh.resolve(),
        arguments.output.resolve(),
        arguments.cross_sections,
        arguments.source_point,
    )


if __name__ == "__main__":
    main()
