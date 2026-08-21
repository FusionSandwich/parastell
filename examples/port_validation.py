"""Generate the representative ParaStell engineering-port validation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import parastell.invessel_build as ivb
import parastell.magnet_coils as magnet_coils
import parastell.parastell as ps


class ValidationReferenceSurface(ivb.ReferenceSurface):
    """Small analytic toroidal surface used by the integration validation."""

    def angles_to_xyz(self, toroidal_angles, poloidal_angles, s, scale):
        phi = np.asarray(toroidal_angles, dtype=float)
        theta = np.asarray(poloidal_angles, dtype=float)
        scalar_phi = phi.ndim == 0
        major_radius = 6.0 * scale
        minor_radius = 1.6 * (1.0 + 0.2 * (s - 1.0)) * scale
        phi = np.atleast_1d(phi)[:, None]
        theta = theta[None, :]
        x = (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
        y = (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
        z = minor_radius * np.sin(theta)
        points = np.stack([x, y, z], axis=-1)
        return points[0] if scalar_phi else points


def create_validation_invessel_build(native=False):
    """Build the four-layer, lined, plasma-to-exterior validation port."""
    toroidal_angles = [0.0, 10.0, 20.0, 30.0]
    poloidal_angles = [0.0, 90.0, 180.0, 270.0, 360.0]
    thickness = np.ones((4, 5)) * 5.0
    radial_build = ivb.RadialBuild(
        toroidal_angles,
        poloidal_angles,
        1.08,
        {
            name: {
                "thickness_matrix": thickness,
                "mat_tag": name,
            }
            for name in (
                "first_wall",
                "breeder",
                "shield",
                "vacuum_vessel",
            )
        },
    )
    port = {
        "name": "real_heating_port",
        "placement": {
            "mode": "surface",
            "anchor": {
                "reference": "plasma_surface",
                "toroidal_angle": 15.0,
                "poloidal_angle": 0.0,
            },
            "axis": {
                "mode": "outward_normal",
                "poloidal_tilt": 0.0,
                "toroidal_tilt": 0.0,
            },
            "roll": 0.0,
            "max_search_length": 500.0,
        },
        "cross_section": {
            "shape": "circle",
            "radius": 3.0,
            "dimensions_are": "clear_aperture",
        },
        "extent": {
            "start": {"reference": "plasma_surface"},
            "end": {
                "reference": "layer",
                "layer": "vacuum_vessel",
                "fraction": 1.0,
            },
            "outer_extension": 25.0,
        },
        "liner": {
            "enabled": True,
            "thickness": 1.0,
            "mat_tag": "SS316L",
        },
        "fill": {"mat_tag": "Vacuum"},
        "repetition": {"mode": "single"},
        "collision": {
            "magnet_policy": "report",
            "clearance_policy": "report",
            "minimum_magnet_clearance": 5.0,
        },
        "expected_layers": [
            "first_wall",
            "breeder",
            "shield",
            "vacuum_vessel",
        ],
    }
    model = ivb.InVesselBuild(
        ValidationReferenceSurface(),
        radial_build,
        num_ribs=13,
        num_rib_pts=49,
        ports=[port],
    )
    model.populate_surfaces()
    model.calculate_loci()
    if native:
        model.use_pydagmc = True
        model.generate_components()
    else:
        model.generate_components_cadquery()
    return model


def create_validation_stellarator():
    """Build the visual validation assembly, including filament magnets."""
    model = create_validation_invessel_build()

    coils_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "files_for_tests"
        / "coils.example"
    )
    magnets = magnet_coils.MagnetSetFromFilaments(
        coils_path,
        width=40.0,
        thickness=50.0,
        toroidal_extent=30.0,
        sample_mod=10,
        mat_tag="magnets",
    )
    magnets.populate_magnet_coils()
    magnets.build_magnet_coils()

    stellarator = ps.Stellarator.__new__(ps.Stellarator)
    stellarator.invessel_build = model
    stellarator.magnet_set = magnets
    stellarator.port_magnet_collision_report = ()
    stellarator.component_ledger = ()
    stellarator.use_pydagmc = False
    stellarator._logger = model._logger
    stellarator.check_port_magnet_clearance()
    return stellarator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stellarator = create_validation_stellarator()
    stellarator.export_port_local_validation(args.output_dir)


if __name__ == "__main__":
    main()
