import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageStat
import pytest

import parastell.magnet_coils as magnet_coils
import parastell.parastell as ps
from parastell.component_ledger import ComponentRecord, component_sort_key

from test_ports import (
    _SyntheticReferenceSurface,
    _explicit_box_port,
)
import parastell.invessel_build as ivb
import numpy as np


def _visual_stellarator(tmp_path):
    toroidal_angles = [0.0, 10.0, 20.0, 30.0]
    poloidal_angles = [0.0, 90.0, 180.0, 270.0, 360.0]
    thickness = np.ones((4, 5)) * 5.0
    radial_build = ivb.RadialBuild(
        toroidal_angles,
        poloidal_angles,
        1.08,
        {
            name: {"thickness_matrix": thickness}
            for name in (
                "first_wall",
                "breeder",
                "shield",
                "vacuum_vessel",
            )
        },
    )
    phi = np.deg2rad(15.0)
    outward = np.array([np.cos(phi), np.sin(phi), 0.0])
    port = _explicit_box_port(
        "visual_port",
        anchor=outward * 760.0,
        axis=outward,
        start={"reference": "plasma_surface"},
        end={
            "reference": "layer",
            "layer": "vacuum_vessel",
            "fraction": 1.0,
        },
        outer_extension=25.0,
        cross_section={
            "shape": "circle",
            "radius": 3.0,
            "dimensions_are": "clear_aperture",
        },
        liner={"enabled": True, "thickness": 1.0, "mat_tag": "SS316L"},
        expected_layers=[
            "first_wall",
            "breeder",
            "shield",
            "vacuum_vessel",
        ],
        collision={
            "magnet_policy": "report",
            "clearance_policy": "report",
            "minimum_magnet_clearance": 5.0,
        },
    )
    port["placement"] = {
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
    }
    model = ivb.InVesselBuild(
        _SyntheticReferenceSurface(),
        radial_build,
        num_ribs=13,
        num_rib_pts=49,
        ports=[port],
    )
    model.populate_surfaces()
    model.calculate_loci()
    model.generate_components_cadquery()

    coils_path = Path(__file__).parent / "files_for_tests" / "coils.example"
    magnets = magnet_coils.MagnetSetFromFilaments(
        coils_path,
        width=40.0,
        thickness=50.0,
        toroidal_extent=30.0,
        sample_mod=10,
    )
    magnets.populate_magnet_coils()
    magnets.build_magnet_coils()
    stellarator = ps.Stellarator.__new__(ps.Stellarator)
    stellarator.invessel_build = model
    stellarator.magnet_set = magnets
    stellarator.port_magnet_collision_report = ()
    stellarator.component_ledger = ()
    stellarator._logger = model._logger
    stellarator.check_port_magnet_clearance()
    return stellarator


def test_stable_component_ledger_has_named_port_records(tmp_path):
    stellarator = _visual_stellarator(tmp_path)
    stellarator.build_cad_to_dagmc_model()
    ledger = stellarator.component_ledger
    assert ledger == tuple(sorted(ledger, key=component_sort_key))
    assert all(isinstance(item, ComponentRecord) for item in ledger)
    assert len(ledger) == len(stellarator._material_tags)
    assert len(ledger) == len(stellarator._geometry.Solids())
    by_kind = {item.kind: item for item in ledger}
    assert by_kind["port_void"].material_tag == "Vacuum"
    assert by_kind["port_liner"].material_tag == "SS316L"


def test_visual_package_exports_named_color_assembly(tmp_path):
    stellarator = _visual_stellarator(tmp_path)
    manifest = stellarator.export_port_visual_validation(tmp_path)

    required = {
        "ported_sector_colored.step",
        "ported_sector_colored.glb",
        "ported_sector_cutaway.glb",
        "port_only.glb",
        "port_isometric.png",
        "port_axis_section.png",
        "port_blanket_cutaway.png",
        "port_magnet_clearance.png",
        "port_layers_exploded.png",
        "port_axis_transverse.png",
        "port_visual_manifest.json",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    assert set(manifest["intersected_layers"]) == {
        "first_wall",
        "breeder",
        "shield",
        "vacuum_vessel",
    }
    assert {"port_void", "port_liner", "magnet_conductor"} <= {
        item["kind"] for item in manifest["components"]
    }
    visual_only = [
        item for item in manifest["components"] if item["visual_only_geometry"]
    ]
    assert visual_only
    assert all(not item["neutronics_geometry"] for item in visual_only)
    assert all(not item["volumetric_mesh_geometry"] for item in visual_only)

    round_trip = json.loads(
        (tmp_path / "port_visual_manifest.json").read_text()
    )
    assert round_trip == manifest
    for filename in required:
        path = tmp_path / filename
        assert path.stat().st_size > 0
        if path.suffix == ".png":
            image = Image.open(path).convert("RGB")
            assert image.width >= 1200 and image.height >= 800
            extrema = ImageStat.Stat(image).extrema
            assert any(low != high for low, high in extrema)


def test_port_local_views_have_semantic_geometry_checks(tmp_path):
    stellarator = _visual_stellarator(tmp_path)
    manifest = stellarator.export_port_local_validation(tmp_path)
    required = {
        "port_local_longitudinal_u.png",
        "port_local_longitudinal_v.png",
        "port_local_transverse_inner.png",
        "port_local_transverse_blanket.png",
        "port_local_transverse_outer.png",
        "port_local_isometric_cutaway.png",
        "port_local_magnet_clearance.png",
        "port_surface_anchor.png",
    }
    assert required == set(manifest["generated_files"])
    for filename in required:
        image = Image.open(tmp_path / filename).convert("RGB")
        assert image.size == (1600, 1000)
        assert any(low != high for low, high in ImageStat.Stat(image).extrema)
    dimensions = manifest["recovered_dimensions"]
    assert dimensions["inner_width"] == pytest.approx(6.0)
    assert dimensions["inner_height"] == pytest.approx(6.0)
    assert dimensions["liner_thickness_u"] == pytest.approx(1.0)
    assert dimensions["liner_thickness_v"] == pytest.approx(1.0)
    assert manifest["local_frame"]["handedness"] == pytest.approx(1.0)
    assert manifest["local_frame"][
        "axis_normal_angle_degrees"
    ] == pytest.approx(0.0)
    assert manifest["views"]["longitudinal_u"]["centerline_slope"] == 0.0
    assert manifest["views"]["transverse_blanket"]["port_center_uv"] == [
        0.0,
        0.0,
    ]
    assert (
        manifest["views"]["transverse_blanket"]["aperture_width_fraction"]
        >= 0.2
    )
    assert not manifest["views"]["isometric"]["contains_global_sector"]
    assert manifest["views"]["magnet_clearance"][
        "collision_solid_identity_preserved"
    ]
