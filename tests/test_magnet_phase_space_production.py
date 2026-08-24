import sys
from types import SimpleNamespace

import numpy as np
import pytest

from parastell.dagmc_envelope import discover_magnet_volumes
from parastell.dagmc_envelope import select_winding_pack_volumes
from parastell.dagmc_envelope import validate_dagmc_watertightness
from parastell.combined_openmc16_model import _magnet_material_tag
from parastell.magnet_boundary_envelope import CorrelatedBoundaryBank
from parastell.magnet_boundary_envelope import EnvelopeSurface
from parastell.magnet_boundary_envelope import MagnetBoundaryEnvelope
from parastell.magnet_boundary_envelope import assign_adaptive_surface_patches
from parastell.magnet_handoff_cli import build_parser
from parastell.openmc16_export import _select_envelope_records
from parastell.production_handoff import validate_no_port_configuration


def test_requires_explicit_no_port_configuration():
    with pytest.raises(ValueError, match="geometry_features.ports: false"):
        validate_no_port_configuration({"geometry_features": {}})

    audit = validate_no_port_configuration(
        {"geometry_features": {"ports": False}, "source": {"kind": "dt"}}
    )

    assert audit.port_free is True


def test_rejects_hidden_port_configuration():
    with pytest.raises(ValueError, match="port-related configuration"):
        validate_no_port_configuration(
            {
                "geometry_features": {"ports": False},
                "geometry": {"port_geometry": {"enabled": False}},
            }
        )


def test_production_inventory_subcommands_are_public():
    parser = build_parser()

    config = parser.parse_args(
        ["validate-production-config", "--config", "production.yaml"]
    )
    listing = parser.parse_args(["list-magnets", "--dagmc", "model.h5m"])
    inspect = parser.parse_args(
        ["inspect-magnet", "--dagmc", "model.h5m", "--volume-id", "17"]
    )

    assert config.command == "validate-production-config"
    assert listing.winding_pack_material == ["winding_pack"]
    assert inspect.volume_id == 17

    combined = parser.parse_args(
        [
            "run-combined",
            "--config",
            "production.yaml",
            "--vmec",
            "wout.nc",
            "--coils",
            "coils.example",
            "--output-dir",
            "run",
            "--cross-sections",
            "cross_sections.xml",
            "--volume-id",
            "17",
            "--frames",
            "frames.json",
            "--parastell-commit",
            "abc123",
        ]
    )
    assert combined.source_mesh_shape == [11, 81, 61]
    assert combined.casing_thickness_cm == 0.0
    assert combined.minimum_mesh_size_cm == 5.0
    assert combined.maximum_mesh_size_cm == 20.0
    assert combined.neutron_groups == "smoke-7"
    assert combined.photon_groups == "smoke-42"


@pytest.mark.parametrize(
    ("case_thickness", "expected_tag"),
    [(0.0, "winding_pack"), (5.0, ("magnet_casing", "winding_pack"))],
)
def test_combined_geometry_material_tags_follow_casing(
    case_thickness, expected_tag
):
    assert _magnet_material_tag(case_thickness) == expected_tag


def test_empty_selected_envelope_records_fail_explicitly():
    records = np.zeros(1, dtype=[("surf_id", np.int64)])
    records["surf_id"] = 41

    with pytest.raises(ValueError, match="no surface-source records crossed"):
        _select_envelope_records(records, (73, 74))


def test_inventory_uses_material_tags_and_never_guesses(tmp_path, monkeypatch):
    dagmc = tmp_path / "combined.h5m"
    dagmc.write_bytes(b"dagmc")

    def volume(volume_id, material, surfaces):
        return SimpleNamespace(
            id=volume_id,
            material=material,
            surfaces=[SimpleNamespace(id=item) for item in surfaces],
        )

    volumes = {
        4: volume(4, "blanket", [1]),
        17: volume(17, "winding_pack", [31, 32]),
        18: volume(18, "magnet_casing", [33, 34]),
        27: volume(27, "winding-pack", [41, 42]),
    }
    module = SimpleNamespace(
        Model=lambda path: SimpleNamespace(volumes_by_id=volumes)
    )
    monkeypatch.setitem(sys.modules, "pydagmc", module)

    inventory = discover_magnet_volumes(dagmc)

    assert [item.volume_id for item in inventory.winding_packs] == [17, 27]
    assert [item.volume_id for item in inventory.casings] == [18]
    with pytest.raises(ValueError, match="explicit volume IDs"):
        select_winding_pack_volumes(inventory)
    assert select_winding_pack_volumes(inventory, [27])[0].surface_ids == (
        41,
        42,
    )


def test_global_watertightness_rejects_open_volume(tmp_path, monkeypatch):
    dagmc = tmp_path / "open.h5m"
    dagmc.write_bytes(b"dagmc")
    triangle = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    volume = SimpleNamespace(
        surfaces=[SimpleNamespace(id=1, triangle_coords=triangle)]
    )
    module = SimpleNamespace(
        Model=lambda path: SimpleNamespace(volumes_by_id={1: volume})
    )
    monkeypatch.setitem(sys.modules, "pydagmc", module)

    report = validate_dagmc_watertightness(dagmc)

    assert report.passes is False
    assert report.leaky_volume_ids == (1,)
    assert report.unmatched_edge_count == 3


def test_population_statistics_report_weighted_count_and_ess():
    count = 2
    columns = {
        "record_id": np.arange(count),
        "history_id": np.arange(count),
        "position_global_cm": np.zeros((count, 3)),
        "position_local_cm": np.zeros((count, 3)),
        "direction_global": np.tile([1.0, 0.0, 0.0], (count, 1)),
        "direction_local": np.tile([0.0, 0.0, -1.0], (count, 1)),
        "outward_normal_global": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "energy_eV": np.asarray([1.0e6, 2.0e6]),
        "weight": np.asarray([1.0, 3.0]),
        "weight_std_dev": np.zeros(count),
        "particle": np.asarray(["neutron", "neutron"]),
        "particle_pdg": np.asarray([2112, 2112]),
        "surface_id": np.asarray([31, 31]),
        "envelope_id": np.asarray(["magnet-17", "magnet-17"]),
        "crossing_sense": np.asarray(["incoming", "incoming"]),
        "surface_role": np.asarray(["plasma_facing", "plasma_facing"]),
        "time_s": np.zeros(count),
        "mu": np.asarray([-1.0, -1.0]),
        "azimuth_rad": np.zeros(count),
        "grazing": np.zeros(count, dtype=bool),
        "patch_id": np.zeros(count, dtype=int),
        "energy_group": np.zeros(count, dtype=int),
        "angle_bin_id": np.zeros(count, dtype=int),
    }
    statistics = CorrelatedBoundaryBank(columns).population_statistics()

    assert statistics["overall"]["record_count"] == 2
    assert statistics["overall"]["weighted_count"] == pytest.approx(4.0)
    assert statistics["overall"]["effective_sample_size"] == pytest.approx(1.6)
    assert statistics["overall"]["status"] == "INSUFFICIENT_STATISTICS"


def test_adaptive_patches_conserve_area_records_and_current():
    surface = EnvelopeSurface(
        surface_id=31,
        role="plasma_facing",
        area_cm2=4.0,
        centroid_global_cm=(0.0, 0.0, 0.0),
        outward_normal_global=(0.0, 0.0, 1.0),
        toroidal_direction_global=(1.0, 0.0, 0.0),
        poloidal_direction_global=(0.0, 1.0, 0.0),
        u_edges_cm=(-1.0, 1.0),
        v_edges_cm=(-1.0, 1.0),
        vector_area_global_cm2=(0.0, 0.0, 0.0),
    )
    envelope = MagnetBoundaryEnvelope(
        "magnet-17",
        "winding-pack-17",
        17,
        (surface,),
        "0" * 64,
        metadata={"edge_multiplicity_proof": {}},
    )
    count = 8
    columns = {
        "record_id": np.arange(count),
        "history_id": np.arange(count),
        "position_global_cm": np.zeros((count, 3)),
        "position_local_cm": np.column_stack(
            (
                np.asarray([-0.8, -0.7, -0.6, -0.5, 0.5, 0.6, 0.7, 0.8]),
                np.zeros(count),
                np.zeros(count),
            )
        ),
        "direction_global": np.tile([0.0, 0.0, -1.0], (count, 1)),
        "direction_local": np.tile([0.0, 0.0, -1.0], (count, 1)),
        "outward_normal_global": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "energy_eV": np.full(count, 14.1e6),
        "weight": np.ones(count),
        "weight_std_dev": np.zeros(count),
        "particle": np.full(count, "neutron"),
        "particle_pdg": np.full(count, 2112),
        "surface_id": np.full(count, 31),
        "envelope_id": np.full(count, "magnet-17"),
        "crossing_sense": np.full(count, "incoming"),
        "surface_role": np.full(count, "plasma_facing"),
        "time_s": np.zeros(count),
        "mu": np.full(count, -1.0),
        "azimuth_rad": np.zeros(count),
        "grazing": np.zeros(count, dtype=bool),
        "patch_id": np.zeros(count, dtype=int),
        "energy_group": np.zeros(count, dtype=int),
        "angle_bin_id": np.zeros(count, dtype=int),
    }
    original = CorrelatedBoundaryBank(columns)

    adapted = assign_adaptive_surface_patches(
        envelope,
        original,
        target_effective_sample_size=2.0,
        minimum_records=2,
        maximum_depth=2,
    )

    patches = adapted.metadata["adaptive_surface_patches"]["patches"]
    assert len(patches) == 2
    assert sum(item["area_cm2"] for item in patches) == pytest.approx(4.0)
    assert sum(item["record_count"] for item in patches) == count
    assert adapted.integrated_current == original.integrated_current


def test_production_modules_do_not_import_port_geometry():
    module_names = (
        "parastell.production_handoff",
        "parastell.combined_openmc16_model",
        "parastell.dagmc_envelope",
        "parastell.openmc16_export",
    )
    for name in module_names:
        module = sys.modules.get(name)
        if module is None:
            __import__(name)
            module = sys.modules[name]
        source = open(module.__file__, encoding="utf-8").read()
        assert "from .ports" not in source
        assert "import parastell.ports" not in source


def test_raw_directional_statepoint_respects_dagmc_surface_sense(tmp_path):
    import h5py
    from types import SimpleNamespace

    from parastell.openmc16_export import _directional_current_from_statepoint

    path = tmp_path / "statepoint.h5"
    with h5py.File(path, "w") as statepoint:
        tallies = statepoint.create_group("tallies")
        filters = tallies.create_group("filters")
        definitions = (
            (1, "surface", [11, 12]),
            (2, "musurface", [-1.0, 0.0, 1.0]),
            (3, "particle", [b"neutron"]),
            (4, "energy", [0.0, 20.0e6]),
        )
        for filter_id, filter_type, bins in definitions:
            group = filters.create_group(f"filter {filter_id}")
            group["type"] = np.bytes_(filter_type)
            group["bins"] = bins
            group["n_bins"] = len(bins) - (
                filter_type in {"musurface", "energy"}
            )
        tally = tallies.create_group("tally 1")
        tally["name"] = np.bytes_("pstl_envelope_neutron_directional_current")
        tally["filters"] = [1, 2, 3, 4]
        tally["n_realizations"] = 10
        means = np.asarray([-0.2, 0.3, -0.4, 0.5])
        results = np.zeros((4, 1, 2))
        results[:, 0, 0] = 10.0 * means
        results[:, 0, 1] = 10.0 * means**2
        tally["results"] = results

    signs = {11: 1, 12: -1}
    schema = SimpleNamespace(
        surface=lambda surface_id: SimpleNamespace(
            openmc_normal_sign=signs[surface_id]
        )
    )
    envelope = SimpleNamespace(envelope=schema)
    current = _directional_current_from_statepoint(path, "neutron", envelope)

    assert current[11]["incoming"][0] == pytest.approx(0.2)
    assert current[11]["outgoing"][0] == pytest.approx(0.3)
    assert current[12]["incoming"][0] == pytest.approx(0.5)
    assert current[12]["outgoing"][0] == pytest.approx(0.4)
