from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import pytest
import openmc

from parastell.magnet_spectral_handoff import (
    _compatible_mesh_filter_dataframe,
)
from parastell.magnet_spectral_handoff import CoordinateFrame
from parastell.magnet_spectral_handoff import MagnetSpectralHandoff


@pytest.fixture(autouse=True)
def reset_openmc_ids():
    reset = getattr(openmc, "reset_auto_ids", None)
    if reset is not None:
        reset()


def handoff_mapping(**tally_overrides):
    tallies = {
        "cell_flux": True,
        "mesh_flux": True,
        "boundary_current": True,
        "heating": True,
        "damage_energy": True,
        "gas_production": False,
    }
    tallies.update(tally_overrides)
    return {
        "energy_bounds_eV": [0.0, 1.0e3, 1.0e6, 2.0e7],
        "particles": ["neutron", "photon"],
        "mu_bounds": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "normalization": {"source_rate_per_s": 100.0},
        "tallies": tallies,
        "regions": [
            {
                "name": "coil A winding pack",
                "source_region_id": "interface-a",
                "magnet_id": "magnet-a",
                "cell_ids": [10],
                "surface_ids": [20, 21],
                "volume_cm3": 10.0,
                "surface_areas_cm2": {20: 2.0, 21: 4.0},
                "surface_normal_signs": {20: 1, 21: -1},
                "surface_outward_normals_global": {
                    20: [1.0, 0.0, 0.0],
                    21: [0.0, -1.0, 0.0],
                },
                "damage_nuclides": ["Cu63", "Cu65"],
                "coordinate_frame": {
                    "origin_cm": [10.0, 0.0, 0.0],
                    "x_axis": [0.0, 1.0, 0.0],
                    "y_axis": [-1.0, 0.0, 0.0],
                    "z_axis": [0.0, 0.0, 1.0],
                    "labels": ["tape_width", "tape_length", "tape_normal"],
                },
                "mesh": {
                    "kind": "regular",
                    "lower_left_cm": [0.0, 0.0, 0.0],
                    "upper_right_cm": [2.0, 2.0, 2.0],
                    "dimension": [2, 2, 2],
                    "mesh_id": 700,
                },
            }
        ],
    }


def test_accepts_named_openmc_energy_group_structure():
    data = handoff_mapping()
    data.pop("energy_bounds_eV")
    data["energy_group_structure"] = "CASMO-2"

    handoff = MagnetSpectralHandoff.from_mapping(data)

    assert handoff.energy_group_structure == "CASMO-2"
    assert handoff.energy_bounds_eV == (0.0, 0.625, 2.0e7)


def test_builds_directional_and_spatial_tallies():
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())
    tallies = handoff.build_tallies()
    by_name = {tally.name: tally for tally in tallies}

    assert set(by_name) == {
        "pstl_magnet_coil_a_winding_pack_cell_flux",
        "pstl_magnet_coil_a_winding_pack_mesh_flux",
        "pstl_magnet_coil_a_winding_pack_boundary_current",
        "pstl_magnet_coil_a_winding_pack_heating",
        "pstl_magnet_coil_a_winding_pack_damage_energy",
    }

    boundary = by_name["pstl_magnet_coil_a_winding_pack_boundary_current"]
    assert boundary.scores == ["current"]
    assert boundary.estimator == "analog"
    assert any(
        isinstance(item, openmc.SurfaceFilter) for item in boundary.filters
    )
    assert any(
        isinstance(item, openmc.MuSurfaceFilter) for item in boundary.filters
    )

    mesh_flux = by_name["pstl_magnet_coil_a_winding_pack_mesh_flux"]
    mesh_filter = next(
        item
        for item in mesh_flux.filters
        if isinstance(item, openmc.MeshFilter)
    )
    assert tuple(mesh_filter.mesh.dimension) == (2, 2, 2)

    damage = by_name["pstl_magnet_coil_a_winding_pack_damage_energy"]
    assert damage.nuclides == ["Cu63", "Cu65"]


def test_attaches_tallies_and_enables_photon_transport():
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())
    model = openmc.Model()

    handoff.attach_to_model(model)

    assert len(model.tallies) == 5
    assert model.settings.photon_transport is True
    with pytest.raises(ValueError, match="already contains"):
        handoff.attach_to_model(model)


@pytest.mark.parametrize(
    "direction, selector",
    [
        ("incoming", "cellto"),
        ("outgoing", "cellfrom"),
        ("both", "cell"),
    ],
)
def test_configures_directional_surface_source(direction, selector):
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())
    settings = openmc.Settings()

    config = handoff.configure_surface_source(
        settings,
        "coil A winding pack",
        direction=direction,
        max_particles=500,
        max_source_files=2,
    )

    assert config[selector] == 10
    assert config["surface_ids"] == [20, 21]
    assert config["max_particles"] == 500
    assert config["max_source_files"] == 2
    assert settings.surf_source_write == config


def test_multiple_cells_require_phase_space_cell_id():
    data = handoff_mapping()
    data["regions"][0]["cell_ids"] = [10, 11]
    data["regions"][0].pop("phase_space_cell_id", None)
    handoff = MagnetSpectralHandoff.from_mapping(data)

    with pytest.raises(ValueError, match="phase_space_cell_id"):
        handoff.configure_surface_source(
            openmc.Settings(), "coil A winding pack"
        )


def test_coordinate_frame_is_right_handed():
    with pytest.raises(ValueError, match="right-handed"):
        CoordinateFrame(
            x_axis=(1.0, 0.0, 0.0),
            y_axis=(0.0, 1.0, 0.0),
            z_axis=(0.0, 0.0, -1.0),
        )


def _write_source_file(path: Path, particle_codes=(2112, 22)):
    vector_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    source_dtype = np.dtype(
        [
            ("r", vector_dtype),
            ("u", vector_dtype),
            ("E", "<f8"),
            ("time", "<f8"),
            ("wgt", "<f8"),
            ("delayed_group", "<i4"),
            ("surf_id", "<i4"),
            ("particle", "<i4"),
        ]
    )
    bank = np.zeros(2, dtype=source_dtype)
    bank["r"]["x"] = [10.0, 11.0]
    bank["r"]["y"] = [2.0, 3.0]
    bank["r"]["z"] = [3.0, 4.0]
    bank["u"]["x"] = [1.0, 0.0]
    bank["u"]["y"] = [0.0, 1.0]
    bank["u"]["z"] = [0.0, 0.0]
    bank["E"] = [14.1e6, 1.0e6]
    bank["time"] = [0.0, 1.0e-8]
    bank["wgt"] = [1.0, 0.5]
    bank["surf_id"] = [20, 21]
    bank["particle"] = particle_codes

    with h5py.File(path, "w") as output:
        output.attrs["filetype"] = np.bytes_("source")
        output.attrs["version"] = np.asarray([1, 0], dtype=int)
        output.create_dataset("source_bank", data=bank)


def test_exports_surface_source_in_local_coordinates(tmp_path):
    source_path = tmp_path / "surface_source.h5"
    output_path = tmp_path / "magnet_phase_space.h5"
    _write_source_file(source_path)
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())

    handoff.export_surface_source(
        source_path,
        output_path,
        region_name="coil A winding pack",
        selection="both",
    )

    with h5py.File(output_path, "r") as output:
        phase_space = output["phase_space"]
        assert output.attrs["selection"] == "both"
        assert np.allclose(
            phase_space["position_local_cm"][0], [2.0, 0.0, 3.0]
        )
        assert np.allclose(phase_space["direction_local"][0], [0.0, -1.0, 0.0])
        assert phase_space["particle_pdg"][:].tolist() == [2112, 22]
        assert phase_space["particle_code_raw"][:].tolist() == [2112, 22]
        assert phase_space["particle_name"].asstr()[:].tolist() == [
            "neutron",
            "photon",
        ]
        assert np.allclose(phase_space["mu_outward"][:], [1.0, -1.0])
        assert phase_space["magnet_direction"].asstr()[:].tolist() == [
            "outgoing",
            "incoming",
        ]
        assert phase_space["source_region_id"].asstr()[:].tolist() == [
            "interface-a",
            "interface-a",
        ]
        assert phase_space["record_id"][:].tolist() == [0, 1]
        validation = json.loads(output.attrs["direction_validation_json"])
        assert validation["counts"]["incoming"] == 1
        assert validation["counts"]["outgoing"] == 1


def test_normalizes_openmc_015_particle_codes(tmp_path):
    source_path = tmp_path / "surface_source_legacy.h5"
    output_path = tmp_path / "magnet_phase_space_legacy.h5"
    _write_source_file(source_path, particle_codes=(0, 1))
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())

    handoff.export_surface_source(
        source_path,
        output_path,
        region_name="coil A winding pack",
        selection="incoming",
    )

    with h5py.File(output_path, "r") as output:
        phase_space = output["phase_space"]
        assert phase_space["particle_code_raw"][:].tolist() == [0, 1]
        assert phase_space["particle_pdg"][:].tolist() == [2112, 22]
        assert phase_space["particle_name"].asstr()[:].tolist() == [
            "neutron",
            "photon",
        ]
        metadata = json.loads(output["metadata_json"].asstr()[()])
        assert (
            metadata["source_files"][0]["particle_code_encoding"]
            == "openmc_0.15_enum"
        )


def test_rejects_empty_cell_ids():
    data = handoff_mapping()
    data["regions"][0]["cell_ids"] = []

    with pytest.raises(ValueError, match="at least one OpenMC cell"):
        MagnetSpectralHandoff.from_mapping(data)


def test_rejects_region_names_with_colliding_slugs():
    data = handoff_mapping()
    second = dict(data["regions"][0])
    second["name"] = "coil-A winding pack"
    second["cell_ids"] = [11]
    second["surface_ids"] = [22]
    second["surface_areas_cm2"] = {22: 2.0}
    second["surface_normal_signs"] = {22: 1}
    second["surface_outward_normals_global"] = {22: [1.0, 0.0, 0.0]}
    second["mesh"] = dict(second["mesh"], mesh_id=701)
    data["regions"].append(second)

    with pytest.raises(ValueError, match="slugification"):
        MagnetSpectralHandoff.from_mapping(data)


def test_phase_space_requires_boundary_current():
    handoff = MagnetSpectralHandoff.from_mapping(
        handoff_mapping(boundary_current=False)
    )

    with pytest.raises(ValueError, match="boundary_current=true"):
        handoff.configure_surface_source(
            openmc.Settings(), "coil A winding pack"
        )


def test_mcpl_is_rejected_for_deterministic_contract():
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())

    with pytest.raises(ValueError, match="requires OpenMC HDF5"):
        handoff.configure_surface_source(
            openmc.Settings(), "coil A winding pack", mcpl=True
        )


def test_surface_source_rejects_unexpected_surface(tmp_path):
    source_path = tmp_path / "unexpected_surface.h5"
    output_path = tmp_path / "magnet_phase_space.h5"
    _write_source_file(source_path)
    with h5py.File(source_path, "r+") as source:
        bank = source["source_bank"][()]
        bank["surf_id"][1] = 99
        source["source_bank"][...] = bank
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())

    with pytest.raises(ValueError, match="outside the configured"):
        handoff.export_surface_source(
            source_path,
            output_path,
            region_name="coil A winding pack",
            selection="both",
        )


def test_manifest_records_handoff_contract():
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())
    manifest = handoff.to_manifest()

    assert manifest["minimum_openmc_version"] == "0.15.1"
    assert manifest["tally_switches"]["boundary_current"] is True
    assert "history_id" in manifest["native_surface_source_fields_unavailable"]
    assert (
        "mu_outward"
        in manifest["phase_space_output_fields_added_by_parastell"]
    )


class FakeTally:
    def __init__(self, name, tally_id, dataframe, filters=None):
        self.name = name
        self.id = tally_id
        self.filters = list(filters or ())
        self._dataframe = dataframe

    def get_pandas_dataframe(self, paths=False):
        assert paths is False
        return self._dataframe.copy()


class FakeStatePoint:
    tallies = {}

    def __init__(self, path):
        self.path = path
        self.closed = False

    def get_tally(self, name=None, id=None):
        key = id if id is not None else name
        try:
            return self.tallies[key]
        except KeyError as exc:
            raise LookupError(key) from exc

    def close(self):
        self.closed = True


def test_exports_normalized_cell_flux(monkeypatch, tmp_path):
    data = handoff_mapping(
        mesh_flux=False,
        boundary_current=False,
        heating=False,
        damage_energy=False,
    )
    data["regions"][0].pop("mesh")
    handoff = MagnetSpectralHandoff.from_mapping(data)
    tally_name = "pstl_magnet_coil_a_winding_pack_cell_flux"
    dataframe = pd.DataFrame(
        {
            "cell": [10, 10],
            "particle": ["neutron", "neutron"],
            "energy low [eV]": [0.0, 1.0],
            "energy high [eV]": [1.0, 11.0],
            "nuclide": ["total", "total"],
            "score": ["flux", "flux"],
            "mean": [2.0, 4.0],
            "std. dev.": [0.2, 0.4],
        }
    )
    tally_id = handoff.tally_catalog()[0]["id"]
    FakeStatePoint.tallies = {
        tally_id: FakeTally(tally_name, tally_id, dataframe)
    }
    monkeypatch.setattr(openmc, "StatePoint", FakeStatePoint)

    output_path = tmp_path / "magnet_spectra.h5"
    handoff.export_statepoint(tmp_path / "statepoint.h5", output_path)

    with h5py.File(output_path, "r") as output:
        group = output["tallies"][tally_name]
        assert np.allclose(group["flux_cm_2_s_1"][:], [20.0, 40.0])
        assert np.allclose(group["energy_width_eV"][:], [1.0, 10.0])
        assert np.allclose(group["relative_error"][:], [0.1, 0.1])


def test_exports_volume_normalized_mesh_flux(monkeypatch, tmp_path):
    data = handoff_mapping(
        cell_flux=False,
        boundary_current=False,
        heating=False,
        damage_energy=False,
    )
    data["regions"][0]["mesh"]["upper_right_cm"] = [4.0, 1.0, 1.0]
    data["regions"][0]["mesh"]["dimension"] = [2, 1, 1]
    handoff = MagnetSpectralHandoff.from_mapping(data)
    tally_name = "pstl_magnet_coil_a_winding_pack_mesh_flux"

    mesh = openmc.RegularMesh(mesh_id=700)
    mesh.lower_left = (0.0, 0.0, 0.0)
    mesh.upper_right = (4.0, 1.0, 1.0)
    mesh.dimension = (2, 1, 1)
    dataframe = pd.DataFrame(
        {
            "mesh 700 x": [1, 2],
            "mesh 700 y": [1, 1],
            "mesh 700 z": [1, 1],
            "particle": ["neutron", "neutron"],
            "energy low [eV]": [0.0, 0.0],
            "energy high [eV]": [1.0e3, 1.0e3],
            "nuclide": ["total", "total"],
            "score": ["flux", "flux"],
            "mean": [4.0, 6.0],
            "std. dev.": [0.4, 0.6],
        }
    )
    tally_id = handoff.tally_catalog()[0]["id"]
    FakeStatePoint.tallies = {
        tally_id: FakeTally(
            tally_name,
            tally_id,
            dataframe,
            filters=[openmc.MeshFilter(mesh)],
        )
    }
    monkeypatch.setattr(openmc, "StatePoint", FakeStatePoint)

    output_path = tmp_path / "magnet_spectra.h5"
    handoff.export_statepoint(tmp_path / "statepoint.h5", output_path)

    with h5py.File(output_path, "r") as output:
        group = output["tallies"][tally_name]
        assert np.allclose(group["mesh_volume_cm3"][:], [2.0, 2.0])
        assert np.allclose(group["flux_per_source_cm_2"][:], [2.0, 3.0])
        assert np.allclose(group["flux_cm_2_s_1"][:], [200.0, 300.0])


def test_maps_mu_bins_to_magnet_direction(monkeypatch, tmp_path):
    data = handoff_mapping(
        cell_flux=False,
        mesh_flux=False,
        heating=False,
        damage_energy=False,
    )
    data["regions"][0].pop("mesh")
    handoff = MagnetSpectralHandoff.from_mapping(data)
    tally_name = "pstl_magnet_coil_a_winding_pack_boundary_current"
    dataframe = pd.DataFrame(
        {
            "surface": [20, 20, 21, 21],
            "particle": ["neutron"] * 4,
            "energy low [eV]": [0.0] * 4,
            "energy high [eV]": [1.0e3] * 4,
            "musurface low": [-1.0, 0.0, -1.0, 0.0],
            "musurface high": [0.0, 1.0, 0.0, 1.0],
            "nuclide": ["total"] * 4,
            "score": ["current"] * 4,
            "mean": [1.0] * 4,
            "std. dev.": [0.1] * 4,
        }
    )
    tally_id = handoff.tally_catalog()[0]["id"]
    FakeStatePoint.tallies = {
        tally_id: FakeTally(tally_name, tally_id, dataframe)
    }
    monkeypatch.setattr(openmc, "StatePoint", FakeStatePoint)

    output_path = tmp_path / "magnet_spectra.h5"
    handoff.export_statepoint(tmp_path / "statepoint.h5", output_path)

    with h5py.File(output_path, "r") as output:
        labels = output["tallies"][tally_name]["magnet_direction"].asstr()[:]
        assert labels.tolist() == [
            "incoming",
            "outgoing",
            "outgoing",
            "incoming",
        ]


def test_yaml_resolves_unstructured_mesh_relative_to_config(tmp_path):
    mesh_path = tmp_path / "mesh" / "coil_mesh.h5m"
    config_path = tmp_path / "handoff.yaml"
    config_path.write_text(
        """
energy_bounds_eV: [0.0, 20000000.0]
regions:
  - name: coil
    cell_ids: [10]
    mesh:
      kind: unstructured
      filename: mesh/coil_mesh.h5m
""".lstrip(),
        encoding="utf-8",
    )

    handoff = MagnetSpectralHandoff.from_yaml(config_path)

    assert handoff.regions[0].mesh.filename == str(mesh_path.resolve())


def test_openmc_015_unstructured_mesh_dataframe_compatibility():
    mesh = openmc.UnstructuredMesh("coil_mesh.h5m", "moab", mesh_id=811)
    mesh.dimension = (3,)
    mesh_filter = openmc.MeshFilter(mesh)

    dataframe = _compatible_mesh_filter_dataframe(
        mesh_filter, data_size=6, stride=2
    )

    assert dataframe[("mesh 811", "element")].tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


def test_direction_is_inferred_from_cell_selection_without_normals(
    tmp_path,
):
    data = handoff_mapping()
    data["regions"][0].pop("surface_outward_normals_global")
    source_path = tmp_path / "surface_source.h5"
    output_path = tmp_path / "magnet_phase_space.h5"
    _write_source_file(source_path)
    handoff = MagnetSpectralHandoff.from_mapping(data)

    handoff.export_surface_source(
        source_path,
        output_path,
        region_name="coil A winding pack",
        selection="incoming",
    )

    with h5py.File(output_path, "r") as output:
        phase_space = output["phase_space"]
        assert phase_space["magnet_direction"].asstr()[:].tolist() == [
            "incoming",
            "incoming",
        ]
        assert phase_space["direction_label_basis"].asstr()[:].tolist() == [
            "openmc_cell_selection",
            "openmc_cell_selection",
        ]


def test_exports_per_record_surface_normals(tmp_path):
    data = handoff_mapping()
    data["regions"][0].pop("surface_outward_normals_global")
    source_path = tmp_path / "surface_source.h5"
    output_path = tmp_path / "magnet_phase_space.h5"
    _write_source_file(source_path)
    handoff = MagnetSpectralHandoff.from_mapping(data)
    normals = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    handoff.export_surface_source(
        source_path,
        output_path,
        region_name="coil A winding pack",
        selection="all",
        record_outward_normals_global=normals,
        record_normal_basis="test_per_record_normals",
    )

    with h5py.File(output_path, "r") as output:
        phase = output["phase_space"]
        assert np.allclose(phase["surface_outward_normal_global"][:], normals)
        assert np.allclose(phase["mu_outward"][:], [-1.0, -1.0])
        assert phase["magnet_direction"].asstr()[:].tolist() == [
            "incoming",
            "incoming",
        ]
        assert phase["direction_label_basis"].asstr()[:].tolist() == [
            "test_per_record_normals",
            "test_per_record_normals",
        ]


def test_rejects_invalid_per_record_surface_normals(tmp_path):
    source_path = tmp_path / "surface_source.h5"
    _write_source_file(source_path)
    handoff = MagnetSpectralHandoff.from_mapping(handoff_mapping())

    with pytest.raises(ValueError, match="must have shape"):
        handoff.export_surface_source(
            source_path,
            tmp_path / "output.h5",
            region_name="coil A winding pack",
            selection="all",
            record_outward_normals_global=np.asarray([[1.0, 0.0, 0.0]]),
        )
