import json
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from parastell import coil_frame, dagmc_envelope, magnet_local_mesh
from parastell import magnet_stage_handlers
from parastell.magnet_stage_handlers import build_tally_meshes
from parastell.magnet_stage_handlers import (
    _association_cad_group_identity,
)
from parastell.magnet_stage_handlers import (
    _validate_cad_group_signature,
)
from parastell.magnet_stage_handlers import (
    _validate_enabled_weight_window_qualification,
)
from parastell.magnet_stage_handlers import prepare_production_model
from parastell.magnet_stage_handlers import prepare_weight_window_generation
from parastell.magnet_stage_handlers import qualify_weight_windows
from parastell.magnet_stage_handlers import run_unbiased_model
from parastell.magnet_stage_handlers import run_weight_window_generation
from parastell.magnet_stage_handlers import (
    run_weight_window_qualification_campaign,
)


class _CadSolid:
    def __init__(self, volume_cm3, lower, upper):
        self._volume_cm3 = volume_cm3
        self._bounds = SimpleNamespace(
            xmin=lower[0],
            ymin=lower[1],
            zmin=lower[2],
            xmax=upper[0],
            ymax=upper[1],
            zmax=upper[2],
        )

    def Volume(self):
        return self._volume_cm3

    def BoundingBox(self):
        return self._bounds


def _cad_identity_pair(group_index, casing, winding_pack):
    signature = {
        "magnet_casing": {
            "volume_cm3": casing.Volume(),
            "bounding_box_cm": [
                [
                    casing.BoundingBox().xmin,
                    casing.BoundingBox().ymin,
                    casing.BoundingBox().zmin,
                ],
                [
                    casing.BoundingBox().xmax,
                    casing.BoundingBox().ymax,
                    casing.BoundingBox().zmax,
                ],
            ],
        },
        "winding_pack": {
            "volume_cm3": winding_pack.Volume(),
            "bounding_box_cm": [
                [
                    winding_pack.BoundingBox().xmin,
                    winding_pack.BoundingBox().ymin,
                    winding_pack.BoundingBox().zmin,
                ],
                [
                    winding_pack.BoundingBox().xmax,
                    winding_pack.BoundingBox().ymax,
                    winding_pack.BoundingBox().zmax,
                ],
            ],
        },
    }
    provenance = {
        "ordered_filament_index": group_index,
        "cad_solid_group_index": group_index,
        "cad_to_dagmc_identity": {
            "method": (
                "global_role_closed_boundary_volume_and_bounding_box_assignment"
            ),
            "cad_signature": signature,
        },
    }
    return {
        "coil_id": f"coil-{group_index:04d}",
        "winding_pack": {"source_coil_provenance": provenance},
        "casing": {
            "source_coil_provenance": json.loads(json.dumps(provenance))
        },
    }


def test_ww_cad_identity_distinguishes_equal_volume_coils_by_full_signature(
    monkeypatch,
):
    monkeypatch.setattr(
        magnet_stage_handlers,
        "_solid_cad_signature",
        lambda solid: {
            "volume_cm3": solid.Volume(),
            "bounding_box_cm": np.asarray(
                [
                    [
                        solid.BoundingBox().xmin,
                        solid.BoundingBox().ymin,
                        solid.BoundingBox().zmin,
                    ],
                    [
                        solid.BoundingBox().xmax,
                        solid.BoundingBox().ymax,
                        solid.BoundingBox().zmax,
                    ],
                ]
            ),
        },
    )
    group_zero = (
        _CadSolid(20.0, (0, 0, 0), (2, 2, 2)),
        _CadSolid(10.0, (0.5, 0.5, 0.5), (1.5, 1.5, 1.5)),
    )
    group_one = (
        _CadSolid(20.0, (10, 0, 0), (12, 2, 2)),
        _CadSolid(10.0, (10.5, 0.5, 0.5), (11.5, 1.5, 1.5)),
    )
    identity = _association_cad_group_identity(
        _cad_identity_pair(1, *group_one), magnet_id="magnet-one"
    )
    assert identity["cad_solid_group_index"] == 1
    _validate_cad_group_signature(
        group_one,
        identity["cad_signature"],
        magnet_id="magnet-one",
        volume_relative_tolerance=0.02,
        bounding_box_tolerance_cm=1.0e-6,
    )
    with pytest.raises(ValueError, match="signature disagrees"):
        _validate_cad_group_signature(
            group_zero,
            identity["cad_signature"],
            magnet_id="magnet-one",
            volume_relative_tolerance=0.02,
            bounding_box_tolerance_cm=1.0e-6,
        )


def test_ww_cad_signature_matches_roles_when_step_import_reorders_solids(
    monkeypatch,
):
    monkeypatch.setattr(
        magnet_stage_handlers,
        "_solid_cad_signature",
        lambda solid: {
            "volume_cm3": solid.Volume(),
            "bounding_box_cm": np.asarray(
                [
                    [
                        solid.BoundingBox().xmin,
                        solid.BoundingBox().ymin,
                        solid.BoundingBox().zmin,
                    ],
                    [
                        solid.BoundingBox().xmax,
                        solid.BoundingBox().ymax,
                        solid.BoundingBox().zmax,
                    ],
                ]
            ),
        },
    )
    casing = _CadSolid(20.0, (10, 0, 0), (12, 2, 2))
    winding_pack = _CadSolid(10.0, (10.5, 0.5, 0.5), (11.5, 1.5, 1.5))
    identity = _association_cad_group_identity(
        _cad_identity_pair(1, casing, winding_pack),
        magnet_id="magnet-one",
    )
    assignment = _validate_cad_group_signature(
        (winding_pack, casing),
        identity["cad_signature"],
        magnet_id="magnet-one",
        volume_relative_tolerance=1.0e-8,
        bounding_box_tolerance_cm=1.0e-6,
    )
    assert assignment == {"magnet_casing": 1, "winding_pack": 0}


def test_ww_cad_identity_rejects_role_provenance_disagreement():
    casing = _CadSolid(20.0, (10, 0, 0), (12, 2, 2))
    winding_pack = _CadSolid(10.0, (10.5, 0.5, 0.5), (11.5, 1.5, 1.5))
    pair = _cad_identity_pair(1, casing, winding_pack)
    pair["casing"]["source_coil_provenance"]["cad_solid_group_index"] = 0
    with pytest.raises(ValueError, match="disagrees across roles"):
        _association_cad_group_identity(pair, magnet_id="magnet-one")


def test_ww_cad_identity_rejects_legacy_occ_mass_assignment():
    casing = _CadSolid(20.0, (10, 0, 0), (12, 2, 2))
    winding_pack = _CadSolid(10.0, (10.5, 0.5, 0.5), (11.5, 1.5, 1.5))
    pair = _cad_identity_pair(1, casing, winding_pack)
    pair["winding_pack"]["source_coil_provenance"]["cad_to_dagmc_identity"][
        "method"
    ] = "global_role_volume_and_bounding_box_assignment"
    pair["casing"]["source_coil_provenance"]["cad_to_dagmc_identity"][
        "method"
    ] = "global_role_volume_and_bounding_box_assignment"
    with pytest.raises(ValueError, match="unsupported CAD identity"):
        _association_cad_group_identity(pair, magnet_id="magnet-one")


def test_build_tally_meshes_writes_declared_collection(tmp_path, monkeypatch):
    association_path = tmp_path / "associations.json"
    association_path.write_text(
        json.dumps(
            {
                "associations": {"8": {}},
                "centreline_points_by_coil": {"coil-0001": [[0, 0, 0]]},
            }
        ),
        encoding="utf-8",
    )
    pair = SimpleNamespace(
        coil_id="coil-0001",
        magnet_id="magnet-0001",
        winding_pack=SimpleNamespace(
            centroid_global_cm=(0.0, 0.0, 0.0),
            bounding_box_cm=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        ),
    )
    inventory = SimpleNamespace(canonical_geometry_fingerprint="a" * 64)
    monkeypatch.setattr(
        dagmc_envelope,
        "discover_magnet_volumes",
        lambda *args, **kwargs: inventory,
    )
    monkeypatch.setattr(
        dagmc_envelope, "select_magnet_pairs", lambda *args, **kwargs: (pair,)
    )
    monkeypatch.setattr(
        coil_frame,
        "parallel_transport_frame",
        lambda points: SimpleNamespace(sample=lambda point: object()),
    )
    monkeypatch.setattr(
        magnet_local_mesh,
        "build_local_mesh_definition",
        lambda *args, **kwargs: SimpleNamespace(
            bin_count=1, to_dict=lambda: {"mesh": "ok"}
        ),
    )
    monkeypatch.setattr(
        magnet_local_mesh,
        "qualify_local_mesh_nonoverlap",
        lambda *args, **kwargs: {
            "nonoverlap_qualified": False,
            "status": "NOT_QUALIFIED",
        },
    )
    output = tmp_path / "local_meshes.json"
    result = build_tally_meshes(
        {
            "associations_path": str(association_path),
            "dagmc_path": str(tmp_path / "geometry.h5m"),
            "local_mesh_manifest_path": str(output),
            "selection": "all",
        },
        tmp_path,
    )
    assert output.is_file()
    assert result["meshes"] == {"magnet-0001": {"mesh": "ok"}}
    assert result["nonoverlap_qualified"] is False
    assert result["nonoverlap_qualification"]["status"] == "NOT_QUALIFIED"


def test_run_unbiased_model_validates_secondary_photons(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "geometry.xml",
        "materials.xml",
        "settings.xml",
        "tallies.xml",
    ):
        (model / name).write_text("<xml/>", encoding="utf-8")
    statepoint = model / "statepoint.2.h5"
    with h5py.File(statepoint, "w") as stream:
        stream.attrs["photon_transport"] = 1
        stream.attrs["tallies_present"] = 1
        for name, value in (
            ("current_batch", 2),
            ("n_batches", 2),
            ("n_particles", 10),
            ("n_realizations", 2),
        ):
            stream.create_dataset(name, data=value)
        tallies = stream.create_group("tallies")
        for index, name in enumerate(
            (
                "pstl_magnet_neutron_ccfe_709_volume_flux",
                "pstl_magnet_photon_configured_volume_flux",
                "pstl_magnet_production_photon",
            ),
            start=1,
        ):
            tally = tallies.create_group(f"tally {index}")
            tally.create_dataset("name", data=name)
            tally.create_dataset("n_realizations", data=2)
            tally.create_dataset("results", data=np.asarray([[[2.0, 4.0]]]))
    vector = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8")])
    bank_type = np.dtype(
        [("particle", "i8"), ("surf_id", "i8"), ("u", vector)]
    )
    bank = np.zeros(2, dtype=bank_type)
    bank["particle"] = [2112, 22]
    bank["surf_id"] = [1, 2]
    bank["u"]["x"] = 1.0
    source_path = model / "surface_source.h5"
    with h5py.File(source_path, "w") as stream:
        stream.create_dataset("source_bank", data=bank)
    monkeypatch.setitem(
        sys.modules,
        "openmc",
        SimpleNamespace(run=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "OpenMC version 0.16.0\n"
                "Commit hash: 617d35a5063c57796b43428bc401e627d2011046\n"
            )
        ),
    )
    report_path = tmp_path / "transport.json"
    result = run_unbiased_model(
        {
            "model_directory": str(model),
            "statepoint_path": str(statepoint),
            "surface_source_paths": [str(source_path)],
            "transport_report_path": str(report_path),
            "max_surface_particles": 100,
            "max_surface_files": 1,
            "expected_openmc_commit": "617d35a5063c57796b43428bc401e627d2011046",
        },
        tmp_path,
    )
    assert result["status"] == "PASS"
    assert result["surface_source"]["particle_counts"] == {
        "neutron": 1,
        "photon": 1,
    }
    assert report_path.is_file()


def test_rejected_weight_window_mesh_propagates_to_unbiased_production(
    tmp_path,
):
    mesh_status = tmp_path / "mesh-status.json"
    mesh_status.write_text(
        json.dumps(
            {
                "classification": "REJECTED_INSTABILITY",
                "weight_windows_enabled": False,
                "production_transport": "UNBIASED",
                "reason": "conformal tetrahedralization failed",
                "evidence": [{"candidate": "hxt-fused"}],
            }
        ),
        encoding="utf-8",
    )
    generation_contract = tmp_path / "generation-contract.json"
    prepared = prepare_weight_window_generation(
        {
            "weight_window_mesh_status_path": str(mesh_status),
            "generation_contract_path": str(generation_contract),
        },
        tmp_path,
    )
    assert prepared["status"] == "SKIPPED_UNBIASED_FALLBACK"

    generation_report = tmp_path / "generation-report.json"
    generated = run_weight_window_generation(
        {
            "generation_contract_path": str(generation_contract),
            "generation_report_path": str(generation_report),
        },
        tmp_path,
    )
    assert generated["classification"] == "REJECTED_INSTABILITY"

    unbiased_responses = tmp_path / "unbiased-responses.json"
    weighted_responses = tmp_path / "weighted-responses.json"
    campaign_report = tmp_path / "campaign.json"
    campaign = run_weight_window_qualification_campaign(
        {
            "generation_report_path": str(generation_report),
            "unbiased_responses_path": str(unbiased_responses),
            "weight_window_responses_path": str(weighted_responses),
            "campaign_report_path": str(campaign_report),
        },
        tmp_path,
    )
    assert campaign["status"] == "SKIPPED_UNBIASED_FALLBACK"
    assert json.loads(unbiased_responses.read_text())["responses"] == []

    qualification_path = tmp_path / "qualification.json"
    qualification = qualify_weight_windows(
        {
            "campaign_report_path": str(campaign_report),
            "qualification_path": str(qualification_path),
        },
        tmp_path,
    )
    assert qualification["weight_windows_enabled"] is False

    baseline = tmp_path / "unbiased-model"
    baseline.mkdir()
    for name in (
        "geometry.xml",
        "materials.xml",
        "settings.xml",
        "tallies.xml",
    ):
        (baseline / name).write_text(f"<{name}/>", encoding="utf-8")
    production = tmp_path / "production-model"
    production_manifest = tmp_path / "production-model.json"
    result = prepare_production_model(
        {
            "qualification_path": str(qualification_path),
            "unbiased_model_directory": str(baseline),
            "production_model_directory": str(production),
            "production_model_manifest_path": str(production_manifest),
        },
        tmp_path,
    )
    assert result["production_transport"] == "UNBIASED_FALLBACK"
    assert sorted(path.name for path in production.glob("*.xml")) == [
        "geometry.xml",
        "materials.xml",
        "settings.xml",
        "tallies.xml",
    ]


def test_enabled_weight_window_qualification_is_campaign_and_particle_bound():
    qualification = {
        "classification": "QUALIFIED_AND_ENABLED",
        "weight_windows_enabled": True,
        "run_contract_sha256": "a" * 64,
        "artifact_particle_type": "neutron",
    }
    campaign = {"status": "PASS", "run_contract_sha256": "a" * 64}
    _validate_enabled_weight_window_qualification(
        qualification, campaign, expected_particle_type="neutron"
    )
    with pytest.raises(ValueError, match="selected campaign"):
        _validate_enabled_weight_window_qualification(
            qualification,
            {**campaign, "run_contract_sha256": "b" * 64},
            expected_particle_type="neutron",
        )
    with pytest.raises(ValueError, match="particle type"):
        _validate_enabled_weight_window_qualification(
            qualification, campaign, expected_particle_type="photon"
        )
