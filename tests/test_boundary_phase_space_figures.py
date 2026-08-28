import copy
import hashlib
import json
import sys
import types

import h5py
import numpy as np
import pytest

from parastell.boundary_phase_space_figures import summarize_phase_space
from parastell.boundary_phase_space_figures import (
    surface_current_by_particle_and_sense,
)
from parastell.boundary_phase_space_figures import validate_figure_inputs
from parastell.boundary_phase_space_figures import write_phase_space_figures
from parastell.boundary_phase_space_figures import (
    _write_phase_space_figures_from_legacy_audit,
)


def _phase_manifest(histories=10):
    return {
        "schema": "parastell.openmc16_surface_phase_space/v1.0.0",
        "raw_phase_space_pass": True,
        "source_histories": histories,
        "history_binding": {
            "kind": "fixed_source_run",
            "run_id": "bounded-test",
            "source_histories": histories,
            "openmc_version": "0.16.0",
            "settings_payload_sha256": "a" * 64,
            "statepoint_sha256": "b" * 64,
        },
    }


def _records():
    return {
        "position_global_cm": np.asarray(
            [[1.5, 2.5, 3.0], [0.5, 4.0, 4.0], [2.5, 2.5, 5.0]]
        ),
        "position_local_cm": np.asarray(
            [[1.5, 2.5, 0.0], [0.5, 4.0, 0.0], [2.5, 2.5, 0.0]]
        ),
        "direction_global": np.asarray(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
        "outward_normal_global": np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        "energy_eV": np.asarray([14.1e6, 1.0e6, 1.0]),
        "time_s": np.asarray([0.0, 1.0e-9, 2.0e-9]),
        "weight": np.asarray([1.0, 2.0, 3.0]),
        "weight_per_source_history": np.asarray([0.1, 0.2, 0.3]),
        "particle": np.asarray([2112, 2112, 22]),
        "surface_id": np.asarray([11, 11, 12]),
        "mu": np.asarray([-1.0, 1.0, 0.0]),
        "facet_id": np.asarray([7, 8, 9]),
        "barycentric_coordinates": np.asarray(
            [[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.0, 0.5, 0.5]]
        ),
        "direction_local": np.asarray(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
        "local_frame_global": np.repeat(np.eye(3)[None, :, :], 3, axis=0),
    }


def _facet_catalog():
    return {
        "normal_source": "dagmc_forward_reverse_topology",
        "topology_manifest_sha256": "c" * 64,
        "facet_id": np.asarray([7, 8, 9]),
        "surface_id": np.asarray([11, 11, 12]),
        "vertices_global_cm": np.asarray(
            [
                [[0.0, 0.0, 3.0], [5.0, 0.0, 3.0], [0.0, 5.0, 3.0]],
                [[0.0, 0.0, 4.0], [5.0, 0.0, 4.0], [0.0, 5.0, 4.0]],
                [[0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [0.0, 5.0, 5.0]],
            ]
        ),
        "outward_normal_global": np.asarray([[0.0, 0.0, 1.0]] * 3),
    }


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_audit(tmp_path):
    geometry = tmp_path / "geometry.h5m"
    bank = tmp_path / "surface_source.h5"
    settings = tmp_path / "settings.xml"
    statepoint = tmp_path / "statepoint.h5"
    topology = tmp_path / "topology.json"
    localized = tmp_path / "localized.npz"
    catalog = tmp_path / "catalog.npz"
    geometry.write_bytes(b"geometry")
    bank.write_bytes(b"bank")
    settings.write_bytes(b"settings")
    statepoint.write_bytes(b"statepoint")
    topology.write_text('{"topology":"verified"}\n', encoding="utf-8")
    records = _records()
    records["openmc_weight"] = records.pop("weight")
    records["particle_pdg"] = records.pop("particle")
    np.savez(localized, **records)
    facet_catalog = _facet_catalog()
    facet_catalog.pop("normal_source")
    facet_catalog.pop("topology_manifest_sha256")
    np.savez(catalog, **facet_catalog)
    phase_manifest = _phase_manifest()
    phase_manifest["source_files"] = [
        {"path": str(bank), "sha256": _digest(bank)}
    ]
    binding = phase_manifest["history_binding"]
    binding.update(
        {
            "settings_payload_path": str(settings),
            "settings_payload_sha256": _digest(settings),
            "statepoint_path": str(statepoint),
            "statepoint_sha256": _digest(statepoint),
        }
    )
    localization_binding = {
        "geometry_sha256": _digest(geometry),
        "source_bank_sha256s": [_digest(bank)],
        "source_histories": 10,
        "settings_payload_sha256": _digest(settings),
        "statepoint_sha256": _digest(statepoint),
        "localized_records_sha256": _digest(localized),
        "facet_catalog_sha256": _digest(catalog),
        "topology_manifest_sha256": _digest(topology),
    }
    audit = {
        "schema": "parastell.openmc16_surface_run_audit/v1.0.0",
        "status": "COMPLETE_CROSSING_BANK",
        "geometry_label": "synthetic-variant",
        "geometry": {"path": str(geometry), "sha256": _digest(geometry)},
        "source_banks": [{"path": str(bank), "sha256": _digest(bank)}],
        "phase_space_manifest": phase_manifest,
        "localized_records": {
            "path": str(localized),
            "sha256": _digest(localized),
        },
        "facet_catalog": {"path": str(catalog), "sha256": _digest(catalog)},
        "topology_manifest": {
            "path": str(topology),
            "sha256": _digest(topology),
        },
        "localization_topology_binding": localization_binding,
    }
    audit_path = tmp_path / "surface_run_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit_path, audit


def _strict_renderer_fixture(tmp_path, monkeypatch):
    geometry = tmp_path / "trusted.h5m"
    model = tmp_path / "model.xml"
    statepoint = tmp_path / "statepoint.h5"
    terminal = tmp_path / "openmc.log"
    inventory = tmp_path / "inventory.json"
    acceptance = tmp_path / "acceptance.json"
    bank = tmp_path / "bank.h5"
    geometry.write_bytes(b"h5m")
    model.write_bytes(b"model")
    statepoint.write_bytes(b"statepoint")
    terminal.write_bytes(b"terminal")
    inventory.write_bytes(b"inventory")
    acceptance.write_bytes(b"acceptance")
    vector = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    dtype = np.dtype(
        [
            ("r", vector),
            ("u", vector),
            ("E", "<f8"),
            ("time", "<f8"),
            ("wgt", "<f8"),
            ("delayed_group", "<i4"),
            ("surf_id", "<i4"),
            ("particle", "<i4"),
        ]
    )
    native = np.zeros(3, dtype=dtype)
    records = _records()
    for axis_index, axis in enumerate("xyz"):
        native["r"][axis] = records["position_global_cm"][:, axis_index]
        native["u"][axis] = records["direction_global"][:, axis_index]
    native["E"] = records["energy_eV"]
    native["time"] = records["time_s"]
    native["wgt"] = records["weight"]
    native["surf_id"] = records["surface_id"]
    native["particle"] = records["particle"]
    with h5py.File(bank, "w") as source:
        source.create_dataset("source_bank", data=native)

    localized = tmp_path / "localized_strict.npz"
    records["openmc_weight"] = records.pop("weight")
    records["particle_pdg"] = records.pop("particle")
    records["delayed_group"] = np.zeros(3, dtype=int)
    records["source_file_index"] = np.zeros(3, dtype=int)
    records["source_record_index"] = np.arange(3, dtype=int)
    np.savez(localized, **records)
    catalog_values = _facet_catalog()
    catalog_values.pop("normal_source")
    catalog_values.pop("topology_manifest_sha256")
    catalog_values["canonical_facet_id"] = catalog_values.pop("facet_id")
    catalog_values["triangle_vertices_global_cm"] = catalog_values.pop(
        "vertices_global_cm"
    )
    catalog = tmp_path / "catalog_strict.npz"
    np.savez(catalog, **catalog_values)
    requests = tmp_path / "envelopes.json"
    requests.write_text(
        json.dumps({"envelope_requests": [{"volume_id": 1}]}),
        encoding="utf-8",
    )
    strict = {
        "schema": "parastell.openmc16_surface_run_audit/v1.0.0",
        "classification": "COMPLETE_CROSSING_BANK",
        "dagmc": {
            "path": str(geometry),
            "sha256": _digest(geometry),
            "surface_ids": list(range(1, 73)),
        },
        "model": {
            "path": str(model),
            "sha256": _digest(model),
            "source_histories": 10,
        },
        "statepoint": {"path": str(statepoint), "sha256": _digest(statepoint)},
        "terminal_log": {"path": str(terminal), "sha256": _digest(terminal)},
        "surface_source_files": [{"path": str(bank), "sha256": _digest(bank)}],
        "accepted_magnet_inventory": {
            "path": str(inventory),
            "sha256": _digest(inventory),
        },
        "root_acceptance_receipt": {
            "path": str(acceptance),
            "sha256": _digest(acceptance),
        },
        "required_particles": ["neutron", "photon"],
        "localization_topology_binding": {"manifest_sha256": "e" * 64},
    }
    audit_path = tmp_path / "strict.json"
    audit_path.write_text(
        json.dumps({"strict_audit": strict}), encoding="utf-8"
    )

    class Artifacts:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Envelope:
        def facet_metadata(self):
            return catalog_values

    fake = types.ModuleType("parastell.openmc16_export")
    fake.StrictSurfaceRunArtifacts = Artifacts
    fake.audit_openmc16_surface_run = lambda *args, **kwargs: copy.deepcopy(
        strict
    )
    fake._extract_envelopes_from_h5m = lambda *args, **kwargs: [Envelope()]
    monkeypatch.setitem(sys.modules, "parastell.openmc16_export", fake)
    return {
        "audit": audit_path,
        "requests": requests,
        "localized": localized,
        "catalog": catalog,
        "strict": strict,
    }


def test_phase_space_validation_preserves_all_required_correlations():
    values = validate_figure_inputs(_records())
    summary = summarize_phase_space(
        values,
        phase_space_manifest=_phase_manifest(),
        grazing_tolerance=1.0e-8,
    )

    assert summary["record_count"] == 3
    assert summary["incoming_count"] == 1
    assert summary["outgoing_count"] == 1
    assert summary["grazing_count"] == 1
    assert summary["per_source_weight_sum"] == pytest.approx(0.6)
    assert summary["facet_localization_present"] is True
    assert summary["barycentric_localization_present"] is True
    assert summary["local_coordinates_present"] is True


def test_phase_space_validation_accepts_root_auditor_column_names():
    records = _records()
    records["openmc_weight"] = records.pop("weight")
    records["particle_pdg"] = records.pop("particle")

    values = validate_figure_inputs(records)

    assert values.weight.tolist() == [1.0, 2.0, 3.0]
    assert values.particle.tolist() == [2112, 2112, 22]


def test_phase_space_validation_is_global_frame_and_surface_id_neutral():
    records = _records()
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([1700.0, -900.0, 250.0])
    records["position_global_cm"] = (
        records["position_global_cm"] @ rotation.T + translation
    )
    records["direction_global"] = records["direction_global"] @ rotation.T
    records["outward_normal_global"] = (
        records["outward_normal_global"] @ rotation.T
    )
    records["surface_id"] = np.asarray([103, 103, 905])

    values = validate_figure_inputs(records)
    summary = summarize_phase_space(
        values,
        phase_space_manifest=_phase_manifest(),
        grazing_tolerance=1.0e-8,
    )

    assert values.mu.tolist() == [-1.0, 1.0, 0.0]
    assert summary["surface_ids"] == [103, 905]
    assert summary["incoming_count"] == 1
    assert summary["outgoing_count"] == 1


def test_phase_space_validation_rejects_mu_inconsistent_with_normal_sense():
    records = copy.deepcopy(_records())
    records["mu"][0] = 1.0

    with pytest.raises(ValueError, match="stored mu"):
        validate_figure_inputs(records)


def test_phase_space_validation_rejects_invalid_barycentric_coordinates():
    records = copy.deepcopy(_records())
    records["barycentric_coordinates"][0] = [-0.1, 0.4, 0.7]

    with pytest.raises(ValueError, match="barycentric"):
        validate_figure_inputs(records)


def test_matched_figures_are_hash_bound(tmp_path):
    audit_path, _ = _verified_audit(tmp_path)
    output = tmp_path / "figures"
    manifest = _write_phase_space_figures_from_legacy_audit(
        audit_path,
        output,
        expected_run_audit_sha256=_digest(audit_path),
    )

    assert len(manifest["figures"]) == 5
    assert all((output / row["path"]).is_file() for row in manifest["figures"])
    assert (output / "FIGURE_MANIFEST.json").is_file()
    assert manifest["facet_catalog_overlay"] is True
    assert len(manifest["topology_manifest_sha256"]) == 64
    assert manifest["surface_run_audit_sha256"] == _digest(audit_path)
    assert manifest["surface_current"]["currents"]["neutron_incoming"] == [
        0.1,
        0.0,
    ]


def test_current_by_surface_preserves_particle_and_sense():
    values = validate_figure_inputs(_records())
    result = surface_current_by_particle_and_sense(
        values, source_histories=10, grazing_tolerance=1.0e-8
    )

    assert result["surface_ids"] == [11, 12]
    assert result["currents"] == {
        "neutron_incoming": [0.1, 0.0],
        "neutron_outgoing": [0.2, 0.0],
        "photon_incoming": [0.0, 0.0],
        "photon_outgoing": [0.0, 0.0],
    }


def test_renderer_rejects_noncomplete_run_audit(tmp_path):
    audit_path, audit = _verified_audit(tmp_path)
    audit["status"] = "SAMPLED_CROSSING_BANK"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="not COMPLETE_CROSSING_BANK"):
        _write_phase_space_figures_from_legacy_audit(
            audit_path,
            tmp_path / "figures",
            expected_run_audit_sha256=_digest(audit_path),
        )


def test_renderer_rejects_forged_localized_rows(tmp_path):
    audit_path, audit = _verified_audit(tmp_path)
    localized = tmp_path / "localized.npz"
    records = _records()
    records["position_global_cm"][0, 0] += 1.0
    records["openmc_weight"] = records.pop("weight")
    records["particle_pdg"] = records.pop("particle")
    np.savez(localized, **records)
    new_hash = _digest(localized)
    audit["localized_records"]["sha256"] = new_hash
    audit["localization_topology_binding"][
        "localized_records_sha256"
    ] = new_hash
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="barycentric reconstruction"):
        _write_phase_space_figures_from_legacy_audit(
            audit_path,
            tmp_path / "figures",
            expected_run_audit_sha256=_digest(audit_path),
        )


def test_renderer_rejects_stale_expected_audit_hash(tmp_path):
    audit_path, _ = _verified_audit(tmp_path)
    with pytest.raises(ValueError, match="audit file/hash binding"):
        _write_phase_space_figures_from_legacy_audit(
            audit_path,
            tmp_path / "figures",
            expected_run_audit_sha256="f" * 64,
        )


@pytest.mark.parametrize("filename", ["surface_source.h5", "topology.json"])
def test_renderer_rejects_tampered_bound_artifact(tmp_path, filename):
    audit_path, _ = _verified_audit(tmp_path)
    (tmp_path / filename).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="path/hash binding failed"):
        _write_phase_space_figures_from_legacy_audit(
            audit_path,
            tmp_path / "figures",
            expected_run_audit_sha256=_digest(audit_path),
        )


def test_trusted_renderer_rechecks_native_rows_and_lists_all_surfaces(
    tmp_path, monkeypatch
):
    fixture = _strict_renderer_fixture(tmp_path, monkeypatch)
    output = tmp_path / "trusted_figures"
    manifest = write_phase_space_figures(
        fixture["audit"],
        output,
        expected_strict_audit_sha256=_digest(fixture["audit"]),
        envelope_request_path=fixture["requests"],
        expected_envelope_request_sha256=_digest(fixture["requests"]),
        localized_records_path=fixture["localized"],
        expected_localized_records_sha256=_digest(fixture["localized"]),
        facet_catalog_path=fixture["catalog"],
        expected_facet_catalog_sha256=_digest(fixture["catalog"]),
        geometry_label="trusted-test",
    )

    current = manifest["surface_current"]
    assert current["surface_ids"] == list(range(1, 73))
    assert current["qualification"][0] == "NO_RECORDS"
    assert current["qualification"][10] == "INSUFFICIENT_STATISTICS"
    assert current["qualification"][11] == "INSUFFICIENT_STATISTICS"
    assert current["qualification"].count("NO_RECORDS") == 70


def test_trusted_renderer_rejects_native_row_forgery(tmp_path, monkeypatch):
    fixture = _strict_renderer_fixture(tmp_path, monkeypatch)
    with np.load(fixture["localized"], allow_pickle=False) as archive:
        localized = {name: archive[name] for name in archive.files}
    localized["energy_eV"][0] += 1.0
    np.savez(fixture["localized"], **localized)

    with pytest.raises(ValueError, match="differs from native row"):
        write_phase_space_figures(
            fixture["audit"],
            tmp_path / "figures",
            expected_strict_audit_sha256=_digest(fixture["audit"]),
            envelope_request_path=fixture["requests"],
            expected_envelope_request_sha256=_digest(fixture["requests"]),
            localized_records_path=fixture["localized"],
            expected_localized_records_sha256=_digest(fixture["localized"]),
            facet_catalog_path=fixture["catalog"],
            expected_facet_catalog_sha256=_digest(fixture["catalog"]),
            geometry_label="forged-test",
        )


def test_trusted_renderer_rejects_supplied_catalog_forgery(
    tmp_path, monkeypatch
):
    fixture = _strict_renderer_fixture(tmp_path, monkeypatch)
    with np.load(fixture["catalog"], allow_pickle=False) as archive:
        catalog = {name: archive[name] for name in archive.files}
    catalog["outward_normal_global"][0] *= -1.0
    np.savez(fixture["catalog"], **catalog)

    with pytest.raises(ValueError, match="supplied facet catalog differs"):
        write_phase_space_figures(
            fixture["audit"],
            tmp_path / "figures",
            expected_strict_audit_sha256=_digest(fixture["audit"]),
            envelope_request_path=fixture["requests"],
            expected_envelope_request_sha256=_digest(fixture["requests"]),
            localized_records_path=fixture["localized"],
            expected_localized_records_sha256=_digest(fixture["localized"]),
            facet_catalog_path=fixture["catalog"],
            expected_facet_catalog_sha256=_digest(fixture["catalog"]),
            geometry_label="forged-test",
        )


def test_summary_rejects_independently_mismatched_history_binding():
    values = validate_figure_inputs(_records())
    manifest = _phase_manifest(histories=10)
    manifest["history_binding"]["source_histories"] = 11

    with pytest.raises(ValueError, match="manifest and history binding"):
        summarize_phase_space(
            values,
            phase_space_manifest=manifest,
            grazing_tolerance=1.0e-8,
        )


def test_summary_rejects_legacy_independent_history_count():
    values = validate_figure_inputs(_records())

    with pytest.raises(TypeError, match="source_histories"):
        summarize_phase_space(
            values,
            source_histories=10,
            grazing_tolerance=1.0e-8,
        )


def test_summary_rejects_mismatched_derived_normalized_weights():
    records = _records()
    records["weight_per_source_history"] = np.asarray([0.1, 0.2, 0.4])
    values = validate_figure_inputs(records)

    with pytest.raises(ValueError, match="weight_per_source_history"):
        summarize_phase_space(
            values,
            phase_space_manifest=_phase_manifest(),
            grazing_tolerance=1.0e-8,
        )
