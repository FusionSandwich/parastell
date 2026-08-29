from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from parastell.activation_campaign import build_activation_campaign
from parastell.alara_activation import (
    build_alara_handoff,
    validate_alara_result_files,
    write_alara_package,
)
from parastell.delayed_photon_source import (
    DelayedPhotonSourceError,
    build_delayed_photon_solver_handoff,
    validate_delayed_photon_solver_handoff,
    write_delayed_photon_solver_package,
)
from parastell.energy_groups import (
    VITAMIN_J_175_EDGES_EV,
    VITAMIN_J_175_EDGES_SHA256,
)


TIME_LABELS = (
    "shutdown",
    "1 s",
    "60 s",
    "1 h",
    "1 d",
    "1 w",
    "30 d",
    "1 y",
    "5 y",
    "10 y",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance():
    return {
        "raw_h5m_sha256": "a" * 64,
        "canonical_geometry_fingerprint": "b" * 64,
        "statepoint_sha256": "c" * 64,
        "nuclear_data_sha256": "d" * 64,
        "physical_source_rate_per_s": 2.0,
        "source_rate_scope": "modeled_90_degree_period",
        "normalization": "per_source_history",
        "statistics_classification": "WORKFLOW_SMOKE_ONLY",
        "provenance_origin": "SYNTHETIC_WORKFLOW_FIXTURE",
    }


def _activation_row(domain_id, volume):
    return {
        "domain_id": domain_id,
        "volume_cm3": volume,
        "energy_edges_eV": list(VITAMIN_J_175_EDGES_EV),
        "bin_integrated_flux_per_source": [1.0] * 175,
        "physical_source_rate_per_s": 2.0,
        "material": {
            "material_id": f"material-{domain_id}",
            "composition_sha256": "e" * 64,
            "alara_constituents": [
                {
                    "element_symbol": "cu:63",
                    "relative_density": 1.0,
                    "volume_fraction": 1.0,
                }
            ],
        },
    }


def _photon_text(*, missing_last_total=False):
    lines = []
    for zone_index, nuclide in enumerate(("cu-64", "co-60")):
        for time_index, label in enumerate(TIME_LABELS):
            lines.append(
                "\t".join([nuclide, f"  {label}  ", "0", "0", "0", "0", "0"])
            )
        for time_index, label in enumerate(TIME_LABELS):
            if missing_last_total and zone_index == 1 and time_index == 9:
                continue
            base = float((zone_index + 1) * (time_index + 1))
            values = [base * value for value in (1, 2, 3, 4, 5)]
            lines.append(
                "\t".join(
                    ["TOTAL", f"  {label}  "]
                    + [f"{value:.6g}" for value in values]
                )
            )
    return "\n".join(lines) + "\n"


def _valid_files(
    tmp_path,
    *,
    photon_text=None,
    bind_result=True,
    delayed_photon_energy_edges_eV=None,
):
    campaign = build_activation_campaign()
    handoff_kwargs = {}
    if delayed_photon_energy_edges_eV is not None:
        handoff_kwargs["delayed_photon_energy_edges_eV"] = (
            delayed_photon_energy_edges_eV
        )
    handoff = build_alara_handoff(
        activation_rows=[
            _activation_row("magnet-0000", 10.0),
            _activation_row("magnet-0001", 20.0),
        ],
        campaign=campaign,
        group_structure_sha256=VITAMIN_J_175_EDGES_SHA256,
        openmc_provenance=_provenance(),
        **handoff_kwargs,
    )
    package_root = tmp_path / "alara-package"
    write_alara_package(package_root, handoff)
    manifest_path = package_root / "alara_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = campaign["irradiation_checkpoint_s"][0]
    relative_input = f"irradiation_{checkpoint:012d}s/alara.inp"
    input_sha256 = next(
        row["sha256"]
        for row in manifest["files"]
        if row["path"] == relative_input
    )
    output = tmp_path / "alara.out"
    output.write_text(
        "Response Units: Bq /cm3\n"
        "*** Specific Activity [Bq/cm3] ***\n"
        "cu-63 1 0 1 1 1 1 1\n"
        "total 0 0 1 1 1 1 1\n"
        "*** Total Decay Heat [W/cm3] ***\n"
        "cu-63 1 0 1 1 1 1 1\n"
        "total 0 0 1 1 1 1 1\n"
        "*** Photon Source Distribution [gammas/s/cm3] : photons\n",
        encoding="utf-8",
    )
    stderr = tmp_path / "alara.stderr"
    stderr.write_text("", encoding="utf-8")
    photon = tmp_path / "delayed_photon_source.txt"
    photon.write_text(photon_text or _photon_text(), encoding="utf-8")
    kwargs = {}
    if bind_result:
        kwargs = {
            "irradiation_checkpoint_s": checkpoint,
            "alara_input_sha256": input_sha256,
            "activation_zone_ids": ["magnet-0000", "magnet-0001"],
            "cooling_offset_s": campaign["cooling_offset_s"],
        }
    receipt = validate_alara_result_files(
        output_path=output,
        stderr_path=stderr,
        delayed_photon_path=photon,
        input_manifest_sha256=_sha256(manifest_path),
        remote_run_root="/fresh/run",
        **kwargs,
    )
    receipt_path = tmp_path / "result.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return photon, receipt_path, manifest_path


def test_bridge_preserves_zone_time_units_and_absolute_rates(tmp_path):
    photon, receipt, manifest = _valid_files(tmp_path)
    bundle = build_delayed_photon_solver_handoff(
        delayed_photon_path=photon,
        alara_result_receipt_path=receipt,
        alara_input_manifest_path=manifest,
    )
    assert bundle["status"] == "SOLVER_SOURCE_CONTRACTS_READY"
    assert bundle["energy"]["energy_edges_eV"] == [
        0.0,
        1.0e3,
        1.0e4,
        1.0e5,
        1.0e6,
        1.0e7,
    ]
    one_hour = bundle["snapshots"][3]
    assert one_hour["cooling_offset_s"] == 3_600
    assert [row["zone_id"] for row in one_hour["zones"]] == [
        "magnet-0000",
        "magnet-0001",
    ]
    assert one_hour["zones"][0]["group_emission_rate_photons_per_s"] == [
        40.0,
        80.0,
        120.0,
        160.0,
        200.0,
    ]
    assert one_hour["zones"][0][
        "group_emission_rate_density_photons_per_cm3_s"
    ] == [4.0, 8.0, 12.0, 16.0, 20.0]
    assert one_hour["total_emission_rate_photons_per_s"] == 3_000.0
    assert (
        bundle["absolute_normalization"]["additional_rate_multiplier_allowed"]
        is False
    )
    assert (
        bundle["absolute_normalization"][
            "bound_zone_volume_applied_exactly_once"
        ]
        is True
    )
    assert bundle["uncertainty"]["status"] == ("UNAVAILABLE_NOT_PROPAGATED")
    assert bundle["covariance"]["status"] == ("UNAVAILABLE_NOT_FABRICATED")
    assert set(bundle["solver_routes"]) == {
        "openmc",
        "mcnp",
        "geant4",
        "opensn",
        "radiant",
    }


def test_bridge_writes_create_only_solver_contracts(tmp_path):
    photon, receipt, manifest = _valid_files(tmp_path)
    bundle = build_delayed_photon_solver_handoff(
        delayed_photon_path=photon,
        alara_result_receipt_path=receipt,
        alara_input_manifest_path=manifest,
    )
    output = tmp_path / "solver-sources"
    paths = write_delayed_photon_solver_package(output, bundle)
    assert len(paths) == 7
    assert (output / "openmc_delayed_photon_source_contract.json").is_file()
    assert (output / "radiant_delayed_photon_source_contract.json").is_file()
    validate_delayed_photon_solver_handoff(
        json.loads(
            (output / "delayed_photon_source.json").read_text(encoding="utf-8")
        )
    )
    with pytest.raises(FileExistsError):
        write_delayed_photon_solver_package(output, bundle)


def test_bridge_rejects_unbound_or_modified_artifacts(tmp_path):
    photon, receipt, manifest = _valid_files(tmp_path, bind_result=False)
    with pytest.raises(DelayedPhotonSourceError, match="execution binding"):
        build_delayed_photon_solver_handoff(
            delayed_photon_path=photon,
            alara_result_receipt_path=receipt,
            alara_input_manifest_path=manifest,
        )

    other = tmp_path / "modified"
    other.mkdir()
    photon, receipt, manifest = _valid_files(other)
    photon.write_text(
        photon.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(DelayedPhotonSourceError, match="hash mismatch"):
        build_delayed_photon_solver_handoff(
            delayed_photon_path=photon,
            alara_result_receipt_path=receipt,
            alara_input_manifest_path=manifest,
        )


def test_bridge_rejects_incomplete_zone_time_totals(tmp_path):
    photon, receipt, manifest = _valid_files(
        tmp_path, photon_text=_photon_text(missing_last_total=True)
    )
    with pytest.raises(DelayedPhotonSourceError, match="incomplete"):
        build_delayed_photon_solver_handoff(
            delayed_photon_path=photon,
            alara_result_receipt_path=receipt,
            alara_input_manifest_path=manifest,
        )


def test_real_format_fixture_uses_bound_groups_and_multiplies_volume_once(
    tmp_path,
):
    fixture = (
        Path(__file__).parent
        / "data"
        / "alara_delayed_photon_source_density.tsv"
    ).read_text(encoding="utf-8")
    photon, receipt, manifest = _valid_files(
        tmp_path,
        photon_text=fixture,
        delayed_photon_energy_edges_eV=[0.0, 1.0e3, 1.0e6, 1.0e7],
    )
    bundle = build_delayed_photon_solver_handoff(
        delayed_photon_path=photon,
        alara_result_receipt_path=receipt,
        alara_input_manifest_path=manifest,
    )
    assert bundle["energy"]["group_count"] == 3
    shutdown = bundle["snapshots"][0]
    assert shutdown["cooling_time_label"] == "shutdown"
    assert shutdown["zones"][0][
        "group_emission_rate_density_photons_per_cm3_s"
    ] == [1.0, 2.0, 3.0]
    assert shutdown["zones"][0]["group_emission_rate_photons_per_s"] == [
        10.0,
        20.0,
        30.0,
    ]
    assert shutdown["zones"][1]["group_emission_rate_photons_per_s"] == [
        40.0,
        80.0,
        120.0,
    ]
    assert shutdown["total_emission_rate_photons_per_s"] == 300.0
    aliases = bundle["near_zero_time_alias_policy"]["accepted_aliases"]
    assert [row["zone_id"] for row in aliases] == [
        "magnet-0000",
        "magnet-0001",
    ]
    assert all(row["alias_label"] == "1e-12 s" for row in aliases)
    assert all(row["spectra_match_status"] == "PASS" for row in aliases)


def test_near_zero_alias_with_different_spectrum_fails_closed(tmp_path):
    fixture = (
        Path(__file__).parent
        / "data"
        / "alara_delayed_photon_source_density.tsv"
    ).read_text(encoding="utf-8")
    fixture = fixture.replace(
        "TOTAL\t1e-12 s\t1\t2\t3",
        "TOTAL\t1e-12 s\t1\t2\t3.000001",
        1,
    )
    photon, receipt, manifest = _valid_files(
        tmp_path,
        photon_text=fixture,
        delayed_photon_energy_edges_eV=[0.0, 1.0e3, 1.0e6, 1.0e7],
    )
    with pytest.raises(
        DelayedPhotonSourceError, match="conflicting duplicate"
    ):
        build_delayed_photon_solver_handoff(
            delayed_photon_path=photon,
            alara_result_receipt_path=receipt,
            alara_input_manifest_path=manifest,
        )


def test_bridge_rejects_old_ambiguous_photon_unit_contract(tmp_path):
    photon, receipt_path, manifest_path = _valid_files(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old = manifest["handoff"]["delayed_photon_output"]
    old.pop("alara_output_normalization")
    old.pop("zone_integrated_rate_operation")
    old["value_semantics"] = "bin_integrated_emission_rate"
    old["unit"] = "photons/s/bin/zone"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["input_manifest_sha256"] = _sha256(manifest_path)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(DelayedPhotonSourceError, match="input handoff"):
        build_delayed_photon_solver_handoff(
            delayed_photon_path=photon,
            alara_result_receipt_path=receipt_path,
            alara_input_manifest_path=manifest_path,
        )


def test_handoff_validation_rejects_tampering(tmp_path):
    photon, receipt, manifest = _valid_files(tmp_path)
    bundle = build_delayed_photon_solver_handoff(
        delayed_photon_path=photon,
        alara_result_receipt_path=receipt,
        alara_input_manifest_path=manifest,
    )
    changed = copy.deepcopy(bundle)
    changed["snapshots"][0]["zones"][0]["group_emission_rate_photons_per_s"][
        0
    ] += 1.0
    with pytest.raises(DelayedPhotonSourceError):
        validate_delayed_photon_solver_handoff(changed)


def test_result_binding_is_all_or_nothing(tmp_path):
    photon, _, manifest = _valid_files(tmp_path)
    output = tmp_path / "minimal.out"
    output.write_text(
        "Response Units: Bq /cm3\n"
        "*** Specific Activity [Bq/cm3] ***\n"
        "cu-63 1 0 1 1 1 1 1\n"
        "total 0 0 1 1 1 1 1\n"
        "*** Total Decay Heat [W/cm3] ***\n"
        "cu-63 1 0 1 1 1 1 1\n"
        "total 0 0 1 1 1 1 1\n"
        "*** Photon Source Distribution [gammas/s/cm3] : photons\n",
        encoding="utf-8",
    )
    stderr = tmp_path / "minimal.stderr"
    stderr.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must be complete"):
        validate_alara_result_files(
            output_path=output,
            stderr_path=stderr,
            delayed_photon_path=photon,
            input_manifest_sha256=_sha256(manifest),
            remote_run_root="/fresh/run",
            irradiation_checkpoint_s=86_400,
        )
