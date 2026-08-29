from __future__ import annotations

import copy

import pytest

from parastell.activation_campaign import build_activation_campaign
from parastell.alara_activation import (
    ALARA_BINARY_SHA256,
    ALARA_PATCH_SHA256,
    build_alara_handoff,
    validate_alara_handoff,
    validate_alara_output_text,
    validate_alara_result_files,
    write_alara_package,
)
from parastell.energy_groups import (
    VITAMIN_J_175_EDGES_EV,
    VITAMIN_J_175_EDGES_SHA256,
)


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


def _row():
    return {
        "domain_id": "magnet-0000",
        "volume_cm3": 30.0,
        "energy_edges_eV": list(VITAMIN_J_175_EDGES_EV),
        "bin_integrated_flux_per_source": list(range(175)),
        "physical_source_rate_per_s": 2.0,
        "material": {
            "material_id": "copper-smoke",
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


def _handoff():
    return build_alara_handoff(
        activation_rows=[_row()],
        campaign=build_activation_campaign(),
        group_structure_sha256=VITAMIN_J_175_EDGES_SHA256,
        openmc_provenance=_provenance(),
    )


def test_alara_handoff_reverses_groups_and_applies_rate_once():
    handoff = _handoff()
    assert (
        handoff["zones"][0]["physical_flux_descending_per_cm2_s"][0] == 348.0
    )
    assert handoff["zones"][0]["physical_flux_descending_per_cm2_s"][-1] == 0.0
    assert handoff["normalization"]["surface_bank_used_as_flux"] is False
    assert handoff["runtime"]["binary_sha256"] == ALARA_BINARY_SHA256
    assert handoff["runtime"]["compiler_patch_sha256"] == ALARA_PATCH_SHA256


def test_alara_package_has_one_independent_deck_per_checkpoint(tmp_path):
    root = tmp_path / "alara"
    paths = write_alara_package(root, _handoff())
    assert len(paths) == 11
    branches = sorted(root.glob("irradiation_*s"))
    assert len(branches) == 5
    text = (branches[0] / "alara.inp").read_text(encoding="utf-8")
    assert "units Bq cm3" in text
    assert "element cu:63" in text
    assert "  10 y" in text
    assert "315576000 s" not in text
    cooling_lines = text.split("cooling", 1)[1].split("end", 1)[0].splitlines()
    assert "  0 s" not in cooling_lines


def test_alara_handoff_rejects_wrong_group_count_or_double_rate():
    row = _row()
    row["bin_integrated_flux_per_source"] = [1.0] * 174
    with pytest.raises(ValueError, match="176-edge"):
        build_alara_handoff(
            activation_rows=[row],
            campaign=build_activation_campaign(),
            group_structure_sha256=VITAMIN_J_175_EDGES_SHA256,
            openmc_provenance=_provenance(),
        )
    row = _row()
    row["physical_source_rate_per_s"] = 4.0
    with pytest.raises(ValueError, match="disagrees"):
        build_alara_handoff(
            activation_rows=[row],
            campaign=build_activation_campaign(),
            group_structure_sha256=VITAMIN_J_175_EDGES_SHA256,
            openmc_provenance=_provenance(),
        )


def test_alara_handoff_rejects_natural_element_substitution():
    row = _row()
    row["material"]["alara_constituents"][0]["element_symbol"] = "cu"
    with pytest.raises(ValueError, match="explicit isotope"):
        build_alara_handoff(
            activation_rows=[row],
            campaign=build_activation_campaign(),
            group_structure_sha256=VITAMIN_J_175_EDGES_SHA256,
            openmc_provenance=_provenance(),
        )


def test_alara_runtime_identity_is_immutable():
    handoff = copy.deepcopy(_handoff())
    handoff["runtime"]["binary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="qualified runtime"):
        validate_alara_handoff(handoff)


def test_alara_output_audit_requires_units_and_well_formed_isotopes():
    audit = validate_alara_output_text(
        "Response Units: Bq /cm3\n"
        "*** Specific Activity [Bq/cm3] ***\n"
        "isotope t_1/2(s) pre-irrad shutdown 1 s 60 s 1 h 1 d\n"
        "cu-63 1 0 1 1 1 1 1\n"
        "total 0 0 1 1 1 1 1\n"
        "*** Total Decay Heat [W/cm3] ***\n"
        "isotope t_1/2(s) pre-irrad shutdown 1 s 60 s 1 h 1 d\n"
        "cu-63 1 0 1 1 1 1 1\n"
        "total 0 0 1 1 1 1 1\n"
        "*** Photon Source Distribution\n"
    )
    assert audit["status"] == "PASS"
    with pytest.raises(ValueError, match="units"):
        validate_alara_output_text("cu-63 1 0 1")


def test_terminal_result_files_require_empty_stderr_and_delayed_photons(
    tmp_path,
):
    output = tmp_path / "alara.out"
    output.write_text(
        "Response Units: Bq /cm3\n"
        "*** Specific Activity [Bq/cm3] ***\n"
        "isotope t_1/2(s) pre-irrad shutdown 1 s 60 s 1 h 1 d\n"
        "cu-63 1 0 1 1 1 1 1\n"
        "total 0 0 1 1 1 1 1\n"
        "*** Total Decay Heat [W/cm3] ***\n"
        "isotope t_1/2(s) pre-irrad shutdown 1 s 60 s 1 h 1 d\n"
        "cu-63 1 0 1 1 1 1 1\n"
        "total 0 0 1 1 1 1 1\n"
        "*** Photon Source Distribution\n",
        encoding="utf-8",
    )
    stderr = tmp_path / "alara.stderr"
    stderr.write_text("", encoding="utf-8")
    photons = tmp_path / "photons"
    photons.write_text("positive", encoding="utf-8")
    receipt = validate_alara_result_files(
        output_path=output,
        stderr_path=stderr,
        delayed_photon_path=photons,
        input_manifest_sha256="a" * 64,
        remote_run_root="/fresh/run",
    )
    assert receipt["status"] == "PASSED"
    stderr.write_text("failure", encoding="utf-8")
    with pytest.raises(ValueError, match="stderr"):
        validate_alara_result_files(
            output_path=output,
            stderr_path=stderr,
            delayed_photon_path=photons,
            input_manifest_sha256="a" * 64,
            remote_run_root="/fresh/run",
        )
