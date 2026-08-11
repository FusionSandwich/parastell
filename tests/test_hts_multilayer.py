from pathlib import Path
import json

import h5py
import numpy as np
import pytest

from parastell.hts_multilayer import analytic_transmission
from parastell.hts_multilayer import constant_response_library
from parastell.hts_multilayer import HTSLayer
from parastell.hts_multilayer import MultilayerStack
from parastell.hts_multilayer import replay_phase_space
from parastell.hts_multilayer import verification_rebco_stack
from parastell.hts_multilayer import zero_response_library


def _write_phase_space(path: Path, *, mu=(-1.0, -1.0)) -> None:
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as output:
        output.attrs["region"] = "test_region"
        output.attrs["record_count"] = 2
        phase = output.create_group("phase_space")
        phase.create_dataset(
            "position_local_cm",
            data=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        )
        phase.create_dataset(
            "direction_global",
            data=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        phase.create_dataset(
            "direction_local",
            data=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        )
        phase.create_dataset("energy_eV", data=[14.1e6, 14.1e6])
        phase.create_dataset("weight", data=[1.0, 3.0])
        phase.create_dataset("surface_id_abs", data=[101, 101])
        phase.create_dataset("mu_outward", data=mu)
        phase.create_dataset("record_id", data=[0, 1])
        phase.create_dataset(
            "particle_name",
            data=np.asarray(["neutron", "neutron"], dtype=object),
            dtype=string_dtype,
        )
        phase.create_dataset(
            "magnet_direction",
            data=np.asarray(["incoming", "incoming"], dtype=object),
            dtype=string_dtype,
        )
        output.create_dataset(
            "metadata_json",
            data=json.dumps({"test": True}),
            dtype=string_dtype,
        )


def _write_spectra(path: Path, current=1.0) -> None:
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as output:
        tallies = output.create_group("tallies")
        tally = tallies.create_group("boundary")
        tally.attrs["role"] = "boundary_current"
        tally.attrs["region"] = "test_region"
        tally.create_dataset(
            "particle",
            data=np.asarray(["neutron", "neutron"], dtype=object),
            dtype=string_dtype,
        )
        tally.create_dataset("surface", data=[101, 101])
        tally.create_dataset("energy_low_ev", data=[0.0, 0.0])
        tally.create_dataset("energy_high_ev", data=[20.0e6, 20.0e6])
        tally.create_dataset(
            "magnet_direction",
            data=np.asarray(["incoming", "outgoing"], dtype=object),
            dtype=string_dtype,
        )
        tally.create_dataset("mean", data=[current, 0.0])
        tally.create_dataset("std_dev", data=[0.0, 0.0])


def test_verification_stack_is_explicit_and_ordered():
    stack = verification_rebco_stack()

    assert [layer.name for layer in stack.layers] == [
        "reactor_side_copper",
        "silver_cap",
        "rebco",
        "buffer_stack",
        "hastelloy_substrate",
        "backside_copper",
        "solder",
        "insulation",
    ]
    assert stack.total_thickness_cm > 0.0
    assert stack.metadata["manufacturer_specific"] is False


def test_rejects_nonpositive_layer_thickness():
    with pytest.raises(ValueError, match="thickness_cm"):
        HTSLayer("bad", "material", 0.0)


def test_zero_response_replay_closes_current(tmp_path):
    phase = tmp_path / "phase.h5"
    spectra = tmp_path / "spectra.h5"
    output = tmp_path / "replay.h5"
    _write_phase_space(phase)
    _write_spectra(spectra, current=2.0)
    stack = verification_rebco_stack()
    library = zero_response_library(stack, [0.0, 20.0e6], ["neutron"])

    summary = replay_phase_space(
        phase,
        spectra,
        output,
        stack=stack,
        response_library=library,
    )

    assert summary.input_current_per_source == pytest.approx(2.0)
    assert summary.transmitted_current_per_source == pytest.approx(2.0)
    assert summary.removed_current_per_source == pytest.approx(0.0)
    assert summary.relative_balance_error <= 1.0e-15
    with h5py.File(output, "r") as replay:
        response = replay["response"]["incident_current_per_source"]
        assert response.shape[2] == len(stack.layers)
        assert replay["records"]["normalised_current_per_source"][
            :
        ].sum() == pytest.approx(2.0)


def test_constant_response_matches_exact_characteristic(tmp_path):
    phase = tmp_path / "phase.h5"
    spectra = tmp_path / "spectra.h5"
    output = tmp_path / "replay.h5"
    _write_phase_space(phase)
    _write_spectra(spectra)
    stack = MultilayerStack(
        "one_layer",
        (HTSLayer("layer", "test", 0.2),),
    )
    library = constant_response_library(
        stack,
        [0.0, 20.0e6],
        ["neutron"],
        3.0,
    )

    summary = replay_phase_space(
        phase,
        spectra,
        output,
        stack=stack,
        response_library=library,
    )
    expected = analytic_transmission(1.0, 3.0, 0.2, 1.0)

    assert summary.transmitted_current_per_source == pytest.approx(expected)
    assert summary.relative_balance_error <= 1.0e-15


def test_per_record_normals_resolve_incident_cosine(tmp_path):
    phase = tmp_path / "phase.h5"
    spectra = tmp_path / "spectra.h5"
    output = tmp_path / "replay.h5"
    _write_phase_space(phase, mu=(np.nan, np.nan))
    _write_spectra(spectra)
    stack = MultilayerStack(
        "one_layer",
        (HTSLayer("layer", "test", 0.1),),
    )
    library = constant_response_library(
        stack,
        [0.0, 20.0e6],
        ["neutron"],
        2.0,
    )
    normals = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    summary = replay_phase_space(
        phase,
        spectra,
        output,
        stack=stack,
        response_library=library,
        per_record_inward_normals_global=normals,
    )

    assert summary.direction_basis == "per_record_inward_normal_global"
    assert summary.transmitted_current_per_source == pytest.approx(
        np.exp(-0.2)
    )


def test_nonzero_tally_bin_without_bank_records_is_rejected(tmp_path):
    phase = tmp_path / "phase.h5"
    spectra = tmp_path / "spectra.h5"
    _write_phase_space(phase)
    _write_spectra(spectra)
    with h5py.File(spectra, "r+") as source:
        source["tallies"]["boundary"]["surface"][:] = [999, 999]
    stack = verification_rebco_stack()
    library = zero_response_library(stack, [0.0, 20.0e6], ["neutron"])

    with pytest.raises(RuntimeError, match="no sampled source records"):
        replay_phase_space(
            phase,
            spectra,
            tmp_path / "replay.h5",
            stack=stack,
            response_library=library,
        )
