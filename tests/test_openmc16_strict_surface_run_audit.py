import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from parastell.dagmc_envelope import DagmcEnvelope, FacetedSurface
from parastell.magnet_boundary_envelope import (
    EnvelopeSurface,
    MagnetBoundaryEnvelope,
)
from parastell import openmc16_export
from parastell.openmc16_export import (
    StrictSurfaceRunArtifacts,
    _capacity_proof,
    audit_openmc16_surface_run,
)


VECTOR = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
SOURCE_DTYPE = np.dtype(
    [
        ("r", VECTOR),
        ("u", VECTOR),
        ("E", "<f8"),
        ("time", "<f8"),
        ("wgt", "<f8"),
        ("delayed_group", "<i4"),
        ("surf_id", "<i4"),
        ("particle", "<i4"),
    ]
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _envelope(dagmc_path: Path) -> DagmcEnvelope:
    triangle = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    faceted = FacetedSurface(
        17,
        "outer",
        triangle,
        np.asarray([[0.0, 0.0, 1.0]]),
        np.asarray([0.5]),
        np.asarray([1.0 / 3.0, 1.0 / 3.0, 0.0]),
        ("facet-exact",),
        8,
    )
    surface = EnvelopeSurface(
        17,
        "outer",
        0.5,
        (1.0 / 3.0, 1.0 / 3.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        openmc_normal_sign=1,
        vector_area_global_cm2=(0.0, 0.0, 0.0),
    )
    envelope = MagnetBoundaryEnvelope(
        "magnet-8-envelope",
        "magnet-8",
        8,
        (surface,),
        _sha(dagmc_path),
        metadata={"canonical_geometry_fingerprint": "c" * 64},
    )
    return DagmcEnvelope(envelope, (faceted,), 3, 0, 0.0)


def _write_model(path: Path, *, max_particles=10, max_files=None):
    max_files_xml = (
        ""
        if max_files is None
        else f"<max_source_files>{max_files}</max_source_files>"
    )
    path.write_text(
        "<model><geometry><dagmc_universe id='1' filename='fake.h5m'/>"
        "</geometry><settings><run_mode>fixed source</run_mode>"
        "<particles>1</particles><batches>1</batches><seed>17</seed>"
        "<surf_source_write><surface_ids>17</surface_ids>"
        f"<max_particles>{max_particles}</max_particles>{max_files_xml}"
        "</surf_source_write></settings><tallies>"
        "<filter id='1' type='surface'><bins>17</bins></filter>"
        "<filter id='2' type='musurface'><bins>-1 0 1</bins></filter>"
        "<filter id='3' type='particle'><bins>neutron</bins></filter>"
        "<filter id='4' type='energy'><bins>0 20000000</bins></filter>"
        "<tally id='1' name='pstl_envelope_neutron_directional_current'>"
        "<filters>1 2 3 4</filters><scores>current</scores></tally>"
        "</tallies></model>"
    )


def _write_bank(path: Path):
    record = np.zeros(1, dtype=SOURCE_DTYPE)
    record["r"]["x"] = 0.25
    record["r"]["y"] = 0.25
    record["u"]["z"] = 1.0
    record["E"] = 14.1e6
    record["wgt"] = 1.0
    record["surf_id"] = 17
    record["particle"] = 2112
    with h5py.File(path, "x") as target:
        target.attrs["filetype"] = np.bytes_(b"source")
        target.attrs["version"] = np.asarray([18, 2], dtype=int)
        target.create_dataset("source_bank", data=record)


def _filter(parent, identifier, kind, bins):
    group = parent.create_group(f"filter {identifier}")
    group.create_dataset("type", data=np.bytes_(kind.encode()))
    group.create_dataset("bins", data=bins)
    group.create_dataset(
        "n_bins",
        data=len(bins) - 1 if kind in {"musurface", "energy"} else len(bins),
    )


def _write_statepoint(path: Path):
    with h5py.File(path, "x") as target:
        target.attrs["filetype"] = np.bytes_(b"statepoint")
        target.attrs["openmc_version"] = np.asarray([0, 16, 0])
        target.create_dataset("run_mode", data=np.bytes_(b"fixed source"))
        target.create_dataset("n_particles", data=1)
        target.create_dataset("n_batches", data=1)
        target.create_dataset("current_batch", data=1)
        target.create_dataset("seed", data=17)
        tallies = target.create_group("tallies")
        filters = tallies.create_group("filters")
        _filter(filters, 1, "surface", np.asarray([17]))
        _filter(filters, 2, "musurface", np.asarray([-1.0, 0.0, 1.0]))
        _filter(filters, 3, "particle", np.asarray([np.bytes_(b"neutron")]))
        _filter(filters, 4, "energy", np.asarray([0.0, 20.0e6]))
        tally = tallies.create_group("tally 1")
        tally.create_dataset(
            "name",
            data=np.bytes_(b"pstl_envelope_neutron_directional_current"),
        )
        tally.create_dataset("filters", data=np.asarray([1, 2, 3, 4]))
        tally.create_dataset(
            "score_bins", data=np.asarray([np.bytes_(b"current")])
        )
        tally.create_dataset(
            "nuclides", data=np.asarray([np.bytes_(b"total")])
        )
        tally.create_dataset("n_realizations", data=1)
        results = np.zeros((2, 1, 2))
        results[1, 0, 0] = 1.0
        results[1, 0, 1] = 1.0
        tally.create_dataset("results", data=results)


def _fixture(tmp_path, *, mpi_ranks=2, max_particles=10):
    dagmc = tmp_path / "fake.h5m"
    dagmc.write_bytes(b"unit-test-h5m-placeholder")
    model = tmp_path / "model.xml"
    _write_model(model, max_particles=max_particles)
    statepoint = tmp_path / "statepoint.1.h5"
    _write_statepoint(statepoint)
    bank = tmp_path / "surface_source.h5"
    _write_bank(bank)
    mpi_line = f"MPI Processes | {mpi_ranks}\n" if mpi_ranks != 1 else ""
    log = tmp_path / "openmc.log"
    log.write_text(
        "Version | 0.16.0\n"
        + mpi_line
        + "Loading file fake.h5m\n"
        + "FIXED SOURCE TRANSPORT SIMULATION\n"
        + "Creating state point statepoint.1.h5...\n"
        + "Creating source file surface_source.h5 with 1 particles ...\n"
        + "RESULTS\n"
    )
    artifacts = StrictSurfaceRunArtifacts(
        dagmc, model, statepoint, log, [bank]
    )
    request = {
        "volume_id": 8,
        "envelope_id": "magnet-8-envelope",
        "magnet_id": "magnet-8",
        "plasma_direction_global": (1.0, 0.0, 0.0),
        "toroidal_direction_global": (0.0, 1.0, 0.0),
        "poloidal_direction_global": (0.0, 0.0, 1.0),
    }
    return artifacts, request, _envelope(dagmc)


def test_strict_audit_parses_artifacts_and_binds_localization(
    monkeypatch, tmp_path
):
    artifacts, request, envelope = _fixture(tmp_path, mpi_ranks=2)
    monkeypatch.setattr(
        openmc16_export,
        "_extract_envelopes_from_h5m",
        lambda path, requests: (envelope,),
    )
    result = audit_openmc16_surface_run(
        artifacts, envelope_requests=[request], required_particles=["neutron"]
    )
    assert result["classification"] == "COMPLETE_CROSSING_BANK"
    assert result["model"]["source_histories"] == 1
    assert result["model"]["max_source_files"] == 1
    assert result["model"]["max_source_files_source"] == "openmc_0.16_default"
    assert result["capacity"]["mpi_ranks_from_log"] == 2
    binding = result["localization_topology_binding"]
    assert binding["dagmc_path"] == str(Path(artifacts.dagmc_path).resolve())
    assert binding["dagmc_sha256"] == _sha(Path(artifacts.dagmc_path))
    assert len(binding["envelopes"][0]["canonical_facet_catalog_sha256"]) == 64
    assert (
        result["same_run_integrity"]["localization_topology_manifest_sha256"]
        == binding["manifest_sha256"]
    )


def test_mpi_capacity_is_proved_below_cap_and_fails_at_cap():
    below = _capacity_proof(
        [{"record_count": 9}],
        max_particles_per_process=10,
        max_source_files=1,
        mpi_ranks=4,
    )
    at_cap = _capacity_proof(
        [{"record_count": 10}],
        max_particles_per_process=10,
        max_source_files=1,
        mpi_ranks=4,
    )
    assert below["capacity_proven_not_reached"]
    assert not at_cap["capacity_proven_not_reached"]


def test_arbitrary_bank_bytes_cannot_produce_complete(monkeypatch, tmp_path):
    artifacts, request, envelope = _fixture(tmp_path)
    Path(artifacts.surface_source_paths[0]).write_bytes(b"not an HDF5 bank")
    monkeypatch.setattr(
        openmc16_export,
        "_extract_envelopes_from_h5m",
        lambda path, requests: (envelope,),
    )
    with pytest.raises(OSError):
        audit_openmc16_surface_run(
            artifacts,
            envelope_requests=[request],
            required_particles=["neutron"],
        )


def test_caller_assertion_accounting_is_disabled():
    from parastell.surface_source_accounting import (
        validate_surface_source_accounting,
    )

    with pytest.raises(RuntimeError, match="caller-asserted"):
        validate_surface_source_accounting(
            {"classification": "COMPLETE_CROSSING_BANK"}
        )
