import json

import h5py
import numpy as np
import pytest

from parastell.magnet_boundary_envelope import (
    EnvelopeSurface,
    MagnetBoundaryEnvelope,
    build_correlated_bank,
    classify_crossing_bank,
    write_handoff,
)
from parastell.photon_calibration import (
    garwood_poisson_interval,
    qualify_photon_calibration,
    summarize_boundary_handoffs,
    validate_photon_calibration,
    write_photon_calibration,
)


def _surface(sid, centroid, normal, toroidal, poloidal, edges):
    return EnvelopeSurface(
        sid,
        "envelope",
        4.0,
        centroid,
        normal,
        toroidal,
        poloidal,
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        topology_edge_ids=edges,
    )


def _box_envelope():
    surfaces = (
        _surface(
            1,
            (1, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            ("a", "b", "c", "d"),
        ),
        _surface(
            2,
            (-1, 0, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, 1),
            ("e", "f", "g", "h"),
        ),
        _surface(
            3,
            (0, 1, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (0, 0, 1),
            ("a", "e", "i", "j"),
        ),
        _surface(
            4,
            (0, -1, 0),
            (0, -1, 0),
            (1, 0, 0),
            (0, 0, 1),
            ("b", "f", "k", "l"),
        ),
        _surface(
            5,
            (0, 0, 1),
            (0, 0, 1),
            (1, 0, 0),
            (0, 1, 0),
            ("c", "g", "i", "k"),
        ),
        _surface(
            6,
            (0, 0, -1),
            (0, 0, -1),
            (-1, 0, 0),
            (0, 1, 0),
            ("d", "h", "j", "l"),
        ),
    )
    return MagnetBoundaryEnvelope(
        "winding-pack-coil-0005",
        "example-stellarator-sector-00-90deg-coil-0005",
        18,
        surfaces,
        "a" * 64,
    )


def _handoff(tmp_path, name, photon_count, *, histories=100_000):
    envelope = _box_envelope()
    particles = ["neutron", "neutron"] + ["photon"] * photon_count
    directions = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]] + [
        [-0.05, 1.0, 0.0]
    ] * photon_count
    count = len(particles)
    bank = build_correlated_bank(
        envelope,
        position_global_cm=np.asarray([[1.0, 0.0, 0.0]] * count),
        direction_global=np.asarray(directions),
        energy_eV=[14.1e6, 13.0e6] + [1.0e6] * photon_count,
        raw_weight=np.asarray([1.0] * count) / histories,
        particle=particles,
        surface_id=[1] * count,
        energy_edges_by_particle={
            "neutron": [0.0, 20.0e6],
            "photon": [0.0, 20.0e6],
        },
        facet_mapping={
            "facet_id": [f"facet-{index}" for index in range(count)],
            "facet_index": list(range(count)),
            "barycentric_coordinates": [[0.2, 0.3, 0.5]] * count,
            "distance_to_facet_residual_cm": [0.0] * count,
            "facet_mapping_status": ["CONTAINING_FACET"] * count,
        },
    )
    bank.metadata["surface_bank_completeness"] = classify_crossing_bank(
        stored_record_count=count,
        selected_record_count=count,
        max_particles_per_file=1000,
        max_source_files=1,
        source_file_count=1,
        mpi_ranks=1,
    )
    output = tmp_path / f"{name}.h5"
    write_handoff(
        output,
        envelope,
        bank,
        provenance={
            "histories": histories,
            "openmc_version": "0.16.0",
            "parastell_commit": "b" * 40,
            "dagmc_geometry_sha256": "a" * 64,
            "source_definition_sha256": "c" * 64,
        },
        normalization={
            "basis": "per source history",
            "particles_per_source_history": 1.0,
            "area_basis": "surface integrated",
            "energy_bin_width": "group integrated",
            "solid_angle_measure": "angle integrated",
            "quantity": "partial crossing current",
            "time_basis": "none",
            "physical_source_rate_per_s": 2.4e20,
        },
    )
    return output


def test_exact_garwood_interval_and_particle_sense_patch_aggregation(tmp_path):
    path = _handoff(tmp_path, "replica", 3)
    result = summarize_boundary_handoffs([path], histories=100_000)

    photon_entry = next(
        row
        for row in result["by_particle_and_crossing_sense"]
        if row["particle"] == "photon" and row["crossing_sense"] == "incoming"
    )
    assert photon_entry["boundary_motion"] == "entry"
    assert photon_entry["raw_count"] == 3
    assert photon_entry["summed_weight_per_source"] == pytest.approx(3.0e-5)
    assert photon_entry["effective_sample_size"] == pytest.approx(3.0)
    expected = garwood_poisson_interval(3)
    observed = photon_entry["garwood_count_mean_interval"]
    assert (observed["lower"], observed["upper"]) == pytest.approx(expected)

    photon_exit = next(
        row
        for row in result["by_particle_and_crossing_sense"]
        if row["particle"] == "photon" and row["crossing_sense"] == "outgoing"
    )
    assert photon_exit["raw_count"] == 0
    assert photon_exit["garwood_count_mean_interval"]["upper"] == (
        pytest.approx(3.6888794541139354)
    )
    detail = result["by_magnet_surface_patch_particle_sense_and_grazing"]
    assert any(
        row["particle"] == "photon"
        and row["surface_id"] == 1
        and row["patch_id"] >= 0
        and row["angular_grazing"]
        for row in detail
    )


def test_three_complete_replicas_produce_a_finite_photon_planning_rule(
    tmp_path,
):
    replicas = [
        {
            "seed": seed,
            "histories": 100_000,
            "handoff_paths": [_handoff(tmp_path, f"seed-{seed}", count)],
        }
        for seed, count in zip((11, 13, 17), (2, 3, 4))
    ]
    result = qualify_photon_calibration(replicas)
    assert result["status"] == "PARASTELL_PHOTON_CALIBRATION_PASS"
    assert result["pooled_incoming_photon"]["raw_count"] == 9
    assert result["pooled_incoming_photon"][
        "effective_sample_size"
    ] == pytest.approx(9.0)
    assert result["planning_rule"]["finite"]
    assert (
        result["planning_rule"]["required_histories_point_estimate"]
        == 33_333_334
    )
    assert (
        result["planning_rule"]["required_histories_confidence_conservative"]
        > 33_333_334
    )

    output = tmp_path / "PHOTON_CALIBRATION.json"
    written = write_photon_calibration(output, replicas)
    validated = validate_photon_calibration(output, verify_bound_files=True)
    assert validated["evidence_sha256"] == written["evidence_sha256"]
    assert validated["finite_planning_rule"]
    assert validated["verified_bound_bank_count"] == 3
    value = json.loads(output.read_text(encoding="utf-8"))
    value["pooled_incoming_photon"]["raw_count"] += 1
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash"):
        validate_photon_calibration(output)


def test_sampled_or_cap_reached_handoff_cannot_qualify(tmp_path):
    path = _handoff(tmp_path, "incomplete", 1)
    with h5py.File(path, "r+") as handle:
        manifest = json.loads(handle["manifest_json"].asstr()[()])
        completeness = manifest["bank_metadata"]["surface_bank_completeness"]
        completeness["sampling_applied"] = True
        completeness["classification"] = "SAMPLED_CROSSING_BANK"
        handle["manifest_json"][()] = json.dumps(manifest, sort_keys=True)
    with pytest.raises(ValueError, match="complete, uncapped, unsampled"):
        summarize_boundary_handoffs([path], histories=100_000)


def test_three_distinct_replicas_are_required(tmp_path):
    path = _handoff(tmp_path, "one", 1)
    replica = {"seed": 11, "histories": 100_000, "handoff_paths": [path]}
    with pytest.raises(ValueError, match="insufficient"):
        qualify_photon_calibration([replica, {**replica, "seed": 13}])
    with pytest.raises(ValueError, match="distinct"):
        qualify_photon_calibration([replica, replica, replica])
