import numpy as np
import pytest

from parastell.magnet_boundary_envelope import CorrelatedBoundaryBank
from parastell.magnet_boundary_envelope import EnvelopeSurface
from parastell.magnet_boundary_envelope import MagnetBoundaryEnvelope
from parastell.magnet_boundary_envelope import build_correlated_bank
from parastell.magnet_boundary_envelope import condition_on_independent_current
from parastell.magnet_boundary_envelope import classify_crossing_bank
from parastell.magnet_boundary_envelope import conservative_projection
from parastell.magnet_boundary_envelope import production_mu_edges
from parastell.magnet_boundary_envelope import production_phi_edges
from parastell.magnet_boundary_envelope import read_handoff
from parastell.magnet_boundary_envelope import write_handoff


def _surface(sid, role, centroid, normal, toroidal, poloidal, edges):
    return EnvelopeSurface(
        sid,
        role,
        4.0,
        centroid,
        normal,
        toroidal,
        poloidal,
        (-1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        topology_edge_ids=edges,
    )


def box_envelope():
    surfaces = (
        _surface(
            1,
            "plasma_facing",
            (1, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            ("a", "b", "c", "d"),
        ),
        _surface(
            2,
            "outer",
            (-1, 0, 0),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, 1),
            ("e", "f", "g", "h"),
        ),
        _surface(
            3,
            "toroidal_side",
            (0, 1, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (0, 0, 1),
            ("a", "e", "i", "j"),
        ),
        _surface(
            4,
            "toroidal_side",
            (0, -1, 0),
            (0, -1, 0),
            (1, 0, 0),
            (0, 0, 1),
            ("b", "f", "k", "l"),
        ),
        _surface(
            5,
            "poloidal_side",
            (0, 0, 1),
            (0, 0, 1),
            (1, 0, 0),
            (0, 1, 0),
            ("c", "g", "i", "k"),
        ),
        _surface(
            6,
            "poloidal_side",
            (0, 0, -1),
            (0, 0, -1),
            (-1, 0, 0),
            (0, 1, 0),
            ("d", "h", "j", "l"),
        ),
    )
    return MagnetBoundaryEnvelope(
        "coil-0", "winding_pack", 88, surfaces, "0" * 64
    )


def test_closed_envelope_and_two_sided_classification():
    envelope = box_envelope()
    assert envelope.classify(1, (-1, 0, 0))[1] == "incoming"
    assert envelope.classify(1, (1, 0, 0))[1] == "outgoing"
    assert envelope.classify(1, (0, 1, 0))[1] == "grazing"


def test_open_and_duplicate_envelopes_fail():
    envelope = box_envelope()
    with pytest.raises(ValueError, match="duplicate"):
        MagnetBoundaryEnvelope(
            "x", "m", 1, envelope.surfaces + (envelope.surfaces[0],), "0" * 64
        )
    bad = list(envelope.surfaces)
    bad[0] = EnvelopeSurface(
        **{**bad[0].to_dict(), "topology_edge_ids": ("missing",)}
    )
    with pytest.raises(ValueError, match="non-manifold"):
        MagnetBoundaryEnvelope("x", "m", 1, tuple(bad), "0" * 64)


def test_production_angle_resolution_has_grazing_bins():
    mu = production_mu_edges()
    assert len(mu) - 1 >= 20
    assert np.sum((mu[:-1] >= -0.1) & (mu[1:] <= 0.1)) >= 6
    assert len(production_phi_edges()) - 1 >= 16


def test_correlated_condition_projection_and_round_trip(tmp_path):
    envelope = box_envelope()
    bank = build_correlated_bank(
        envelope,
        position_global_cm=np.array(
            [[1, 0.25, 0.25], [1, -0.25, -0.25], [1, 0.5, 0.5]]
        ),
        direction_global=np.array(
            [[-1, 0, 0], [-1, 1, 0], [1, 0, 0]], dtype=float
        ),
        energy_eV=[14.1e6, 1.0e6, 14.1e6],
        raw_weight=[1.0, 3.0, 2.0],
        particle=["neutron", "photon", "neutron"],
        surface_id=[1, 1, 1],
        energy_edges_eV=[0.0, 2.0e6, 20.0e6],
    )
    rows = [
        {
            "surface_id": 1,
            "particle": "neutron",
            "energy_group": 1,
            "crossing_sense": "incoming",
            "mean": 0.4,
            "std_dev": 0.04,
        },
        {
            "surface_id": 1,
            "particle": "photon",
            "energy_group": 0,
            "crossing_sense": "incoming",
            "mean": 0.2,
            "std_dev": 0.02,
        },
        {
            "surface_id": 1,
            "particle": "neutron",
            "energy_group": 1,
            "crossing_sense": "outgoing",
            "mean": 0.1,
            "std_dev": 0.01,
        },
    ]
    conditioned = condition_on_independent_current(bank, rows)
    assert conditioned.integrated_current == pytest.approx(0.7)
    projection = conservative_projection(conditioned)
    assert projection["mean"].sum() == pytest.approx(0.7)
    assert projection["mean"].shape == (
        len(envelope.surface_ids),
        4,
        2,
        (len(production_mu_edges()) - 1) * (len(production_phi_edges()) - 1),
        2,
        3,
    )
    assert projection["surface_ids"].tolist() == list(envelope.surface_ids)
    assert projection["surface_patch_counts"].tolist() == [4] * 6
    assert np.count_nonzero(projection["mean"][1:]) == 0
    output = tmp_path / "handoff.h5"
    write_handoff(
        output,
        envelope,
        conditioned,
        provenance={"openmc_version": "test", "parastell_commit": "test"},
        normalization={
            "basis": "per_source",
            "particles_per_source_history": 1.0,
            "area_basis": "surface area cm2",
            "energy_bin_width": "explicit edges in eV",
            "solid_angle_measure": "explicit mu/phi bin solid angle sr",
            "quantity": "partial current",
            "time_basis": "per source history",
        },
    )
    manifest, restored_envelope, restored = read_handoff(output)
    assert manifest["schema_version"] == "2.1.0"
    assert manifest["canonical_bank"] is False
    assert manifest["field_availability"]["history_id"]["available"] is False
    assert restored_envelope.surface_ids == envelope.surface_ids
    assert restored.integrated_current == pytest.approx(0.7)


def test_positive_tally_without_records_fails():
    envelope = box_envelope()
    bank = build_correlated_bank(
        envelope,
        position_global_cm=np.array([[1, 0, 0]]),
        direction_global=np.array([[-1, 0, 0]]),
        energy_eV=[1.0],
        raw_weight=[1.0],
        particle=["neutron"],
        surface_id=[1],
        energy_edges_eV=[0.0, 2.0],
    )
    with pytest.raises(ValueError, match="no bank records"):
        condition_on_independent_current(
            bank,
            [
                {
                    "surface_id": 2,
                    "particle": "neutron",
                    "energy_group": 0,
                    "crossing_sense": "incoming",
                    "mean": 1.0,
                }
            ],
        )


def test_missing_bank_field_rejected():
    with pytest.raises(ValueError, match="missing fields"):
        CorrelatedBoundaryBank({"record_id": np.array([0])})


def test_optional_unavailable_fields_are_not_fabricated():
    bank = build_correlated_bank(
        box_envelope(),
        position_global_cm=np.array([[1.0, 0.0, 0.0]]),
        direction_global=np.array([[-1.0, 0.0, 0.0]]),
        energy_eV=[1.0],
        raw_weight=[0.25],
        particle=["neutron"],
        surface_id=[1],
        energy_edges_eV=[0.0, 2.0],
    )
    assert "history_id" not in bank.columns
    assert "weight_std_dev" not in bank.columns
    assert "time_s" not in bank.columns
    assert bank.metadata["field_availability"]["history_id"] == {
        "available": False,
        "origin": "unavailable",
    }


def test_surface_bank_completeness_is_fail_closed():
    complete = classify_crossing_bank(
        stored_record_count=10,
        selected_record_count=4,
        max_particles_per_file=100,
        max_source_files=1,
        source_file_count=1,
    )
    sampled = classify_crossing_bank(
        stored_record_count=10,
        selected_record_count=4,
        max_particles_per_file=None,
        max_source_files=None,
        source_file_count=1,
    )
    truncated = classify_crossing_bank(
        stored_record_count=100,
        selected_record_count=4,
        max_particles_per_file=100,
        max_source_files=1,
        source_file_count=1,
    )
    assert complete["classification"] == "COMPLETE_CROSSING_BANK"
    assert sampled["classification"] == "SAMPLED_CROSSING_BANK"
    assert truncated["classification"] == "TRUNCATED_INVALID_BANK"
