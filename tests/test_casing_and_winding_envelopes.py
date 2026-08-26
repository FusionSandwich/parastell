from types import SimpleNamespace

import pytest

from parastell.dagmc_envelope import component_union_boundary_surface_ids
from parastell.dagmc_envelope import MagnetPairRecord
from parastell.dagmc_envelope import MagnetVolumeRecord
from parastell.dagmc_envelope import magnet_pair_canonical_facet_ids
from parastell.magnet_radiation_field import MagnetRadiationFieldProducer
from parastell.magnet_radiation_field_bundle import _require_boundary_coverage


def _volume(volume_id, surface_ids):
    return SimpleNamespace(
        id=volume_id,
        surfaces=tuple(SimpleNamespace(id=value) for value in surface_ids),
    )


def test_component_union_removes_only_shared_material_interfaces():
    casing = _volume(41, (260, 261, 262, 263, 264, 265, 266))
    winding = _volume(42, (264, 265, 266, 267))
    assert component_union_boundary_surface_ids((casing, winding)) == (
        260,
        261,
        262,
        263,
        267,
    )


def test_component_union_requires_a_conformal_shared_interface():
    with pytest.raises(ValueError, match="no conformal shared interface"):
        component_union_boundary_surface_ids(
            (_volume(1, (1, 2)), _volume(2, (3, 4)))
        )


def _record(volume_id, role, facets):
    return MagnetVolumeRecord(
        volume_id=volume_id,
        material=role,
        component_role=role,
        surface_ids=(volume_id,),
        canonical_facet_ids=tuple(facets),
    )


def test_geometry_interchange_uses_real_pair_facet_ids_and_fails_if_absent():
    pair = MagnetPairRecord(
        "magnet-A",
        "coil-A",
        _record(8, "winding_pack", ("facet-w2", "facet-w1")),
        _record(7, "magnet_casing", ("facet-c1",)),
        "fixture",
    )
    assert magnet_pair_canonical_facet_ids(pair, require_casing=True) == (
        "facet-c1",
        "facet-w1",
        "facet-w2",
    )

    missing = MagnetPairRecord(
        "magnet-A",
        "coil-A",
        _record(8, "winding_pack", ()),
        _record(7, "magnet_casing", ("facet-c1",)),
        "fixture",
    )
    with pytest.raises(
        ValueError, match="canonical facet IDs are unavailable"
    ):
        magnet_pair_canonical_facet_ids(missing, require_casing=True)


def test_openmc_whole_component_tallies_include_pack_and_casing(monkeypatch):
    captured = {}

    def add_envelope_tallies(model, **kwargs):
        captured.update(kwargs)
        return "inventory"

    monkeypatch.setattr(
        "parastell.openmc16.configure_transport", lambda settings: None
    )
    monkeypatch.setattr(
        "parastell.openmc16.add_envelope_tallies", add_envelope_tallies
    )
    producer = object.__new__(MagnetRadiationFieldProducer)
    producer.envelopes = (
        SimpleNamespace(envelope=SimpleNamespace(surface_ids=(10, 11))),
        SimpleNamespace(envelope=SimpleNamespace(surface_ids=(11, 12))),
    )
    producer.selected_pairs = (
        SimpleNamespace(
            winding_pack=SimpleNamespace(volume_id=8),
            casing=SimpleNamespace(volume_id=7),
        ),
        SimpleNamespace(
            winding_pack=SimpleNamespace(volume_id=10),
            casing=SimpleNamespace(volume_id=9),
        ),
    )
    producer.selection = SimpleNamespace(
        local_mesh=True, tally_profile="activation_ready"
    )
    producer.tally_inventory = None
    result = producer.attach_openmc(
        SimpleNamespace(settings=object()),
        neutron_edges_eV=(0.0, 20.0e6),
        photon_edges_eV=(0.0, 1.0e9),
        local_mesh_filters_by_cell={8: "mesh-A"},
    )
    assert result == "inventory"
    assert captured["surface_ids"] == [10, 11, 12]
    assert captured["cell_ids"] == [8, 7, 10, 9]
    assert captured["local_mesh_filters_by_cell"] == {8: "mesh-A"}


def test_openmc_tallies_preserve_winding_only_geometry(monkeypatch):
    captured = {}

    def add_envelope_tallies(model, **kwargs):
        captured.update(kwargs)
        return "inventory"

    monkeypatch.setattr(
        "parastell.openmc16.configure_transport", lambda settings: None
    )
    monkeypatch.setattr(
        "parastell.openmc16.add_envelope_tallies", add_envelope_tallies
    )
    producer = object.__new__(MagnetRadiationFieldProducer)
    producer.envelopes = (
        SimpleNamespace(envelope=SimpleNamespace(surface_ids=(10, 11))),
    )
    producer.selected_pairs = (
        SimpleNamespace(
            winding_pack=SimpleNamespace(volume_id=8),
            casing=None,
        ),
    )
    producer.selection = SimpleNamespace(
        local_mesh=True, tally_profile="activation_ready"
    )
    producer.tally_inventory = None
    producer.attach_openmc(
        SimpleNamespace(settings=object()),
        neutron_edges_eV=(0.0, 20.0e6),
        photon_edges_eV=(0.0, 1.0e9),
        local_mesh_filters_by_cell={8: "mesh-A"},
    )
    assert captured["cell_ids"] == [8]


def test_default_envelope_roles_request_outer_union_and_pack(monkeypatch):
    calls = []

    def outer(**kwargs):
        calls.append(("outer_magnet", kwargs["volume_ids"]))
        return SimpleNamespace(envelope=SimpleNamespace(metadata={}))

    def winding(*args, **kwargs):
        calls.append(("winding_pack", (args[1],)))
        return SimpleNamespace(envelope=SimpleNamespace(metadata={}))

    monkeypatch.setattr(
        "parastell.magnet_radiation_field.extract_closed_component_union",
        outer,
    )
    monkeypatch.setattr(
        "parastell.magnet_radiation_field.extract_closed_envelope", winding
    )
    producer = object.__new__(MagnetRadiationFieldProducer)
    producer.inventory = object()
    producer.dagmc_path = "geometry.h5m"
    producer.selected_pairs = (
        SimpleNamespace(
            magnet_id="magnet-A",
            coil_id="coil-A",
            winding_pack=SimpleNamespace(
                volume_id=8, centroid_global_cm=(0.0, 0.0, 0.0)
            ),
            casing=SimpleNamespace(volume_id=7),
        ),
    )
    producer.centreline_frames = {}
    producer.canonical_geometry_policy = {
        "coordinate_quantum_cm": 1.0e-6,
        "faceting_tolerances": {},
    }
    producer.envelopes = ()
    result = producer.build_envelopes(
        frames_by_magnet={
            "magnet-A": {
                "plasma_direction_global": (1.0, 0.0, 0.0),
                "toroidal_direction_global": (0.0, 1.0, 0.0),
                "poloidal_direction_global": (0.0, 0.0, 1.0),
            }
        }
    )
    assert len(result) == 2
    assert calls == [("outer_magnet", (7, 8)), ("winding_pack", (8,))]
    assert result[0].envelope.metadata["boundary_role"] == "outer_magnet"
    assert result[0].envelope.metadata["construction_kind"] is None
    assert result[1].envelope.metadata["boundary_role"] == "winding_pack"


def test_bundle_role_routes_require_both_boundaries_per_magnet():
    products = [
        {
            "kind": "boundary_phase_space",
            "magnet_id": "magnet-A",
            "boundary_role": role,
        }
        for role in ("outer_magnet", "winding_pack")
    ]
    _require_boundary_coverage(products, ["magnet-A"])
    with pytest.raises(
        ValueError, match="cover outer_magnet and winding_pack"
    ):
        _require_boundary_coverage(products[:1], ["magnet-A"])
