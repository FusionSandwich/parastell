from types import SimpleNamespace

from parastell.openmc16_export import _directional_current_for_envelope


def test_native_directional_current_is_oriented_by_each_envelope_normal():
    surfaces = {
        10: SimpleNamespace(openmc_normal_sign=1),
        11: SimpleNamespace(openmc_normal_sign=-1),
    }
    inner = SimpleNamespace(
        surface_ids=[10, 11],
        surface=lambda surface_id: surfaces[surface_id],
    )
    envelope = SimpleNamespace(envelope=inner)
    native = {
        10: {-1: (1.0, 0.1), 1: (2.0, 0.2)},
        11: {-1: (3.0, 0.3), 1: (4.0, 0.4)},
    }

    result = _directional_current_for_envelope(native, envelope)

    assert result[10]["incoming"] == (1.0, 0.1)
    assert result[10]["outgoing"] == (2.0, 0.2)
    assert result[11]["incoming"] == (4.0, 0.4)
    assert result[11]["outgoing"] == (3.0, 0.3)
