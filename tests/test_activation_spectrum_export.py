import json

import pytest

from parastell.activation.spectrum_export import ActivationSpectrum
from parastell.activation.spectrum_export import load_activation_spectrum


def _spectrum():
    return ActivationSpectrum.create(
        "test",
        "neutron",
        [1.0, 2.0, 4.0, 8.0],
        [1.0, 2.0, 3.0],
        "magnet-1",
        2.0e20,
    )


def test_activation_spectrum_round_trip_and_normalization(tmp_path):
    path = _spectrum().write_json(tmp_path / "spectrum.json")
    restored = load_activation_spectrum(path)
    assert restored.total_flux_cm2_s == 6.0
    assert restored.content_sha256 == _spectrum().content_sha256


def test_fispact_arbitrary_flux_is_descending_and_conservative(tmp_path):
    path = _spectrum().write_fispact_arb_flux(tmp_path / "arb_flux")
    lines = path.read_text(encoding="ascii").splitlines()
    assert [float(value) for value in lines[0].split()] == [8.0, 4.0, 2.0, 1.0]
    assert sum(float(value) for value in lines[1].split()) == pytest.approx(
        6.0
    )


def test_tampered_spectrum_is_rejected(tmp_path):
    path = _spectrum().write_json(tmp_path / "spectrum.json")
    data = json.loads(path.read_text(encoding="ascii"))
    data["group_flux_cm2_s"][0] = 99.0
    path.write_text(json.dumps(data), encoding="ascii")
    with pytest.raises(ValueError, match="checksum"):
        load_activation_spectrum(path)
