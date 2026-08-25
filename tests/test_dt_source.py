from types import SimpleNamespace

import numpy as np
import pytest

from parastell.dt_source import source_convergence_observables


class _Spectrum:
    def __init__(self, temperature):
        self.mean_value = 14.0e6 + temperature
        self.std_dev = 1.0e5 + temperature / 10.0


def test_source_convergence_observables_use_strength_weighted_moments(
    monkeypatch,
):
    calls = []

    def spectrum(*, ion_temp, reactants):
        calls.append((ion_temp, reactants))
        return _Spectrum(ion_temp)

    monkeypatch.setitem(
        __import__("sys").modules,
        "openmc",
        SimpleNamespace(
            stats=SimpleNamespace(fusion_neutron_spectrum=spectrum)
        ),
    )
    source = SimpleNamespace(
        strengths=[1.0, 3.0],
        source_element_centroids_cm=[[0.0, 2.0, -1.0], [4.0, 6.0, 3.0]],
        ion_temperatures_eV=[1000.0, 1000.0],
    )

    result = source_convergence_observables(source)

    assert calls == [(1000.0, "DT")]
    assert result["spatial_moments"] == {
        "mean_x_cm": 3.0,
        "mean_y_cm": 5.0,
        "mean_z_cm": 2.0,
        "mean_x2_cm2": 12.0,
        "mean_y2_cm2": 28.0,
        "mean_z2_cm2": 7.0,
    }
    mean = 14.001e6
    standard_deviation = 100100.0
    assert result["dt_energy_moments"]["mean_eV"] == mean
    assert result["dt_energy_moments"]["mean_squared_eV2"] == pytest.approx(
        mean**2 + standard_deviation**2
    )


@pytest.mark.parametrize(
    "strengths,centroids,temperatures",
    [
        ([], [], []),
        ([0.0], [[0.0, 0.0, 0.0]], [1000.0]),
        ([-1.0], [[0.0, 0.0, 0.0]], [1000.0]),
        ([1.0], np.zeros((2, 3)), [1000.0]),
    ],
)
def test_source_convergence_observables_fail_closed_on_invalid_arrays(
    monkeypatch, strengths, centroids, temperatures
):
    monkeypatch.setitem(
        __import__("sys").modules,
        "openmc",
        SimpleNamespace(
            stats=SimpleNamespace(
                fusion_neutron_spectrum=lambda **kwargs: _Spectrum(1000.0)
            )
        ),
    )
    source = SimpleNamespace(
        strengths=strengths,
        source_element_centroids_cm=centroids,
        ion_temperatures_eV=temperatures,
    )
    with pytest.raises(ValueError):
        source_convergence_observables(source)
