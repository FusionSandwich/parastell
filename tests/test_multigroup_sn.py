import numpy as np
import pytest

from parastell.multigroup_sn import LayerTransport
from parastell.multigroup_sn import solve_multilayer_sn


def _incoming(groups, ordinates=8, value=0.0):
    return np.full((groups, ordinates // 2), value, dtype=float)


def test_unequal_neutron_and_photon_grids_transmit_in_vacuum():
    layer = LayerTransport(
        "vacuum",
        "vacuum",
        1.0,
        {"neutron": np.zeros(2), "photon": np.zeros(3)},
        {"neutron": np.zeros(2), "photon": np.zeros(3)},
        {"neutron": np.zeros((2, 2)), "photon": np.zeros((3, 3))},
    )
    result = solve_multilayer_sn(
        [layer],
        {"neutron": _incoming(2, value=1.0), "photon": _incoming(3)},
        {"neutron": [0.0, 1.0, 2.0], "photon": [0.0, 1.0, 2.0, 3.0]},
        ordinates=8,
    )
    assert result.group_counts == {"neutron": 2, "photon": 3}
    assert result.outgoing_current == pytest.approx(result.incoming_current)
    assert result.reflected_current == pytest.approx(0.0)
    assert result.interface_current.shape == (2, 2, 3)
    assert result.interface_current[0] == pytest.approx(
        result.interface_current[1]
    )
    assert result.particle_balance_error == pytest.approx(0.0)
    assert result.energy_balance_error == pytest.approx(0.0)


def test_downscatter_and_neutron_to_photon_sources_are_retained():
    neutron_scatter = np.zeros((2, 2))
    neutron_scatter[0, 1] = 0.05
    production = np.zeros((3, 2))
    production[1, 1] = 0.02
    layer = LayerTransport(
        "coupled",
        "verification",
        0.1,
        {"neutron": np.full(2, 0.1), "photon": np.full(3, 0.1)},
        {"neutron": np.full(2, 0.05), "photon": np.full(3, 0.1)},
        {"neutron": neutron_scatter, "photon": np.zeros((3, 3))},
        neutron_to_photon_xs_cm_1=production,
    )
    neutron = _incoming(2)
    neutron[1] = 1.0
    result = solve_multilayer_sn(
        [layer],
        {"neutron": neutron, "photon": _incoming(3)},
        {"neutron": [0.0, 1.0, 2.0], "photon": [0.0, 1.0, 2.0, 3.0]},
        ordinates=8,
    )
    assert result.scalar_flux[0, 0, 0] > 0.0
    assert result.scalar_flux[0, 1, 1] > 0.0
    assert result.neutron_to_photon_rate[0, 1] > 0.0
    assert np.all(np.isfinite(result.heating_eV_per_source))


def test_scattering_reflection_is_included_in_particle_balance():
    layer = LayerTransport(
        "scattering",
        "verification",
        1.0,
        {"neutron": np.array([1.0])},
        {"neutron": np.array([0.2])},
        {"neutron": np.array([[0.8]])},
    )
    result = solve_multilayer_sn(
        [layer],
        {"neutron": _incoming(1, value=1.0)},
        {"neutron": [1.0, 2.0]},
        ordinates=8,
        tolerance=1.0e-12,
    )

    assert result.reflected_current.sum() > 0.0
    assert result.particle_balance_error < 1.0e-9
    assert result.energy_balance_error < 1.0e-9
