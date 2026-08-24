"""Small solver-neutral 1-D multigroup neutron/photon reference solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LayerTransport:
    name: str
    material: str
    thickness_cm: float
    total_xs_cm_1: Mapping[str, np.ndarray]
    absorption_xs_cm_1: Mapping[str, np.ndarray]
    scattering_xs_cm_1: Mapping[str, np.ndarray]
    neutron_to_photon_xs_cm_1: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.thickness_cm <= 0.0:
            raise ValueError("layer thickness must be positive")


@dataclass(frozen=True)
class SNResult:
    particles: tuple[str, ...]
    group_counts: Mapping[str, int]
    scalar_flux: np.ndarray
    incoming_current: np.ndarray
    outgoing_current: np.ndarray
    reflected_current: np.ndarray
    interface_current: np.ndarray
    absorption_rate: np.ndarray
    neutron_to_photon_rate: np.ndarray
    heating_eV_per_source: np.ndarray
    particle_balance_error: float
    energy_balance_error: float
    iterations: int


def project_incoming_bank_to_ordinates(
    bank,
    particle: str,
    energy_edges_eV: Sequence[float],
    *,
    ordinates: int = 16,
) -> np.ndarray:
    """Conservatively map incoming correlated records to positive slab ordinates.

    The slab coordinate points through the tape, opposite the envelope outward
    normal.  Each record is assigned to its nearest positive Gauss-Legendre
    ordinate and converted from partial current to angular flux exactly.
    """
    if ordinates < 2 or ordinates % 2:
        raise ValueError("ordinates must be a positive even integer")
    edges = np.asarray(energy_edges_eV, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy edges must be strictly increasing")
    columns = bank.columns
    names = np.asarray(columns["particle"]).astype(str)
    mu_inward = -np.asarray(columns["mu"], dtype=float)
    energies = np.asarray(columns["energy_eV"], dtype=float)
    current = np.asarray(columns["weight"], dtype=float)
    selected = (names == particle) & (mu_inward > 0.0) & (current > 0.0)
    mu, weights = np.polynomial.legendre.leggauss(ordinates)
    positive_mu = mu[mu > 0.0]
    positive_weights = weights[mu > 0.0]
    result = np.zeros((len(edges) - 1, len(positive_mu)))
    groups = np.searchsorted(edges, energies[selected], side="right") - 1
    groups[energies[selected] == edges[-1]] = len(edges) - 2
    if np.any(groups < 0) or np.any(groups >= len(edges) - 1):
        raise ValueError(f"{particle} record lies outside deterministic grid")
    angle_indices = np.argmin(
        np.abs(mu_inward[selected, None] - positive_mu[None, :]), axis=1
    )
    for group, angle, value in zip(groups, angle_indices, current[selected]):
        result[group, angle] += value / (
            positive_mu[angle] * positive_weights[angle]
        )
    projected_current = np.sum(
        result * (positive_mu * positive_weights)[None, :]
    )
    expected_current = float(current[selected].sum())
    if not np.isclose(
        projected_current, expected_current, rtol=1e-12, atol=1e-15
    ):
        raise RuntimeError(
            "ordinate projection does not conserve incoming current"
        )
    return result


def solve_multilayer_sn(
    layers: Sequence[LayerTransport],
    incoming_angular_flux: Mapping[str, np.ndarray],
    energy_edges: Mapping[str, Sequence[float]],
    *,
    ordinates: int = 16,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 1000,
) -> SNResult:
    """Solve slab transport by source iteration and exponential characteristics."""
    if not layers or ordinates < 2 or ordinates % 2:
        raise ValueError(
            "layers are required and ordinates must be positive even"
        )
    mu, weights = np.polynomial.legendre.leggauss(ordinates)
    positive = mu > 0.0
    particles = tuple(energy_edges)
    groups = {p: len(np.asarray(energy_edges[p])) - 1 for p in particles}
    group_count = max(groups.values())
    layer_count = len(layers)
    angular = np.zeros((layer_count, len(particles), group_count, ordinates))
    old_scalar = np.zeros((layer_count, len(particles), group_count))
    incoming_current = np.zeros((len(particles), group_count))
    pindex = {name: index for index, name in enumerate(particles)}
    boundary = {}
    right_boundary = np.zeros((len(particles), group_count, ordinates))
    forward_interfaces = np.zeros(
        (layer_count + 1, len(particles), group_count, ordinates)
    )
    backward_interfaces = np.zeros_like(forward_interfaces)
    for particle in particles:
        particle_groups = groups[particle]
        values = np.asarray(incoming_angular_flux[particle], dtype=float)
        if values.shape != (
            particle_groups,
            int(np.count_nonzero(positive)),
        ):
            raise ValueError(f"invalid incoming angular flux for {particle}")
        boundary[particle] = values
        incoming_current[pindex[particle], :particle_groups] = np.sum(
            values * (mu[positive] * weights[positive])[None, :], axis=1
        )
    for iteration in range(1, maximum_iterations + 1):
        scalar = np.sum(angular * weights[None, None, None, :], axis=-1)
        source = np.zeros_like(scalar)
        for li, layer in enumerate(layers):
            for particle, pi in pindex.items():
                particle_groups = groups[particle]
                scatter = np.asarray(
                    layer.scattering_xs_cm_1[particle], dtype=float
                )
                if scatter.shape != (particle_groups, particle_groups):
                    raise ValueError(
                        "scattering matrices use [outgoing, incoming]"
                    )
                source[li, pi, :particle_groups] += (
                    0.5 * scatter @ scalar[li, pi, :particle_groups]
                )
            if (
                "neutron" in pindex
                and "photon" in pindex
                and layer.neutron_to_photon_xs_cm_1 is not None
            ):
                matrix = np.asarray(
                    layer.neutron_to_photon_xs_cm_1, dtype=float
                )
                expected = (groups["photon"], groups["neutron"])
                if matrix.shape != expected:
                    raise ValueError(
                        "neutron-to-photon matrices use [photon, neutron]"
                    )
                source[li, pindex["photon"], : groups["photon"]] += (
                    0.5
                    * matrix
                    @ scalar[li, pindex["neutron"], : groups["neutron"]]
                )
        for particle, pi in pindex.items():
            particle_groups = groups[particle]
            pos_indices = np.flatnonzero(positive)
            forward_interfaces[0, pi, :particle_groups, pos_indices] = (
                boundary[particle].T
            )
            for local_index, ordinate_index in enumerate(pos_indices):
                psi = boundary[particle][:, local_index].copy()
                for li, layer in enumerate(layers):
                    total = np.asarray(
                        layer.total_xs_cm_1[particle], dtype=float
                    )
                    tau = total * layer.thickness_cm / mu[ordinate_index]
                    attenuation = np.exp(-tau)
                    equilibrium = np.divide(
                        source[li, pi, :particle_groups],
                        total,
                        out=np.zeros_like(total),
                        where=total > 0.0,
                    )
                    outgoing = psi * attenuation + equilibrium * (
                        1.0 - attenuation
                    )
                    angular[li, pi, :particle_groups, ordinate_index] = (
                        np.where(
                            tau > 1.0e-14,
                            equilibrium
                            + (psi - equilibrium)
                            * np.divide(
                                1.0 - attenuation,
                                tau,
                                out=np.ones_like(tau),
                                where=tau > 1.0e-14,
                            ),
                            0.5 * (psi + outgoing),
                        )
                    )
                    psi = outgoing
                    forward_interfaces[
                        li + 1, pi, :particle_groups, ordinate_index
                    ] = psi
                right_boundary[pi, :particle_groups, ordinate_index] = psi
            for ordinate_index in np.flatnonzero(~positive):
                psi = np.zeros(particle_groups)
                for li in range(layer_count - 1, -1, -1):
                    layer = layers[li]
                    total = np.asarray(
                        layer.total_xs_cm_1[particle], dtype=float
                    )
                    tau = total * layer.thickness_cm / abs(mu[ordinate_index])
                    attenuation = np.exp(-tau)
                    equilibrium = np.divide(
                        source[li, pi, :particle_groups],
                        total,
                        out=np.zeros_like(total),
                        where=total > 0.0,
                    )
                    outgoing = psi * attenuation + equilibrium * (
                        1.0 - attenuation
                    )
                    angular[li, pi, :particle_groups, ordinate_index] = (
                        np.where(
                            tau > 1.0e-14,
                            equilibrium
                            + (psi - equilibrium)
                            * np.divide(
                                1.0 - attenuation,
                                tau,
                                out=np.ones_like(tau),
                                where=tau > 1.0e-14,
                            ),
                            0.5 * (psi + outgoing),
                        )
                    )
                    psi = outgoing
                    backward_interfaces[
                        li, pi, :particle_groups, ordinate_index
                    ] = psi
        new_scalar = np.sum(angular * weights[None, None, None, :], axis=-1)
        error = float(np.max(np.abs(new_scalar - old_scalar)))
        scale = max(float(np.max(np.abs(new_scalar))), 1.0)
        old_scalar = new_scalar
        if error <= tolerance * scale:
            break
    else:
        raise RuntimeError("multigroup source iteration did not converge")
    outgoing_current = np.zeros_like(incoming_current)
    reflected_current = np.sum(
        backward_interfaces[0][..., ~positive]
        * (np.abs(mu[~positive]) * weights[~positive])[None, None, :],
        axis=-1,
    )
    interface_current = np.sum(
        forward_interfaces[..., positive]
        * (mu[positive] * weights[positive])[None, None, None, :],
        axis=-1,
    ) - np.sum(
        backward_interfaces[..., ~positive]
        * (np.abs(mu[~positive]) * weights[~positive])[None, None, None, :],
        axis=-1,
    )
    absorption = np.zeros((layer_count, len(particles), group_count))
    neutron_to_photon = np.zeros((layer_count, group_count))
    heating = np.zeros_like(absorption)
    energy_centers = np.zeros((len(particles), group_count))
    for particle, pi in pindex.items():
        particle_groups = groups[particle]
        outgoing_current[pi, :particle_groups] = np.sum(
            right_boundary[pi, :particle_groups][:, positive]
            * (mu[positive] * weights[positive])[None, :],
            axis=1,
        )
        centers = 0.5 * (
            np.asarray(energy_edges[particle][:-1])
            + np.asarray(energy_edges[particle][1:])
        )
        energy_centers[pi, :particle_groups] = centers
        for li, layer in enumerate(layers):
            absorption[li, pi, :particle_groups] = (
                np.asarray(layer.absorption_xs_cm_1[particle])
                * new_scalar[li, pi, :particle_groups]
                * layer.thickness_cm
            )
            heating[li, pi, :particle_groups] = (
                absorption[li, pi, :particle_groups] * centers
            )
            scattering_rate = (
                np.asarray(layer.scattering_xs_cm_1[particle])
                * new_scalar[li, pi, :particle_groups][None, :]
                * layer.thickness_cm
            )
            heating[li, pi, :particle_groups] += np.sum(
                scattering_rate * (centers[None, :] - centers[:, None]),
                axis=0,
            )
    if "neutron" in pindex and "photon" in pindex:
        for li, layer in enumerate(layers):
            if layer.neutron_to_photon_xs_cm_1 is not None:
                neutron_to_photon[li, : groups["photon"]] = (
                    np.asarray(layer.neutron_to_photon_xs_cm_1)
                    @ new_scalar[li, pindex["neutron"], : groups["neutron"]]
                    * layer.thickness_cm
                )
    total_in = float(incoming_current.sum())
    total_out = float(outgoing_current.sum() + reflected_current.sum())
    total_abs = float(absorption.sum())
    total_produced = float(neutron_to_photon.sum())
    balance = abs(total_in + total_produced - total_out - total_abs) / max(
        total_in + total_produced, 1.0e-300
    )
    energy_in = float(np.sum(incoming_current * energy_centers))
    energy_out = float(
        np.sum((outgoing_current + reflected_current) * energy_centers)
    )
    produced_energy = 0.0
    if "photon" in pindex:
        produced_energy = float(
            np.sum(
                neutron_to_photon[:, : groups["photon"]]
                * energy_centers[pindex["photon"], : groups["photon"]]
            )
        )
    deposited_energy = float(heating.sum())
    energy_balance = abs(
        energy_in + produced_energy - energy_out - deposited_energy
    ) / max(energy_in + produced_energy, 1.0e-300)
    return SNResult(
        particles=particles,
        group_counts=groups,
        scalar_flux=new_scalar,
        incoming_current=incoming_current,
        outgoing_current=outgoing_current,
        reflected_current=reflected_current,
        interface_current=interface_current,
        absorption_rate=absorption,
        neutron_to_photon_rate=neutron_to_photon,
        heating_eV_per_source=heating,
        particle_balance_error=float(balance),
        energy_balance_error=float(energy_balance),
        iterations=iteration,
    )
