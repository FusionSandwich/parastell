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
    scalar_flux: np.ndarray
    incoming_current: np.ndarray
    outgoing_current: np.ndarray
    absorption_rate: np.ndarray
    heating_eV_per_source: np.ndarray
    particle_balance_error: float
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
    if len(set(groups.values())) != 1:
        raise ValueError(
            "reference solver currently requires equal padded group axes"
        )
    group_count = next(iter(groups.values()))
    layer_count = len(layers)
    angular = np.zeros((layer_count, len(particles), group_count, ordinates))
    old_scalar = np.zeros((layer_count, len(particles), group_count))
    incoming_current = np.zeros((len(particles), group_count))
    pindex = {name: index for index, name in enumerate(particles)}
    boundary = {}
    right_boundary = np.zeros((len(particles), group_count, ordinates))
    for particle in particles:
        values = np.asarray(incoming_angular_flux[particle], dtype=float)
        if values.shape != (group_count, int(np.count_nonzero(positive))):
            raise ValueError(f"invalid incoming angular flux for {particle}")
        boundary[particle] = values
        incoming_current[pindex[particle]] = np.sum(
            values * (mu[positive] * weights[positive])[None, :], axis=1
        )
    for iteration in range(1, maximum_iterations + 1):
        scalar = np.sum(angular * weights[None, None, None, :], axis=-1)
        source = np.zeros_like(scalar)
        for li, layer in enumerate(layers):
            for particle, pi in pindex.items():
                scatter = np.asarray(
                    layer.scattering_xs_cm_1[particle], dtype=float
                )
                if scatter.shape != (group_count, group_count):
                    raise ValueError(
                        "scattering matrices use [outgoing, incoming]"
                    )
                source[li, pi] += 0.5 * scatter @ scalar[li, pi]
            if (
                "neutron" in pindex
                and "photon" in pindex
                and layer.neutron_to_photon_xs_cm_1 is not None
            ):
                matrix = np.asarray(
                    layer.neutron_to_photon_xs_cm_1, dtype=float
                )
                source[li, pindex["photon"]] += (
                    0.5 * matrix @ scalar[li, pindex["neutron"]]
                )
        for particle, pi in pindex.items():
            pos_indices = np.flatnonzero(positive)
            for local_index, ordinate_index in enumerate(pos_indices):
                psi = boundary[particle][:, local_index].copy()
                for li, layer in enumerate(layers):
                    total = np.asarray(
                        layer.total_xs_cm_1[particle], dtype=float
                    )
                    tau = total * layer.thickness_cm / mu[ordinate_index]
                    attenuation = np.exp(-tau)
                    equilibrium = np.divide(
                        source[li, pi],
                        total,
                        out=np.zeros_like(total),
                        where=total > 0.0,
                    )
                    outgoing = psi * attenuation + equilibrium * (
                        1.0 - attenuation
                    )
                    angular[li, pi, :, ordinate_index] = np.divide(
                        psi - outgoing,
                        tau,
                        out=0.5 * (psi + outgoing),
                        where=tau > 1.0e-14,
                    )
                    psi = outgoing
                right_boundary[pi, :, ordinate_index] = psi
            for ordinate_index in np.flatnonzero(~positive):
                psi = np.zeros(group_count)
                for li in range(layer_count - 1, -1, -1):
                    layer = layers[li]
                    total = np.asarray(
                        layer.total_xs_cm_1[particle], dtype=float
                    )
                    tau = total * layer.thickness_cm / abs(mu[ordinate_index])
                    attenuation = np.exp(-tau)
                    equilibrium = np.divide(
                        source[li, pi],
                        total,
                        out=np.zeros_like(total),
                        where=total > 0.0,
                    )
                    outgoing = psi * attenuation + equilibrium * (
                        1.0 - attenuation
                    )
                    angular[li, pi, :, ordinate_index] = np.divide(
                        psi - outgoing,
                        tau,
                        out=0.5 * (psi + outgoing),
                        where=tau > 1.0e-14,
                    )
                    psi = outgoing
        new_scalar = np.sum(angular * weights[None, None, None, :], axis=-1)
        error = float(np.max(np.abs(new_scalar - old_scalar)))
        scale = max(float(np.max(np.abs(new_scalar))), 1.0)
        old_scalar = new_scalar
        if error <= tolerance * scale:
            break
    else:
        raise RuntimeError("multigroup source iteration did not converge")
    outgoing_current = np.zeros_like(incoming_current)
    absorption = np.zeros((layer_count, len(particles), group_count))
    heating = np.zeros_like(absorption)
    for particle, pi in pindex.items():
        outgoing_current[pi] = np.sum(
            right_boundary[pi][:, positive]
            * (mu[positive] * weights[positive])[None, :],
            axis=1,
        )
        centers = 0.5 * (
            np.asarray(energy_edges[particle][:-1])
            + np.asarray(energy_edges[particle][1:])
        )
        for li, layer in enumerate(layers):
            absorption[li, pi] = (
                np.asarray(layer.absorption_xs_cm_1[particle])
                * new_scalar[li, pi]
                * layer.thickness_cm
            )
            heating[li, pi] = absorption[li, pi] * centers
    total_in = float(incoming_current.sum())
    total_out = float(outgoing_current.sum())
    total_abs = float(absorption.sum())
    balance = abs(total_in - total_out - total_abs) / max(total_in, 1.0e-300)
    return SNResult(
        particles=particles,
        scalar_flux=new_scalar,
        incoming_current=incoming_current,
        outgoing_current=outgoing_current,
        absorption_rate=absorption,
        heating_eV_per_source=heating,
        particle_balance_error=float(balance),
        iterations=iteration,
    )
