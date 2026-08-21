"""Export OpenMC 0.16 closed-envelope records without tally conditioning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .dagmc_envelope import DagmcEnvelope
from .magnet_boundary_envelope import build_correlated_bank
from .magnet_boundary_envelope import write_handoff
from .openmc16 import PDG_PARTICLES


def _hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vectors(values, field: str) -> np.ndarray:
    return np.column_stack([values[field][axis] for axis in "xyz"])


def export_openmc16_handoff(
    output_path: str | Path,
    *,
    statepoint_path: str | Path,
    surface_source_paths: Sequence[str | Path],
    envelope: DagmcEnvelope,
    histories: int,
    energy_edges_by_particle: Mapping[str, Sequence[float]],
    physical_source_rate_per_s: float | None = None,
    parastell_commit: str,
    source_definition_sha256: str,
) -> dict:
    """Write a correlated v2 handoff and independent crossing closure."""
    import openmc

    if histories <= 0:
        raise ValueError("histories must be positive")
    arrays = []
    source_hashes = []
    for path in surface_source_paths:
        source_path = Path(path)
        with h5py.File(source_path) as source:
            if "source_bank" not in source:
                raise ValueError(f"{source_path} has no source_bank")
            arrays.append(source["source_bank"][:])
        source_hashes.append(_hash(source_path))
    if not arrays:
        raise ValueError("at least one surface-source file is required")
    records = np.concatenate(arrays)
    surface_ids = np.asarray(records["surf_id"]).reshape(-1)
    keep = np.isin(surface_ids, envelope.envelope.surface_ids)
    records = records[keep]
    surface_ids = surface_ids[keep]
    pdg = np.asarray(records["particle"]).reshape(-1)
    inverse_pdg = {value: name for name, value in PDG_PARTICLES.items()}
    unsupported = set(np.unique(pdg)) - set(inverse_pdg)
    if unsupported:
        raise ValueError(
            f"unsupported PDG particle IDs: {sorted(unsupported)}"
        )
    particles = np.asarray([inverse_pdg[int(value)] for value in pdg])
    positions = _vectors(records, "r")
    directions = _vectors(records, "u")
    normals = envelope.normals_at(surface_ids, positions)
    transport_weights = np.asarray(records["wgt"]).reshape(-1) / histories
    bank = build_correlated_bank(
        envelope.envelope,
        position_global_cm=positions,
        direction_global=directions,
        energy_eV=np.asarray(records["E"]).reshape(-1),
        raw_weight=transport_weights,
        particle=particles,
        surface_id=surface_ids,
        time_s=np.asarray(records["time"]).reshape(-1),
        energy_edges_by_particle=energy_edges_by_particle,
        outward_normal_global=normals,
    )
    # Independent event-count uncertainty: summing these record contributions
    # in quadrature yields sqrt(sum(w_i**2))/histories in each projected bin.
    bank.columns["weight_std_dev"] = transport_weights.copy()

    statepoint = openmc.StatePoint(statepoint_path)
    closure = {}
    for particle in sorted(set(particles)):
        tally = statepoint.get_tally(
            name=f"pstl_envelope_{particle}_directional_current"
        )
        tally_total = float(np.sum(np.abs(tally.mean)))
        tally_sigma = float(np.sqrt(np.sum(tally.std_dev**2)))
        mask = particles == particle
        bank_total = float(transport_weights[mask].sum())
        bank_sigma = float(np.sqrt(np.sum(transport_weights[mask] ** 2)))
        combined = float(np.hypot(tally_sigma, bank_sigma))
        difference = bank_total - tally_total
        closure[particle] = {
            "tally_total_crossing_current": tally_total,
            "tally_std_dev_quadrature": tally_sigma,
            "bank_total_crossing_current": bank_total,
            "bank_poisson_std_dev": bank_sigma,
            "difference": difference,
            "combined_std_dev": combined,
            "z_score": difference / combined if combined else 0.0,
            "passes_three_sigma": abs(difference) <= 3.0 * combined + 1.0e-14,
        }
    statepoint.close()
    bank.metadata.update(
        {
            "independent_closure": closure,
            "continuous_energy_authoritative": True,
            "continuous_direction_authoritative": True,
            "source_surface_sha256": source_hashes,
        }
    )
    normalization = {
        "basis": "per source history",
        "particles_per_source_history": 1.0,
        "area_basis": "surface integrated; patch area stored separately",
        "energy_bin_width": "not divided; group-integrated partial current",
        "solid_angle_measure": "not divided; angular-bin solid angle in metadata",
        "quantity": "partial crossing current",
        "time_basis": "none for per-source values",
        "physical_source_rate_per_s": physical_source_rate_per_s,
        "physical_scaling_contract": (
            "multiply per-source quantities by physical_source_rate_per_s"
            if physical_source_rate_per_s is not None
            else "no physical scaling supplied"
        ),
    }
    provenance = {
        "parastell_commit": parastell_commit,
        "openmc_version": openmc.__version__,
        "openmc_statepoint_sha256": _hash(statepoint_path),
        "dagmc_geometry_sha256": envelope.envelope.dagmc_geometry_sha256,
        "source_definition_sha256": source_definition_sha256,
        "histories": histories,
        "surface_source_sha256": source_hashes,
    }
    return write_handoff(
        output_path,
        envelope.envelope,
        bank,
        provenance=provenance,
        normalization=normalization,
    )
