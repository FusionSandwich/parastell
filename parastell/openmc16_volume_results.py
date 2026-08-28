"""Geometry-neutral extraction of cell scalar spectra from OpenMC 0.16.

This reader intentionally consumes HDF5 statepoint data rather than relying on
unstable Python object ordering.  It exports bin-integrated scalar flux per
source history after division by an independently audited cell volume.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cell_scalar_flux_spectrum(
    statepoint_path: str | Path, tally_name: str
) -> dict[str, Any]:
    """Read one cell/particle/energy flux tally in declared filter order."""
    path = Path(statepoint_path).resolve()
    statepoint_sha256 = _sha256(path)
    with h5py.File(path, "r") as statepoint:
        if _text(statepoint.attrs.get("filetype", "")) != "statepoint":
            raise ValueError("input is not an OpenMC statepoint")
        statepoint_version = tuple(
            int(value)
            for value in np.asarray(
                statepoint.attrs.get("version", ()), dtype=int
            ).reshape(-1)
        )
        if not statepoint_version or statepoint_version[0] != 18:
            raise ValueError("unsupported OpenMC statepoint file version")
        version = tuple(
            int(value)
            for value in np.asarray(
                statepoint.attrs.get("openmc_version", ()), dtype=int
            ).tolist()
        )
        if version != (0, 16, 0):
            raise ValueError("statepoint is not from OpenMC 0.16.0")
        run_mode = _text(statepoint["run_mode"][()])
        particles_per_batch = int(statepoint["n_particles"][()])
        batches = int(statepoint["n_batches"][()])
        current_batch = int(statepoint["current_batch"][()])
        seed = int(statepoint["seed"][()])
        if (
            run_mode != "fixed source"
            or min(particles_per_batch, batches, seed) <= 0
            or current_batch != batches
        ):
            raise ValueError("statepoint fixed-source run is incomplete")
        tallies = statepoint.get("tallies")
        if tallies is None:
            raise ValueError("statepoint has no tallies group")
        tally = next(
            (
                value
                for key, value in tallies.items()
                if key.startswith("tally ")
                and "name" in value
                and _text(value["name"][()]) == tally_name
            ),
            None,
        )
        if tally is None:
            raise ValueError(f"statepoint has no tally named {tally_name!r}")
        required_tally_fields = {
            "score_bins",
            "nuclides",
            "estimator",
            "filters",
            "n_realizations",
            "results",
        }
        if not required_tally_fields.issubset(tally):
            raise ValueError(f"{tally_name} tally metadata is incomplete")
        scores = [_text(value) for value in tally["score_bins"][:]]
        nuclides = [_text(value).strip() for value in tally["nuclides"][:]]
        estimator = _text(tally["estimator"][()])
        if scores != ["flux"] or nuclides != ["total"]:
            raise ValueError(f"{tally_name} does not score only total flux")
        if estimator != "tracklength":
            raise ValueError(f"{tally_name} is not a track-length tally")
        filters = [
            tallies[f"filters/filter {int(value)}"]
            for value in tally["filters"][:]
        ]
        types = [_text(value["type"][()]) for value in filters]
        if len(types) != 3 or set(types) != {"cell", "particle", "energy"}:
            raise ValueError(
                f"{tally_name} is not a cell/particle/energy scalar-flux tally"
            )
        dimensions = tuple(int(value["n_bins"][()]) for value in filters)
        realizations = int(tally["n_realizations"][()])
        if realizations <= 1:
            raise ValueError(
                f"{tally_name} needs at least two realizations for uncertainty"
            )
        if realizations != current_batch or realizations != batches:
            raise ValueError(
                f"{tally_name} realizations do not match completed batches"
            )
        results = np.asarray(tally["results"][:], dtype=float)
        expected_results_shape = (int(np.prod(dimensions)), 1, 2)
        if results.shape != expected_results_shape:
            raise ValueError(
                f"{tally_name} results shape {results.shape} is not "
                f"{expected_results_shape}"
            )
        sums = results[..., 0].reshape(dimensions + (-1,))
        squares = results[..., 1].reshape(dimensions + (-1,))
        means = sums / realizations
        variance = (squares / realizations - means**2) / (realizations - 1)
        scale = np.maximum(np.abs(squares / realizations), np.abs(means**2))
        tolerance = 1.0e-12 * np.maximum(scale, 1.0)
        if np.any(variance < -tolerance):
            raise ValueError(
                f"{tally_name} contains inconsistent tally moments"
            )
        std_dev = np.sqrt(np.where(variance < 0.0, 0.0, variance))
        cell_axis = types.index("cell")
        particle_axis = types.index("particle")
        energy_axis = types.index("energy")
        cell_ids = np.asarray(
            filters[cell_axis]["bins"][:], dtype=int
        ).reshape(-1)
        scored_particles = [
            _text(value) for value in filters[particle_axis]["bins"][:]
        ]
        if len(scored_particles) != 1 or scored_particles[0] not in {
            "neutron",
            "photon",
        }:
            raise ValueError(
                f"{tally_name} must contain one supported particle"
            )
        energy_edges = np.asarray(filters[energy_axis]["bins"][:], dtype=float)
        if (
            len(energy_edges) != dimensions[energy_axis] + 1
            or np.any(~np.isfinite(energy_edges))
            or np.any(np.diff(energy_edges) <= 0.0)
        ):
            raise ValueError(f"{tally_name} energy edges are invalid")
        output_mean = np.zeros((len(cell_ids), len(energy_edges) - 1))
        output_std = np.zeros_like(output_mean)
        for cell_index in range(len(cell_ids)):
            for energy_index in range(len(energy_edges) - 1):
                selection = [0] * len(dimensions)
                selection[cell_axis] = cell_index
                selection[particle_axis] = 0
                selection[energy_axis] = energy_index
                output_mean[cell_index, energy_index] = float(
                    np.sum(means[tuple(selection)])
                )
                output_std[cell_index, energy_index] = float(
                    np.sqrt(np.sum(std_dev[tuple(selection)] ** 2))
                )
    return {
        "tally_id": tally_name,
        "statepoint_path": str(path),
        "statepoint_sha256": statepoint_sha256,
        "openmc_version": "0.16.0",
        "statepoint_file_version": list(statepoint_version),
        "run_mode": run_mode,
        "particles_per_batch": particles_per_batch,
        "batches": batches,
        "source_histories": particles_per_batch * batches,
        "seed": seed,
        "particle": scored_particles[0],
        "cell_ids": cell_ids.tolist(),
        "energy_edges_eV": energy_edges.tolist(),
        "track_length_mean_cm_per_source": output_mean.tolist(),
        "track_length_std_dev_cm_per_source": output_std.tolist(),
        "realizations": realizations,
    }


def scalar_flux_estimators(
    spectrum: Mapping[str, Any],
    *,
    domains_by_cell_id: Mapping[int, Mapping[str, Any]],
    geometry_binding: Mapping[str, Any],
    tally_definition_sha256: str,
    run_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Normalize cell track length by audited volumes for consumer handoff."""
    for name, value in {
        "raw_h5m_sha256": geometry_binding.get("raw_h5m_sha256"),
        "canonical_geometry_fingerprint": geometry_binding.get(
            "canonical_geometry_fingerprint"
        ),
        "tally_definition_sha256": tally_definition_sha256,
        "model_xml_sha256": run_binding.get("model_xml_sha256"),
        "settings_xml_sha256": run_binding.get("settings_xml_sha256"),
        "strict_run_audit_sha256": run_binding.get("strict_run_audit_sha256"),
        "root_acceptance_receipt_sha256": run_binding.get(
            "root_acceptance_receipt_sha256"
        ),
    }.items():
        text = str(value)
        if len(text) != 64 or any(
            character not in "0123456789abcdef" for character in text
        ):
            raise ValueError(f"{name} is not a SHA-256 digest")
    if run_binding.get("statepoint_sha256") != spectrum.get(
        "statepoint_sha256"
    ):
        raise ValueError("run binding used another statepoint")
    cells = [int(value) for value in spectrum["cell_ids"]]
    means = np.asarray(
        spectrum["track_length_mean_cm_per_source"], dtype=float
    )
    deviations = np.asarray(
        spectrum["track_length_std_dev_cm_per_source"], dtype=float
    )
    if means.shape != deviations.shape or means.shape[0] != len(cells):
        raise ValueError("cell scalar-flux arrays do not align")
    outputs = []
    for index, cell_id in enumerate(cells):
        try:
            domain = domains_by_cell_id[cell_id]
        except KeyError as exc:
            raise ValueError(
                f"no audited domain for OpenMC cell {cell_id}"
            ) from exc
        domain_id = str(domain.get("domain_id", "")).strip()
        volume = float(domain.get("volume_cm3", 0.0))
        receipt = str(domain.get("volume_receipt_sha256", ""))
        if (
            not domain_id
            or not np.isfinite(volume)
            or volume <= 0.0
            or len(receipt) != 64
        ):
            raise ValueError(f"OpenMC cell {cell_id} domain/volume is invalid")
        for key in ("raw_h5m_sha256", "canonical_geometry_fingerprint"):
            if domain.get(key) != geometry_binding.get(key):
                raise ValueError(
                    f"OpenMC cell {cell_id} volume used another geometry"
                )
        row_mean = means[index] / volume
        row_std = deviations[index] / volume
        if np.any(~np.isfinite(row_mean)) or np.any(row_mean < 0.0):
            raise ValueError(f"OpenMC cell {cell_id} scalar flux is invalid")
        outputs.append(
            {
                "domain_id": domain_id,
                "observable": "scalar_flux_spectrum",
                "particle": spectrum["particle"],
                "estimator": "tracklength",
                "normalization": "per_source_history",
                "unit": "1/cm2/source_history/bin",
                "energy_edges_eV": list(spectrum["energy_edges_eV"]),
                "mean_per_source": row_mean.tolist(),
                "std_dev_per_source": row_std.tolist(),
                "tally_id": spectrum["tally_id"],
                "openmc_cell_id": cell_id,
                "volume_cm3": volume,
                "volume_receipt_sha256": domain.get("volume_receipt_sha256"),
                "realizations": int(spectrum["realizations"]),
                "statepoint_sha256": spectrum["statepoint_sha256"],
                "raw_h5m_sha256": geometry_binding["raw_h5m_sha256"],
                "canonical_geometry_fingerprint": geometry_binding[
                    "canonical_geometry_fingerprint"
                ],
                "tally_definition_sha256": tally_definition_sha256,
                "model_xml_sha256": run_binding["model_xml_sha256"],
                "settings_xml_sha256": run_binding["settings_xml_sha256"],
                "strict_run_audit_sha256": run_binding[
                    "strict_run_audit_sha256"
                ],
                "root_acceptance_receipt_sha256": run_binding[
                    "root_acceptance_receipt_sha256"
                ],
                "source_histories": int(spectrum["source_histories"]),
            }
        )
    return outputs
