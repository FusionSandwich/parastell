"""Generic, filter-preserving OpenMC 0.16 tally result extraction.

Unlike the scalar-flux convenience reader, this module retains every declared
filter, nuclide, score, first moment, and uncertainty.  It never manufactures
cross-bin covariance: unavailable covariance is explicit in the result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEMA = "parastell.openmc16_response_results/v1.0.0"
SCORE_SEMANTICS = {
    "flux": ("scalar_flux_spectrum", "1/cm2/source_history/bin", True),
    "heating": ("heating", "eV/source_history/bin", False),
    "damage-energy": (
        "damage_energy",
        "eV/source_history/bin",
        False,
    ),
    "H1-production": ("H1_production", "atoms/source_history/bin", False),
    "H2-production": ("H2_production", "atoms/source_history/bin", False),
    "H3-production": ("H3_production", "atoms/source_history/bin", False),
    "He3-production": (
        "He3_production",
        "atoms/source_history/bin",
        False,
    ),
    "He4-production": (
        "He4_production",
        "atoms/source_history/bin",
        False,
    ),
    "events": (
        "reaction_or_particle_production",
        "events/source_history/bin",
        False,
    ),
}


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _native(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return _text(value)
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_filter(
    node: h5py.Group, filter_id: int, tally_name: str
) -> dict[str, Any]:
    """Read one OpenMC filter without discarding compound bin identities."""
    if "type" not in node or "n_bins" not in node:
        raise ValueError(f"{tally_name} contains an invalid filter")
    filter_type = _text(node["type"][()]).strip().lower()
    n_bins = int(node["n_bins"][()])
    if not filter_type or n_bins <= 0:
        raise ValueError(f"{tally_name} contains an invalid filter")

    if filter_type == "particleproduction":
        if "particles" not in node:
            raise ValueError(
                f"{tally_name} particle-production identities are missing"
            )
        particles = [
            _text(value).strip().lower() for value in node["particles"][:]
        ]
        if not particles or any(not value for value in particles):
            raise ValueError(
                f"{tally_name} particle-production identities are invalid"
            )
        energies = None
        if "energies" in node:
            numeric = np.asarray(node["energies"][:], dtype=float).reshape(-1)
            if (
                len(numeric) < 2
                or np.any(~np.isfinite(numeric))
                or np.any(np.diff(numeric) <= 0.0)
            ):
                raise ValueError(
                    f"{tally_name} outgoing-energy edges are invalid"
                )
            energies = numeric.tolist()
        energy_bins = 1 if energies is None else len(energies) - 1
        if n_bins != len(particles) * energy_bins:
            raise ValueError(
                f"{tally_name} particle-production bin count is inconsistent"
            )
        bins: list[dict[str, Any]] = []
        for particle in particles:
            if energies is None:
                bins.append({"produced_particle": particle})
            else:
                for low, high in pairwise(energies):
                    bins.append(
                        {
                            "produced_particle": particle,
                            "outgoing_energy_low_eV": low,
                            "outgoing_energy_high_eV": high,
                        }
                    )
        return {
            "filter_id": filter_id,
            "type": filter_type,
            "n_bins": n_bins,
            "bins": bins,
            "particles": particles,
            "outgoing_energy_edges_eV": energies,
        }

    if "bins" not in node:
        raise ValueError(f"{tally_name} {filter_type} filter bins are missing")
    bins = _native(node["bins"][:])
    if filter_type == "energy":
        numeric = np.asarray(bins, dtype=float).reshape(-1)
        if (
            len(numeric) != n_bins + 1
            or np.any(~np.isfinite(numeric))
            or np.any(np.diff(numeric) <= 0.0)
        ):
            raise ValueError(f"{tally_name} energy edges are invalid")
        bins = numeric.tolist()
    elif len(bins) != n_bins:
        raise ValueError(
            f"{tally_name} {filter_type} bin count is inconsistent"
        )
    return {
        "filter_id": filter_id,
        "type": filter_type,
        "n_bins": n_bins,
        "bins": bins,
    }


def _run_metadata(statepoint: h5py.File) -> dict[str, Any]:
    if _text(statepoint.attrs.get("filetype", "")) != "statepoint":
        raise ValueError("input is not an OpenMC statepoint")
    version = tuple(
        int(value)
        for value in np.asarray(
            statepoint.attrs.get("openmc_version", ()), dtype=int
        ).tolist()
    )
    if version != (0, 16, 0):
        raise ValueError("statepoint is not from OpenMC 0.16.0")
    run_mode = _text(statepoint["run_mode"][()])
    particles = int(statepoint["n_particles"][()])
    batches = int(statepoint["n_batches"][()])
    current_batch = int(statepoint["current_batch"][()])
    seed = int(statepoint["seed"][()])
    if (
        run_mode != "fixed source"
        or min(particles, batches, seed) <= 0
        or current_batch != batches
    ):
        raise ValueError("statepoint fixed-source run is incomplete")
    return {
        "openmc_version": "0.16.0",
        "run_mode": run_mode,
        "particles_per_batch": particles,
        "batches": batches,
        "source_histories": particles * batches,
        "seed": seed,
    }


def _find_tally(tallies: h5py.Group, tally_name: str) -> h5py.Group:
    matches = [
        value
        for key, value in tallies.items()
        if key.startswith("tally ")
        and isinstance(value, h5py.Group)
        and "name" in value
        and _text(value["name"][()]) == tally_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"statepoint needs exactly one tally named {tally_name!r}"
        )
    return matches[0]


def read_openmc16_tally(
    statepoint_path: str | Path, tally_name: str
) -> dict[str, Any]:
    """Read one named tally and retain its full declared response axes."""
    path = Path(statepoint_path).resolve(strict=True)
    with h5py.File(path, "r") as statepoint:
        run = _run_metadata(statepoint)
        tallies = statepoint.get("tallies")
        if not isinstance(tallies, h5py.Group):
            raise TypeError("statepoint has no tallies group")
        tally = _find_tally(tallies, tally_name)
        required = {
            "score_bins",
            "nuclides",
            "estimator",
            "filters",
            "n_realizations",
            "results",
        }
        if not required.issubset(tally):
            raise ValueError(f"{tally_name} tally metadata is incomplete")
        scores = [_text(value).strip() for value in tally["score_bins"][:]]
        nuclides = [_text(value).strip() for value in tally["nuclides"][:]]
        if (
            not scores
            or not nuclides
            or any(not value for value in scores + nuclides)
        ):
            raise ValueError(f"{tally_name} score/nuclide axes are empty")
        filters = []
        dimensions = []
        for filter_id in tally["filters"][:]:
            node = tallies[f"filters/filter {int(filter_id)}"]
            filter_row = _read_filter(node, int(filter_id), tally_name)
            filters.append(filter_row)
            dimensions.append(int(filter_row["n_bins"]))
        realizations = int(tally["n_realizations"][()])
        if realizations <= 1:
            raise ValueError(
                f"{tally_name} needs at least two realizations for uncertainty"
            )
        if realizations != run["batches"]:
            raise ValueError(
                f"{tally_name} realizations do not match completed batches"
            )
        results = np.asarray(tally["results"][:], dtype=float)
        response_count = len(nuclides) * len(scores)
        expected = (int(np.prod(dimensions)), response_count, 2)
        if results.shape != expected:
            raise ValueError(
                f"{tally_name} results shape {results.shape} is not {expected}"
            )
        sums = results[..., 0] / realizations
        second = results[..., 1] / realizations
        variance = (second - sums**2) / (realizations - 1)
        scale = np.maximum(np.abs(second), np.abs(sums**2))
        tolerance = 1.0e-12 * np.maximum(scale, 1.0)
        if np.any(variance < -tolerance):
            raise ValueError(f"{tally_name} contains inconsistent moments")
        deviation = np.sqrt(np.where(variance < 0.0, 0.0, variance))
        shape = tuple(dimensions) + (len(nuclides), len(scores))
        means = sums.reshape(shape)
        deviations = deviation.reshape(shape)
        if np.any(~np.isfinite(means)) or np.any(~np.isfinite(deviations)):
            raise ValueError(f"{tally_name} contains non-finite results")
        negative_unsigned = any(
            score != "current" and np.any(means[..., index] < 0.0)
            for index, score in enumerate(scores)
        )
        if negative_unsigned or np.any(deviations < 0.0):
            raise ValueError(f"{tally_name} contains negative result moments")
        output = {
            "schema": SCHEMA,
            "tally_id": tally_name,
            "estimator": _text(tally["estimator"][()]),
            "filters": filters,
            "nuclides": nuclides,
            "scores": scores,
            "mean_per_source": means.tolist(),
            "std_dev_per_source": deviations.tolist(),
            "result_shape": list(shape),
            "realizations": realizations,
            "covariance": {
                "status": "UNAVAILABLE",
                "reason": "OpenMC statepoint contains independent tally moments only",
            },
            **run,
        }
    output["statepoint_path"] = str(path)
    output["statepoint_sha256"] = _sha256(path)
    return output


def read_openmc16_response_set(
    statepoint_path: str | Path,
    tally_names: Sequence[str],
) -> dict[str, Any]:
    """Read a complete declared tally set with duplicate-safe identities."""
    names = [str(value).strip() for value in tally_names]
    if (
        not names
        or any(not value for value in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("tally names must be nonempty and unique")
    results = [read_openmc16_tally(statepoint_path, name) for name in names]
    metadata_keys = (
        "statepoint_sha256",
        "openmc_version",
        "run_mode",
        "particles_per_batch",
        "batches",
        "source_histories",
        "seed",
    )
    metadata = {key: {item[key] for item in results} for key in metadata_keys}
    if any(len(values) != 1 for values in metadata.values()):
        raise ValueError("response results are not from one run")
    return {
        "schema": "parastell.openmc16_response_set/v1.0.0",
        "status": "SMOKE_RESULT" if results else "MISSING",
        "normalization": "per_source_history",
        **{key: next(iter(values)) for key, values in metadata.items()},
        "responses": results,
        "missing_response_semantics": "MISSING_IS_NOT_ZERO",
    }


def validate_response_coverage(
    response_set: Mapping[str, Any],
    required_tally_names: Sequence[str],
) -> dict[str, Any]:
    """Classify explicit result coverage without turning absence into zero."""
    available = {
        str(item.get("tally_id"))
        for item in response_set.get("responses", ())
        if isinstance(item, Mapping)
    }
    required = {str(value) for value in required_tally_names}
    missing = sorted(required - available)
    unexpected = sorted(available - required)
    return {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "required_tallies": sorted(required),
        "available_tallies": sorted(available),
        "missing_tallies": missing,
        "unexpected_tallies": unexpected,
        "missing_response_semantics": "MISSING_IS_NOT_ZERO",
    }


def tally_result_to_domain_estimators(
    result: Mapping[str, Any],
    *,
    domains_by_cell_id: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project one cell-filtered result into portable per-domain estimators."""
    if result.get("schema") != SCHEMA:
        raise ValueError("unsupported OpenMC response-result schema")
    filters = result.get("filters")
    if not isinstance(filters, list):
        raise TypeError("response-result filters are missing")
    types = [str(item.get("type")) for item in filters]
    if types.count("cell") != 1:
        raise ValueError("domain estimator requires exactly one cell filter")
    cell_axis = types.index("cell")
    cell_ids = [int(value) for value in filters[cell_axis]["bins"]]
    dimensions = [int(item["n_bins"]) for item in filters]
    nuclides = [str(value) for value in result["nuclides"]]
    scores = [str(value) for value in result["scores"]]
    expected_shape = tuple(dimensions) + (len(nuclides), len(scores))
    means = np.asarray(result["mean_per_source"], dtype=float)
    deviations = np.asarray(result["std_dev_per_source"], dtype=float)
    if means.shape != expected_shape or deviations.shape != expected_shape:
        raise ValueError("response-result arrays do not match declared axes")
    outputs = []
    for cell_index, cell_id in enumerate(cell_ids):
        if cell_id not in domains_by_cell_id:
            raise ValueError(f"no audited domain for OpenMC cell {cell_id}")
        domain = domains_by_cell_id[cell_id]
        domain_id = str(domain.get("domain_id", "")).strip()
        volume = float(domain.get("volume_cm3", 0.0))
        if not domain_id or not np.isfinite(volume) or volume <= 0.0:
            raise ValueError(f"OpenMC cell {cell_id} domain/volume is invalid")
        for nuclide_index, nuclide in enumerate(nuclides):
            for score_index, score in enumerate(scores):
                if score not in SCORE_SEMANTICS:
                    raise ValueError(f"unsupported OpenMC score {score!r}")
                observable, unit, divide_by_volume = SCORE_SEMANTICS[score]
                selection: list[Any] = [slice(None)] * len(dimensions)
                selection[cell_axis] = cell_index
                selected_mean = means[
                    tuple(selection) + (nuclide_index, score_index)
                ]
                selected_std = deviations[
                    tuple(selection) + (nuclide_index, score_index)
                ]
                if divide_by_volume:
                    selected_mean = selected_mean / volume
                    selected_std = selected_std / volume
                retained_filters = [
                    dict(item)
                    for index, item in enumerate(filters)
                    if index != cell_axis
                ]
                outputs.append(
                    {
                        "domain_id": domain_id,
                        "openmc_cell_id": cell_id,
                        "volume_cm3": volume,
                        "tally_id": result["tally_id"],
                        "observable": observable,
                        "score": score,
                        "nuclide": nuclide,
                        "estimator": result["estimator"],
                        "normalization": "per_source_history",
                        "unit": unit,
                        "filter_axes": retained_filters,
                        "mean_per_source": np.asarray(selected_mean).tolist(),
                        "std_dev_per_source": np.asarray(
                            selected_std
                        ).tolist(),
                        "result_state": (
                            "EMPTY_OR_UNDER_SAMPLED"
                            if np.all(np.asarray(selected_mean) == 0.0)
                            else "SCORED"
                        ),
                        "covariance": dict(result["covariance"]),
                        "statepoint_sha256": result["statepoint_sha256"],
                        "source_histories": int(result["source_histories"]),
                    }
                )
    return outputs
