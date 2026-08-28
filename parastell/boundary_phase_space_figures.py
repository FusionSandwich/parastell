"""Geometry-neutral visual evidence for correlated magnet-boundary banks.

The routines in this module do not discover surfaces or infer facet topology.
They accept an already localized record table and make the raw and derived
phase-space relationships visible without destroying particle correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "parastell.boundary_phase_space_figure_manifest/v1.0.0"
PARTICLE_LABELS = {2112: "neutron", 22: "photon"}


@dataclass(frozen=True)
class BoundaryFigureInputs:
    """Validated arrays used by every phase-space figure."""

    position_cm: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    energy_eV: np.ndarray
    time_s: np.ndarray
    weight: np.ndarray
    particle: np.ndarray
    surface_id: np.ndarray
    mu: np.ndarray
    local_position_cm: np.ndarray | None
    facet_id: np.ndarray | None
    barycentric: np.ndarray | None


def _array(
    records: Mapping[str, Any], name: str, *, shape_tail: tuple[int, ...] = ()
) -> np.ndarray:
    if name not in records:
        raise ValueError(f"localized bank is missing {name!r}")
    values = np.asarray(records[name])
    if values.ndim != 1 + len(shape_tail) or values.shape[1:] != shape_tail:
        raise ValueError(f"{name} has invalid shape {values.shape}")
    return values


def _array_alias(
    records: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    shape_tail: tuple[int, ...] = (),
) -> np.ndarray:
    for name in names:
        if name in records:
            return _array(records, name, shape_tail=shape_tail)
    raise ValueError(f"localized bank is missing one of {names!r}")


def _optional_array(
    records: Mapping[str, Any], name: str, *, shape_tail: tuple[int, ...] = ()
) -> np.ndarray | None:
    if name not in records:
        return None
    return _array(records, name, shape_tail=shape_tail)


def _consistent_length(arrays: list[np.ndarray | None]) -> int:
    lengths = {len(value) for value in arrays if value is not None}
    if len(lengths) != 1:
        raise ValueError("localized bank columns have inconsistent lengths")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("localized bank contains no records")
    return next(iter(lengths))


def validate_figure_inputs(
    records: Mapping[str, Any], *, mu_tolerance: float = 1.0e-10
) -> BoundaryFigureInputs:
    """Validate a localized, correlated boundary-record table.

    OpenMC's raw bank supplies position, direction, energy, time, weight,
    particle, and surface ID. Normal, facet ID, barycentric coordinates, and
    local coordinates are localization products and must be derived from the
    same immutable DAGMC geometry before calling this function.
    """

    position = _array(records, "position_global_cm", shape_tail=(3,)).astype(
        float
    )
    direction = _array(records, "direction_global", shape_tail=(3,)).astype(
        float
    )
    normal = _array(records, "outward_normal_global", shape_tail=(3,)).astype(
        float
    )
    energy = _array(records, "energy_eV").astype(float)
    time = _array(records, "time_s").astype(float)
    weight = _array_alias(records, ("openmc_weight", "weight")).astype(float)
    particle = _array_alias(records, ("particle_pdg", "particle"))
    surface_id = _array(records, "surface_id").astype(np.int64)
    supplied_mu = _optional_array(records, "mu")
    local = _optional_array(records, "position_local_cm", shape_tail=(3,))
    facet_id = _optional_array(records, "facet_id")
    barycentric = _optional_array(
        records, "barycentric_coordinates", shape_tail=(3,)
    )
    count = _consistent_length(
        [
            position,
            direction,
            normal,
            energy,
            time,
            weight,
            particle,
            surface_id,
            supplied_mu,
            local,
            facet_id,
            barycentric,
        ]
    )
    numeric = [position, direction, normal, energy, time, weight]
    if local is not None:
        numeric.append(local)
    if barycentric is not None:
        numeric.append(barycentric)
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("localized bank contains non-finite values")
    direction_norm = np.linalg.norm(direction, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    if not np.allclose(direction_norm, 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("particle directions must be unit vectors")
    if not np.allclose(normal_norm, 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("outward normals must be unit vectors")
    if np.any(energy <= 0.0) or np.any(weight < 0.0) or np.any(time < 0.0):
        raise ValueError("energy must be positive; weight/time nonnegative")
    if np.any(surface_id <= 0):
        raise ValueError("surface IDs must be positive")
    mu = np.einsum("ij,ij->i", direction, normal)
    if supplied_mu is not None and not np.allclose(
        supplied_mu.astype(float), mu, atol=mu_tolerance, rtol=0.0
    ):
        raise ValueError(
            "stored mu does not equal direction dot outward normal"
        )
    if barycentric is not None:
        if not np.allclose(
            barycentric.sum(axis=1), 1.0, atol=1.0e-9, rtol=0.0
        ) or np.any(barycentric < -1.0e-9):
            raise ValueError("invalid facet barycentric coordinates")
    if facet_id is not None:
        if np.issubdtype(np.asarray(facet_id).dtype, np.number):
            if np.any(np.asarray(facet_id, dtype=int) < 0):
                raise ValueError("facet IDs must be nonnegative")
        elif any(not str(value) for value in facet_id):
            raise ValueError("facet IDs must be nonempty")
    if count != len(mu):
        raise AssertionError("internal record-count mismatch")
    return BoundaryFigureInputs(
        position_cm=position,
        direction=direction,
        normal=normal,
        energy_eV=energy,
        time_s=time,
        weight=weight,
        particle=particle,
        surface_id=surface_id,
        mu=mu,
        local_position_cm=None if local is None else local.astype(float),
        facet_id=None if facet_id is None else np.asarray(facet_id),
        barycentric=(
            None if barycentric is None else barycentric.astype(float)
        ),
    )


def summarize_phase_space(
    values: BoundaryFigureInputs,
    *,
    source_histories: int,
    grazing_tolerance: float,
) -> dict[str, Any]:
    """Return auditable finite-list statistics without marginal resampling."""

    if source_histories <= 0:
        raise ValueError("source_histories must be positive")
    if not 0.0 <= grazing_tolerance < 1.0:
        raise ValueError("grazing_tolerance must be in [0, 1)")
    incoming = values.mu < -grazing_tolerance
    outgoing = values.mu > grazing_tolerance
    grazing = ~(incoming | outgoing)
    normalized_weight = values.weight / float(source_histories)
    return {
        "record_count": int(len(values.mu)),
        "incoming_count": int(incoming.sum()),
        "outgoing_count": int(outgoing.sum()),
        "grazing_count": int(grazing.sum()),
        "surface_ids": sorted(int(item) for item in set(values.surface_id)),
        "raw_weight_sum": float(values.weight.sum()),
        "per_source_weight_sum": float(normalized_weight.sum()),
        "source_histories": int(source_histories),
        "normalization": "raw OpenMC weight / exact source histories",
        "energy_range_eV": [
            float(values.energy_eV.min()),
            float(values.energy_eV.max()),
        ],
        "time_range_s": [
            float(values.time_s.min()),
            float(values.time_s.max()),
        ],
        "mu_range": [float(values.mu.min()), float(values.mu.max())],
        "facet_localization_present": values.facet_id is not None,
        "barycentric_localization_present": values.barycentric is not None,
        "local_coordinates_present": values.local_position_cm is not None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _particle_labels(values: np.ndarray) -> np.ndarray:
    labels = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            labels.append(value.decode())
        elif isinstance(value, str):
            labels.append(value)
        else:
            labels.append(PARTICLE_LABELS.get(int(value), str(int(value))))
    return np.asarray(labels)


def write_phase_space_figures(
    records: Mapping[str, Any],
    output_directory: str | Path,
    *,
    geometry_label: str,
    geometry_sha256: str,
    source_bank_sha256: str,
    source_histories: int,
    grazing_tolerance: float = 1.0e-8,
    status: str = "BOUNDED_TEST_ONLY",
) -> dict[str, Any]:
    """Write matched PNG evidence and a hash-bound manifest."""

    import matplotlib.pyplot as plt

    values = validate_figure_inputs(records)
    summary = summarize_phase_space(
        values,
        source_histories=source_histories,
        grazing_tolerance=grazing_tolerance,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    labels = _particle_labels(values.particle)
    sense = np.where(
        values.mu < -grazing_tolerance,
        "incoming",
        np.where(values.mu > grazing_tolerance, "outgoing", "grazing"),
    )
    figures: list[dict[str, Any]] = []

    def save(figure: Any, filename: str, content: str) -> None:
        path = output / filename
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        figures.append(
            {
                "path": filename,
                "sha256": _sha256(path),
                "content": content,
                "units": "cm; eV; s; dimensionless mu",
                "normalization": summary["normalization"],
                "status": status,
            }
        )

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    colors = {
        "incoming": "#2166ac",
        "outgoing": "#b2182b",
        "grazing": "#666666",
    }
    for category in ("incoming", "outgoing", "grazing"):
        selected = sense == category
        if np.any(selected):
            axis.scatter(
                *values.position_cm[selected].T,
                c=colors[category],
                s=16,
                alpha=0.8,
                label=f"{category} ({int(selected.sum())})",
            )
    axis.set(xlabel="x [cm]", ylabel="y [cm]", zlabel="z [cm]")
    axis.legend(loc="best")
    axis.set_title(f"{geometry_label}: correlated magnet-envelope crossings")
    save(
        figure,
        "01_crossing_positions.png",
        "global crossing positions by mu sense",
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    for particle in sorted(set(labels)):
        selected = labels == particle
        axis.scatter(
            values.mu[selected],
            values.energy_eV[selected],
            s=20,
            alpha=0.75,
            label=f"{particle} ({int(selected.sum())})",
        )
    axis.axvspan(-grazing_tolerance, grazing_tolerance, color="0.85")
    axis.set_yscale("log")
    axis.set(xlabel=r"$\mu=\Omega\cdot n_{out}$", ylabel="energy [eV]")
    axis.legend(loc="best")
    axis.set_title(f"{geometry_label}: energy-angle correlation")
    save(
        figure,
        "02_mu_energy.png",
        "correlated energy versus outward-normal cosine",
    )

    coordinates = (
        values.local_position_cm
        if values.local_position_cm is not None
        else values.position_cm
    )
    coordinate_label = (
        "local" if values.local_position_cm is not None else "global"
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values.mu,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        s=24,
    )
    figure.colorbar(points, ax=axis, label=r"$\mu=\Omega\cdot n_{out}$")
    axis.set(
        xlabel=f"{coordinate_label} coordinate 1 [cm]",
        ylabel=f"{coordinate_label} coordinate 2 [cm]",
    )
    axis.set_title(
        f"{geometry_label}: entry map in {coordinate_label} coordinates"
    )
    save(
        figure,
        "03_local_entry_map.png",
        f"{coordinate_label} position colored by mu",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hist(
        values.mu, bins=np.linspace(-1.0, 1.0, 31), weights=values.weight
    )
    axes[0].set(xlabel=r"$\mu$", ylabel="raw OpenMC crossing weight")
    axes[1].hist(np.log10(values.energy_eV), bins=30, weights=values.weight)
    axes[1].set(
        xlabel="log10 energy [eV]", ylabel="raw OpenMC crossing weight"
    )
    figure.suptitle(
        f"{geometry_label}: weighted phase-space marginals (diagnostic only)"
    )
    save(
        figure,
        "04_weighted_diagnostics.png",
        "mu and energy weighted diagnostics",
    )

    manifest = {
        "schema": SCHEMA,
        "geometry_label": geometry_label,
        "geometry_sha256": geometry_sha256,
        "source_bank_sha256": source_bank_sha256,
        "status": status,
        "grazing_tolerance": grazing_tolerance,
        "summary": summary,
        "figures": figures,
        "correlation_warning": (
            "figures are projections of one correlated record table; the "
            "marginals are not independently resampled source definitions"
        ),
    }
    manifest_path = output / "FIGURE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest
