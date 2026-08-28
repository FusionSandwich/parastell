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
    weight_per_source_history: np.ndarray | None
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
    normalized_weight = _optional_array(records, "weight_per_source_history")
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
            normalized_weight,
            particle,
            surface_id,
            supplied_mu,
            local,
            facet_id,
            barycentric,
        ]
    )
    numeric = [position, direction, normal, energy, time, weight]
    if normalized_weight is not None:
        numeric.append(normalized_weight)
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
        weight_per_source_history=(
            None
            if normalized_weight is None
            else normalized_weight.astype(float)
        ),
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
    phase_space_manifest: Mapping[str, Any],
    grazing_tolerance: float,
) -> dict[str, Any]:
    """Return auditable finite-list statistics without marginal resampling."""

    source_histories = _bound_source_histories(phase_space_manifest)
    if not 0.0 <= grazing_tolerance < 1.0:
        raise ValueError("grazing_tolerance must be in [0, 1)")
    incoming = values.mu < -grazing_tolerance
    outgoing = values.mu > grazing_tolerance
    grazing = ~(incoming | outgoing)
    normalized_weight = values.weight / float(source_histories)
    if values.weight_per_source_history is not None and not np.allclose(
        values.weight_per_source_history,
        normalized_weight,
        atol=0.0,
        rtol=1.0e-14,
    ):
        raise ValueError(
            "weight_per_source_history disagrees with raw OpenMC weight "
            "divided by the exact history binding"
        )
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


def _bound_source_histories(manifest: Mapping[str, Any]) -> int:
    """Return the exact history count from a strict phase-space manifest."""

    if not isinstance(manifest, Mapping):
        raise ValueError("verified phase_space_manifest is required")
    expected_schema = "parastell.openmc16_surface_phase_space/v1.0.0"
    if manifest.get("schema") != expected_schema:
        raise ValueError("unsupported phase-space manifest schema")
    if manifest.get("raw_phase_space_pass") is not True:
        raise ValueError("phase-space manifest is not verified PASS")
    histories = manifest.get("source_histories")
    if (
        isinstance(histories, bool)
        or not isinstance(histories, (int, np.integer))
        or int(histories) <= 0
    ):
        raise ValueError(
            "manifest source_histories must be a positive integer"
        )
    binding = manifest.get("history_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("kind") != "fixed_source_run"
    ):
        raise ValueError("a fixed-source-run history binding is required")
    if binding.get("source_histories") != histories:
        raise ValueError("phase manifest and history binding disagree")
    if binding.get("openmc_version") != "0.16.0":
        raise ValueError("history binding is not OpenMC 0.16.0")
    if not str(binding.get("run_id", "")).strip():
        raise ValueError("history binding omits run_id")
    for name in ("settings_payload_sha256", "statepoint_sha256"):
        digest = str(binding.get(name, "")).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"history binding has invalid {name}")
    return int(histories)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: object, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return digest


def _verified_facet_mesh(
    facet_catalog: Mapping[str, Any] | None,
    topology_manifest_sha256: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if facet_catalog is None:
        if topology_manifest_sha256 is not None:
            raise ValueError(
                "topology hash was supplied without a facet catalog"
            )
        return None, None, None
    if topology_manifest_sha256 is None:
        raise ValueError("facet catalog requires topology_manifest_sha256")
    digest = _sha256_text(topology_manifest_sha256, "topology_manifest_sha256")
    if facet_catalog.get("topology_manifest_sha256") != digest:
        raise ValueError("facet catalog and topology manifest hash disagree")
    if facet_catalog.get("normal_source") != "dagmc_forward_reverse_topology":
        raise ValueError("facet catalog normals are not topology-derived")
    triangles = np.asarray(
        facet_catalog.get("vertices_global_cm"), dtype=float
    )
    surfaces = np.asarray(facet_catalog.get("surface_id"), dtype=np.int64)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or surfaces.shape != (len(triangles),)
        or len(triangles) == 0
        or np.any(~np.isfinite(triangles))
    ):
        raise ValueError("facet catalog mesh arrays are invalid")
    return triangles, surfaces, digest


def surface_current_by_particle_and_sense(
    values: BoundaryFigureInputs,
    *,
    source_histories: int,
    grazing_tolerance: float,
) -> dict[str, Any]:
    """Compute crossing current without angular or tally conditioning."""

    if (
        isinstance(source_histories, bool)
        or not isinstance(source_histories, (int, np.integer))
        or int(source_histories) <= 0
    ):
        raise ValueError("source_histories must be a positive integer")
    normalized_weight = values.weight / float(source_histories)
    labels = _particle_labels(values.particle)
    surfaces = np.asarray(sorted(set(values.surface_id)), dtype=np.int64)
    groups = (
        (
            "neutron_incoming",
            labels == "neutron",
            values.mu < -grazing_tolerance,
        ),
        (
            "neutron_outgoing",
            labels == "neutron",
            values.mu > grazing_tolerance,
        ),
        (
            "photon_incoming",
            labels == "photon",
            values.mu < -grazing_tolerance,
        ),
        ("photon_outgoing", labels == "photon", values.mu > grazing_tolerance),
    )
    currents = {}
    for name, particle_mask, sense_mask in groups:
        currents[name] = [
            float(
                normalized_weight[
                    particle_mask & sense_mask & (values.surface_id == surface)
                ].sum()
            )
            for surface in surfaces
        ]
    return {
        "surface_ids": surfaces.tolist(),
        "normalization": "raw OpenMC weight / exact source histories",
        "units": "particle/source history",
        "currents": currents,
    }


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


def _write_phase_space_figures(
    records: Mapping[str, Any],
    output_directory: str | Path,
    *,
    geometry_label: str,
    geometry_sha256: str,
    source_bank_sha256: str,
    phase_space_manifest: Mapping[str, Any],
    source_bank_sha256s: list[str] | None = None,
    facet_catalog: Mapping[str, Any] | None = None,
    topology_manifest_sha256: str | None = None,
    run_audit_sha256: str | None = None,
    grazing_tolerance: float = 1.0e-8,
    status: str = "BOUNDED_TEST_ONLY",
) -> dict[str, Any]:
    """Write matched PNG evidence and a hash-bound manifest."""

    import matplotlib.pyplot as plt

    values = validate_figure_inputs(records)
    facet_triangles, facet_surfaces, topology_hash = _verified_facet_mesh(
        facet_catalog, topology_manifest_sha256
    )
    if facet_surfaces is not None and not set(values.surface_id).issubset(
        set(facet_surfaces)
    ):
        raise ValueError("facet catalog omits one or more crossing surfaces")
    summary = summarize_phase_space(
        values,
        phase_space_manifest=phase_space_manifest,
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
    if facet_triangles is not None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = Poly3DCollection(
            facet_triangles,
            facecolor="#bdbdbd",
            edgecolor="#737373",
            linewidth=0.15,
            alpha=0.22,
        )
        axis.add_collection3d(mesh)
        mesh_points = facet_triangles.reshape(-1, 3)
        axis.auto_scale_xyz(
            mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2]
        )
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

    current_summary = surface_current_by_particle_and_sense(
        values,
        source_histories=summary["source_histories"],
        grazing_tolerance=grazing_tolerance,
    )
    surface_ids = np.asarray(current_summary["surface_ids"], dtype=np.int64)
    x = np.arange(len(surface_ids), dtype=float)
    width = 0.19
    figure, axis = plt.subplots(figsize=(max(8, len(surface_ids) * 0.55), 5.5))
    for offset, (name, current) in enumerate(
        current_summary["currents"].items()
    ):
        label = name.replace("_", " ")
        axis.bar(x + (offset - 1.5) * width, current, width, label=label)
    axis.set_xticks(x, [str(value) for value in surface_ids], rotation=45)
    axis.set(
        xlabel="DAGMC surface ID",
        ylabel="crossing current [particle/source history]",
    )
    axis.legend(loc="best", fontsize="small")
    axis.set_title(f"{geometry_label}: current by envelope surface and sense")
    save(
        figure,
        "05_current_by_surface.png",
        "incoming/outgoing neutron and photon crossing current by surface",
    )

    manifest = {
        "schema": SCHEMA,
        "geometry_label": geometry_label,
        "geometry_sha256": geometry_sha256,
        "source_bank_sha256": source_bank_sha256,
        "source_bank_sha256s": (
            [source_bank_sha256]
            if source_bank_sha256s is None
            else source_bank_sha256s
        ),
        "phase_space_history_binding": phase_space_manifest["history_binding"],
        "facet_catalog_overlay": facet_triangles is not None,
        "topology_manifest_sha256": topology_hash,
        "surface_run_audit_sha256": run_audit_sha256,
        "surface_current": current_summary,
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


def _resolved_verified_path(row: Mapping[str, Any], label: str) -> Path:
    if not isinstance(row, Mapping):
        raise ValueError(f"{label} binding must be a mapping")
    path = Path(str(row.get("path", ""))).resolve()
    expected = _sha256_text(row.get("sha256"), f"{label} sha256")
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} path/hash binding failed")
    return path


def _load_npz_mapping(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _verify_localized_rows_against_catalog(
    records: Mapping[str, np.ndarray], catalog: Mapping[str, np.ndarray]
) -> None:
    required_catalog = {
        "facet_id",
        "surface_id",
        "vertices_global_cm",
        "outward_normal_global",
    }
    missing = required_catalog - set(catalog)
    if missing:
        raise ValueError(f"facet catalog omits {sorted(missing)}")
    facet_ids = np.asarray(catalog["facet_id"]).astype(str)
    if len(facet_ids) != len(set(facet_ids)):
        raise ValueError("facet catalog IDs are not unique")
    lookup = {value: index for index, value in enumerate(facet_ids)}
    catalog_surfaces = np.asarray(catalog["surface_id"], dtype=np.int64)
    triangles = np.asarray(catalog["vertices_global_cm"], dtype=float)
    normals = np.asarray(catalog["outward_normal_global"], dtype=float)
    if (
        catalog_surfaces.shape != (len(facet_ids),)
        or triangles.shape != (len(facet_ids), 3, 3)
        or normals.shape != (len(facet_ids), 3)
    ):
        raise ValueError("facet catalog columns are misaligned")

    positions = np.asarray(records["position_global_cm"], dtype=float)
    directions = np.asarray(records["direction_global"], dtype=float)
    record_surfaces = np.asarray(records["surface_id"], dtype=np.int64)
    record_facets = np.asarray(records["facet_id"]).astype(str)
    barycentric = np.asarray(records["barycentric_coordinates"], dtype=float)
    record_normals = np.asarray(records["outward_normal_global"], dtype=float)
    local_positions = np.asarray(records["position_local_cm"], dtype=float)
    local_directions = np.asarray(records["direction_local"], dtype=float)
    frames = np.asarray(records["local_frame_global"], dtype=float)
    for record_index, facet_id in enumerate(record_facets):
        if facet_id not in lookup:
            raise ValueError("localized row names an unknown facet")
        facet_index = lookup[facet_id]
        if record_surfaces[record_index] != catalog_surfaces[facet_index]:
            raise ValueError("localized facet/surface identity mismatch")
        reconstructed = (
            barycentric[record_index, :, None] * triangles[facet_index]
        ).sum(axis=0)
        if not np.allclose(
            reconstructed, positions[record_index], atol=1.0e-8, rtol=0.0
        ):
            raise ValueError("localized barycentric reconstruction mismatch")
        if not np.allclose(
            record_normals[record_index],
            normals[facet_index],
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("localized outward normal mismatch")
        frame = frames[record_index]
        if frame.shape != (3, 3) or not np.allclose(
            np.cross(frame[0], frame[1]), frame[2], atol=1.0e-10, rtol=0.0
        ):
            raise ValueError("localized frame is not right handed")
        if not np.allclose(
            frame[2], record_normals[record_index], atol=1.0e-10, rtol=0.0
        ):
            raise ValueError("localized frame normal mismatch")
        if not np.allclose(
            local_directions[record_index],
            frame @ directions[record_index],
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("localized direction roundtrip mismatch")
        if not np.allclose(
            local_positions[record_index],
            frame @ (positions[record_index] - triangles[facet_index, 0]),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError("localized position roundtrip mismatch")


def write_phase_space_figures(
    run_audit_path: str | Path,
    output_directory: str | Path,
    *,
    expected_run_audit_sha256: str,
    grazing_tolerance: float = 1.0e-8,
    status: str = "BOUNDED_TEST_ONLY",
) -> dict[str, Any]:
    """Render only from a hash-bound COMPLETE surface-run audit artifact."""

    audit_path = Path(run_audit_path).resolve()
    expected_audit_hash = _sha256_text(
        expected_run_audit_sha256, "expected_run_audit_sha256"
    )
    if not audit_path.is_file() or _sha256(audit_path) != expected_audit_hash:
        raise ValueError("surface-run audit file/hash binding failed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != "parastell.openmc16_surface_run_audit/v1.0.0":
        raise ValueError("unsupported surface-run audit schema")
    if audit.get("status") != "COMPLETE_CROSSING_BANK":
        raise ValueError("surface-run audit is not COMPLETE_CROSSING_BANK")

    geometry_path = _resolved_verified_path(audit.get("geometry"), "geometry")
    geometry_hash = _sha256(geometry_path)
    bank_rows = audit.get("source_banks")
    if not isinstance(bank_rows, list) or not bank_rows:
        raise ValueError("surface-run audit has no source-bank bindings")
    bank_paths = [
        _resolved_verified_path(row, f"source bank {index}")
        for index, row in enumerate(bank_rows)
    ]
    bank_hashes = [_sha256(path) for path in bank_paths]

    phase_manifest = audit.get("phase_space_manifest")
    histories = _bound_source_histories(phase_manifest)
    source_rows = phase_manifest.get("source_files")
    if not isinstance(source_rows, list):
        raise ValueError("phase manifest omits source file bindings")
    if [
        str(row.get("sha256", "")).lower() for row in source_rows
    ] != bank_hashes:
        raise ValueError("phase manifest source banks disagree with run audit")
    binding = phase_manifest["history_binding"]
    settings_path = _resolved_verified_path(
        {
            "path": binding.get("settings_payload_path"),
            "sha256": binding.get("settings_payload_sha256"),
        },
        "settings payload",
    )
    statepoint_path = _resolved_verified_path(
        {
            "path": binding.get("statepoint_path"),
            "sha256": binding.get("statepoint_sha256"),
        },
        "statepoint",
    )

    localized_path = _resolved_verified_path(
        audit.get("localized_records"), "localized records"
    )
    catalog_path = _resolved_verified_path(
        audit.get("facet_catalog"), "facet catalog"
    )
    topology_path = _resolved_verified_path(
        audit.get("topology_manifest"), "topology manifest"
    )
    localized_hash = _sha256(localized_path)
    catalog_hash = _sha256(catalog_path)
    topology_hash = _sha256(topology_path)
    localization_binding = audit.get("localization_topology_binding")
    expected_localization_binding = {
        "geometry_sha256": geometry_hash,
        "source_bank_sha256s": bank_hashes,
        "source_histories": histories,
        "settings_payload_sha256": _sha256(settings_path),
        "statepoint_sha256": _sha256(statepoint_path),
        "localized_records_sha256": localized_hash,
        "facet_catalog_sha256": catalog_hash,
        "topology_manifest_sha256": topology_hash,
    }
    if localization_binding != expected_localization_binding:
        raise ValueError("localization_topology_binding is inconsistent")

    records = _load_npz_mapping(localized_path)
    required_localized = {
        "position_global_cm",
        "direction_global",
        "energy_eV",
        "time_s",
        "openmc_weight",
        "weight_per_source_history",
        "particle_pdg",
        "surface_id",
        "facet_id",
        "barycentric_coordinates",
        "outward_normal_global",
        "mu",
        "position_local_cm",
        "direction_local",
        "local_frame_global",
    }
    missing = required_localized - set(records)
    if missing:
        raise ValueError(f"localized record table omits {sorted(missing)}")
    catalog = _load_npz_mapping(catalog_path)
    _verify_localized_rows_against_catalog(records, catalog)
    catalog["normal_source"] = "dagmc_forward_reverse_topology"
    catalog["topology_manifest_sha256"] = topology_hash

    geometry_label = str(audit.get("geometry_label", "")).strip()
    if not geometry_label:
        raise ValueError("surface-run audit omits geometry_label")
    return _write_phase_space_figures(
        records,
        output_directory,
        geometry_label=geometry_label,
        geometry_sha256=geometry_hash,
        source_bank_sha256=bank_hashes[0],
        source_bank_sha256s=bank_hashes,
        phase_space_manifest=phase_manifest,
        facet_catalog=catalog,
        topology_manifest_sha256=topology_hash,
        run_audit_sha256=expected_audit_hash,
        grazing_tolerance=grazing_tolerance,
        status=status,
    )
