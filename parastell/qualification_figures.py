"""Reproducible Prompt-A geometry and radiation-field figure set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dagmc_envelope import _openmc_to_outward_normal_sign
from .dagmc_envelope import _triangles


FIGURE_FILENAMES = (
    "01_full_sector_isometric.png",
    "02_full_sector_top.png",
    "03_full_sector_side.png",
    "04_cutaway_reactor_layers_and_magnets.png",
    "05_all_magnet_ids.png",
    "06_representative_magnet_casing_and_winding_pack.png",
    "07_outer_casing_external_envelope.png",
    "08_winding_pack_envelope.png",
    "09_outer_and_inner_casing_surface_classification.png",
    "10_surface_normals_and_roles.png",
    "11_closest_vessel_casing_clearance.png",
    "12_source_mesh_inside_plasma.png",
    "13_all_magnet_neutron_flux.png",
    "14_all_magnet_photon_flux.png",
    "15_all_magnet_neutron_heating.png",
    "16_all_magnet_photon_heating.png",
    "17_all_magnet_damage_energy.png",
    "18_representative_magnet_neutron_spectrum.png",
    "19_representative_magnet_photon_spectrum.png",
    "20_surface_current_by_face.png",
    "21_neutron_entry_locations.png",
    "22_photon_entry_locations.png",
    "23_mu_angle_distribution.png",
    "24_grazing_entry_map.png",
    "25_local_mesh_neutron_flux_slice.png",
    "26_local_mesh_photon_flux_slice.png",
    "27_local_mesh_heating_slice.png",
    "28_relative_uncertainty_map.png",
    "29_effective_sample_size_map.png",
    "30_prompt_vs_delayed_photon_context.png",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_figure_manifest(
    manifest: Mapping[str, Any], *, expected_files: Any = FIGURE_FILENAMES
) -> None:
    required = {
        "filename",
        "sha256",
        "units",
        "normalization_basis",
        "geometry_sha256",
        "source_mesh_sha256",
        "run_hash",
        "particle",
        "energy_integration_range_eV",
        "uncertainty_meaning",
        "simulation_status",
        "data_status",
    }
    rows = manifest.get("figures", [])
    filenames = [row.get("filename") for row in rows]
    if tuple(filenames) != tuple(expected_files):
        raise ValueError("figure manifest is incomplete or out of order")
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"figure {row.get('filename')} lacks {sorted(missing)}"
            )
        digest = str(row["sha256"])
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError("figure SHA-256 is malformed")
        if not all(
            str(row[name]).strip()
            for name in required - {"energy_integration_range_eV"}
        ):
            raise ValueError("figure metadata cannot be empty")


def _sample_points(volume: Any, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(volume.triangle_coords, dtype=float)
    if len(values) <= limit:
        return values
    rng = np.random.default_rng(seed)
    return values[np.sort(rng.choice(len(values), size=limit, replace=False))]


def _surface_points(
    volume: Any, surface_ids: Any, limit: int = 5000
) -> np.ndarray:
    selected = [
        np.asarray(surface.triangle_coords, dtype=float)
        for surface in volume.surfaces
        if int(surface.id) in {int(value) for value in surface_ids}
    ]
    if not selected:
        return np.empty((0, 3))
    values = np.concatenate(selected)
    return values[:: max(1, len(values) // limit)]


def _style_3d(ax: Any) -> None:
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_box_aspect((1, 1, 0.65))


def _finalize(
    fig: Any,
    path: Path,
    title: str,
    footer: str,
) -> None:
    fig.suptitle(title, fontsize=12)
    fig.text(0.01, 0.006, footer, fontsize=5.8, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _bar_figure(labels: Any, values: Any, ylabel: str, title: str, log: bool):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(labels))
    positive = np.asarray(values, dtype=float)
    ax.bar(x, positive, color="#3568a8")
    ax.set_xticks(x, labels, rotation=60, ha="right")
    ax.set_ylabel(ylabel)
    if log and np.any(positive > 0):
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    return fig


def generate_qualification_figures(
    *,
    artifact_dir: str | Path,
    worktree_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    import pydagmc
    from pymoab import core, types

    artifact = Path(artifact_dir).resolve()
    worktree = Path(worktree_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometry_path = artifact / "combined_reactor_magnets.h5m"
    source_path = artifact / "source_mesh_11x81x61.h5m"
    geometry_sha = _sha(geometry_path)
    source_sha = _sha(source_path)
    run_hash = (
        "00b93392c4004b0203e839909db2e2984a6d1ab9854ff5e0a2f6386146775370"
    )
    analysis = json.loads(
        (
            worktree
            / "validation_output/20260826T123526-0400/SHORT_RUN_SCIENTIFIC_ANALYSIS.json"
        ).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (
            worktree
            / "validation_output/20260826T123526-0400/ALL_MAGNET_SURFACE_INVENTORY.json"
        ).read_text(encoding="utf-8")
    )
    overlap = json.loads(
        (
            worktree
            / "validation_output/20260826T123526-0400/GEOMETRY_AND_SURFACE_AUDIT.json"
        ).read_text(encoding="utf-8")
    )["native_overlap_check"]
    model = pydagmc.Model(str(geometry_path))
    rep = next(
        row for row in inventory["magnets"] if row["coil_id"] == "coil-0005"
    )
    casing = model.volumes_by_id[int(rep["casing_volume_id"])]
    winding = model.volumes_by_id[int(rep["winding_pack_volume_id"])]
    footer_base = (
        f"geometry={geometry_sha[:12]} | source={source_sha[:12]} | run={run_hash[:12]} | "
        "direct 90-degree sector; not symmetry-expanded | uncertainty: OpenMC 1-sigma, "
        "energy-sum shown with conservative L1 bound"
    )
    rows = []

    def emit(
        index: int,
        title: str,
        draw: Callable[[], Any],
        *,
        units: str,
        particle: str,
        energy: Any,
        normalization: str,
        uncertainty: str,
        data_status: str = "DIRECTLY_SIMULATED",
    ) -> None:
        filename = FIGURE_FILENAMES[index - 1]
        path = output / filename
        fig = draw()
        footer = (
            footer_base
            + f" | particle={particle} | E={energy} | units={units} | norm={normalization}"
        )
        _finalize(fig, path, title, footer)
        rows.append(
            {
                "filename": filename,
                "title": title,
                "path": str(path),
                "sha256": _sha(path),
                "size_bytes": path.stat().st_size,
                "units": units,
                "normalization_basis": normalization,
                "geometry_sha256": geometry_sha,
                "source_mesh_sha256": source_sha,
                "run_hash": run_hash,
                "particle": particle,
                "energy_integration_range_eV": energy,
                "uncertainty_meaning": uncertainty,
                "simulation_status": "directly_simulated_90_degree_sector_not_symmetry_expanded",
                "data_status": data_status,
            }
        )

    volume_points = {
        volume_id: _sample_points(volume, 2500, 1000 + volume_id)
        for volume_id, volume in model.volumes_by_id.items()
        if volume_id != 44
    }

    def full3d():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        for volume_id in range(1, 7):
            p = volume_points[volume_id]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.3, alpha=0.18)
        for volume_id in range(7, 43, 2):
            p = volume_points[volume_id]
            ax.scatter(
                p[:, 0], p[:, 1], p[:, 2], s=0.7, c="#b23b3b", alpha=0.45
            )
        _style_3d(ax)
        return fig

    emit(
        1,
        "Full port-free sector — isometric",
        full3d,
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="native DAGMC coordinates",
        uncertainty="not applicable",
    )

    def projection(a: int, b: int, xlabel: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(9, 7))
        for volume_id in range(1, 7):
            p = volume_points[volume_id]
            ax.scatter(p[:, a], p[:, b], s=0.3, alpha=0.20)
        for volume_id in range(7, 43, 2):
            p = volume_points[volume_id]
            ax.scatter(p[:, a], p[:, b], s=0.6, c="#b23b3b", alpha=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        return fig

    emit(
        2,
        "Full port-free sector — top",
        lambda: projection(0, 1, "x (cm)", "y (cm)"),
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="native DAGMC coordinates",
        uncertainty="not applicable",
    )
    emit(
        3,
        "Full port-free sector — side",
        lambda: projection(0, 2, "x (cm)", "z (cm)"),
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="native DAGMC coordinates",
        uncertainty="not applicable",
    )

    def cutaway():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        for volume_id in range(1, 7):
            p = volume_points[volume_id]
            p = p[p[:, 1] <= np.median(p[:, 1])]
            ax.scatter(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                s=0.6,
                alpha=0.35,
                label=f"layer {volume_id}",
            )
        for volume_id in range(7, 43, 2):
            p = volume_points[volume_id]
            ax.scatter(
                p[:, 0], p[:, 1], p[:, 2], s=0.8, c="#b23b3b", alpha=0.5
            )
        _style_3d(ax)
        ax.legend(fontsize=6, ncol=2)
        return fig

    emit(
        4,
        "Cutaway reactor layers and magnets",
        cutaway,
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="native DAGMC coordinates",
        uncertainty="not applicable",
    )

    def magnet_ids():
        fig, ax = plt.subplots(figsize=(9, 7))
        for row in inventory["magnets"]:
            p = np.asarray(
                model.volumes_by_id[
                    int(row["casing_volume_id"])
                ].triangle_coords
            ).mean(axis=0)
            ax.scatter(p[0], p[1], c="#b23b3b", s=24)
            ax.text(
                p[0], p[1], row["coil_id"].replace("coil-", ""), fontsize=7
            )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_aspect("equal", adjustable="box")
        return fig

    emit(
        5,
        "All 18 magnet IDs",
        magnet_ids,
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="native DAGMC casing centroids",
        uncertainty="not applicable",
    )

    def rep_geometry():
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for volume, color, label in (
            (casing, "#4466aa", "casing"),
            (winding, "#d8872d", "winding pack"),
        ):
            p = _sample_points(volume, 9000, int(volume.id))
            ax.scatter(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                s=0.7,
                c=color,
                alpha=0.45,
                label=label,
            )
        _style_3d(ax)
        ax.legend()
        return fig

    emit(
        6,
        "Representative magnet: casing and winding pack",
        rep_geometry,
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="native DAGMC coordinates",
        uncertainty="not applicable",
    )

    def classified_surfaces(groups: Any):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for ids, color, label in groups:
            p = _surface_points(
                casing if label != "exposed winding" else winding, ids
            )
            if len(p):
                ax.scatter(
                    p[:, 0],
                    p[:, 1],
                    p[:, 2],
                    s=0.9,
                    c=color,
                    alpha=0.55,
                    label=label,
                )
        _style_3d(ax)
        ax.legend()
        return fig

    emit(
        7,
        "Outer-casing external envelope — open after exclusions",
        lambda: classified_surfaces(
            [
                (
                    rep["outer_casing_external_surface_ids"],
                    "#2e8b57",
                    "selected external casing",
                ),
                (
                    rep["casing_internal_surface_ids"],
                    "#b23b3b",
                    "excluded internal",
                ),
            ]
        ),
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="topology-selected casing faces",
        uncertainty="not applicable",
        data_status="FAIL_OPEN_MANIFOLD",
    )

    def winding_envelope():
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        p = _surface_points(winding, rep["winding_pack_surface_ids"], 12000)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.8, c="#d8872d", alpha=0.55)
        _style_3d(ax)
        return fig

    emit(
        8,
        "Closed winding-pack envelope",
        winding_envelope,
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="all DAGMC winding-pack boundary faces",
        uncertainty="not applicable",
    )
    exposed = sorted(
        set(rep["winding_pack_surface_ids"])
        - set(rep["casing_internal_surface_ids"])
    )
    emit(
        9,
        "External, internal, and exposed-winding face classification",
        lambda: classified_surfaces(
            [
                (
                    rep["outer_casing_external_surface_ids"],
                    "#2e8b57",
                    "external casing",
                ),
                (
                    rep["casing_internal_surface_ids"],
                    "#b23b3b",
                    "internal casing/WP",
                ),
                (exposed, "#d8872d", "exposed winding"),
            ]
        ),
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="DAGMC forward/reverse adjacency",
        uncertainty="not applicable",
        data_status="28_DIRECT_WINDING_TO_VACUUM_FACES_ALL_MAGNETS",
    )

    def normals():
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for surface in casing.surfaces:
            triangles = _triangles(surface.triangle_coords)
            centroid = triangles.reshape((-1, 3)).mean(axis=0)
            cross = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            vector = cross.sum(axis=0) * _openmc_to_outward_normal_sign(
                surface, int(casing.id)
            )
            vector /= np.linalg.norm(vector)
            ax.quiver(*centroid, *(80 * vector), color="#243b6b")
            ax.text(*centroid, str(int(surface.id)), fontsize=6)
        _style_3d(ax)
        return fig

    emit(
        10,
        "Surface IDs and topology-derived outward normals",
        normals,
        units="cm and unit vectors",
        particle="not applicable",
        energy="not applicable",
        normalization="DAGMC forward/reverse sense",
        uncertainty="not applicable",
    )

    def clearance():
        fig, ax = plt.subplots(figsize=(9, 7))
        vessel = volume_points[6]
        ax.scatter(
            vessel[:, 0], vessel[:, 2], s=0.4, alpha=0.2, label="vacuum vessel"
        )
        p = _sample_points(casing, 7000, 17)
        ax.scatter(
            p[:, 0], p[:, 2], s=0.8, alpha=0.4, label="coil-0005 casing"
        )
        bad = np.asarray(
            [
                row["location_cm"]
                for row in overlap["locations"]
                if 6 in row["volume_ids"] or 17 in row["volume_ids"]
            ]
        )
        ax.scatter(
            bad[:, 0],
            bad[:, 2],
            marker="x",
            s=55,
            c="#c43131",
            label="native overlap finding",
        )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("z (cm)")
        ax.legend()
        return fig

    emit(
        11,
        "Closest vessel/casing clearance — native overlaps present",
        clearance,
        units="cm",
        particle="not applicable",
        energy="not applicable",
        normalization="native overlap_check p=2",
        uncertainty="non-exhaustive native overlap search",
        data_status="FAIL_UNINTENDED_OVERLAPS",
    )

    def source_mesh():
        mesh = core.Core()
        mesh.load_file(str(source_path))
        tets = mesh.get_entities_by_type(0, types.MBTET)
        chosen = tets[:: max(1, len(tets) // 9000)]
        centroids = []
        for handle in chosen:
            conn = mesh.get_connectivity(handle)
            centroids.append(
                mesh.get_coords(conn).reshape((-1, 3)).mean(axis=0)
            )
        centroids = np.asarray(centroids)
        plasma = volume_points[1]
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            plasma[:, 0],
            plasma[:, 1],
            plasma[:, 2],
            s=0.3,
            alpha=0.15,
            label="plasma volume",
        )
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            s=1.0,
            c="#d45113",
            alpha=0.5,
            label="source tetrahedra",
        )
        _style_3d(ax)
        ax.legend()
        return fig

    emit(
        12,
        "11×81×61 source mesh inside plasma",
        source_mesh,
        units="cm",
        particle="neutron",
        energy="D-T source distribution",
        normalization="physical source 2.693734274881251e20 s^-1",
        uncertainty="not applicable",
    )

    components = analysis["all_magnet_components"]
    labels = [
        f"{row['magnet_id'][-4:]}-{row['component_role'][0]}"
        for row in components
    ]
    for index, metric, title, ylabel, particle in (
        (
            13,
            "neutron_scalar_flux",
            "All-magnet neutron scalar flux",
            "flux (cm$^{-2}$ s$^{-1}$; log)",
            "neutron",
        ),
        (
            14,
            "photon_scalar_flux",
            "All-magnet photon scalar flux",
            "flux (cm$^{-2}$ s$^{-1}$; log)",
            "photon",
        ),
        (
            15,
            "neutron_heating",
            "All-magnet neutron heating",
            "heating (W; log)",
            "neutron",
        ),
        (
            16,
            "photon_heating",
            "All-magnet photon heating",
            "heating (W; log)",
            "photon",
        ),
        (
            17,
            "damage_energy",
            "All-magnet damage energy",
            "damage energy (J/s; log)",
            "neutron",
        ),
    ):
        values = [row[metric]["value"] for row in components]
        emit(
            index,
            title,
            lambda labels=labels, values=values, ylabel=ylabel, title=title: _bar_figure(
                labels, values, ylabel, title, True
            ),
            units=ylabel,
            particle=particle,
            energy="energy-integrated",
            normalization="physical source 2.693734274881251e20 s^-1",
            uncertainty="OpenMC 1-sigma; zero scores are insufficient statistics",
        )

    bank = analysis["representative_magnet"]["winding_bank"]

    def spectrum(name: str):
        values = bank["spectra"][name]
        edges = np.asarray(values["energy_edges_eV"])
        current = np.asarray(values["current_per_source"])
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        ax.stairs(current, edges, color="#3568a8")
        ax.set_xscale("log")
        if np.any(current > 0):
            ax.set_yscale("log")
        ax.set_xlabel("energy (eV; log)")
        ax.set_ylabel("crossing current per source")
        return fig

    emit(
        18,
        "Representative winding-pack neutron spectrum",
        lambda: spectrum("neutron"),
        units="particles per source",
        particle="neutron",
        energy="0–20 MeV",
        normalization="raw bank weights / 500000 histories",
        uncertainty="weighted event-counting ESS",
    )
    emit(
        19,
        "Representative winding-pack photon spectrum",
        lambda: spectrum("photon"),
        units="particles per source",
        particle="photon",
        energy="0–1 GeV",
        normalization="raw bank weights / 500000 histories",
        uncertainty="weighted event-counting ESS",
    )

    def currents():
        entries = [
            row
            for row in bank["current_by_surface_particle_sense"]
            if row["particle"] == "neutron"
            and row["sense"] in {"incoming", "outgoing"}
        ]
        surfaces = sorted(set(row["surface_id"] for row in entries))
        incoming = [
            next(
                row["sum_weights_per_source"]
                for row in entries
                if row["surface_id"] == sid and row["sense"] == "incoming"
            )
            for sid in surfaces
        ]
        outgoing = [
            next(
                row["sum_weights_per_source"]
                for row in entries
                if row["surface_id"] == sid and row["sense"] == "outgoing"
            )
            for sid in surfaces
        ]
        fig, ax = plt.subplots(figsize=(9, 5.8))
        x = np.arange(len(surfaces))
        ax.bar(x - 0.2, incoming, 0.4, label="incoming")
        ax.bar(x + 0.2, outgoing, 0.4, label="outgoing")
        ax.set_xticks(x, [str(value) for value in surfaces])
        ax.set_xlabel("DAGMC surface ID")
        ax.set_ylabel("current per source")
        ax.legend()
        return fig

    emit(
        20,
        "Neutron surface current by winding-pack face",
        currents,
        units="particles per source",
        particle="neutron",
        energy="0–20 MeV",
        normalization="raw bank weights / 500000 histories",
        uncertainty="weighted event-counting ESS",
    )

    positions = np.asarray(bank["position_global_cm"])
    particles = np.asarray(bank["particle"])
    with h5py.File(
        artifact / "medium_500k_winding_coil0005.h5", "r"
    ) as source:
        senses = source["records/crossing_sense"].asstr()[()]
        mu_values = source["records/mu"][()]

    def entries(name: str):
        mask = (particles == name) & (senses == "incoming")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(positions[mask, 0], positions[mask, 2], s=25, alpha=0.7)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("z (cm)")
        return fig

    emit(
        21,
        "Neutron entry locations",
        lambda: entries("neutron"),
        units="cm",
        particle="neutron",
        energy="0–20 MeV",
        normalization="complete correlated bank",
        uncertainty="finite record locations; no interpolation",
    )
    emit(
        22,
        "Photon entry locations",
        lambda: entries("photon"),
        units="cm",
        particle="photon",
        energy="0–1 GeV",
        normalization="complete correlated bank",
        uncertainty="finite record locations; no interpolation",
    )

    def mu_plot():
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.hist(
            mu_values[particles == "neutron"],
            bins=np.linspace(-1, 1, 27),
            alpha=0.65,
            label="neutron",
        )
        ax.hist(
            mu_values[particles == "photon"],
            bins=np.linspace(-1, 1, 27),
            alpha=0.65,
            label="photon",
        )
        ax.axvspan(
            -0.1, 0.1, alpha=0.12, color="#d45113", label="near-grazing"
        )
        ax.set_xlabel(r"$\mu=\Omega\cdot n_{out}$")
        ax.set_ylabel("records")
        ax.legend()
        return fig

    emit(
        23,
        "Boundary-crossing mu distribution",
        mu_plot,
        units="dimensionless and record count",
        particle="neutron and photon",
        energy="particle-specific full ranges",
        normalization="complete correlated bank",
        uncertainty="finite weighted-event sample",
    )

    def grazing():
        mask = np.abs(mu_values) <= 0.1
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            positions[:, 0],
            positions[:, 2],
            c="#b8b8b8",
            s=14,
            alpha=0.35,
            label="all bank crossings",
        )
        if np.any(mask):
            ax.scatter(
                positions[mask, 0],
                positions[mask, 2],
                c=mu_values[mask],
                cmap="coolwarm",
                s=35,
                label="|mu| <= 0.1",
            )
        else:
            ax.text(
                0.5,
                0.90,
                "No |mu| <= 0.1 records in the complete 53-record bank;\n"
                "a physical zero is not inferred.",
                ha="center",
                va="top",
                transform=ax.transAxes,
            )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("z (cm)")
        ax.legend()
        return fig

    emit(
        24,
        "Near-grazing entry map (|mu|≤0.1)",
        grazing,
        units="cm; mu dimensionless",
        particle="neutron and photon",
        energy="particle-specific full ranges",
        normalization="complete correlated bank",
        uncertainty="finite weighted-event sample",
    )

    local = analysis["representative_magnet"]["local_mesh_flux"]

    def local_map(name: str, quantity: str):
        entries_data = local[name]
        coords = np.asarray(
            [row["local_centreline_coordinates_cm"] for row in entries_data]
        )
        if quantity == "flux":
            values = np.asarray([row["value"] for row in entries_data])
            label = f"{name} flux (cm$^{{-2}}$ s$^{{-1}}$)"
            values = np.log10(
                np.clip(
                    values,
                    np.min(values[values > 0]) if np.any(values > 0) else 1,
                    None,
                )
            )
        elif quantity == "relative":
            values = np.asarray(
                [
                    (
                        row["conservative_relative_uncertainty_upper_bound"]
                        if row["conservative_relative_uncertainty_upper_bound"]
                        is not None
                        else np.nan
                    )
                    for row in entries_data
                ]
            )
            label = "conservative relative uncertainty"
        else:
            values = np.asarray(
                [
                    (
                        row["batch_moment_effective_sample_size_proxy"]
                        if row["batch_moment_effective_sample_size_proxy"]
                        is not None
                        else np.nan
                    )
                    for row in entries_data
                ]
            )
            label = "batch-moment ESS proxy"
        fig, ax = plt.subplots(figsize=(8.5, 6))
        finite = np.isfinite(values)
        ax.scatter(
            coords[~finite, 0],
            coords[~finite, 1],
            c="#b8b8b8",
            s=18,
            alpha=0.45,
            label="no finite estimate",
        )
        if np.any(finite):
            plot = ax.scatter(
                coords[finite, 0],
                coords[finite, 1],
                c=values[finite],
                cmap="viridis",
                s=26,
            )
            fig.colorbar(plot, ax=ax, label=label)
        else:
            ax.text(
                0.5,
                0.94,
                "No finite values; all bins are INSUFFICIENT_STATISTICS.",
                ha="center",
                va="top",
                transform=ax.transAxes,
            )
        ax.set_xlabel("centreline arclength (cm)")
        ax.set_ylabel("local radial coordinate (cm)")
        if np.any(~finite):
            ax.legend(loc="lower right")
        if quantity in {"relative", "ess"}:
            ax.text(
                0.5,
                0.97,
                f"{int(np.sum(finite))}/{len(values)} finite; all bins "
                "INSUFFICIENT_STATISTICS",
                ha="center",
                va="top",
                transform=ax.transAxes,
            )
        return fig

    emit(
        25,
        "Local-mesh neutron flux slice",
        lambda: local_map("neutron", "flux"),
        units="log10(cm^-2 s^-1)",
        particle="neutron",
        energy="0–20 MeV",
        normalization="physical source; component-intersection volume",
        uncertainty="all bins insufficient at 500k",
    )
    emit(
        26,
        "Local-mesh photon flux slice",
        lambda: local_map("photon", "flux"),
        units="log10(cm^-2 s^-1)",
        particle="photon",
        energy="0–1 GeV",
        normalization="physical source; component-intersection volume",
        uncertainty="all bins insufficient at 500k",
    )

    def not_run():
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.axis("off")
        ax.text(0.5, 0.60, "NOT RUN", ha="center", va="center", fontsize=24)
        ax.text(
            0.5,
            0.43,
            "No local-mesh heating tally exists in the 500k statepoint.\nNo physical zero is claimed.",
            ha="center",
            va="center",
        )
        return fig

    emit(
        27,
        "Local-mesh heating slice",
        not_run,
        units="not available",
        particle="neutron and photon",
        energy="not available",
        normalization="not available",
        uncertainty="NOT_RUN",
        data_status="NOT_RUN_LOCAL_MESH_HEATING_TALLY_ABSENT",
    )
    emit(
        28,
        "Local-mesh neutron relative-uncertainty map",
        lambda: local_map("neutron", "relative"),
        units="dimensionless",
        particle="neutron",
        energy="0–20 MeV",
        normalization="OpenMC batch moments",
        uncertainty="conservative L1 relative upper bound",
    )
    emit(
        29,
        "Local-mesh neutron effective-sample-size map",
        lambda: local_map("neutron", "ess"),
        units="dimensionless proxy",
        particle="neutron",
        energy="0–20 MeV",
        normalization="1/(relative uncertainty)^2 proxy",
        uncertainty="event-level ESS unavailable; proxy only",
        data_status="BATCH_MOMENT_PROXY_NOT_EVENT_ESS",
    )

    def prompt_delayed():
        selected = [
            row for row in components if row["magnet_id"].endswith("coil-0005")
        ]
        prompt = sum(row["photon_scalar_flux"]["value"] for row in selected)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.bar([0], [prompt], width=0.55, color="#3568a8")
        ax.text(
            1, max(prompt, 1) * 0.5, "NOT SIMULATED", ha="center", va="center"
        )
        ax.set_xticks([0, 1], ["prompt photon flux", "delayed photon field"])
        ax.set_ylabel("flux (cm$^{-2}$ s$^{-1}$)")
        ax.set_xlim(-0.6, 1.6)
        return fig

    emit(
        30,
        "Prompt versus delayed photon context",
        prompt_delayed,
        units="cm^-2 s^-1 for prompt only",
        particle="photon",
        energy="0–1 GeV prompt; delayed absent",
        normalization="direct prompt transport only",
        uncertainty="delayed field NOT_RUN; not plotted as zero",
        data_status="PROMPT_DIRECT_DELAYED_NOT_RUN",
    )

    manifest = {
        "schema": "parastell.qualification_figure_manifest/v1",
        "figure_count": len(rows),
        "figures": rows,
    }
    validate_figure_manifest(manifest)
    target = Path(manifest_path)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selected_dir = worktree / "figures/selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_names = (
        "06_representative_magnet_casing_and_winding_pack.png",
        "07_outer_casing_external_envelope.png",
        "09_outer_and_inner_casing_surface_classification.png",
        "13_all_magnet_neutron_flux.png",
        "28_relative_uncertainty_map.png",
    )
    selected = []
    for name in selected_names:
        destination = selected_dir / name
        shutil.copy2(output / name, destination)
        selected.append(
            {
                "filename": name,
                "path": str(destination),
                "sha256": _sha(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    manifest["selected_committed_subset"] = selected
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
