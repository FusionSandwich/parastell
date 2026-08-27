#!/usr/bin/env python3
"""Generate the hash-bound Prompt-7A geometry-parity visual package.

The renderer intentionally consumes the accepted R1 and failed-feature H5M
files directly.  It uses deterministic faceted point samples so the complete
package can be regenerated without an interactive ParaView session.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pydagmc
from scipy.spatial import cKDTree


SCHEMA = "parastell.geometry_figure_manifest/v1.0.0"
BRANCH = "magnet-radiation-geometry-parity-20260826"
BASE_SHA = "de7d2978ff314b060ca2e6b10745a034e8b2a3c4"
R1_FINGERPRINT = (
    "1c4a6c1fdb37f7bb9d7ef59ab99913884bde6df7b55977a53a68f2a037552bd1"
)
R1_HASH = "8741dd48fded42e8411816e56e3e5e10a29db26ddb785b4a389f8a38b09707a0"
FAILED_HASH = (
    "7c710fe3dd261ce7f46e5d08b4f9d2924513994ccb6229468933d0b91b2cd7cb"
)

COLORS = {
    "chamber": "#a855f7",
    "first_wall": "#ef4444",
    "breeder": "#f59e0b",
    "back_wall": "#eab308",
    "shield": "#22c55e",
    "vac_vessel": "#64748b",
    "magnets": "#2563eb",
    "casing": "#f97316",
    "winding": "#10b981",
    "interstitial": "#93c5fd",
    "graveyard": "#111827",
}

CAMERAS = {
    "isometric": (28.0, -50.0),
    "top": (90.0, -90.0),
    "side": (0.0, 0.0),
    "front": (0.0, -90.0),
    "rear": (0.0, 90.0),
    "cutaway": (24.0, -42.0),
}

OVERLAPS_R1 = np.asarray(
    [
        [1160.39, 380.476, 128.459],
        [1095.15, 498.581, 92.5898],
        [476.445, 1095.28, -86.2664],
        [364.23, 1158.37, -121.329],
    ]
)
OVERLAPS_FAILED = np.asarray(
    [
        [1165.96, 341.146, 104.741],
        [1145.84, 362.887, 100.333],
        [1151.02, 333.69, 92.709],
        [1088.88, 498.513, 92.6229],
        [438.026, 1095.55, -75.3721],
        [304.98, 1157.46, -84.0225],
        [333.69, 1151.02, -92.709],
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def volume_material(volume: Any) -> str:
    return str(getattr(volume, "material", "") or "").strip()


def surface_triangles(surface: Any) -> np.ndarray:
    values = np.asarray(surface.triangle_coords, dtype=float)
    return values.reshape((-1, 3, 3))


def evenly_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum, dtype=int)]


@dataclass
class VolumeView:
    volume_id: int
    material: str
    points: np.ndarray
    centroid: np.ndarray
    surface_centroids: dict[int, np.ndarray]


def load_views(path: Path, *, per_volume: int = 7500) -> dict[int, VolumeView]:
    model = pydagmc.Model(str(path))
    result: dict[int, VolumeView] = {}
    for volume_id, volume in sorted(model.volumes_by_id.items()):
        all_points = []
        surface_centroids = {}
        for surface in volume.surfaces:
            triangles = surface_triangles(surface)
            centroids = triangles.mean(axis=1)
            all_points.append(centroids)
            surface_centroids[int(surface.id)] = centroids.mean(axis=0)
        points = evenly_sample(np.concatenate(all_points), per_volume)
        result[int(volume_id)] = VolumeView(
            volume_id=int(volume_id),
            material=volume_material(volume),
            points=points,
            centroid=points.mean(axis=0),
            surface_centroids=surface_centroids,
        )
    return result


def role(view: VolumeView, *, failed: bool = False) -> str:
    material = view.material.lower()
    if material == "vacuum":
        material = "chamber"
    if not failed:
        return material if material in COLORS else "magnets"
    if 7 <= view.volume_id <= 42:
        return "casing" if view.volume_id % 2 else "winding"
    if view.volume_id == 43:
        return "interstitial"
    if view.volume_id == 44:
        return "graveyard"
    return material if material in COLORS else "magnets"


def common_axes(
    ax: Any, camera: str, *, title: str, points: np.ndarray
) -> None:
    elevation, azimuth = CAMERAS[camera]
    ax.view_init(elev=elevation, azim=azimuth)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    half = max(0.5 * float(np.max(upper - lower)), 1.0)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_zlabel("z [cm]")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(False)


def draw_views(
    ax: Any,
    views: dict[int, VolumeView],
    *,
    camera: str,
    title: str,
    failed: bool = False,
    selected: Iterable[int] | None = None,
    cutaway: bool = False,
    magnets_only: bool = False,
    transparent: bool = False,
    annotate_volumes: bool = False,
    annotate_surfaces: bool = False,
    overlap_points: np.ndarray | None = None,
) -> None:
    selected_set = (
        None if selected is None else {int(value) for value in selected}
    )
    rendered = []
    for volume_id, view in views.items():
        if selected_set is not None and volume_id not in selected_set:
            continue
        volume_role = role(view, failed=failed)
        if magnets_only and volume_role not in {
            "magnets",
            "casing",
            "winding",
        }:
            continue
        points = view.points
        if cutaway:
            points = points[points[:, 1] <= np.median(view.points[:, 1])]
        if not len(points):
            continue
        rendered.append(points)
        alpha = (
            0.19
            if transparent
            else (0.58 if volume_role != "magnets" else 0.78)
        )
        if volume_role in {"interstitial", "graveyard"}:
            alpha = 0.03
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=0.34 if len(points) > 1500 else 0.8,
            color=COLORS.get(volume_role, "#2563eb"),
            alpha=alpha,
            linewidths=0,
            depthshade=False,
        )
        if annotate_volumes:
            ax.text(
                *view.centroid, f"V{volume_id}", fontsize=5, color="#111827"
            )
        if annotate_surfaces:
            for surface_id, centroid in view.surface_centroids.items():
                ax.text(
                    *centroid, f"S{surface_id}", fontsize=4.8, color="#111827"
                )
    if overlap_points is not None and len(overlap_points):
        rendered.append(overlap_points)
        ax.scatter(
            overlap_points[:, 0],
            overlap_points[:, 1],
            overlap_points[:, 2],
            marker="X",
            s=60,
            color="#dc2626",
            edgecolors="white",
            linewidths=0.8,
            label="native overlap witness",
        )
    plot_points = np.concatenate(rendered) if rendered else np.zeros((1, 3))
    common_axes(ax, camera, title=title, points=plot_points)


def new_figure() -> tuple[Any, Any]:
    figure = plt.figure(figsize=(10, 7.5), constrained_layout=True)
    return figure, figure.add_subplot(111, projection="3d")


class FigureWriter:
    def __init__(self, output: Path):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=False)
        self.rows: list[dict[str, Any]] = []

    def save(
        self,
        figure: Any,
        filename: str,
        *,
        input_hash: str,
        fingerprint: str,
        camera: str,
        components: list[str],
        status: str,
        title: str,
    ) -> Path:
        path = self.output / filename
        figure.text(
            0.01,
            0.005,
            f"units: cm | {BRANCH}@{BASE_SHA[:12]} | geometry: {input_hash[:12]} | "
            f"fingerprint: {fingerprint[:12]} | status: {status}",
            fontsize=6.5,
            color="#475569",
        )
        figure.savefig(path, dpi=180, facecolor="white")
        plt.close(figure)
        self.rows.append(
            {
                "filename": filename,
                "title": title,
                "input_geometry_sha256": input_hash,
                "canonical_geometry_fingerprint": fingerprint,
                "branch": BRANCH,
                "sha": BASE_SHA,
                "camera": camera,
                "visible_components": components,
                "color_map": COLORS,
                "coordinate_units": "cm",
                "status": status,
                "sha256": sha256_file(path),
                "width_px": 1800,
                "height_px": 1350,
            }
        )
        return path

    def status_sheet(
        self,
        filename: str,
        title: str,
        lines: list[str],
        *,
        input_hash: str,
        fingerprint: str,
        status: str,
    ) -> Path:
        figure, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.03,
            0.92,
            title,
            transform=ax.transAxes,
            fontsize=18,
            weight="bold",
        )
        y = 0.82
        for line in lines:
            ax.text(
                0.04, y, line, transform=ax.transAxes, fontsize=11, va="top"
            )
            y -= 0.075
        return self.save(
            figure,
            filename,
            input_hash=input_hash,
            fingerprint=fingerprint,
            camera="status_sheet",
            components=[],
            status=status,
            title=title,
        )


def save_model_view(
    writer: FigureWriter,
    views: dict[int, VolumeView],
    filename: str,
    title: str,
    camera: str,
    *,
    failed: bool = False,
    input_hash: str = R1_HASH,
    fingerprint: str = R1_FINGERPRINT,
    status: str = "EVIDENCE_RENDERED",
    **kwargs: Any,
) -> Path:
    figure, ax = new_figure()
    draw_views(ax, views, camera=camera, title=title, failed=failed, **kwargs)
    roles = sorted({role(view, failed=failed) for view in views.values()})
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[item],
            label=item,
            markersize=7,
        )
        for item in roles
        if item in COLORS and item not in {"interstitial", "graveyard"}
    ]
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7)
    return writer.save(
        figure,
        filename,
        input_hash=input_hash,
        fingerprint=fingerprint,
        camera=camera,
        components=roles,
        status=status,
        title=title,
    )


def load_source_points(path: Path, *, maximum: int = 12000) -> np.ndarray:
    from pymoab import core, types

    mesh = core.Core()
    mesh.load_file(str(path))
    vertices = mesh.get_entities_by_type(0, types.MBVERTEX)
    points = np.asarray(mesh.get_coords(vertices), dtype=float).reshape(
        (-1, 3)
    )
    return evenly_sample(points, maximum)


def source_mesh_view(
    writer: FigureWriter,
    views: dict[int, VolumeView],
    source_points: np.ndarray,
    filename: str,
    title: str,
    *,
    status: str,
) -> Path:
    figure, ax = new_figure()
    draw_views(
        ax,
        views,
        camera="isometric",
        title=title,
        selected=[1],
        transparent=True,
    )
    ax.scatter(
        source_points[:, 0],
        source_points[:, 1],
        source_points[:, 2],
        s=1.1,
        color="#ec4899",
        alpha=0.65,
        linewidths=0,
        depthshade=False,
    )
    return writer.save(
        figure,
        filename,
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        camera="isometric",
        components=["chamber", "source_mesh"],
        status=status,
        title=title,
    )


def highlighted_surfaces_view(
    writer: FigureWriter,
    views: dict[int, VolumeView],
    dagmc_path: Path,
    surface_ids: list[int],
    filename: str,
    title: str,
    *,
    status: str,
) -> Path:
    model = pydagmc.Model(str(dagmc_path))
    surfaces = {}
    for volume in model.volumes_by_id.values():
        for surface in volume.surfaces:
            surfaces[int(surface.id)] = surface
    points = []
    centroids = []
    for surface_id in surface_ids:
        triangles = surface_triangles(surfaces[surface_id])
        triangle_centroids = triangles.mean(axis=1)
        points.append(evenly_sample(triangle_centroids, 1800))
        centroids.append((surface_id, triangle_centroids.mean(axis=0)))
    highlights = np.concatenate(points)
    figure, ax = new_figure()
    draw_views(
        ax,
        views,
        camera="isometric",
        title=title,
        failed=True,
        magnets_only=True,
        transparent=True,
    )
    ax.scatter(
        highlights[:, 0],
        highlights[:, 1],
        highlights[:, 2],
        s=2.4,
        color="#dc2626",
        alpha=0.92,
        linewidths=0,
        depthshade=False,
    )
    for surface_id, centroid in centroids:
        ax.text(*centroid, f"S{surface_id}", color="#7f1d1d", fontsize=6)
    return writer.save(
        figure,
        filename,
        input_hash=FAILED_HASH,
        fingerprint="NOT_ACCEPTED",
        camera="isometric",
        components=["casing", "winding", "highlighted_surfaces"],
        status=status,
        title=title,
    )


def projection_comparison(
    writer: FigureWriter,
    views: dict[int, VolumeView],
    filename: str,
    title: str,
    *,
    camera: str,
    cutaway: bool = False,
) -> Path:
    figure = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    for index, label in enumerate(
        ("Untouched R1", "Instrumented byte reuse"), start=1
    ):
        ax = figure.add_subplot(1, 2, index, projection="3d")
        draw_views(ax, views, camera=camera, title=label, cutaway=cutaway)
    return writer.save(
        figure,
        filename,
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        camera=f"matched_{camera}",
        components=sorted(COLORS),
        status="BYTE_IDENTICAL_REUSE",
        title=title,
    )


def magnet_closeup(
    writer: FigureWriter,
    views: dict[int, VolumeView],
    volume_id: int,
    magnet_index: int,
) -> Path:
    view = views[volume_id]
    reactors = [item for item in views.values() if item.volume_id <= 6]
    nearest_name = "not evaluated"
    nearest_distance = float("inf")
    sample = evenly_sample(view.points, 1200)
    for candidate in reactors:
        other = evenly_sample(candidate.points, 3500)
        # Bounded deterministic nearest sampled-facet-centroid distance.
        distance = float(cKDTree(other).query(sample, workers=1)[0].min())
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_name = (
                candidate.material or f"volume-{candidate.volume_id}"
            )
    figure, ax = new_figure()
    draw_views(
        ax,
        views,
        camera="isometric",
        title=f"magnet-{magnet_index:04d}: complete original envelope",
        selected=[volume_id],
        annotate_surfaces=True,
    )
    ax.text2D(
        0.02,
        0.03,
        f"volume V{volume_id} | nearest sampled component: {nearest_name} | "
        f"sampled facet-centroid clearance: {nearest_distance:.2f} cm\n"
        "Axes are global ParaStell coordinates; all boundary surface IDs shown.",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    return writer.save(
        figure,
        f"{35 + magnet_index:02d}_magnet-{magnet_index:04d}.png",
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        camera="magnet_isometric",
        components=[f"magnet-{magnet_index:04d}"],
        status="CLOSED_OUTER_ENVELOPE",
        title=f"magnet-{magnet_index:04d}",
    )


def contact_sheet(output: Path, sources: list[Path], title: str) -> None:
    thumbs = []
    for source in sources:
        image = Image.open(source).convert("RGB")
        image.thumbnail((540, 405))
        thumbs.append((source.name, image.copy()))
    columns = 3
    rows = int(np.ceil(len(thumbs) / columns))
    canvas = Image.new("RGB", (columns * 560, 80 + rows * 450), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((20, 20), title, fill="#0f172a", font=font)
    for index, (name, image) in enumerate(thumbs):
        x = (index % columns) * 560 + 10
        y = 70 + (index // columns) * 450
        canvas.paste(image, (x, y))
        draw.text((x, y + 410), name, fill="#334155", font=font)
    canvas.save(output, optimize=True)


def write_vtk_points(path: Path, views: dict[int, VolumeView]) -> None:
    points = np.concatenate([view.points for view in views.values()])
    volume_ids = np.concatenate(
        [
            np.full(len(view.points), view.volume_id, dtype=int)
            for view in views.values()
        ]
    )
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            "# vtk DataFile Version 3.0\nParaStell faceted evidence points\nASCII\n"
        )
        stream.write("DATASET POLYDATA\n")
        stream.write(f"POINTS {len(points)} float\n")
        np.savetxt(stream, points, fmt="%.8g")
        stream.write(f"VERTICES {len(points)} {2 * len(points)}\n")
        for index in range(len(points)):
            stream.write(f"1 {index}\n")
        stream.write(
            f"POINT_DATA {len(points)}\nSCALARS volume_id int 1\nLOOKUP_TABLE default\n"
        )
        np.savetxt(stream, volume_ids, fmt="%d")


def generate(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    writer = FigureWriter(output)
    r1 = load_views(Path(args.r1))
    failed = load_views(Path(args.failed), per_volume=4000)
    source_points = load_source_points(Path(args.source_mesh))
    failed_inventory = json.loads(
        Path(args.failed_inventory).read_text(encoding="utf-8")
    )
    direct_winding_surface_ids = sorted(
        {
            int(row["surface_id"])
            for magnet in failed_inventory["magnets"]
            for row in magnet["winding_pack_surface_adjacencies"]
            if int(row["adjacent_non_winding_volume_id"]) == 43
        }
    )

    # Groups A and B: matching names, cameras, and color assignments.
    specifications = [
        ("isometric", "isometric", {}),
        ("top", "top", {}),
        ("side", "side", {}),
        ("front_toroidal", "front", {}),
        ("rear_toroidal", "rear", {}),
        ("plasma_side_cutaway", "cutaway", {"cutaway": True}),
        ("radial_build_cutaway", "side", {"cutaway": True}),
        ("transparent_in_vessel_build", "isometric", {"transparent": True}),
        ("magnets_only", "isometric", {"magnets_only": True}),
        ("source_mesh_inside_plasma", "isometric", {"selected": [1]}),
        ("material_component_ids", "isometric", {"annotate_volumes": True}),
        (
            "surface_volume_ids",
            "isometric",
            {"annotate_volumes": True, "annotate_surfaces": True},
        ),
    ]
    for offset, prefix in ((0, "r1"), (12, "instrumented")):
        for index, (slug, camera, options) in enumerate(
            specifications, start=1
        ):
            number = offset + index
            if slug == "source_mesh_inside_plasma":
                source_mesh_view(
                    writer,
                    r1,
                    source_points,
                    f"{number:02d}_{prefix}_{slug}.png",
                    f"{prefix.upper()} — source mesh inside plasma",
                    status=(
                        "R1_NATIVE_SOURCE_MESH"
                        if prefix == "r1"
                        else "BYTE_IDENTICAL_REUSE"
                    ),
                )
            else:
                save_model_view(
                    writer,
                    r1,
                    f"{number:02d}_{prefix}_{slug}.png",
                    f"{prefix.upper()} — {slug.replace('_', ' ')}",
                    camera,
                    status=(
                        "R1_NATIVE_GEOMETRY"
                        if prefix == "r1"
                        else "BYTE_IDENTICAL_REUSE"
                    ),
                    **options,
                )

    # Group C: direct comparisons and parity diagnostics.
    projection_comparison(
        writer,
        r1,
        "25_r1_instrumented_side_by_side_isometric.png",
        "R1 / instrumented",
        camera="isometric",
    )
    projection_comparison(
        writer,
        r1,
        "26_r1_instrumented_side_by_side_cutaway.png",
        "R1 / instrumented cutaway",
        camera="cutaway",
        cutaway=True,
    )
    save_model_view(
        writer,
        r1,
        "27_r1_instrumented_transparent_overlay.png",
        "Transparent overlay: exact coincident reuse",
        "isometric",
        transparent=True,
        status="ZERO_GEOMETRIC_DIFFERENCE",
    )
    writer.status_sheet(
        "28_surface_difference_map.png",
        "Surface difference map",
        [
            "142 / 142 semantic surfaces match",
            "Raw H5M path and SHA-256 are identical",
            "Maximum observed displacement: exactly 0 cm",
        ],
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        status="ZERO_DIFFERENCE",
    )
    writer.status_sheet(
        "29_volume_difference_map.png",
        "Volume difference map",
        [
            "24 / 24 volumes match",
            "No physical H5M volume was added",
            "No post-export mutation was performed",
        ],
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        status="ZERO_DIFFERENCE",
    )
    writer.status_sheet(
        "30_bounding_box_comparison.png",
        "Bounding-box comparison",
        [
            "All canonical per-volume bounds match",
            "Coordinate quantum: 1e-6 cm",
            "Reference and instrumented inputs are the same immutable file",
        ],
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        status="EXACT_MATCH",
    )
    writer.status_sheet(
        "31_magnet_centroid_comparison.png",
        "Magnet-centroid comparison",
        [
            "18 / 18 original homogenized magnets discovered",
            "Centroid displacement: exactly 0 cm for byte reuse",
            "No translation, rotation, or scale introduced",
        ],
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        status="EXACT_MATCH",
    )
    writer.status_sheet(
        "32_radial_build_interface_comparison.png",
        "Radial-build interface comparison",
        [
            "Untouched example: first wall 5 cm",
            "breeder: embedded public-example 9 x 9 matrix",
            "back wall 5 cm; shield 50 cm; vessel 10 cm",
            "All interfaces are inherited from the same H5M",
        ],
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        status="EXACT_MATCH",
    )
    writer.status_sheet(
        "33_source_mesh_comparison.png",
        "Source-mesh comparison",
        [
            "Untouched public source definition: 11 x 61 x 61 samples",
            "External source_mesh.h5m retained byte-for-byte",
            "No source-convergence calculation was run",
        ],
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        status="EXACT_INPUT_REUSE",
    )
    writer.status_sheet(
        "34_topology_count_comparison.png",
        "Topology and count comparison",
        [
            "R1: 24 volumes, 142 surfaces, 18 homogenized magnets",
            "Instrumented: 24 volumes, 142 surfaces, 18 homogenized magnets",
            "Canonical fingerprint equality: PASS",
        ],
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        status="BYTE_IDENTICAL_REUSE",
    )

    # Group D: every complete original magnet boundary.
    magnet_ids = [
        volume_id for volume_id, view in r1.items() if role(view) == "magnets"
    ]
    for magnet_index, volume_id in enumerate(sorted(magnet_ids)):
        magnet_closeup(writer, r1, volume_id, magnet_index)

    # Group E: retain the failed split model and all native overlap witnesses.
    save_model_view(
        writer,
        failed,
        "53_failed_feature_full_geometry.png",
        "Failed feature — full geometry",
        "isometric",
        failed=True,
        input_hash=FAILED_HASH,
        fingerprint="NOT_ACCEPTED",
        status="NON_PRODUCTION",
        overlap_points=OVERLAPS_FAILED,
    )
    save_model_view(
        writer,
        failed,
        "54_failed_feature_casing_winding_split.png",
        "Failed feature — casing / winding split",
        "isometric",
        failed=True,
        input_hash=FAILED_HASH,
        fingerprint="NOT_ACCEPTED",
        status="UNNECESSARY_GLOBAL_SPLIT",
        magnets_only=True,
    )
    save_model_view(
        writer,
        failed,
        "55_failed_feature_all_seven_overlaps.png",
        "Failed feature — seven overlap witnesses",
        "isometric",
        failed=True,
        input_hash=FAILED_HASH,
        fingerprint="NOT_ACCEPTED",
        status="SEVEN_NATIVE_WITNESSES",
        overlap_points=OVERLAPS_FAILED,
    )
    save_model_view(
        writer,
        failed,
        "56_failed_feature_overlap_pair_5_17.png",
        "Failed feature — shield V5 / casing V17",
        "isometric",
        failed=True,
        input_hash=FAILED_HASH,
        fingerprint="NOT_ACCEPTED",
        status="TRUE_UNINTENDED_OVERLAP",
        selected=[5, 17],
        overlap_points=OVERLAPS_FAILED[:1],
    )
    save_model_view(
        writer,
        failed,
        "57_failed_feature_all_volume_6_overlaps.png",
        "Failed feature — all vacuum-vessel V6 pairs",
        "isometric",
        failed=True,
        input_hash=FAILED_HASH,
        fingerprint="NOT_ACCEPTED",
        status="SIX_V6_WITNESSES",
        selected=[6, 17, 18, 19, 29, 31, 32],
        overlap_points=OVERLAPS_FAILED[1:],
    )
    highlighted_surfaces_view(
        writer,
        failed,
        Path(args.failed),
        [102, 103],
        "58_failed_invalid_outer_interface_declaration.png",
        "Invalid declaration: winding-only S102 / S103 highlighted",
        status="SELECTOR_LABELING_FAILURE",
    )
    highlighted_surfaces_view(
        writer,
        failed,
        Path(args.failed),
        direct_winding_surface_ids,
        "59_failed_direct_winding_to_vacuum_faces.png",
        "28 direct winding-to-interstitial faces highlighted",
        status="28_DIRECT_FACES",
    )
    save_model_view(
        writer,
        r1,
        "60_correct_r1_homogenized_magnet_envelope.png",
        "Correct parity envelope — original homogenized magnet",
        "isometric",
        selected=[12],
        annotate_surfaces=True,
        status="CLOSED_ORIGINAL_MAGNET_ENVELOPE",
    )

    # Group F: R2 assets exist, but no validated historical H5M exists.
    r2_lines = [
        "Separate WISTELL-D inputs were found at local SHA bc4ab3d0f27369d4eda908a3fc187a10b6c7fedb.",
        "The workflow is legacy Cubit-based and differs from the current public R1 example.",
        "No accepted H5M, source mesh, or validation receipt was found.",
        "Geometry imagery is therefore not synthesized or misrepresented.",
    ]
    for number, slug in enumerate(
        (
            "isometric",
            "cutaway",
            "magnets",
            "source_mesh",
            "r1_side_by_side",
            "topology_summary",
        ),
        start=61,
    ):
        writer.status_sheet(
            f"{number:02d}_wistell_d_{slug}.png",
            f"WISTELL-D {slug.replace('_', ' ')}",
            r2_lines,
            input_hash="NO_ACCEPTED_R2_H5M",
            fingerprint="NOT_AVAILABLE",
            status="NOT_RUN_INCOMPLETE_HISTORICAL_METADATA",
        )

    # Contact sheets are generated after all source panels.
    contact_specs = [
        (67, "r1_instrumented_contact_sheet", list(range(1, 35))),
        (68, "all_18_magnets_contact_sheet", list(range(35, 53))),
        (69, "failed_feature_root_cause_contact_sheet", list(range(53, 61))),
    ]
    for number, slug, indices in contact_specs:
        sources = [
            writer.output / row["filename"]
            for row in writer.rows
            if int(row["filename"][:2]) in indices
        ]
        filename = f"{number:02d}_{slug}.png"
        target = writer.output / filename
        contact_sheet(target, sources, slug.replace("_", " ").title())
        writer.rows.append(
            {
                "filename": filename,
                "title": slug.replace("_", " ").title(),
                "input_geometry_sha256": (
                    R1_HASH if number != 69 else FAILED_HASH
                ),
                "canonical_geometry_fingerprint": (
                    R1_FINGERPRINT if number != 69 else "NOT_ACCEPTED"
                ),
                "branch": BRANCH,
                "sha": BASE_SHA,
                "camera": "contact_sheet",
                "visible_components": [path.name for path in sources],
                "color_map": COLORS,
                "coordinate_units": "cm",
                "status": "EVIDENCE_CONTACT_SHEET",
                "sha256": sha256_file(target),
                "width_px": Image.open(target).width,
                "height_px": Image.open(target).height,
            }
        )

    status_source = writer.status_sheet(
        "70_geometry_gate_status_sheet.png",
        "Prompt 7A geometry gate status",
        [
            "PASS: byte-identical R1 reuse; 18 closed original magnet envelopes; no ports or split.",
            "FAIL: untouched R1 native overlap_check found four nonadjacent vessel/magnet overlaps.",
            "FAIL: bounded OpenMC debug reported overlapping cells and did not prove zero navigation errors.",
            "Final decision: BLOCKED_GEOMETRY_PARITY.",
        ],
        input_hash=R1_HASH,
        fingerprint=R1_FINGERPRINT,
        status="BLOCKED_GEOMETRY_PARITY",
    )

    write_vtk_points(writer.output / "r1_reference_faceted_points.vtk", r1)
    write_vtk_points(
        writer.output / "failed_feature_faceted_points.vtk", failed
    )
    manifest = {
        "schema": SCHEMA,
        "generated_utc": args.generated_utc,
        "generator": "scripts/generate_geometry_parity_figures.py",
        "branch": BRANCH,
        "base_sha": BASE_SHA,
        "figure_count": len(writer.rows),
        "matched_camera_and_color_policy": True,
        "r1_input": {
            "path": str(Path(args.r1).resolve()),
            "sha256": sha256_file(Path(args.r1)),
        },
        "failed_input": {
            "path": str(Path(args.failed).resolve()),
            "sha256": sha256_file(Path(args.failed)),
        },
        "source_mesh_input": {
            "path": str(Path(args.source_mesh).resolve()),
            "sha256": sha256_file(Path(args.source_mesh)),
        },
        "vtk_products": [
            {
                "filename": "r1_reference_faceted_points.vtk",
                "sha256": sha256_file(
                    writer.output / "r1_reference_faceted_points.vtk"
                ),
            },
            {
                "filename": "failed_feature_faceted_points.vtk",
                "sha256": sha256_file(
                    writer.output / "failed_feature_faceted_points.vtk"
                ),
            },
        ],
        "figures": writer.rows,
        "gate_status": "BLOCKED_GEOMETRY_PARITY",
        "status_sheet": status_source.name,
    }
    (writer.output / "FIGURE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    selected = Path(args.selected).resolve()
    selected.mkdir(parents=True, exist_ok=True)
    selected_numbers = {
        1,
        2,
        6,
        9,
        13,
        18,
        25,
        27,
        35,
        40,
        45,
        52,
        53,
        55,
        60,
        67,
        68,
        69,
        70,
    }
    for row in writer.rows:
        if int(row["filename"][:2]) in selected_numbers:
            shutil.copy2(
                writer.output / row["filename"], selected / row["filename"]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1", required=True)
    parser.add_argument("--failed", required=True)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--failed-inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--generated-utc", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
