"""Artifact export and local-frame rendering for native port meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pymoab import core, types


SURFACE_COLORS = {
    "radial_surface": "#8dbb78",
    "sector_cap": "#777777",
    "void_liner_interface": "#00d5ff",
    "liner_blanket_interface": "#f28e2b",
    "void_blanket_interface": "#00d5ff",
    "plasma_connection": "#4ea3ff",
    "blind_termination": "#8a2be2",
    "external_termination": "#d45087",
    "liner_termination": "#f28e2b",
}


def _loaded_surface_triangles(complex_, filename):
    mb = core.Core()
    mb.load_file(str(filename))
    root = mb.get_root_set()
    dimension = mb.tag_get_handle("GEOM_DIMENSION")
    global_id = mb.tag_get_handle("GLOBAL_ID")
    by_id = {}
    for meshset in mb.get_entities_by_type(root, types.MBENTITYSET):
        try:
            dim = int(mb.tag_get_data(dimension, [meshset], flat=True)[0])
        except RuntimeError:
            continue
        if dim != 2:
            continue
        surface_id = int(mb.tag_get_data(global_id, [meshset], flat=True)[0])
        triangles = mb.get_entities_by_type(meshset, types.MBTRI)
        by_id[surface_id] = np.asarray(
            [
                mb.get_coords(mb.get_connectivity(triangle)).reshape((3, 3))
                for triangle in triangles
            ]
        )
    return [by_id[index] for index in range(1, len(complex_.surfaces) + 1)]


def _local_points(points, anchor, axis, reference, normal):
    relative = np.asarray(points) - anchor
    return np.stack(
        (
            relative @ axis,
            relative @ reference,
            relative @ normal,
        ),
        axis=-1,
    )


def _plane_segments(triangles, coordinate, plane_value, tolerance=1e-8):
    segments = []
    for triangle in triangles:
        values = triangle[:, coordinate] - plane_value
        points = []
        for left, right in ((0, 1), (1, 2), (2, 0)):
            a, b = values[left], values[right]
            if abs(a) <= tolerance:
                points.append(triangle[left])
            if a * b < -(tolerance**2):
                fraction = -a / (b - a)
                points.append(
                    triangle[left]
                    + fraction * (triangle[right] - triangle[left])
                )
        unique = []
        for point in points:
            if not any(
                np.linalg.norm(point - other) <= tolerance for other in unique
            ):
                unique.append(point)
        if len(unique) == 2:
            segments.append(np.asarray(unique))
    return segments


def render_native_dagmc(complex_, h5m_filename, output_dir):
    """Render local views from facets independently reloaded from H5M."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    port = complex_.port
    anchor = np.asarray(port.placement.anchor)
    axis = np.asarray(port.placement.local_axis)
    reference = np.asarray(port.placement.local_reference)
    normal = np.asarray(port.placement.local_normal)
    loaded = _loaded_surface_triangles(complex_, h5m_filename)
    local = [
        _local_points(triangles, anchor, axis, reference, normal)
        for triangles in loaded
    ]
    aperture = (
        port.cross_section.radius
        if port.cross_section.shape == "circle"
        else max(port.cross_section.width, port.cross_section.height) / 2.0
    )
    radial_limit = max(4.0 * (aperture + port.liner.thickness), 18.0)
    start = complex_.radial_data["port_result"].resolved_start
    end = complex_.radial_data["port_result"].resolved_end
    w_limits = (start - 8.0, end + port.extent.outer_extension + 8.0)

    def decorate_2d(ax, title, xlabel, ylabel):
        ax.set_title(f"{title} — {port.name} (model units: cm)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="port centerline",
        )
        ax.axvline(
            start, color="#0066cc", linestyle=":", linewidth=1.5, label="start"
        )
        ax.axvline(
            end, color="#cc0066", linestyle=":", linewidth=1.5, label="end"
        )
        ax.grid(alpha=0.2)

    figure, ax = plt.subplots(figsize=(14, 9), dpi=100)
    for record, triangles in zip(complex_.surfaces, local):
        segments = _plane_segments(triangles, 2, 0.0)
        if segments:
            ax.add_collection(
                LineCollection(
                    [segment[:, [0, 1]] for segment in segments],
                    colors=SURFACE_COLORS.get(record.kind, "#555555"),
                    linewidths=1.2,
                    label=record.kind,
                )
            )
    decorate_2d(ax, "Native DAGMC longitudinal section", "w (axis)", "u")
    ax.set_xlim(*w_limits)
    ax.set_ylim(-radial_limit, radial_limit)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    axis_path = output_dir / "native_dagmc_port_axis_section.png"
    figure.tight_layout()
    figure.savefig(axis_path)
    plt.close(figure)

    plane_w = float(
        np.median(
            [
                np.mean((loop.inner_points[:-1] - anchor) @ axis)
                for loop in complex_.loops[1:-1]
            ]
        )
    )
    figure, ax = plt.subplots(figsize=(14, 9), dpi=100)
    for record, triangles in zip(complex_.surfaces, local):
        segments = _plane_segments(triangles, 0, plane_w)
        if segments:
            ax.add_collection(
                LineCollection(
                    [segment[:, [1, 2]] for segment in segments],
                    colors=SURFACE_COLORS.get(record.kind, "#555555"),
                    linewidths=1.5,
                    label=record.kind,
                )
            )
    ax.scatter(
        [0.0], [0.0], color="black", marker="+", s=120, label="port centerline"
    )
    ax.set_title(
        f"Native DAGMC transverse section at w={plane_w:.3f} — {port.name} (cm)"
    )
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_aspect("equal")
    ax.set_xlim(-radial_limit, radial_limit)
    ax.set_ylim(-radial_limit, radial_limit)
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    transverse_path = output_dir / "native_dagmc_port_transverse.png"
    figure.tight_layout()
    figure.savefig(transverse_path)
    plt.close(figure)

    figure = plt.figure(figsize=(14, 9), dpi=100)
    ax = figure.add_subplot(111, projection="3d")
    for record, triangles in zip(complex_.surfaces, local):
        centers = triangles.mean(axis=1)
        keep = (
            (centers[:, 0] >= w_limits[0])
            & (centers[:, 0] <= w_limits[1])
            & np.all(np.abs(triangles[:, :, 1]) <= radial_limit, axis=1)
            & np.all(np.abs(triangles[:, :, 2]) <= radial_limit, axis=1)
        )
        if np.any(keep):
            ax.add_collection3d(
                Poly3DCollection(
                    triangles[keep],
                    facecolor=SURFACE_COLORS.get(record.kind, "#777777"),
                    edgecolor="none",
                    alpha=0.35 if record.kind == "radial_surface" else 0.85,
                )
            )
    ax.plot([w_limits[0], w_limits[1]], [0, 0], [0, 0], "k--", linewidth=2)
    ax.scatter([start, end], [0, 0], [0, 0], c=["#0066cc", "#cc0066"], s=60)
    ax.set_xlim(*w_limits)
    ax.set_ylim(-radial_limit, radial_limit)
    ax.set_zlim(-radial_limit, radial_limit)
    ax.set_xlabel("w (axis)")
    ax.set_ylabel("u")
    ax.set_zlabel("v")
    ax.set_title(f"Native DAGMC local faceted model — {port.name} (cm)")
    ax.view_init(elev=24, azim=-58)
    iso_path = output_dir / "native_dagmc_port_isometric.png"
    figure.tight_layout()
    figure.savefig(iso_path)
    plt.close(figure)
    return (iso_path, axis_path, transverse_path)


def export_native_port_artifacts(
    model,
    output_dir,
    *,
    tetrahedralize=True,
    min_mesh_size=5.0,
    max_mesh_size=25.0,
):
    """Export and validate native DAGMC and discrete-PLC volume artifacts."""
    if not hasattr(model, "native_port_complex"):
        raise ValueError("Generate native PyDAGMC components before export")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complex_ = model.native_port_complex
    dagmc_path = complex_.write_dagmc(
        output_dir / "ported_sector_native_dagmc.h5m"
    )
    complex_.dag_model.mb.write_file(str(dagmc_path.with_suffix(".vtk")))
    complex_.dag_model.volumes_by_id[
        complex_.volume_ids[complex_.port.name + "__void"]
    ].to_vtk(str(output_dir / "port_void_native_dagmc.vtk"))
    if complex_.port.liner.enabled:
        complex_.dag_model.volumes_by_id[
            complex_.volume_ids[complex_.port.name + "__liner"]
        ].to_vtk(str(output_dir / "port_liner_native_dagmc.vtk"))
    structural = complex_.validate()
    file_audit = complex_.validate_dagmc_file(dagmc_path)
    complex_.write_validation(
        output_dir / "native_dagmc_validation.json", structural
    )
    ledger = {
        "source": "point_cloud_aperture_loops",
        "volumes": [record.__dict__ for record in complex_.volumes],
        "surfaces": [
            {
                "name": record.name,
                "kind": record.kind,
                "global_id": complex_.surface_ids[record.name],
                "reverse_volume": record.reverse_volume,
                "forward_volume": record.forward_volume,
                "triangle_count": record.triangle_count,
                "bounding_box": [
                    record.triangles.min(axis=(0, 1)).tolist(),
                    record.triangles.max(axis=(0, 1)).tolist(),
                ],
            }
            for record in complex_.surfaces
        ],
        "file_audit": file_audit,
    }
    ledger_path = output_dir / "native_dagmc_component_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    images = render_native_dagmc(complex_, dagmc_path, output_dir)
    result = {
        "dagmc_h5m": str(dagmc_path),
        "dagmc_vtk": str(dagmc_path.with_suffix(".vtk")),
        "ledger": str(ledger_path),
        "images": [str(path) for path in images],
        "dagmc_validation": structural.to_dict(),
        "dagmc_file_audit": file_audit,
    }
    if tetrahedralize:
        mesh = complex_.tetrahedralize(min_mesh_size, max_mesh_size)
        mesh_validation = mesh.validate()
        mesh_path = mesh.write(
            output_dir / "ported_sector_native_volume_mesh.h5m"
        )
        mesh_file_audit = mesh.validate_file(mesh_path)
        mesh_validation_path = (
            output_dir / "native_volume_mesh_validation.json"
        )
        mesh_validation_path.write_text(
            json.dumps(mesh_validation.to_dict(), indent=2) + "\n"
        )
        result.update(
            {
                "volume_mesh_h5m": str(mesh_path),
                "volume_mesh_vtk": str(mesh_path.with_suffix(".vtk")),
                "volume_mesh_validation": mesh_validation.to_dict(),
                "volume_mesh_file_audit": mesh_file_audit,
            }
        )
    manifest_path = output_dir / "native_port_artifact_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    result["manifest"] = str(manifest_path)
    return result
