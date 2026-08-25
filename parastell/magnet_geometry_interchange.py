"""Versioned, consumer-neutral magnet geometry interchange contract.

The contract exposes engineering geometry needed to construct downstream
explicit-layer models without exporting ParaStell implementation objects.  Its
parallel-transport frame is an engineering frame only; it never represents
the unresolved twist of an individual conductor or REBCO tape.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .coil_frame import CentrelineFrame


SCHEMA_NAME = "parastell.magnet_geometry_interchange"
SCHEMA_VERSION = "1.0.0"
SCHEMA_URI = f"{SCHEMA_NAME}/v{SCHEMA_VERSION}"
FRAME_KIND = "engineering_parallel_transport"
ARTIFACT_NAMES = (
    "outer_step",
    "outer_stl",
    "outer_h5m",
    "winding_pack_step",
    "winding_pack_stl",
)
_SEGMENT_KINDS = {
    "full_filament",
    "field_period_segment",
    "sector_segment",
    "explicit_open_segment",
}
_HEX = frozenset("0123456789abcdef")


def _require_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_fields(
    value: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    result = _require_object(value, label)
    missing = expected - set(result)
    unexpected = set(result) - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise ValueError(f"{label} has " + " and ".join(details))
    return result


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a nonnegative integer") from exc
    if result != value or result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _require_sha256(value: Any, label: str) -> str:
    text = _require_text(value, label)
    if (
        len(text) != 64
        or text != text.lower()
        or any(c not in _HEX for c in text)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _vector(value: Any, label: str, *, size: int = 3) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite {size}-vector") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite {size}-vector")
    return result


def _unit_vector(value: Any, label: str) -> np.ndarray:
    result = _vector(value, label)
    if not np.isclose(np.linalg.norm(result), 1.0, atol=1.0e-10):
        raise ValueError(f"{label} must be a unit vector")
    return result


def _rigid_transform(value: Any, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite 4-by-4 matrix") from exc
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 4-by-4 matrix")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12):
        raise ValueError(f"{label} must be a homogeneous transform")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-10):
        raise ValueError(f"{label} rotation must be right handed")
    return result


def _validate_units(value: Any, label: str) -> None:
    units = _require_exact_fields(value, {"length", "angle"}, label)
    if units["length"] != "cm" or units["angle"] != "rad":
        raise ValueError(f"{label} must use cm and rad")


def _validate_polygon(value: Any, label: str) -> None:
    polygon = _require_exact_fields(
        value,
        {"definition_kind", "coordinate_axes", "vertices_local_cm", "closed"},
        label,
    )
    if polygon["definition_kind"] != "polygon":
        raise ValueError(f"{label} definition_kind must be polygon")
    if polygon["coordinate_axes"] != ["width_direction", "normal_direction"]:
        raise ValueError(
            f"{label} coordinate_axes must be width_direction, normal_direction"
        )
    if polygon["closed"] is not True:
        raise ValueError(f"{label} must be closed")
    try:
        vertices = np.asarray(polygon["vertices_local_cm"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} vertices must be finite two-vectors"
        ) from exc
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (2,)
        or len(vertices) < 3
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError(
            f"{label} vertices must contain at least three finite points"
        )
    if np.allclose(vertices[0], vertices[-1]):
        raise ValueError(
            f"{label} must not repeat its implicit closing vertex"
        )
    x, y = vertices[:, 0], vertices[:, 1]
    area_twice = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    if abs(float(area_twice)) <= 1.0e-14:
        raise ValueError(f"{label} polygon has zero area")


def _validate_artifact(value: Any, label: str) -> None:
    artifact = _require_object(value, label)
    status = artifact.get("status")
    if status == "available":
        artifact = _require_exact_fields(
            artifact, {"status", "sha256", "size_bytes"}, label
        )
        _require_sha256(artifact["sha256"], f"{label}.sha256")
        if (
            _require_nonnegative_integer(
                artifact["size_bytes"], f"{label}.size_bytes"
            )
            <= 0
        ):
            raise ValueError(f"{label}.size_bytes must be positive")
    elif status == "unavailable":
        artifact = _require_exact_fields(artifact, {"status", "reason"}, label)
        _require_text(artifact["reason"], f"{label}.reason")
    else:
        raise ValueError(f"{label}.status must be available or unavailable")


def _validate_closure(value: Any, label: str) -> tuple[bool, bool, str]:
    closure = _require_exact_fields(
        value,
        {
            "source_filament_closed",
            "represented_segment_closed",
            "represented_segment_kind",
        },
        label,
    )
    source_closed = _require_bool(
        closure["source_filament_closed"], f"{label}.source_filament_closed"
    )
    segment_closed = _require_bool(
        closure["represented_segment_closed"],
        f"{label}.represented_segment_closed",
    )
    kind = closure["represented_segment_kind"]
    if kind not in _SEGMENT_KINDS:
        raise ValueError(f"{label}.represented_segment_kind is unsupported")
    if segment_closed and not source_closed:
        raise ValueError(
            "an open source filament cannot produce a closed segment"
        )
    if kind == "full_filament" and segment_closed != source_closed:
        raise ValueError(
            "full_filament closure must match source-filament closure"
        )
    if kind != "full_filament" and segment_closed:
        raise ValueError(
            "a field-period, sector, or explicit segment must be open"
        )
    return source_closed, segment_closed, str(kind)


def _validate_centreline(
    value: Any, label: str, *, represented_closed: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    centreline = _require_exact_fields(
        value,
        {
            "definition_kind",
            "points_global_cm",
            "arc_length_cm",
            "total_length_cm",
            "represented_segment_closed",
        },
        label,
    )
    if centreline["definition_kind"] != "polyline":
        raise ValueError(f"{label}.definition_kind must be polyline")
    if centreline["represented_segment_closed"] is not represented_closed:
        raise ValueError(f"{label} closure disagrees with closure contract")
    try:
        points = np.asarray(centreline["points_global_cm"], dtype=float)
        arc = np.asarray(centreline["arc_length_cm"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} points and arc length must be finite"
        ) from exc
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(f"{label} must contain at least three finite points")
    if arc.shape != (len(points),) or not np.all(np.isfinite(arc)):
        raise ValueError(f"{label}.arc_length_cm must align with its points")
    if arc[0] != 0.0 or np.any(np.diff(arc) <= 0.0):
        raise ValueError(
            f"{label}.arc_length_cm must start at zero and increase"
        )
    edge_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(edge_lengths <= 1.0e-12):
        raise ValueError(f"{label} contains duplicate adjacent points")
    expected_arc = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    if not np.allclose(arc, expected_arc, rtol=1.0e-12, atol=1.0e-10):
        raise ValueError(
            f"{label}.arc_length_cm disagrees with the centreline polyline"
        )
    total = _require_float(
        centreline["total_length_cm"], f"{label}.total_length_cm"
    )
    expected_total = float(expected_arc[-1])
    if represented_closed:
        closing_length = float(np.linalg.norm(points[0] - points[-1]))
        if closing_length <= 1.0e-12:
            raise ValueError(
                f"{label} repeats its first point instead of using a closing edge"
            )
        expected_total += closing_length
    if not np.isclose(total, expected_total, rtol=1.0e-12, atol=1.0e-10):
        raise ValueError(
            f"{label}.total_length_cm disagrees with the centreline polyline"
        )
    return points, arc, total


def _validate_frame(
    value: Any,
    label: str,
    *,
    points: np.ndarray,
    arc: np.ndarray,
    total_length: float,
    represented_closed: bool,
) -> None:
    frame = _require_exact_fields(
        value,
        {
            "frame_kind",
            "tape_twist_resolved",
            "handedness",
            "continuity",
            "closure_twist_rad",
            "quality_status",
            "samples",
        },
        label,
    )
    if frame["frame_kind"] != FRAME_KIND:
        raise ValueError(f"{label}.frame_kind must be {FRAME_KIND}")
    if frame["tape_twist_resolved"] is not False:
        raise ValueError("engineering frame cannot claim resolved tape twist")
    if frame["handedness"] != "right" or frame["continuity"] != "continuous":
        raise ValueError(f"{label} must be continuous and right handed")
    _require_float(frame["closure_twist_rad"], f"{label}.closure_twist_rad")
    _require_text(frame["quality_status"], f"{label}.quality_status")
    samples = frame["samples"]
    if not isinstance(samples, list) or len(samples) != len(points):
        raise ValueError(f"{label}.samples must align with centreline points")
    tangent_rows = []
    width_rows = []
    normal_rows = []
    for index, sample_value in enumerate(samples):
        sample_label = f"{label}.samples[{index}]"
        sample = _require_exact_fields(
            sample_value,
            {
                "s_cm",
                "normalized_s",
                "point_global_cm",
                "tangent_global",
                "width_direction_global",
                "normal_direction_global",
            },
            sample_label,
        )
        s_cm = _require_float(sample["s_cm"], f"{sample_label}.s_cm")
        normalized = _require_float(
            sample["normalized_s"], f"{sample_label}.normalized_s"
        )
        if not np.isclose(s_cm, arc[index], atol=1.0e-10):
            raise ValueError(f"{sample_label}.s_cm disagrees with centreline")
        if not np.isclose(normalized, s_cm / total_length, atol=1.0e-12):
            raise ValueError(f"{sample_label}.normalized_s is inconsistent")
        point = _vector(
            sample["point_global_cm"], f"{sample_label}.point_global_cm"
        )
        if not np.allclose(point, points[index], atol=1.0e-10):
            raise ValueError(
                f"{sample_label}.point_global_cm disagrees with centreline"
            )
        tangent = _unit_vector(
            sample["tangent_global"], f"{sample_label}.tangent_global"
        )
        width = _unit_vector(
            sample["width_direction_global"],
            f"{sample_label}.width_direction_global",
        )
        normal = _unit_vector(
            sample["normal_direction_global"],
            f"{sample_label}.normal_direction_global",
        )
        if (
            abs(float(np.dot(tangent, width))) > 1.0e-10
            or abs(float(np.dot(tangent, normal))) > 1.0e-10
            or abs(float(np.dot(width, normal))) > 1.0e-10
            or not np.allclose(np.cross(tangent, width), normal, atol=1.0e-10)
        ):
            raise ValueError(
                f"{sample_label} is not a right-handed orthonormal frame"
            )
        tangent_rows.append(tangent)
        width_rows.append(width)
        normal_rows.append(normal)
    tangent_rows = np.asarray(tangent_rows)
    width_rows = np.asarray(width_rows)
    edges = (
        np.roll(points, -1, axis=0) - points
        if represented_closed
        else np.diff(points, axis=0)
    )
    unit_edges = edges / np.linalg.norm(edges, axis=1)[:, None]
    if represented_closed:
        tangent_sums = np.roll(unit_edges, 1, axis=0) + unit_edges
        tangent_norms = np.linalg.norm(tangent_sums, axis=1)
        if np.any(tangent_norms <= 1.0e-12):
            raise ValueError(f"{label} centreline contains a tangent reversal")
        expected_tangents = tangent_sums / tangent_norms[:, None]
    else:
        expected_tangents = np.empty_like(points)
        expected_tangents[0], expected_tangents[-1] = (
            unit_edges[0],
            unit_edges[-1],
        )
        if len(points) > 2:
            tangent_sums = unit_edges[:-1] + unit_edges[1:]
            tangent_norms = np.linalg.norm(tangent_sums, axis=1)
            if np.any(tangent_norms <= 1.0e-12):
                raise ValueError(
                    f"{label} centreline contains a tangent reversal"
                )
            expected_tangents[1:-1] = tangent_sums / tangent_norms[:, None]
    if not np.allclose(
        tangent_rows, expected_tangents, rtol=1.0e-12, atol=1.0e-10
    ):
        raise ValueError(
            f"{label} tangent samples disagree with the centreline polyline"
        )
    if len(samples) > 1:
        if np.any(
            np.sum(tangent_rows[:-1] * tangent_rows[1:], axis=1) <= -1.0
        ):
            raise ValueError(f"{label} contains a tangent reversal")
        if np.any(np.sum(width_rows[:-1] * width_rows[1:], axis=1) <= 0.0):
            raise ValueError(f"{label} width direction is not sign-continuous")
    if (
        represented_closed
        and float(np.dot(width_rows[-1], width_rows[0])) <= 0.0
    ):
        raise ValueError(f"{label} is not sign-continuous across closure")


def _validate_identifier_lists(magnet: Mapping[str, Any], label: str) -> None:
    surfaces = magnet["surface_ids"]
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError(f"{label}.surface_ids must be a nonempty list")
    normalised_surfaces = []
    for index, value in enumerate(surfaces):
        sid = _require_nonnegative_integer(
            value, f"{label}.surface_ids[{index}]"
        )
        if sid <= 0:
            raise ValueError(f"{label}.surface_ids must be positive")
        normalised_surfaces.append(sid)
    if len(normalised_surfaces) != len(set(normalised_surfaces)):
        raise ValueError(f"{label}.surface_ids contains duplicates")
    facets = magnet["facet_ids"]
    if not isinstance(facets, list):
        raise ValueError(f"{label}.facet_ids must be a list")
    normalised_facets = []
    for index, value in enumerate(facets):
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(
                f"{label}.facet_ids[{index}] must be a string or integer"
            )
        if isinstance(value, int) and value < 0:
            raise ValueError(f"{label}.facet_ids[{index}] cannot be negative")
        if isinstance(value, str) and not value:
            raise ValueError(f"{label}.facet_ids[{index}] cannot be empty")
        normalised_facets.append((type(value).__name__, value))
    if len(normalised_facets) != len(set(normalised_facets)):
        raise ValueError(f"{label}.facet_ids contains duplicates")


def _validate_components(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    names = []
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        component = _require_exact_fields(
            item, {"component_name", "material_name"}, item_label
        )
        names.append(
            _require_text(
                component["component_name"], f"{item_label}.component_name"
            )
        )
        _require_text(
            component["material_name"], f"{item_label}.material_name"
        )
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate component names")


def _validate_magnet(value: Any, index: int) -> None:
    label = f"magnets[{index}]"
    magnet = _require_exact_fields(
        value,
        {
            "magnet_id",
            "coil_filament_source_sha256",
            "closure",
            "native_global_transform",
            "centreline",
            "cross_sections",
            "frame",
            "field_period_sector_transform",
            "artifacts",
            "surface_ids",
            "facet_ids",
            "components",
            "units",
        },
        label,
    )
    _require_text(magnet["magnet_id"], f"{label}.magnet_id")
    _require_sha256(
        magnet["coil_filament_source_sha256"],
        f"{label}.coil_filament_source_sha256",
    )
    _, represented_closed, _ = _validate_closure(
        magnet["closure"], f"{label}.closure"
    )
    _rigid_transform(
        magnet["native_global_transform"], f"{label}.native_global_transform"
    )
    points, arc, total = _validate_centreline(
        magnet["centreline"],
        f"{label}.centreline",
        represented_closed=represented_closed,
    )
    cross_sections = _require_exact_fields(
        magnet["cross_sections"],
        {"outer_casing", "outer_winding_pack", "casing_thickness_cm"},
        f"{label}.cross_sections",
    )
    _validate_polygon(
        cross_sections["outer_casing"], f"{label}.cross_sections.outer_casing"
    )
    _validate_polygon(
        cross_sections["outer_winding_pack"],
        f"{label}.cross_sections.outer_winding_pack",
    )
    if (
        _require_float(
            cross_sections["casing_thickness_cm"],
            f"{label}.cross_sections.casing_thickness_cm",
        )
        <= 0.0
    ):
        raise ValueError(
            f"{label}.cross_sections.casing_thickness_cm must be positive"
        )
    _validate_frame(
        magnet["frame"],
        f"{label}.frame",
        points=points,
        arc=arc,
        total_length=total,
        represented_closed=represented_closed,
    )
    sector = _require_exact_fields(
        magnet["field_period_sector_transform"],
        {"field_period_index", "sector_index", "transform_to_native_global"},
        f"{label}.field_period_sector_transform",
    )
    _require_nonnegative_integer(
        sector["field_period_index"],
        f"{label}.field_period_sector_transform.field_period_index",
    )
    _require_nonnegative_integer(
        sector["sector_index"],
        f"{label}.field_period_sector_transform.sector_index",
    )
    _rigid_transform(
        sector["transform_to_native_global"],
        f"{label}.field_period_sector_transform.transform_to_native_global",
    )
    artifacts = _require_exact_fields(
        magnet["artifacts"], set(ARTIFACT_NAMES), f"{label}.artifacts"
    )
    for name in ARTIFACT_NAMES:
        _validate_artifact(artifacts[name], f"{label}.artifacts.{name}")
    _validate_identifier_lists(magnet, label)
    _validate_components(magnet["components"], f"{label}.components")
    _validate_units(magnet["units"], f"{label}.units")


def _content_sha256(document: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(document))
    payload.pop("content_sha256", None)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "geometry interchange must contain canonical JSON values"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_magnet_geometry_interchange(
    document: Mapping[str, Any], *, verify_content_hash: bool = True
) -> Mapping[str, Any]:
    """Validate every required interchange field and its hash binding."""
    expected = {
        "schema",
        "schema_version",
        "units",
        "geometry_fingerprint",
        "magnets",
        "provenance",
        "content_sha256",
    }
    interchange = _require_exact_fields(document, expected, "interchange")
    if interchange["schema"] != SCHEMA_URI:
        raise ValueError(f"interchange schema must be {SCHEMA_URI}")
    if interchange["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"interchange schema_version must be {SCHEMA_VERSION}"
        )
    _validate_units(interchange["units"], "interchange.units")
    _require_sha256(
        interchange["geometry_fingerprint"], "interchange.geometry_fingerprint"
    )
    magnets = interchange["magnets"]
    if not isinstance(magnets, list) or not magnets:
        raise ValueError("interchange.magnets must be a nonempty list")
    for index, magnet in enumerate(magnets):
        _validate_magnet(magnet, index)
    magnet_ids = [magnet["magnet_id"] for magnet in magnets]
    if len(magnet_ids) != len(set(magnet_ids)):
        raise ValueError("interchange contains duplicate magnet IDs")
    _require_object(interchange["provenance"], "interchange.provenance")
    expected_hash = _require_sha256(
        interchange["content_sha256"], "interchange.content_sha256"
    )
    if verify_content_hash and _content_sha256(interchange) != expected_hash:
        raise ValueError("geometry interchange content SHA-256 mismatch")
    return document


def build_magnet_geometry_record(
    *,
    magnet_id: str,
    coil_filament_source_sha256: str,
    source_filament_closed: bool,
    represented_segment_kind: str,
    centreline_frame: CentrelineFrame,
    native_global_transform: Sequence[Sequence[float]],
    field_period_index: int,
    sector_index: int,
    field_period_sector_transform: Sequence[Sequence[float]],
    outer_casing_cross_section_local_cm: Sequence[Sequence[float]],
    outer_winding_pack_cross_section_local_cm: Sequence[Sequence[float]],
    casing_thickness_cm: float,
    artifacts: Mapping[str, Mapping[str, Any]],
    surface_ids: Sequence[int],
    facet_ids: Sequence[int | str],
    components: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Translate a ParaStell centreline frame into one neutral magnet record."""
    if not isinstance(centreline_frame, CentrelineFrame):
        raise ValueError("centreline_frame must be a CentrelineFrame")
    point_count = len(centreline_frame.points_cm)
    arc = np.asarray(centreline_frame.arclength_cm[:point_count], dtype=float)
    total = float(centreline_frame.total_length_cm)
    samples = []
    for index in range(point_count):
        samples.append(
            {
                "s_cm": float(arc[index]),
                "normalized_s": float(arc[index] / total),
                "point_global_cm": centreline_frame.points_cm[index].tolist(),
                "tangent_global": centreline_frame.tangents[index].tolist(),
                "width_direction_global": centreline_frame.radial_directions[
                    index
                ].tolist(),
                "normal_direction_global": centreline_frame.transverse_directions[
                    index
                ].tolist(),
            }
        )
    polygon_common = {
        "definition_kind": "polygon",
        "coordinate_axes": ["width_direction", "normal_direction"],
        "closed": True,
    }
    record = {
        "magnet_id": magnet_id,
        "coil_filament_source_sha256": coil_filament_source_sha256,
        "closure": {
            "source_filament_closed": source_filament_closed,
            "represented_segment_closed": bool(centreline_frame.closed),
            "represented_segment_kind": represented_segment_kind,
        },
        "native_global_transform": np.asarray(
            native_global_transform, dtype=float
        ).tolist(),
        "centreline": {
            "definition_kind": "polyline",
            "points_global_cm": centreline_frame.points_cm.tolist(),
            "arc_length_cm": arc.tolist(),
            "total_length_cm": total,
            "represented_segment_closed": bool(centreline_frame.closed),
        },
        "cross_sections": {
            "outer_casing": {
                **polygon_common,
                "vertices_local_cm": np.asarray(
                    outer_casing_cross_section_local_cm, dtype=float
                ).tolist(),
            },
            "outer_winding_pack": {
                **polygon_common,
                "vertices_local_cm": np.asarray(
                    outer_winding_pack_cross_section_local_cm, dtype=float
                ).tolist(),
            },
            "casing_thickness_cm": float(casing_thickness_cm),
        },
        "frame": {
            "frame_kind": FRAME_KIND,
            "tape_twist_resolved": False,
            "handedness": "right",
            "continuity": "continuous",
            "closure_twist_rad": float(centreline_frame.closure_twist_rad),
            "quality_status": str(centreline_frame.quality_status),
            "samples": samples,
        },
        "field_period_sector_transform": {
            "field_period_index": field_period_index,
            "sector_index": sector_index,
            "transform_to_native_global": np.asarray(
                field_period_sector_transform, dtype=float
            ).tolist(),
        },
        "artifacts": deepcopy(dict(artifacts)),
        "surface_ids": list(surface_ids),
        "facet_ids": list(facet_ids),
        "components": [dict(item) for item in components],
        "units": {"length": "cm", "angle": "rad"},
    }
    _validate_magnet(record, 0)
    return record


def seal_magnet_geometry_interchange(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy with a deterministic hash binding over all content."""
    result = deepcopy(dict(document))
    result["content_sha256"] = "0" * 64
    validate_magnet_geometry_interchange(result, verify_content_hash=False)
    result["content_sha256"] = _content_sha256(result)
    validate_magnet_geometry_interchange(result)
    return result


def build_magnet_geometry_interchange(
    magnets: Sequence[Mapping[str, Any]],
    *,
    geometry_fingerprint: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and seal an interchange document from neutral magnet records."""
    return seal_magnet_geometry_interchange(
        {
            "schema": SCHEMA_URI,
            "schema_version": SCHEMA_VERSION,
            "units": {"length": "cm", "angle": "rad"},
            "geometry_fingerprint": geometry_fingerprint,
            "magnets": [deepcopy(dict(item)) for item in magnets],
            "provenance": deepcopy(dict(provenance)),
        }
    )


def write_magnet_geometry_interchange(
    path: str | Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Write an already sealed, validated interchange as canonical JSON."""
    validate_magnet_geometry_interchange(document)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = deepcopy(dict(document))
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def read_magnet_geometry_interchange(path: str | Path) -> dict[str, Any]:
    """Read a geometry interchange, rejecting malformed or tampered content."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"geometry interchange is not valid JSON: {source}"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError("geometry interchange root must be an object")
    validate_magnet_geometry_interchange(document)
    return document


def _barycentric_coordinates(
    point: np.ndarray, triangle: np.ndarray
) -> tuple[np.ndarray, float]:
    if triangle.shape != (3, 3) or not np.all(np.isfinite(triangle)):
        raise ValueError("facet_vertices_global_cm must have shape (3, 3)")
    first = triangle[1] - triangle[0]
    second = triangle[2] - triangle[0]
    gram = np.asarray(
        [
            [np.dot(first, first), np.dot(first, second)],
            [np.dot(first, second), np.dot(second, second)],
        ]
    )
    if abs(float(np.linalg.det(gram))) <= 1.0e-20:
        raise ValueError("facet triangle is degenerate")
    right = np.asarray(
        [
            np.dot(point - triangle[0], first),
            np.dot(point - triangle[0], second),
        ]
    )
    second_weight, third_weight = np.linalg.solve(gram, right)
    weights = np.asarray(
        [1.0 - second_weight - third_weight, second_weight, third_weight]
    )
    projection = weights @ triangle
    return weights, float(np.linalg.norm(point - projection))


def derive_boundary_geometry_fields(
    record: Mapping[str, Any],
    *,
    centreline_frame: CentrelineFrame | None = None,
    facet_vertices_global_cm: Sequence[Sequence[float]] | None = None,
    tolerance_cm: float = 1.0e-7,
) -> dict[str, Any]:
    """Derive neutral facet and local-frame fields for one boundary record.

    Direction vectors are expressed in the global basis; the ``local_*`` names
    identify their engineering role.  Existing producer field names are
    accepted as aliases.  Facet IDs and missing geometry are never fabricated.
    """
    result = deepcopy(dict(_require_object(record, "boundary record")))
    position = _vector(result.get("position_global_cm"), "position_global_cm")
    if centreline_frame is not None:
        sampled = centreline_frame.sample(position)
        result.update(
            {
                "centreline_arclength_cm": float(
                    sampled["centreline_arclength_cm"]
                ),
                "normalized_arclength": float(sampled["normalized_arclength"]),
                "local_centreline_coordinates_cm": np.asarray(
                    sampled["local_centreline_coordinates_cm"]
                ).tolist(),
                "centreline_tangent": np.asarray(
                    sampled["centreline_tangent"]
                ).tolist(),
                "centreline_radial": np.asarray(
                    sampled["centreline_radial"]
                ).tolist(),
                "centreline_transverse": np.asarray(
                    sampled["centreline_transverse"]
                ).tolist(),
            }
        )
    required = (
        "centreline_arclength_cm",
        "local_centreline_coordinates_cm",
        "centreline_tangent",
        "centreline_radial",
        "centreline_transverse",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"boundary record lacks centreline fields {missing}")
    s_cm = _require_float(
        result["centreline_arclength_cm"], "centreline_arclength_cm"
    )
    local = _vector(
        result["local_centreline_coordinates_cm"],
        "local_centreline_coordinates_cm",
    )
    tangent = _unit_vector(result["centreline_tangent"], "centreline_tangent")
    width = _unit_vector(result["centreline_radial"], "centreline_radial")
    normal = _unit_vector(
        result["centreline_transverse"], "centreline_transverse"
    )
    if not np.allclose(np.cross(tangent, width), normal, atol=1.0e-10):
        raise ValueError("boundary engineering frame is not right handed")
    result.update(
        {
            "s_cm": s_cm,
            "local_cross_section_coordinates_cm": [
                float(local[1]),
                float(local[2]),
            ],
            "local_tangent": tangent.tolist(),
            "local_width_direction": width.tolist(),
            "local_normal_direction": normal.tolist(),
            "frame_kind": FRAME_KIND,
            "tape_twist_resolved": False,
        }
    )
    tolerance = _require_float(tolerance_cm, "tolerance_cm")
    if tolerance <= 0.0:
        raise ValueError("tolerance_cm must be positive")
    if facet_vertices_global_cm is not None:
        if "facet_id" not in result:
            raise ValueError(
                "facet_id is required when deriving barycentric coordinates"
            )
        triangle = np.asarray(facet_vertices_global_cm, dtype=float)
        barycentric, residual = _barycentric_coordinates(position, triangle)
        if residual > tolerance or np.any(barycentric < -1.0e-10):
            raise ValueError("boundary point is not on the supplied facet")
        result["barycentric_coordinates"] = barycentric.tolist()
        result["distance_to_facet_residual_cm"] = residual
        result["facet_mapping_status"] = "MAPPED_BARYCENTRIC"
    elif "barycentric_coordinates" in result:
        barycentric = _vector(
            result["barycentric_coordinates"], "barycentric_coordinates"
        )
        if not np.isclose(barycentric.sum(), 1.0, atol=1.0e-10) or np.any(
            barycentric < -1.0e-10
        ):
            raise ValueError(
                "barycentric coordinates must be nonnegative and sum to one"
            )
        if "facet_id" not in result:
            raise ValueError("barycentric coordinates require facet_id")
    return result
