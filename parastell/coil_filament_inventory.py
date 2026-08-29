"""Hash-bound, geometry-neutral coil-filament inventories.

The inventory is deliberately separate from the accepted global DAGMC model.
It describes candidate local magnet centrelines; it never claims that those
filaments are physical volumes in a continuous radial-envelope global model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "parastell.coil_filament_inventory/v1.0.0"
PARSER_SCHEMA = "parastell.coil_filament_text_parser/v1.0.0"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coordinate_identity(coordinates_cm: Any) -> dict[str, Any]:
    """Return an order-preserving identity for one closed filament."""
    array = np.asarray(coordinates_cm, dtype="<f8")
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or len(array) < 4
        or not np.isfinite(array).all()
        or not np.array_equal(array[0], array[-1])
    ):
        raise ValueError("filament coordinates must be finite and closed")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    centroid = np.mean(array[:-1], axis=0)
    segments = np.diff(array, axis=0)
    return {
        "coordinates_sha256": digest.hexdigest(),
        "coordinate_count": len(array),
        "centroid_cm": centroid.tolist(),
        "centroid_toroidal_angle_degrees": float(
            np.degrees(np.arctan2(centroid[1], centroid[0])) % 360.0
        ),
        "bounding_box_cm": {
            "lower": np.min(array, axis=0).tolist(),
            "upper": np.max(array, axis=0).tolist(),
        },
        "centreline_length_cm": float(np.linalg.norm(segments, axis=1).sum()),
    }


def _parse_filaments(path: Path, *, start_line: int, scale_to_cm: float):
    lines = path.read_text(encoding="utf-8").splitlines()[start_line:]
    result: list[np.ndarray] = []
    current: list[list[float]] = []
    found_end = False
    for line_number, raw in enumerate(lines, start=start_line + 1):
        columns = raw.split()
        if not columns:
            continue
        if columns[0].lower() == "end":
            found_end = True
            break
        if len(columns) < 4:
            raise ValueError(
                f"coil line {line_number} has fewer than 4 fields"
            )
        try:
            xyz = [float(value) * scale_to_cm for value in columns[:3]]
            continuation = float(columns[3])
        except ValueError as exc:
            raise ValueError(
                f"coil line {line_number} is not numeric"
            ) from exc
        if not all(math.isfinite(value) for value in (*xyz, continuation)):
            raise ValueError(f"coil line {line_number} is not finite")
        if continuation != 0.0:
            current.append(xyz)
            continue
        if len(current) < 3:
            raise ValueError(
                f"filament ending at line {line_number} is too short"
            )
        current.append(current[0])
        result.append(np.asarray(current, dtype="<f8"))
        current = []
    if current:
        raise ValueError("coil file ends with an unterminated filament")
    if not found_end:
        raise ValueError("coil file lacks an end marker")
    if not result:
        raise ValueError("coil file contains no filaments")
    return result


def _angle_in_sector(angle: float, start: float, extent: float) -> bool:
    if math.isclose(extent, 360.0):
        return True
    delta = (angle - start) % 360.0
    return delta <= extent or math.isclose(delta, extent)


def build_coil_filament_inventory(
    coil_file: str | Path,
    *,
    start_line: int = 3,
    scale_to_cm: float = 100.0,
    input_coordinate_units: str = "m",
    sector_start_degrees: float = 0.0,
    sector_extent_degrees: float = 360.0,
) -> dict[str, Any]:
    """Parse every filament and return a deterministic, self-sealed inventory.

    Sector membership uses the filament centroid only and is descriptive. CAD
    clipping and finite-cross-section inclusion are intentionally deferred to
    the local geometry case.
    """
    path = Path(coil_file).resolve(strict=True)
    if (
        isinstance(start_line, bool)
        or not isinstance(start_line, int)
        or start_line < 0
    ):
        raise ValueError("start_line must be a nonnegative integer")
    controls = (
        float(scale_to_cm),
        float(sector_start_degrees),
        float(sector_extent_degrees),
    )
    if (
        any(not math.isfinite(value) for value in controls)
        or controls[0] <= 0.0
        or not 0.0 < controls[2] <= 360.0
        or not str(input_coordinate_units).strip()
    ):
        raise ValueError("coil parser units, scale, or sector are invalid")
    filaments = _parse_filaments(
        path, start_line=start_line, scale_to_cm=controls[0]
    )
    rows = []
    for source_ordinal, coordinates in enumerate(filaments):
        identity = coordinate_identity(coordinates)
        digest = identity["coordinates_sha256"]
        rows.append(
            {
                "filament_id": f"filament-{digest[:16]}",
                "source_ordinal": source_ordinal,
                **identity,
                "centroid_inside_declared_sector": _angle_in_sector(
                    identity["centroid_toroidal_angle_degrees"],
                    controls[1],
                    controls[2],
                ),
            }
        )
    hashes = [row["coordinates_sha256"] for row in rows]
    ids = [row["filament_id"] for row in rows]
    if len(set(hashes)) != len(rows) or len(set(ids)) != len(rows):
        raise ValueError("coil inventory has duplicated filament identities")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "coil_input": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "parser": {
            "schema": PARSER_SCHEMA,
            "start_line": start_line,
            "scale_to_cm": controls[0],
            "input_coordinate_units": str(input_coordinate_units),
            "output_coordinate_units": "cm",
            "termination_semantics": "fourth_field_zero_closes_with_first_point",
        },
        "sector": {
            "start_degrees": controls[1] % 360.0,
            "extent_degrees": controls[2],
            "membership_semantics": "centroid_only_no_cross_section_padding",
            "membership_is_geometry_selection": False,
        },
        "ordering": "source_file_ordinal",
        "filament_count": len(rows),
        "filaments": rows,
        "claims": {
            "physical_global_volumes": False,
            "continuous_global_layer_modified": False,
            "cad_constructed": False,
        },
    }
    payload["inventory_content_sha256"] = canonical_sha256(payload)
    return payload


def validate_coil_filament_inventory(
    inventory: Mapping[str, Any], *, expected_coil_sha256: str | None = None
) -> None:
    """Validate structure, self-seal, source hash, and unique identities."""
    value = dict(inventory)
    seal = value.pop("inventory_content_sha256", None)
    rows = inventory.get("filaments")
    coil = inventory.get("coil_input", {})
    parser = inventory.get("parser", {})
    sector = inventory.get("sector", {})
    if (
        inventory.get("schema") != SCHEMA
        or seal != canonical_sha256(value)
        or parser.get("schema") != PARSER_SCHEMA
        or parser.get("output_coordinate_units") != "cm"
        or sector.get("membership_is_geometry_selection") is not False
        or not isinstance(rows, list)
        or not rows
        or inventory.get("filament_count") != len(rows)
        or inventory.get("claims", {}).get("physical_global_volumes")
        is not False
    ):
        raise ValueError(
            "coil filament inventory structure or seal is invalid"
        )
    source_path = Path(str(coil.get("path", ""))).resolve(strict=True)
    source_hash = sha256_file(source_path)
    if (
        source_hash != coil.get("sha256")
        or source_path.stat().st_size != coil.get("bytes")
        or (
            expected_coil_sha256 is not None
            and source_hash != expected_coil_sha256
        )
    ):
        raise ValueError("coil filament inventory source binding is invalid")
    rebuilt = build_coil_filament_inventory(
        source_path,
        start_line=int(parser.get("start_line", -1)),
        scale_to_cm=float(parser.get("scale_to_cm", math.nan)),
        input_coordinate_units=str(parser.get("input_coordinate_units", "")),
        sector_start_degrees=float(sector.get("start_degrees", math.nan)),
        sector_extent_degrees=float(sector.get("extent_degrees", math.nan)),
    )
    if rebuilt != inventory:
        raise ValueError(
            "coil filament inventory does not reproduce from source"
        )
    hashes = []
    ids = []
    for index, row in enumerate(rows):
        digest = str(row.get("coordinates_sha256", ""))
        filament_id = str(row.get("filament_id", ""))
        if (
            row.get("source_ordinal") != index
            or len(digest) != 64
            or filament_id != f"filament-{digest[:16]}"
            or row.get("coordinate_count", 0) < 4
            or not math.isfinite(
                float(row.get("centreline_length_cm", math.nan))
            )
            or float(row["centreline_length_cm"]) <= 0.0
        ):
            raise ValueError("coil filament inventory row is invalid")
        hashes.append(digest)
        ids.append(filament_id)
    if len(set(hashes)) != len(rows) or len(set(ids)) != len(rows):
        raise ValueError("coil filament inventory identities are duplicated")


def write_inventory_create_only(
    path: str | Path, inventory: Mapping[str, Any]
) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(inventory, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
