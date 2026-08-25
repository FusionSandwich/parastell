"""Fail-closed native-CAD overlap audit for combined transport solids."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _bounds(solid: Any) -> tuple[np.ndarray, np.ndarray]:
    box = solid.BoundingBox()
    lower = np.asarray([box.xmin, box.ymin, box.zmin], dtype=float)
    upper = np.asarray([box.xmax, box.ymax, box.zmax], dtype=float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(upper < lower)
    ):
        raise ValueError("CAD solid has an invalid bounding box")
    return lower, upper


def audit_native_cad_overlaps(
    solids: Sequence[Any],
    labels: Sequence[str],
    *,
    absolute_volume_tolerance_cm3: float = 1.0e-7,
    relative_volume_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Prove no native solid pair shares material volume above tolerance.

    Axis-aligned bounding boxes cheaply reject separated pairs. Every remaining
    pair is checked using the CAD kernel's Boolean intersection, so shared
    faces are allowed but positive shared volumes are not.
    """
    objects = tuple(solids)
    names = tuple(str(label) for label in labels)
    if len(objects) < 2 or len(names) != len(objects):
        raise ValueError("overlap audit requires at least two labelled solids")
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("overlap audit labels must be nonempty and unique")
    if (
        not np.isfinite(absolute_volume_tolerance_cm3)
        or absolute_volume_tolerance_cm3 < 0.0
        or not np.isfinite(relative_volume_tolerance)
        or relative_volume_tolerance < 0.0
    ):
        raise ValueError(
            "overlap volume tolerances must be finite and nonnegative"
        )

    bounds = [_bounds(solid) for solid in objects]
    volumes = np.asarray(
        [float(solid.Volume()) for solid in objects], dtype=float
    )
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError(
            "overlap audit requires positive finite solid volumes"
        )

    candidates = 0
    maximum_intersection = 0.0
    overlap_rows = []
    pair_count = len(objects) * (len(objects) - 1) // 2
    for first in range(len(objects)):
        first_lower, first_upper = bounds[first]
        for second in range(first + 1, len(objects)):
            second_lower, second_upper = bounds[second]
            if np.any(first_upper < second_lower) or np.any(
                second_upper < first_lower
            ):
                continue
            candidates += 1
            try:
                intersection_volume = float(
                    objects[first].intersect(objects[second]).Volume()
                )
            except Exception as exc:
                raise RuntimeError(
                    "CAD Boolean intersection failed for overlap candidate "
                    f"{names[first]!r} and {names[second]!r}"
                ) from exc
            if (
                not np.isfinite(intersection_volume)
                or intersection_volume < 0.0
            ):
                raise ValueError(
                    "CAD Boolean intersection returned an invalid volume"
                )
            maximum_intersection = max(
                maximum_intersection, intersection_volume
            )
            tolerance = max(
                float(absolute_volume_tolerance_cm3),
                float(relative_volume_tolerance)
                * min(volumes[first], volumes[second]),
            )
            if intersection_volume > tolerance:
                overlap_rows.append(
                    {
                        "first": names[first],
                        "second": names[second],
                        "intersection_volume_cm3": intersection_volume,
                        "allowed_volume_cm3": tolerance,
                    }
                )
    if overlap_rows:
        sample = overlap_rows[:5]
        raise ValueError(
            f"native CAD overlap audit found {len(overlap_rows)} positive-volume "
            f"overlaps; sample={sample}"
        )
    return {
        "status": "PASS",
        "method": "native CAD Boolean intersection after AABB candidate filtering",
        "solid_count": len(objects),
        "pair_count": pair_count,
        "aabb_candidate_pair_count": candidates,
        "positive_volume_overlap_count": 0,
        "maximum_intersection_volume_cm3": maximum_intersection,
        "absolute_volume_tolerance_cm3": float(absolute_volume_tolerance_cm3),
        "relative_volume_tolerance": float(relative_volume_tolerance),
        "shared_faces_allowed": True,
    }
