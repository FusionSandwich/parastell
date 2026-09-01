"""Deterministic preprocessing-versus-query benchmark for facet patch atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from parastell.facet_patch_atlas import (  # noqa: E402
    build_facet_patch_atlas,
    localize_surface_crossings_cached,
)
from parastell.source_conformal_interface import (  # noqa: E402
    write_compact_json_create_only,
)
from parastell.surface_source_localization import (  # noqa: E402
    localize_surface_crossings,
)


def _grid_fixture(side: int) -> tuple[dict, dict]:
    if side < 2:
        raise ValueError("grid side must be at least two")
    vertices = np.asarray(
        [[x, y, 0.0] for y in range(side + 1) for x in range(side + 1)],
        dtype=float,
    )
    triangles = []
    vertex_ids = []
    for y in range(side):
        for x in range(side):
            lower_left = y * (side + 1) + x
            lower_right = lower_left + 1
            upper_left = lower_left + side + 1
            upper_right = upper_left + 1
            for row in (
                (lower_left, lower_right, upper_left),
                (upper_right, upper_left, lower_right),
            ):
                vertex_ids.append(row)
                triangles.append(vertices[list(row)])
    triangles = np.asarray(triangles)
    count = len(triangles)
    catalog = {
        "facet_id": [f"fixture-facet-{index:06d}" for index in range(count)],
        "surface_id": np.full(count, 7),
        "vertex_id": np.asarray(vertex_ids),
        "vertices_global_cm": triangles,
        "outward_normal_global": np.repeat(
            np.asarray([[0.0, 0.0, 1.0]]), count, axis=0
        ),
        "normal_source": "dagmc_forward_reverse_topology",
    }
    positions = np.mean(triangles, axis=1)
    phase = {
        "position_global_cm": positions,
        "direction_global": np.repeat(
            np.asarray([[0.0, 0.0, -1.0]]), count, axis=0
        ),
        "surface_id": np.full(count, 7),
    }
    return catalog, phase


def benchmark(side: int, repeats: int) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    catalog, phase = _grid_fixture(side)
    start = perf_counter_ns()
    atlas = build_facet_patch_atlas(catalog)
    preprocessing_ns = perf_counter_ns() - start

    cached, cached_audit = localize_surface_crossings_cached(phase, atlas)
    uncached, _ = localize_surface_crossings(phase, catalog)
    if not (
        np.array_equal(cached["facet_id"], uncached["facet_id"])
        and np.allclose(
            cached["barycentric_coordinates"],
            uncached["barycentric_coordinates"],
        )
    ):
        raise RuntimeError("cached and uncached localization disagree")

    cached_times = []
    uncached_times = []
    for _ in range(repeats):
        start = perf_counter_ns()
        localize_surface_crossings_cached(phase, atlas)
        cached_times.append(perf_counter_ns() - start)
        start = perf_counter_ns()
        localize_surface_crossings(phase, catalog)
        uncached_times.append(perf_counter_ns() - start)
    cached_median = int(statistics.median(cached_times))
    uncached_median = int(statistics.median(uncached_times))
    return {
        "schema": "parastell.facet_patch_atlas_benchmark/v1.0.0",
        "status": "DIAGNOSTIC_ONLY",
        "fixture_only": True,
        "grid_side": side,
        "facet_count": atlas.facet_count,
        "query_count_per_repeat": len(phase["surface_id"]),
        "repeats": repeats,
        "preprocessing_nanoseconds": preprocessing_ns,
        "cached_query_median_nanoseconds": cached_median,
        "uncached_query_median_nanoseconds": uncached_median,
        "cached_queries_per_second": (
            len(phase["surface_id"]) * 1.0e9 / cached_median
        ),
        "uncached_queries_per_second": (
            len(phase["surface_id"]) * 1.0e9 / uncached_median
        ),
        "query_speedup_ratio": uncached_median / cached_median,
        "candidate_facets_considered": cached_audit[
            "candidate_facets_considered"
        ],
        "full_surface_facets_available": cached_audit[
            "full_surface_facets_available"
        ],
        "cached_uncached_equivalence_pass": True,
        "arbitrary_polyhedral_radiant_sweep_claimed": False,
        "physical_qualification_claimed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = benchmark(args.side, args.repeats)
    if args.output is None:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        write_compact_json_create_only(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
