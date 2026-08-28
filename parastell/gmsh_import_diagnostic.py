"""Bounded diagnostics for CadQuery-to-Gmsh solid identity matching."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping

from .reference_geometry import sha256_file
from .source_cad_refaceting import COMPONENT_STEP_FILES
from .source_cad_refaceting import MATERIAL_TAGS
from .source_cad_refaceting import SOURCE_CAD_FILES
from .source_cad_refaceting import _cadquery_solid_signature
from .source_cad_refaceting import _gmsh_volume_signature
from .source_cad_refaceting import _semantic_roles
from .source_cad_refaceting import _source_import_signature_match
from .source_cad_refaceting import _volume_signature_match


SCHEMA = "parastell.gmsh_import_mapping_diagnostic/v1.0.0"


def volume_signature_delta(
    source: Mapping[str, Any], imported: Mapping[str, Any]
) -> dict[str, Any]:
    """Return lossless scalar comparisons without declaring acceptance."""

    source_mass = float(source["mass_cm3"])
    imported_mass = float(imported["mass_cm3"])
    center_deltas = [
        float(imported_value) - float(source_value)
        for source_value, imported_value in zip(
            source["center_cm"], imported["center_cm"]
        )
    ]
    bounds_deltas = [
        float(imported_value) - float(source_value)
        for source_value, imported_value in zip(
            source["bounding_box_cm"], imported["bounding_box_cm"]
        )
    ]
    inertia_deltas = [
        float(imported_value) - float(source_value)
        for source_value, imported_value in zip(
            source["inertia_cm5"], imported["inertia_cm5"]
        )
    ]
    values = [
        source_mass,
        imported_mass,
        *center_deltas,
        *bounds_deltas,
        *inertia_deltas,
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "volume-signature comparison contains non-finite data"
        )
    if source_mass <= 0.0 or imported_mass <= 0.0:
        raise ValueError("volume-signature masses must be positive")
    return {
        "source_mass_cm3": source_mass,
        "imported_mass_cm3": imported_mass,
        "mass_absolute_delta_cm3": imported_mass - source_mass,
        "mass_ratio_imported_over_source": imported_mass / source_mass,
        "linear_scale_from_mass_ratio": (imported_mass / source_mass)
        ** (1.0 / 3.0),
        "center_delta_cm": center_deltas,
        "maximum_absolute_center_delta_cm": max(
            abs(value) for value in center_deltas
        ),
        "bounding_box_delta_cm": bounds_deltas,
        "maximum_absolute_bounding_box_delta_cm": max(
            abs(value) for value in bounds_deltas
        ),
        "inertia_delta_cm5": inertia_deltas,
        "maximum_absolute_inertia_delta_cm5": max(
            abs(value) for value in inertia_deltas
        ),
        "source_import_identity_match": _source_import_signature_match(
            source, imported
        ),
        "existing_strict_match": _volume_signature_match(source, imported),
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(serialized, encoding="utf-8")
    if json.loads(temporary.read_text(encoding="utf-8")) != value:
        raise RuntimeError("diagnostic JSON readback failed")
    temporary.replace(path)


def diagnose_source_gmsh_import(
    *, source_dir: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Import the immutable source CAD once and record every signature delta."""

    import cad_to_dagmc
    import cadquery as cq
    import gmsh as gmsh_module

    source_root = Path(source_dir).resolve()
    destination = Path(output_path).resolve()
    before = {
        name: {
            "sha256": sha256_file(source_root / name),
            "size_bytes": (source_root / name).stat().st_size,
        }
        for name in SOURCE_CAD_FILES
    }
    solids = []
    for name in COMPONENT_STEP_FILES:
        loaded = (
            cq.importers.importStep(str(source_root / name)).solids().vals()
        )
        if len(loaded) != 1:
            raise RuntimeError(
                f"expected one solid in {name}, got {len(loaded)}"
            )
        solids.append(loaded[0])
    magnets = (
        cq.importers.importStep(str(source_root / "magnet_set.step"))
        .solids()
        .vals()
    )
    if len(magnets) != 18:
        raise RuntimeError(f"expected 18 magnet solids, got {len(magnets)}")
    solids.extend(magnets)
    roles = _semantic_roles(list(MATERIAL_TAGS))
    source_signatures = [_cadquery_solid_signature(solid) for solid in solids]
    geometry = cq.Compound.makeCompound(solids)
    gmsh = cad_to_dagmc.init_gmsh()
    try:
        _, imported = cad_to_dagmc.get_volumes(
            gmsh, geometry, method="in memory"
        )
        imported_volumes = [tuple(value) for value in imported]
        imported_signatures = [
            _gmsh_volume_signature(gmsh, volume) for volume in imported_volumes
        ]
        pair_rows = []
        source_rows = []
        for source_index, (role, source_signature) in enumerate(
            zip(roles, source_signatures)
        ):
            strict_matches = []
            comparisons = []
            for imported_index, imported_signature in enumerate(
                imported_signatures
            ):
                delta = volume_signature_delta(
                    source_signature, imported_signature
                )
                row = {
                    "source_index": source_index,
                    "semantic_role": role,
                    "imported_index": imported_index,
                    "imported_dimtag": list(imported_volumes[imported_index]),
                    **delta,
                }
                pair_rows.append(row)
                comparisons.append(row)
                if delta["existing_strict_match"]:
                    strict_matches.append(imported_index)
            source_rows.append(
                {
                    "source_index": source_index,
                    "semantic_role": role,
                    "source_signature": source_signature,
                    "strict_match_indices": strict_matches,
                    "strict_match_count": len(strict_matches),
                    "source_import_identity_match_indices": [
                        row["imported_index"]
                        for row in comparisons
                        if row["source_import_identity_match"]
                    ],
                    "nearest_by_relative_mass": min(
                        comparisons,
                        key=lambda row: abs(
                            row["mass_ratio_imported_over_source"] - 1.0
                        ),
                    )["imported_index"],
                    "nearest_by_center": min(
                        comparisons,
                        key=lambda row: row[
                            "maximum_absolute_center_delta_cm"
                        ],
                    )["imported_index"],
                    "nearest_by_bounds": min(
                        comparisons,
                        key=lambda row: row[
                            "maximum_absolute_bounding_box_delta_cm"
                        ],
                    )["imported_index"],
                }
            )
        after = {
            name: {
                "sha256": sha256_file(source_root / name),
                "size_bytes": (source_root / name).stat().st_size,
            }
            for name in SOURCE_CAD_FILES
        }
        report = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scientific_decision": "NOT_PERFORMED_DIAGNOSTIC_ONLY",
            "source_dir": str(source_root),
            "source_artifacts_before": before,
            "source_artifacts_after": after,
            "source_immutability_pass": before == after,
            "source_solid_count": len(source_signatures),
            "gmsh_imported_volume_count": len(imported_volumes),
            "source_rows": source_rows,
            "imported_volumes": [
                {
                    "imported_index": index,
                    "dimtag": list(volume),
                    "signature": signature,
                }
                for index, (volume, signature) in enumerate(
                    zip(imported_volumes, imported_signatures)
                )
            ],
            "pairwise_signature_deltas": pair_rows,
            "existing_strict_bijection_pass": (
                len(source_signatures) == len(imported_volumes)
                and all(row["strict_match_count"] == 1 for row in source_rows)
                and len(
                    {
                        row["strict_match_indices"][0]
                        for row in source_rows
                        if row["strict_match_count"] == 1
                    }
                )
                == len(source_rows)
            ),
            "source_import_identity_bijection_pass": (
                len(source_signatures) == len(imported_volumes)
                and all(
                    len(row["source_import_identity_match_indices"]) == 1
                    for row in source_rows
                )
                and len(
                    {
                        row["source_import_identity_match_indices"][0]
                        for row in source_rows
                    }
                )
                == len(source_rows)
            ),
            "runtime": {
                "python": platform.python_version(),
                "cadquery": getattr(cq, "__version__", None),
                "cad_to_dagmc": getattr(cad_to_dagmc, "__version__", None),
                "gmsh": getattr(gmsh_module, "__version__", None),
            },
        }
    finally:
        gmsh.finalize()
    _write_json_create_only(destination, report)
    return report


__all__ = ["SCHEMA", "diagnose_source_gmsh_import", "volume_signature_delta"]
