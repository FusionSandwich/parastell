"""Validated, solver-neutral heterogeneous REBCO tape-stack geometry.

No tape architecture is assumed by this module.  A caller must provide every
layer, dimension, and material binding.  The generated boxes use a local,
right-handed coordinate system and still require solver-native geometry and
navigation qualification before transport.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .material_identity import validate_material_record

DEFINITION_SCHEMA = "parastell.magnet_tape_stack_definition/v1.0.0"
MANIFEST_SCHEMA = "parastell.magnet_tape_stack_manifest/v1.0.0"
USER_CLASSIFICATION = "USER_SUPPLIED_NOT_YET_TRANSPORT_QUALIFIED"
PROVISIONAL_CLASSIFICATION = (
    "PROVISIONAL_VERIFICATION_FIXTURE_NOT_MANUFACTURER_SPECIFICATION"
)
CLASSIFICATIONS = {USER_CLASSIFICATION, PROVISIONAL_CLASSIFICATION}
SIDES = ("below_rebco", "rebco", "above_rebco")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_UM_TO_CM = Decimal("0.0001")
_MM_TO_CM = Decimal("0.1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identifier(value: Any, label: str) -> str:
    text = str(value).strip()
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError(f"{label} is not a safe stable identifier")
    return text


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive decimal string")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a positive decimal string") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def validate_tape_stack_definition(
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one explicit local tape-stack definition."""
    required = {
        "schema",
        "stack_id",
        "magnet_id",
        "classification",
        "geometry",
        "material_bindings",
        "layers",
        "provenance",
    }
    if not isinstance(definition, Mapping) or set(definition) != required:
        raise ValueError("tape-stack definition keys differ")
    if definition.get("schema") != DEFINITION_SCHEMA:
        raise ValueError("unsupported tape-stack definition schema")
    stack_id = _identifier(definition.get("stack_id"), "stack_id")
    magnet_id = _identifier(definition.get("magnet_id"), "magnet_id")
    classification = str(definition.get("classification", ""))
    if classification not in CLASSIFICATIONS:
        raise ValueError("tape-stack classification is not explicit")

    geometry = definition.get("geometry")
    geometry_keys = {
        "tape_width_mm",
        "longitudinal_mode",
        "model_length_mm",
        "stack_axis",
        "reference_plane",
        "layer_order",
    }
    if not isinstance(geometry, Mapping) or set(geometry) != geometry_keys:
        raise ValueError("tape-stack geometry fields differ")
    width = _decimal(geometry.get("tape_width_mm"), "tape_width_mm")
    length = _decimal(geometry.get("model_length_mm"), "model_length_mm")
    if geometry.get("longitudinal_mode") not in {
        "finite_segment",
        "unit_length_normalization",
    }:
        raise ValueError("longitudinal_mode is unsupported")
    if (
        geometry.get("stack_axis") != "local_z"
        or geometry.get("reference_plane") != "rebco_midplane"
        or geometry.get("layer_order") != "below_to_above"
    ):
        raise ValueError("local tape coordinate convention is ambiguous")

    bindings = definition.get("material_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("material bindings are required")
    normalized_bindings: dict[str, str] = {}
    for alias, material_id in bindings.items():
        key = _identifier(alias, "material binding alias")
        if key in normalized_bindings:
            raise ValueError("material binding aliases are duplicated")
        normalized_bindings[key] = _identifier(material_id, "material_id")

    layers = definition.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise TypeError("layers must be an ordered nonempty list")
    normalized_layers = []
    layer_ids: set[str] = set()
    seen_sides: list[str] = []
    used_bindings: set[str] = set()
    for row in layers:
        if not isinstance(row, Mapping) or set(row) != {
            "layer_id",
            "role",
            "side",
            "material_binding",
            "thickness_um",
        }:
            raise ValueError("tape layer fields differ")
        layer_id = _identifier(row.get("layer_id"), "layer_id")
        role = _identifier(row.get("role"), "layer role")
        side = str(row.get("side", ""))
        binding = _identifier(
            row.get("material_binding"), "material binding alias"
        )
        thickness = _decimal(row.get("thickness_um"), "layer thickness_um")
        if layer_id in layer_ids:
            raise ValueError("layer IDs are duplicated")
        if side not in SIDES:
            raise ValueError("layer side must be explicit relative to REBCO")
        if binding not in normalized_bindings:
            raise ValueError("layer uses an unknown material binding")
        layer_ids.add(layer_id)
        seen_sides.append(side)
        used_bindings.add(binding)
        normalized_layers.append(
            {
                "layer_id": layer_id,
                "role": role,
                "side": side,
                "material_binding": binding,
                "thickness_um": _decimal_text(thickness),
            }
        )
    if not normalized_layers:
        raise ValueError("layers must be an ordered nonempty list")
    if seen_sides.count("rebco") != 1:
        raise ValueError("exactly one REBCO layer is required")
    rebco_index = seen_sides.index("rebco")
    rebco = normalized_layers[rebco_index]
    if rebco["role"] != "rebco_active":
        raise ValueError("the REBCO layer role must be rebco_active")
    if (
        rebco_index == 0
        or rebco_index == len(normalized_layers) - 1
        or any(side != "below_rebco" for side in seen_sides[:rebco_index])
        or any(side != "above_rebco" for side in seen_sides[rebco_index + 1 :])
    ):
        raise ValueError("ordered layers must explicitly bracket REBCO")
    if used_bindings != set(normalized_bindings):
        raise ValueError("material bindings must be used exactly by the stack")

    provenance = definition.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "source",
        "citation",
        "provisional",
        "design_authority",
        "manufacturer_specification",
    }:
        raise ValueError("tape-stack provenance fields differ")
    source = str(provenance.get("source", "")).strip()
    citation = provenance.get("citation")
    if not source or (citation is not None and not str(citation).strip()):
        raise ValueError("tape-stack source and citation are invalid")
    if (
        not isinstance(provenance.get("provisional"), bool)
        or not isinstance(provenance.get("design_authority"), bool)
        or not isinstance(provenance.get("manufacturer_specification"), bool)
    ):
        raise TypeError("tape-stack provenance flags must be booleans")
    provisional = provenance["provisional"]
    if (classification == PROVISIONAL_CLASSIFICATION) != provisional:
        raise ValueError("classification conflicts with provisional status")
    if provisional and provenance["design_authority"]:
        raise ValueError("a provisional fixture cannot be design authority")
    if provisional and provenance["manufacturer_specification"]:
        raise ValueError("a provisional fixture cannot be a manufacturer spec")

    return {
        "schema": DEFINITION_SCHEMA,
        "stack_id": stack_id,
        "magnet_id": magnet_id,
        "classification": classification,
        "geometry": {
            "tape_width_mm": _decimal_text(width),
            "longitudinal_mode": geometry["longitudinal_mode"],
            "model_length_mm": _decimal_text(length),
            "stack_axis": "local_z",
            "reference_plane": "rebco_midplane",
            "layer_order": "below_to_above",
        },
        "material_bindings": dict(sorted(normalized_bindings.items())),
        "layers": normalized_layers,
        "provenance": {
            "source": source,
            "citation": None if citation is None else str(citation).strip(),
            "provisional": provisional,
            "design_authority": provenance["design_authority"],
            "manufacturer_specification": provenance[
                "manufacturer_specification"
            ],
        },
    }


def _validated_materials(
    definition: Mapping[str, Any],
    material_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(material_records, Sequence) or isinstance(
        material_records, (str, bytes)
    ):
        raise TypeError("material_records must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in material_records:
        normalized = validate_material_record(row)
        material_id = normalized["material_id"]
        if material_id in by_id:
            raise ValueError("material record IDs are duplicated")
        by_id[material_id] = normalized
    expected = set(definition["material_bindings"].values())
    if set(by_id) != expected:
        raise ValueError(
            "material records do not exactly match stack bindings"
        )
    return by_id


def _assemble_manifest(
    definition: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
    definition_binding: Mapping[str, Any],
) -> dict[str, Any]:
    width_cm = Decimal(definition["geometry"]["tape_width_mm"]) * _MM_TO_CM
    length_cm = Decimal(definition["geometry"]["model_length_mm"]) * _MM_TO_CM
    thicknesses = [
        Decimal(row["thickness_um"]) * _UM_TO_CM
        for row in definition["layers"]
    ]
    rebco_index = next(
        index
        for index, row in enumerate(definition["layers"])
        if row["side"] == "rebco"
    )
    rebco_lower_unshifted = sum(thicknesses[:rebco_index], Decimal(0))
    shift = -(rebco_lower_unshifted + thicknesses[rebco_index] / 2)
    x_lower, x_upper = -width_cm / 2, width_cm / 2
    y_lower, y_upper = -length_cm / 2, length_cm / 2
    cursor = shift
    generated_layers = []
    for row, thickness_cm in zip(definition["layers"], thicknesses):
        lower_z, upper_z = cursor, cursor + thickness_cm
        material_id = definition["material_bindings"][row["material_binding"]]
        material = materials[material_id]
        volume = width_cm * length_cm * thickness_cm
        mass = volume * Decimal(str(material["density_g_cm3"]))
        generated_layers.append(
            {
                **row,
                "material_id": material_id,
                "material_record_sha256": material["record_sha256"],
                "material_composition_sha256": material["composition_sha256"],
                "density_g_cm3": material["density_g_cm3"],
                "thickness_cm": float(thickness_cm),
                "bounds_cm": {
                    "lower": [float(x_lower), float(y_lower), float(lower_z)],
                    "upper": [float(x_upper), float(y_upper), float(upper_z)],
                },
                "exact_z_bounds_cm": [
                    _decimal_text(lower_z),
                    _decimal_text(upper_z),
                ],
                "volume_cm3": float(volume),
                "exact_volume_cm3": _decimal_text(volume),
                "mass_g": float(mass),
            }
        )
        cursor = upper_z
    total = sum(thicknesses, Decimal(0))
    below = sum(thicknesses[:rebco_index], Decimal(0))
    above = sum(thicknesses[rebco_index + 1 :], Decimal(0))
    rebco_row = generated_layers[rebco_index]
    result: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "GEOMETRY_AND_MATERIAL_BINDINGS_VALIDATED_NOT_TRANSPORT_EXECUTED",
        "definition_binding": dict(definition_binding),
        "definition": dict(definition),
        "magnet_id": definition["magnet_id"],
        "stack_id": definition["stack_id"],
        "classification": definition["classification"],
        "local_coordinate_system": {
            "units": "cm",
            "basis_columns": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "axes": {
                "x": "tape_width",
                "y": "longitudinal",
                "z": "below_to_above",
            },
            "determinant": 1.0,
            "right_handed": True,
            "global_spatial_mapping_applied": False,
        },
        "geometry": {
            "kind": "contiguous_axis_aligned_layer_boxes",
            "tape_width_cm": float(width_cm),
            "model_length_cm": float(length_cm),
            "longitudinal_mode": definition["geometry"]["longitudinal_mode"],
            "total_thickness_um": _decimal_text(total / _UM_TO_CM),
            "total_thickness_cm": float(total),
            "exact_total_thickness_cm": _decimal_text(total),
            "layer_count": len(generated_layers),
            "closure": {
                "contiguous": True,
                "gaps_cm": "0",
                "overlaps_cm": "0",
                "cumulative_decimal_arithmetic": True,
            },
        },
        "layers": generated_layers,
        "material_records": [materials[key] for key in sorted(materials)],
        "rebco_focus": {
            "rebco_layer_id": rebco_row["layer_id"],
            "rebco_bounds_cm": rebco_row["bounds_cm"],
            "directly_below_layer_id": generated_layers[rebco_index - 1][
                "layer_id"
            ],
            "directly_above_layer_id": generated_layers[rebco_index + 1][
                "layer_id"
            ],
            "below_rebco_total_thickness_um": _decimal_text(below / _UM_TO_CM),
            "above_rebco_total_thickness_um": _decimal_text(above / _UM_TO_CM),
            "recommended_distinct_tally_regions": [
                generated_layers[rebco_index - 1]["layer_id"],
                rebco_row["layer_id"],
                generated_layers[rebco_index + 1]["layer_id"],
            ],
        },
        "qualification": {
            "definition_and_material_bindings": "PASS",
            "analytic_layer_closure": "PASS",
            "solver_native_geometry": "NOT_RUN",
            "solver_navigation": "NOT_RUN",
            "transport": "NOT_RUN",
            "production_authorized": False,
        },
        "claims": {
            "manufacturer_specification": definition["provenance"][
                "manufacturer_specification"
            ],
            "global_magnet_geometry": False,
            "global_geometry_mutated": False,
            "cad_constructed": False,
            "transport_executed": False,
        },
    }
    result["manifest_sha256"] = _canonical_sha256(result)
    return result


def build_tape_stack_manifest(
    definition_path: str | Path,
    material_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Generate a deterministic hash-bound layer-box manifest."""
    source = Path(definition_path).resolve(strict=True)
    raw = json.loads(source.read_text(encoding="utf-8"))
    definition = validate_tape_stack_definition(raw)
    materials = _validated_materials(definition, material_records)
    binding = {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }
    return _assemble_manifest(definition, materials, binding)


def validate_tape_stack_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the complete self-sealed stack manifest and its sources."""
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != MANIFEST_SCHEMA
    ):
        raise ValueError("unsupported tape-stack manifest schema")
    binding = manifest.get("definition_binding")
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise ValueError("definition binding is incomplete")
    path = Path(str(binding["path"])).resolve(strict=True)
    if (
        path.stat().st_size != binding["bytes"]
        or _sha256(path) != binding["sha256"]
    ):
        raise ValueError("tape-stack definition binding changed")
    definition = validate_tape_stack_definition(manifest.get("definition", {}))
    materials = _validated_materials(
        definition, manifest.get("material_records", [])
    )
    expected = _assemble_manifest(definition, materials, binding)
    if dict(manifest) != expected:
        raise ValueError("tape-stack manifest seal or geometry is invalid")
    return expected


def write_manifest_create_only(
    path: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """Validate and write a manifest without overwriting existing evidence."""
    validated = validate_tape_stack_manifest(manifest)
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(validated, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output
