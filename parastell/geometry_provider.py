"""Fail-closed geometry-provider contracts for transport workflows.

This module deliberately contains no CAD, DAGMC, OpenMC, or Paramak import.
It validates immutable geometry receipts before a transport-facing caller may
load an artifact.  In particular, the WISTELL-D provider never discovers a
geometry by filename, recency, or fallback search.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np


PROVIDER_VERSION = "1.0.0"
PROVIDER_SCHEMA = "parastell.geometry_provider/v1.0.0"
WISTELL_D_ACCEPTANCE_SCHEMA = "wistell_d.geometry_acceptance/v1.0.0"
WISTELL_D_GEOMETRY_INPUT_MODE = "REBUILD_FROM_AUTHORITATIVE_WISTELL_D_INPUTS"
DIRECT90_DOCKER_IMAGE_ID = (
    "sha256:ca0c3b1fba39ce27af6ebdb79df14795041922e72521f232cdd770ff1c416191"
)
DIRECT90_RUNTIME_RECEIPT_SHA256 = (
    "7fe93d056604d5c279f55678615e772201a9b3737855891f88708580221be07e"
)
DIRECT90_REFERENCE_MANIFEST_SHA256 = (
    "b6e723cdb9ac95d789a838abbf44590d210c4fdbe718c3b459777d38768e0499"
)
DIRECT45_REFERENCE_MANIFEST_SHA256 = (
    "f330bbd06a0c8234a3b52932ee48e8dcdec7e2842c3d12a5c75d3052028920b4"
)

WISTELL_D_SOURCE_HASHES = {
    "wout_wistell-d.nc": (
        "9231969001203a8133255ee0a275bf552b114cc12524dda0608ab2f12047f7ac"
    ),
    "coils.wistell-d": (
        "7748369407d28a70f35b5c4a7c0ab860495a08fd0030002112ea933fe570159b"
    ),
    "nwl.npy": (
        "56baa090d61b67273efba61213849b7516beabb2a57fc2ad4751a6f3a32b2db4"
    ),
    "blanket_boundary.npy": (
        "fdb85b2c0c8cd72f5d000302e0b67349ebf72679f98f9c4d7739e5d8484cdde3"
    ),
    "magnet_boundary.npy": (
        "3579e5d8fe97dd74c8700e5676964159f00f07989ca6436528f60462889f05bd"
    ),
    "source_mesh.h5m": (
        "65264e15669d09c43f107c3b43c2af24ffbd15173e3bbd0e990b527bfa0b5322"
    ),
    "strengths.npy": (
        "0ed18ab58bcc1e9884bf1b5c8bf19a7b7558ce7afe1869f1a2b01710148af6df"
    ),
}

# These hashes identify the generic ParaStell example input pair.  They may
# remain as isolated software fixtures, but can never select scientific data.
KNOWN_EXAMPLE_SOURCE_HASHES = frozenset(
    {
        "1cebb8d46e60d77df4a6904662a9c9f943137a9fb59f7290e5309af15fa04797",
        "69f508b216f0b674368ca8731d390c9d514736ff092f0ebecd854e8772ae04ab",
        "83902138eccefc266df638cf8662e00eb8a11cb7c66848d80c0d9e19805383e5",
    }
)

EXPECTED_LAYER_ORDER = (
    "chamber",
    "first_wall",
    "breeder",
    "back_wall",
    "high_temperature_shield",
    "vacuum_vessel",
    "low_temperature_shield",
    "vacuum_gap",
    "magnet_envelope",
)


class GeometryProvenanceError(ValueError):
    """Raised when geometry provenance is incomplete or excluded."""


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _is_git_sha(value: Any) -> bool:
    text = str(value)
    return len(text) == 40 and all(
        character in "0123456789abcdef" for character in text
    )


def _finite_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for malformed/non-finite input."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without mutating *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Hash a JSON-compatible value using deterministic compact encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _homogeneous(linear: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    return matrix


@dataclass(frozen=True)
class RigidTransform:
    """Orthogonal homogeneous transform with polar/axial mapping helpers."""

    transform_id: str
    matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("rigid transform matrix must be 4x4")
        if not np.isfinite(matrix).all():
            raise ValueError("rigid transform matrix must be finite")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-14):
            raise ValueError("invalid homogeneous transform final row")
        linear = matrix[:3, :3]
        if not np.allclose(linear.T @ linear, np.eye(3), atol=1e-12):
            raise ValueError("rigid transform linear part is not orthogonal")
        if not np.isclose(abs(np.linalg.det(linear)), 1.0, atol=1e-12):
            raise ValueError(
                "rigid transform determinant must have unit magnitude"
            )

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)

    @property
    def linear(self) -> np.ndarray:
        return self.array[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.array[:3, 3]

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.linear))

    @property
    def inverse(self) -> "RigidTransform":
        linear = self.linear.T
        translation = -linear @ self.translation
        matrix = _homogeneous(linear, translation)
        return RigidTransform(
            transform_id=f"{self.transform_id}.inverse",
            matrix=tuple(tuple(float(x) for x in row) for row in matrix),
        )

    def transform_points(self, points: Any) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,):
            raise ValueError("points must end in a three-coordinate axis")
        return points @ self.linear.T + self.translation

    def transform_directions(self, directions: Any) -> np.ndarray:
        directions = np.asarray(directions, dtype=float)
        if directions.shape[-1:] != (3,):
            raise ValueError("directions must end in a three-coordinate axis")
        return directions @ self.linear.T

    def transform_axial_vectors(self, vectors: Any) -> np.ndarray:
        return self.determinant * self.transform_directions(vectors)

    def transform_tetrahedra(
        self, tetrahedra: Any, *, positive_orientation: bool = True
    ) -> np.ndarray:
        transformed = self.transform_points(tetrahedra)
        if transformed.shape[-2:] != (4, 3):
            raise ValueError("tetrahedra must end in a (4, 3) vertex array")
        if positive_orientation and self.determinant < 0.0:
            transformed = transformed.copy()
            transformed[..., [1, 2], :] = transformed[..., [2, 1], :]
        return transformed

    def receipt(self) -> dict[str, Any]:
        inverse = self.inverse.array
        product = self.array @ inverse
        return {
            "transform_id": self.transform_id,
            "matrix": self.array.tolist(),
            "inverse_matrix": inverse.tolist(),
            "determinant": self.determinant,
            "orthogonality_inf_norm": float(
                np.linalg.norm(
                    self.linear.T @ self.linear - np.eye(3), ord=np.inf
                )
            ),
            "inverse_closure_inf_norm": float(
                np.linalg.norm(product - np.eye(4), ord=np.inf)
            ),
        }


def derive_wistell_d_transforms(
    *, nfp: int, lasym: bool, phase_origin_degrees: float = 0.0
) -> dict[str, RigidTransform]:
    """Derive period and half-period stellarator-symmetry generators.

    The canonical half-period is ``[phase_origin, phase_origin + 180/nfp]``.
    Its mate is obtained by a 180-degree rotation about the radial line at the
    shared seam.  This is derived from live ``nfp``/``lasym`` evidence; callers
    must not substitute a simple rotation by the half-period angle.
    """
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    if not lasym:
        raise GeometryProvenanceError(
            "WISTELL-D half-period expansion requires live stellarator symmetry"
        )

    period_degrees = 360.0 / nfp
    seam_degrees = phase_origin_degrees + period_degrees / 2.0
    seam_radians = math.radians(seam_degrees)
    axis = np.array(
        [math.cos(seam_radians), math.sin(seam_radians), 0.0], dtype=float
    )
    half_linear = 2.0 * np.outer(axis, axis) - np.eye(3)

    period_radians = math.radians(period_degrees)
    c = math.cos(period_radians)
    s = math.sin(period_radians)
    period_linear = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )

    identity = np.eye(4)
    half = _homogeneous(half_linear, np.zeros(3))
    period = _homogeneous(period_linear, np.zeros(3))
    return {
        "identity": RigidTransform(
            "identity", tuple(tuple(float(x) for x in row) for row in identity)
        ),
        "half_period_mate": RigidTransform(
            "half_period_mate",
            tuple(tuple(float(x) for x in row) for row in half),
        ),
        "field_period": RigidTransform(
            "field_period",
            tuple(tuple(float(x) for x in row) for row in period),
        ),
    }


def complete_pairwise_audit(
    components: Mapping[str, Any],
    intersection_volume: Callable[[Any, Any], float],
    *,
    tolerance_cm3: float = 1.0e-5,
) -> dict[str, Any]:
    """Evaluate every unordered component pair and report positive overlaps."""
    if tolerance_cm3 < 0.0 or not math.isfinite(tolerance_cm3):
        raise ValueError("overlap tolerance must be finite and nonnegative")
    names = tuple(components)
    if len(names) < 2:
        raise ValueError("at least two components are required")
    pairs: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(names[:-1]):
        for right_index, right_name in enumerate(
            names[left_index + 1 :], start=left_index + 1
        ):
            try:
                volume = float(
                    intersection_volume(
                        components[left_name], components[right_name]
                    )
                )
                if not math.isfinite(volume) or volume < 0.0:
                    raise ValueError(
                        "intersection volume must be finite and nonnegative"
                    )
                error = None
            except (
                Exception
            ) as exc:  # fail closed and preserve the exact error
                volume = None
                error = f"{type(exc).__name__}: {exc}"
            pairs.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "left_index": left_index,
                    "right_index": right_index,
                    "adjacent_in_radial_stack": right_index == left_index + 1,
                    "intersection_volume_cm3": volume,
                    "boolean_error": error,
                    "overlap": volume is not None and volume > tolerance_cm3,
                }
            )

    expected = len(names) * (len(names) - 1) // 2
    return {
        "component_order": list(names),
        "component_count": len(names),
        "expected_pair_count": expected,
        "evaluated_pair_count": len(pairs),
        "tolerance_cm3": tolerance_cm3,
        "boolean_failure_count": sum(
            row["boolean_error"] is not None for row in pairs
        ),
        "overlap_count": sum(row["overlap"] for row in pairs),
        "nonadjacent_overlap_count": sum(
            row["overlap"] and not row["adjacent_in_radial_stack"]
            for row in pairs
        ),
        "pairs": pairs,
    }


def require_complete_pairwise_acceptance(report: Mapping[str, Any]) -> None:
    """Reject missing pairs, Boolean failures, and overlaps of any pair."""
    expected = int(report.get("expected_pair_count", -1))
    evaluated = int(report.get("evaluated_pair_count", -1))
    pairs = report.get("pairs")
    if not isinstance(pairs, list) or expected <= 0 or evaluated != expected:
        raise GeometryProvenanceError("component-pair audit is incomplete")
    if len(pairs) != expected:
        raise GeometryProvenanceError(
            "component-pair rows do not match expected count"
        )
    if int(report.get("boolean_failure_count", -1)) != 0:
        raise GeometryProvenanceError(
            "component-pair Boolean audit contains failures"
        )
    overlaps = [row for row in pairs if row.get("overlap")]
    if overlaps:
        pair = overlaps[0]
        raise GeometryProvenanceError(
            "positive component overlap: "
            f"{pair.get('left')}-{pair.get('right')} = "
            f"{pair.get('intersection_volume_cm3')} cm^3"
        )


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        strings: list[str] = []
        for key, item in value.items():
            strings.extend((str(key), *_all_strings(item)))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(_all_strings(item))
        return strings
    return [str(value)]


def _validate_prompt02_local_frames(local_frames: Any) -> None:
    if not isinstance(local_frames, Mapping):
        raise GeometryProvenanceError("missing Prompt-2 local-frame contract")
    if local_frames.get("status") != "PASS":
        raise GeometryProvenanceError(
            "Prompt-2 local-frame contract is not accepted"
        )
    frames = local_frames.get("frames")
    if not isinstance(frames, list) or not frames:
        raise GeometryProvenanceError(
            "Prompt-2 local-frame inventory is empty"
        )
    if int(local_frames.get("frame_count", -1)) != len(frames):
        raise GeometryProvenanceError("Prompt-2 local-frame count mismatch")
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise GeometryProvenanceError(
                f"local frame {index} is not a mapping"
            )
        control_id = frame.get("canonical_control_point_id")
        arc_interval = frame.get("canonical_arc_interval")
        if control_id is None and arc_interval is None:
            raise GeometryProvenanceError(
                f"local frame {index} has no control point or arc interval"
            )
        if control_id is not None and not 0 <= int(control_id) < 452:
            raise GeometryProvenanceError(
                f"local frame {index} has an invalid canonical control point"
            )
        for key in (
            "canonical_45_patch_id",
            "symmetry_instance_id",
            "transform_id",
            "local_engineering_frame_id",
            "frame_kind",
        ):
            if not isinstance(frame.get(key), str) or not frame[key]:
                raise GeometryProvenanceError(
                    f"local frame {index} is missing {key}"
                )
        if frame.get("symmetry_instance_id") not in ("canonical", "mate"):
            raise GeometryProvenanceError(
                f"local frame {index} has an unsupported symmetry instance"
            )
        for key in ("global_90_surface_ids", "global_90_facet_ids"):
            identifiers = frame.get(key)
            if not isinstance(identifiers, list) or not identifiers:
                raise GeometryProvenanceError(
                    f"local frame {index} has no {key}"
                )
            if any(int(identifier) < 0 for identifier in identifiers):
                raise GeometryProvenanceError(
                    f"local frame {index} has an invalid {key}"
                )
        forward = np.asarray(frame.get("forward_transform"), dtype=float)
        inverse = np.asarray(frame.get("inverse_transform"), dtype=float)
        if forward.shape != (4, 4) or inverse.shape != (4, 4):
            raise GeometryProvenanceError(
                f"local frame {index} transforms are not 4x4"
            )
        if not np.isfinite(forward).all() or not np.isfinite(inverse).all():
            raise GeometryProvenanceError(
                f"local frame {index} transforms are not finite"
            )
        if not np.allclose(forward @ inverse, np.eye(4), atol=1.0e-10):
            raise GeometryProvenanceError(
                f"local frame {index} forward/inverse transforms do not close"
            )
        if not isinstance(frame.get("tape_twist_resolved"), bool):
            raise GeometryProvenanceError(
                f"local frame {index} has no Boolean tape-twist status"
            )


def validate_wistell_d_manifest(
    manifest: Mapping[str, Any],
    *,
    required_extent_degrees: float | None = None,
    require_local_frames: bool = False,
) -> None:
    """Validate a new-lane scientific WISTELL-D geometry manifest."""
    if manifest.get("schema") != WISTELL_D_ACCEPTANCE_SCHEMA:
        raise GeometryProvenanceError(
            "unsupported WISTELL-D acceptance schema"
        )
    if manifest.get("geometry_input_mode") != WISTELL_D_GEOMETRY_INPUT_MODE:
        raise GeometryProvenanceError(
            "unqualified WISTELL-D geometry input mode"
        )

    flattened = "\n".join(_all_strings(manifest)).lower()
    forbidden_markers = (
        "examples/wout_vmec.nc",
        "examples\\wout_vmec.nc",
        "coils.example",
        "parastell_multi_config_20260827_01",
        "newest artifact",
        "fallback geometry",
    )
    if any(marker in flattened for marker in forbidden_markers):
        raise GeometryProvenanceError(
            "example-derived or fallback geometry marker"
        )
    if any(digest in flattened for digest in KNOWN_EXAMPLE_SOURCE_HASHES):
        raise GeometryProvenanceError(
            "known generic-example source fingerprint"
        )

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise GeometryProvenanceError("missing WISTELL-D source receipt")
    if source.get("lineage_commit") != (
        "398032b8c0b4e7c0459c602f2af1e73b3fca0b9a"
    ):
        raise GeometryProvenanceError("wrong WISTELL-D lineage commit")
    source_hashes = source.get("hashes")
    if not isinstance(source_hashes, Mapping):
        raise GeometryProvenanceError("missing authoritative source hashes")
    if dict(source_hashes) != WISTELL_D_SOURCE_HASHES:
        raise GeometryProvenanceError("authoritative source hash set mismatch")

    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise GeometryProvenanceError("missing WISTELL-D model receipt")
    if model.get("device") != "WISTELL-D":
        raise GeometryProvenanceError("geometry device is not WISTELL-D")
    if float(model.get("wall_s", math.nan)) != 1.0:
        raise GeometryProvenanceError("WISTELL-D geometry requires wall_s=1.0")
    if int(model.get("canonical_control_count", -1)) != 452:
        raise GeometryProvenanceError(
            "WISTELL-D canonical control count mismatch"
        )
    extent = float(model.get("toroidal_extent_degrees", math.nan))
    if model.get("source_cad_grid") not in ([61, 121], [80, 90]):
        raise GeometryProvenanceError(
            "scientific source CAD must use 61x121 half-period or 80x90 direct-period grid"
        )
    if extent == 90.0 and (
        model.get("source_cad_grid") != [80, 90]
        or model.get("resolved_control_grid") != [80, 90]
        or int(model.get("resolved_matrix_locations_per_layer", -1)) != 7200
        or model.get("construction_method")
        != "direct_ParaStell_0_to_90_degrees"
        or model.get("derived_from_finished_45_degree_CAD") is not False
        or model.get("component_order") != list(EXPECTED_LAYER_ORDER)
        or model.get("complete_geometry_representation")
        != "nine_component_step_set"
        or not isinstance(model.get("combined_step_exported"), bool)
    ):
        raise GeometryProvenanceError(
            "90-degree WISTELL-D geometry must be constructed directly by ParaStell"
        )
    if extent == 45.0 and model.get("source_cad_grid") != [61, 121]:
        raise GeometryProvenanceError(
            "45-degree WISTELL-D source geometry must use the 61x121 grid"
        )
    if model.get("magnet_representation") != (
        "continuous_30_cm_magnet_envelope"
    ):
        raise GeometryProvenanceError("wrong WISTELL-D magnet representation")
    if model.get("global_explicit_coils") is not False:
        raise GeometryProvenanceError("global explicit coils are excluded")
    if extent not in (45.0, 90.0):
        raise GeometryProvenanceError("unsupported WISTELL-D model extent")
    if required_extent_degrees is not None and not math.isclose(
        extent, required_extent_degrees, abs_tol=1e-12
    ):
        raise GeometryProvenanceError("geometry extent does not match request")

    local_frames = manifest.get("local_frames")
    if require_local_frames or (
        isinstance(local_frames, Mapping)
        and local_frames.get("status") == "PASS"
    ):
        _validate_prompt02_local_frames(local_frames)

    if extent == 90.0:
        adjacent = manifest.get("adjacent_pair_audit")
        pairs = (
            adjacent.get("pairs") if isinstance(adjacent, Mapping) else None
        )
        expected_pairs = list(
            zip(EXPECTED_LAYER_ORDER[:-1], EXPECTED_LAYER_ORDER[1:])
        )
        if (
            not isinstance(adjacent, Mapping)
            or adjacent.get("audit_scope") != "adjacent_radial_pairs_only"
            or int(adjacent.get("component_count", -1)) != 9
            or int(adjacent.get("expected_pair_count", -1)) != 8
            or int(adjacent.get("evaluated_pair_count", -1)) != 8
            or int(adjacent.get("boolean_failure_count", -1)) != 0
            or int(adjacent.get("overlap_count", -1)) != 0
            or not isinstance(pairs, list)
            or len(pairs) != 8
            or any(
                (row.get("left"), row.get("right")) != expected_pair
                or not row.get("adjacent_in_radial_stack")
                or row.get("boolean_error") is not None
                or row.get("overlap")
                or not math.isfinite(
                    float(row.get("intersection_volume_cm3", math.nan))
                )
                or float(row.get("intersection_volume_cm3", -1.0)) < 0.0
                for row, expected_pair in zip(pairs, expected_pairs)
            )
        ):
            raise GeometryProvenanceError(
                "direct-90 adjacent radial-pair audit is not accepted"
            )
        thickness = manifest.get("thickness_validation", {})
        if (
            thickness.get("all_thicknesses_strictly_positive") is not True
            or thickness.get("strict_radial_order_at_all_resolved_locations")
            is not True
            or manifest.get("nonadjacent_separation", {}).get(
                "native_all_volume_overlap_gate"
            )
            != "REQUIRED_AFTER_IMPRINTED_DAGMC_EXPORT"
        ):
            raise GeometryProvenanceError(
                "direct-90 nested radial separation proof is incomplete"
            )

        live_vmec = source.get("live_vmec_metadata")
        if (
            not isinstance(live_vmec, Mapping)
            or int(live_vmec.get("nfp", -1)) != 4
            or live_vmec.get("lasym_vmec_logical") is not False
            or live_vmec.get("stellarator_symmetric") is not True
        ):
            raise GeometryProvenanceError(
                "direct-90 live VMEC symmetry proof is incomplete"
            )

        lane = manifest.get("lane")
        runtime = manifest.get("container_runtime")
        runtime_receipt = (
            runtime.get("runtime_receipt")
            if isinstance(runtime, Mapping)
            else None
        )
        attestation = (
            runtime.get("launch_attestation")
            if isinstance(runtime, Mapping)
            else None
        )
        modules = (
            runtime.get("modules") if isinstance(runtime, Mapping) else None
        )
        head_sha = (
            lane.get("head_sha_at_build")
            if isinstance(lane, Mapping)
            else None
        )
        expected_attestation = {
            "PARASTELL_CONTAINER_IMAGE_ID": DIRECT90_DOCKER_IMAGE_ID,
            "PARASTELL_CONTAINER_NETWORK": "none",
            "PARASTELL_MOUNTED_SOURCE_REVISION": head_sha,
            "PARASTELL_DOCKER_ATTESTATION": "direct90-create-only-v1",
        }
        required_modules = {
            "ParaStell",
            "CadQuery",
            "OCP",
            "SciPy",
            "NumPy",
            "cad_to_dagmc",
            "Gmsh",
            "PyMOAB",
            "PyDAGMC",
            "OpenMC",
        }
        if (
            not isinstance(lane, Mapping)
            or not _is_git_sha(head_sha)
            or not isinstance(runtime, Mapping)
            or runtime.get("image_id") != DIRECT90_DOCKER_IMAGE_ID
            or runtime.get("network") != "none"
            or runtime.get("container_execution_proof") != "/.dockerenv"
            or not isinstance(runtime_receipt, Mapping)
            or runtime_receipt.get("schema")
            != "wistell_d.docker_runtime_receipt/v1.0.0"
            or runtime_receipt.get("sha256") != DIRECT90_RUNTIME_RECEIPT_SHA256
            or not isinstance(attestation, Mapping)
            or dict(attestation) != expected_attestation
            or runtime.get("python", {}).get("version") != "3.12.13"
            or set(modules or {}) != required_modules
            or any(
                not isinstance(row, Mapping)
                or not _is_sha256(row.get("sha256"))
                for row in (modules or {}).values()
            )
        ):
            raise GeometryProvenanceError(
                "direct-90 qualified Docker runtime proof is incomplete"
            )

        periodicity = manifest.get("cad_periodicity")
        loci = (
            periodicity.get("generated_loci")
            if isinstance(periodicity, Mapping)
            else None
        )
        cut_faces = (
            periodicity.get("boundary_cut_faces")
            if isinstance(periodicity, Mapping)
            else None
        )
        loci_rows = loci.get("surfaces") if isinstance(loci, Mapping) else None
        if (
            not isinstance(loci, Mapping)
            or loci.get("pass") is not True
            or not math.isfinite(
                float(loci.get("maximum_residual_cm", math.nan))
            )
            or float(loci.get("maximum_residual_cm", math.inf))
            > float(loci.get("tolerance_cm", -math.inf))
            or not isinstance(loci_rows, list)
            or {row.get("surface") for row in loci_rows}
            != set(EXPECTED_LAYER_ORDER)
            or not isinstance(cut_faces, Mapping)
            or set(cut_faces) != set(EXPECTED_LAYER_ORDER)
            or any(
                not isinstance(row, Mapping)
                or row.get("pass") is not True
                or not row.get("phi_0_faces")
                or not row.get("phi_90_faces")
                or float(row.get("area_residual_cm2", math.inf))
                > float(row.get("area_tolerance_cm2", -math.inf))
                for row in (cut_faces or {}).values()
            )
        ):
            raise GeometryProvenanceError(
                "direct-90 CAD periodicity proof is incomplete"
            )

        regression = manifest.get("reference_volume_regression")
        reference_receipt = (
            regression.get("reference_manifest")
            if isinstance(regression, Mapping)
            else None
        )
        regression_components = (
            regression.get("components")
            if isinstance(regression, Mapping)
            else None
        )
        if (
            not isinstance(regression, Mapping)
            or regression.get("pass") is not True
            or not isinstance(reference_receipt, Mapping)
            or reference_receipt.get("sha256")
            != DIRECT90_REFERENCE_MANIFEST_SHA256
            or not Path(str(reference_receipt.get("path", ""))).is_absolute()
            or not isinstance(regression_components, Mapping)
            or set(regression_components) != set(EXPECTED_LAYER_ORDER)
            or any(
                not isinstance(row, Mapping)
                or row.get("pass") is not True
                or float(row.get("reference_volume_cm3", 0.0)) <= 0.0
                or float(row.get("candidate_volume_cm3", 0.0)) <= 0.0
                or float(row.get("relative_difference", math.inf))
                > float(row.get("tolerance", -math.inf))
                or float(row.get("tolerance", math.inf)) > 1.0e-7
                for row in (regression_components or {}).values()
            )
        ):
            raise GeometryProvenanceError(
                "direct-90 reference-volume regression is incomplete"
            )

        physical_measure = manifest.get("full_period_physical_measure")
        half_reference = (
            physical_measure.get("half_period_reference_manifest")
            if isinstance(physical_measure, Mapping)
            else None
        )
        ratios = (
            physical_measure.get("components")
            if isinstance(physical_measure, Mapping)
            else None
        )
        manifest_components = manifest.get("components")
        physical_measure_rows_valid = True
        if (
            not isinstance(ratios, Mapping)
            or set(ratios) != set(EXPECTED_LAYER_ORDER)
            or not isinstance(manifest_components, Mapping)
            or set(manifest_components) != set(EXPECTED_LAYER_ORDER)
        ):
            physical_measure_rows_valid = False
        else:
            for name in EXPECTED_LAYER_ORDER:
                row = ratios[name]
                component = manifest_components[name]
                if not isinstance(row, Mapping) or not isinstance(
                    component, Mapping
                ):
                    physical_measure_rows_valid = False
                    break
                volume_45 = _finite_float(row.get("volume_45_cm3"))
                volume_90 = _finite_float(row.get("volume_90_cm3"))
                stored_ratio = _finite_float(row.get("ratio"))
                expected_ratio = _finite_float(row.get("expected_ratio", 2.0))
                tolerance = _finite_float(row.get("relative_tolerance"))
                component_volume = _finite_float(component.get("volume_cm3"))
                if (
                    row.get("pass") is not True
                    or volume_45 is None
                    or volume_45 <= 0.0
                    or volume_90 is None
                    or volume_90 <= 0.0
                    or stored_ratio is None
                    or stored_ratio <= 0.0
                    or expected_ratio != 2.0
                    or tolerance is None
                    or tolerance <= 0.0
                    or tolerance > 0.025
                    or component_volume is None
                    or component_volume <= 0.0
                    or not math.isclose(
                        stored_ratio,
                        volume_90 / volume_45,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                    or not math.isclose(
                        volume_90,
                        component_volume,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                    or abs(stored_ratio / 2.0 - 1.0) > tolerance
                ):
                    physical_measure_rows_valid = False
                    break
        top_expected_ratio = _finite_float(
            physical_measure.get("expected_ratio")
            if isinstance(physical_measure, Mapping)
            else None
        )
        top_tolerance = _finite_float(
            physical_measure.get("relative_tolerance")
            if isinstance(physical_measure, Mapping)
            else None
        )
        if (
            not isinstance(physical_measure, Mapping)
            or physical_measure.get("pass") is not True
            or top_expected_ratio != 2.0
            or top_tolerance is None
            or top_tolerance <= 0.0
            or top_tolerance > 0.025
            or not isinstance(half_reference, Mapping)
            or half_reference.get("sha256")
            != DIRECT45_REFERENCE_MANIFEST_SHA256
            or not Path(str(half_reference.get("path", ""))).is_absolute()
            or not physical_measure_rows_valid
        ):
            raise GeometryProvenanceError(
                "direct-90 full-period physical-measure proof is incomplete"
            )
    else:
        pairwise = manifest.get("complete_pairwise_audit")
        if not isinstance(pairwise, Mapping):
            raise GeometryProvenanceError(
                "missing complete component-pair audit"
            )
        require_complete_pairwise_acceptance(pairwise)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise GeometryProvenanceError(
            "accepted geometry has no artifact inventory"
        )
    for role, row in artifacts.items():
        if not isinstance(row, Mapping):
            raise GeometryProvenanceError(f"invalid artifact row for {role}")
        path = row.get("path")
        digest = str(row.get("sha256", "")).lower()
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise GeometryProvenanceError(
                f"artifact {role} path is not absolute"
            )
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise GeometryProvenanceError(
                f"artifact {role} has invalid SHA-256"
            )
    if extent == 90.0:
        required_component_roles = {
            f"component_step:{name}" for name in EXPECTED_LAYER_ORDER
        }
        actual_component_roles = {
            role for role in artifacts if role.startswith("component_step:")
        }
        if actual_component_roles != required_component_roles:
            raise GeometryProvenanceError(
                "direct-90 manifest lacks the complete component STEP set"
            )
        if model.get("combined_step_exported") is not (
            "source_step" in artifacts
        ):
            raise GeometryProvenanceError(
                "direct-90 combined STEP declaration contradicts its artifact set"
            )


@runtime_checkable
class GeometryProvider(Protocol):
    """Minimum transport-facing geometry provider contract."""

    def validate(self) -> None: ...

    def component_manifest(self) -> Mapping[str, Any]: ...

    def material_manifest(self) -> Mapping[str, Any]: ...

    def source_domain(self) -> Mapping[str, Any]: ...

    def physical_boundary_roles(self) -> Mapping[str, Any]: ...

    def global_transforms(self) -> Mapping[str, Any]: ...

    def canonical_patch_map(self) -> Sequence[Mapping[str, Any]]: ...

    def local_frames(self) -> Mapping[str, Any]: ...

    def geometry_fingerprint(self) -> str: ...

    def export_or_load_dagmc(self) -> Path: ...


@runtime_checkable
class ParamakGeometryAdapter(Protocol):
    """Optional adapter protocol; ParaStell never imports Paramak."""

    def to_geometry_provider(self) -> GeometryProvider: ...


class WistellDGeometryProvider:
    """Read-only provider for a strictly accepted lane-owned WISTELL-D model."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self._manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def validate(self) -> None:
        validate_wistell_d_manifest(self._manifest)
        row = self._manifest["artifacts"].get("selected_h5m")
        if row is None:
            raise GeometryProvenanceError(
                "no explicitly selected DAGMC artifact"
            )
        path = Path(row["path"]).resolve()
        if not path.is_file():
            raise GeometryProvenanceError(
                "selected DAGMC artifact does not exist"
            )
        if sha256_file(path) != row["sha256"]:
            raise GeometryProvenanceError(
                "selected DAGMC artifact hash mismatch"
            )

    def validate_selected_patch_contract(self) -> None:
        """Require the optional surface/frame contract for selected consumers."""
        validate_wistell_d_manifest(self._manifest, require_local_frames=True)

    def component_manifest(self) -> Mapping[str, Any]:
        return self._manifest["components"]

    def material_manifest(self) -> Mapping[str, Any]:
        return self._manifest["materials"]

    def source_domain(self) -> Mapping[str, Any]:
        return self._manifest["source_domain"]

    def physical_boundary_roles(self) -> Mapping[str, Any]:
        return self._manifest["physical_boundaries"]

    def global_transforms(self) -> Mapping[str, Any]:
        return self._manifest["transforms"]

    def canonical_patch_map(self) -> Sequence[Mapping[str, Any]]:
        return self._manifest["canonical_patch_map"]

    def local_frames(self) -> Mapping[str, Any]:
        return self._manifest.get("local_frames", {})

    def geometry_fingerprint(self) -> str:
        identity = {
            "provider_schema": PROVIDER_SCHEMA,
            "provider_version": PROVIDER_VERSION,
            "source": self._manifest["source"],
            "model": self._manifest["model"],
            "artifacts": self._manifest["artifacts"],
            "transforms": self._manifest["transforms"],
        }
        return canonical_json_sha256(identity)

    def export_or_load_dagmc(self) -> Path:
        """Return only the explicitly selected, hash-verified H5M.

        There is intentionally no export, discovery, or example fallback path.
        """
        self.validate()
        return Path(
            self._manifest["artifacts"]["selected_h5m"]["path"]
        ).resolve()


class ExistingGeometryProvider(WistellDGeometryProvider):
    """Read-only existing STEP/DAGMC provider using the same strict receipt."""


def canonical_patch_instances(
    *, canonical_count: int = 452, instance_ids: Sequence[str] = ("canonical",)
) -> list[dict[str, Any]]:
    """Create stable canonical-control and symmetry-instance identities."""
    if canonical_count <= 0:
        raise ValueError("canonical control count must be positive")
    if not instance_ids or len(set(instance_ids)) != len(instance_ids):
        raise ValueError("symmetry instance IDs must be nonempty and unique")
    return [
        {
            "canonical_control_id": canonical_id,
            "symmetry_instance_id": instance_id,
            "transform_id": (
                "identity"
                if instance_id == "canonical"
                else "half_period_mate"
            ),
        }
        for instance_id in instance_ids
        for canonical_id in range(canonical_count)
    ]
