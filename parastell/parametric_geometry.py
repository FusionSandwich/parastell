"""Device-neutral, fail-closed ParaStell parametric geometry plans.

This module separates reusable model specification and resolved thickness
fields from device-specific reconstruction/acceptance logic.  It intentionally
does not weaken the existing WISTELL-D acceptance contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


PLAN_SCHEMA = "parastell.parametric_geometry_plan/v1.0.0"
MANIFEST_SCHEMA = "parastell.parametric_source_cad/v1.0.0"
SAFETY_POLICY = {
    "ports": False,
    "post_build_transforms": False,
    "imported_physical_solids": False,
    "create_only": True,
}
MAGNET_REPRESENTATIONS = frozenset({"radial_envelope", "swept_filaments"})
SECTOR_MODES = frozenset({"stellarator_half_period", "full_periods", "direct"})
CONTAINER_ATTESTATION = "parastell-parametric-create-only-v1"
RADIAL_BUILD_MODES = frozenset({"monolithic", "isolated_cumulative_shells"})


class ParametricGeometryError(ValueError):
    """Raised when a parametric geometry plan is ambiguous or unsafe."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_source_revision(source_revision: str) -> None:
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ParametricGeometryError("source_revision must be a full Git SHA")


def validate_repository_revision(
    repository_root: str | Path, source_revision: str
) -> dict[str, Any]:
    """Prove that a Docker command binds a clean checkout at the named SHA."""
    _validate_source_revision(source_revision)
    repository_root = Path(repository_root).resolve()
    if not repository_root.is_dir():
        raise FileNotFoundError(repository_root)
    try:
        head = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ParametricGeometryError(
            "repository revision could not be verified"
        ) from error
    if head != source_revision:
        raise ParametricGeometryError(
            f"repository HEAD {head} differs from source_revision {source_revision}"
        )
    if dirty:
        raise ParametricGeometryError(
            "repository must be clean before a Docker geometry build"
        )
    return {
        "repository_root": str(repository_root),
        "head": head,
        "clean": True,
    }


def _positive_integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ParametricGeometryError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ParametricGeometryError(f"{label} must be an integer") from error
    if result != value or result < minimum:
        raise ParametricGeometryError(
            f"{label} must be an integer >= {minimum}"
        )
    return result


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ParametricGeometryError(f"{label} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ParametricGeometryError(f"{label} must be finite and positive")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_exact_keys(
    value: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ParametricGeometryError(
            f"{label} contains unsupported keys: {sorted(unknown)}"
        )


@dataclass(frozen=True)
class GeometryPlan:
    """Validated immutable parametric geometry plan."""

    config_path: Path
    input_root: Path
    config: Mapping[str, Any]
    input_files: Mapping[str, Path]
    input_hashes: Mapping[str, str]
    extent_degrees: float
    field_period_degrees: float
    periodic_boundary_candidate: bool
    plan_sha256: str
    config_sha256: str

    @property
    def grid_shape(self) -> tuple[int, int]:
        grid = self.config["sector"]["cad_grid"]
        return int(grid[0]), int(grid[1])

    @property
    def component_order(self) -> tuple[str, ...]:
        layers = tuple(layer["name"] for layer in self.config["layers"])
        if self.config["magnets"]["representation"] == "swept_filaments":
            return ("chamber", *layers, "magnets")
        return ("chamber", *layers)

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "model_id": self.config["model_id"],
            "device": self.config["device"],
            "plan_sha256": self.plan_sha256,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "input_root": str(self.input_root),
            "inputs": {
                role: {"path": str(self.input_files[role]), "sha256": digest}
                for role, digest in self.input_hashes.items()
            },
            "extent_degrees": self.extent_degrees,
            "field_period_degrees": self.field_period_degrees,
            "periodic_boundary_candidate": self.periodic_boundary_candidate,
            "grid_shape": list(self.grid_shape),
            "component_order": list(self.component_order),
            "magnet_representation": self.config["magnets"]["representation"],
            "radial_build_mode": self.config.get("construction", {}).get(
                "radial_build_mode", "monolithic"
            ),
            "safety": dict(SAFETY_POLICY),
        }


@dataclass(frozen=True)
class ResolvedGeometry:
    """A plan plus final layer matrices used directly by ParaStell."""

    plan: GeometryPlan
    thickness_cm: Mapping[str, np.ndarray]

    def statistics(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "shape": list(values.shape),
                "minimum_cm": float(np.min(values)),
                "maximum_cm": float(np.max(values)),
                "mean_cm": float(np.mean(values)),
                "finite": bool(np.isfinite(values).all()),
                "strictly_positive": bool(np.all(values > 0.0)),
                "poloidally_closed": bool(
                    np.allclose(values[:, 0], values[:, -1], atol=1.0e-10)
                ),
            }
            for name, values in self.thickness_cm.items()
        }


def _validate_inputs(
    config: Mapping[str, Any], input_root: Path, verify_files: bool
) -> tuple[dict[str, Path], dict[str, str]]:
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or "vmec" not in inputs:
        raise ParametricGeometryError("inputs.vmec is required")
    files: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role, row in inputs.items():
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(row, Mapping)
        ):
            raise ParametricGeometryError("input roles must be named objects")
        _require_exact_keys(row, {"path", "sha256"}, f"input {role}")
        relative = Path(str(row.get("path", "")))
        if (
            not str(relative)
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ParametricGeometryError(
                f"input {role} path must be relative to input_root"
            )
        digest = str(row.get("sha256", ""))
        if not _is_sha256(digest):
            raise ParametricGeometryError(f"input {role} SHA-256 is invalid")
        path = (input_root / relative).resolve()
        try:
            path.relative_to(input_root)
        except ValueError as error:
            raise ParametricGeometryError(
                f"input {role} escapes input_root"
            ) from error
        if verify_files:
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256_file(path) != digest:
                raise ParametricGeometryError(f"input {role} hash mismatch")
        files[role] = path
        hashes[role] = digest
    return files, hashes


def _validate_sector(config: Mapping[str, Any]) -> tuple[float, float, bool]:
    sector = config.get("sector")
    if not isinstance(sector, Mapping):
        raise ParametricGeometryError("sector must be an object")
    nfp = _positive_integer(sector.get("n_field_periods"), "n_field_periods")
    stellarator_symmetric = sector.get("stellarator_symmetric")
    if not isinstance(stellarator_symmetric, bool):
        raise ParametricGeometryError("stellarator_symmetric must be boolean")
    start = float(sector.get("start_degrees", math.nan))
    if not math.isfinite(start) or not math.isclose(start, 0.0, abs_tol=1e-12):
        raise ParametricGeometryError(
            "nonzero sector starts are blocked until phase-origin CAD support is qualified"
        )
    mode = sector.get("mode")
    if mode not in SECTOR_MODES:
        raise ParametricGeometryError(f"unsupported sector mode: {mode}")
    common_keys = {
        "mode",
        "start_degrees",
        "n_field_periods",
        "stellarator_symmetric",
        "cad_grid",
    }
    mode_key = {
        "stellarator_half_period": set(),
        "full_periods": {"period_count"},
        "direct": {"extent_degrees"},
    }[mode]
    _require_exact_keys(sector, common_keys | mode_key, "sector")
    period = 360.0 / nfp
    if mode == "stellarator_half_period":
        if not stellarator_symmetric:
            raise ParametricGeometryError(
                "half-period construction requires stellarator symmetry"
            )
        extent = period / 2.0
        openmc_periodic = False
    elif mode == "full_periods":
        periods = _positive_integer(sector.get("period_count"), "period_count")
        extent = periods * period
        if extent > 360.0 + 1.0e-12:
            raise ParametricGeometryError(
                "modeled full periods exceed 360 degrees"
            )
        openmc_periodic = True
    else:
        extent = _positive_float(
            sector.get("extent_degrees"), "extent_degrees"
        )
        if extent > 360.0:
            raise ParametricGeometryError("direct extent exceeds 360 degrees")
        openmc_periodic = False
    cad_grid = sector.get("cad_grid")
    if (
        not isinstance(cad_grid, list)
        or len(cad_grid) != 2
        or any(
            _positive_integer(value, "cad_grid entry", minimum=2) < 2
            for value in cad_grid
        )
    ):
        raise ParametricGeometryError(
            "cad_grid must contain two integers >= 2"
        )
    return extent, period, openmc_periodic


def _validate_layers_and_magnets(config: Mapping[str, Any]) -> None:
    chamber_material = config.get("chamber_material")
    if not isinstance(chamber_material, str) or not chamber_material:
        raise ParametricGeometryError("chamber_material must be explicit")
    layers = config.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ParametricGeometryError("layers must be a nonempty ordered list")
    names = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ParametricGeometryError(f"layer {index} must be an object")
        _require_exact_keys(
            layer, {"name", "material", "thickness"}, f"layer {index}"
        )
        name = layer.get("name")
        material = layer.get("material")
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise ParametricGeometryError(f"layer {index} name is invalid")
        if not isinstance(material, str) or not material:
            raise ParametricGeometryError(f"layer {name} material is invalid")
        if name in names or name == "chamber":
            raise ParametricGeometryError(
                f"layer name is duplicated/reserved: {name}"
            )
        names.append(name)
        thickness = layer.get("thickness")
        if not isinstance(thickness, Mapping):
            raise ParametricGeometryError(f"layer {name} thickness is missing")
        kind = thickness.get("kind")
        if kind == "constant":
            _require_exact_keys(
                thickness, {"kind", "value_cm"}, f"layer {name} thickness"
            )
            _positive_float(thickness.get("value_cm"), f"{name} thickness")
        elif kind == "array":
            _require_exact_keys(
                thickness,
                {"kind", "input", "array_key"},
                f"layer {name} thickness",
            )
            role = thickness.get("input")
            if role not in config["inputs"]:
                raise ParametricGeometryError(
                    f"layer {name} references unknown array input {role}"
                )
        else:
            raise ParametricGeometryError(
                f"layer {name} thickness kind must be constant or array"
            )
    magnets = config.get("magnets")
    if not isinstance(magnets, Mapping):
        raise ParametricGeometryError("magnets must be an object")
    representation = magnets.get("representation")
    if representation not in MAGNET_REPRESENTATIONS:
        raise ParametricGeometryError("unsupported magnet representation")
    if representation == "radial_envelope":
        _require_exact_keys(magnets, {"representation", "layer"}, "magnets")
        layer_name = magnets.get("layer")
        if layer_name not in names or layer_name != names[-1]:
            raise ParametricGeometryError(
                "radial-envelope magnet must identify the final radial layer"
            )
        forbidden = {
            "width_cm",
            "thickness_cm",
            "case_thickness_cm",
            "sample_mod",
            "start_line",
            "scale",
        }
        if forbidden.intersection(magnets):
            raise ParametricGeometryError(
                "radial-envelope magnets cannot define swept-filament fields"
            )
    else:
        _require_exact_keys(
            magnets,
            {
                "representation",
                "material",
                "width_cm",
                "thickness_cm",
                "case_thickness_cm",
                "sample_mod",
                "start_line",
                "scale",
            },
            "magnets",
        )
        if "layer" in magnets:
            raise ParametricGeometryError(
                "swept-filament magnets cannot also select a radial layer"
            )
        if "coils" not in config["inputs"]:
            raise ParametricGeometryError(
                "swept-filament magnets require a hash-bound coils input"
            )
        material = magnets.get("material")
        if not isinstance(material, str) or not material:
            raise ParametricGeometryError(
                "swept-filament magnets require an explicit material"
            )
        for key in ("width_cm", "thickness_cm"):
            _positive_float(magnets.get(key), f"magnets.{key}")
        _positive_integer(magnets.get("sample_mod", 1), "magnets.sample_mod")
        _positive_integer(
            magnets.get("start_line", 3), "magnets.start_line", minimum=0
        )
        _positive_float(magnets.get("scale", 100.0), "magnets.scale")
        case_thickness = float(magnets.get("case_thickness_cm", 0.0))
        if not math.isfinite(case_thickness) or case_thickness < 0.0:
            raise ParametricGeometryError(
                "magnets.case_thickness_cm must be finite and nonnegative"
            )
        if case_thickness > 0.0 and 2.0 * case_thickness >= min(
            float(magnets["width_cm"]), float(magnets["thickness_cm"])
        ):
            raise ParametricGeometryError(
                "twice the casing thickness must be smaller than both coil dimensions"
            )


def _validate_runtime_and_safety(config: Mapping[str, Any]) -> None:
    if config.get("safety") != SAFETY_POLICY:
        raise ParametricGeometryError(
            "ports, transforms, imported solids, and overwrite paths are forbidden"
        )
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("kind") != "docker":
        raise ParametricGeometryError("runtime.kind must be docker")
    _require_exact_keys(
        runtime,
        {"kind", "image_id", "network", "cpus", "memory_bytes"},
        "runtime",
    )
    image = str(runtime.get("image_id", ""))
    if not image.startswith("sha256:") or not _is_sha256(image[7:]):
        raise ParametricGeometryError(
            "Docker image must use an immutable image ID"
        )
    if runtime.get("network") != "none":
        raise ParametricGeometryError(
            "Docker geometry builds require network=none"
        )
    _positive_float(runtime.get("cpus"), "runtime.cpus")
    _positive_integer(runtime.get("memory_bytes"), "runtime.memory_bytes")
    source = config.get("source_mesh", {})
    if not isinstance(source, Mapping) or not isinstance(
        source.get("enabled", False), bool
    ):
        raise ParametricGeometryError("source_mesh.enabled must be boolean")
    allowed_source = {"enabled"}
    if source.get("enabled"):
        allowed_source |= {
            "radial_points",
            "poloidal_points",
            "toroidal_points",
        }
    _require_exact_keys(source, allowed_source, "source_mesh")
    if source.get("enabled"):
        for key in ("radial_points", "poloidal_points", "toroidal_points"):
            _positive_integer(source.get(key), f"source_mesh.{key}", minimum=2)
    construction = config.get("construction", {})
    if not isinstance(construction, Mapping):
        raise ParametricGeometryError("construction must be an object")
    _require_exact_keys(construction, {"radial_build_mode"}, "construction")
    mode = construction.get("radial_build_mode", "monolithic")
    if mode not in RADIAL_BUILD_MODES:
        raise ParametricGeometryError(
            f"unsupported radial-build construction mode: {mode}"
        )


def load_plan(
    config_path: str | Path,
    input_root: str | Path,
    *,
    verify_inputs: bool = True,
) -> GeometryPlan:
    """Load and validate one generic create-only geometry plan."""
    config_path = Path(config_path).resolve()
    input_root = Path(input_root).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if verify_inputs and not input_root.is_dir():
        raise FileNotFoundError(input_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ParametricGeometryError("plan must be a JSON object")
    _require_exact_keys(
        config,
        {
            "schema",
            "model_id",
            "device",
            "inputs",
            "sector",
            "wall_s",
            "chamber_material",
            "layers",
            "magnets",
            "source_mesh",
            "construction",
            "runtime",
            "safety",
        },
        "plan",
    )
    if config.get("schema") != PLAN_SCHEMA:
        raise ParametricGeometryError("unsupported parametric geometry schema")
    for key in ("model_id", "device"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ParametricGeometryError(f"{key} must be a nonempty string")
    files, hashes = _validate_inputs(config, input_root, verify_inputs)
    extent, period, openmc_periodic = _validate_sector(config)
    wall_s = _positive_float(config.get("wall_s"), "wall_s")
    if wall_s < 1.0:
        raise ParametricGeometryError("wall_s must be at least 1.0")
    _validate_layers_and_magnets(config)
    _validate_runtime_and_safety(config)
    normalized = json.loads(json.dumps(config, allow_nan=False))
    return GeometryPlan(
        config_path=config_path,
        input_root=input_root,
        config=normalized,
        input_files=MappingProxyType(files),
        input_hashes=MappingProxyType(hashes),
        extent_degrees=extent,
        field_period_degrees=period,
        periodic_boundary_candidate=openmc_periodic,
        plan_sha256=canonical_json_sha256(normalized),
        config_sha256=sha256_file(config_path),
    )


def validate_vmec_metadata(
    plan: GeometryPlan, *, n_field_periods: int, lasym: bool
) -> dict[str, Any]:
    """Cross-check plan symmetry against live VMEC metadata.

    VMEC ``lasym=False`` means stellarator symmetry is present; the explicit
    result avoids propagating that historically confusing inverted name.
    """
    expected_nfp = int(plan.config["sector"]["n_field_periods"])
    if n_field_periods != expected_nfp:
        raise ParametricGeometryError(
            f"live VMEC nfp={n_field_periods} differs from plan {expected_nfp}"
        )
    stellarator_symmetric = not bool(lasym)
    expected_symmetry = bool(plan.config["sector"]["stellarator_symmetric"])
    if stellarator_symmetric != expected_symmetry:
        raise ParametricGeometryError(
            "live VMEC stellarator symmetry differs from the plan"
        )
    return {
        "n_field_periods": n_field_periods,
        "lasym_vmec_logical": bool(lasym),
        "stellarator_symmetric": stellarator_symmetric,
        "pass": True,
    }


def read_vmec_metadata(plan: GeometryPlan) -> dict[str, Any]:
    """Read only nfp/lasym from the hash-bound VMEC input."""
    from scipy.io import netcdf_file

    with netcdf_file(plan.input_files["vmec"], mode="r", mmap=False) as data:
        nfp = int(np.asarray(data.variables["nfp"].data))
        lasym = bool(np.asarray(data.variables["lasym__logical__"].data))
    return validate_vmec_metadata(plan, n_field_periods=nfp, lasym=lasym)


def _load_array(path: Path, key: str | None) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            if not key or key not in payload.files:
                raise ParametricGeometryError(
                    f"NPZ input requires an existing array_key: {path}"
                )
            values = np.asarray(payload[key], dtype=float)
        finally:
            payload.close()
        return values
    if key:
        raise ParametricGeometryError("array_key is only valid for NPZ inputs")
    return np.asarray(payload, dtype=float)


def resolve_geometry(plan: GeometryPlan) -> ResolvedGeometry:
    """Resolve every layer to the exact direct ParaStell CAD grid."""
    shape = plan.grid_shape
    matrices: dict[str, np.ndarray] = {}
    for layer in plan.config["layers"]:
        thickness = layer["thickness"]
        if thickness["kind"] == "constant":
            values = np.full(shape, float(thickness["value_cm"]))
        else:
            role = thickness["input"]
            values = _load_array(
                plan.input_files[role], thickness.get("array_key")
            )
        if values.shape != shape:
            raise ParametricGeometryError(
                f"layer {layer['name']} shape {values.shape} != {shape}"
            )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ParametricGeometryError(
                f"layer {layer['name']} contains nonfinite/nonpositive thickness"
            )
        if not np.allclose(values[:, 0], values[:, -1], atol=1.0e-10):
            raise ParametricGeometryError(
                f"layer {layer['name']} is not poloidally closed"
            )
        mode = plan.config["sector"]["mode"]
        if (
            mode == "full_periods" or math.isclose(plan.extent_degrees, 360.0)
        ) and not np.allclose(values[0, :], values[-1, :], atol=1.0e-10):
            raise ParametricGeometryError(
                f"layer {layer['name']} is not toroidally periodic"
            )
        if mode == "stellarator_half_period":
            for endpoint in (values[0, :], values[-1, :]):
                if not np.allclose(endpoint, endpoint[::-1], atol=1.0e-10):
                    raise ParametricGeometryError(
                        f"layer {layer['name']} violates half-period reflection symmetry"
                    )
        frozen = np.array(values, dtype=float, copy=True)
        frozen.setflags(write=False)
        matrices[layer["name"]] = frozen
    return ResolvedGeometry(plan=plan, thickness_cm=matrices)


def _source_tree_rows(repository_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every Python source file executable by the geometry builder."""
    paths = sorted((repository_root / "parastell").rglob("*.py"))
    paths.append(repository_root / "scripts" / "build_parametric_geometry.py")
    rows = {}
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(repository_root).as_posix()
        rows[relative] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return rows


def _source_tree_sha256(rows: Mapping[str, Mapping[str, Any]]) -> str:
    return canonical_json_sha256(
        {relative: row["sha256"] for relative, row in rows.items()}
    )


def _validate_plan_snapshot(resolved: ResolvedGeometry) -> None:
    plan = resolved.plan
    if sha256_file(plan.config_path) != plan.config_sha256:
        raise ParametricGeometryError("plan configuration changed after load")
    if canonical_json_sha256(plan.config) != plan.plan_sha256:
        raise ParametricGeometryError("resolved plan was mutated after load")
    expected_files = {
        role: (plan.input_root / row["path"]).resolve()
        for role, row in plan.config["inputs"].items()
    }
    expected_hashes = {
        role: row["sha256"] for role, row in plan.config["inputs"].items()
    }
    if (
        dict(plan.input_files) != expected_files
        or dict(plan.input_hashes) != expected_hashes
    ):
        raise ParametricGeometryError(
            "plan input path/hash mappings differ from canonical configuration"
        )
    current_inputs = {
        role: sha256_file(path) for role, path in expected_files.items()
    }
    if current_inputs != expected_hashes:
        raise ParametricGeometryError("geometry input changed after plan load")
    expected = resolve_geometry(plan)
    if tuple(expected.thickness_cm) != tuple(resolved.thickness_cm):
        raise ParametricGeometryError("resolved layer inventory was mutated")
    for name, values in resolved.thickness_cm.items():
        if values.flags.writeable or not np.array_equal(
            values, expected.thickness_cm[name]
        ):
            raise ParametricGeometryError(
                f"resolved layer {name} is mutable or changed after validation"
            )


def validate_container_attestation(
    plan: GeometryPlan,
    source_revision: str,
    *,
    repository_root: str | Path,
    container_marker: str | Path = "/.dockerenv",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Require the exact networkless Docker launch contract inside a build."""
    environment = os.environ if environ is None else environ
    if not Path(container_marker).is_file():
        raise ParametricGeometryError(
            "source CAD construction is permitted only inside Docker"
        )
    expected = {
        "PARASTELL_PARAMETRIC_ATTESTATION": CONTAINER_ATTESTATION,
        "PARASTELL_PARAMETRIC_IMAGE_ID": plan.config["runtime"]["image_id"],
        "PARASTELL_PARAMETRIC_NETWORK": "none",
        "PARASTELL_PARAMETRIC_SOURCE_REVISION": source_revision,
    }
    for name, value in expected.items():
        if environment.get(name) != value:
            raise ParametricGeometryError(
                f"container attestation mismatch for {name}"
            )
    declared_tree_sha256 = environment.get(
        "PARASTELL_PARAMETRIC_SOURCE_TREE_SHA256", ""
    )
    if not _is_sha256(declared_tree_sha256):
        raise ParametricGeometryError(
            "container source-tree attestation is missing/invalid"
        )
    actual_rows = _source_tree_rows(Path(repository_root).resolve())
    actual_tree_sha256 = _source_tree_sha256(actual_rows)
    if declared_tree_sha256 != actual_tree_sha256:
        raise ParametricGeometryError(
            "container source tree differs from host attestation"
        )
    return {
        "environment": expected,
        "source_tree_sha256": actual_tree_sha256,
        "implementation_files": actual_rows,
    }


def _artifact_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in {
            "BUILD_STARTED.json",
            "PARAMETRIC_GEOMETRY_MANIFEST.json",
            "BUILD_FAILED.json",
        }:
            rows[path.name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return rows


def _solid_evidence(solid: Any, label: str) -> dict[str, Any]:
    solids = list(solid.Solids())
    valid = bool(solid.isValid())
    volume = float(solid.Volume())
    if (
        len(solids) != 1
        or not valid
        or not math.isfinite(volume)
        or volume <= 0.0
    ):
        raise RuntimeError(
            f"{label} is not one valid positive-volume CAD solid"
        )
    return {"solid_count": 1, "brep_valid": True, "volume_cm3": volume}


def _matrix_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _export_step_solid(solid: Any, path: Path) -> None:
    import cadquery as cq

    cq.exporters.export(solid, str(path))


def _construct_isolated_cumulative_shells(
    *,
    ps: Any,
    plan: GeometryPlan,
    resolved: ResolvedGeometry,
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Construct exact cumulative boundaries while bounding live CAD.

    ParaStell defines every radial surface from the cumulative thickness
    matrix. For layer ``k`` the isolated build therefore uses the same
    cumulative matrix through ``k-1`` as an unexported interior and the exact
    layer-``k`` matrix as the target shell. This changes object lifetime only;
    it does not change either bounding surface.
    """
    toroidal = np.linspace(0.0, plan.extent_degrees, plan.grid_shape[0])
    poloidal = np.linspace(0.0, 360.0, plan.grid_shape[1])
    cumulative = np.zeros(plan.grid_shape, dtype=float)
    evidence: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []
    for index, layer in enumerate(config["layers"]):
        name = layer["name"]
        target = resolved.thickness_cm[name]
        radial_build: dict[str, dict[str, Any]] = {}
        if index:
            radial_build["__cumulative_interior__"] = {
                "thickness_matrix": np.array(cumulative, copy=True),
                "mat_tag": config["chamber_material"],
            }
        radial_build[name] = {
            "thickness_matrix": target,
            "mat_tag": layer["material"],
        }
        stellarator = ps.Stellarator(str(plan.input_files["vmec"]))
        stellarator.construct_invessel_build(
            toroidal,
            poloidal,
            float(config["wall_s"]),
            radial_build,
            split_chamber=False,
            chamber_mat_tag=config["chamber_material"],
            num_ribs=plan.grid_shape[0],
            num_rib_pts=plan.grid_shape[1],
        )
        expected = (
            ("chamber", name)
            if index == 0
            else ("chamber", "__cumulative_interior__", name)
        )
        actual = tuple(stellarator.invessel_build.Components)
        if actual != expected:
            raise RuntimeError(
                f"isolated stage {name} components {actual} != {expected}"
            )
        if index == 0:
            chamber = stellarator.invessel_build.Components["chamber"]
            evidence["chamber"] = _solid_evidence(chamber, "component chamber")
            _export_step_solid(chamber, output_root / "chamber.step")
        target_solid = stellarator.invessel_build.Components[name]
        evidence[name] = _solid_evidence(target_solid, f"component {name}")
        _export_step_solid(target_solid, output_root / f"{name}.step")
        cumulative_before = np.array(cumulative, copy=True)
        cumulative += np.asarray(target, dtype=float)
        stages.append(
            {
                "stage_index": index,
                "component": name,
                "cumulative_inner_matrix_sha256": _matrix_sha256(
                    cumulative_before
                ),
                "layer_matrix_sha256": _matrix_sha256(target),
                "cumulative_outer_matrix_sha256": _matrix_sha256(cumulative),
                "unexported_helper_component": (
                    "__cumulative_interior__" if index else None
                ),
                "exported_components": (
                    ["chamber", name] if index == 0 else [name]
                ),
            }
        )
        del target_solid, stellarator
        gc.collect()
    return evidence, stages


def _filament_identity(coords: Any) -> dict[str, Any]:
    """Return a coordinate-order-preserving identity for one source filament."""
    array = np.asarray(coords, dtype="<f8")
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or len(array) < 4
        or np.any(~np.isfinite(array))
    ):
        raise RuntimeError("magnet source filament coordinates are invalid")
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    centroid_points = array[:-1] if np.allclose(array[0], array[-1]) else array
    centroid = np.mean(centroid_points, axis=0)
    return {
        "coordinate_count": int(len(array)),
        "coordinates_sha256": digest.hexdigest(),
        "centroid_cm": centroid.tolist(),
        "centroid_toroidal_angle_degrees": float(
            np.degrees(np.arctan2(centroid[1], centroid[0])) % 360.0
        ),
    }


def build_source_cad(
    resolved: ResolvedGeometry,
    output_root: str | Path,
    *,
    source_revision: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Build direct source CAD using public ParaStell APIs.

    The output is geometry-only and remains transport-ineligible until the
    separate imprint, native DAGMC, overlap, source-domain, and OpenMC gates.
    """
    output_root = Path(output_root).resolve()
    repository_root = Path(repository_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    _validate_source_revision(source_revision)
    _validate_plan_snapshot(resolved)
    attestation = validate_container_attestation(
        resolved.plan,
        source_revision,
        repository_root=repository_root,
    )
    before = {
        role: sha256_file(path)
        for role, path in resolved.plan.input_files.items()
    }
    if before != dict(resolved.plan.input_hashes):
        raise ParametricGeometryError(
            "geometry inputs changed before CAD construction"
        )
    output_root.mkdir(parents=False)
    start_receipt = {
        "schema": "parastell.parametric_build_started/v1.0.0",
        "created_utc": _utc_now(),
        "plan": resolved.plan.receipt(),
        "source_revision": source_revision,
        "process_id": os.getpid(),
    }
    (output_root / "BUILD_STARTED.json").write_text(
        json.dumps(start_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        import parastell.parastell as ps

        plan = resolved.plan
        config = plan.config
        live_vmec = read_vmec_metadata(plan)
        expected_invessel = (
            "chamber",
            *(layer["name"] for layer in config["layers"]),
        )
        radial_build_mode = config.get("construction", {}).get(
            "radial_build_mode", "monolithic"
        )
        construction_stages: list[dict[str, Any]] = []
        if radial_build_mode == "isolated_cumulative_shells":
            component_evidence, construction_stages = (
                _construct_isolated_cumulative_shells(
                    ps=ps,
                    plan=plan,
                    resolved=resolved,
                    config=config,
                    output_root=output_root,
                )
            )
            actual_invessel = expected_invessel
            stellarator = ps.Stellarator(str(plan.input_files["vmec"]))
        else:
            stellarator = ps.Stellarator(str(plan.input_files["vmec"]))
            toroidal = np.linspace(
                0.0, plan.extent_degrees, plan.grid_shape[0]
            )
            poloidal = np.linspace(0.0, 360.0, plan.grid_shape[1])
            radial_build = {
                layer["name"]: {
                    "thickness_matrix": resolved.thickness_cm[layer["name"]],
                    "mat_tag": layer["material"],
                }
                for layer in config["layers"]
            }
            stellarator.construct_invessel_build(
                toroidal,
                poloidal,
                float(config["wall_s"]),
                radial_build,
                split_chamber=False,
                chamber_mat_tag=config["chamber_material"],
                num_ribs=plan.grid_shape[0],
                num_rib_pts=plan.grid_shape[1],
            )
            actual_invessel = tuple(stellarator.invessel_build.Components)
            if actual_invessel != expected_invessel:
                raise RuntimeError(
                    f"actual in-vessel components {actual_invessel} != {expected_invessel}"
                )
            component_evidence = {}
            for name, solid in stellarator.invessel_build.Components.items():
                component_evidence[name] = _solid_evidence(
                    solid, f"component {name}"
                )
            stellarator.export_invessel_build_step(export_dir=str(output_root))
        expected_steps = {f"{name}.step" for name in expected_invessel}
        magnets = config["magnets"]
        magnet_inventory = {
            "representation": magnets["representation"],
            "material": (
                config["layers"][-1]["material"]
                if magnets["representation"] == "radial_envelope"
                else magnets["material"]
            ),
        }
        if magnets["representation"] == "swept_filaments":
            stellarator.construct_magnets_from_filaments(
                str(plan.input_files["coils"]),
                float(magnets["width_cm"]),
                float(magnets["thickness_cm"]),
                plan.extent_degrees,
                sample_mod=int(magnets.get("sample_mod", 1)),
                start_line=int(magnets.get("start_line", 3)),
                scale=float(magnets.get("scale", 100.0)),
                mat_tag=magnets["material"],
                case_thickness=float(magnets.get("case_thickness_cm", 0.0)),
            )
            stellarator.export_magnets_step(
                filename="magnet_set", export_dir=str(output_root)
            )
            expected_steps.add("magnet_set.step")
            grouped_coil_solids = stellarator.magnet_set.coil_solids
            flat_coil_solids = stellarator.magnet_set.all_coil_solids
            if not grouped_coil_solids or not flat_coil_solids:
                raise RuntimeError("swept-filament build produced zero coils")
            if any(len(solids) != 1 for solids in grouped_coil_solids):
                raise RuntimeError(
                    "homogenized swept-coil mode requires one solid per coil"
                )
            magnet_coils = stellarator.magnet_set.magnet_coils
            if len(magnet_coils) != len(grouped_coil_solids):
                raise RuntimeError(
                    "coil solids lost their source-filament identity"
                )
            magnet_rows = []
            for index, (coil, solids) in enumerate(
                zip(magnet_coils, grouped_coil_solids)
            ):
                row = {
                    "magnet_id": f"magnet-{index:04d}",
                    "material": magnets["material"],
                    "casing_winding_split": False,
                    "cross_section_cm": {
                        "width": float(magnets["width_cm"]),
                        "thickness": float(magnets["thickness_cm"]),
                    },
                    **_filament_identity(coil.filament.coords),
                    **_solid_evidence(solids[0], f"magnet solid {index}"),
                }
                magnet_rows.append(row)
            magnet_inventory.update(
                {
                    "coil_count": len(grouped_coil_solids),
                    "solid_count": len(flat_coil_solids),
                    "one_homogenized_solid_per_coil": True,
                    "casing_winding_split": False,
                    "magnets": magnet_rows,
                }
            )
        source = config.get("source_mesh", {})
        if source.get("enabled") is True:
            stellarator.construct_source_mesh(
                np.linspace(0.0, 1.0, int(source["radial_points"])),
                np.linspace(0.0, 360.0, int(source["poloidal_points"])),
                np.linspace(
                    0.0,
                    plan.extent_degrees,
                    int(source["toroidal_points"]),
                ),
            )
            stellarator.export_source_mesh(
                filename="source_mesh", export_dir=str(output_root)
            )
            source_path = output_root / "source_mesh.h5m"
            if not source_path.is_file() or source_path.stat().st_size <= 0:
                raise RuntimeError(
                    "source mesh export did not produce a nonempty H5M"
                )
        for filename in sorted(expected_steps):
            path = output_root / filename
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(
                    f"expected nonempty source-CAD artifact is absent: {filename}"
                )
        actual_steps = {path.name for path in output_root.glob("*.step")}
        if actual_steps != expected_steps:
            raise RuntimeError(
                f"emitted STEP inventory {sorted(actual_steps)} != {sorted(expected_steps)}"
            )
        np.savez(
            output_root / "resolved_thickness_arrays_cm.npz",
            **resolved.thickness_cm,
        )
        after = {
            role: sha256_file(path) for role, path in plan.input_files.items()
        }
        if before != after or before != dict(plan.input_hashes):
            raise RuntimeError(
                "parametric geometry inputs changed during build"
            )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING",
            "created_utc": _utc_now(),
            "transport_eligible": False,
            "plan": plan.receipt(),
            "source_revision": source_revision,
            "container_attestation": attestation,
            "live_vmec_metadata": live_vmec,
            "resolved_layers": resolved.statistics(),
            "actual_invessel_component_order": list(actual_invessel),
            "radial_build_mode": radial_build_mode,
            "radial_build_construction_stages": construction_stages,
            "component_evidence": component_evidence,
            "magnet_inventory": magnet_inventory,
            "artifacts": _artifact_rows(output_root),
            "required_successor_gates": [
                "COMPLETE_SOURCE_CAD_PAIR_AUDIT",
                "IMPRINTED_DAGMC_EXPORT",
                "NATIVE_WATERTIGHT_OVERLAP_AND_SENSES",
                "SOURCE_DOMAIN_CONTAINMENT",
                "OPENMC_0_16_TWO_SEED_NAVIGATION",
            ],
        }
        manifest_path = output_root / "PARAMETRIC_GEOMETRY_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception as error:
        failure_path = output_root / "BUILD_FAILED.json"
        if not failure_path.exists():
            failure_path.write_text(
                json.dumps(
                    {
                        "schema": "parastell.parametric_build_failure/v1.0.0",
                        "created_utc": _utc_now(),
                        "transport_eligible": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise


def docker_build_argv(
    plan: GeometryPlan,
    output_root: str | Path,
    *,
    repository_root: str | Path,
    source_revision: str,
) -> list[str]:
    """Return a shell-free, network-disabled Docker source-CAD command."""
    output_root = Path(output_root).resolve()
    repository_root = Path(repository_root).resolve()
    revision_receipt = validate_repository_revision(
        repository_root, source_revision
    )
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-only output root exists: {output_root}")
    _validate_source_revision(source_revision)
    if not output_root.parent.is_dir() or not repository_root.is_dir():
        raise FileNotFoundError("output parent and repository root must exist")
    runtime = plan.config["runtime"]
    config_parent = plan.config_path.parent
    source_tree_sha256 = _source_tree_sha256(
        _source_tree_rows(repository_root)
    )
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        str(runtime["cpus"]),
        "--memory",
        str(runtime["memory_bytes"]),
        "--memory-swap",
        str(runtime["memory_bytes"]),
        "--env",
        f"PARASTELL_PARAMETRIC_ATTESTATION={CONTAINER_ATTESTATION}",
        "--env",
        f"PARASTELL_PARAMETRIC_IMAGE_ID={runtime['image_id']}",
        "--env",
        "PARASTELL_PARAMETRIC_NETWORK=none",
        "--env",
        f"PARASTELL_PARAMETRIC_SOURCE_REVISION={revision_receipt['head']}",
        "--env",
        f"PARASTELL_PARAMETRIC_SOURCE_TREE_SHA256={source_tree_sha256}",
        "--mount",
        f"type=bind,source={repository_root},target=/repo,readonly",
        "--mount",
        f"type=bind,source={plan.input_root},target=/inputs,readonly",
        "--mount",
        f"type=bind,source={config_parent},target=/config,readonly",
        "--mount",
        f"type=bind,source={output_root.parent},target=/outputs",
        runtime["image_id"],
        "python",
        "/repo/scripts/build_parametric_geometry.py",
        "build-source-cad",
        "--config",
        f"/config/{plan.config_path.name}",
        "--input-root",
        "/inputs",
        "--output-root",
        f"/outputs/{output_root.name}",
        "--source-revision",
        revision_receipt["head"],
    ]


def execute_docker_build(argv: list[str]) -> int:
    """Execute an already validated Docker argv without a shell."""
    completed = subprocess.run(argv, check=False)
    return int(completed.returncode)
