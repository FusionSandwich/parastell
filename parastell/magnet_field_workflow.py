"""Restartable, hash-bound stages for magnet-radiation production.

This orchestration layer provides small restart-safe stage contracts around
geometry, transport preparation and execution, post-processing, and
diagnostics.  Transport is still performed only by an explicitly requested
execution stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .production_handoff import validate_no_port_configuration


SCHEMA = "parastell.magnet_radiation_stage/v1.0.0"
WORKFLOW_SCHEMA = "parastell.magnet_radiation_workflow/v1.0.0"


class StageStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    STALE_INPUT = "STALE_INPUT"


class StaleStageError(RuntimeError):
    """Raised when an existing stage was built from incompatible inputs."""


STAGES = (
    "validate_inputs",
    "build_source",
    "build_source_convergence_ladder",
    "qualify_source_convergence",
    "build_geometry",
    "validate_geometry",
    "inventory_magnets",
    "build_tally_meshes",
    "build_weight_window_mesh",
    "prepare_unbiased_model",
    "run_unbiased_model",
    "run_unbiased_qualification_campaign",
    "qualify_production_statistics",
    "prepare_weight_window_generation",
    "run_weight_window_generation",
    "run_weight_window_qualification_campaign",
    "qualify_weight_windows",
    "prepare_production_model",
    "postprocess",
    "render_diagnostics",
    "export_bundle",
)


STAGE_DEPENDENCIES = {
    "build_source": ("validate_inputs",),
    "build_source_convergence_ladder": ("validate_inputs",),
    "qualify_source_convergence": (
        "build_source",
        "build_source_convergence_ladder",
    ),
    "build_geometry": ("validate_inputs",),
    "validate_geometry": ("build_geometry",),
    "inventory_magnets": ("validate_geometry",),
    "build_tally_meshes": ("inventory_magnets",),
    "build_weight_window_mesh": ("inventory_magnets",),
    "prepare_unbiased_model": (
        "build_source",
        "inventory_magnets",
        "build_tally_meshes",
    ),
    "run_unbiased_model": ("prepare_unbiased_model",),
    "run_unbiased_qualification_campaign": ("prepare_unbiased_model",),
    "qualify_production_statistics": ("run_unbiased_qualification_campaign",),
    "prepare_weight_window_generation": (
        "prepare_unbiased_model",
        "build_weight_window_mesh",
    ),
    "run_weight_window_generation": ("prepare_weight_window_generation",),
    "run_weight_window_qualification_campaign": (
        "run_unbiased_model",
        "run_weight_window_generation",
    ),
    "qualify_weight_windows": ("run_weight_window_qualification_campaign",),
    "prepare_production_model": (
        "prepare_unbiased_model",
        "qualify_production_statistics",
        "qualify_weight_windows",
    ),
    "postprocess": ("run_unbiased_model",),
    "render_diagnostics": ("postprocess",),
    "export_bundle": ("postprocess",),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    if resolved.is_file():
        return {
            "path": str(resolved),
            "exists": True,
            "kind": "file",
            "size_bytes": resolved.stat().st_size,
            "sha256": _file_hash(resolved),
        }
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    entries = [
        {
            "path": item.relative_to(resolved).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _file_hash(item),
        }
        for item in files
    ]
    return {
        "path": str(resolved),
        "exists": True,
        "kind": "directory",
        "file_count": len(entries),
        "tree_sha256": _canonical_hash(entries),
    }


def _resolve_paths(value: Any, base: Path) -> Any:
    if isinstance(value, Mapping):
        return {key: _resolve_paths(item, base) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if expanded.startswith("./") or expanded.startswith("../"):
            return str((base / expanded).resolve())
        return expanded
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]):
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _replace_strings(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _replace_strings(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def _inside_git_checkout(path: Path) -> bool:
    current = path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return True
    return False


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: StageStatus
    manifest_path: Path
    reused: bool
    manifest: Mapping[str, Any]


class MagnetRadiationWorkflow:
    """Hash all stage boundaries and reject incompatible resumptions."""

    def __init__(self, config: Mapping[str, Any], *, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = _resolve_paths(dict(config), self.config_path.parent)
        validate_no_port_configuration(self.config)
        output = self.config.get("output_directory")
        if not output:
            raise ValueError(
                "workflow configuration requires output_directory"
            )
        if "$" in str(output):
            raise ValueError(
                "output_directory contains an unresolved environment variable"
            )
        self.output_directory = Path(output).resolve()
        if _inside_git_checkout(self.output_directory) and not bool(
            self.config.get("allow_artifacts_inside_git_checkout", False)
        ):
            raise ValueError(
                "output_directory must be outside a Git checkout unless explicitly allowed"
            )
        self.manifest_directory = self.output_directory / "stage_manifests"

    @classmethod
    def from_json(cls, path: str | Path) -> "MagnetRadiationWorkflow":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if "template" in data:
            template_path = (source.parent / data["template"]).resolve()
            template = json.loads(template_path.read_text(encoding="utf-8"))
            template = _replace_strings(
                template, data.get("path_replacements", {})
            )
            override = {
                key: value
                for key, value in data.items()
                if key not in {"template", "path_replacements"}
            }
            data = _deep_merge(template, override)
            data["template_binding"] = {
                "path": str(template_path),
                "sha256": _file_hash(template_path),
            }
        return cls(data, config_path=source)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MagnetRadiationWorkflow":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to read YAML configuration"
            ) from exc
        source = Path(path)
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("workflow YAML root must be a mapping")
        return cls(data, config_path=source)

    @classmethod
    def from_file(cls, path: str | Path) -> "MagnetRadiationWorkflow":
        source = Path(path)
        return (
            cls.from_json(source)
            if source.suffix.lower() == ".json"
            else cls.from_yaml(source)
        )

    @property
    def manifest_path(self) -> Path:
        return self.output_directory / "workflow_manifest.json"

    def validate_configuration(self) -> dict[str, Any]:
        """Validate the declared stage graph and artifact containment."""
        configured = self.config.get("stages", {})
        if not isinstance(configured, Mapping):
            raise ValueError("workflow stages must be a mapping")
        unknown = set(configured) - set(STAGES)
        if unknown:
            raise ValueError(
                f"workflow contains unknown stages: {sorted(unknown)}"
            )
        declared_outputs: dict[Path, str] = {}
        for stage, value in configured.items():
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"stage {stage!r} configuration must be a mapping"
                )
            for field in ("inputs", "outputs"):
                paths = value.get(field, [])
                if not isinstance(paths, list) or any(
                    not isinstance(path, str) or not path for path in paths
                ):
                    raise ValueError(
                        f"stage {stage!r} {field} must be a path list"
                    )
                if len(paths) != len(set(paths)):
                    raise ValueError(
                        f"stage {stage!r} repeats a declared {field} path"
                    )
            for raw_path in value.get("outputs", []):
                path = Path(raw_path).resolve()
                try:
                    path.relative_to(self.output_directory)
                except ValueError as exc:
                    raise ValueError(
                        f"stage {stage!r} output is outside output_directory: {path}"
                    ) from exc
                if path in declared_outputs:
                    raise ValueError(
                        f"stages {declared_outputs[path]!r} and {stage!r} "
                        f"declare the same output {path}"
                    )
                declared_outputs[path] = stage
            missing_dependencies = set(
                STAGE_DEPENDENCIES.get(stage, ())
            ) - set(configured)
            if missing_dependencies:
                raise ValueError(
                    f"stage {stage!r} has undeclared dependencies: "
                    f"{sorted(missing_dependencies)}"
                )
        return {
            "status": "PASS",
            "configured_stage_count": len(configured),
            "configured_stages": [
                stage for stage in STAGES if stage in configured
            ],
            "declared_output_count": len(declared_outputs),
            "all_outputs_outside_git_checkout": all(
                not _inside_git_checkout(path) for path in declared_outputs
            ),
            "output_directory": str(self.output_directory),
            "port_free": True,
        }

    def _stage_config(self, stage: str) -> Mapping[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        return dict(self.config.get("stages", {}).get(stage, {}))

    def _stage_manifest_path(self, stage: str) -> Path:
        return self.manifest_directory / f"{stage}.json"

    def _read_stage_manifest(self, stage: str) -> dict[str, Any] | None:
        path = self._stage_manifest_path(stage)
        return (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else None
        )

    def _write_stage_manifest(
        self, stage: str, value: Mapping[str, Any]
    ) -> Path:
        self.manifest_directory.mkdir(parents=True, exist_ok=True)
        path = self._stage_manifest_path(stage)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def _inputs(self, stage: str) -> dict[str, Any]:
        stage_config = self._stage_config(stage)
        explicit = [Path(value) for value in stage_config.get("inputs", [])]
        upstream = {
            dependency: self._read_stage_manifest(dependency)
            for dependency in STAGE_DEPENDENCIES.get(stage, ())
        }
        return {
            "workflow_config": _path_identity(self.config_path),
            "stage_config": stage_config,
            "explicit_paths": [_path_identity(path) for path in explicit],
            "upstream_manifest_sha256": {
                name: _canonical_hash(value) if value is not None else None
                for name, value in upstream.items()
            },
        }

    def _outputs(self, stage: str) -> list[Path]:
        return [
            Path(value).resolve()
            for value in self._stage_config(stage).get("outputs", [])
        ]

    def inspect(self) -> dict[str, Any]:
        rows = {}
        for stage in STAGES:
            manifest = self._read_stage_manifest(stage)
            rows[stage] = (
                {"status": StageStatus.NOT_RUN.value}
                if manifest is None
                else {
                    "status": manifest["status"],
                    "manifest": str(self._stage_manifest_path(stage)),
                    "input_contract_sha256": manifest.get(
                        "input_contract_sha256"
                    ),
                }
            )
        return {"schema": WORKFLOW_SCHEMA, "stages": rows}

    def run_stage(
        self,
        stage: str,
        action: Callable[[Mapping[str, Any], Path], Any],
        *,
        force: bool = False,
    ) -> StageResult:
        """Run one callable stage or reuse an exactly matching prior result."""
        inputs = self._inputs(stage)
        input_hash = _canonical_hash(inputs)
        prior = self._read_stage_manifest(stage)
        outputs = self._outputs(stage)
        if (
            not force
            and prior is not None
            and prior.get("status") == StageStatus.PASS.value
        ):
            recorded_outputs = prior.get("outputs", [])
            current_outputs = [_path_identity(path) for path in outputs]
            if (
                prior.get("input_contract_sha256") == input_hash
                and recorded_outputs == current_outputs
            ):
                return StageResult(
                    stage,
                    StageStatus.PASS,
                    self._stage_manifest_path(stage),
                    True,
                    prior,
                )
            stale = {
                **prior,
                "status": StageStatus.STALE_INPUT.value,
                "stale_detected_utc": _utc_now(),
                "current_input_contract_sha256": input_hash,
                "current_outputs": current_outputs,
            }
            path = self._write_stage_manifest(stage, stale)
            raise StaleStageError(
                f"stage {stage!r} has stale inputs or outputs; inspect {path}"
            )
        blocked = [
            dependency
            for dependency in STAGE_DEPENDENCIES.get(stage, ())
            if (self._read_stage_manifest(dependency) or {}).get("status")
            != StageStatus.PASS.value
        ]
        if blocked:
            manifest = {
                "schema": SCHEMA,
                "stage": stage,
                "status": StageStatus.BLOCKED.value,
                "blocked_by": blocked,
                "created_utc": _utc_now(),
                "input_contract_sha256": input_hash,
                "inputs": inputs,
            }
            path = self._write_stage_manifest(stage, manifest)
            return StageResult(
                stage, StageStatus.BLOCKED, path, False, manifest
            )
        running = {
            "schema": SCHEMA,
            "stage": stage,
            "status": StageStatus.RUNNING.value,
            "started_utc": _utc_now(),
            "input_contract_sha256": input_hash,
            "inputs": inputs,
            "restart": {
                "config": str(self.config_path),
                "stage": stage,
            },
        }
        self._write_stage_manifest(stage, running)
        try:
            result = action(self._stage_config(stage), self.output_directory)
            missing = [str(path) for path in outputs if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"stage did not create outputs: {missing}"
                )
            manifest = {
                **running,
                "status": StageStatus.PASS.value,
                "completed_utc": _utc_now(),
                "outputs": [_path_identity(path) for path in outputs],
                "result": result,
            }
        except Exception as exc:
            manifest = {
                **running,
                "status": StageStatus.FAIL.value,
                "completed_utc": _utc_now(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "outputs": [_path_identity(path) for path in outputs],
            }
            self._write_stage_manifest(stage, manifest)
            raise
        path = self._write_stage_manifest(stage, manifest)
        self.write_workflow_manifest()
        return StageResult(stage, StageStatus.PASS, path, False, manifest)

    def write_workflow_manifest(self) -> dict[str, Any]:
        manifest = {
            "schema": WORKFLOW_SCHEMA,
            "updated_utc": _utc_now(),
            "config": _path_identity(self.config_path),
            "output_directory": str(self.output_directory),
            "stages": {
                stage: self._read_stage_manifest(stage) for stage in STAGES
            },
        }
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    # Named methods make the public stage graph discoverable without coupling
    # this layer to one scheduler or transport execution backend.
    def __getattr__(self, name: str):
        if name in STAGES:
            return lambda action, force=False: self.run_stage(
                name, action, force=force
            )
        raise AttributeError(name)
