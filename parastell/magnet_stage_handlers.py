"""Built-in callable handlers for restartable magnet-field workflow stages."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from itertools import permutations
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dict(value)


def _normalise_cad_role_signature(
    signature: Mapping[str, Any], *, magnet_id: str
) -> dict[str, dict[str, Any]]:
    normalised = {}
    for role, value in signature.items():
        if role not in {"magnet_casing", "winding_pack"}:
            raise ValueError(
                f"unsupported source CAD role {role!r} for {magnet_id!r}"
            )
        if not isinstance(value, Mapping):
            raise ValueError(
                f"source CAD signature for {magnet_id!r}/{role} is invalid"
            )
        volume = float(value.get("volume_cm3", float("nan")))
        bounding_box = np.asarray(value.get("bounding_box_cm"), dtype=float)
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError(
                f"source CAD volume for {magnet_id!r}/{role} must be positive"
            )
        if (
            bounding_box.shape != (2, 3)
            or not np.all(np.isfinite(bounding_box))
            or np.any(bounding_box[1] < bounding_box[0])
        ):
            raise ValueError(
                f"source CAD bounding box for {magnet_id!r}/{role} is invalid"
            )
        normalised[role] = {
            "volume_cm3": volume,
            "bounding_box_cm": bounding_box,
        }
    return normalised


def _association_cad_group_identity(
    pair: Mapping[str, Any], *, magnet_id: str
) -> dict[str, Any]:
    expected_roles = {"winding_pack"}
    if pair.get("casing") is not None:
        expected_roles.add("magnet_casing")
    records = [pair.get("winding_pack")]
    if pair.get("casing") is not None:
        records.append(pair.get("casing"))
    provenances = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("source_coil_provenance"), Mapping
        ):
            raise ValueError(
                f"association for {magnet_id!r} lacks source CAD provenance"
            )
        provenances.append(dict(record["source_coil_provenance"]))
    if any(value != provenances[0] for value in provenances[1:]):
        raise ValueError(
            f"source CAD provenance disagrees across roles for {magnet_id!r}"
        )
    provenance = provenances[0]
    raw_group_index = provenance.get("cad_solid_group_index")
    if isinstance(raw_group_index, bool) or not isinstance(
        raw_group_index, int
    ):
        raise ValueError(
            f"association for {magnet_id!r} lacks a valid CAD group index"
        )
    group_index = raw_group_index
    if group_index < 0:
        raise ValueError(
            f"association for {magnet_id!r} has an invalid CAD group index"
        )
    filament_index = provenance.get("ordered_filament_index")
    if (
        isinstance(filament_index, bool)
        or not isinstance(filament_index, int)
        or filament_index != group_index
    ):
        raise ValueError(
            f"CAD and filament indices disagree for {magnet_id!r}"
        )
    expected_coil_id = f"coil-{group_index:04d}"
    if str(pair.get("coil_id")) != expected_coil_id:
        raise ValueError(
            f"coil identity disagrees with the CAD group for {magnet_id!r}"
        )
    identity = provenance.get("cad_to_dagmc_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(
            f"association for {magnet_id!r} lacks CAD identity evidence"
        )
    if (
        identity.get("method")
        != "global_role_closed_boundary_volume_and_bounding_box_assignment"
    ):
        raise ValueError(
            f"association for {magnet_id!r} uses unsupported CAD identity evidence"
        )
    signature = identity.get("cad_signature")
    if not isinstance(signature, Mapping):
        raise ValueError(
            f"association for {magnet_id!r} lacks a full CAD signature"
        )
    signature = _normalise_cad_role_signature(signature, magnet_id=magnet_id)
    if set(signature) != expected_roles:
        raise ValueError(
            f"source CAD roles disagree with the association for {magnet_id!r}"
        )
    return {
        "cad_solid_group_index": group_index,
        "cad_signature": signature,
    }


def _solid_cad_signature(solid: Any) -> dict[str, Any]:
    from .magnet_radiation_field import cad_solid_boundary_signature

    signature = cad_solid_boundary_signature(solid)
    return {
        "volume_cm3": float(signature["volume_cm3"]),
        "bounding_box_cm": np.asarray(
            signature["bounding_box_cm"], dtype=float
        ),
    }


def _validate_cad_group_signature(
    solids: Any,
    expected_signature: Mapping[str, Mapping[str, Any]],
    *,
    magnet_id: str,
    volume_relative_tolerance: float,
    bounding_box_tolerance_cm: float,
) -> dict[str, int]:
    if volume_relative_tolerance <= 0.0:
        raise ValueError("magnet identity volume tolerance must be positive")
    if bounding_box_tolerance_cm <= 0.0:
        raise ValueError(
            "magnet identity bounding-box tolerance must be positive"
        )
    actual = tuple(_solid_cad_signature(solid) for solid in solids)
    roles = tuple(sorted(expected_signature))
    if len(actual) != len(roles):
        raise ValueError(
            f"reconstructed CAD solid count disagrees for {magnet_id!r}"
        )
    candidates = []
    for assignment in permutations(range(len(actual))):
        volume_error = 0.0
        box_error = 0.0
        for role, solid_index in zip(roles, assignment):
            expected = expected_signature[role]
            volume_error = max(
                volume_error,
                abs(
                    actual[solid_index]["volume_cm3"]
                    - float(expected["volume_cm3"])
                )
                / float(expected["volume_cm3"]),
            )
            box_error = max(
                box_error,
                float(
                    np.max(
                        np.abs(
                            actual[solid_index]["bounding_box_cm"]
                            - np.asarray(
                                expected["bounding_box_cm"], dtype=float
                            )
                        )
                    )
                ),
            )
        if (
            volume_error <= volume_relative_tolerance
            and box_error <= bounding_box_tolerance_cm
        ):
            score = (
                volume_error / volume_relative_tolerance
                + box_error / bounding_box_tolerance_cm
            )
            candidates.append((score, volume_error, box_error, assignment))
    if not candidates:
        raise ValueError(
            f"reconstructed CAD signature disagrees for {magnet_id!r}"
        )
    candidates.sort(key=lambda item: (item[0], item[3]))
    if len(candidates) > 1 and np.isclose(
        candidates[0][0], candidates[1][0], rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            f"reconstructed CAD role assignment is ambiguous for {magnet_id!r}"
        )
    return {role: int(index) for role, index in zip(roles, candidates[0][3])}


def _canonical_geometry_policy(
    stage: Mapping[str, Any], artifact_policy: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve one policy and reject explicit disagreement with an artifact."""
    from .dagmc_envelope import canonical_geometry_policy

    artifact = dict(artifact_policy or {})
    configured_quantum = stage.get("coordinate_quantum_cm")
    configured_tolerances = stage.get("faceting_tolerances")
    policy = canonical_geometry_policy(
        (
            configured_quantum
            if configured_quantum is not None
            else artifact.get("coordinate_quantum_cm", 1.0e-6)
        ),
        (
            configured_tolerances
            if configured_tolerances is not None
            else artifact.get("faceting_tolerances", {})
        ),
    )
    if artifact:
        expected = canonical_geometry_policy(
            artifact.get("coordinate_quantum_cm", 1.0e-6),
            artifact.get("faceting_tolerances", {}),
        )
        if policy != expected:
            raise ValueError(
                "configured canonical geometry policy disagrees with the "
                "upstream geometry artifact"
            )
    return policy


_DISABLED_WEIGHT_WINDOW_CLASSIFICATIONS = {
    "NO_MATERIAL_BENEFIT_DISABLE",
    "REJECTED_BIAS",
    "REJECTED_INSTABILITY",
    "INSUFFICIENT_PILOT_STATISTICS",
}


def _disabled_weight_window_status(path: str | Path) -> dict[str, Any] | None:
    source = Path(path).resolve()
    if not source.is_file():
        return None
    value = json.loads(source.read_text(encoding="utf-8"))
    classification = value.get("classification")
    if (
        classification in _DISABLED_WEIGHT_WINDOW_CLASSIFICATIONS
        and value.get("weight_windows_enabled") is False
    ):
        return value
    return None


def _skipped_weight_window_stage(
    *, stage_name: str, disabled: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": f"parastell.magnet_{stage_name}/v1.0.0",
        "status": "SKIPPED_UNBIASED_FALLBACK",
        "classification": disabled["classification"],
        "weight_windows_enabled": False,
        "production_transport": "UNBIASED",
        "reason": disabled.get("reason"),
        "evidence": list(disabled.get("evidence", ())),
        "execution_performed": False,
    }


def _validate_enabled_weight_window_qualification(
    qualification: Mapping[str, Any],
    campaign: Mapping[str, Any],
    *,
    expected_particle_type: str,
) -> None:
    if campaign.get("status") != "PASS":
        raise ValueError(
            "enabled production weight windows require a completed campaign"
        )
    if qualification.get("run_contract_sha256") != campaign.get(
        "run_contract_sha256"
    ):
        raise ValueError(
            "weight-window qualification does not bind the selected campaign"
        )
    if qualification.get("artifact_particle_type") != expected_particle_type:
        raise ValueError(
            "qualified particle type differs from the WW artifact contract"
        )
    if qualification.get("classification") not in {
        "QUALIFIED_AND_ENABLED",
        "QUALIFIED_FOR_NEUTRONS_ONLY",
        "QUALIFIED_FOR_SELECTED_MAGNETS_ONLY",
    }:
        raise ValueError(
            "weight_windows_enabled contradicts the qualification classification"
        )


def validate_inputs(stage: Mapping[str, Any], output_root: Path):
    from .production_handoff import load_and_validate_no_port_configuration

    configuration, audit = load_and_validate_no_port_configuration(
        stage["geometry_config"]
    )
    paths = {
        name: Path(value).resolve()
        for name, value in stage["required_files"].items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"required production inputs are absent: {missing}"
        )
    result = {
        "status": "PASS",
        "port_free": audit.to_dict(),
        "geometry_paths_supported": [
            "filament",
            "custom_step",
            "cad_to_dagmc",
            "native_pydagmc",
            "gmsh_moab",
            "optional_cubit",
        ],
        "required_files": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "configuration_keys": sorted(configuration),
    }
    return _write(stage["report_path"], result)


def _build_source_candidate(stage: Mapping[str, Any]) -> dict[str, Any]:
    """Build one source mesh and its hash-bound convergence manifest."""
    from .dt_source import source_convergence_observables
    from .parastell import Stellarator
    from .utils import read_yaml_config

    config = read_yaml_config(stage["geometry_config"])
    stellarator = Stellarator(str(Path(stage["vmec_path"]).resolve()))
    radial, poloidal, toroidal = (int(value) for value in stage["mesh_shape"])
    extent = float(config["source_mesh"].get("toroidal_extent", 90.0))
    stellarator.construct_source_mesh(
        np.linspace(0.0, 1.0, radial),
        np.linspace(0.0, 360.0, poloidal),
        np.linspace(0.0, extent, toroidal),
    )
    source_path = Path(stage["source_mesh_path"]).resolve()
    stellarator.export_source_mesh(
        source_path.stem, export_dir=source_path.parent
    )
    manifest = {
        "schema": "parastell.magnet_source_stage/v1.0.0",
        "source_mesh": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "size_bytes": source_path.stat().st_size,
            "shape": [radial, poloidal, toroidal],
            "tetrahedra": len(stellarator.source_mesh.strengths),
            "physical_source_rate_per_s": float(
                sum(stellarator.source_mesh.strengths)
            ),
        },
        "vmec": {
            "path": str(Path(stage["vmec_path"]).resolve()),
            "sha256": _sha256(stage["vmec_path"]),
        },
        "convergence_observables": source_convergence_observables(
            stellarator.source_mesh
        ),
        "fallback_source_used": False,
        "temperature_dependent_dt_spectrum_ready": True,
    }
    return _write(stage["source_manifest_path"], manifest)


def build_source(stage: Mapping[str, Any], output_root: Path):
    return _build_source_candidate(stage)


def build_source_convergence_ladder(
    stage: Mapping[str, Any], output_root: Path
):
    """Build the three required source candidates not owned by build_source."""
    from .source_convergence import REQUIRED_CANDIDATE_SHAPES

    primary_shape = stage.get("primary_shape")
    candidates = stage.get("candidates")
    if (
        not isinstance(primary_shape, list)
        or tuple(primary_shape) not in REQUIRED_CANDIDATE_SHAPES
    ):
        raise ValueError(
            "source convergence ladder primary_shape is not in the required ladder"
        )
    expected = set(REQUIRED_CANDIDATE_SHAPES) - {tuple(primary_shape)}
    if not isinstance(candidates, list) or len(candidates) != len(expected):
        raise ValueError(
            "source convergence ladder requires exactly three non-primary candidates"
        )

    declared_shapes = []
    output_paths = []
    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError(
                "source convergence ladder candidates must be mappings"
            )
        shape = candidate.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in shape
            )
        ):
            raise TypeError(
                "source convergence ladder shapes must contain three integers"
            )
        source_path = Path(candidate["source_mesh_path"]).resolve()
        manifest_path = Path(candidate["source_manifest_path"]).resolve()
        declared_shapes.append(tuple(shape))
        output_paths.extend((source_path, manifest_path))
        normalized.append(
            {
                **stage,
                "mesh_shape": shape,
                "source_mesh_path": str(source_path),
                "source_manifest_path": str(manifest_path),
            }
        )
    if set(declared_shapes) != expected or len(set(declared_shapes)) != len(
        expected
    ):
        raise ValueError(
            "source convergence ladder candidates must be the exact non-primary shapes"
        )
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("source convergence ladder outputs must be distinct")
    declared_outputs = [
        Path(value).resolve() for value in stage.get("outputs", [])
    ]
    if set(declared_outputs) != set(output_paths) or len(
        declared_outputs
    ) != len(output_paths):
        raise ValueError(
            "source convergence ladder must declare exactly its six candidate outputs"
        )

    primary_paths = {
        Path(value).resolve()
        for name in (
            "primary_source_mesh_path",
            "primary_source_manifest_path",
        )
        if (value := stage.get(name)) is not None
    }
    if primary_paths.intersection(output_paths):
        raise ValueError(
            "source convergence ladder cannot overwrite primary source outputs"
        )

    built = []
    for candidate in normalized:
        manifest = _build_source_candidate(candidate)
        manifest_path = Path(candidate["source_manifest_path"]).resolve()
        source_path = Path(candidate["source_mesh_path"]).resolve()
        built.append(
            {
                "shape": list(manifest["source_mesh"]["shape"]),
                "source_mesh": {
                    "path": str(source_path),
                    "sha256": _sha256(source_path),
                    "size_bytes": source_path.stat().st_size,
                },
                "source_manifest": {
                    "path": str(manifest_path),
                    "sha256": _sha256(manifest_path),
                    "size_bytes": manifest_path.stat().st_size,
                },
                "convergence_observables": manifest["convergence_observables"],
            }
        )
    return {
        "schema": "parastell.magnet_source_convergence_ladder_stage/v1.0.0",
        "primary_shape": list(primary_shape),
        "built_candidate_count": len(built),
        "candidates": built,
    }


def qualify_source_convergence(stage: Mapping[str, Any], output_root: Path):
    """Evaluate the four-shape, hash-bound source convergence campaign."""
    from .source_convergence import MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS
    from .source_convergence import REQUIRED_CANDIDATE_SHAPES
    from .source_convergence import validate_source_mesh_convergence
    from .source_convergence import write_source_mesh_convergence

    configured = stage.get("candidates")
    required_response_ids = stage.get("required_response_metric_ids")
    if required_response_ids != list(MANDATORY_WHOLE_MAGNET_RESPONSE_METRICS):
        raise ValueError(
            "source convergence must explicitly require whole-magnet current, "
            "flux, heating, hotspot, and computational-cost metrics"
        )
    if not isinstance(configured, list) or len(configured) != len(
        REQUIRED_CANDIDATE_SHAPES
    ):
        raise ValueError(
            "source convergence requires four configured candidate manifests"
        )
    declared_shapes = []
    available = []
    missing_files = []
    for candidate in configured:
        if not isinstance(candidate, Mapping):
            raise TypeError("source convergence candidates must be mappings")
        shape = candidate.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in shape
            )
        ):
            raise TypeError(
                "configured source convergence shapes must contain three integers"
            )
        declared_shapes.append(tuple(shape))
        manifest_path = Path(candidate["source_manifest_path"]).resolve()
        response_path_value = candidate.get("response_report_path")
        response_path = (
            Path(response_path_value).resolve()
            if response_path_value is not None
            else None
        )
        if response_path is not None and not response_path.is_file():
            missing_files.append(str(response_path))
        if not manifest_path.is_file():
            missing_files.append(str(manifest_path))
            continue
        normalized = {
            "source_manifest_path": str(manifest_path),
            "expected_shape": shape,
        }
        if candidate.get("source_manifest_sha256") is not None:
            normalized["source_manifest_sha256"] = candidate[
                "source_manifest_sha256"
            ]
        if response_path is not None and response_path.is_file():
            normalized["response_report_path"] = str(response_path)
            if candidate.get("response_report_sha256") is not None:
                normalized["response_report_sha256"] = candidate[
                    "response_report_sha256"
                ]
        available.append(normalized)
    if set(declared_shapes) != set(REQUIRED_CANDIDATE_SHAPES) or len(
        set(declared_shapes)
    ) != len(REQUIRED_CANDIDATE_SHAPES):
        raise ValueError(
            "configured source convergence shapes must exactly match the required ladder"
        )

    output = Path(stage["qualification_path"]).resolve()
    result = write_source_mesh_convergence(
        output,
        available,
        source_metric_tolerances=stage.get("source_metric_tolerances"),
        response_metric_tolerances=stage.get("response_metric_tolerances"),
        required_response_metric_ids=required_response_ids,
    )
    validation = validate_source_mesh_convergence(
        output, verify_bound_files=True
    )
    return {
        **validation,
        "schema": "parastell.magnet_source_convergence_stage/v1.0.0",
        "artifact_schema": validation["schema"],
        "gate": "SOURCE_MESH_CONVERGENCE",
        "qualification_path": str(output),
        "configured_candidate_count": len(configured),
        "available_candidate_count": len(available),
        "missing_configured_files": sorted(set(missing_files)),
        "evidence_sha256": result["evidence_sha256"],
    }


def build_geometry(stage: Mapping[str, Any], output_root: Path):
    from .combined_openmc16_model import _magnet_material_tag
    from .dagmc_envelope import discover_magnet_volumes
    from .dagmc_envelope import require_watertight_dagmc
    from .dagmc_graveyard import close_dagmc_file
    from .geometry_overlap import audit_native_cad_overlaps
    from .magnet_radiation_field import filament_associations
    from .parastell import Stellarator
    from .production_handoff import load_and_validate_no_port_configuration

    config, _ = load_and_validate_no_port_configuration(
        stage["geometry_config"]
    )
    output = Path(stage["dagmc_path"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_stage = dict(stage)
    actual_faceting = {
        "minimum_mesh_size_cm": float(stage.get("minimum_mesh_size_cm", 5.0)),
        "maximum_mesh_size_cm": float(stage.get("maximum_mesh_size_cm", 20.0)),
    }
    declared_faceting = dict(stage.get("faceting_tolerances", {}))
    for name, value in actual_faceting.items():
        if (
            name in declared_faceting
            and float(declared_faceting[name]) != value
        ):
            raise ValueError(
                f"canonical {name} disagrees with the geometry build setting"
            )
    policy_stage["faceting_tolerances"] = {
        **declared_faceting,
        **actual_faceting,
    }
    geometry_policy = _canonical_geometry_policy(policy_stage)
    backend = (
        str(stage.get("geometry_backend", "cad-to-dagmc")).strip().lower()
    )
    backend_options = {
        "cad-to-dagmc": False,
        "pydagmc-invessel-cad-to-dagmc-magnets": True,
    }
    if backend not in backend_options:
        raise ValueError(
            f"unsupported geometry_backend {backend!r}; expected one of "
            f"{sorted(backend_options)}"
        )
    invessel = dict(config["invessel_build"])
    legacy_export_flag = invessel.pop("export_cad_to_dagmc", None)
    if "use_pydagmc" in invessel:
        configured_pydagmc = bool(invessel["use_pydagmc"])
        if configured_pydagmc != backend_options[backend]:
            raise ValueError(
                "geometry_backend conflicts with invessel_build.use_pydagmc"
            )
    invessel["use_pydagmc"] = backend_options[backend]
    stellarator = Stellarator(str(Path(stage["vmec_path"]).resolve()))
    stellarator.construct_invessel_build(**invessel)
    magnet = dict(config["magnet_coils"])
    casing = float(
        stage.get("casing_thickness_cm", magnet.get("case_thickness", 0.0))
    )
    stellarator.construct_magnets_from_filaments(
        str(Path(stage["coils_path"]).resolve()),
        float(magnet["width"]),
        float(magnet["thickness"]),
        float(magnet["toroidal_extent"]),
        case_thickness=casing,
        sample_mod=int(magnet.get("sample_mod", 6)),
        mat_tag=_magnet_material_tag(casing),
    )
    invessel_solids, invessel_materials = (
        stellarator.invessel_build.extract_solids_and_mat_tags()
    )
    overlap_solids = list(invessel_solids)
    overlap_labels = [
        f"invessel:{material}:{index}"
        for index, material in enumerate(invessel_materials)
    ]
    for coil_index, solids in enumerate(stellarator.magnet_set.coil_solids):
        roles = (
            ("magnet_casing", "winding_pack")
            if len(solids) == 2
            else ("winding_pack",)
        )
        for role, solid in zip(roles, solids):
            overlap_solids.append(solid)
            overlap_labels.append(f"magnet:{coil_index:04d}:{role}")
    overlap_audit = audit_native_cad_overlaps(
        overlap_solids,
        overlap_labels,
        absolute_volume_tolerance_cm3=float(
            stage.get("overlap_absolute_volume_tolerance_cm3", 1.0e-7)
        ),
        relative_volume_tolerance=float(
            stage.get("overlap_relative_volume_tolerance", 1.0e-10)
        ),
    )
    if stellarator.use_pydagmc:
        stellarator.build_pydagmc_model(
            magnet_exporter="cad_to_dagmc",
            filename=output.stem + "_magnet_components",
            export_dir=output.parent,
            min_mesh_size=float(stage.get("minimum_mesh_size_cm", 5.0)),
            max_mesh_size=float(stage.get("maximum_mesh_size_cm", 20.0)),
        )
        stellarator.export_pydagmc_model(
            filename=output.stem, export_dir=output.parent
        )
    else:
        stellarator.build_cad_to_dagmc_model()
        stellarator.export_cad_to_dagmc(
            filename=output.stem,
            export_dir=output.parent,
            min_mesh_size=float(stage.get("minimum_mesh_size_cm", 5.0)),
            max_mesh_size=float(stage.get("maximum_mesh_size_cm", 20.0)),
        )
    closed = output.with_name(output.stem + "_closed.h5m")
    graveyard = close_dagmc_file(
        output, closed, margin_cm=float(stage.get("graveyard_margin_cm", 50.0))
    )
    closed.replace(output)
    watertight = require_watertight_dagmc(output)
    initial = discover_magnet_volumes(output, **geometry_policy)
    associations = filament_associations(
        initial,
        stellarator.magnet_set,
        coils_path=stage["coils_path"],
        machine_id=str(stage.get("machine_id", "machine")),
        sector_id=str(stage.get("sector_id", "sector")),
    )
    inventory = discover_magnet_volumes(
        output, associations=associations, **geometry_policy
    )
    association_manifest = {
        "schema": "parastell.magnet_associations/v1.0.0",
        "dagmc_sha256": _sha256(output),
        "canonical_geometry_fingerprint": inventory.canonical_geometry_fingerprint,
        "canonical_geometry_policy": geometry_policy,
        "associations": {
            str(key): value for key, value in associations.items()
        },
        "centreline_points_by_coil": {
            f"coil-{index:04d}": np.asarray(coil.coords, dtype=float).tolist()
            for index, coil in enumerate(stellarator.magnet_set.magnet_coils)
        },
        "inventory": inventory.to_dict(),
    }
    _write(stage["associations_path"], association_manifest)
    result = {
        "schema": "parastell.magnet_geometry_stage/v1.0.0",
        "dagmc": {
            "path": str(output),
            "raw_h5m_sha256": _sha256(output),
            "canonical_geometry_fingerprint": inventory.canonical_geometry_fingerprint,
            "canonical_geometry_policy": geometry_policy,
            "size_bytes": output.stat().st_size,
        },
        "watertightness": watertight.to_dict(),
        "native_cad_overlap_audit": overlap_audit,
        "graveyard": graveyard,
        "magnet_count": len(inventory.pairs),
        "winding_pack_count": len(inventory.winding_packs),
        "casing_count": len(inventory.casings),
        "associations_path": str(Path(stage["associations_path"]).resolve()),
        "ports": False,
        "geometry_backend": backend,
        "legacy_export_cad_to_dagmc_config": legacy_export_flag,
    }
    return _write(stage["geometry_manifest_path"], result)


def validate_geometry(stage: Mapping[str, Any], output_root: Path):
    from .dagmc_envelope import (
        canonical_dagmc_fingerprint,
        require_watertight_dagmc,
    )

    path = Path(stage["dagmc_path"]).resolve()
    association = None
    associations_path = stage.get("associations_path")
    if associations_path is not None:
        association = json.loads(
            Path(associations_path).read_text(encoding="utf-8")
        )
    geometry_policy = _canonical_geometry_policy(
        stage,
        (
            association.get("canonical_geometry_policy")
            if association is not None
            else None
        ),
    )
    fingerprint = canonical_dagmc_fingerprint(path, **geometry_policy)
    expected_fingerprint = (
        association.get("canonical_geometry_fingerprint")
        if association is not None
        else None
    )
    if (
        expected_fingerprint
        and fingerprint["canonical_fingerprint"] != expected_fingerprint
    ):
        raise ValueError(
            "geometry validation fingerprint disagrees with the geometry "
            "association artifact"
        )
    fingerprint_summary = {
        "algorithm": fingerprint["algorithm"],
        "canonical_fingerprint": fingerprint["canonical_fingerprint"],
        "raw_h5m_sha256": fingerprint["raw_h5m_sha256"],
        "coordinate_quantum_cm": geometry_policy["coordinate_quantum_cm"],
        "surface_count": fingerprint["surface_count"],
        "volume_count": fingerprint["volume_count"],
        "faceting_tolerances": geometry_policy["faceting_tolerances"],
    }
    watertight = require_watertight_dagmc(path)
    return _write(
        stage["validation_report_path"],
        {
            "status": "PASS",
            "fingerprint": fingerprint_summary,
            "watertightness": watertight.to_dict(),
        },
    )


def inventory_magnets(stage: Mapping[str, Any], output_root: Path):
    from .dagmc_envelope import discover_magnet_volumes, select_magnet_pairs

    association = json.loads(
        Path(stage["associations_path"]).read_text(encoding="utf-8")
    )
    geometry_policy = _canonical_geometry_policy(
        stage, association.get("canonical_geometry_policy")
    )
    inventory = discover_magnet_volumes(
        stage["dagmc_path"],
        associations={
            int(key): value
            for key, value in association["associations"].items()
        },
        **geometry_policy,
    )
    expected_fingerprint = association.get("canonical_geometry_fingerprint")
    if (
        expected_fingerprint
        and inventory.canonical_geometry_fingerprint != expected_fingerprint
    ):
        raise ValueError(
            "magnet inventory canonical geometry fingerprint disagrees with "
            "the association artifact"
        )
    selected = select_magnet_pairs(inventory, stage.get("selection", "all"))
    result = {
        "schema": "parastell.magnet_inventory/v1.0.0",
        **inventory.to_dict(),
        "selected_magnet_ids": [item.magnet_id for item in selected],
    }
    return _write(stage["inventory_path"], result)


def build_tally_meshes(stage: Mapping[str, Any], output_root: Path):
    from .coil_frame import parallel_transport_frame
    from .dagmc_envelope import discover_magnet_volumes, select_magnet_pairs
    from .magnet_local_mesh import build_local_mesh_definition

    association = json.loads(
        Path(stage["associations_path"]).read_text(encoding="utf-8")
    )
    geometry_policy = _canonical_geometry_policy(
        stage, association.get("canonical_geometry_policy")
    )
    inventory = discover_magnet_volumes(
        stage["dagmc_path"],
        associations={
            int(key): value
            for key, value in association["associations"].items()
        },
        **geometry_policy,
    )
    expected_fingerprint = association.get("canonical_geometry_fingerprint")
    if (
        expected_fingerprint
        and inventory.canonical_geometry_fingerprint != expected_fingerprint
    ):
        raise ValueError(
            "tally-mesh canonical geometry fingerprint disagrees with the "
            "association artifact"
        )
    selected = select_magnet_pairs(inventory, stage.get("selection", "all"))
    meshes = {}
    total_bins = 0
    maximum_bins_per_magnet = int(
        stage.get("maximum_bins_per_magnet", 1_000_000)
    )
    maximum_total_bins = int(stage.get("maximum_total_bins", 10_000_000))
    for pair in selected:
        points = association["centreline_points_by_coil"][pair.coil_id]
        frame = parallel_transport_frame(points)
        sample = frame.sample(pair.winding_pack.centroid_global_cm)
        mesh = build_local_mesh_definition(
            pair.magnet_id,
            bounding_box_global_cm=pair.winding_pack.bounding_box_cm,
            centreline_sample=sample,
            resolution_cm=float(stage.get("resolution_cm", 5.0)),
            padding_cm=float(stage.get("padding_cm", 0.0)),
        )
        if mesh.bin_count > maximum_bins_per_magnet:
            raise ValueError(
                f"local mesh for {pair.magnet_id} has {mesh.bin_count} bins, "
                f"exceeding maximum_bins_per_magnet={maximum_bins_per_magnet}; "
                "increase the resolution or explicitly raise the budget"
            )
        total_bins += mesh.bin_count
        meshes[pair.magnet_id] = mesh.to_dict()
    if total_bins > maximum_total_bins:
        raise ValueError(
            f"local mesh collection has {total_bins} bins, exceeding "
            f"maximum_total_bins={maximum_total_bins}; increase the resolution, "
            "select fewer magnets, or explicitly raise the budget"
        )
    return _write(
        stage["local_mesh_manifest_path"],
        {
            "schema": "parastell.magnet_local_mesh_collection/v1.0.0",
            "canonical_geometry_fingerprint": inventory.canonical_geometry_fingerprint,
            "resolution_cm": float(stage.get("resolution_cm", 5.0)),
            "candidate_resolutions_cm": [5.0, 2.0, 1.0, 0.5],
            "total_bin_count": total_bins,
            "maximum_bins_per_magnet": maximum_bins_per_magnet,
            "maximum_total_bins": maximum_total_bins,
            "spatial_estimator": "track_length_flux_on_full_rotated_mesh_voxels",
            "mesh_volume_basis": "full_geometric_voxel_volume",
            "cell_filter_applied": False,
            "meshes": meshes,
        },
    )


def build_weight_window_mesh(stage: Mapping[str, Any], output_root: Path):
    from .weight_windows import build_weight_window_mesh_from_step
    from .weight_windows import weight_window_disabled_fallback

    if not bool(stage.get("enabled", False)):
        result = weight_window_disabled_fallback(
            str(
                stage.get(
                    "disabled_reason",
                    "weight-window study disabled by configuration",
                )
            )
        )
        return _write(stage["weight_window_mesh_status_path"], result)
    component_step_paths = dict(stage.get("component_step_paths", {}))
    characteristic_length_cm = dict(stage.get("characteristic_length_cm", {}))
    association = None
    if "associations_path" in stage:
        association = json.loads(
            Path(stage["associations_path"]).read_text(encoding="utf-8")
        )
    configured_targets = stage["target_magnet_ids"]
    target_magnet_ids = (
        [
            pair["magnet_id"]
            for pair in association["inventory"]["magnet_pairs"]
        ]
        if configured_targets == "all" and association is not None
        else list(configured_targets)
    )
    geometry_fingerprint = stage.get("canonical_geometry_fingerprint")
    if geometry_fingerprint in {None, "AUTO"} and association is not None:
        geometry_fingerprint = association["canonical_geometry_fingerprint"]
    if not geometry_fingerprint:
        raise ValueError("WW mesh requires a canonical geometry fingerprint")
    if association is None:
        raise ValueError("WW mesh requires the validated magnet associations")

    association_pairs = {
        str(pair["magnet_id"]): pair
        for pair in association["inventory"]["magnet_pairs"]
    }
    missing_target_associations = set(target_magnet_ids) - set(
        association_pairs
    )
    if missing_target_associations:
        raise ValueError(
            "WW target magnets are absent from the association inventory: "
            f"{sorted(missing_target_associations)}"
        )

    reconstruct_component_steps = bool(
        stage.get("reconstruct_component_steps", False)
    )
    identity_volume_tolerance = float(
        stage.get("magnet_identity_volume_tolerance", 0.02)
    )
    identity_box_tolerance_cm = float(
        stage.get("magnet_identity_bounding_box_tolerance_cm", 1.0e-4)
    )
    target_cad_identities = (
        {
            magnet_id: _association_cad_group_identity(
                association_pairs[magnet_id], magnet_id=magnet_id
            )
            for magnet_id in target_magnet_ids
        }
        if reconstruct_component_steps
        else {}
    )
    reused_component_steps = False
    if reconstruct_component_steps and bool(
        stage.get("reuse_existing_component_steps", False)
    ):
        step_directory = Path(stage["component_step_directory"]).resolve()
        invessel_components = tuple(stage.get("invessel_components", ()))
        if not invessel_components:
            raise ValueError(
                "STEP reuse requires an explicit invessel_components list"
            )
        expected_components = (*invessel_components, *target_magnet_ids)
        existing_paths = {
            component: step_directory / f"{component}.step"
            for component in expected_components
        }
        target_steps_match = False
        if all(path.is_file() for path in existing_paths.values()):
            import cadquery as cq

            target_steps_match = True
            for magnet_id in target_magnet_ids:
                imported = cq.importers.importStep(
                    str(existing_paths[magnet_id])
                )
                try:
                    _validate_cad_group_signature(
                        imported.solids().vals(),
                        target_cad_identities[magnet_id]["cad_signature"],
                        magnet_id=magnet_id,
                        volume_relative_tolerance=identity_volume_tolerance,
                        bounding_box_tolerance_cm=identity_box_tolerance_cm,
                    )
                except ValueError:
                    target_steps_match = False
                    break
        if target_steps_match:
            for component in invessel_components:
                component_step_paths[component] = existing_paths[component]
                characteristic_length_cm.setdefault(
                    component,
                    float(
                        stage.get("invessel_characteristic_length_cm", 100.0)
                    ),
                )
            for magnet_id in target_magnet_ids:
                component_step_paths[magnet_id] = existing_paths[magnet_id]
                characteristic_length_cm.setdefault(
                    magnet_id,
                    float(stage.get("magnet_characteristic_length_cm", 100.0)),
                )
            reused_component_steps = True
    if reconstruct_component_steps and not reused_component_steps:
        import cadquery as cq

        from .combined_openmc16_model import _magnet_material_tag
        from .parastell import Stellarator
        from .production_handoff import load_and_validate_no_port_configuration

        config, _ = load_and_validate_no_port_configuration(
            stage["geometry_config"]
        )
        step_directory = Path(stage["component_step_directory"]).resolve()
        step_directory.mkdir(parents=True, exist_ok=True)
        stellarator = Stellarator(str(Path(stage["vmec_path"]).resolve()))
        invessel = dict(config["invessel_build"])
        invessel.pop("export_cad_to_dagmc", None)
        invessel["use_pydagmc"] = False
        stellarator.construct_invessel_build(**invessel)
        ivb_components = tuple(
            stage.get(
                "invessel_components",
                stellarator.invessel_build.Components.keys(),
            )
        )
        unknown = set(ivb_components) - set(
            stellarator.invessel_build.Components
        )
        if unknown:
            raise ValueError(
                f"unknown in-vessel WW components: {sorted(unknown)}"
            )
        for component in ivb_components:
            path = step_directory / f"{component}.step"
            cq.exporters.export(
                stellarator.invessel_build.Components[component], str(path)
            )
            component_step_paths[component] = path
            characteristic_length_cm.setdefault(
                component,
                float(stage.get("invessel_characteristic_length_cm", 100.0)),
            )
        magnet = dict(config["magnet_coils"])
        casing = float(
            stage.get("casing_thickness_cm", magnet.get("case_thickness", 0.0))
        )
        stellarator.construct_magnets_from_filaments(
            str(Path(stage["coils_path"]).resolve()),
            float(magnet["width"]),
            float(magnet["thickness"]),
            float(magnet["toroidal_extent"]),
            case_thickness=casing,
            sample_mod=int(magnet.get("sample_mod", 6)),
            mat_tag=_magnet_material_tag(casing),
        )
        reconstructed_targets = []
        used_group_indices = set()
        for magnet_id in target_magnet_ids:
            identity = target_cad_identities[magnet_id]
            index = identity["cad_solid_group_index"]
            if index >= len(stellarator.magnet_set.coil_solids):
                raise ValueError(
                    f"source CAD group index is out of range for {magnet_id!r}"
                )
            if index in used_group_indices:
                raise ValueError(
                    "multiple WW target magnets claim reconstructed CAD group "
                    f"{index}"
                )
            solids = stellarator.magnet_set.coil_solids[index]
            _validate_cad_group_signature(
                solids,
                identity["cad_signature"],
                magnet_id=magnet_id,
                volume_relative_tolerance=identity_volume_tolerance,
                bounding_box_tolerance_cm=identity_box_tolerance_cm,
            )
            used_group_indices.add(index)
            path = step_directory / f"{magnet_id}.step"
            cq.exporters.export(cq.Compound.makeCompound(solids), str(path))
            component_step_paths[magnet_id] = path
            characteristic_length_cm.setdefault(
                magnet_id,
                float(stage.get("magnet_characteristic_length_cm", 100.0)),
            )
            reconstructed_targets.append(magnet_id)
        if set(reconstructed_targets) != set(target_magnet_ids):
            missing = set(target_magnet_ids) - set(reconstructed_targets)
            raise ValueError(
                f"target magnet STEP reconstruction is incomplete: {sorted(missing)}"
            )
    try:
        result = build_weight_window_mesh_from_step(
            component_step_paths,
            stage["weight_window_mesh_path"],
            target_magnet_ids=target_magnet_ids,
            geometry_fingerprint=geometry_fingerprint,
            characteristic_length_cm=characteristic_length_cm,
            mode=stage.get("mode", "all_magnet_coarse"),
            volume_meshing_algorithm=int(
                stage.get("volume_meshing_algorithm", 1)
            ),
            fuse_multi_volume_components=bool(
                stage.get("fuse_multi_volume_components", True)
            ),
        )
    except (RuntimeError, ValueError) as exc:
        if stage.get("failure_policy") != "document_and_disable":
            raise
        from . import weight_windows as weight_window_module

        evidence = {
            "candidate": {
                "mode": stage.get("mode", "all_magnet_coarse"),
                "target_magnet_ids": target_magnet_ids,
                "characteristic_length_cm": {
                    name: float(value)
                    for name, value in sorted(characteristic_length_cm.items())
                },
                "volume_meshing_algorithm": int(
                    stage.get("volume_meshing_algorithm", 1)
                ),
                "fuse_multi_volume_components": bool(
                    stage.get("fuse_multi_volume_components", True)
                ),
            },
            "geometry_fingerprint": geometry_fingerprint,
            "component_inputs": {
                name: {
                    "path": str(Path(path).resolve()),
                    "sha256": _sha256(path),
                }
                for name, path in sorted(component_step_paths.items())
            },
            "implementation": {
                "weight_windows_py_sha256": _sha256(
                    Path(weight_window_module.__file__).resolve()
                ),
                "stage_handlers_py_sha256": _sha256(Path(__file__).resolve()),
            },
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        result = weight_window_disabled_fallback(
            "The declared combined conformal WW-mesh candidate was rejected "
            "by fail-closed mesh validation.",
            classification="REJECTED_INSTABILITY",
            evidence=(evidence,),
        )
    return _write(stage["weight_window_mesh_status_path"], result)


def prepare_unbiased_model(stage: Mapping[str, Any], output_root: Path):
    from .energy_groups import get_structure
    from .magnet_openmc_model import prepare_magnet_openmc_model
    from .material_manifest import (
        audit_nuclear_data,
        resolve_material_manifest,
    )

    material = resolve_material_manifest(
        stage["material_config"],
        output_path=stage["resolved_material_manifest_path"],
    )
    nuclear = audit_nuclear_data(
        material,
        cross_sections_path=stage["cross_sections_path"],
        approved_library=stage["approved_library"],
        evaluation_release=stage["evaluation_release"],
        photon_evaluation_release=stage.get("photon_evaluation_release"),
        approved_mixed_case=bool(stage.get("approved_mixed_case", False)),
        temperature_method=stage.get("temperature_method", "nearest"),
        temperature_tolerance_K=float(
            stage.get("temperature_tolerance_K", 1000.0)
        ),
    )
    _write(stage["nuclear_data_audit_path"], nuclear)
    if nuclear["status"] != "PASS":
        raise ValueError(f"nuclear-data audit failed: {nuclear['status']}")
    neutron_edges = stage.get("neutron_edges_eV")
    if neutron_edges is None:
        neutron_edges = get_structure(
            stage.get("neutron_structure", "smoke-7"), particle="neutron"
        ).edges_eV
    photon_edges = stage.get("photon_edges_eV")
    if photon_edges is None:
        photon_edges = get_structure(
            stage.get("photon_structure", "photon-master-v1"),
            particle="photon",
        ).edges_eV
    result = prepare_magnet_openmc_model(
        stage["model_directory"],
        dagmc_path=stage["dagmc_path"],
        source_mesh_path=stage["source_mesh_path"],
        material_manifest_path=stage["resolved_material_manifest_path"],
        cross_sections_path=stage["cross_sections_path"],
        associations_path=stage["associations_path"],
        magnet_selection=stage.get("selection", "all"),
        tally_profile=stage.get("tally_profile", "magnet_damage_and_handoff"),
        neutron_edges_eV=neutron_edges,
        photon_edges_eV=photon_edges,
        particles_per_batch=int(stage["particles_per_batch"]),
        batches=int(stage["batches"]),
        seed=int(stage["seed"]),
        max_surface_particles=int(stage["max_surface_particles"]),
        max_surface_files=int(stage.get("max_surface_files", 1)),
        local_mesh_manifest_path=stage.get("local_mesh_manifest_path"),
        supported_responses=stage.get("supported_responses"),
        temperature_method=stage.get("temperature_method", "nearest"),
        temperature_tolerance_K=float(
            stage.get("temperature_tolerance_K", 1000.0)
        ),
        temperature_policy_justification=str(
            stage.get("temperature_policy_justification", "")
        ),
        coordinate_quantum_cm=stage.get("coordinate_quantum_cm"),
        faceting_tolerances=stage.get("faceting_tolerances"),
    )
    return result


def run_unbiased_model(stage: Mapping[str, Any], output_root: Path):
    """Execute and fail-closed validate one prepared OpenMC model."""
    import subprocess
    import time

    import h5py

    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required to execute unbiased transport"
        ) from exc

    model_directory = Path(stage["model_directory"]).resolve()
    required_xml = tuple(
        model_directory / name
        for name in (
            "geometry.xml",
            "materials.xml",
            "settings.xml",
            "tallies.xml",
        )
    )
    missing_xml = [str(path) for path in required_xml if not path.is_file()]
    if missing_xml:
        raise FileNotFoundError(
            f"prepared OpenMC XML is absent: {missing_xml}"
        )
    executable = str(stage.get("openmc_executable", "openmc"))
    version = subprocess.run(
        [executable, "--version"],
        cwd=model_directory,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_version = str(stage.get("expected_openmc_version", "0.16.0"))
    expected_commit = stage.get("expected_openmc_commit")
    if f"OpenMC version {expected_version}" not in version:
        raise ValueError(
            f"OpenMC executable is not expected version {expected_version}"
        )
    if (
        expected_commit is not None
        and f"Commit hash: {expected_commit}" not in version
    ):
        raise ValueError(
            "OpenMC executable commit does not match configuration"
        )
    commit_lines = [
        line.partition(":")[2].strip()
        for line in version.splitlines()
        if line.strip().lower().startswith("commit hash:")
    ]
    if (
        len(commit_lines) != 1
        or len(commit_lines[0]) not in {40, 64}
        or any(
            character not in "0123456789abcdef"
            for character in commit_lines[0].lower()
        )
    ):
        raise ValueError(
            "OpenMC executable did not report one full commit SHA"
        )
    actual_openmc_commit = commit_lines[0].lower()

    started = time.perf_counter()
    openmc.run(
        cwd=model_directory,
        threads=int(stage.get("threads", 1)),
        openmc_exec=executable,
        output=bool(stage.get("stream_output", True)),
    )
    elapsed = time.perf_counter() - started

    statepoint_path = Path(stage["statepoint_path"]).resolve()
    surface_source_paths = [
        Path(value).resolve() for value in stage["surface_source_paths"]
    ]
    if not statepoint_path.is_file():
        raise FileNotFoundError(statepoint_path)
    missing_sources = [
        str(path) for path in surface_source_paths if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"OpenMC surface-source files are absent: {missing_sources}"
        )
    lost_particle_files = sorted(model_directory.glob("particle_*.h5"))
    if lost_particle_files:
        raise RuntimeError(
            "OpenMC wrote lost-particle files: "
            + ", ".join(path.name for path in lost_particle_files)
        )

    def decode(value):
        return (
            value.decode()
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
        )

    tally_summary = {}
    with h5py.File(statepoint_path) as statepoint:
        if not bool(statepoint.attrs.get("photon_transport", False)):
            raise ValueError("statepoint did not use coupled photon transport")
        if not bool(statepoint.attrs.get("tallies_present", False)):
            raise ValueError("statepoint contains no tallies")
        tallies = statepoint["tallies"]
        for key, tally in tallies.items():
            if not key.startswith("tally "):
                continue
            name = decode(tally["name"][()])
            realizations = int(tally["n_realizations"][()])
            if realizations <= 0:
                raise ValueError(f"tally {name!r} has no realizations")
            mean = tally["results"][:][..., 0] / realizations
            tally_summary[name] = {
                "realizations": realizations,
                "nonzero_bins": int(np.count_nonzero(mean)),
                "sum_mean_per_source": float(mean.sum()),
            }
        statepoint_summary = {
            "current_batch": int(statepoint["current_batch"][()]),
            "n_batches": int(statepoint["n_batches"][()]),
            "n_particles_per_batch": int(statepoint["n_particles"][()]),
            "n_realizations": int(statepoint["n_realizations"][()]),
            "photon_transport": True,
            "tally_count": len(tally_summary),
        }
    required_tallies = tuple(
        stage.get(
            "required_tally_names",
            (
                "pstl_magnet_neutron_ccfe_709_volume_flux",
                "pstl_magnet_photon_configured_volume_flux",
                "pstl_magnet_production_photon",
            ),
        )
    )
    missing_tallies = sorted(set(required_tallies) - set(tally_summary))
    if missing_tallies:
        raise ValueError(
            f"statepoint omits required tallies {missing_tallies}"
        )
    if bool(stage.get("require_secondary_photons", True)):
        photon_tallies = (
            "pstl_magnet_photon_configured_volume_flux",
            "pstl_magnet_production_photon",
        )
        zero = [
            name
            for name in photon_tallies
            if tally_summary.get(name, {}).get("sum_mean_per_source", 0.0)
            <= 0.0
        ]
        if zero:
            raise ValueError(
                f"secondary-photon evidence is absent from {zero}"
            )

    particle_counts = {"neutron": 0, "photon": 0}
    surface_ids = set()
    maximum_direction_norm_error = 0.0
    stored_records = 0
    for path in surface_source_paths:
        with h5py.File(path) as source:
            if "source_bank" not in source:
                raise ValueError(f"{path} has no source_bank")
            bank = source["source_bank"][:]
        stored_records += len(bank)
        pdg = np.asarray(bank["particle"]).reshape(-1)
        particle_counts["neutron"] += int(np.count_nonzero(pdg == 2112))
        particle_counts["photon"] += int(np.count_nonzero(pdg == 22))
        unknown = set(np.unique(pdg)) - {22, 2112}
        if unknown:
            raise ValueError(
                f"surface source contains unsupported particles {sorted(unknown)}"
            )
        surface_ids.update(int(value) for value in bank["surf_id"])
        if len(bank):
            directions = np.column_stack([bank["u"][axis] for axis in "xyz"])
            maximum_direction_norm_error = max(
                maximum_direction_norm_error,
                float(
                    np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0))
                ),
            )
    configured_capacity = int(stage["max_surface_particles"]) * int(
        stage.get("max_surface_files", len(surface_source_paths))
    )
    if stored_records >= configured_capacity:
        raise RuntimeError(
            "surface-source capacity was reached; bank is truncated and invalid"
        )
    if maximum_direction_norm_error > float(
        stage.get("direction_norm_tolerance", 1.0e-12)
    ):
        raise ValueError("surface-source directions are not normalized")

    report = {
        "schema": "parastell.magnet_transport_execution/v1.0.0",
        "status": "PASS",
        "execution_performed": True,
        "transport": "UNBIASED",
        "openmc_executable_version": version,
        "openmc_version": expected_version,
        "openmc_commit": actual_openmc_commit,
        "elapsed_wall_seconds": elapsed,
        "threads": int(stage.get("threads", 1)),
        "statepoint": {
            "path": str(statepoint_path),
            "sha256": _sha256(statepoint_path),
            "size_bytes": statepoint_path.stat().st_size,
            **statepoint_summary,
        },
        "surface_source": {
            "paths": [str(path) for path in surface_source_paths],
            "sha256": [_sha256(path) for path in surface_source_paths],
            "stored_record_count": stored_records,
            "particle_counts": particle_counts,
            "surface_count": len(surface_ids),
            "maximum_direction_norm_error": maximum_direction_norm_error,
            "configured_capacity": configured_capacity,
            "capacity_reached": False,
        },
        "lost_particle_files": [],
        "dagmc_navigation_failures": 0,
        "required_tallies": {
            name: tally_summary[name] for name in required_tallies
        },
        "secondary_photon_evidence": {
            name: tally_summary[name]
            for name in (
                "pstl_magnet_photon_configured_volume_flux",
                "pstl_magnet_production_photon",
            )
        },
    }
    return _write(stage["transport_report_path"], report)


_UNBIASED_SEED_REPORT_SCHEMA = "parastell.unbiased_response_report/v1.0.0"
_UNBIASED_CAMPAIGN_SCHEMA = "parastell.unbiased_qualification_campaign/v1.0.0"


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_unbiased_campaign_response_specs(
    values: Any,
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise ValueError("unbiased campaign requires key response specs")
    specs = []
    response_ids = []
    for source in values:
        if not isinstance(source, Mapping):
            raise TypeError(
                "unbiased campaign response specs must be mappings"
            )
        spec = dict(source)
        response_id = str(spec.get("response_id", ""))
        if not response_id:
            raise ValueError("unbiased campaign response IDs must be nonempty")
        response_ids.append(response_id)
        kind = str(spec.get("kind", "tally"))
        if kind not in {"tally", "surface_bank_weight"}:
            raise ValueError(f"unsupported unbiased response kind {kind!r}")
        particle = str(spec.get("particle", ""))
        if particle not in {"neutron", "photon"}:
            raise ValueError(
                f"response {response_id!r} has an unsupported particle"
            )
        for name in ("quantity", "units", "normalization"):
            if not str(spec.get(name, "")):
                raise ValueError(f"response {response_id!r} requires {name}")
        if spec.get("is_key_response", True) is not True:
            raise ValueError(
                f"response {response_id!r} is not marked as a key response"
            )
        if kind == "tally":
            if not str(spec.get("tally_name", "")):
                raise ValueError(
                    f"tally response {response_id!r} requires tally_name"
                )
            reduction = str(spec.get("reduction", "sum"))
            if reduction not in {"sum", "mean"}:
                raise ValueError(
                    f"unsupported unbiased response reduction {reduction!r}"
                )
            flat_bins = spec.get("flat_bin_indices")
            if flat_bins is not None:
                if (
                    not isinstance(flat_bins, list)
                    or not flat_bins
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in flat_bins
                    )
                    or len(set(flat_bins)) != len(flat_bins)
                ):
                    raise ValueError(
                        f"response {response_id!r} has invalid flat bins"
                    )
        else:
            surface_ids = spec.get("surface_ids", [])
            if (
                not isinstance(surface_ids, list)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in surface_ids
                )
                or len(set(surface_ids)) != len(surface_ids)
            ):
                raise ValueError(
                    f"response {response_id!r} has invalid surface IDs"
                )
        spec["kind"] = kind
        spec["particle"] = particle
        spec["is_key_response"] = True
        specs.append(spec)
    if len(set(response_ids)) != len(response_ids):
        raise ValueError("unbiased campaign response IDs must be unique")
    return specs


def _validate_declared_unassessed_metrics(values: Any) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("declared unassessed metrics must be a mapping")
    result = {str(name): str(reason) for name, reason in values.items()}
    if any(not name or not reason for name, reason in result.items()):
        raise ValueError(
            "declared unassessed metric IDs and reasons must be nonempty"
        )
    return dict(sorted(result.items()))


def _validate_unbiased_seed_report(
    path: str | Path,
    *,
    expected_seed: int,
    expected_run_contract_sha256: str,
    expected_response_ids: set[str],
) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != _UNBIASED_SEED_REPORT_SCHEMA:
        raise ValueError("unsupported unbiased seed response report")
    if value.get("status") != "PASS" or value.get("variant") != "unbiased":
        raise ValueError("unbiased seed response report did not pass")
    if value.get("seed") != expected_seed:
        raise ValueError("unbiased seed response report seed changed")
    if value.get("run_contract_sha256") != expected_run_contract_sha256:
        raise ValueError("unbiased seed response report contract changed")
    stable = dict(value)
    evidence_sha256 = stable.pop("evidence_sha256", None)
    if evidence_sha256 != _canonical_json_sha256(stable):
        raise ValueError("unbiased seed response evidence hash is invalid")
    responses = value.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("unbiased seed response report has no responses")
    if {row.get("response_id") for row in responses} != expected_response_ids:
        raise ValueError("unbiased seed response inventory changed")
    if any(
        row.get("variant") != "unbiased"
        or row.get("is_key_response") is not True
        or row.get("seed") != expected_seed
        or row.get("run_contract_sha256") != expected_run_contract_sha256
        for row in responses
    ):
        raise ValueError("unbiased seed response row binding is invalid")
    run_artifact = value.get("run_artifact_sha256")
    if (
        not isinstance(run_artifact, str)
        or len(run_artifact) != 64
        or any(
            character not in "0123456789abcdef" for character in run_artifact
        )
        or any(
            row.get("run_artifact_sha256") != run_artifact for row in responses
        )
    ):
        raise ValueError("unbiased seed run artifact binding is invalid")
    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("unbiased seed report lacks execution evidence")
    artifacts = [execution.get("statepoint"), execution.get("log")]
    artifacts.extend(execution.get("surface_sources", []))
    artifacts.extend(execution.get("xml_files", []))
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("unbiased seed execution artifact is invalid")
        artifact_path = Path(str(artifact.get("path", ""))).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if artifact_path.stat().st_size != artifact.get("size_bytes"):
            raise ValueError("unbiased seed execution artifact size changed")
        if _sha256(artifact_path) != artifact.get("sha256"):
            raise ValueError("unbiased seed execution artifact hash changed")
    if execution["statepoint"].get("sha256") != run_artifact:
        raise ValueError("unbiased seed statepoint hash binding is invalid")
    diagnostics = value.get("run_diagnostics")
    if not isinstance(diagnostics, Mapping) or any(
        diagnostics.get(name) != expected
        for name, expected in {
            "lost_particles": 0,
            "dagmc_navigation_failures": 0,
            "runaway_histories": 0,
            "surface_source_capacity_reached": False,
        }.items()
    ):
        raise ValueError("unbiased seed diagnostics are not passing")
    return value


def run_unbiased_qualification_campaign(
    stage: Mapping[str, Any], output_root: Path
):
    """Run isolated unbiased seeds and write restartable Gate-I evidence."""
    from contextlib import redirect_stdout
    import subprocess
    import time

    import h5py

    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required to execute the unbiased campaign"
        ) from exc

    raw_seeds = stage.get("seeds")
    if not isinstance(raw_seeds, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_seeds
    ):
        raise TypeError("unbiased campaign seeds must be an integer list")
    seeds = tuple(raw_seeds)
    raw_minimum_seed_count = stage.get("minimum_seed_count", 3)
    if isinstance(raw_minimum_seed_count, bool) or not isinstance(
        raw_minimum_seed_count, int
    ):
        raise TypeError(
            "unbiased campaign minimum seed count must be an integer"
        )
    minimum_seed_count = raw_minimum_seed_count
    if minimum_seed_count < 3:
        raise ValueError("unbiased statistical campaigns require three seeds")
    if (
        len(seeds) < minimum_seed_count
        or len(set(seeds)) != len(seeds)
        or any(seed <= 0 for seed in seeds)
    ):
        raise ValueError(
            "unbiased campaign requires distinct positive independent seeds"
        )
    report_paths = [
        Path(value).resolve() for value in stage["result_report_paths"]
    ]
    if len(report_paths) != len(seeds) or len(set(report_paths)) != len(
        report_paths
    ):
        raise ValueError(
            "each unbiased campaign seed requires one result report"
        )
    response_specs = _validate_unbiased_campaign_response_specs(
        stage.get("responses")
    )
    unassessed_metrics = _validate_declared_unassessed_metrics(
        stage.get("unassessed_metrics")
    )
    response_ids = {str(value["response_id"]) for value in response_specs}

    baseline = Path(stage["model_directory"]).resolve()
    model_files = {
        name: baseline / f"{name}.xml"
        for name in ("geometry", "materials", "settings", "tallies")
    }
    if (baseline / "plots.xml").is_file():
        model_files["plots"] = baseline / "plots.xml"
    missing = [
        str(path) for path in model_files.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"unbiased campaign baseline XML is absent: {missing}"
        )
    manifest_path = Path(
        stage.get(
            "model_manifest_path",
            baseline / "magnet_openmc_model_manifest.json",
        )
    ).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "parastell.magnet_openmc_model/v1.0.0":
        raise ValueError(
            "unbiased campaign model manifest schema is unsupported"
        )
    if manifest.get("execution_performed", False) is not False:
        raise ValueError(
            "unbiased campaign requires a prepared model manifest"
        )
    for name, path in model_files.items():
        recorded = manifest.get("xml_files", {}).get(path.name)
        if not isinstance(recorded, Mapping) or recorded.get(
            "sha256"
        ) != _sha256(path):
            raise ValueError(
                f"prepared model manifest does not bind {path.name}"
            )

    particles_per_batch = int(stage["particles_per_batch"])
    batches = int(stage["batches"])
    if particles_per_batch <= 0 or batches <= 0:
        raise ValueError(
            "unbiased campaign histories and batches must be positive"
        )
    prepared_max_surface_particles = int(
        manifest.get("surface_source", {}).get("max_particles", 0)
    )
    prepared_max_surface_files = int(
        manifest.get("surface_source", {}).get("max_source_files", 0)
    )
    max_surface_particles = int(
        stage.get(
            "max_surface_particles",
            prepared_max_surface_particles,
        )
    )
    max_surface_files = int(
        stage.get(
            "max_surface_files",
            prepared_max_surface_files,
        )
    )
    if max_surface_particles <= 0 or max_surface_files <= 0:
        raise ValueError(
            "unbiased campaign surface-bank caps must be positive"
        )
    if (
        max_surface_particles != prepared_max_surface_particles
        or max_surface_files != prepared_max_surface_files
    ):
        raise ValueError(
            "unbiased campaign surface-bank caps changed from the prepared model"
        )
    configured_capacity = max_surface_particles * max_surface_files

    executable = str(stage.get("openmc_executable", "openmc"))
    version = subprocess.run(
        [executable, "--version"],
        cwd=baseline,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_version = str(stage.get("expected_openmc_version", "0.16.0"))
    if f"OpenMC version {expected_version}" not in version:
        raise ValueError(
            f"OpenMC executable is not expected version {expected_version}"
        )
    commit_lines = [
        line.partition(":")[2].strip().lower()
        for line in version.splitlines()
        if line.strip().lower().startswith("commit hash:")
    ]
    if (
        len(commit_lines) != 1
        or len(commit_lines[0]) not in {40, 64}
        or any(
            character not in "0123456789abcdef"
            for character in commit_lines[0]
        )
    ):
        raise ValueError(
            "OpenMC executable did not report one full commit SHA"
        )
    openmc_commit = commit_lines[0]
    expected_commit = stage.get("expected_openmc_commit")
    if (
        expected_commit is not None
        and openmc_commit != str(expected_commit).lower()
    ):
        raise ValueError(
            "OpenMC executable commit does not match configuration"
        )

    contract_payload = {
        "schema": "parastell.unbiased_campaign_run_contract/v1.0.0",
        "prepared_model_manifest_sha256": _sha256(manifest_path),
        "xml_sha256": {
            name: _sha256(path) for name, path in sorted(model_files.items())
        },
        "particles_per_batch": particles_per_batch,
        "batches": batches,
        "histories_per_seed": particles_per_batch * batches,
        "max_surface_particles": max_surface_particles,
        "max_surface_files": max_surface_files,
        "response_specs": response_specs,
        "unassessed_metrics": unassessed_metrics,
        "openmc_version": expected_version,
        "openmc_commit": openmc_commit,
        "handler_sha256": _sha256(Path(__file__).resolve()),
    }
    run_contract_sha256 = _canonical_json_sha256(contract_payload)
    campaign_root = (
        Path(stage["campaign_directory"]).resolve() / run_contract_sha256[:16]
    )
    campaign_root.mkdir(parents=True, exist_ok=True)
    run_reports = []

    for seed, report_path in zip(seeds, report_paths, strict=True):
        try:
            cached = _validate_unbiased_seed_report(
                report_path,
                expected_seed=seed,
                expected_run_contract_sha256=run_contract_sha256,
                expected_response_ids=response_ids,
            )
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            cached = None
        if cached is None:
            seed_root = campaign_root / f"seed-{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            attempts = [
                int(path.name.partition("-")[2])
                for path in seed_root.glob("attempt-*")
                if path.is_dir() and path.name.partition("-")[2].isdigit()
            ]
            run_directory = (
                seed_root / f"attempt-{max(attempts, default=0) + 1}"
            )
            run_directory.mkdir(parents=True, exist_ok=False)
            model = openmc.Model.from_xml(**model_files)
            model.settings.seed = seed
            model.settings.particles = particles_per_batch
            model.settings.batches = batches
            model.settings.inactive = 0
            model.settings.weight_window_generators = []
            model.settings.weight_windows = []
            model.settings.weight_windows_file = None
            model.settings.weight_windows_on = False
            model.export_to_xml(directory=run_directory)

            log_path = run_directory / "openmc.log"
            started = time.perf_counter()
            with (
                log_path.open("w", encoding="utf-8") as log,
                redirect_stdout(log),
            ):
                openmc.run(
                    cwd=run_directory,
                    threads=int(stage.get("threads", 1)),
                    openmc_exec=executable,
                    output=True,
                )
            elapsed = time.perf_counter() - started
            statepoint_path = run_directory / f"statepoint.{batches}.h5"
            if not statepoint_path.is_file():
                raise FileNotFoundError(statepoint_path)
            lost = sorted(run_directory.glob("particle_*.h5"))
            if lost:
                raise RuntimeError(
                    f"unbiased seed {seed} wrote lost-particle files"
                )
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            lowered_log = log_text.lower()
            unstable_patterns = {
                "dagmc error": "DAGMC navigation failure",
                "could not find the cell": "cell navigation failure",
                "could not be located in any cell": "lost particle diagnostic",
                "maximum number of surface crossings": "runaway history",
                "maximum number of splits": "unexpected split instability",
            }
            matched = [
                label
                for pattern, label in unstable_patterns.items()
                if pattern in lowered_log
            ]
            if matched:
                raise RuntimeError(
                    f"unbiased seed {seed} is unstable: {sorted(matched)}"
                )

            with h5py.File(statepoint_path) as statepoint_hdf5:
                for name, expected in (
                    ("current_batch", batches),
                    ("n_batches", batches),
                    ("n_particles", particles_per_batch),
                ):
                    if (
                        name not in statepoint_hdf5
                        or int(statepoint_hdf5[name][()]) != expected
                    ):
                        raise ValueError(
                            f"unbiased seed {seed} has incomplete {name}"
                        )
                if not bool(
                    statepoint_hdf5.attrs.get("tallies_present", False)
                ):
                    raise ValueError(
                        f"unbiased seed {seed} statepoint has no tallies"
                    )
                if any(
                    spec["particle"] == "photon" for spec in response_specs
                ) and not bool(
                    statepoint_hdf5.attrs.get("photon_transport", False)
                ):
                    raise ValueError(
                        f"unbiased seed {seed} omitted photon transport"
                    )

            surface_banks = sorted(run_directory.glob("surface_source*.h5"))
            if not surface_banks:
                raise FileNotFoundError(
                    f"unbiased seed {seed} produced no surface-source bank"
                )
            if len(surface_banks) > max_surface_files:
                raise RuntimeError(
                    f"unbiased seed {seed} exceeded the surface-source file cap"
                )
            banks = []
            for source_path in surface_banks:
                with h5py.File(source_path) as source:
                    if "source_bank" not in source:
                        raise ValueError(f"{source_path} has no source_bank")
                    bank = source["source_bank"][:]
                required_fields = {"particle", "surf_id", "wgt"}
                if bank.dtype.names is None or not required_fields.issubset(
                    bank.dtype.names
                ):
                    raise ValueError(
                        f"{source_path} has an unsupported source-bank schema"
                    )
                particles = np.asarray(bank["particle"]).reshape(-1)
                unknown_particles = set(np.unique(particles)) - {22, 2112}
                if unknown_particles:
                    raise ValueError(
                        "unbiased campaign source bank contains unsupported "
                        f"particles {sorted(unknown_particles)}"
                    )
                weights = np.asarray(bank["wgt"], dtype=float).reshape(-1)
                if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                    raise ValueError(
                        "unbiased campaign source bank has invalid weights"
                    )
                banks.append(bank)
            stored_records = sum(len(bank) for bank in banks)
            if stored_records >= configured_capacity:
                raise RuntimeError(
                    f"unbiased seed {seed} reached the surface-source capacity"
                )

            responses = []
            with openmc.StatePoint(
                statepoint_path, autolink=False
            ) as statepoint:
                for spec in response_specs:
                    kind = str(spec["kind"])
                    if kind == "surface_bank_weight":
                        pdg = 2112 if spec["particle"] == "neutron" else 22
                        surface_ids = {
                            int(value) for value in spec.get("surface_ids", [])
                        }
                        selected_weights = []
                        for bank in banks:
                            particle = np.asarray(bank["particle"]).reshape(-1)
                            mask = particle == pdg
                            if surface_ids:
                                mask &= np.isin(
                                    bank["surf_id"], list(surface_ids)
                                )
                            selected_weights.extend(
                                np.asarray(bank["wgt"])[mask]
                                .astype(float)
                                .tolist()
                            )
                        weights = np.asarray(selected_weights, dtype=float)
                        histories = particles_per_batch * batches
                        estimate = float(np.sum(weights) / histories)
                        within_std = float(
                            np.sqrt(np.sum(weights * weights)) / histories
                        )
                        sum_squared = float(np.sum(weights * weights))
                        effective_sample_size = (
                            float(np.sum(weights) ** 2 / sum_squared)
                            if sum_squared > 0.0
                            else 0.0
                        )
                        effective_sample_size_status = (
                            "AVAILABLE_EXACT_WEIGHTED_SURFACE_BANK"
                        )
                        raw_count = int(len(weights))
                    else:
                        tally = statepoint.get_tally(
                            name=str(spec["tally_name"])
                        )
                        mean = np.asarray(tally.mean, dtype=float).reshape(-1)
                        std = np.asarray(tally.std_dev, dtype=float).reshape(
                            -1
                        )
                        selected_bins = spec.get("flat_bin_indices")
                        if selected_bins is not None:
                            selected = np.asarray(selected_bins, dtype=int)
                            if np.any(selected >= len(mean)):
                                raise ValueError(
                                    f"response {spec['response_id']!r} selects absent bins"
                                )
                            mean = mean[selected]
                            std = std[selected]
                        if (
                            not len(mean)
                            or not np.all(np.isfinite(mean))
                            or not np.all(np.isfinite(std))
                        ):
                            raise ValueError(
                                f"response {spec['response_id']!r} contains invalid tally data"
                            )
                        if np.any(std < 0.0):
                            raise ValueError(
                                f"response {spec['response_id']!r} has negative uncertainty"
                            )
                        if str(spec.get("reduction", "sum")) == "sum":
                            estimate = float(np.sum(mean))
                            within_std = float(np.sum(np.abs(std)))
                        else:
                            estimate = float(np.mean(mean))
                            within_std = float(np.sum(np.abs(std)) / len(std))
                        effective_sample_size = None
                        effective_sample_size_status = (
                            "UNAVAILABLE_TALLY_DOES_NOT_REPORT_EVENT_WEIGHTS"
                        )
                        raw_count = None
                    responses.append(
                        {
                            "variant": "unbiased",
                            "is_key_response": True,
                            "response_id": str(spec["response_id"]),
                            "response_definition_sha256": _canonical_json_sha256(
                                spec
                            ),
                            "run_contract_sha256": run_contract_sha256,
                            "run_artifact_sha256": _sha256(statepoint_path),
                            "seed": seed,
                            "particle": str(spec["particle"]),
                            "quantity": str(spec["quantity"]),
                            "units": str(spec["units"]),
                            "normalization": str(spec["normalization"]),
                            "estimate": estimate,
                            "within_run_std_dev": within_std,
                            "effective_sample_size": effective_sample_size,
                            "effective_sample_size_status": (
                                effective_sample_size_status
                            ),
                            "raw_count": raw_count,
                        }
                    )

            def artifact(path: Path) -> dict[str, Any]:
                return {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }

            xml_artifacts = [
                artifact(path) for path in sorted(run_directory.glob("*.xml"))
            ]
            statepoint_artifact = artifact(statepoint_path)
            seed_report = {
                "schema": _UNBIASED_SEED_REPORT_SCHEMA,
                "status": "PASS",
                "variant": "unbiased",
                "seed": seed,
                "run_contract_sha256": run_contract_sha256,
                "run_artifact_sha256": statepoint_artifact["sha256"],
                "histories": particles_per_batch * batches,
                "responses": responses,
                "execution": {
                    "run_directory": str(run_directory),
                    "elapsed_wall_seconds": elapsed,
                    "statepoint": statepoint_artifact,
                    "log": artifact(log_path),
                    "surface_sources": [
                        artifact(path) for path in surface_banks
                    ],
                    "xml_files": xml_artifacts,
                    "stored_surface_source_records": stored_records,
                    "configured_surface_source_capacity": configured_capacity,
                    "capacity_reached": False,
                },
                "run_diagnostics": {
                    "lost_particles": 0,
                    "dagmc_navigation_failures": 0,
                    "runaway_histories": 0,
                    "surface_source_capacity_reached": False,
                },
            }
            seed_report["evidence_sha256"] = _canonical_json_sha256(
                seed_report
            )
            _write(report_path, seed_report)
            cached = _validate_unbiased_seed_report(
                report_path,
                expected_seed=seed,
                expected_run_contract_sha256=run_contract_sha256,
                expected_response_ids=response_ids,
            )
        run_reports.append(
            {
                "seed": seed,
                "path": str(report_path),
                "sha256": _sha256(report_path),
                "size_bytes": report_path.stat().st_size,
                "run_artifact_sha256": cached["run_artifact_sha256"],
                "evidence_sha256": cached["evidence_sha256"],
            }
        )

    campaign_report = {
        "schema": _UNBIASED_CAMPAIGN_SCHEMA,
        "status": "PASS",
        "transport": "UNBIASED_ONLY",
        "run_contract": contract_payload,
        "run_contract_sha256": run_contract_sha256,
        "seeds": list(seeds),
        "minimum_seed_count": minimum_seed_count,
        "histories_per_seed": particles_per_batch * batches,
        "unassessed_metrics": unassessed_metrics,
        "result_reports": run_reports,
        "openmc_executable_version": version,
        "run_diagnostics": {
            "lost_particles": 0,
            "dagmc_navigation_failures": 0,
            "runaway_histories": 0,
            "surface_source_capacity_reached": False,
        },
    }
    campaign_report["evidence_sha256"] = _canonical_json_sha256(
        campaign_report
    )
    return _write(stage["campaign_report_path"], campaign_report)


def prepare_weight_window_generation(
    stage: Mapping[str, Any], output_root: Path
):
    disabled = _disabled_weight_window_status(
        stage["weight_window_mesh_status_path"]
    )
    if disabled is not None:
        return _write(
            stage["generation_contract_path"],
            _skipped_weight_window_stage(
                stage_name="weight_window_generation_preparation",
                disabled=disabled,
            ),
        )
    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required to prepare WW generation"
        ) from exc
    from .weight_windows import WeightWindowArtifactContract
    from .weight_windows import configure_magic_generator

    baseline = Path(stage["unbiased_model_directory"]).resolve()
    model_files = dict(
        geometry=baseline / "geometry.xml",
        materials=baseline / "materials.xml",
        settings=baseline / "settings.xml",
        tallies=baseline / "tallies.xml",
    )
    if (baseline / "plots.xml").is_file():
        model_files["plots"] = baseline / "plots.xml"
    model = openmc.Model.from_xml(**model_files)
    configure_magic_generator(
        model.settings,
        mesh_path=stage["weight_window_mesh_path"],
        energy_bounds_eV=stage["energy_bounds_eV"],
        particle_type=stage.get("particle_type", "neutron"),
        batches=int(stage["batches"]),
        update_interval=int(stage.get("update_interval", 1)),
        max_history_splits=int(stage["max_history_splits"]),
    )
    model.settings.particles = int(stage["particles_per_batch"])
    model.settings.batches = int(stage["batches"])
    model.settings.inactive = 0
    model.settings.seed = int(stage["seed"])
    output = Path(stage["generation_model_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.export_to_xml(directory=output)
    contract = WeightWindowArtifactContract(
        canonical_geometry_fingerprint=stage["canonical_geometry_fingerprint"],
        raw_h5m_sha256=stage["raw_h5m_sha256"],
        source_definition_sha256=stage["source_definition_sha256"],
        source_mesh_sha256=stage["source_mesh_sha256"],
        physical_source_rate_per_s=float(stage["physical_source_rate_per_s"]),
        material_manifest_sha256=stage["material_manifest_sha256"],
        nuclear_data_manifest_sha256=stage["nuclear_data_manifest_sha256"],
        weight_window_mesh_sha256=_sha256(stage["weight_window_mesh_path"]),
        particle_type=stage.get("particle_type", "neutron"),
        energy_bounds_eV=tuple(
            float(value) for value in stage["energy_bounds_eV"]
        ),
        generator_method="magic",
        generator_settings={
            "update_interval": int(stage.get("update_interval", 1)),
            "max_history_splits": int(stage["max_history_splits"]),
            "on_the_fly": True,
            "checkpoints": {"collision": True, "surface": True},
            "max_split": int(stage.get("max_split", 10)),
            "survival_ratio": float(stage.get("survival_ratio", 3.0)),
            "max_lower_bound_ratio": stage.get("max_lower_bound_ratio"),
            "weight_cutoff": float(stage.get("weight_cutoff", 1.0e-38)),
        },
        openmc_version=openmc.__version__,
        openmc_source_sha=stage["openmc_source_sha"],
        generation_histories=int(stage["particles_per_batch"])
        * int(stage["batches"]),
        generation_batches=int(stage["batches"]),
        generation_seed=int(stage["seed"]),
        selected_magnet_ids=tuple(stage["selected_magnet_ids"]),
    )
    return _write(
        stage["generation_contract_path"],
        {
            **contract.to_dict(),
            "contract_sha256": contract.sha256,
            "xml_files": {
                path.name: {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(output.glob("*.xml"))
            },
            "execution_performed": False,
        },
    )


def run_weight_window_generation(stage: Mapping[str, Any], output_root: Path):
    """Execute MAGIC and finalize one semantically validated WW artifact."""
    disabled = _disabled_weight_window_status(
        stage["generation_contract_path"]
    )
    if disabled is not None:
        return _write(
            stage["generation_report_path"],
            _skipped_weight_window_stage(
                stage_name="weight_window_generation_execution",
                disabled=disabled,
            ),
        )
    from contextlib import redirect_stdout
    import subprocess
    import time

    import h5py

    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required to execute WW generation"
        ) from exc
    from .weight_windows import validate_weight_window_hdf5
    from .weight_windows import weight_window_contract_from_mapping
    from .weight_windows import write_weight_window_contract

    model_directory = Path(stage["generation_model_directory"]).resolve()
    required_xml = tuple(
        model_directory / name
        for name in (
            "geometry.xml",
            "materials.xml",
            "settings.xml",
            "tallies.xml",
        )
    )
    missing_xml = [str(path) for path in required_xml if not path.is_file()]
    if missing_xml:
        raise FileNotFoundError(
            f"MAGIC generation XML is absent: {missing_xml}"
        )
    prepared = json.loads(
        Path(stage["generation_contract_path"]).read_text(encoding="utf-8")
    )
    contract = weight_window_contract_from_mapping(prepared)
    executable = str(stage.get("openmc_executable", "openmc"))
    version = subprocess.run(
        [executable, "--version"],
        cwd=model_directory,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_version = str(stage.get("expected_openmc_version", "0.16.0"))
    expected_commit = stage.get("expected_openmc_commit")
    if f"OpenMC version {expected_version}" not in version:
        raise ValueError(
            f"OpenMC executable is not expected version {expected_version}"
        )
    if expected_commit and f"Commit hash: {expected_commit}" not in version:
        raise ValueError(
            "OpenMC executable commit does not match configuration"
        )
    if openmc.__version__.split("+")[0] != expected_version:
        raise ValueError("OpenMC Python API and executable versions differ")

    log_path = Path(stage["generation_log_path"]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
        openmc.run(
            cwd=model_directory,
            threads=int(stage.get("threads", 1)),
            openmc_exec=executable,
            output=True,
        )
    elapsed = time.perf_counter() - started

    statepoint_path = Path(stage["generation_statepoint_path"]).resolve()
    artifact_path = Path(stage["weight_window_artifact_path"]).resolve()
    if not statepoint_path.is_file():
        raise FileNotFoundError(statepoint_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    lost = sorted(model_directory.glob("particle_*.h5"))
    if lost:
        raise RuntimeError(
            "MAGIC generation wrote lost-particle files: "
            + ", ".join(path.name for path in lost)
        )
    with h5py.File(statepoint_path) as statepoint:
        current_batch = int(statepoint["current_batch"][()])
        n_batches = int(statepoint["n_batches"][()])
        particles = int(statepoint["n_particles"][()])
    if (
        current_batch != contract.generation_batches
        or n_batches != current_batch
    ):
        raise ValueError("MAGIC generation statepoint is incomplete")
    if particles * n_batches != contract.generation_histories:
        raise ValueError(
            "MAGIC generation histories do not match the contract"
        )
    semantic = validate_weight_window_hdf5(
        artifact_path, expected_contract=contract
    )
    finalized = write_weight_window_contract(
        stage["weight_window_artifact_contract_path"],
        contract,
        weight_window_path=artifact_path,
    )
    report = {
        "schema": "parastell.magnet_weight_window_generation_execution/v1.0.0",
        "status": "PASS",
        "execution_performed": True,
        "openmc_executable_version": version,
        "elapsed_wall_seconds": elapsed,
        "statepoint": {
            "path": str(statepoint_path),
            "sha256": _sha256(statepoint_path),
            "size_bytes": statepoint_path.stat().st_size,
            "current_batch": current_batch,
            "n_batches": n_batches,
            "particles_per_batch": particles,
        },
        "log": {
            "path": str(log_path),
            "sha256": _sha256(log_path),
            "size_bytes": log_path.stat().st_size,
        },
        "weight_window_artifact": finalized["weight_window_artifact"],
        "artifact_contract_sha256": finalized["contract_sha256"],
        "semantic_validation": semantic,
        "lost_particle_files": [],
    }
    return _write(stage["generation_report_path"], report)


def run_weight_window_qualification_campaign(
    stage: Mapping[str, Any], output_root: Path
):
    """Run isolated, paired unbiased/fixed-WW seeds and extract responses."""
    disabled = _disabled_weight_window_status(stage["generation_report_path"])
    if disabled is not None:
        empty = {
            "schema": "parastell.magnet_weight_window_responses/v1.0.0",
            "status": "SKIPPED_UNBIASED_FALLBACK",
            "classification": disabled["classification"],
            "responses": [],
        }
        _write(stage["unbiased_responses_path"], empty)
        _write(stage["weight_window_responses_path"], empty)
        return _write(
            stage["campaign_report_path"],
            _skipped_weight_window_stage(
                stage_name="weight_window_campaign", disabled=disabled
            ),
        )
    from contextlib import redirect_stdout
    import time

    import h5py

    try:
        import openmc
    except ImportError as exc:
        raise RuntimeError(
            "OpenMC 0.16 is required to execute WW qualification"
        ) from exc
    from .weight_windows import aggregate_weight_window_campaign_rows
    from .weight_windows import require_compatible_weight_window
    from .weight_windows import validate_weight_window_hdf5
    from .weight_windows import weight_window_contract_from_mapping

    seeds = tuple(int(value) for value in stage["seeds"])
    minimum_seed_count = int(stage.get("minimum_seed_count", 3))
    if len(seeds) < minimum_seed_count or len(set(seeds)) != len(seeds):
        raise ValueError(
            "WW qualification requires distinct independent seeds"
        )
    prepared = json.loads(
        Path(stage["generation_contract_path"]).read_text(encoding="utf-8")
    )
    expected_contract = weight_window_contract_from_mapping(prepared)
    if expected_contract.generation_seed in seeds:
        raise ValueError(
            "MAGIC generation seed must differ from qualification seeds"
        )
    compatible = require_compatible_weight_window(
        stage["weight_window_artifact_contract_path"], expected_contract
    )
    artifact_path = Path(
        compatible["resolved_weight_window_artifact_path"]
    ).resolve()
    semantic = validate_weight_window_hdf5(
        artifact_path, expected_contract=expected_contract
    )

    baseline = Path(stage["unbiased_model_directory"]).resolve()
    model_files = dict(
        geometry=baseline / "geometry.xml",
        materials=baseline / "materials.xml",
        settings=baseline / "settings.xml",
        tallies=baseline / "tallies.xml",
    )
    if (baseline / "plots.xml").is_file():
        model_files["plots"] = baseline / "plots.xml"
    missing = [
        str(path) for path in model_files.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"qualification baseline XML is absent: {missing}"
        )
    response_specs = [dict(value) for value in stage["responses"]]
    response_ids = [
        str(value.get("response_id", "")) for value in response_specs
    ]
    if not response_ids or any(not value for value in response_ids):
        raise ValueError("qualification responses require stable response IDs")
    if len(set(response_ids)) != len(response_ids):
        raise ValueError("qualification response IDs must be unique")

    particles_per_batch = int(stage["particles_per_batch"])
    batches = int(stage["batches"])
    if particles_per_batch <= 0 or batches <= 0:
        raise ValueError(
            "qualification histories and batches must be positive"
        )
    contract_payload = {
        "geometry_xml_sha256": _sha256(model_files["geometry"]),
        "materials_xml_sha256": _sha256(model_files["materials"]),
        "tallies_xml_sha256": _sha256(model_files["tallies"]),
        "particles_per_batch": particles_per_batch,
        "batches": batches,
        "response_specs": response_specs,
        "weight_window_contract_sha256": compatible["contract_sha256"],
    }
    run_contract_sha256 = hashlib.sha256(
        json.dumps(
            contract_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    executable = str(stage.get("openmc_executable", "openmc"))
    campaign_directory = Path(stage["campaign_directory"]).resolve()
    campaign_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    run_reports = []

    for variant in ("unbiased", "weight_window"):
        for seed in seeds:
            run_directory = campaign_directory / variant / f"seed-{seed}"
            run_directory.mkdir(parents=True, exist_ok=True)
            model = openmc.Model.from_xml(**model_files)
            model.settings.seed = seed
            model.settings.particles = particles_per_batch
            model.settings.batches = batches
            model.settings.inactive = 0
            model.settings.weight_window_generators = []
            model.settings.weight_windows = []
            model.settings.weight_windows_file = None
            model.settings.weight_windows_on = variant == "weight_window"
            if variant == "weight_window":
                model.settings.weight_windows_file = str(artifact_path)
                model.settings.max_history_splits = int(
                    stage["max_history_splits"]
                )
            model.export_to_xml(directory=run_directory)
            log_path = run_directory / "openmc.log"
            started = time.perf_counter()
            with (
                log_path.open("w", encoding="utf-8") as log,
                redirect_stdout(log),
            ):
                openmc.run(
                    cwd=run_directory,
                    threads=int(stage.get("threads", 1)),
                    openmc_exec=executable,
                    output=True,
                )
            elapsed = time.perf_counter() - started
            statepoint_path = run_directory / f"statepoint.{batches}.h5"
            if not statepoint_path.is_file():
                raise FileNotFoundError(statepoint_path)
            lost = sorted(run_directory.glob("particle_*.h5"))
            if lost:
                raise RuntimeError(
                    f"qualification {variant} seed {seed} wrote lost particles"
                )
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if (
                "DAGMC ERROR" in log_text
                or "Maximum number of splits" in log_text
            ):
                raise RuntimeError(
                    f"qualification {variant} seed {seed} is unstable"
                )

            surface_banks = sorted(run_directory.glob("surface_source*.h5"))
            with openmc.StatePoint(
                statepoint_path, autolink=False
            ) as statepoint:
                for spec in response_specs:
                    response_id = str(spec["response_id"])
                    definition_hash = hashlib.sha256(
                        json.dumps(
                            spec, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest()
                    if spec.get("kind", "tally") == "surface_bank_weight":
                        pdg = 2112 if spec["particle"] == "neutron" else 22
                        surface_ids = {
                            int(value) for value in spec.get("surface_ids", [])
                        }
                        weights = []
                        for source_path in surface_banks:
                            with h5py.File(source_path) as source:
                                bank = source["source_bank"][:]
                            mask = (
                                np.asarray(bank["particle"]).reshape(-1) == pdg
                            )
                            if surface_ids:
                                mask &= np.isin(
                                    bank["surf_id"], list(surface_ids)
                                )
                            weights.extend(
                                np.asarray(bank["wgt"])[mask]
                                .astype(float)
                                .tolist()
                            )
                        weights = np.asarray(weights, dtype=float)
                        histories = particles_per_batch * batches
                        estimate = float(np.sum(weights) / histories)
                        within_std = float(
                            np.sqrt(np.sum(weights * weights)) / histories
                        )
                        effective_sample_size = (
                            float(
                                np.sum(weights) ** 2
                                / np.sum(weights * weights)
                            )
                            if len(weights) and np.sum(weights * weights) > 0.0
                            else 0.0
                        )
                    else:
                        tally = statepoint.get_tally(
                            name=str(spec["tally_name"])
                        )
                        mean = np.asarray(tally.mean, dtype=float).reshape(-1)
                        std = np.asarray(tally.std_dev, dtype=float).reshape(
                            -1
                        )
                        selected_bins = spec.get("flat_bin_indices")
                        if selected_bins is not None:
                            selected_bins = np.asarray(
                                selected_bins, dtype=int
                            )
                            mean = mean[selected_bins]
                            std = std[selected_bins]
                        reduction = str(spec.get("reduction", "sum"))
                        if reduction == "sum":
                            estimate = float(np.sum(mean))
                            within_std = float(np.sqrt(np.sum(std * std)))
                        elif reduction == "mean":
                            estimate = float(np.mean(mean))
                            within_std = float(
                                np.sqrt(np.sum(std * std)) / len(std)
                            )
                        else:
                            raise ValueError(
                                f"unsupported qualification reduction {reduction!r}"
                            )
                        effective_sample_size = None
                    row = {
                        "variant": variant,
                        "seed": seed,
                        "response_id": response_id,
                        "definition_hash": definition_hash,
                        "run_contract_sha256": run_contract_sha256,
                        "particle": str(spec["particle"]),
                        "primary": bool(spec.get("primary", False)),
                        "critical": bool(spec.get("critical", False)),
                        "estimate": estimate,
                        "within_run_std_dev": within_std,
                        "wall_time_s": elapsed,
                        "effective_sample_size": effective_sample_size,
                        "statepoint_sha256": _sha256(statepoint_path),
                    }
                    rows.append(row)
            run_reports.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "wall_time_s": elapsed,
                    "statepoint_path": str(statepoint_path),
                    "statepoint_sha256": _sha256(statepoint_path),
                    "log_path": str(log_path),
                    "log_sha256": _sha256(log_path),
                    "surface_bank_paths": [
                        str(path) for path in surface_banks
                    ],
                    "surface_bank_sha256": [
                        _sha256(path) for path in surface_banks
                    ],
                    "lost_particles": 0,
                    "dagmc_navigation_failures": 0,
                    "runaway_histories": 0,
                    "pathological_split_behavior": False,
                }
            )
    aggregated = aggregate_weight_window_campaign_rows(
        rows, minimum_seed_count=minimum_seed_count
    )
    _write(
        stage["unbiased_responses_path"], {"responses": aggregated["unbiased"]}
    )
    _write(
        stage["weight_window_responses_path"],
        {"responses": aggregated["weight_window"]},
    )
    report = {
        "schema": "parastell.magnet_weight_window_campaign/v1.0.0",
        "status": "PASS",
        "run_contract_sha256": run_contract_sha256,
        "seeds": list(seeds),
        "minimum_seed_count": minimum_seed_count,
        "weight_window_semantic_validation": semantic,
        "run_reports": run_reports,
        "run_diagnostics": {
            "lost_particles": 0,
            "dagmc_navigation_failures": 0,
            "runaway_histories": 0,
            "pathological_split_behavior": False,
        },
    }
    return _write(stage["campaign_report_path"], report)


def prepare_production_model(stage: Mapping[str, Any], output_root: Path):
    import shutil

    qualification = json.loads(
        Path(stage["qualification_path"]).read_text(encoding="utf-8")
    )
    baseline = Path(stage["unbiased_model_directory"]).resolve()
    output = Path(stage["production_model_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    enabled = bool(qualification.get("weight_windows_enabled", False))
    if enabled:
        try:
            import openmc
        except ImportError as exc:
            raise RuntimeError(
                "OpenMC 0.16 is required to prepare production XML"
            ) from exc
        from .weight_windows import require_compatible_weight_window
        from .weight_windows import validate_weight_window_hdf5
        from .weight_windows import weight_window_contract_from_mapping

        prepared_contract = json.loads(
            Path(stage["generation_contract_path"]).read_text(encoding="utf-8")
        )
        expected_contract = weight_window_contract_from_mapping(
            prepared_contract
        )
        campaign = json.loads(
            Path(stage["campaign_report_path"]).read_text(encoding="utf-8")
        )
        _validate_enabled_weight_window_qualification(
            qualification,
            campaign,
            expected_particle_type=expected_contract.particle_type,
        )
        compatible = require_compatible_weight_window(
            stage["weight_window_artifact_contract_path"], expected_contract
        )
        qualified_magnets = set(expected_contract.selected_magnet_ids)
        production_magnets = set(stage["production_selected_magnet_ids"])
        if production_magnets != qualified_magnets:
            raise ValueError(
                "production magnet selection differs from the qualified WW "
                "artifact scope"
            )
        model_files = dict(
            geometry=baseline / "geometry.xml",
            materials=baseline / "materials.xml",
            settings=baseline / "settings.xml",
            tallies=baseline / "tallies.xml",
        )
        if (baseline / "plots.xml").is_file():
            model_files["plots"] = baseline / "plots.xml"
        model = openmc.Model.from_xml(**model_files)
        ww_path = Path(
            compatible["resolved_weight_window_artifact_path"]
        ).resolve()
        configured_path = Path(stage["weight_window_artifact_path"]).resolve()
        if configured_path != ww_path:
            raise ValueError(
                "configured WW artifact path differs from the finalized contract"
            )
        if not ww_path.is_file():
            raise FileNotFoundError(ww_path)
        semantic = validate_weight_window_hdf5(
            ww_path, expected_contract=expected_contract
        )
        model.settings.weight_window_generators = []
        model.settings.weight_windows = []
        model.settings.weight_windows_file = str(ww_path)
        model.settings.weight_windows_on = True
        model.settings.max_history_splits = int(stage["max_history_splits"])
        model.settings.seed = int(stage["production_seed"])
        model.export_to_xml(directory=output)
        transport = "QUALIFIED_WEIGHT_WINDOWS"
    else:
        for path in baseline.glob("*.xml"):
            shutil.copy2(path, output / path.name)
        transport = "UNBIASED_FALLBACK"
        semantic = None
    result = {
        "schema": "parastell.magnet_production_model/v1.0.0",
        "qualification_classification": qualification["classification"],
        "weight_windows_enabled": enabled,
        "production_transport": transport,
        "weight_window_semantic_validation": semantic,
        "xml_files": {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(output.glob("*.xml"))
        },
        "execution_performed": False,
    }
    return _write(stage["production_model_manifest_path"], result)


def qualify_weight_windows(stage: Mapping[str, Any], output_root: Path):
    from .weight_windows import qualify_weight_windows as qualify

    campaign = json.loads(
        Path(stage["campaign_report_path"]).read_text(encoding="utf-8")
    )
    if campaign.get("status") == "SKIPPED_UNBIASED_FALLBACK":
        return _write(
            stage["qualification_path"],
            {
                **campaign,
                "schema": "parastell.magnet_weight_window_qualification/v1.0.0",
            },
        )
    if campaign.get("status") != "PASS":
        raise ValueError("WW qualification campaign is not complete")

    unbiased_value = json.loads(
        Path(stage["unbiased_responses_path"]).read_text(encoding="utf-8")
    )
    weighted_value = json.loads(
        Path(stage["weight_window_responses_path"]).read_text(encoding="utf-8")
    )
    unbiased = (
        unbiased_value["responses"]
        if isinstance(unbiased_value, Mapping)
        else unbiased_value
    )
    weighted = (
        weighted_value["responses"]
        if isinstance(weighted_value, Mapping)
        else weighted_value
    )
    result = qualify(
        unbiased,
        weighted,
        alpha=float(stage.get("alpha", 0.05)),
        minimum_geometric_mean_fom_ratio=float(
            stage.get("minimum_geometric_mean_fom_ratio", 2.0)
        ),
        minimum_improved_fraction=float(
            stage.get("minimum_improved_fraction", 0.75)
        ),
        minimum_critical_fom_ratio=float(
            stage.get("minimum_critical_fom_ratio", 0.8)
        ),
        minimum_seed_count=int(stage.get("minimum_seed_count", 3)),
        run_diagnostics=campaign["run_diagnostics"],
        selected_magnets_only=bool(stage.get("selected_magnets_only", False)),
        artifact_particle_type=str(
            stage.get("artifact_particle_type", "neutron")
        ),
    )
    return _write(stage["qualification_path"], result)


def qualify_production_statistics(stage: Mapping[str, Any], output_root: Path):
    """Aggregate hash-bound, independent unbiased response reports for Gate I."""
    from .production_statistics import (
        validate_production_statistical_qualification,
        write_production_statistical_qualification,
    )

    report_values = stage.get("result_report_paths")
    if not isinstance(report_values, list) or not report_values:
        raise ValueError(
            "production statistical qualification requires result_report_paths"
        )
    minimum_seed_count = int(stage.get("minimum_seed_count", 3))
    if len(report_values) < minimum_seed_count:
        raise ValueError(
            "production statistical qualification requires at least "
            f"{minimum_seed_count} result reports"
        )
    campaign_path = Path(stage["campaign_report_path"]).resolve()
    if not campaign_path.is_file():
        raise FileNotFoundError(campaign_path)
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    unassessed_metrics = _validate_declared_unassessed_metrics(
        stage.get("unassessed_metrics")
    )
    if (
        campaign.get("schema") != _UNBIASED_CAMPAIGN_SCHEMA
        or campaign.get("status") != "PASS"
        or campaign.get("transport") != "UNBIASED_ONLY"
    ):
        raise ValueError("unbiased statistical campaign did not pass")
    campaign_stable = dict(campaign)
    campaign_evidence = campaign_stable.pop("evidence_sha256", None)
    if campaign_evidence != _canonical_json_sha256(campaign_stable):
        raise ValueError("unbiased statistical campaign hash is invalid")
    if campaign.get("unassessed_metrics", {}) != unassessed_metrics:
        raise ValueError(
            "unbiased campaign and qualification unassessed metrics disagree"
        )
    campaign_reports = campaign.get("result_reports")
    if not isinstance(campaign_reports, list):
        raise ValueError("unbiased campaign lacks result-report bindings")
    configured_reports = [
        str(Path(value).resolve()) for value in report_values
    ]
    if [value.get("path") for value in campaign_reports] != configured_reports:
        raise ValueError("unbiased campaign result-report inventory changed")
    for binding in campaign_reports:
        report_path = Path(str(binding.get("path", ""))).resolve()
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        if report_path.stat().st_size != binding.get("size_bytes"):
            raise ValueError("unbiased campaign result-report size changed")
        if _sha256(report_path) != binding.get("sha256"):
            raise ValueError("unbiased campaign result-report hash changed")

    rows: list[dict[str, Any]] = []
    source_reports = []
    for raw_path in report_values:
        report_path = Path(raw_path).resolve()
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        value = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            report_rows = value.get("responses", value.get("rows"))
        else:
            report_rows = value
        if not isinstance(report_rows, list) or not report_rows:
            raise ValueError(
                f"unbiased result report has no response rows: {report_path}"
            )
        if any(not isinstance(row, Mapping) for row in report_rows):
            raise TypeError(
                f"unbiased result report rows must be mappings: {report_path}"
            )
        report_rows = [dict(row) for row in report_rows]
        seeds = {row.get("seed") for row in report_rows}
        if len(seeds) != 1:
            raise ValueError(
                "each unbiased result report must contain exactly one seed"
            )
        seed = next(iter(seeds))
        if isinstance(value, Mapping) and value.get("seed") not in {
            None,
            seed,
        }:
            raise ValueError(
                "result report seed disagrees with its response rows"
            )
        run_artifacts = {row.get("run_artifact_sha256") for row in report_rows}
        if len(run_artifacts) != 1:
            raise ValueError(
                "one unbiased result report changed run artifact hash"
            )
        run_artifact = next(iter(run_artifacts))
        if isinstance(value, Mapping) and value.get(
            "run_artifact_sha256"
        ) not in {None, run_artifact}:
            raise ValueError(
                "result report run artifact hash disagrees with response rows"
            )
        source_reports.append(
            {
                "path": str(report_path),
                "sha256": _sha256(report_path),
                "size_bytes": report_path.stat().st_size,
                "seed": seed,
                "row_count": len(report_rows),
                "run_artifact_sha256": run_artifact,
            }
        )
        rows.extend(report_rows)

    row_contracts = {row.get("run_contract_sha256") for row in rows}
    if row_contracts != {campaign.get("run_contract_sha256")}:
        raise ValueError("unbiased campaign and response contracts disagree")
    row_seeds = sorted({row.get("seed") for row in rows})
    if row_seeds != sorted(campaign.get("seeds", [])):
        raise ValueError("unbiased campaign and response seeds disagree")

    output = Path(stage["qualification_path"]).resolve()
    result = write_production_statistical_qualification(
        output,
        rows,
        qualification_thresholds=stage.get("qualification_thresholds"),
        minimum_seed_count=minimum_seed_count,
        source_reports=source_reports,
        declared_unassessed_metrics=unassessed_metrics,
    )
    validation = validate_production_statistical_qualification(
        output, verify_source_reports=True
    )
    return {
        "schema": "parastell.magnet_gate_i_stage/v1.0.0",
        "gate": "I_STATISTICAL_QUALIFICATION",
        "qualification_path": str(output),
        "evidence_sha256": result["evidence_sha256"],
        **validation,
    }


def postprocess(stage: Mapping[str, Any], output_root: Path):
    """Export neutral scalar flux, heating, reactions, and boundary records."""
    from .energy_groups import get_structure
    from .magnet_heating import export_magnet_heating
    from .magnet_damage_gas import (
        export_magnet_damage_gas,
        validate_magnet_damage_gas,
    )
    from .magnet_radiation_field import MagnetRadiationFieldProducer
    from .magnet_radiation_field import ProducerSelection
    from .magnet_reaction_production import (
        export_magnet_reaction_production,
        validate_magnet_reaction_production,
    )
    from .magnet_volume_flux import (
        _read_volume_flux_tally,
        build_scalar_flux_fields_from_statepoint,
        export_scalar_flux_fields,
        validate_spectra_pka_ready_flux,
    )
    from .openmc16_export import export_openmc16_handoffs

    statepoint_path = Path(stage["statepoint_path"]).resolve()
    model_manifest_path = Path(stage["model_manifest_path"]).resolve()
    associations_path = Path(stage["associations_path"]).resolve()
    local_mesh_manifest_path = Path(
        stage["local_mesh_manifest_path"]
    ).resolve()
    nuclear_data_manifest_path = Path(
        stage["nuclear_data_manifest_path"]
    ).resolve()
    source_manifest_path = Path(stage["source_manifest_path"]).resolve()
    transport_report_path = Path(stage["transport_report_path"]).resolve()
    required_inputs = (
        statepoint_path,
        model_manifest_path,
        associations_path,
        local_mesh_manifest_path,
        nuclear_data_manifest_path,
        source_manifest_path,
        transport_report_path,
    )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"postprocess inputs are absent: {missing}")
    model = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    association = json.loads(associations_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    transport_report = json.loads(
        transport_report_path.read_text(encoding="utf-8")
    )
    pairs = association["inventory"]["magnet_pairs"]
    pair_by_cell = {
        int(pair["winding_pack_volume_id"]): pair for pair in pairs
    }
    rate = float(model["physical_source_rate_per_s"])
    source_definition_sha256 = model["xml_files"]["settings.xml"]["sha256"]
    statepoint_sha256 = _sha256(statepoint_path)
    if (
        transport_report.get("status") != "PASS"
        or transport_report.get("statepoint", {}).get("sha256")
        != statepoint_sha256
    ):
        raise ValueError(
            "postprocess statepoint is not bound to a passing transport run"
        )
    if (
        source_manifest["source_mesh"]["sha256"]
        != model["source"]["mesh_sha256"]
    ):
        raise ValueError(
            "source manifest and OpenMC model use different source meshes"
        )
    coil_hashes = {
        pair["winding_pack"]["source_coil_provenance"]["coils_sha256"]
        for pair in pairs
    }
    if len(coil_hashes) != 1:
        raise ValueError(
            "magnet associations do not bind one common coil file"
        )
    provenance = {
        "parastell_commit": str(stage["parastell_commit"]),
        "raw_h5m_sha256": model["dagmc"]["raw_h5m_sha256"],
        "canonical_geometry_fingerprint": model["dagmc"][
            "canonical_geometry_fingerprint"
        ],
        "source_definition_sha256": source_definition_sha256,
        "source_mesh_sha256": model["source"]["mesh_sha256"],
        "nuclear_data_manifest_sha256": _sha256(nuclear_data_manifest_path),
        "statepoint_sha256": statepoint_sha256,
        "openmc_version": model["openmc_version"],
        "openmc_commit": transport_report["openmc_commit"],
        "vmec_sha256": source_manifest["vmec"]["sha256"],
        "coils_sha256": coil_hashes.pop(),
        "histories": int(model["histories"]),
        "batches": int(model["batches"]),
        "seeds": [int(model["seed"])],
    }

    neutron_structure = str(
        stage.get("configured_neutron_structure", "smoke-7")
    )
    photon_structure = str(
        stage.get("configured_photon_structure", "photon-master-v1")
    )
    fields = build_scalar_flux_fields_from_statepoint(
        statepoint_path,
        associations_path=associations_path,
        local_mesh_manifest_path=local_mesh_manifest_path,
        local_energy_structures={
            "neutron": neutron_structure,
            "photon": photon_structure,
        },
        minimum_realizations=int(stage.get("minimum_realizations", 10)),
    )
    scalar_path = Path(stage["scalar_flux_path"]).resolve()
    scalar_manifest = export_scalar_flux_fields(
        scalar_path,
        fields=fields,
        physical_source_rate_per_s=rate,
        provenance=provenance,
        material_manifest_sha256=model["material_manifest"]["sha256"],
    )
    ccfe_validation = validate_spectra_pka_ready_flux(
        scalar_path, field_name="neutron_ccfe_709"
    )

    cell_volumes = {
        cell: float(pair["winding_pack"]["volume_cm3"])
        for cell, pair in pair_by_cell.items()
    }
    cell_magnets = {
        cell: pair["magnet_id"] for cell, pair in pair_by_cell.items()
    }
    heating_path = Path(stage["heating_path"]).resolve()
    heating_manifest = export_magnet_heating(
        heating_path,
        statepoint_path=statepoint_path,
        tally_names=(
            "pstl_magnet_neutron_heating",
            "pstl_magnet_photon_heating",
        ),
        cell_volumes_cm3=cell_volumes,
        cell_magnet_ids=cell_magnets,
        physical_source_rate_per_s=rate,
        provenance=provenance,
    )

    reference_flux = _read_volume_flux_tally(
        statepoint_path, "pstl_magnet_neutron_ccfe_709_volume_flux"
    )
    reaction_path = Path(stage["reaction_production_path"]).resolve()
    export_magnet_reaction_production(
        statepoint_path,
        reaction_path,
        magnet_ids=[
            cell_magnets[int(cell)] for cell in reference_flux["cell_ids"]
        ],
        physical_source_rate_per_s=rate,
        transported_particles={
            "neutron": True,
            "photon": True,
            "electron": False,
            "positron": False,
        },
        provenance=provenance,
    )
    reaction_validation = validate_magnet_reaction_production(reaction_path)

    damage_gas_path = export_magnet_damage_gas(
        statepoint_path,
        stage["damage_gas_path"],
        tally_inventory=model["tallies"],
        cell_magnet_ids=cell_magnets,
        cell_volumes_cm3=cell_volumes,
        physical_source_rate_per_s=rate,
        provenance=provenance,
    )
    damage_gas_validation = validate_magnet_damage_gas(damage_gas_path)

    model_geometry_policy = model["dagmc"].get(
        "canonical_geometry_policy",
        {
            "coordinate_quantum_cm": 1.0e-6,
            "faceting_tolerances": {},
        },
    )
    producer = MagnetRadiationFieldProducer(
        model["dagmc"]["path"],
        selection=ProducerSelection(
            magnet_selection=stage.get("selection", "all")
        ),
        associations={
            int(key): value
            for key, value in association["associations"].items()
        },
        centreline_points_by_coil=association["centreline_points_by_coil"],
        **model_geometry_policy,
        expected_canonical_geometry_fingerprint=model["dagmc"][
            "canonical_geometry_fingerprint"
        ],
    )
    producer.discover()
    envelopes = producer.build_envelopes(
        spatial_bins=tuple(stage.get("boundary_spatial_bins", (4, 4)))
    )
    boundary_neutron_edges = stage.get("boundary_neutron_edges_eV")
    if boundary_neutron_edges is None:
        boundary_neutron_edges = get_structure(
            stage.get("boundary_neutron_structure", neutron_structure),
            particle="neutron",
        ).edges_eV
    boundary_photon_edges = stage.get("boundary_photon_edges_eV")
    if boundary_photon_edges is None:
        boundary_photon_edges = get_structure(
            stage.get("boundary_photon_structure", photon_structure),
            particle="photon",
        ).edges_eV
    centreline_frames = {
        envelope.envelope.envelope_id: producer.centreline_frames[
            envelope.envelope.magnet_component
        ]
        for envelope in envelopes
    }
    boundary_collection = export_openmc16_handoffs(
        stage["boundary_directory"],
        statepoint_path=statepoint_path,
        surface_source_paths=stage["surface_source_paths"],
        envelopes=envelopes,
        histories=int(model["histories"]),
        energy_edges_by_particle={
            "neutron": boundary_neutron_edges,
            "photon": boundary_photon_edges,
        },
        physical_source_rate_per_s=rate,
        parastell_commit=str(stage["parastell_commit"]),
        source_definition_sha256=source_definition_sha256,
        adaptive_patch_target_ess=stage.get("adaptive_patch_target_ess"),
        adaptive_patch_minimum_records=int(
            stage.get("adaptive_patch_minimum_records", 4)
        ),
        adaptive_patch_maximum_depth=int(
            stage.get("adaptive_patch_maximum_depth", 5)
        ),
        surface_source_max_particles=int(
            model["surface_source"]["max_particles"]
        ),
        surface_source_max_files=int(
            model["surface_source"]["max_source_files"]
        ),
        surface_source_sampling_applied=False,
        mpi_ranks=1,
        centreline_frames=centreline_frames,
    )
    empty_handoffs = sum(
        item["record_count"] == 0 for item in boundary_collection["handoffs"]
    )
    report = {
        "schema": "parastell.magnet_radiation_postprocess/v1.0.0",
        "status": "PASS",
        "statepoint_sha256": provenance["statepoint_sha256"],
        "physical_source_rate_per_s": rate,
        "products": {
            "scalar_flux": {
                "path": str(scalar_path),
                "sha256": _sha256(scalar_path),
                "fields": scalar_manifest["fields"],
                "spectra_pka_ccfe_709": ccfe_validation,
            },
            "heating": {
                "path": str(heating_path),
                "sha256": _sha256(heating_path),
                "manifest": heating_manifest,
            },
            "reaction_production": {
                "path": str(reaction_path),
                "sha256": _sha256(reaction_path),
                "validation": reaction_validation,
            },
            "damage_and_gas_production": {
                "path": str(damage_gas_path),
                "sha256": _sha256(damage_gas_path),
                "validation": damage_gas_validation,
                "is_dpa": False,
                "is_appm": False,
            },
            "boundary_phase_space": {
                **boundary_collection,
                "empty_handoff_count": empty_handoffs,
                "empty_is_physical_zero": False,
            },
        },
        "provenance": provenance,
    }
    return _write(stage["postprocess_report_path"], report)


def render_diagnostics(stage: Mapping[str, Any], output_root: Path):
    from .magnet_diagnostics import render_response_comparison
    from .magnet_diagnostics import write_figure_manifest
    from .material_manifest import deterministic_component_colors

    unbiased = json.loads(
        Path(stage["unbiased_responses_path"]).read_text(encoding="utf-8")
    )
    weighted = json.loads(
        Path(stage["weight_window_responses_path"]).read_text(encoding="utf-8")
    )
    unbiased_rows = (
        unbiased.get("responses", [])
        if isinstance(unbiased, Mapping)
        else unbiased
    )
    weighted_rows = (
        weighted.get("responses", [])
        if isinstance(weighted, Mapping)
        else weighted
    )
    if not isinstance(unbiased_rows, list) or not isinstance(
        weighted_rows, list
    ):
        raise ValueError("diagnostic response payloads must contain lists")
    left = {str(item["response_id"]): item for item in unbiased_rows}
    right = {str(item["response_id"]): item for item in weighted_rows}
    if set(left) != set(right):
        raise ValueError("diagnostic response inventories do not match")
    identifiers = sorted(left)
    colors = deterministic_component_colors(
        stage.get("component_names", ()), stage.get("component_colors", {})
    )
    figure = render_response_comparison(
        stage["comparison_figure_path"],
        response_ids=identifiers,
        unbiased=[left[name]["mean"] for name in identifiers],
        weight_window=[right[name]["mean"] for name in identifiers],
        component_names=stage.get("component_names", ()),
        explicit_colors=colors,
    )
    return write_figure_manifest(
        stage["figure_manifest_path"],
        input_paths=(
            stage["unbiased_responses_path"],
            stage["weight_window_responses_path"],
        ),
        figures=(figure,),
        component_colors=colors,
        rendering_parameters={
            "dpi": 160,
            "random_colors": False,
            "status": (
                "RENDERED"
                if identifiers
                else "NO_COMPARABLE_WEIGHT_WINDOW_RESPONSES"
            ),
            "unbiased_payload_status": (
                unbiased.get("status")
                if isinstance(unbiased, Mapping)
                else None
            ),
            "weight_window_payload_status": (
                weighted.get("status")
                if isinstance(weighted, Mapping)
                else None
            ),
        },
    )


def export_bundle(stage: Mapping[str, Any], output_root: Path):
    from .magnet_radiation_field_bundle import write_radiation_field_bundle

    if "postprocess_report_path" in stage:
        postprocess_report = json.loads(
            Path(stage["postprocess_report_path"]).read_text(encoding="utf-8")
        )
        transport_report = json.loads(
            Path(stage["transport_report_path"]).read_text(encoding="utf-8")
        )
        association = json.loads(
            Path(stage["associations_path"]).read_text(encoding="utf-8")
        )
        material = json.loads(
            Path(stage["material_manifest_path"]).read_text(encoding="utf-8")
        )
        nuclear = json.loads(
            Path(stage["nuclear_data_manifest_path"]).read_text(
                encoding="utf-8"
            )
        )
        products_report = postprocess_report["products"]
        products = [
            {
                "kind": "volume_scalar_flux",
                "magnet_id": "all-selected-magnets",
                "path": products_report["scalar_flux"]["path"],
                "quantity": "volume_scalar_flux",
                "units": "particles/cm2/s",
                "normalization": "physical_source_rate",
            },
            {
                "kind": "heating",
                "magnet_id": "all-selected-magnets",
                "path": products_report["heating"]["path"],
                "quantity": "heating",
                "units": "W/cm3",
                "normalization": "physical_source_rate",
            },
            {
                "kind": "reaction_production",
                "magnet_id": "all-selected-magnets",
                "path": products_report["reaction_production"]["path"],
                "quantity": "reaction_and_particle_production",
                "units": "events/s",
                "normalization": "physical_source_rate",
            },
            {
                "kind": "damage_and_gas_production",
                "magnet_id": "all-selected-magnets",
                "path": products_report["damage_and_gas_production"]["path"],
                "quantity": "damage_energy_and_gas_production",
                "units": "mixed_explicit_in_product",
                "normalization": "physical_source_rate",
            },
        ]
        products.extend(
            {
                "kind": "boundary_phase_space",
                "magnet_id": item["magnet_component"],
                "path": item["path"],
                "quantity": "partial_crossing_current",
                "units": "crossings/source",
                "normalization": "per_source_history",
                "record_count": item["record_count"],
            }
            for item in products_report["boundary_phase_space"]["handoffs"]
        )
        provenance = {
            **postprocess_report["provenance"],
            "parastell_commit": str(stage["parastell_commit"]),
            "transport_report_sha256": _sha256(stage["transport_report_path"]),
            "postprocess_report_sha256": _sha256(
                stage["postprocess_report_path"]
            ),
        }
        geometry = {
            "raw_h5m_sha256": provenance["raw_h5m_sha256"],
            "canonical_geometry_fingerprint": provenance[
                "canonical_geometry_fingerprint"
            ],
        }
        source = {
            "physical_source_rate_per_s": postprocess_report[
                "physical_source_rate_per_s"
            ],
            "source_definition_sha256": provenance["source_definition_sha256"],
            "source_mesh_sha256": provenance["source_mesh_sha256"],
        }
        nuclear_data = {
            "status": nuclear["status"],
            "manifest_sha256": provenance["nuclear_data_manifest_sha256"],
            "approved_library": nuclear.get("approved_library"),
            "evaluation_release": nuclear.get("evaluation_release"),
            "temperature_policy": nuclear.get("temperature_policy"),
        }
        materials = {
            "resolved_manifest_sha256": material["resolved_manifest_sha256"],
            "file_sha256": _sha256(stage["material_manifest_path"]),
            "materials": material["materials"],
        }
        magnet_inventory = [
            {
                "magnet_id": pair["magnet_id"],
                "coil_id": pair["coil_id"],
                "winding_pack_volume_id": pair["winding_pack_volume_id"],
                "casing_volume_id": pair["casing_volume_id"],
                "winding_pack_volume_cm3": pair["winding_pack"]["volume_cm3"],
            }
            for pair in association["inventory"]["magnet_pairs"]
        ]
        verification = {
            "transport_status": transport_report["status"],
            "lost_particle_files": transport_report["lost_particle_files"],
            "dagmc_navigation_failures": transport_report[
                "dagmc_navigation_failures"
            ],
            "secondary_photon_evidence": transport_report[
                "secondary_photon_evidence"
            ],
            "spectra_pka_ccfe_709": products_report["scalar_flux"][
                "spectra_pka_ccfe_709"
            ],
            "empty_boundary_handoffs_are_physical_zero": False,
        }
    else:
        provenance = stage["provenance"]
        geometry = stage["geometry"]
        source = stage["source"]
        nuclear_data = stage["nuclear_data"]
        materials = stage["materials"]
        magnet_inventory = stage["magnet_inventory"]
        products = stage["products"]
        verification = stage["verification"]
    manifest = write_radiation_field_bundle(
        stage["bundle_directory"],
        provenance=provenance,
        geometry=geometry,
        source=source,
        nuclear_data=nuclear_data,
        materials=materials,
        magnet_inventory=magnet_inventory,
        products=products,
        verification=verification,
    )
    receipt = stage.get("bundle_receipt_path")
    return _write(receipt, manifest) if receipt else manifest
