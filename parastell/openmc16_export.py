"""Export OpenMC 0.16 closed-envelope records without tally conditioning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .dagmc_envelope import DagmcEnvelope
from .dagmc_envelope import extract_closed_envelope
from .magnet_boundary_envelope import build_correlated_bank
from .magnet_boundary_envelope import assign_adaptive_surface_patches
from .magnet_boundary_envelope import classify_crossing_bank
from .magnet_boundary_envelope import write_handoff
from .openmc16 import PDG_PARTICLES


OPENMC16_SOURCE_FORMAT = (18, 2)
OPENMC16_VERSION = (0, 16, 0)
OPENMC16_SOURCE_FIELDS = frozenset(
    {"r", "u", "E", "time", "wgt", "delayed_group", "surf_id", "particle"}
)
_WRITER_PATTERN = re.compile(
    r"Creating source file\s+(?P<name>\S+)\s+with\s+"
    r"(?P<count>\d+)\s+particles"
)


@dataclass(frozen=True)
class StrictSurfaceRunArtifacts:
    """Create-only paths consumed by the strict OpenMC 0.16 run audit.

    Counts, histories, requested surfaces, writer capacity, and tally values
    are deliberately absent. They are parsed from the bound artifacts.
    """

    dagmc_path: str | Path
    model_xml_path: str | Path
    statepoint_path: str | Path
    terminal_log_path: str | Path
    surface_source_paths: Sequence[str | Path]
    accepted_magnet_inventory_path: str | Path
    root_acceptance_receipt_path: str | Path
    expected_root_acceptance_receipt_sha256: str


def _resolved_file(path: str | Path, label: str) -> Path:
    result = Path(path).resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _localization_topology_binding(
    dagmc_path: Path, envelopes: Sequence[DagmcEnvelope]
) -> dict[str, Any]:
    """Hash canonical facet catalogs derived directly from the H5M."""
    dagmc_hash = _hash(dagmc_path)
    rows = []
    for item in envelopes:
        envelope_manifest = item.envelope.to_dict()
        facet_catalog = item.facet_metadata()
        rows.append(
            {
                "envelope_id": item.envelope.envelope_id,
                "magnet_id": item.envelope.magnet_component,
                "dagmc_volume_id": item.envelope.dagmc_volume_id,
                "surface_ids": list(item.envelope.surface_ids),
                "canonical_geometry_fingerprint": item.envelope.metadata[
                    "canonical_geometry_fingerprint"
                ],
                "envelope_manifest_sha256": _canonical_json_sha256(
                    envelope_manifest
                ),
                "canonical_facet_catalog_sha256": _canonical_json_sha256(
                    facet_catalog
                ),
                "canonical_facet_count": int(
                    len(facet_catalog["canonical_facet_id"])
                ),
                "normal_source": "DAGMC forward/reverse topology",
            }
        )
    binding = {
        "schema": "parastell.localization_topology_binding/v1.0.0",
        "dagmc_path": str(dagmc_path),
        "dagmc_sha256": dagmc_hash,
        "parser": "pydagmc.Model via extract_closed_envelope",
        "envelopes": rows,
    }
    binding["manifest_sha256"] = _canonical_json_sha256(binding)
    return binding


def _canonical_material_tag(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("mat:"):
        text = text[4:]
    return text.replace("-", "_").replace(" ", "_")


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and not (set(text) - set("0123456789abcdef"))


def _discover_h5m_material_group(
    dagmc_path: Path, material_tags: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    """Discover all H5M volumes in an accepted material group."""
    import pydagmc

    accepted = {_canonical_material_tag(value) for value in material_tags}
    if not accepted or "" in accepted:
        raise ValueError("magnet material group must be nonempty")
    model = pydagmc.Model(str(dagmc_path))
    rows = []
    for volume_id, volume in sorted(model.volumes_by_id.items()):
        material = str(getattr(volume, "material", "") or "")
        if _canonical_material_tag(material) not in accepted:
            continue
        surface_ids = tuple(
            sorted(int(surface.id) for surface in volume.surfaces)
        )
        if not surface_ids:
            raise ValueError(
                f"magnet material volume {volume_id} has no boundary surfaces"
            )
        rows.append(
            {
                "dagmc_volume_id": int(volume_id),
                "material_tag": material,
                "surface_ids": list(surface_ids),
            }
        )
    if not rows:
        raise ValueError(
            "H5M has no volume in the accepted magnet material group"
        )
    return tuple(rows)


def _verify_root_acceptance_receipt(
    receipt_path: Path,
    expected_receipt_sha256: str,
    inventory_path: Path,
    dagmc_path: Path,
) -> dict[str, Any]:
    """Verify the external frozen-control anchor before inventory semantics."""
    expected_receipt_sha256 = str(expected_receipt_sha256).lower()
    if (
        not _valid_sha256(expected_receipt_sha256)
        or _hash(receipt_path) != expected_receipt_sha256
    ):
        raise ValueError(
            "root acceptance receipt does not match frozen run-control SHA"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("root acceptance receipt is not valid JSON") from exc
    required = {
        "schema",
        "acceptance_authority",
        "accepted_magnet_inventory_sha256",
        "dagmc_sha256",
        "canonical_geometry_fingerprint",
        "accepted_canonical_material_tags",
    }
    missing = required - set(receipt)
    if missing:
        raise ValueError(f"root acceptance receipt omits {sorted(missing)}")
    if (
        receipt["schema"] != "parastell.root_accepted_magnet_inventory/v1.0.0"
        or receipt["acceptance_authority"] != "ROOT_GEOMETRY_GATE_ACCEPTED"
    ):
        raise ValueError("root acceptance receipt has no accepted authority")
    inventory_hash = str(receipt["accepted_magnet_inventory_sha256"]).lower()
    if (
        not _valid_sha256(inventory_hash)
        or _hash(inventory_path) != inventory_hash
    ):
        raise ValueError(
            "accepted magnet inventory is not bound by root receipt"
        )
    dagmc_hash = str(receipt["dagmc_sha256"]).lower()
    if not _valid_sha256(dagmc_hash) or _hash(dagmc_path) != dagmc_hash:
        raise ValueError("root acceptance receipt H5M hash mismatch")
    raw_tags = receipt["accepted_canonical_material_tags"]
    if not isinstance(raw_tags, list):
        raise ValueError(
            "root acceptance receipt material tags must be a canonical list"
        )
    canonical_tags = tuple(
        sorted(_canonical_material_tag(value) for value in raw_tags)
    )
    if (
        not canonical_tags
        or "" in canonical_tags
        or len(canonical_tags) != len(set(canonical_tags))
    ):
        raise ValueError(
            "root acceptance receipt material tags are not canonical/unique"
        )
    if tuple(raw_tags) != canonical_tags:
        raise ValueError(
            "root acceptance receipt material tags must already be canonical"
        )
    fingerprint = str(receipt["canonical_geometry_fingerprint"]).lower()
    if not _valid_sha256(fingerprint):
        raise ValueError(
            "root acceptance receipt canonical fingerprint is invalid"
        )
    result = {
        "path": str(receipt_path),
        "sha256": _hash(receipt_path),
        "expected_sha256_from_frozen_run_control": expected_receipt_sha256,
        "schema": receipt["schema"],
        "acceptance_authority": receipt["acceptance_authority"],
        "accepted_magnet_inventory_sha256": inventory_hash,
        "dagmc_sha256": dagmc_hash,
        "canonical_geometry_fingerprint": fingerprint,
        "accepted_canonical_material_tags": list(canonical_tags),
    }
    result["verified_acceptance_sha256"] = _canonical_json_sha256(result)
    return result


def _verify_accepted_magnet_inventory(
    inventory_path: Path,
    root_acceptance: Mapping[str, Any],
    dagmc_path: Path,
    envelopes: Sequence[DagmcEnvelope],
    envelope_requests: Sequence[Mapping[str, Any]],
    localization_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind root-selected magnet semantics to independently parsed H5M rows."""
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "accepted magnet inventory is not valid JSON"
        ) from exc
    required = {
        "schema",
        "geometry_gate_status",
        "dagmc_sha256",
        "canonical_geometry_fingerprint",
        "magnet_material_tags",
        "components",
    }
    missing = required - set(inventory)
    if missing:
        raise ValueError(f"accepted magnet inventory omits {sorted(missing)}")
    if (
        inventory["schema"]
        != "parastell.accepted_magnet_component_inventory/v1.0.0"
        or inventory["geometry_gate_status"] != "PASS"
    ):
        raise ValueError(
            "magnet inventory is not a root-accepted geometry gate"
        )
    dagmc_hash = _hash(dagmc_path)
    if str(inventory["dagmc_sha256"]).lower() != dagmc_hash:
        raise ValueError("magnet inventory H5M hash mismatch")
    if dagmc_hash != root_acceptance["dagmc_sha256"]:
        raise ValueError("magnet inventory H5M is not root-authorized")
    fingerprints = {
        str(item.envelope.metadata["canonical_geometry_fingerprint"]).lower()
        for item in envelopes
    }
    inventory_fingerprint = str(
        inventory["canonical_geometry_fingerprint"]
    ).lower()
    if fingerprints != {inventory_fingerprint}:
        raise ValueError("magnet inventory canonical fingerprint mismatch")
    if (
        inventory_fingerprint
        != root_acceptance["canonical_geometry_fingerprint"]
    ):
        raise ValueError(
            "magnet inventory canonical fingerprint is not root-authorized"
        )
    material_tags = tuple(
        str(item) for item in inventory["magnet_material_tags"]
    )
    canonical_material_tags = tuple(
        _canonical_material_tag(item) for item in material_tags
    )
    if (
        not material_tags
        or "" in canonical_material_tags
        or len(canonical_material_tags) != len(set(canonical_material_tags))
    ):
        raise ValueError(
            "magnet inventory material tags must be unique/nonempty"
        )
    if (
        sorted(canonical_material_tags)
        != root_acceptance["accepted_canonical_material_tags"]
    ):
        raise ValueError(
            "magnet inventory material tags are not root-authorized"
        )
    discovered = _discover_h5m_material_group(dagmc_path, material_tags)
    discovered_by_id = {
        int(item["dagmc_volume_id"]): item for item in discovered
    }
    components = list(inventory["components"])
    if not components:
        raise ValueError("accepted magnet inventory has no components")
    component_by_id = {}
    semantic_ids = set()
    component_ids = set()
    for component in components:
        component_required = {
            "magnet_id",
            "component_id",
            "dagmc_volume_id",
            "material_tag",
            "surface_ids",
        }
        component_missing = component_required - set(component)
        if component_missing:
            raise ValueError(
                f"magnet inventory component omits {sorted(component_missing)}"
            )
        volume_id = int(component["dagmc_volume_id"])
        magnet_id = str(component["magnet_id"]).strip()
        component_id = str(component["component_id"]).strip()
        if (
            volume_id <= 0
            or not magnet_id
            or not component_id
            or volume_id in component_by_id
            or magnet_id in semantic_ids
            or component_id in component_ids
        ):
            raise ValueError(
                "magnet inventory component identities are invalid"
            )
        semantic_ids.add(magnet_id)
        component_ids.add(component_id)
        has_source_cad = "source_cad" in component
        has_source_geometry = "source_geometry" in component
        if has_source_cad == has_source_geometry:
            raise ValueError(
                "magnet inventory component must declare exactly one of "
                "source_cad or source_geometry"
            )
        source_geometry = component[
            "source_cad" if has_source_cad else "source_geometry"
        ]
        if not isinstance(source_geometry, Mapping):
            raise ValueError("magnet source geometry must be a mapping")
        source_kind = str(
            source_geometry.get("kind", "source_cad" if has_source_cad else "")
        ).strip()
        if source_kind not in {
            "source_cad",
            "parametric_reference_manifest",
        }:
            raise ValueError("unsupported magnet source geometry kind")
        source_path = _resolved_file(
            source_geometry.get("path", ""), "magnet source geometry"
        )
        source_hash = str(source_geometry.get("sha256", "")).lower()
        if not _valid_sha256(source_hash) or _hash(source_path) != source_hash:
            label = "source CAD" if has_source_cad else "source geometry"
            raise ValueError(f"magnet inventory {label} hash mismatch")
        component_by_id[volume_id] = {
            "magnet_id": magnet_id,
            "component_id": component_id,
            "dagmc_volume_id": volume_id,
            "material_tag": str(component["material_tag"]),
            "surface_ids": sorted(
                int(value) for value in component["surface_ids"]
            ),
            "source_geometry": {
                "kind": source_kind,
                "path": str(source_path),
                "sha256": source_hash,
            },
        }
    if set(component_by_id) != set(discovered_by_id):
        raise ValueError(
            "accepted inventory does not enumerate every H5M magnet-material volume"
        )
    for volume_id, component in component_by_id.items():
        actual = discovered_by_id[volume_id]
        if _canonical_material_tag(
            component["material_tag"]
        ) != _canonical_material_tag(actual["material_tag"]) or component[
            "surface_ids"
        ] != sorted(
            actual["surface_ids"]
        ):
            raise ValueError(
                f"magnet inventory topology/material mismatch for volume {volume_id}"
            )
    request_by_id = {
        int(item["volume_id"]): str(item["magnet_id"]).strip()
        for item in envelope_requests
    }
    if set(request_by_id) != set(component_by_id):
        raise ValueError(
            "envelope requests do not equal the complete magnet material group"
        )
    envelope_by_id = {
        item.envelope.dagmc_volume_id: item for item in envelopes
    }
    if set(envelope_by_id) != set(component_by_id):
        raise ValueError("extracted envelopes do not equal accepted inventory")
    topology_by_id = {
        int(item["dagmc_volume_id"]): item
        for item in localization_binding["envelopes"]
    }
    for volume_id, component in component_by_id.items():
        envelope = envelope_by_id[volume_id]
        if (
            request_by_id[volume_id] != component["magnet_id"]
            or envelope.envelope.magnet_component != component["magnet_id"]
            or sorted(envelope.envelope.surface_ids)
            != component["surface_ids"]
            or volume_id not in topology_by_id
        ):
            raise ValueError(
                f"semantic magnet/envelope mismatch for volume {volume_id}"
            )
    result = {
        "path": str(inventory_path),
        "sha256": _hash(inventory_path),
        "accepted_sha256_from_root_receipt": root_acceptance[
            "accepted_magnet_inventory_sha256"
        ],
        "root_acceptance_receipt_sha256": root_acceptance["sha256"],
        "verified_root_acceptance_sha256": root_acceptance[
            "verified_acceptance_sha256"
        ],
        "schema": inventory["schema"],
        "geometry_gate_status": inventory["geometry_gate_status"],
        "dagmc_sha256": dagmc_hash,
        "canonical_geometry_fingerprint": next(iter(fingerprints)),
        "magnet_material_tags": list(material_tags),
        "components": [
            component_by_id[key] for key in sorted(component_by_id)
        ],
        "all_material_group_volumes_selected": True,
    }
    result["verified_inventory_sha256"] = _canonical_json_sha256(result)
    return result


def _positive_int_text(parent, name: str) -> int:
    element = parent.find(name)
    if element is None or element.text is None:
        raise ValueError(f"OpenMC settings omit {name}")
    try:
        value = int(element.text.strip())
    except ValueError as exc:
        raise ValueError(f"OpenMC settings {name} is not an integer") from exc
    if value <= 0:
        raise ValueError(f"OpenMC settings {name} must be positive")
    return value


def _parse_int_bins(element, label: str) -> tuple[int, ...]:
    if element is None or element.text is None:
        raise ValueError(f"OpenMC model omits {label}")
    try:
        values = tuple(int(item) for item in element.text.split())
    except ValueError as exc:
        raise ValueError(f"OpenMC {label} contains a noninteger") from exc
    if not values or len(values) != len(set(values)) or min(values) <= 0:
        raise ValueError(f"OpenMC {label} must be unique positive IDs")
    return values


def _parse_model_xml(model_path: Path, dagmc_path: Path) -> dict[str, Any]:
    """Parse the exact monolithic model used for transport."""
    try:
        root = ET.parse(model_path).getroot()
    except ET.ParseError as exc:
        raise ValueError("OpenMC model XML is malformed") from exc
    if root.tag != "model":
        raise ValueError("strict surface audit requires monolithic model.xml")
    settings = root.find("settings")
    geometry = root.find("geometry")
    tallies = root.find("tallies")
    if settings is None or geometry is None or tallies is None:
        raise ValueError(
            "model.xml must contain settings, geometry, and tallies"
        )
    if (settings.findtext("run_mode") or "").strip() != "fixed source":
        raise ValueError("surface-source audit requires fixed source mode")
    particles = _positive_int_text(settings, "particles")
    batches = _positive_int_text(settings, "batches")
    seed = _positive_int_text(settings, "seed")
    writer = settings.find("surf_source_write")
    if writer is None:
        raise ValueError("model.xml omits surf_source_write")
    requested_surface_ids = _parse_int_bins(
        writer.find("surface_ids"), "surf_source_write surface_ids"
    )
    max_particles = _positive_int_text(writer, "max_particles")
    max_source_files_element = writer.find("max_source_files")
    if max_source_files_element is None:
        # OpenMC 0.16 serializes this element only when the configured value
        # differs from its documented default of one source file.
        max_source_files = 1
        max_source_files_source = "openmc_0.16_default"
    else:
        max_source_files = _positive_int_text(writer, "max_source_files")
        max_source_files_source = "model_xml"

    dagmc_universes = geometry.findall(".//dagmc_universe")
    if len(dagmc_universes) != 1:
        raise ValueError("strict audit requires exactly one DAGMC universe")
    declared = dagmc_universes[0].get("filename")
    if not declared:
        raise ValueError("DAGMC universe omits its filename")
    declared_path = Path(declared)
    if not declared_path.is_absolute():
        declared_path = model_path.parent / declared_path
    declared_path = declared_path.resolve()
    if not declared_path.is_file() or _hash(declared_path) != _hash(
        dagmc_path
    ):
        raise ValueError(
            "model.xml DAGMC filename does not resolve to the audited H5M"
        )

    filters: dict[int, dict[str, Any]] = {}
    for element in tallies.findall("filter"):
        filter_id = int(element.get("id", "0"))
        filter_type = str(element.get("type", "")).strip().lower()
        bins_text = element.findtext("bins")
        if filter_id <= 0 or not filter_type or bins_text is None:
            raise ValueError("model.xml contains a malformed tally filter")
        if filter_id in filters:
            raise ValueError("model.xml contains duplicate tally filter IDs")
        filters[filter_id] = {"type": filter_type, "bins": bins_text.split()}
    tally_rows = {}
    for element in tallies.findall("tally"):
        name = str(element.get("name", "")).strip()
        if not name:
            raise ValueError("model.xml contains an unnamed tally")
        if name in tally_rows:
            raise ValueError(
                f"model.xml contains duplicate tally name {name!r}"
            )
        tally_rows[name] = {
            "filter_ids": tuple(
                int(item)
                for item in (element.findtext("filters") or "").split()
            ),
            "scores": tuple((element.findtext("scores") or "").split()),
        }
    return {
        "path": str(model_path),
        "sha256": _hash(model_path),
        "particles_per_batch": particles,
        "batches": batches,
        "source_histories": particles * batches,
        "seed": seed,
        "requested_surface_ids": requested_surface_ids,
        "max_particles_per_process": max_particles,
        "max_source_files": max_source_files,
        "max_source_files_source": max_source_files_source,
        "dagmc_declared_filename": declared,
        "dagmc_resolved_path": str(declared_path),
        "filters": filters,
        "tallies": tally_rows,
    }


def _hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vectors(values, field: str) -> np.ndarray:
    return np.column_stack([values[field][axis] for axis in "xyz"])


def _select_envelope_records(records, surface_ids):
    record_surface_ids = np.asarray(records["surf_id"]).reshape(-1)
    keep = np.isin(
        record_surface_ids, tuple(int(item) for item in surface_ids)
    )
    selected = records[keep]
    return selected, record_surface_ids[keep], keep


def _decode_hdf5_text(value) -> str:
    return (
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
    )


def _statepoint_header(
    statepoint_path: Path, model: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate exact OpenMC 0.16 run identity before reading scores."""
    with h5py.File(statepoint_path, "r") as statepoint:
        filetype = _decode_hdf5_text(statepoint.attrs.get("filetype", ""))
        version = tuple(
            int(value)
            for value in np.asarray(
                statepoint.attrs.get("openmc_version", ()), dtype=int
            ).reshape(-1)
        )
        required = {
            "run_mode",
            "n_particles",
            "n_batches",
            "current_batch",
            "seed",
            "tallies",
        }
        missing = required - set(statepoint)
        if missing:
            raise ValueError(f"statepoint omits {sorted(missing)}")
        values = {
            "run_mode": _decode_hdf5_text(statepoint["run_mode"][()]),
            "particles_per_batch": int(statepoint["n_particles"][()]),
            "batches": int(statepoint["n_batches"][()]),
            "current_batch": int(statepoint["current_batch"][()]),
            "seed": int(statepoint["seed"][()]),
        }
    if filetype != "statepoint" or version != OPENMC16_VERSION:
        raise ValueError("statepoint is not an exact OpenMC 0.16.0 statepoint")
    expected = {
        "run_mode": "fixed source",
        "particles_per_batch": model["particles_per_batch"],
        "batches": model["batches"],
        "current_batch": model["batches"],
        "seed": model["seed"],
    }
    if values != expected:
        raise ValueError("statepoint run identity disagrees with model.xml")
    return {
        "path": str(statepoint_path),
        "sha256": _hash(statepoint_path),
        "openmc_version": "0.16.0",
        **values,
    }


def _filter_bins_for_model(
    model_filter: Mapping[str, Any], statepoint_filter, filter_type: str
) -> None:
    """Require the statepoint filter to reproduce the model definition."""
    state_type = _decode_hdf5_text(statepoint_filter["type"][()]).lower()
    if state_type != filter_type or model_filter["type"] != filter_type:
        raise ValueError(f"statepoint/model {filter_type} filter mismatch")
    model_bins = model_filter["bins"]
    state_bins = statepoint_filter["bins"][:]
    if filter_type in {"surface"}:
        equal = np.array_equal(
            np.asarray([int(item) for item in model_bins], dtype=int),
            np.asarray(state_bins, dtype=int),
        )
    elif filter_type in {"musurface", "energy"}:
        equal = np.array_equal(
            np.asarray([float(item) for item in model_bins], dtype=float),
            np.asarray(state_bins, dtype=float),
        )
    else:
        equal = tuple(model_bins) == tuple(
            _decode_hdf5_text(item) for item in state_bins
        )
    if not equal:
        raise ValueError(f"statepoint/model {filter_type} bins disagree")


def _directional_rows_from_statepoint(
    statepoint_path: Path,
    model: Mapping[str, Any],
    envelopes: Sequence[DagmcEnvelope],
    required_particles: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Parse every directional-current bin and orient it from H5M topology."""
    surfaces = {
        int(surface.surface_id): surface
        for envelope in envelopes
        for surface in envelope.envelope.surfaces
    }
    if len(surfaces) != sum(len(item.envelope.surfaces) for item in envelopes):
        raise ValueError("selected envelopes contain duplicate DAGMC surfaces")
    rows = []
    energy_edges = {}
    with h5py.File(statepoint_path, "r") as statepoint:
        tallies = statepoint["tallies"]
        for particle in required_particles:
            tally_name = f"pstl_envelope_{particle}_directional_current"
            model_tally = model["tallies"].get(tally_name)
            if model_tally is None or model_tally["scores"] != ("current",):
                raise ValueError(
                    f"model.xml omits exact current tally {tally_name!r}"
                )
            candidates = [
                value
                for name, value in tallies.items()
                if name.startswith("tally ")
                and _decode_hdf5_text(value["name"][()]) == tally_name
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"statepoint must contain exactly one {tally_name!r}"
                )
            tally = candidates[0]
            scores = tuple(
                _decode_hdf5_text(value) for value in tally["score_bins"][:]
            )
            nuclides = tuple(
                _decode_hdf5_text(value) for value in tally["nuclides"][:]
            )
            if scores != ("current",) or nuclides != ("total",):
                raise ValueError(
                    f"{tally_name} has unexpected score/nuclide bins"
                )
            filter_ids = tuple(int(value) for value in tally["filters"][:])
            if filter_ids != tuple(model_tally["filter_ids"]):
                raise ValueError(
                    f"{tally_name} filter IDs disagree with model.xml"
                )
            filters = [
                tallies[f"filters/filter {value}"] for value in filter_ids
            ]
            filter_types = [
                _decode_hdf5_text(item["type"][()]).lower() for item in filters
            ]
            if filter_types != ["surface", "musurface", "particle", "energy"]:
                raise ValueError(
                    f"{tally_name} filters must be surface/musurface/particle/energy"
                )
            for filter_id, filter_type, state_filter in zip(
                filter_ids, filter_types, filters
            ):
                if filter_id not in model["filters"]:
                    raise ValueError(f"model.xml omits filter {filter_id}")
                _filter_bins_for_model(
                    model["filters"][filter_id], state_filter, filter_type
                )
            scored_particles = tuple(
                _decode_hdf5_text(value) for value in filters[2]["bins"][:]
            )
            if scored_particles != (particle,):
                raise ValueError(f"{tally_name} particle filter is not exact")
            surface_ids = np.asarray(filters[0]["bins"][:], dtype=int)
            if set(surface_ids.tolist()) != set(surfaces):
                raise ValueError(
                    f"{tally_name} surfaces disagree with closed H5M envelopes"
                )
            mu_edges = np.asarray(filters[1]["bins"][:], dtype=float)
            if not np.array_equal(mu_edges, [-1.0, 0.0, 1.0]):
                raise ValueError(f"{tally_name} must split mu exactly at zero")
            edges = np.asarray(filters[3]["bins"][:], dtype=float)
            if (
                edges.ndim != 1
                or len(edges) < 2
                or np.any(~np.isfinite(edges))
                or np.any(np.diff(edges) <= 0.0)
            ):
                raise ValueError(f"{tally_name} energy bins are malformed")
            energy_edges[particle] = edges
            dimensions = tuple(int(item["n_bins"][()]) for item in filters)
            result = np.asarray(tally["results"])
            means = result[..., 0].reshape(dimensions + (-1,))
            realizations = int(tally["n_realizations"][()])
            if realizations != int(model["batches"]):
                raise ValueError(
                    f"{tally_name} realizations disagree with batches"
                )
            means = means / realizations
            for surface_index, surface_id in enumerate(surface_ids):
                normal_sign = surfaces[int(surface_id)].openmc_normal_sign
                for native_mu_index, native_sign in ((0, -1), (1, 1)):
                    canonical_direction = (
                        "outgoing"
                        if normal_sign * native_sign > 0
                        else "incoming"
                    )
                    for energy_index in range(len(edges) - 1):
                        selection = (
                            surface_index,
                            native_mu_index,
                            0,
                            energy_index,
                        )
                        value = float(abs(np.asarray(means[selection]).sum()))
                        rows.append(
                            {
                                "surface_id": int(surface_id),
                                "particle": particle,
                                "energy_bin": energy_index,
                                "energy_low_eV": float(edges[energy_index]),
                                "energy_high_eV": float(
                                    edges[energy_index + 1]
                                ),
                                "canonical_direction": canonical_direction,
                                "openmc_native_mu_sign": native_sign,
                                "openmc_normal_sign": int(normal_sign),
                                "tally_current_per_source": value,
                            }
                        )
    return rows, energy_edges


def _read_surface_banks(
    paths: Sequence[Path],
) -> tuple[np.ndarray, list[dict]]:
    arrays = []
    manifest = []
    dtype = None
    for path in paths:
        with h5py.File(path, "r") as source:
            filetype = _decode_hdf5_text(source.attrs.get("filetype", ""))
            version = tuple(
                int(value)
                for value in np.asarray(
                    source.attrs.get("version", ())
                ).reshape(-1)
            )
            if filetype != "source" or version != OPENMC16_SOURCE_FORMAT:
                raise ValueError(
                    f"{path} is not an exact OpenMC 0.16 source file"
                )
            if "source_bank" not in source:
                raise ValueError(f"{path} omits source_bank")
            dataset = source["source_bank"]
            fields = frozenset(dataset.dtype.names or ())
            if not OPENMC16_SOURCE_FIELDS.issubset(fields):
                raise ValueError(
                    f"{path} source_bank omits {sorted(OPENMC16_SOURCE_FIELDS - fields)}"
                )
            if dtype is not None and dataset.dtype != dtype:
                raise ValueError(
                    "surface-source files use inconsistent dtypes"
                )
            dtype = dataset.dtype
            array = dataset[:]
        arrays.append(array)
        manifest.append(
            {
                "path": str(path),
                "sha256": _hash(path),
                "record_count": int(len(array)),
                "openmc_source_format": list(OPENMC16_SOURCE_FORMAT),
                "dtype_fields": list(dtype.names),
            }
        )
    if not arrays:
        raise ValueError(
            "strict surface audit requires at least one source file"
        )
    return np.concatenate(arrays), manifest


def _parse_terminal_log(
    path: Path,
    source_files: Sequence[Mapping[str, Any]],
    dagmc_declared_filename: str,
    statepoint_path: Path,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="strict")
    if not re.search(r"Version\s*\|\s*0\.16\.0\b", text):
        raise ValueError("terminal log does not identify OpenMC 0.16.0")
    if (
        "FIXED SOURCE TRANSPORT SIMULATION" not in text
        or "RESULTS" not in text
    ):
        raise ValueError(
            "terminal log does not prove a completed fixed-source run"
        )
    if re.search(r"\bERROR\b|lost particle|navigation error", text, re.I):
        raise ValueError("terminal log contains a fatal/navigation diagnostic")
    loaded = re.findall(r"^Loading file\s+(.+?\.h5m)\s*$", text, re.M)
    loaded_normalized = loaded[0].replace("\\", "/") if loaded else ""
    declared_normalized = dagmc_declared_filename.replace("\\", "/")
    if len(loaded) != 1 or loaded_normalized != declared_normalized:
        raise ValueError(
            "terminal log does not bind the declared DAGMC filename"
        )
    statepoints = re.findall(r"Creating state point\s+(\S+\.h5)", text)
    if not statepoints or Path(statepoints[-1]).name != statepoint_path.name:
        raise ValueError("terminal log does not bind the final statepoint")
    writer_rows = [
        {
            "name": match.group("name"),
            "record_count": int(match.group("count")),
        }
        for match in _WRITER_PATTERN.finditer(text)
    ]
    expected = [
        {"name": Path(item["path"]).name, "record_count": item["record_count"]}
        for item in source_files
    ]
    if writer_rows != expected:
        raise ValueError(
            "terminal writer lines disagree with source-bank files"
        )
    mpi = re.findall(r"MPI Processes\s*\|\s*(\d+)", text)
    if len(mpi) > 1:
        raise ValueError("terminal log contains ambiguous MPI process counts")
    return {
        "path": str(path),
        "sha256": _hash(path),
        "openmc_version": "0.16.0",
        "run_completed": True,
        "writer_rows": writer_rows,
        "mpi_ranks": int(mpi[0]) if mpi else 1,
        "dagmc_loaded_filename": loaded[0],
        "final_statepoint_filename": statepoints[-1],
    }


def _extract_envelopes_from_h5m(
    dagmc_path: Path, envelope_requests: Sequence[Mapping[str, Any]]
) -> tuple[DagmcEnvelope, ...]:
    """Reload the H5M and derive every closed envelope from native senses."""
    requests = tuple(envelope_requests)
    if not requests:
        raise ValueError("at least one H5M envelope request is required")
    volume_ids = [int(item.get("volume_id", 0)) for item in requests]
    if min(volume_ids) <= 0 or len(volume_ids) != len(set(volume_ids)):
        raise ValueError("envelope volume IDs must be unique and positive")
    output = []
    for item in requests:
        required = {
            "volume_id",
            "envelope_id",
            "magnet_id",
            "plasma_direction_global",
            "toroidal_direction_global",
            "poloidal_direction_global",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(f"envelope request omits {sorted(missing)}")
        output.append(
            extract_closed_envelope(
                dagmc_path,
                int(item["volume_id"]),
                envelope_id=str(item["envelope_id"]),
                magnet_id=str(item["magnet_id"]),
                plasma_direction_global=item["plasma_direction_global"],
                toroidal_direction_global=item["toroidal_direction_global"],
                poloidal_direction_global=item["poloidal_direction_global"],
                coordinate_quantum_cm=float(
                    item.get("coordinate_quantum_cm", 1.0e-6)
                ),
                faceting_tolerances=item.get("faceting_tolerances"),
                centreline_frame=item.get("centreline_frame"),
            )
        )
    digest = _hash(dagmc_path)
    if any(item.envelope.dagmc_geometry_sha256 != digest for item in output):
        raise RuntimeError(
            "extracted envelope did not bind the exact H5M hash"
        )
    return tuple(output)


def _bank_tally_integrity_rows(
    records: np.ndarray,
    envelopes: Sequence[DagmcEnvelope],
    tally_rows: Sequence[Mapping[str, Any]],
    energy_edges: Mapping[str, np.ndarray],
    histories: int,
    *,
    grazing_tolerance: float = 1.0e-12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive bank currents from raw records and compare every tally bin."""
    by_surface = {
        int(surface_id): envelope
        for envelope in envelopes
        for surface_id in envelope.envelope.surface_ids
    }
    requested = set(by_surface)
    surface_ids = np.asarray(records["surf_id"], dtype=int).reshape(-1)
    foreign = set(surface_ids.tolist()) - requested
    if foreign:
        raise ValueError(
            f"surface bank contains records outside H5M envelopes: {sorted(foreign)}"
        )
    positions = _vectors(records, "r").astype(float, copy=False)
    directions = _vectors(records, "u").astype(float, copy=False)
    energy = np.asarray(records["E"], dtype=float).reshape(-1)
    time = np.asarray(records["time"], dtype=float).reshape(-1)
    weights = np.asarray(records["wgt"], dtype=float).reshape(-1)
    delayed_groups = np.asarray(records["delayed_group"], dtype=int).reshape(
        -1
    )
    pdg = np.asarray(records["particle"], dtype=int).reshape(-1)
    if any(
        np.any(~np.isfinite(value))
        for value in (positions, directions, energy, time, weights)
    ):
        raise ValueError("surface bank contains nonfinite phase-space values")
    if (
        np.any(np.abs(np.linalg.norm(directions, axis=1) - 1.0) > 1.0e-12)
        or np.any(energy <= 0.0)
        or np.any(time < 0.0)
        or np.any(weights < 0.0)
        or np.any(delayed_groups < 0)
    ):
        raise ValueError("surface bank contains invalid phase-space values")
    inverse_pdg = {value: key for key, value in PDG_PARTICLES.items()}
    particles = []
    unknown = sorted(set(pdg.tolist()) - set(inverse_pdg))
    if unknown:
        raise ValueError(f"surface bank contains unknown PDG IDs {unknown}")
    particles = np.asarray([inverse_pdg[int(value)] for value in pdg])
    unsupported = set(particles.tolist()) - set(energy_edges)
    if unsupported:
        raise ValueError(
            f"surface bank contains particles without tallies: {sorted(unsupported)}"
        )

    bank_values = {
        (
            int(row["surface_id"]),
            str(row["particle"]),
            int(row["energy_bin"]),
            str(row["canonical_direction"]),
        ): 0.0
        for row in tally_rows
    }
    facet_match_classes = []
    maximum_residual = 0.0
    for index, (
        surface_id,
        point,
        direction,
        particle,
        value,
        weight,
    ) in enumerate(
        zip(surface_ids, positions, directions, particles, energy, weights)
    ):
        envelope = by_surface[int(surface_id)]
        mapping = envelope.surface(int(surface_id)).locate(point)
        if mapping["mapping_status"] == "NO_VALID_FACET_MATCH":
            raise ValueError(
                f"surface-source record {index} has no valid H5M facet match"
            )
        facet_match_classes.append(mapping["mapping_status"])
        maximum_residual = max(
            maximum_residual, float(mapping["nearest_point_residual_cm"])
        )
        mu = float(np.dot(direction, mapping["outward_normal_global"]))
        if abs(mu) <= grazing_tolerance:
            raise ValueError(
                f"surface-source record {index} is grazing and cannot be reconciled"
            )
        canonical_direction = "outgoing" if mu > 0.0 else "incoming"
        edges = energy_edges[str(particle)]
        energy_bin = int(np.searchsorted(edges, value, side="right") - 1)
        if value == edges[-1]:
            energy_bin = len(edges) - 2
        if (
            energy_bin < 0
            or energy_bin >= len(edges) - 1
            or value < edges[energy_bin]
            or value > edges[energy_bin + 1]
        ):
            raise ValueError(
                f"surface-source record {index} energy is outside tally bins"
            )
        key = (int(surface_id), str(particle), energy_bin, canonical_direction)
        if key not in bank_values:
            raise ValueError(f"surface-source record {index} has no tally bin")
        bank_values[key] += float(weight) / histories

    compared = []
    all_pass = True
    maximum_difference = 0.0
    for row in tally_rows:
        key = (
            int(row["surface_id"]),
            str(row["particle"]),
            int(row["energy_bin"]),
            str(row["canonical_direction"]),
        )
        bank = float(bank_values[key])
        tally = float(row["tally_current_per_source"])
        difference = bank - tally
        tolerance = max(1.0e-14, 1.0e-10 * max(abs(bank), abs(tally), 1.0e-12))
        passes = abs(difference) <= tolerance
        all_pass = all_pass and passes
        maximum_difference = max(maximum_difference, abs(difference))
        compared.append(
            {
                **dict(row),
                "bank_current_per_source": bank,
                "difference": difference,
                "numerical_tolerance": tolerance,
                "passes_integrity_tolerance": passes,
            }
        )
    return compared, {
        "passes": bool(all_pass),
        "maximum_absolute_difference": maximum_difference,
        "record_count": int(len(records)),
        "facet_mapping_status_counts": {
            value: facet_match_classes.count(value)
            for value in sorted(set(facet_match_classes))
        },
        "maximum_facet_residual_cm": maximum_residual,
        "canonical_weight": "raw OpenMC wgt / model-derived source_histories",
        "tally_conditioning": False,
    }


def _capacity_proof(
    source_files: Sequence[Mapping[str, Any]],
    *,
    max_particles_per_process: int,
    max_source_files: int,
    mpi_ranks: int,
) -> dict[str, Any]:
    if min(max_particles_per_process, max_source_files, mpi_ranks) <= 0:
        raise ValueError("writer limits and MPI ranks must be positive")
    file_count_valid = 0 < len(source_files) <= max_source_files
    counts = [int(item["record_count"]) for item in source_files]
    if any(value < 0 for value in counts):
        raise ValueError("surface-source record counts cannot be negative")
    # Even for MPI>1, sum(rank_counts) < one_rank_cap implies every
    # nonnegative rank_count < one_rank_cap. At or above the cap the native
    # files do not carry enough per-rank identity, so the proof fails closed.
    capacity_proven_not_reached = file_count_valid and all(
        value < max_particles_per_process for value in counts
    )
    return {
        "max_particles_per_process": int(max_particles_per_process),
        "max_source_files": int(max_source_files),
        "mpi_ranks_from_log": int(mpi_ranks),
        "configured_global_upper_bound": int(
            max_particles_per_process * max_source_files * mpi_ranks
        ),
        "source_file_count_valid": file_count_valid,
        "capacity_proven_not_reached": capacity_proven_not_reached,
        "proof_rule": (
            "every global file record count is below one process's "
            "max_particles; this conservatively proves no rank reached cap"
        ),
    }


def audit_openmc16_surface_run(
    artifacts: StrictSurfaceRunArtifacts,
    *,
    envelope_requests: Sequence[Mapping[str, Any]],
    required_particles: Sequence[str] = ("neutron", "photon"),
) -> dict[str, Any]:
    """Qualify one real OpenMC surface run from artifacts, not assertions.

    This is the only path in this module allowed to emit
    ``COMPLETE_CROSSING_BANK``. Every quantity used in that classification is
    parsed from a hash-bound H5M, model.xml, statepoint, native source HDF5, or
    terminal OpenMC log.
    """
    dagmc_path = _resolved_file(artifacts.dagmc_path, "DAGMC H5M")
    model_path = _resolved_file(artifacts.model_xml_path, "OpenMC model.xml")
    statepoint_path = _resolved_file(artifacts.statepoint_path, "statepoint")
    log_path = _resolved_file(artifacts.terminal_log_path, "terminal log")
    source_paths = tuple(
        _resolved_file(path, "surface-source bank")
        for path in artifacts.surface_source_paths
    )
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("surface-source bank paths must be unique")
    inventory_path = _resolved_file(
        artifacts.accepted_magnet_inventory_path,
        "accepted magnet inventory",
    )
    root_acceptance_receipt_path = _resolved_file(
        artifacts.root_acceptance_receipt_path,
        "root acceptance receipt",
    )
    particles = tuple(str(item) for item in required_particles)
    if (
        not particles
        or len(particles) != len(set(particles))
        or set(particles) - {"neutron", "photon"}
    ):
        raise ValueError(
            "required_particles must be unique neutron/photon names"
        )

    model = _parse_model_xml(model_path, dagmc_path)
    envelopes = _extract_envelopes_from_h5m(dagmc_path, envelope_requests)
    localization_binding = _localization_topology_binding(
        dagmc_path, envelopes
    )
    root_acceptance = _verify_root_acceptance_receipt(
        root_acceptance_receipt_path,
        artifacts.expected_root_acceptance_receipt_sha256,
        inventory_path,
        dagmc_path,
    )
    accepted_inventory = _verify_accepted_magnet_inventory(
        inventory_path,
        root_acceptance,
        dagmc_path,
        envelopes,
        envelope_requests,
        localization_binding,
    )
    surface_ids = tuple(
        sorted(
            int(value)
            for envelope in envelopes
            for value in envelope.envelope.surface_ids
        )
    )
    if surface_ids != tuple(sorted(model["requested_surface_ids"])):
        raise ValueError(
            "surf_source_write IDs are not the complete selected H5M envelopes"
        )
    statepoint = _statepoint_header(statepoint_path, model)
    tally_rows, energy_edges = _directional_rows_from_statepoint(
        statepoint_path, model, envelopes, particles
    )
    records, source_files = _read_surface_banks(source_paths)
    log = _parse_terminal_log(
        log_path,
        source_files,
        model["dagmc_declared_filename"],
        statepoint_path,
    )
    compared_rows, integrity = _bank_tally_integrity_rows(
        records,
        envelopes,
        tally_rows,
        energy_edges,
        int(model["source_histories"]),
    )

    file_limit = int(model["max_particles_per_process"])
    capacity = _capacity_proof(
        source_files,
        max_particles_per_process=file_limit,
        max_source_files=int(model["max_source_files"]),
        mpi_ranks=int(log["mpi_ranks"]),
    )
    classification = (
        "COMPLETE_CROSSING_BANK"
        if integrity["passes"] and capacity["capacity_proven_not_reached"]
        else "TRUNCATED_INVALID_BANK"
    )
    dagmc_hash = _hash(dagmc_path)
    return {
        "schema": "parastell.openmc16_surface_run_audit/v1.0.0",
        "classification": classification,
        "complete_classification_source": "parsed_artifacts_only",
        "dagmc": {
            "path": str(dagmc_path),
            "sha256": dagmc_hash,
            "parser": "pydagmc.Model via extract_closed_envelope",
            "volume_ids": [
                item.envelope.dagmc_volume_id for item in envelopes
            ],
            "surface_ids": list(surface_ids),
            "canonical_geometry_fingerprints": [
                item.envelope.metadata["canonical_geometry_fingerprint"]
                for item in envelopes
            ],
            "edge_closure_pass": all(
                item.maximum_edge_multiplicity_error == 0 for item in envelopes
            ),
            "vector_area_closure_pass": all(
                item.vector_area_closure_relative <= 1.0e-7
                for item in envelopes
            ),
        },
        "localization_topology_binding": localization_binding,
        "root_acceptance_receipt": root_acceptance,
        "accepted_magnet_inventory": accepted_inventory,
        "model": {
            key: value
            for key, value in model.items()
            if key not in {"filters", "tallies"}
        },
        "statepoint": statepoint,
        "terminal_log": log,
        "surface_source_files": source_files,
        "stored_record_count": int(len(records)),
        "selected_record_count": int(len(records)),
        "foreign_record_count": 0,
        "required_particles": list(particles),
        "energy_edges_eV": {
            key: value.tolist() for key, value in energy_edges.items()
        },
        "same_run_integrity": {
            **integrity,
            "rows": compared_rows,
            "statepoint_sha256": statepoint["sha256"],
            "surface_source_sha256": [item["sha256"] for item in source_files],
            "dagmc_sha256": dagmc_hash,
            "model_xml_sha256": model["sha256"],
            "localization_topology_manifest_sha256": localization_binding[
                "manifest_sha256"
            ],
            "accepted_magnet_inventory_sha256": accepted_inventory["sha256"],
            "verified_magnet_inventory_sha256": accepted_inventory[
                "verified_inventory_sha256"
            ],
            "root_acceptance_receipt_sha256": root_acceptance["sha256"],
            "verified_root_acceptance_sha256": root_acceptance[
                "verified_acceptance_sha256"
            ],
        },
        "capacity": capacity,
        "native_phase_fields": sorted(OPENMC16_SOURCE_FIELDS),
        "native_phase_limitations": {
            "parent_history_id": "not stored by OpenMC 0.16 SourceParticle",
            "polarization": "not represented by OpenMC 0.16 SourceParticle",
        },
    }


def _directional_current_from_statepoint(
    statepoint_path: str | Path,
    particle: str,
    envelope: DagmcEnvelope,
) -> dict[int, dict[str, tuple[float, float]]]:
    """Read one directional-current tally without loading unrelated filters."""
    tally_name = f"pstl_envelope_{particle}_directional_current"
    with h5py.File(statepoint_path) as statepoint:
        tallies = statepoint["tallies"]
        tally = next(
            (
                value
                for key, value in tallies.items()
                if key.startswith("tally ")
                and _decode_hdf5_text(value["name"][()]) == tally_name
            ),
            None,
        )
        if tally is None:
            raise ValueError(f"statepoint has no tally named {tally_name!r}")
        filter_ids = [int(value) for value in tally["filters"][:]]
        filters = [tallies[f"filters/filter {value}"] for value in filter_ids]
        filter_types = [
            _decode_hdf5_text(value["type"][()]) for value in filters
        ]
        required = {"surface", "musurface", "particle", "energy"}
        if not required.issubset(filter_types):
            raise ValueError(
                f"{tally_name} filters {filter_types} omit {sorted(required)}"
            )
        particle_filter = filters[filter_types.index("particle")]
        scored_particles = {
            _decode_hdf5_text(value) for value in particle_filter["bins"][:]
        }
        if scored_particles != {particle}:
            raise ValueError(
                f"{tally_name} particle filter is {sorted(scored_particles)}"
            )
        dimensions = tuple(int(value["n_bins"][()]) for value in filters)
        results = tally["results"][:]
        sums = results[..., 0].reshape(dimensions + (-1,))
        sum_squares = results[..., 1].reshape(dimensions + (-1,))
        realizations = int(tally["n_realizations"][()])
        if realizations <= 0:
            raise ValueError(f"{tally_name} has no realizations")
        mean = sums / realizations
        if realizations == 1:
            standard_deviation = np.zeros_like(mean)
        else:
            variance = np.maximum(
                0.0,
                (sum_squares / realizations - mean**2) / (realizations - 1),
            )
            standard_deviation = np.sqrt(variance)
        surface_axis = filter_types.index("surface")
        mu_axis = filter_types.index("musurface")
        surface_ids = np.asarray(filters[surface_axis]["bins"][:], dtype=int)
        envelope_surface_ids = {
            int(value) for value in envelope.envelope.surface_ids
        }
        missing_surface_ids = envelope_surface_ids - set(surface_ids)
        if missing_surface_ids:
            raise ValueError(
                f"{tally_name} omits envelope surfaces {sorted(missing_surface_ids)}"
            )
        mu_edges = np.asarray(filters[mu_axis]["bins"][:], dtype=float)
        if len(mu_edges) != dimensions[mu_axis] + 1:
            raise ValueError(f"{tally_name} has malformed mu-surface edges")
        output = {
            int(surface_id): {
                "incoming": (0.0, 0.0),
                "outgoing": (0.0, 0.0),
            }
            for surface_id in surface_ids
            if int(surface_id) in envelope_surface_ids
        }
        for surface_index, surface_id in enumerate(surface_ids):
            if int(surface_id) not in envelope_surface_ids:
                continue
            normal_sign = envelope.envelope.surface(
                int(surface_id)
            ).openmc_normal_sign
            for mu_index, (mu_low, mu_high) in enumerate(
                zip(mu_edges[:-1], mu_edges[1:])
            ):
                if mu_high <= 0.0:
                    native_sense = -1
                elif mu_low >= 0.0:
                    native_sense = 1
                else:
                    raise ValueError(
                        f"{tally_name} has a mu bin straddling zero"
                    )
                selection = [slice(None)] * mean.ndim
                selection[surface_axis] = surface_index
                selection[mu_axis] = mu_index
                values = mean[tuple(selection)]
                deviations = standard_deviation[tuple(selection)]
                value = float(abs(values.sum()))
                deviation = float(np.sqrt(np.sum(deviations**2)))
                sense = (
                    "outgoing"
                    if normal_sign * native_sense > 0
                    else "incoming"
                )
                prior_value, prior_deviation = output[int(surface_id)][sense]
                output[int(surface_id)][sense] = (
                    prior_value + value,
                    float(np.hypot(prior_deviation, deviation)),
                )
    return output


def _closure_quantity(tally, tally_std, bank, bank_std):
    difference = float(bank - tally)
    tolerance = max(1.0e-14, 1.0e-10 * max(abs(tally), abs(bank), 1.0e-12))
    return {
        "tally_current_per_source": float(tally),
        "tally_std_dev": float(tally_std),
        "bank_current_per_source": float(bank),
        "bank_poisson_std_dev": float(bank_std),
        "difference": difference,
        "numerical_tolerance": tolerance,
        "passes_integrity_tolerance": abs(difference) <= tolerance,
        "statistical_relationship": (
            "same OpenMC histories; covariance unavailable; uncertainties are not combined"
        ),
    }


def _same_run_directional_integrity(
    statepoint_path,
    envelope,
    bank,
    record_particles,
    particle_types,
    surface_ids,
    transport_weights,
):
    crossing_sense = np.asarray(bank.columns["crossing_sense"]).astype(str)
    closure = {}
    for particle in sorted(set(particle_types)):
        tally = _directional_current_from_statepoint(
            statepoint_path, particle, envelope
        )
        by_surface = []
        surface_values = []
        for surface_id in envelope.envelope.surface_ids:
            values = {}
            for sense in ("incoming", "outgoing"):
                tally_value, tally_std = tally[int(surface_id)][sense]
                mask = (
                    (record_particles == particle)
                    & (surface_ids == int(surface_id))
                    & (crossing_sense == sense)
                )
                weights = transport_weights[mask]
                values[sense] = (
                    tally_value,
                    tally_std,
                    float(weights.sum()),
                    float(np.sqrt(np.sum(weights**2))),
                )
            incoming = values["incoming"]
            outgoing = values["outgoing"]
            surface = {
                "surface_id": int(surface_id),
                "openmc_normal_sign": envelope.envelope.surface(
                    int(surface_id)
                ).openmc_normal_sign,
                "incoming": _closure_quantity(*incoming),
                "outgoing": _closure_quantity(*outgoing),
                "net": _closure_quantity(
                    outgoing[0] - incoming[0],
                    float(np.hypot(outgoing[1], incoming[1])),
                    outgoing[2] - incoming[2],
                    float(np.hypot(outgoing[3], incoming[3])),
                ),
                "total_crossing": _closure_quantity(
                    outgoing[0] + incoming[0],
                    float(np.hypot(outgoing[1], incoming[1])),
                    outgoing[2] + incoming[2],
                    float(np.hypot(outgoing[3], incoming[3])),
                ),
            }
            surface["passes_integrity_tolerance"] = all(
                surface[name]["passes_integrity_tolerance"]
                for name in ("incoming", "outgoing", "net", "total_crossing")
            )
            by_surface.append(surface)
            surface_values.append(values)
        whole_values = {}
        for sense in ("incoming", "outgoing"):
            selected = [value[sense] for value in surface_values]
            whole_values[sense] = (
                float(sum(value[0] for value in selected)),
                float(np.sqrt(sum(value[1] ** 2 for value in selected))),
                float(sum(value[2] for value in selected)),
                float(np.sqrt(sum(value[3] ** 2 for value in selected))),
            )
        incoming = whole_values["incoming"]
        outgoing = whole_values["outgoing"]
        whole = {
            "incoming": _closure_quantity(*incoming),
            "outgoing": _closure_quantity(*outgoing),
            "net": _closure_quantity(
                outgoing[0] - incoming[0],
                float(np.hypot(outgoing[1], incoming[1])),
                outgoing[2] - incoming[2],
                float(np.hypot(outgoing[3], incoming[3])),
            ),
            "total_crossing": _closure_quantity(
                outgoing[0] + incoming[0],
                float(np.hypot(outgoing[1], incoming[1])),
                outgoing[2] + incoming[2],
                float(np.hypot(outgoing[3], incoming[3])),
            ),
        }
        whole["passes_integrity_tolerance"] = all(
            whole[name]["passes_integrity_tolerance"]
            for name in ("incoming", "outgoing", "net", "total_crossing")
        )
        closure[particle] = {
            "whole_envelope": whole,
            "by_surface": by_surface,
            "passes_integrity_tolerance": whole["passes_integrity_tolerance"]
            and all(
                value["passes_integrity_tolerance"] for value in by_surface
            ),
        }
    return closure


def export_openmc16_handoff(
    output_path: str | Path,
    *,
    statepoint_path: str | Path,
    surface_source_paths: Sequence[str | Path],
    envelope: DagmcEnvelope,
    histories: int,
    energy_edges_by_particle: Mapping[str, Sequence[float]],
    physical_source_rate_per_s: float | None = None,
    parastell_commit: str,
    source_definition_sha256: str,
    adaptive_patch_target_ess: float | None = None,
    adaptive_patch_minimum_records: int = 4,
    adaptive_patch_maximum_depth: int = 5,
    surface_source_max_particles: int | None = None,
    surface_source_max_files: int | None = None,
    surface_source_sampling_applied: bool = False,
    mpi_ranks: int | None = None,
    centreline_frame=None,
    facet_barycentric_tolerance: float = 1.0e-7,
    facet_source_tolerance_cm: float = 1.0e-5,
) -> dict:
    """Write raw correlated records and a same-run integrity comparison."""
    import openmc

    if histories <= 0:
        raise ValueError("histories must be positive")
    arrays = []
    source_hashes = []
    source_file_ids = []
    source_record_indices = []
    for file_id, path in enumerate(surface_source_paths):
        source_path = Path(path)
        with h5py.File(source_path) as source:
            if "source_bank" not in source:
                raise ValueError(f"{source_path} has no source_bank")
            array = source["source_bank"][:]
            arrays.append(array)
            source_file_ids.append(
                np.full(len(array), file_id, dtype=np.int64)
            )
            source_record_indices.append(np.arange(len(array), dtype=np.int64))
        source_hashes.append(_hash(source_path))
    if not arrays:
        raise ValueError("at least one surface-source file is required")
    records = np.concatenate(arrays)
    source_file_ids = np.concatenate(source_file_ids)
    source_record_indices = np.concatenate(source_record_indices)
    stored_record_count = len(records)
    records, surface_ids, keep = _select_envelope_records(
        records, envelope.envelope.surface_ids
    )
    source_file_ids = source_file_ids[keep]
    source_record_indices = source_record_indices[keep]
    pdg = np.asarray(records["particle"]).reshape(-1)
    inverse_pdg = {value: name for name, value in PDG_PARTICLES.items()}
    unsupported = set(np.unique(pdg)) - set(inverse_pdg)
    if unsupported:
        raise ValueError(
            f"unsupported PDG particle IDs: {sorted(unsupported)}"
        )
    particles = np.asarray([inverse_pdg[int(value)] for value in pdg])
    positions = _vectors(records, "r")
    directions = _vectors(records, "u")
    facet_mapping = envelope.facet_mappings(
        surface_ids,
        positions,
        require_valid=True,
        barycentric_tolerance=facet_barycentric_tolerance,
        source_tolerance_cm=facet_source_tolerance_cm,
    )
    normals = facet_mapping.pop("outward_normal_global")
    openmc_weights = np.asarray(records["wgt"]).reshape(-1)
    transport_weights = openmc_weights / histories
    bank = build_correlated_bank(
        envelope.envelope,
        position_global_cm=positions,
        direction_global=directions,
        energy_eV=np.asarray(records["E"]).reshape(-1),
        raw_weight=transport_weights,
        openmc_weight=openmc_weights,
        particle=particles,
        surface_id=surface_ids,
        time_s=np.asarray(records["time"]).reshape(-1),
        delayed_group=np.asarray(records["delayed_group"]).reshape(-1),
        energy_edges_by_particle=energy_edges_by_particle,
        outward_normal_global=normals,
        source_file_id=source_file_ids,
        source_record_index=source_record_indices,
        facet_mapping=facet_mapping,
        centreline_frame=centreline_frame,
    )
    if adaptive_patch_target_ess is not None:
        bank = assign_adaptive_surface_patches(
            envelope.envelope,
            bank,
            target_effective_sample_size=adaptive_patch_target_ess,
            minimum_records=adaptive_patch_minimum_records,
            maximum_depth=adaptive_patch_maximum_depth,
        )
    closure = _same_run_directional_integrity(
        statepoint_path,
        envelope,
        bank,
        particles,
        energy_edges_by_particle,
        surface_ids,
        transport_weights,
    )
    completeness = classify_crossing_bank(
        stored_record_count=stored_record_count,
        selected_record_count=len(records),
        max_particles_per_file=surface_source_max_particles,
        max_source_files=surface_source_max_files,
        source_file_count=len(surface_source_paths),
        mpi_ranks=mpi_ranks,
        sampling_applied=surface_source_sampling_applied,
    )
    if completeness["classification"] == "COMPLETE_CROSSING_BANK":
        completeness["classification"] = "TRUNCATED_INVALID_BANK"
        completeness["legacy_assertion_only_accounting"] = True
        completeness["reason"] = (
            "legacy exporter accepts caller-provided history/capacity values; "
            "use audit_openmc16_surface_run for COMPLETE classification"
        )
    zero_record_interpretation = {
        "status": "EMPTY" if len(records) == 0 else "OBSERVED",
        "physical_zero_claimed": False,
        "confidence_level": 0.95,
        "poisson_upper_bound_expected_crossings_per_source": (
            float(-np.log(0.05) / histories)
            if len(records) == 0
            and completeness["classification"] == "COMPLETE_CROSSING_BANK"
            else None
        ),
        "upper_bound_quantity": "expected crossing records per source history",
        "upper_bound_assumptions": (
            "complete unsampled crossing bank and a Poisson zero-count model; "
            "this is not a claim of zero physical current or scalar flux"
        ),
    }
    bank.metadata.update(
        {
            "same_run_integrity_closure": closure,
            "closure_semantics": (
                "tally and bank are separate output mechanisms from the same histories"
            ),
            "continuous_energy_authoritative": True,
            "continuous_direction_authoritative": True,
            "source_surface_sha256": source_hashes,
            "projection_uncertainty_model": (
                "weighted_event_counting_approximation"
            ),
            "surface_bank_completeness": completeness,
            "zero_record_interpretation": zero_record_interpretation,
            "canonical_bank": True,
            "facet_identity": (
                "canonical geometry fingerprint + DAGMC volume + DAGMC "
                "surface + outward-oriented quantized triangle"
            ),
            "facet_mapping_contract": {
                "schema": "parastell.magnet_boundary_source/v2.2.0",
                "classification": "FACET_COMPLETE_BOUNDARY_PASS",
                "accepted_match_classes": [
                    "EXACT_FACET_MATCH",
                    "EDGE_TOLERANCE_MATCH",
                    "VERTEX_TOLERANCE_MATCH",
                ],
                "fatal_match_class": "NO_VALID_FACET_MATCH",
                "barycentric_tolerance": facet_barycentric_tolerance,
                "source_tolerance_cm": facet_source_tolerance_cm,
            },
            "frame_type": (
                "coil_centerline_parallel_transport"
                if centreline_frame is not None
                else None
            ),
        }
    )
    normalization = {
        "basis": "per source history",
        "particles_per_source_history": 1.0,
        "area_basis": "surface integrated; patch area stored separately",
        "energy_bin_width": "not divided; group-integrated partial current",
        "solid_angle_measure": "not divided; angular-bin solid angle in metadata",
        "quantity": "partial crossing current",
        "time_basis": "none for per-source values",
        "physical_source_rate_per_s": physical_source_rate_per_s,
        "physical_scaling_contract": (
            "multiply per-source quantities by physical_source_rate_per_s"
            if physical_source_rate_per_s is not None
            else "no physical scaling supplied"
        ),
    }
    provenance = {
        "parastell_commit": parastell_commit,
        "openmc_version": openmc.__version__,
        "openmc_statepoint_sha256": _hash(statepoint_path),
        "dagmc_geometry_sha256": envelope.envelope.dagmc_geometry_sha256,
        "canonical_geometry_fingerprint": envelope.envelope.metadata.get(
            "canonical_geometry_fingerprint"
        ),
        "source_definition_sha256": source_definition_sha256,
        "histories": histories,
        "surface_source_sha256": source_hashes,
    }
    return write_handoff(
        output_path,
        envelope.envelope,
        bank,
        provenance=provenance,
        normalization=normalization,
        facet_catalog=envelope.facet_metadata(),
    )


def export_openmc16_handoffs(
    output_directory: str | Path,
    *,
    statepoint_path: str | Path,
    surface_source_paths: Sequence[str | Path],
    envelopes: Sequence[DagmcEnvelope],
    histories: int,
    energy_edges_by_particle: Mapping[str, Sequence[float]],
    physical_source_rate_per_s: float | None = None,
    parastell_commit: str,
    source_definition_sha256: str,
    adaptive_patch_target_ess: float | None = None,
    adaptive_patch_minimum_records: int = 4,
    adaptive_patch_maximum_depth: int = 5,
    surface_source_max_particles: int | None = None,
    surface_source_max_files: int | None = None,
    surface_source_sampling_applied: bool = False,
    mpi_ranks: int | None = None,
    centreline_frames: Mapping[str, Any] | None = None,
    facet_barycentric_tolerance: float = 1.0e-7,
    facet_source_tolerance_cm: float = 1.0e-5,
) -> dict:
    """Write one raw correlated handoff for every selected magnet."""
    selected = tuple(envelopes)
    if not selected:
        raise ValueError("at least one closed magnet envelope is required")
    identifiers = [item.envelope.envelope_id for item in selected]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("magnet envelope IDs must be unique")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for envelope in selected:
        safe_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in envelope.envelope.envelope_id
        )
        path = directory / f"magnet_boundary_{safe_id}.h5"
        manifest = export_openmc16_handoff(
            path,
            statepoint_path=statepoint_path,
            surface_source_paths=surface_source_paths,
            envelope=envelope,
            histories=histories,
            energy_edges_by_particle=energy_edges_by_particle,
            physical_source_rate_per_s=physical_source_rate_per_s,
            parastell_commit=parastell_commit,
            source_definition_sha256=source_definition_sha256,
            adaptive_patch_target_ess=adaptive_patch_target_ess,
            adaptive_patch_minimum_records=adaptive_patch_minimum_records,
            adaptive_patch_maximum_depth=adaptive_patch_maximum_depth,
            surface_source_max_particles=surface_source_max_particles,
            surface_source_max_files=surface_source_max_files,
            surface_source_sampling_applied=surface_source_sampling_applied,
            mpi_ranks=mpi_ranks,
            centreline_frame=(centreline_frames or {}).get(
                envelope.envelope.envelope_id
            ),
            facet_barycentric_tolerance=facet_barycentric_tolerance,
            facet_source_tolerance_cm=facet_source_tolerance_cm,
        )
        outputs.append(
            {
                "envelope_id": envelope.envelope.envelope_id,
                "magnet_component": envelope.envelope.magnet_component,
                "dagmc_volume_id": envelope.envelope.dagmc_volume_id,
                "path": str(path.resolve()),
                "sha256": _hash(path),
                "record_count": manifest["record_count"],
                "integrated_current": manifest["integrated_current"],
            }
        )
    return {
        "schema": "parastell.magnet_boundary_source_collection/v1.0.0",
        "dagmc_geometry_sha256": selected[0].envelope.dagmc_geometry_sha256,
        "handoffs": outputs,
    }
