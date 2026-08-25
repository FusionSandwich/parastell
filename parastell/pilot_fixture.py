"""Portable receipts for real ParaStell magnet-radiation pilot runs.

The receipt deliberately leaves transport artifacts outside the source tree.  It
binds a validated neutral radiation-field bundle by hash and records the
reader-level acceptance checks needed by downstream test-spectrum consumers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "parastell.magnet_pilot_fixture_manifest/v1.0.0"
TIME_SEMANTICS = "prompt particle flight time, not irradiation time"
ACCEPTANCE_SCOPE = "reader_contract_and_normalization_plumbing_only"
QUALIFICATION_SCOPE = (
    "boundary_record_particle_population_only_not_physics_qualification"
)
ACCEPTANCE_GATES = (
    "schema_round_trip",
    "current_conservation",
    "scalar_flux_normalization",
    "coordinate_transform",
    "particle_identity",
    "energy_preservation",
    "direction_preservation",
    "time_semantics",
)
_SHA256_HEX = frozenset("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(
        character not in _SHA256_HEX for character in text
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return text


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _energy_group(edges: Any, energy_eV: float) -> int:
    import numpy as np

    values = np.asarray(edges, dtype=float)
    group = int(np.searchsorted(values, energy_eV, side="right") - 1)
    if energy_eV == values[-1]:
        group = len(values) - 2
    return group


def _evaluate_boundary_acceptance(
    root: Path, products: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Independently check preservation and frame semantics in boundary banks."""
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "h5py and numpy are required for pilot validation"
        ) from exc

    particle_counts = {"neutron": 0, "photon": 0}
    record_count = 0
    bank_count = 0
    time_record_count = 0
    facet_record_count = 0
    for product in products:
        if product.get("kind") != "boundary_phase_space":
            continue
        bank_count += 1
        path = root / str(product["path"])
        with h5py.File(path, "r") as handle:
            manifest = json.loads(_decode(handle["manifest_json"][()]))
            records = handle["records"]
            count = len(records["record_id"])
            record_count += count
            surfaces = {
                int(item["surface_id"]): item
                for item in manifest["envelope"]["surfaces"]
            }
            positions = np.asarray(
                records["position_global_cm"][...], dtype=float
            )
            local_positions = np.asarray(
                records["position_local_cm"][...], dtype=float
            )
            directions = np.asarray(
                records["direction_global"][...], dtype=float
            )
            local_directions = np.asarray(
                records["direction_local"][...], dtype=float
            )
            record_normals = np.asarray(
                records["outward_normal_global"][...], dtype=float
            )
            surface_ids = np.asarray(records["surface_id"][...], dtype=int)
            for surface_id in np.unique(surface_ids):
                mask = surface_ids == surface_id
                surface = surfaces[int(surface_id)]
                frame = np.asarray(
                    (
                        surface["toroidal_direction_global"],
                        surface["poloidal_direction_global"],
                        surface["outward_normal_global"],
                    ),
                    dtype=float,
                )
                centroid = np.asarray(
                    surface["centroid_global_cm"], dtype=float
                )
                if not np.allclose(
                    (positions[mask] - centroid) @ frame.T,
                    local_positions[mask],
                    rtol=0.0,
                    atol=2.0e-10,
                ):
                    raise ValueError(
                        "boundary global/local position transform failed"
                    )
                # Faceted DAGMC surfaces carry a per-record triangle normal.
                # The canonical local direction keeps the surface's toroidal
                # seed but reprojects it against that exact normal.
                normals = record_normals[mask]
                tangent_seed = np.asarray(
                    surface["toroidal_direction_global"], dtype=float
                )
                tangents = tangent_seed[None, :] - (
                    np.sum(tangent_seed[None, :] * normals, axis=1)[:, None]
                    * normals
                )
                tangents /= np.linalg.norm(tangents, axis=1)[:, None]
                poloidal = np.cross(normals, tangents)
                expected_local_direction = np.column_stack(
                    (
                        np.sum(directions[mask] * tangents, axis=1),
                        np.sum(directions[mask] * poloidal, axis=1),
                        np.sum(directions[mask] * normals, axis=1),
                    )
                )
                if not np.allclose(
                    expected_local_direction,
                    local_directions[mask],
                    rtol=0.0,
                    atol=2.0e-10,
                ):
                    raise ValueError(
                        "boundary global/local direction transform failed"
                    )

            if count and not (
                np.allclose(
                    np.linalg.norm(directions, axis=1), 1.0, atol=1.0e-12
                )
                and np.allclose(
                    np.linalg.norm(local_directions, axis=1), 1.0, atol=1.0e-12
                )
            ):
                raise ValueError(
                    "boundary directions were not preserved as unit vectors"
                )

            particles = np.asarray(records["particle"].asstr()[()]).astype(str)
            pdg = np.asarray(records["particle_pdg"][...], dtype=int)
            expected_pdg = np.asarray(
                [
                    {"neutron": 2112, "photon": 22}.get(name, -1)
                    for name in particles
                ]
            )
            if not np.array_equal(pdg, expected_pdg):
                raise ValueError(
                    "boundary particle identity was not preserved"
                )
            for name in particle_counts:
                particle_counts[name] += int(
                    np.count_nonzero(particles == name)
                )

            axes = manifest["energy_axes"]
            energies = np.asarray(records["energy_eV"][...], dtype=float)
            groups = np.asarray(records["energy_group"][...], dtype=int)
            expected_groups = np.asarray(
                [
                    _energy_group(
                        axes[f"{name}_energy_edges_eV"], float(energy)
                    )
                    for name, energy in zip(particles, energies)
                ],
                dtype=int,
            )
            if not np.array_equal(groups, expected_groups):
                raise ValueError(
                    "boundary continuous energy/group identity failed"
                )

            availability = manifest.get("field_availability", {})
            if manifest.get("time_semantics") != TIME_SEMANTICS:
                raise ValueError("boundary prompt-time semantics are absent")
            if "time_s" in records:
                times = np.asarray(records["time_s"][...], dtype=float)
                if np.any(~np.isfinite(times)) or np.any(times < 0.0):
                    raise ValueError(
                        "boundary prompt flight times are invalid"
                    )
                time_record_count += count
            elif availability.get("time_s", {}).get("available") is not False:
                raise ValueError("boundary time availability is not explicit")

            facet_fields = {
                "facet_id",
                "barycentric_coordinates",
                "distance_to_facet_residual_cm",
                "facet_mapping_status",
            }
            if facet_fields.issubset(records):
                barycentric = np.asarray(
                    records["barycentric_coordinates"][...], dtype=float
                )
                if count and (
                    np.any(~np.isfinite(barycentric))
                    or not np.allclose(
                        np.sum(barycentric, axis=1),
                        1.0,
                        rtol=0.0,
                        atol=2.0e-10,
                    )
                ):
                    raise ValueError(
                        "boundary barycentric coordinates are invalid"
                    )
                facet_record_count += count

    pass_result = lambda evidence: {"status": "PASS", "evidence": evidence}
    gates = {
        "current_conservation": pass_result(
            "canonical record weights equal each bank integrated current and projection"
        ),
        "coordinate_transform": pass_result(
            "global positions and directions reconstruct every stored surface-local record"
        ),
        "particle_identity": pass_result(
            "particle names agree with PDG identities"
        ),
        "energy_preservation": pass_result(
            "continuous energies reproduce stored particle-specific group identities"
        ),
        "direction_preservation": pass_result(
            "global and local directions remain normalized and frame-equivalent"
        ),
        "time_semantics": pass_result(
            "prompt flight-time meaning is explicit; availability is explicit per bank"
        ),
    }
    summary = {
        "bank_count": bank_count,
        "record_count": record_count,
        "particle_record_counts": particle_counts,
        "prompt_time_record_count": time_record_count,
        "facet_mapped_record_count": facet_record_count,
    }
    return gates, summary


def _population_qualification(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Separate statistical population from structural reader acceptance."""

    particle_counts = summary.get("particle_record_counts", {})
    observations = {
        "boundary_bank_present": int(summary.get("bank_count", 0)) > 0,
        "boundary_record_present": int(summary.get("record_count", 0)) > 0,
        "neutron_records_present": int(particle_counts.get("neutron", 0)) > 0,
        "photon_records_present": int(particle_counts.get("photon", 0)) > 0,
    }
    criteria = {
        name: {
            "status": "PASS" if passed else "BLOCKED",
            "evidence": (
                "observed in the validated external boundary banks"
                if passed
                else "not observed in the validated external boundary banks"
            ),
        }
        for name, passed in observations.items()
    }
    blockers = [name for name, passed in observations.items() if not passed]
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "requirement": (
            "at least one validated boundary bank and positive neutron and "
            "photon boundary-record populations"
        ),
        "criteria": criteria,
        "blocking_reasons": blockers,
    }


def _scoped_qualification_status(population_status: str) -> str:
    if population_status == "PASS":
        return "STRUCTURAL_PILOT_POPULATION_PASS"
    if population_status == "BLOCKED":
        return "STRUCTURAL_PILOT_POPULATION_BLOCKED"
    raise ValueError("unsupported pilot population status")


def _canonical_bundle_tree(
    root: Path, bundle: Mapping[str, Any]
) -> tuple[str, int, int]:
    manifest_path = root / "manifest.json"
    entries = [
        {
            "path": "manifest.json",
            "sha256": _sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }
    ]
    for product in bundle["products"]:
        entries.append(
            {
                "path": str(product["path"]),
                "sha256": _require_sha256(product["sha256"], "product sha256"),
                "size_bytes": int(product["size_bytes"]),
            }
        )
    entries.sort(key=lambda item: item["path"])
    return (
        _canonical_json_sha256(entries),
        len(entries),
        sum(int(item["size_bytes"]) for item in entries),
    )


def build_pilot_fixture_manifest(
    bundle_directory: str | Path,
    *,
    artifact_uri: str | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Validate one real bundle and return a compact, source-safe receipt."""
    from .magnet_radiation_field_bundle import read_radiation_field_bundle

    root = Path(bundle_directory).resolve()
    bundle = read_radiation_field_bundle(root)
    boundary_gates, boundary_summary = _evaluate_boundary_acceptance(
        root, bundle["products"]
    )
    tree_sha256, file_count, size_bytes = _canonical_bundle_tree(root, bundle)
    manifest_sha256 = _sha256(root / "manifest.json")
    product_kinds: dict[str, int] = {}
    for item in bundle["products"]:
        kind = str(item["kind"])
        product_kinds[kind] = product_kinds.get(kind, 0) + 1
    scalar_products = [
        item
        for item in bundle["products"]
        if item["kind"] == "volume_scalar_flux"
    ]
    if not scalar_products:
        raise ValueError("pilot fixture requires volume scalar flux")
    gates = {
        "schema_round_trip": {
            "status": "PASS",
            "evidence": "neutral bundle and every referenced product passed strict readback",
        },
        "scalar_flux_normalization": {
            "status": "PASS",
            "evidence": (
                "track-length scalar flux retains volume and physical source-rate "
                "normalization and is rejected if encoded as surface current"
            ),
        },
        **boundary_gates,
    }
    pilot_summary = {
        "histories": int(bundle["provenance"]["histories"]),
        "batches": int(bundle["provenance"]["batches"]),
        "physical_source_rate_per_s": float(
            bundle["source"]["physical_source_rate_per_s"]
        ),
        "magnet_count": len(bundle["magnet_inventory"]),
        "product_kind_counts": dict(sorted(product_kinds.items())),
        "bank_completeness": dict(bundle["bank_completeness"]),
        **boundary_summary,
    }
    population = _population_qualification(pilot_summary)
    return {
        "schema": SCHEMA,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "acceptance_scope": ACCEPTANCE_SCOPE,
        "qualification_scope": QUALIFICATION_SCOPE,
        "qualification_status": _scoped_qualification_status(
            population["status"]
        ),
        "qualification_limitations": [
            (
                "reader acceptance and particle population do not establish "
                "statistical convergence, activation readiness, SPECTRA-PKA "
                "eligibility, or production readiness"
            )
        ],
        "external_artifact": {
            "uri": artifact_uri or root.as_uri(),
            "local_path_hint": str(root),
            "bundle_manifest_sha256": manifest_sha256,
            "canonical_tree_sha256": tree_sha256,
            "file_count": file_count,
            "size_bytes": size_bytes,
        },
        "producer_identity": {
            "parastell_commit": bundle["provenance"]["parastell_commit"],
            "openmc_version": bundle["provenance"]["openmc_version"],
            "openmc_commit": bundle["provenance"]["openmc_commit"],
            "geometry_fingerprint": bundle["geometry"][
                "canonical_geometry_fingerprint"
            ],
            "statepoint_sha256": bundle["provenance"]["statepoint_sha256"],
        },
        "pilot_summary": pilot_summary,
        "acceptance": gates,
        "population_qualification": population,
        "committed_payload": {
            "included_large_artifacts": [],
            "boundary_downsample": {
                "status": "OMITTED",
                "reason": (
                    "no inclusion-probability and weight-correction contract was "
                    "qualified for boundary downsampling"
                ),
            },
            "scalar_downsample": {
                "status": "OMITTED",
                "reason": "the external hash-bound scalar product is the qualified pilot",
            },
        },
    }


def validate_pilot_fixture_manifest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a receipt without requiring its external bundle to be mounted."""
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported pilot fixture manifest schema")
    required = {
        "created_utc",
        "acceptance_scope",
        "qualification_scope",
        "qualification_status",
        "qualification_limitations",
        "external_artifact",
        "producer_identity",
        "pilot_summary",
        "acceptance",
        "population_qualification",
        "committed_payload",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"pilot fixture manifest is missing {sorted(missing)}"
        )
    external = value["external_artifact"]
    if not isinstance(external, Mapping) or not str(external.get("uri", "")):
        raise ValueError("pilot fixture external artifact URI is absent")
    for name in ("bundle_manifest_sha256", "canonical_tree_sha256"):
        _require_sha256(external.get(name), f"external_artifact.{name}")
    for name in ("file_count", "size_bytes"):
        raw = external.get(name)
        if isinstance(raw, bool) or int(raw) <= 0 or int(raw) != raw:
            raise ValueError(f"external_artifact.{name} must be positive")
    acceptance = value["acceptance"]
    if not isinstance(acceptance, Mapping) or set(acceptance) != set(
        ACCEPTANCE_GATES
    ):
        raise ValueError("pilot fixture acceptance gate set is incomplete")
    for name, result in acceptance.items():
        if not isinstance(result, Mapping) or result.get("status") != "PASS":
            raise ValueError(
                f"pilot fixture acceptance gate {name!r} did not pass"
            )
        if not str(result.get("evidence", "")):
            raise ValueError(
                f"pilot fixture acceptance gate {name!r} lacks evidence"
            )
    committed = value["committed_payload"]
    if committed.get("included_large_artifacts") != []:
        raise ValueError(
            "pilot fixture receipt cannot include large artifacts"
        )
    summary = value["pilot_summary"]
    rate = float(summary.get("physical_source_rate_per_s", math.nan))
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(
            "pilot fixture source rate must be positive and finite"
        )
    expected_population = _population_qualification(summary)
    if value["population_qualification"] != expected_population:
        raise ValueError(
            "pilot fixture population qualification is inconsistent with "
            "the observed boundary populations"
        )
    if value.get("acceptance_scope") != ACCEPTANCE_SCOPE:
        raise ValueError("pilot fixture acceptance scope is invalid")
    if value.get("qualification_scope") != QUALIFICATION_SCOPE:
        raise ValueError("pilot fixture qualification scope is invalid")
    expected_status = _scoped_qualification_status(
        expected_population["status"]
    )
    if value["qualification_status"] != expected_status:
        raise ValueError(
            "pilot fixture qualification status is inconsistent with its "
            "population qualification"
        )
    limitations = value.get("qualification_limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item for item in limitations)
    ):
        raise ValueError("pilot fixture qualification limitations are absent")
    return json.loads(json.dumps(value))


def write_pilot_fixture_manifest(
    path: str | Path,
    bundle_directory: str | Path,
    *,
    artifact_uri: str | None = None,
) -> dict[str, Any]:
    """Write one JSON receipt; no transport artifact is copied."""
    output = Path(path)
    if output.suffix.lower() != ".json":
        raise ValueError("pilot fixture receipt must be a JSON file")
    value = validate_pilot_fixture_manifest(
        build_pilot_fixture_manifest(
            bundle_directory, artifact_uri=artifact_uri
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**value, "manifest_sha256": _sha256(output)}


def read_pilot_fixture_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a compact pilot receipt."""
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    return {
        **validate_pilot_fixture_manifest(value),
        "manifest_sha256": _sha256(source),
    }
