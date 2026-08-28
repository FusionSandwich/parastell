"""Geometry-bound producer contract for downstream activation workflows.

ParaStell owns geometry and transport-field provenance.  Activation execution
belongs to DPA_workflow.  This module keeps that boundary explicit and, most
importantly, prevents a magnet surface-current bank from being substituted for
the volume scalar flux required by activation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree


ACTIVATION_HANDOFF_SCHEMA = "parastell.activation_ready_metadata/v2.0.0"
ACTIVATION_EXECUTION_OWNER = "DPA_workflow"
ENDFB80_FAST_CHAIN_SHA256 = (
    "5eeb727498d824d7c951ad89864bbc1c2d76ec5e8c9097a820505213ba6a2bf3"
)
FULL_TRANSPORT_CATALOG_SHA256 = (
    "218236803b2c4a21b038992af93dacfdfe5c0c0401cbbc57f3ff3a947c63abc7"
)
FULL_TRANSPORT_PAYLOAD_LEDGER_SHA256 = (
    "ef339cb80fef10d54b9372087c1a0ed4864bc6af2212460d591932aeea40f38b"
)
DIRECT90_REFERENCE_MANIFEST_SHA256 = (
    "b6e723cdb9ac95d789a838abbf44590d210c4fdbe718c3b459777d38768e0499"
)
EXPECTED_DIRECT90_MATERIAL_COUNTS = {
    "Vacuum": 2,
    "back_wall": 1,
    "breeder": 1,
    "first_wall": 1,
    "high_temperature_shield": 1,
    "low_temperature_shield": 1,
    "magnet_envelope": 1,
    "vacuum_vessel": 1,
}
DIRECT90_SOURCE_MESH_SHA256 = (
    "65264e15669d09c43f107c3b43c2af24ffbd15173e3bbd0e990b527bfa0b5322"
)
DIRECT90_STRENGTHS_SHA256 = (
    "0ed18ab58bcc1e9884bf1b5c8bf19a7b7558ce7afe1869f1a2b01710148af6df"
)
DIRECT90_MODELED_SOURCE_RATE_PER_S = 9.427053032700795e19


class ActivationHandoffError(ValueError):
    """Raised when activation provenance is missing, ambiguous, or unsafe."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def inspect_activation_data(
    *,
    chain_path: str | Path,
    cross_sections_path: str | Path,
    hash_transport_payloads: bool = False,
) -> dict[str, Any]:
    """Hash the qualified local activation inputs without modifying them."""
    chain = Path(chain_path).resolve()
    catalog = Path(cross_sections_path).resolve()
    if not chain.is_file():
        raise FileNotFoundError(chain)
    if not catalog.is_file():
        raise FileNotFoundError(catalog)
    chain_hash = _sha256_file(chain)
    catalog_hash = _sha256_file(catalog)
    libraries = ElementTree.parse(catalog).getroot().findall("library")
    payloads = []
    missing = []
    for library in libraries:
        relative_path = str(library.attrib["path"])
        payload = (catalog.parent / relative_path).resolve()
        if not payload.is_file():
            missing.append(relative_path)
            continue
        row = {
            "materials": str(library.attrib.get("materials", "")),
            "path": relative_path,
            "type": str(library.attrib.get("type", "")),
            "size_bytes": payload.stat().st_size,
        }
        if hash_transport_payloads:
            row["sha256"] = _sha256_file(payload)
        payloads.append(row)
    ledger_hash = None
    if hash_transport_payloads and not missing:
        encoded = json.dumps(
            payloads, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        ledger_hash = hashlib.sha256(encoded).hexdigest()
    return {
        "chain": {
            "path": str(chain),
            "sha256": chain_hash,
            "size_bytes": chain.stat().st_size,
            "format": "OpenMC depletion chain XML",
            "qualified": chain_hash == ENDFB80_FAST_CHAIN_SHA256,
        },
        "transport_catalog": {
            "path": str(catalog),
            "sha256": catalog_hash,
            "size_bytes": catalog.stat().st_size,
            "format": "OpenMC cross_sections.xml",
            "qualified": catalog_hash == FULL_TRANSPORT_CATALOG_SHA256,
            "payload_count": len(libraries),
            "present_payload_count": len(payloads),
            "missing_payloads": missing,
            "payloads_all_present": not missing,
            "payloads_all_hashed": hash_transport_payloads and not missing,
            "payload_ledger_sha256": ledger_hash,
            "payload_ledger_algorithm": (
                "sha256(canonical-json ordered cross_sections.xml rows with "
                "relative path, material, type, size, and payload sha256)"
                if hash_transport_payloads and not missing
                else None
            ),
        },
    }


def build_activation_handoff(
    *,
    geometry: Mapping[str, Any],
    volumes: Sequence[Mapping[str, Any]],
    scalar_flux: Mapping[str, Any] | None,
    activation_data: Mapping[str, Any],
    physical_source_rate_per_s: float,
    source_mesh: Mapping[str, Any] | None = None,
    activation_domains: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build and validate a neutral producer-to-activation handoff.

    A not-yet-qualified geometry or field yields an honest blocked handoff,
    while malformed provenance raises immediately.  This permits software and
    data integration before transport without promoting old physical results.
    """
    rows = [dict(row) for row in volumes]
    material_counts = dict(
        Counter(str(row.get("material_tag")) for row in rows)
    )
    geometry_gates_pass = geometry.get("native_geometry_gate") == "PASS" and (
        geometry.get("openmc_navigation_gate") == "PASS"
    )
    domains = [dict(row) for row in activation_domains]
    field_ready = (
        scalar_flux is not None and scalar_flux.get("status") == "PASS"
    )
    binding_ready = source_mesh is not None and bool(domains)
    status = (
        "READY_FOR_DPA_ACTIVATION_QUALIFICATION"
        if geometry_gates_pass and field_ready and binding_ready
        else "BLOCKED_PENDING_ACCEPTED_GEOMETRY_AND_MEDIUM_FIELD"
    )
    handoff = {
        "schema": ACTIVATION_HANDOFF_SCHEMA,
        "status": status,
        "execution_owner": ACTIVATION_EXECUTION_OWNER,
        "geometry": dict(geometry),
        "volume_inventory": rows,
        "material_counts": material_counts,
        "source_mesh": dict(source_mesh) if source_mesh is not None else None,
        "activation_domains": domains,
        "activation_input": {
            "observable": "volume_scalar_flux",
            "estimator": "track_length",
            "normalization": "per_source_history",
            "physical_source_rate_per_s": float(physical_source_rate_per_s),
            "scalar_flux": (
                dict(scalar_flux) if scalar_flux is not None else None
            ),
            "boundary_bank_used_for_activation": False,
        },
        "activation_data": dict(activation_data),
        "consumer": {
            "repository": "DPA_workflow",
            "accepted_operation": "bounded_activation_rebinding",
            "production_activation_authorized": False,
        },
        "old_geometry_activation_results": {
            "status": "REFERENCE_ONLY_REJECTED_GEOMETRY",
            "may_validate_software_path": True,
            "may_supply_final_physical_results": False,
        },
    }
    validate_activation_handoff(handoff)
    return handoff


def validate_activation_handoff(handoff: Mapping[str, Any]) -> None:
    """Fail closed unless geometry, field, data, and ownership are explicit."""
    if handoff.get("schema") != ACTIVATION_HANDOFF_SCHEMA:
        raise ActivationHandoffError("unsupported activation handoff schema")
    if handoff.get("execution_owner") != ACTIVATION_EXECUTION_OWNER:
        raise ActivationHandoffError(
            "activation execution owner must be DPA_workflow"
        )

    geometry = handoff.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ActivationHandoffError("missing accepted-geometry provenance")
    if geometry.get("reference_manifest_sha256") != (
        DIRECT90_REFERENCE_MANIFEST_SHA256
    ):
        raise ActivationHandoffError("wrong direct-90 reference manifest")
    if (
        float(geometry.get("modeled_toroidal_extent_degrees", math.nan))
        != 90.0
    ):
        raise ActivationHandoffError(
            "activation handoff must bind the 90-degree model"
        )
    for key in ("raw_h5m_sha256", "canonical_geometry_fingerprint"):
        if not _is_sha256(geometry.get(key)):
            raise ActivationHandoffError(f"missing or invalid geometry {key}")
    if not _is_sha256(geometry.get("native_volume_audit_receipt_sha256")):
        raise ActivationHandoffError(
            "native volume audit receipt hash is invalid"
        )
    if geometry.get("magnet_representation") != (
        "continuous_30_cm_magnet_envelope"
    ):
        raise ActivationHandoffError("wrong global magnet representation")
    wrapper = geometry.get("transport_periodic_wrapper")
    if not isinstance(wrapper, Mapping) or (
        wrapper.get("outside_h5m") is not True
        or wrapper.get("periodic_planes_paired") is not True
        or int(wrapper.get("n_field_periods", -1)) != 4
        or float(wrapper.get("extent_degrees", math.nan)) != 90.0
    ):
        raise ActivationHandoffError(
            "external 90-degree periodic wrapper semantics are invalid"
        )

    volumes = handoff.get("volume_inventory")
    if not isinstance(volumes, list) or len(volumes) != 9:
        raise ActivationHandoffError(
            "direct-90 volume inventory must contain 9 rows"
        )
    ids = [int(row.get("volume_id", -1)) for row in volumes]
    if any(identifier <= 0 for identifier in ids) or len(set(ids)) != len(ids):
        raise ActivationHandoffError(
            "volume IDs must be unique positive integers"
        )
    for row in volumes:
        _positive_finite(row.get("volume_cm3"), "native audited volume")
        if row.get("volume_audit_receipt_sha256") != geometry.get(
            "native_volume_audit_receipt_sha256"
        ):
            raise ActivationHandoffError(
                "native volume row is not bound to the volume audit"
            )
    if handoff.get("material_counts") != EXPECTED_DIRECT90_MATERIAL_COUNTS:
        raise ActivationHandoffError(
            "direct-90 material multiplicities do not match"
        )

    activation_input = handoff.get("activation_input")
    if not isinstance(activation_input, Mapping):
        raise ActivationHandoffError("missing activation input")
    if activation_input.get("observable") != "volume_scalar_flux":
        raise ActivationHandoffError("activation requires volume scalar flux")
    if activation_input.get("estimator") != "track_length":
        raise ActivationHandoffError(
            "activation scalar flux requires track-length estimator"
        )
    if activation_input.get("normalization") != "per_source_history":
        raise ActivationHandoffError("unsupported scalar-flux normalization")
    if activation_input.get("boundary_bank_used_for_activation") is not False:
        raise ActivationHandoffError(
            "boundary current cannot drive activation"
        )
    rate = float(activation_input.get("physical_source_rate_per_s", math.nan))
    if not math.isfinite(rate) or rate <= 0.0:
        raise ActivationHandoffError(
            "physical source rate must be finite and positive"
        )

    data = handoff.get("activation_data")
    if not isinstance(data, Mapping):
        raise ActivationHandoffError("missing activation-data provenance")
    expected = {
        "chain": ENDFB80_FAST_CHAIN_SHA256,
        "transport_catalog": FULL_TRANSPORT_CATALOG_SHA256,
    }
    for key, digest in expected.items():
        row = data.get(key)
        if not isinstance(row, Mapping) or row.get("sha256") != digest:
            raise ActivationHandoffError(f"unqualified {key} hash")
        if row.get("qualified") is not True:
            raise ActivationHandoffError(f"{key} is not qualified")
    catalog = data["transport_catalog"]
    if (
        int(catalog.get("payload_count", -1)) != 728
        or int(catalog.get("present_payload_count", -1)) != 728
        or catalog.get("payloads_all_present") is not True
        or catalog.get("payloads_all_hashed") is not True
        or catalog.get("payload_ledger_sha256")
        != FULL_TRANSPORT_PAYLOAD_LEDGER_SHA256
    ):
        raise ActivationHandoffError(
            "transport payload hash ledger is incomplete"
        )

    consumer = handoff.get("consumer")
    if not isinstance(consumer, Mapping) or consumer.get("repository") != (
        "DPA_workflow"
    ):
        raise ActivationHandoffError("wrong activation consumer")
    if consumer.get("production_activation_authorized") is not False:
        raise ActivationHandoffError("production activation is not authorized")

    source_mesh = handoff.get("source_mesh")
    domains = handoff.get("activation_domains")
    if not isinstance(domains, list):
        raise ActivationHandoffError("activation_domains must be a list")
    if source_mesh is not None:
        if not isinstance(source_mesh, Mapping):
            raise ActivationHandoffError("source_mesh must be a mapping")
        if source_mesh.get("sha256") != DIRECT90_SOURCE_MESH_SHA256:
            raise ActivationHandoffError("source mesh hash is invalid")
        if source_mesh.get("strengths_sha256") != DIRECT90_STRENGTHS_SHA256:
            raise ActivationHandoffError("source strengths hash is invalid")
        if source_mesh.get("status") != "PASS":
            raise ActivationHandoffError("source mesh is not qualified")
        if source_mesh.get("domain_element_order_audit_status") != "PASS":
            raise ActivationHandoffError(
                "source domain and element-order audit has not passed"
            )
        if not _is_sha256(source_mesh.get("audit_receipt_sha256")):
            raise ActivationHandoffError(
                "source audit receipt hash is invalid"
            )
        if (
            float(source_mesh.get("modeled_toroidal_extent_degrees", math.nan))
            != 90.0
        ):
            raise ActivationHandoffError(
                "source mesh extent is not 90 degrees"
            )
        if source_mesh.get("source_rate_scope") != "modeled_90_degree_period":
            raise ActivationHandoffError("source-rate scope is ambiguous")
        mesh_rate = _positive_finite(
            source_mesh.get("strength_sum_per_s"),
            "source mesh strength sum",
        )
        if not math.isclose(
            mesh_rate,
            DIRECT90_MODELED_SOURCE_RATE_PER_S,
            rel_tol=1.0e-14,
            abs_tol=0.0,
        ):
            raise ActivationHandoffError("source mesh strength sum is wrong")
        if not math.isclose(rate, mesh_rate, rel_tol=1.0e-14, abs_tol=0.0):
            raise ActivationHandoffError(
                "physical source rate does not close to source mesh"
            )
        if source_mesh.get("geometry_fingerprint") != geometry.get(
            "canonical_geometry_fingerprint"
        ):
            raise ActivationHandoffError("source mesh is not geometry-bound")

    domain_ids: set[str] = set()
    volume_rows = {int(row["volume_id"]): row for row in volumes}
    volume_ids = set(volume_rows)
    for domain in domains:
        _validate_activation_domain(
            domain,
            volume_rows=volume_rows,
            volume_ids=volume_ids,
            geometry=geometry,
        )
        domain_id = str(domain["domain_id"])
        if domain_id in domain_ids:
            raise ActivationHandoffError(
                "activation domain IDs must be unique"
            )
        domain_ids.add(domain_id)

    ready = (
        geometry.get("native_geometry_gate") == "PASS"
        and geometry.get("openmc_navigation_gate") == "PASS"
        and isinstance(activation_input.get("scalar_flux"), Mapping)
        and activation_input["scalar_flux"].get("status") == "PASS"
        and source_mesh is not None
        and bool(domains)
    )
    scalar_flux = activation_input.get("scalar_flux")
    if scalar_flux is not None:
        if not isinstance(scalar_flux, Mapping):
            raise ActivationHandoffError(
                "scalar flux artifact must be a mapping"
            )
        if not domains or source_mesh is None:
            raise ActivationHandoffError(
                "scalar flux cannot precede domain and source-mesh binding"
            )
        _validate_scalar_flux_artifact(
            scalar_flux,
            domains_by_id={str(row["domain_id"]): row for row in domains},
            physical_source_rate_per_s=rate,
            geometry=geometry,
            source_mesh=source_mesh,
        )
    if ready:
        whole = [
            row
            for row in domains
            if row.get("domain_kind") == "WHOLE_MAGNET_ENVELOPE"
        ]
        if len(whole) != 1:
            raise ActivationHandoffError(
                "ready direct-90 activation requires one whole magnet envelope"
            )
    expected_status = (
        "READY_FOR_DPA_ACTIVATION_QUALIFICATION"
        if ready
        else "BLOCKED_PENDING_ACCEPTED_GEOMETRY_AND_MEDIUM_FIELD"
    )
    if handoff.get("status") != expected_status:
        raise ActivationHandoffError(
            "activation readiness status is inconsistent"
        )


def _positive_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ActivationHandoffError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ActivationHandoffError(f"{name} must be positive and finite")
    return result


def _validate_activation_domain(
    domain: Mapping[str, Any],
    *,
    volume_rows: Mapping[int, Mapping[str, Any]],
    volume_ids: set[int],
    geometry: Mapping[str, Any],
) -> None:
    if domain.get("domain_kind") not in {
        "WHOLE_MAGNET_ENVELOPE",
        "QUALIFIED_MATERIAL_PATCH",
    }:
        raise ActivationHandoffError("unsupported activation domain kind")
    if not str(domain.get("domain_id", "")):
        raise ActivationHandoffError("activation domain ID is missing")
    dagmc_id = int(domain.get("dagmc_volume_id", -1))
    if dagmc_id not in volume_ids:
        raise ActivationHandoffError(
            "activation domain DAGMC volume is unknown"
        )
    if int(domain.get("openmc_cell_id", -1)) <= 0:
        raise ActivationHandoffError("actual OpenMC cell ID is missing")
    if domain.get("cell_id_mapping_method") != "openmc_geometry_introspection":
        raise ActivationHandoffError(
            "OpenMC cell ID was not independently discovered"
        )
    if domain.get("dagmc_id_inference_used") is not False:
        raise ActivationHandoffError(
            "DAGMC volume ID cannot be inferred as OpenMC cell ID"
        )
    if domain.get("material_tag") != "magnet_envelope":
        raise ActivationHandoffError(
            "whole magnet domain has wrong material tag"
        )
    if volume_rows[dagmc_id].get("material_tag") != domain.get("material_tag"):
        raise ActivationHandoffError(
            "activation domain material disagrees with DAGMC inventory"
        )
    if int(domain.get("openmc_material_id", -1)) <= 0:
        raise ActivationHandoffError("actual OpenMC material ID is missing")
    domain_volume = _positive_finite(
        domain.get("volume_cm3"), "activation domain volume"
    )
    native_volume = float(volume_rows[dagmc_id]["volume_cm3"])
    if not math.isclose(
        domain_volume, native_volume, rel_tol=1.0e-12, abs_tol=1.0e-9
    ):
        raise ActivationHandoffError(
            "activation domain volume disagrees with native audited volume"
        )
    _positive_finite(domain.get("density_g_cm3"), "activation domain density")
    if not _is_sha256(domain.get("isotopic_composition_sha256")):
        raise ActivationHandoffError("isotopic composition hash is invalid")
    if not _is_sha256(domain.get("id_mapping_receipt_sha256")):
        raise ActivationHandoffError("ID mapping receipt hash is invalid")
    if domain.get("raw_h5m_sha256") != geometry.get("raw_h5m_sha256"):
        raise ActivationHandoffError("activation domain is not H5M-bound")
    if domain.get("geometry_fingerprint") != geometry.get(
        "canonical_geometry_fingerprint"
    ):
        raise ActivationHandoffError("activation domain is not geometry-bound")
    surfaces = domain.get("boundary_surface_ids")
    if not isinstance(surfaces, list) or not surfaces:
        raise ActivationHandoffError("activation domain boundary is empty")
    if len({int(value) for value in surfaces}) != len(surfaces):
        raise ActivationHandoffError(
            "activation boundary surface IDs are duplicated"
        )
    if domain.get("boundary_closure") != "EUCLIDEAN_CLOSED":
        raise ActivationHandoffError(
            "DAGMC magnet volume must be physically Euclidean-closed"
        )


def _validate_scalar_flux_artifact(
    scalar_flux: Mapping[str, Any],
    *,
    domains_by_id: Mapping[str, Mapping[str, Any]],
    physical_source_rate_per_s: float,
    geometry: Mapping[str, Any],
    source_mesh: Mapping[str, Any],
) -> None:
    for key in ("sha256", "statepoint_sha256", "tally_definition_sha256"):
        if not _is_sha256(scalar_flux.get(key)):
            raise ActivationHandoffError(f"scalar flux {key} is invalid")
    if scalar_flux.get("raw_h5m_sha256") != geometry.get("raw_h5m_sha256"):
        raise ActivationHandoffError("scalar flux is not H5M-bound")
    if scalar_flux.get("geometry_fingerprint") != geometry.get(
        "canonical_geometry_fingerprint"
    ):
        raise ActivationHandoffError("scalar flux is not geometry-bound")
    if scalar_flux.get("source_mesh_sha256") != source_mesh.get("sha256"):
        raise ActivationHandoffError("scalar flux is not source-mesh-bound")
    if scalar_flux.get("openmc_version") != "0.16.0":
        raise ActivationHandoffError("scalar flux requires OpenMC 0.16.0")
    if (
        scalar_flux.get("particle") != "neutron"
        or scalar_flux.get("score") != "flux"
        or scalar_flux.get("estimator") != "tracklength"
        or scalar_flux.get("filters") != ["cell", "energy"]
    ):
        raise ActivationHandoffError(
            "scalar flux tally semantics are not neutron tracklength cell-energy flux"
        )
    if int(scalar_flux.get("source_histories", -1)) <= 0:
        raise ActivationHandoffError(
            "scalar flux source history count is invalid"
        )
    if not math.isclose(
        float(scalar_flux.get("physical_source_rate_per_s", math.nan)),
        physical_source_rate_per_s,
        rel_tol=1.0e-14,
        abs_tol=0.0,
    ):
        raise ActivationHandoffError("scalar flux source rate is inconsistent")
    rows = scalar_flux.get("domains")
    row_ids = (
        [str(row.get("domain_id")) for row in rows]
        if isinstance(rows, list)
        else []
    )
    if (
        not isinstance(rows, list)
        or len(rows) != len(domains_by_id)
        or len(set(row_ids)) != len(row_ids)
        or set(row_ids) != set(domains_by_id)
    ):
        raise ActivationHandoffError(
            "scalar flux domains do not match activation domains"
        )
    for row in rows:
        volume = _positive_finite(row.get("volume_cm3"), "scalar flux volume")
        domain_volume = float(
            domains_by_id[str(row["domain_id"])]["volume_cm3"]
        )
        if not math.isclose(
            volume, domain_volume, rel_tol=1.0e-12, abs_tol=1.0e-9
        ):
            raise ActivationHandoffError(
                "scalar flux volume disagrees with activation domain"
            )
        edges = [float(value) for value in row.get("energy_edges_eV", ())]
        means = [
            float(value)
            for value in row.get("track_length_mean_cm_per_source", ())
        ]
        stddev = [
            float(value)
            for value in row.get("track_length_std_dev_cm_per_source", ())
        ]
        physical = [
            float(value)
            for value in row.get("physical_scalar_flux_per_cm2_s", ())
        ]
        if len(edges) != len(means) + 1 or len(means) == 0:
            raise ActivationHandoffError(
                "scalar flux energy-group shape is invalid"
            )
        if len(stddev) != len(means) or len(physical) != len(means):
            raise ActivationHandoffError("scalar flux score shapes disagree")
        values = edges + means + stddev + physical
        if not all(math.isfinite(value) for value in values):
            raise ActivationHandoffError("scalar flux scores must be finite")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ActivationHandoffError(
                "scalar flux energy edges are not increasing"
            )
        if any(value < 0.0 for value in means + stddev + physical):
            raise ActivationHandoffError(
                "scalar flux scores cannot be negative"
            )
        expected = [
            value / volume * physical_source_rate_per_s for value in means
        ]
        if any(
            not math.isclose(actual, target, rel_tol=1.0e-12, abs_tol=0.0)
            for actual, target in zip(physical, expected)
        ):
            raise ActivationHandoffError(
                "scalar flux normalization closure failed"
            )


def write_activation_handoff(
    path: str | Path, handoff: Mapping[str, Any]
) -> None:
    """Validate and write a deterministic JSON handoff."""
    validate_activation_handoff(handoff)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(handoff, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
