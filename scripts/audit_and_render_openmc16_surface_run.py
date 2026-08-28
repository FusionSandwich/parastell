"""Audit, localize, and render one bounded OpenMC 0.16 surface run.

The scientific classification is produced by ``audit_openmc16_surface_run``
from the immutable H5M, model, statepoint, native source bank, terminal log,
and root-accepted magnet inventory.  This script only exports the resulting
correlated records and reproducible visual evidence; it cannot upgrade a
non-complete bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from parastell.boundary_phase_space_figures import write_phase_space_figures
from parastell.openmc16_export import (
    StrictSurfaceRunArtifacts,
    _extract_envelopes_from_h5m,
    audit_openmc16_surface_run,
)
from parastell.surface_source_localization import localize_surface_crossings
from parastell.surface_source_phase_space import read_openmc16_surface_sources


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_envelope_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        payload.get("envelope_requests")
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("envelope request JSON must contain a nonempty list")
    return [dict(row) for row in rows]


def _unicode_safe(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "O":
        return array.astype(str)
    return array


def _facet_catalog(envelopes) -> dict[str, np.ndarray]:
    rows = [envelope.facet_metadata() for envelope in envelopes]
    if not rows:
        raise ValueError("no DAGMC envelopes were extracted")
    output = {
        "facet_id": np.concatenate(
            [_unicode_safe(row["canonical_facet_id"]) for row in rows]
        ),
        "surface_id": np.concatenate(
            [np.asarray(row["surface_id"], dtype=np.int64) for row in rows]
        ),
        "vertices_global_cm": np.concatenate(
            [
                np.asarray(row["triangle_vertices_global_cm"], dtype=float)
                for row in rows
            ]
        ),
        "outward_normal_global": np.concatenate(
            [
                np.asarray(row["outward_normal_global"], dtype=float)
                for row in rows
            ]
        ),
    }
    if len(output["facet_id"]) != len(set(output["facet_id"].tolist())):
        raise ValueError("combined canonical facet IDs are not unique")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dagmc_h5m", type=Path)
    parser.add_argument("model_xml", type=Path)
    parser.add_argument("statepoint", type=Path)
    parser.add_argument("terminal_log", type=Path)
    parser.add_argument("accepted_magnet_inventory", type=Path)
    parser.add_argument("root_acceptance_receipt", type=Path)
    parser.add_argument("expected_root_acceptance_receipt_sha256")
    parser.add_argument("envelope_requests", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("surface_sources", nargs="+", type=Path)
    parser.add_argument("--geometry-label", required=True)
    args = parser.parse_args()

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    requests = _load_envelope_requests(
        args.envelope_requests.resolve(strict=True)
    )
    artifacts = StrictSurfaceRunArtifacts(
        dagmc_path=args.dagmc_h5m.resolve(strict=True),
        model_xml_path=args.model_xml.resolve(strict=True),
        statepoint_path=args.statepoint.resolve(strict=True),
        terminal_log_path=args.terminal_log.resolve(strict=True),
        surface_source_paths=[
            path.resolve(strict=True) for path in args.surface_sources
        ],
        accepted_magnet_inventory_path=args.accepted_magnet_inventory.resolve(
            strict=True
        ),
        root_acceptance_receipt_path=args.root_acceptance_receipt.resolve(
            strict=True
        ),
        expected_root_acceptance_receipt_sha256=(
            args.expected_root_acceptance_receipt_sha256.lower()
        ),
    )
    strict = audit_openmc16_surface_run(
        artifacts,
        envelope_requests=requests,
        required_particles=("neutron", "photon"),
    )
    strict_path = output / "STRICT_SURFACE_RUN_AUDIT.json"
    _write_json_create_only(strict_path, strict)
    if strict["classification"] != "COMPLETE_CROSSING_BANK":
        raise RuntimeError(
            "strict audit did not classify the bank COMPLETE_CROSSING_BANK"
        )

    histories = int(strict["model"]["source_histories"])
    history_binding = {
        "kind": "fixed_source_run",
        "run_id": _sha256(strict_path),
        "source_histories": histories,
        "settings_payload_path": str(Path(artifacts.model_xml_path)),
        "settings_payload_sha256": _sha256(Path(artifacts.model_xml_path)),
        "statepoint_path": str(Path(artifacts.statepoint_path)),
        "statepoint_sha256": _sha256(Path(artifacts.statepoint_path)),
    }
    phase_space, phase_manifest = read_openmc16_surface_sources(
        artifacts.surface_source_paths,
        source_histories=histories,
        history_binding=history_binding,
        requested_surface_ids=strict["dagmc"]["surface_ids"],
    )
    envelopes = _extract_envelopes_from_h5m(
        Path(artifacts.dagmc_path), requests
    )
    catalog = _facet_catalog(envelopes)
    localized, localization_audit = localize_surface_crossings(
        phase_space,
        {**catalog, "normal_source": "dagmc_forward_reverse_topology"},
    )

    localized_path = output / "LOCALIZED_SURFACE_PHASE_SPACE.npz"
    np.savez_compressed(
        localized_path,
        **{name: _unicode_safe(values) for name, values in localized.items()},
    )
    catalog_path = output / "DAGMC_FACET_CATALOG.npz"
    np.savez_compressed(catalog_path, **catalog)
    topology_path = output / "LOCALIZATION_TOPOLOGY_MANIFEST.json"
    _write_json_create_only(
        topology_path,
        {
            **strict["localization_topology_binding"],
            "strict_surface_run_audit_sha256": _sha256(strict_path),
            "localization_audit": localization_audit,
        },
    )

    bank_rows = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in strict["surface_source_files"]
    ]
    render_audit = {
        "schema": "parastell.openmc16_surface_run_audit/v1.0.0",
        "status": "COMPLETE_CROSSING_BANK",
        "geometry_label": args.geometry_label,
        "geometry": {
            "path": str(Path(artifacts.dagmc_path)),
            "sha256": strict["dagmc"]["sha256"],
        },
        "source_banks": bank_rows,
        "phase_space_manifest": phase_manifest,
        "localized_records": {
            "path": str(localized_path),
            "sha256": _sha256(localized_path),
        },
        "facet_catalog": {
            "path": str(catalog_path),
            "sha256": _sha256(catalog_path),
        },
        "topology_manifest": {
            "path": str(topology_path),
            "sha256": _sha256(topology_path),
        },
        "strict_surface_run_audit": {
            "path": str(strict_path),
            "sha256": _sha256(strict_path),
        },
    }
    render_audit["localization_topology_binding"] = {
        "geometry_sha256": strict["dagmc"]["sha256"],
        "source_bank_sha256s": [row["sha256"] for row in bank_rows],
        "source_histories": histories,
        "settings_payload_sha256": _sha256(Path(artifacts.model_xml_path)),
        "statepoint_sha256": _sha256(Path(artifacts.statepoint_path)),
        "localized_records_sha256": _sha256(localized_path),
        "facet_catalog_sha256": _sha256(catalog_path),
        "topology_manifest_sha256": _sha256(topology_path),
    }
    render_audit_path = output / "VERIFIED_FIGURE_INPUT_AUDIT.json"
    _write_json_create_only(render_audit_path, render_audit)
    figure_manifest = write_phase_space_figures(
        render_audit_path,
        output / "figures",
        expected_run_audit_sha256=_sha256(render_audit_path),
        status="BOUNDED_TEST_ONLY",
    )
    final_receipt = {
        "schema": "parastell.surface_phase_space_evidence_bundle/v1.0.0",
        "status": "PASS",
        "claim": "BOUNDED_TEST_ONLY",
        "strict_surface_run_audit": {
            "path": str(strict_path),
            "sha256": _sha256(strict_path),
        },
        "verified_figure_input_audit": {
            "path": str(render_audit_path),
            "sha256": _sha256(render_audit_path),
        },
        "figure_manifest": figure_manifest,
        "native_phase_limitations": strict["native_phase_limitations"],
    }
    receipt_path = output / "SURFACE_PHASE_SPACE_EVIDENCE_RECEIPT.json"
    _write_json_create_only(receipt_path, final_receipt)
    print(receipt_path)


if __name__ == "__main__":
    main()
