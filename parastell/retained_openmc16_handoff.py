"""Re-export a hash-bound retained OpenMC 0.16 crossing bank as v2.2.

This module does not rerun transport.  It is deliberately strict about the
geometry, statepoint, and surface-source identity so a newer facet-complete
producer contract cannot be attached to an unrelated historical calculation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .coil_frame import CentrelineFrame, parallel_transport_frame
from .dagmc_envelope import DagmcEnvelope, extract_closed_envelope
from .openmc16_export import export_openmc16_handoff

V21_SCHEMA = "parastell.magnet_boundary_source/v2.1.0"
V22_SCHEMA = "parastell.magnet_boundary_source/v2.2.0"
GEOMETRY_SCHEMA = "parastell.magnet_geometry_interchange/v1.0.0"


def sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_handoff_manifest(path: str | Path) -> dict[str, Any]:
    """Load only the embedded manifest from a ParaStell handoff."""
    with h5py.File(path, "r") as handle:
        if "manifest_json" not in handle:
            raise ValueError(f"handoff has no manifest_json: {path}")
        return json.loads(handle["manifest_json"].asstr()[()])


def _matching_recipe_magnet(
    recipe: Mapping[str, Any], magnet_id: str
) -> Mapping[str, Any]:
    magnets = [
        value
        for value in recipe.get("magnets", [])
        if value.get("magnet_id") == magnet_id
    ]
    if len(magnets) != 1:
        raise ValueError(
            f"geometry recipe must contain exactly one {magnet_id!r} entry"
        )
    return magnets[0]


def reconstruct_hash_bound_frame(
    geometry_recipe_path: str | Path,
    dagmc_path: str | Path,
    *,
    magnet_id: str,
    expected_geometry_fingerprint: str | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any], CentrelineFrame]:
    """Validate the neutral recipe and reproduce its engineering frame."""
    recipe = json.loads(Path(geometry_recipe_path).read_text(encoding="utf-8"))
    if recipe.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError("unsupported geometry-interchange schema")
    fingerprint = str(recipe.get("geometry_fingerprint", ""))
    if expected_geometry_fingerprint not in (None, fingerprint):
        raise ValueError("geometry fingerprint mismatch")
    magnet = _matching_recipe_magnet(recipe, magnet_id)
    artifact = magnet.get("artifacts", {}).get("outer_h5m", {})
    if artifact.get("status") != "available":
        raise ValueError("geometry recipe does not bind an available H5M")
    observed_h5m = sha256(dagmc_path)
    if observed_h5m != str(artifact.get("sha256", "")).lower():
        raise ValueError("DAGMC H5M hash does not match the geometry recipe")
    points = magnet.get("centreline", {}).get("points_global_cm")
    if not points:
        raise ValueError("geometry recipe has no centreline points")
    if not magnet.get("closure", {}).get("represented_segment_closed"):
        raise ValueError(
            "retained production frame must represent a closed coil"
        )
    frame = parallel_transport_frame(points, closed=True)
    samples = magnet.get("frame", {}).get("samples", [])
    if len(samples) != len(frame.points_cm):
        raise ValueError("frame sample count does not match the centreline")
    expected_tangent = np.asarray(
        [sample["tangent_global"] for sample in samples]
    )
    expected_width = np.asarray(
        [sample["width_direction_global"] for sample in samples]
    )
    expected_normal = np.asarray(
        [sample["normal_direction_global"] for sample in samples]
    )
    if not (
        np.allclose(frame.tangents, expected_tangent, rtol=0.0, atol=1.0e-12)
        and np.allclose(
            frame.radial_directions,
            expected_width,
            rtol=0.0,
            atol=1.0e-12,
        )
        and np.allclose(
            frame.transverse_directions,
            expected_normal,
            rtol=0.0,
            atol=1.0e-12,
        )
        and np.isclose(
            frame.closure_twist_rad,
            float(magnet["frame"]["closure_twist_rad"]),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            "reconstructed frame does not match the frozen recipe"
        )
    return recipe, magnet, frame


def _volume_centroid(dagmc_path: str | Path, volume_id: int) -> np.ndarray:
    import pydagmc

    model = pydagmc.Model(str(dagmc_path))
    if volume_id not in model.volumes_by_id:
        raise ValueError(f"DAGMC volume {volume_id} does not exist")
    triangles = np.concatenate(
        [
            np.asarray(surface.triangle_coords, dtype=float).reshape(
                (-1, 3, 3)
            )
            for surface in model.volumes_by_id[volume_id].surfaces
        ]
    )
    return np.mean(triangles.reshape((-1, 3)), axis=0)


def _assert_envelope_matches_v21(
    envelope: DagmcEnvelope,
    retained_manifest: Mapping[str, Any],
) -> None:
    old = retained_manifest["envelope"]
    current = envelope.envelope
    if int(old["dagmc_volume_id"]) != current.dagmc_volume_id:
        raise ValueError("DAGMC volume changed during retained-bank re-export")
    if (
        str(old["dagmc_geometry_sha256"]).lower()
        != current.dagmc_geometry_sha256
    ):
        raise ValueError(
            "DAGMC geometry hash changed during retained-bank re-export"
        )
    if old["envelope_id"] != current.envelope_id:
        raise ValueError(
            "envelope identity changed during retained-bank re-export"
        )
    if old["magnet_component"] != current.magnet_component:
        raise ValueError(
            "magnet identity changed during retained-bank re-export"
        )
    old_surfaces = {
        int(value["surface_id"]): value for value in old["surfaces"]
    }
    new_surfaces = {value.surface_id: value for value in current.surfaces}
    if old_surfaces.keys() != new_surfaces.keys():
        raise ValueError(
            "DAGMC surface set changed during retained-bank re-export"
        )
    for surface_id, earlier in old_surfaces.items():
        later = new_surfaces[surface_id]
        exact_scalars = {
            "role": later.role,
            "openmc_normal_sign": later.openmc_normal_sign,
        }
        for name, value in exact_scalars.items():
            if earlier[name] != value:
                raise ValueError(
                    f"surface {surface_id} {name} changed during re-export"
                )
        comparisons: Sequence[tuple[str, Any]] = (
            ("area_cm2", later.area_cm2),
            ("centroid_global_cm", later.centroid_global_cm),
            ("outward_normal_global", later.outward_normal_global),
            ("toroidal_direction_global", later.toroidal_direction_global),
            ("poloidal_direction_global", later.poloidal_direction_global),
            ("u_edges_cm", later.u_edges_cm),
            ("v_edges_cm", later.v_edges_cm),
            ("vector_area_global_cm2", later.vector_area_global_cm2),
        )
        for name, value in comparisons:
            if not np.allclose(
                np.asarray(earlier[name], dtype=float),
                np.asarray(value, dtype=float),
                rtol=1.0e-12,
                atol=1.0e-9,
            ):
                raise ValueError(
                    f"surface {surface_id} {name} changed during re-export"
                )


def _transport_version_from_statepoint(path: str | Path) -> str:
    import openmc

    with openmc.StatePoint(path) as statepoint:
        version = tuple(int(value) for value in statepoint.version)
    if version != (0, 16, 0):
        raise ValueError(
            "retained-bank exporter requires an OpenMC 0.16.0 statepoint; "
            f"observed {version}"
        )
    return ".".join(str(value) for value in version)


def reexport_retained_handoff_v22(
    output_path: str | Path,
    *,
    geometry_recipe_path: str | Path,
    dagmc_path: str | Path,
    retained_v21_path: str | Path,
    statepoint_path: str | Path,
    surface_source_path: str | Path,
    exporter_commit: str,
) -> dict[str, Any]:
    """Produce one facet-complete v2.2 bank from retained same-run files."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    retained = load_handoff_manifest(retained_v21_path)
    if retained.get("schema") != V21_SCHEMA:
        raise ValueError("retained input is not a v2.1 boundary handoff")
    provenance = retained["provenance"]
    expected = {
        Path(dagmc_path): str(provenance["dagmc_geometry_sha256"]).lower(),
        Path(statepoint_path): str(
            provenance["openmc_statepoint_sha256"]
        ).lower(),
        Path(surface_source_path): str(
            provenance["surface_source_sha256"][0]
        ).lower(),
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(
                f"retained same-run hash mismatch for {path.name}"
            )
    magnet_id = str(retained["envelope"]["magnet_component"])
    fingerprint = None
    recipe, _magnet, frame = reconstruct_hash_bound_frame(
        geometry_recipe_path,
        dagmc_path,
        magnet_id=magnet_id,
        expected_geometry_fingerprint=fingerprint,
    )
    volume_id = int(retained["envelope"]["dagmc_volume_id"])
    sampled = frame.sample(_volume_centroid(dagmc_path, volume_id))
    envelope = extract_closed_envelope(
        dagmc_path,
        volume_id,
        envelope_id=str(retained["envelope"]["envelope_id"]),
        magnet_id=magnet_id,
        plasma_direction_global=-np.asarray(sampled["centreline_radial"]),
        toroidal_direction_global=sampled["centreline_tangent"],
        poloidal_direction_global=sampled["centreline_transverse"],
        spatial_bins=(4, 4),
        centreline_frame=frame,
    )
    if (
        envelope.envelope.metadata.get("canonical_geometry_fingerprint")
        != recipe["geometry_fingerprint"]
    ):
        raise ValueError(
            "canonical DAGMC fingerprint does not match the recipe"
        )
    _assert_envelope_matches_v21(envelope, retained)
    completeness = retained["bank_metadata"]["surface_bank_completeness"]
    transport_version = _transport_version_from_statepoint(statepoint_path)
    result = export_openmc16_handoff(
        output,
        statepoint_path=statepoint_path,
        surface_source_paths=[surface_source_path],
        envelope=envelope,
        histories=int(provenance["histories"]),
        energy_edges_by_particle={
            "neutron": retained["energy_axes"]["neutron_energy_edges_eV"],
            "photon": retained["energy_axes"]["photon_energy_edges_eV"],
        },
        physical_source_rate_per_s=retained["normalization"].get(
            "physical_source_rate_per_s"
        ),
        parastell_commit=str(provenance["parastell_commit"]),
        source_definition_sha256=str(provenance["source_definition_sha256"]),
        surface_source_max_particles=completeness.get(
            "max_particles_per_file"
        ),
        surface_source_max_files=completeness.get("max_source_files"),
        surface_source_sampling_applied=bool(
            completeness.get("sampling_applied", False)
        ),
        mpi_ranks=completeness.get("mpi_ranks"),
        centreline_frame=frame,
        transport_openmc_version=transport_version,
    )
    if result.get("schema") != V22_SCHEMA:
        raise ValueError("re-export did not produce the v2.2 schema")
    facet_status = result.get("facet_complete_boundary", {})
    if not all(
        facet_status.get(name) is True
        for name in (
            "record_fields_complete",
            "all_records_valid",
            "catalog_complete",
        )
    ):
        raise ValueError("re-exported handoff is not facet-complete")
    if int(result.get("record_count", -1)) != int(retained["record_count"]):
        raise ValueError("record count changed during retained-bank re-export")
    if not np.isclose(
        float(result["integrated_current"]),
        float(retained["integrated_current"]),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError(
            "integrated current changed during retained-bank re-export"
        )
    result["retained_reexport_receipt"] = {
        "classification": "FACET_COMPLETE_BOUNDARY_PASS",
        "exporter_commit": exporter_commit,
        "retained_v21_sha256": sha256(retained_v21_path),
        "geometry_recipe_sha256": sha256(geometry_recipe_path),
        "output_v22_sha256": sha256(output),
        "historical_transport_commit": provenance["parastell_commit"],
        "transport_openmc_version": transport_version,
        "canonical_geometry_fingerprint": recipe["geometry_fingerprint"],
    }
    return result
