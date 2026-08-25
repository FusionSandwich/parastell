"""Export OpenMC 0.16 closed-envelope records without tally conditioning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from .dagmc_envelope import DagmcEnvelope
from .magnet_boundary_envelope import build_correlated_bank
from .magnet_boundary_envelope import assign_adaptive_surface_patches
from .magnet_boundary_envelope import classify_crossing_bank
from .magnet_boundary_envelope import write_handoff
from .openmc16 import PDG_PARTICLES


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
    transport_weights = np.asarray(records["wgt"]).reshape(-1) / histories
    bank = build_correlated_bank(
        envelope.envelope,
        position_global_cm=positions,
        direction_global=directions,
        energy_eV=np.asarray(records["E"]).reshape(-1),
        raw_weight=transport_weights,
        particle=particles,
        surface_id=surface_ids,
        time_s=np.asarray(records["time"]).reshape(-1),
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
