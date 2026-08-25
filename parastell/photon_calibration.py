"""Exact-count calibration for mapped ParaStell boundary handoffs.

The canonical v2.1 boundary handoff stores a correlated partial-current bank.
Its record weights are already normalized per OpenMC source history, whereas
the number of stored records is an unweighted Poisson observation.  This
module deliberately keeps those two estimators separate: Garwood intervals
apply to raw counts, and weighted current and effective sample size are
reported without assigning them an exact Poisson confidence interval.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

from .magnet_boundary_envelope import (
    SCHEMA_URI,
    classify_crossing_bank,
    read_handoff,
)


SCHEMA = "parastell.photon_calibration/v1.0.0"
REQUIRED_HANDOFF_SCHEMA = "parastell.magnet_boundary_source/v2.1.0"
CROSSING_SENSES = ("incoming", "outgoing", "grazing")
EXPECTED_PARTICLES = ("neutron", "photon")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"created_utc", "evidence_sha256", "path"}
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def garwood_poisson_interval(
    count: int, *, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Return the exact central Garwood interval for a Poisson mean."""

    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError("Poisson count must be a nonnegative integer")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    alpha = 1.0 - confidence
    lower = 0.0 if count == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * count)
    upper = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (count + 1))
    return float(lower), float(upper)


def _exact_count_summary(
    weights: np.ndarray,
    *,
    histories: int,
    confidence_level: float,
) -> dict[str, Any]:
    count = int(len(weights))
    summed_weight = float(np.sum(weights, dtype=float))
    summed_weight_squared = float(np.dot(weights, weights))
    ess = (
        float(summed_weight * summed_weight / summed_weight_squared)
        if summed_weight_squared > 0.0
        else 0.0
    )
    lower, upper = garwood_poisson_interval(
        count, confidence_level=confidence_level
    )
    return {
        "raw_count": count,
        "summed_weight_per_source": summed_weight,
        "sum_weight_squared_per_source2": summed_weight_squared,
        "effective_sample_size": ess,
        "relative_weighted_counting_uncertainty": (
            math.sqrt(summed_weight_squared) / summed_weight
            if summed_weight > 0.0
            else None
        ),
        "raw_count_rate_per_source_history": count / histories,
        "garwood_count_mean_interval": {
            "confidence_level": float(confidence_level),
            "lower": lower,
            "upper": upper,
        },
        "garwood_raw_count_rate_per_source_history_interval": {
            "confidence_level": float(confidence_level),
            "lower": lower / histories,
            "upper": upper / histories,
        },
    }


def _require_complete_bank(
    manifest: Mapping[str, Any], *, record_count: int, path: Path
) -> dict[str, Any]:
    if SCHEMA_URI != REQUIRED_HANDOFF_SCHEMA:
        raise RuntimeError("the installed boundary producer is not v2.1")
    if manifest.get("schema") != REQUIRED_HANDOFF_SCHEMA:
        raise ValueError(
            f"{path} is not a current v2.1 mapped boundary handoff"
        )
    if int(manifest.get("record_count", -1)) != record_count:
        raise ValueError(f"{path} record count disagrees with its manifest")
    normalization = manifest.get("normalization", {})
    if normalization.get("basis") != "per source history":
        raise ValueError(f"{path} is not normalized per source history")
    completeness = manifest.get("bank_metadata", {}).get(
        "surface_bank_completeness"
    )
    if not isinstance(completeness, Mapping):
        raise ValueError(f"{path} has no surface-bank completeness evidence")
    required_accounting = {
        "stored_record_count",
        "selected_record_count",
        "max_particles_per_file",
        "max_source_files",
        "source_file_count",
    }
    if missing := required_accounting - set(completeness):
        raise ValueError(
            f"{path} bank accounting is missing {sorted(missing)}"
        )
    recomputed = classify_crossing_bank(
        stored_record_count=int(completeness["stored_record_count"]),
        selected_record_count=int(completeness["selected_record_count"]),
        max_particles_per_file=(
            None
            if completeness["max_particles_per_file"] is None
            else int(completeness["max_particles_per_file"])
        ),
        max_source_files=(
            None
            if completeness["max_source_files"] is None
            else int(completeness["max_source_files"])
        ),
        source_file_count=int(completeness["source_file_count"]),
        mpi_ranks=(
            None
            if completeness.get("mpi_ranks") is None
            else int(completeness["mpi_ranks"])
        ),
        sampling_applied=bool(completeness.get("sampling_applied", False)),
    )
    if completeness.get("classification") != recomputed["classification"]:
        raise ValueError(f"{path} bank classification is inconsistent")
    if recomputed["classification"] != "COMPLETE_CROSSING_BANK":
        raise ValueError(
            f"{path} is not a complete, uncapped, unsampled crossing bank"
        )
    if int(completeness.get("selected_record_count", -1)) != record_count:
        raise ValueError(
            f"{path} selected crossing count disagrees with its records"
        )
    return recomputed


def _validate_mapped_crossing_columns(
    columns: Mapping[str, np.ndarray], *, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "facet_id",
        "facet_index",
        "barycentric_coordinates",
        "facet_mapping_status",
        "outward_normal_global",
        "direction_global",
        "mu",
        "crossing_sense",
        "grazing",
    }
    if missing := required - set(columns):
        raise ValueError(
            f"{path} is not facet-mapped; missing fields {sorted(missing)}"
        )
    directions = np.asarray(columns["direction_global"], dtype=float)
    normals = np.asarray(columns["outward_normal_global"], dtype=float)
    if directions.shape != normals.shape or (
        directions.ndim != 2 or directions.shape[1] != 3
    ):
        raise ValueError(f"{path} direction/normal arrays are misaligned")
    if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(normals)):
        raise ValueError(f"{path} direction/normal arrays are not finite")
    direction_norms = np.linalg.norm(directions, axis=1)
    normal_norms = np.linalg.norm(normals, axis=1)
    if not np.allclose(direction_norms, 1.0, rtol=0.0, atol=1.0e-12) or (
        not np.allclose(normal_norms, 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(f"{path} directions and outward normals must be unit")
    derived_mu = np.sum(directions * normals, axis=1)
    if not np.allclose(
        derived_mu,
        np.asarray(columns["mu"], dtype=float),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError(f"{path} mu disagrees with the mapped outward normal")
    derived_sense = np.where(
        derived_mu > 1.0e-12,
        "outgoing",
        np.where(derived_mu < -1.0e-12, "incoming", "grazing"),
    )
    stored_sense = np.asarray(columns["crossing_sense"]).astype(str)
    if not np.array_equal(derived_sense, stored_sense):
        raise ValueError(
            f"{path} crossing sense disagrees with mapped normals"
        )
    derived_grazing = np.abs(derived_mu) <= 0.1
    stored_grazing = np.asarray(columns["grazing"]).astype(bool)
    if not np.array_equal(derived_grazing, stored_grazing):
        raise ValueError(f"{path} angular-grazing flags are inconsistent")
    barycentric = np.asarray(columns["barycentric_coordinates"], dtype=float)
    if barycentric.shape != (len(directions), 3) or not np.all(
        np.isfinite(barycentric)
    ):
        raise ValueError(f"{path} barycentric coordinates are invalid")
    if np.any(barycentric < -1.0e-10) or not np.allclose(
        np.sum(barycentric, axis=1), 1.0, rtol=0.0, atol=1.0e-8
    ):
        raise ValueError(f"{path} barycentric coordinates leave their facet")
    status = np.asarray(columns["facet_mapping_status"]).astype(str)
    facet_ids = np.asarray(columns["facet_id"]).astype(str)
    if np.any(status != "CONTAINING_FACET") or np.any(facet_ids == ""):
        raise ValueError(f"{path} contains unresolved facet mappings")
    return derived_sense.astype(str), derived_grazing


def summarize_boundary_handoffs(
    handoff_paths: Sequence[str | Path],
    *,
    histories: int,
    confidence_level: float = 0.95,
    expected_particles: Sequence[str] = EXPECTED_PARTICLES,
) -> dict[str, Any]:
    """Aggregate complete v2.1 handoffs without changing their schema.

    Detailed rows retain magnet, envelope, surface, patch, particle, exact
    crossing sense, and the independent angular-grazing flag.  Summary rows
    include explicit zero-count neutron/photon entry, exit, and tangent rows.
    """

    if isinstance(histories, bool) or int(histories) != histories:
        raise ValueError("histories must be a positive integer")
    histories = int(histories)
    if histories <= 0:
        raise ValueError("histories must be a positive integer")
    paths = tuple(Path(path).resolve() for path in handoff_paths)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("handoff_paths must be a nonempty unique sequence")
    particles_expected = tuple(str(item) for item in expected_particles)
    if (
        not particles_expected
        or len(particles_expected) != len(set(particles_expected))
        or set(particles_expected) - set(EXPECTED_PARTICLES)
    ):
        raise ValueError(
            "expected_particles must be unique neutron/photon IDs"
        )

    record_blocks: list[dict[str, Any]] = []
    bank_evidence = []
    provenance_history_values = set()
    run_invariants = None
    for path in paths:
        manifest, envelope, bank = read_handoff(path)
        completeness = _require_complete_bank(
            manifest, record_count=len(bank), path=path
        )
        provenance_histories = manifest.get("provenance", {}).get("histories")
        if provenance_histories is not None:
            provenance_history_values.add(int(provenance_histories))
            if int(provenance_histories) != histories:
                raise ValueError(
                    f"{path} provenance histories disagree with the replica"
                )
        provenance = manifest.get("provenance", {})
        normalization = manifest.get("normalization", {})
        candidate_invariants = {
            "dagmc_geometry_sha256": provenance.get("dagmc_geometry_sha256"),
            "source_definition_sha256": provenance.get(
                "source_definition_sha256"
            ),
            "parastell_commit": provenance.get("parastell_commit"),
            "openmc_version": provenance.get("openmc_version"),
            "physical_source_rate_per_s": normalization.get(
                "physical_source_rate_per_s"
            ),
            "energy_axes_sha256": hashlib.sha256(
                json.dumps(
                    manifest.get("energy_axes"),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }
        if any(value is None for value in candidate_invariants.values()):
            raise ValueError(
                f"{path} lacks fixed-problem calibration provenance"
            )
        if (
            envelope.dagmc_geometry_sha256
            != candidate_invariants["dagmc_geometry_sha256"]
        ):
            raise ValueError(
                f"{path} envelope and transport geometry hashes disagree"
            )
        if run_invariants is None:
            run_invariants = candidate_invariants
        elif candidate_invariants != run_invariants:
            raise ValueError(
                "handoffs in one replica do not share fixed problem inputs"
            )
        columns = bank.columns
        senses, grazing = _validate_mapped_crossing_columns(columns, path=path)
        if set(senses) - set(CROSSING_SENSES):
            raise ValueError(f"{path} contains an unknown crossing sense")
        record_blocks.append(
            {
                "magnet_id": np.full(
                    len(bank), envelope.magnet_component, dtype=object
                ),
                "envelope_id": np.full(
                    len(bank), envelope.envelope_id, dtype=object
                ),
                "surface_id": np.asarray(columns["surface_id"], dtype=int),
                "patch_id": np.asarray(columns["patch_id"], dtype=int),
                "particle": np.asarray(columns["particle"]).astype(str),
                "crossing_sense": senses,
                "angular_grazing": grazing.astype(bool),
                "weight": np.asarray(columns["weight"], dtype=float),
            }
        )
        bank_evidence.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "schema": manifest["schema"],
                "magnet_id": envelope.magnet_component,
                "envelope_id": envelope.envelope_id,
                "record_count": len(bank),
                "completeness": completeness,
            }
        )
    if len(provenance_history_values) > 1:
        raise ValueError("handoff provenance histories are inconsistent")

    names = tuple(record_blocks[0])
    records = {
        name: np.concatenate([block[name] for block in record_blocks])
        for name in names
    }
    all_mask = np.ones(len(records["weight"]), dtype=bool)
    particle_sense_rows = []
    for particle in particles_expected:
        for sense in CROSSING_SENSES:
            mask = (records["particle"] == particle) & (
                records["crossing_sense"] == sense
            )
            particle_sense_rows.append(
                {
                    "particle": particle,
                    "crossing_sense": sense,
                    "boundary_motion": {
                        "incoming": "entry",
                        "outgoing": "exit",
                        "grazing": "tangent_grazing",
                    }[sense],
                    "angular_grazing_raw_count": int(
                        np.count_nonzero(mask & records["angular_grazing"])
                    ),
                    **_exact_count_summary(
                        records["weight"][mask],
                        histories=histories,
                        confidence_level=confidence_level,
                    ),
                }
            )

    detailed_rows = []
    dimensions = zip(
        records["magnet_id"],
        records["envelope_id"],
        records["surface_id"],
        records["patch_id"],
        records["particle"],
        records["crossing_sense"],
        records["angular_grazing"],
    )
    unique_dimensions = sorted(
        set(dimensions),
        key=lambda item: (
            str(item[0]),
            str(item[1]),
            int(item[2]),
            int(item[3]),
            str(item[4]),
            str(item[5]),
            bool(item[6]),
        ),
    )
    for (
        magnet_id,
        envelope_id,
        surface_id,
        patch_id,
        particle,
        sense,
        angular_grazing,
    ) in unique_dimensions:
        mask = (
            (records["magnet_id"] == magnet_id)
            & (records["envelope_id"] == envelope_id)
            & (records["surface_id"] == surface_id)
            & (records["patch_id"] == patch_id)
            & (records["particle"] == particle)
            & (records["crossing_sense"] == sense)
            & (records["angular_grazing"] == angular_grazing)
        )
        detailed_rows.append(
            {
                "magnet_id": str(magnet_id),
                "envelope_id": str(envelope_id),
                "surface_id": int(surface_id),
                "patch_id": int(patch_id),
                "particle": str(particle),
                "crossing_sense": str(sense),
                "boundary_motion": {
                    "incoming": "entry",
                    "outgoing": "exit",
                    "grazing": "tangent_grazing",
                }[str(sense)],
                "angular_grazing": bool(angular_grazing),
                **_exact_count_summary(
                    records["weight"][mask],
                    histories=histories,
                    confidence_level=confidence_level,
                ),
            }
        )

    return {
        "handoff_schema": REQUIRED_HANDOFF_SCHEMA,
        "histories": histories,
        "bank_count": len(paths),
        "all_banks_complete_uncapped_unsampled": True,
        "run_invariants": run_invariants,
        "banks": bank_evidence,
        "overall": _exact_count_summary(
            records["weight"][all_mask],
            histories=histories,
            confidence_level=confidence_level,
        ),
        "by_particle_and_crossing_sense": particle_sense_rows,
        "by_magnet_surface_patch_particle_sense_and_grazing": detailed_rows,
        "statistical_contract": {
            "raw_counts": (
                "exact central Garwood confidence interval under an "
                "independent Poisson crossing-count model"
            ),
            "weights": (
                "canonical v2.1 partial-current contributions already "
                "normalized per source history"
            ),
            "weighted_interval": (
                "not claimed exact; summed weights, squared weights, and "
                "effective sample size are reported separately"
            ),
            "grazing": (
                "crossing_sense is the exact normal-sign entry/exit/tangent "
                "classification; angular_grazing independently records "
                "|mu| <= 0.1"
            ),
        },
    }


def _particle_sense_row(
    summary: Mapping[str, Any], particle: str, sense: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in summary.get("by_particle_and_crossing_sense", ())
        if row.get("particle") == particle
        and row.get("crossing_sense") == sense
    ]
    if len(matches) != 1:
        raise ValueError(f"summary lacks one {particle} {sense} row")
    return matches[0]


def qualify_photon_calibration(
    replicas: Sequence[Mapping[str, Any]],
    *,
    confidence_level: float = 0.95,
    minimum_replica_count: int = 3,
    target_incoming_photon_ess: float = 1000.0,
) -> dict[str, Any]:
    """Qualify a planning rate from independent complete-bank replicas."""

    if minimum_replica_count < 3:
        raise ValueError("photon calibration requires at least three replicas")
    if len(replicas) < minimum_replica_count:
        raise ValueError("insufficient independent calibration replicas")
    if target_incoming_photon_ess <= 0.0:
        raise ValueError("target incoming-photon ESS must be positive")
    raw_seeds = [replica["seed"] for replica in replicas]
    if any(isinstance(seed, bool) or int(seed) != seed for seed in raw_seeds):
        raise ValueError("calibration seeds must be integers")
    seeds = [int(seed) for seed in raw_seeds]
    if any(seed <= 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError(
            "calibration seeds must be distinct positive integers"
        )

    normalized_replicas = []
    raw_total = 0
    histories_total = 0
    transport_weight_total = 0.0
    transport_weight_squared_total = 0.0
    fixed_run_invariants = None
    fixed_problem_identity = None
    for replica in replicas:
        paths = replica.get("handoff_paths")
        if paths is None and replica.get("handoff_path") is not None:
            paths = [replica["handoff_path"]]
        if paths is None:
            raise ValueError("replica requires handoff_paths")
        raw_histories = replica["histories"]
        if (
            isinstance(raw_histories, bool)
            or int(raw_histories) != raw_histories
            or raw_histories <= 0
        ):
            raise ValueError("replica histories must be positive integers")
        histories = int(raw_histories)
        summary = summarize_boundary_handoffs(
            paths,
            histories=histories,
            confidence_level=confidence_level,
        )
        if fixed_run_invariants is None:
            fixed_run_invariants = summary["run_invariants"]
        elif summary["run_invariants"] != fixed_run_invariants:
            raise ValueError(
                "calibration replicas do not share fixed transport inputs"
            )
        problem_identity = replica.get("problem_identity", {})
        if not isinstance(problem_identity, Mapping):
            raise TypeError("replica problem_identity must be a mapping")
        normalized_identity = dict(problem_identity)
        if fixed_problem_identity is None:
            fixed_problem_identity = normalized_identity
        elif normalized_identity != fixed_problem_identity:
            raise ValueError(
                "calibration replicas do not share fixed problem identity"
            )
        photon = _particle_sense_row(summary, "photon", "incoming")
        raw_total += int(photon["raw_count"])
        histories_total += histories
        # Handoff weights are per source.  Undo each replica's normalization
        # before pooling differently sized independent replicas.
        transport_weight_total += (
            float(photon["summed_weight_per_source"]) * histories
        )
        transport_weight_squared_total += (
            float(photon["sum_weight_squared_per_source2"])
            * histories
            * histories
        )
        normalized_replicas.append(
            {
                "seed": int(replica["seed"]),
                "histories": histories,
                "incoming_photon": dict(photon),
                "summary": summary,
            }
        )
    lower, upper = garwood_poisson_interval(
        raw_total, confidence_level=confidence_level
    )
    lower_rate = lower / histories_total
    upper_rate = upper / histories_total
    pooled_ess = (
        transport_weight_total
        * transport_weight_total
        / transport_weight_squared_total
        if transport_weight_squared_total > 0.0
        else 0.0
    )
    ess_rate = pooled_ess / histories_total
    finite_rule = bool(raw_total > 0 and lower_rate > 0.0 and ess_rate > 0.0)
    histories_for_target_point = (
        int(math.ceil(target_incoming_photon_ess / ess_rate))
        if finite_rule
        else None
    )
    ess_per_raw_record = pooled_ess / raw_total if raw_total > 0 else 0.0
    conservative_ess_rate = lower_rate * ess_per_raw_record
    histories_for_target_conservative = (
        int(math.ceil(target_incoming_photon_ess / conservative_ess_rate))
        if conservative_ess_rate > 0.0
        else None
    )
    result = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "PARASTELL_PHOTON_CALIBRATION_PASS"
            if finite_rule
            else "INSUFFICIENT_INCOMING_PHOTON_EVIDENCE"
        ),
        "replica_count": len(normalized_replicas),
        "minimum_replica_count": minimum_replica_count,
        "independent_seeds": seeds,
        "total_histories": histories_total,
        "fixed_run_invariants": fixed_run_invariants,
        "fixed_problem_identity": fixed_problem_identity,
        "replicas": normalized_replicas,
        "pooled_incoming_photon": {
            "raw_count": raw_total,
            "summed_transport_weight_before_source_normalization": (
                transport_weight_total
            ),
            "sum_transport_weight_squared_before_source_normalization": (
                transport_weight_squared_total
            ),
            "effective_sample_size": pooled_ess,
            "effective_sample_size_rate_per_history": ess_rate,
            "raw_count_rate_per_history": raw_total / histories_total,
            "garwood_raw_count_rate_per_history_interval": {
                "confidence_level": float(confidence_level),
                "lower": lower_rate,
                "upper": upper_rate,
            },
        },
        "planning_rule": {
            "finite": finite_rule,
            "basis": (
                "pooled incoming-photon ESS per raw record applied to both "
                "the point and exact Garwood-lower raw count rates"
            ),
            "target_incoming_photon_ess": float(target_incoming_photon_ess),
            "effective_sample_size_per_raw_record": ess_per_raw_record,
            "confidence_conservative_effective_sample_size_rate_per_history": (
                conservative_ess_rate
            ),
            "required_histories_point_estimate": histories_for_target_point,
            "required_histories_confidence_conservative": (
                histories_for_target_conservative
            ),
        },
    }
    result["evidence_sha256"] = _payload_sha256(result)
    return result


def write_photon_calibration(
    path: str | Path,
    replicas: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Write a hash-bound neutral photon-calibration report."""

    result = qualify_photon_calibration(replicas, **kwargs)
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**result, "path": str(output)}


def validate_photon_calibration(
    path: str | Path, *, verify_bound_files: bool = False
) -> dict[str, Any]:
    """Validate the report hash and recompute its top-level planning rule."""

    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported photon-calibration schema")
    if value.get("evidence_sha256") != _payload_sha256(value):
        raise ValueError("photon-calibration evidence hash is invalid")
    pooled = value.get("pooled_incoming_photon", {})
    count = int(pooled.get("raw_count", -1))
    histories = int(value.get("total_histories", 0))
    if count < 0 or histories <= 0:
        raise ValueError("photon-calibration counts are invalid")
    interval = pooled.get("garwood_raw_count_rate_per_history_interval", {})
    lower, upper = garwood_poisson_interval(
        count, confidence_level=float(interval.get("confidence_level", 0.0))
    )
    if not np.allclose(
        [interval.get("lower"), interval.get("upper")],
        [lower / histories, upper / histories],
        rtol=1.0e-14,
        atol=0.0,
    ):
        raise ValueError("photon-calibration Garwood interval is inconsistent")
    finite = bool(value.get("planning_rule", {}).get("finite"))
    expected_finite = bool(
        count > 0
        and lower > 0.0
        and float(pooled.get("effective_sample_size_rate_per_history", 0.0))
        > 0.0
    )
    if finite != expected_finite:
        raise ValueError("photon-calibration planning rule is inconsistent")
    bound_bank_count = 0
    if verify_bound_files:
        for replica in value.get("replicas", ()):
            for bank in replica.get("summary", {}).get("banks", ()):
                bound_bank_count += 1
                bank_path = Path(bank["path"]).resolve()
                if not bank_path.is_file():
                    raise FileNotFoundError(bank_path)
                if bank_path.stat().st_size != int(bank["size_bytes"]):
                    raise ValueError("bound calibration bank size changed")
                if _file_sha256(bank_path) != bank["sha256"]:
                    raise ValueError("bound calibration bank hash changed")
    return {
        "schema": SCHEMA,
        "status": value.get("status"),
        "evidence_sha256": value["evidence_sha256"],
        "replica_count": int(value.get("replica_count", 0)),
        "total_histories": histories,
        "finite_planning_rule": finite,
        "verified_bound_bank_count": bound_bank_count,
    }
