"""Deterministic response-preserving multigroup condensation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .energy_groups import validate_edges


@dataclass(frozen=True)
class MergeRecord:
    removed_boundary_eV: float
    groups_after_merge: int
    maximum_relative_error: float
    maximum_integrated_error: float


@dataclass(frozen=True)
class CondensationResult:
    edges_eV: tuple[float, ...]
    collapsed_spectra: dict[str, tuple[float, ...]]
    reference_responses: dict[str, dict[str, float]]
    collapsed_responses: dict[str, dict[str, float]]
    relative_errors: dict[str, dict[str, float]]
    merge_history: tuple[MergeRecord, ...]
    protected_boundaries_eV: tuple[float, ...]
    requested_tolerance: float
    requested_max_groups: int
    qualified: bool

    @property
    def group_count(self) -> int:
        return len(self.edges_eV) - 1

    @property
    def maximum_relative_error(self) -> float:
        return max(
            error
            for spectrum in self.relative_errors.values()
            for error in spectrum.values()
        )

    def write_registry_candidate(
        self,
        path: str | Path,
        *,
        name: str,
        particle: str,
        provenance: Mapping[str, object],
    ) -> None:
        """Write a qualified derived structure in registry-compatible JSON."""

        if not self.qualified:
            raise RuntimeError("unqualified condensation cannot be registered")
        from .energy_groups.registry import _canonical_edge_hash

        payload = {
            "name": name,
            "aliases": [],
            "particle": particle,
            "intended_purposes": ["response-selected deterministic transport"],
            "group_count": self.group_count,
            "edge_units": "eV",
            "minimum_energy": self.edges_eV[0],
            "maximum_energy": self.edges_eV[-1],
            "edge_ordering": "ascending",
            "status": "response-qualified",
            "edge_sha256": _canonical_edge_hash(self.edges_eV),
            "authoritative_source": {
                **dict(provenance),
                "condensation_tolerance": self.requested_tolerance,
                "maximum_observed_relative_error": self.maximum_relative_error,
                "protected_boundaries_eV": list(self.protected_boundaries_eV),
            },
            "edges": list(self.edges_eV),
        }
        Path(path).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="ascii"
        )


@dataclass
class _Segment:
    first: int
    last: int


def _arrays(
    values: Mapping[str, Sequence[float]], groups: int, label: str
) -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(sorted(values))
    if not names:
        raise ValueError(f"at least one {label} is required")
    array = np.asarray([values[name] for name in names], dtype=float)
    if array.shape != (len(names), groups):
        raise ValueError(f"every {label} must have one value per fine group")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} values must be finite")
    return names, array


def condense_groups(
    fine_edges_eV: Sequence[float],
    spectra: Mapping[str, Sequence[float]],
    responses: Mapping[str, Sequence[float]],
    *,
    protected_boundaries_eV: Sequence[float] = (),
    relative_tolerance: float,
    min_groups: int = 1,
    max_groups: int,
) -> CondensationResult:
    """Greedily merge adjacent groups while protecting all supplied responses.

    Spectrum arrays contain integrated particle weight/current in each fine group.
    Response arrays contain fine-group response coefficients. Coarse spectra are
    exact sums, while coefficients are collapsed with an equal-spectrum aggregate
    weighting. Candidate merges are accepted only if every spectrum/response
    integral remains within ``relative_tolerance``.
    """

    edges = np.asarray(validate_edges(fine_edges_eV), dtype=float)
    group_count = len(edges) - 1
    if not (0.0 <= relative_tolerance < 1.0):
        raise ValueError("relative_tolerance must be in [0, 1)")
    if not (1 <= min_groups <= max_groups <= group_count):
        raise ValueError(
            "require 1 <= min_groups <= max_groups <= fine groups"
        )
    spectrum_names, spectrum_values = _arrays(spectra, group_count, "spectrum")
    response_names, response_values = _arrays(
        responses, group_count, "response"
    )
    if np.any(spectrum_values < 0.0):
        raise ValueError("integrated spectrum weights cannot be negative")
    totals = spectrum_values.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("every spectrum must have positive integrated weight")
    normalized = spectrum_values / totals[:, None]
    collapse_weight = normalized.sum(axis=0)
    reference = spectrum_values @ response_values.T
    scales = np.maximum(np.abs(reference), 1.0e-300)

    protected_indices = set()
    protected_values = []
    for boundary in protected_boundaries_eV:
        matches = np.flatnonzero(
            np.isclose(edges, boundary, rtol=0.0, atol=0.0)
        )
        if len(matches) != 1:
            raise ValueError(
                f"protected boundary {boundary} eV is absent from fine grid"
            )
        index = int(matches[0])
        if index not in {0, len(edges) - 1}:
            protected_indices.add(index)
        protected_values.append(float(edges[index]))

    def coefficient(segment: _Segment) -> np.ndarray:
        selection = slice(segment.first, segment.last)
        weight = collapse_weight[selection]
        denominator = weight.sum()
        if denominator > 0.0:
            return (response_values[:, selection] * weight).sum(
                axis=1
            ) / denominator
        return response_values[:, selection].mean(axis=1)

    def contribution(segment: _Segment) -> np.ndarray:
        weights = spectrum_values[:, segment.first : segment.last].sum(axis=1)
        return weights[:, None] * coefficient(segment)[None, :]

    segments = [_Segment(index, index + 1) for index in range(group_count)]
    contributions = [contribution(segment) for segment in segments]
    predicted = sum(contributions, np.zeros_like(reference))
    history = []
    while len(segments) > min_groups:
        best = None
        for index in range(len(segments) - 1):
            boundary_index = segments[index].last
            if boundary_index in protected_indices:
                continue
            merged = _Segment(segments[index].first, segments[index + 1].last)
            merged_contribution = contribution(merged)
            candidate = (
                predicted
                - contributions[index]
                - contributions[index + 1]
                + merged_contribution
            )
            relative = np.abs(candidate - reference) / scales
            score = (
                float(relative.max()),
                float(np.abs(candidate - reference).max()),
            )
            key = (*score, float(edges[boundary_index]), index)
            if best is None or key < best[0]:
                best = (key, merged, merged_contribution, candidate)
        if best is None or best[0][0] > relative_tolerance:
            break
        key, merged, merged_contribution, predicted = best
        index = key[-1]
        history.append(
            MergeRecord(
                removed_boundary_eV=float(edges[segments[index].last]),
                groups_after_merge=len(segments) - 1,
                maximum_relative_error=key[0],
                maximum_integrated_error=key[1],
            )
        )
        segments[index : index + 2] = [merged]
        contributions[index : index + 2] = [merged_contribution]

    selected_edges = [float(edges[segments[0].first])]
    selected_edges.extend(float(edges[segment.last]) for segment in segments)
    collapsed_spectra = {
        name: tuple(
            float(
                spectrum_values[
                    spectrum_index, segment.first : segment.last
                ].sum()
            )
            for segment in segments
        )
        for spectrum_index, name in enumerate(spectrum_names)
    }
    collapsed = {
        spectrum: {
            response: float(predicted[spectrum_index, response_index])
            for response_index, response in enumerate(response_names)
        }
        for spectrum_index, spectrum in enumerate(spectrum_names)
    }
    references = {
        spectrum: {
            response: float(reference[spectrum_index, response_index])
            for response_index, response in enumerate(response_names)
        }
        for spectrum_index, spectrum in enumerate(spectrum_names)
    }
    errors = {
        spectrum: {
            response: float(
                abs(
                    collapsed[spectrum][response]
                    - references[spectrum][response]
                )
                / max(abs(references[spectrum][response]), 1.0e-300)
            )
            for response in response_names
        }
        for spectrum in spectrum_names
    }
    maximum_error = max(
        value for row in errors.values() for value in row.values()
    )
    return CondensationResult(
        edges_eV=tuple(selected_edges),
        collapsed_spectra=collapsed_spectra,
        reference_responses=references,
        collapsed_responses=collapsed,
        relative_errors=errors,
        merge_history=tuple(history),
        protected_boundaries_eV=tuple(sorted(set(protected_values))),
        requested_tolerance=relative_tolerance,
        requested_max_groups=max_groups,
        qualified=len(segments) <= max_groups
        and maximum_error <= relative_tolerance,
    )


def fusion_neutron_protected_boundaries() -> tuple[float, ...]:
    """Return documented regions that a caller should insert into a master grid."""

    return (
        0.025,
        0.55,
        10.0,
        1.0e5,
        1.0e6,
        2.45e6,
        5.0e6,
        10.0e6,
        13.0e6,
        13.5e6,
        14.0e6,
        14.1e6,
        14.2e6,
        14.5e6,
        15.0e6,
        20.0e6,
    )


def photon_protected_boundaries() -> tuple[float, ...]:
    """Return baseline photon thresholds; material edges/lines must be added."""

    return (1.022e6,)
