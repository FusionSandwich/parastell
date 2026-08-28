"""Deterministic clearance constraints for ParaStell radial builds.

This module changes thickness data only.  It deliberately does not construct,
move, rotate, scale, or otherwise modify magnet geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterpolatedThicknessComparison:
    """Dense audit of one layer under ParaStell's cumulative PCHIP rule."""

    toroidal_angles_deg: np.ndarray
    poloidal_angles_deg: np.ndarray
    reference_thickness_cm: np.ndarray
    candidate_thickness_cm: np.ndarray
    reduction_cm: np.ndarray

    def summary(self, *, changed_tolerance_cm: float = 1.0e-10) -> dict:
        tolerance = float(changed_tolerance_cm)
        changed = np.abs(self.reduction_cm) > tolerance
        increased = self.reduction_cm < -tolerance
        indices = np.argwhere(changed)
        minimum_index = np.unravel_index(
            np.argmin(self.candidate_thickness_cm),
            self.candidate_thickness_cm.shape,
        )
        changed_extent = None
        if len(indices):
            changed_extent = {
                "toroidal_min_deg": float(
                    self.toroidal_angles_deg[indices[:, 0]].min()
                ),
                "toroidal_max_deg": float(
                    self.toroidal_angles_deg[indices[:, 0]].max()
                ),
                "poloidal_min_deg": float(
                    self.poloidal_angles_deg[indices[:, 1]].min()
                ),
                "poloidal_max_deg": float(
                    self.poloidal_angles_deg[indices[:, 1]].max()
                ),
            }
        return {
            "sample_shape": list(self.candidate_thickness_cm.shape),
            "toroidal_step_max_deg": float(
                np.max(np.diff(self.toroidal_angles_deg))
            ),
            "poloidal_step_max_deg": float(
                np.max(np.diff(self.poloidal_angles_deg))
            ),
            "minimum_candidate_thickness_cm": float(
                self.candidate_thickness_cm[minimum_index]
            ),
            "minimum_location_deg": {
                "toroidal": float(self.toroidal_angles_deg[minimum_index[0]]),
                "poloidal": float(self.poloidal_angles_deg[minimum_index[1]]),
            },
            "maximum_reduction_cm": float(np.max(self.reduction_cm)),
            "maximum_increase_cm": float(
                np.max(np.maximum(-self.reduction_cm, 0.0))
            ),
            "increased_sample_count": int(np.count_nonzero(increased)),
            "changed_sample_count": int(np.count_nonzero(changed)),
            "sample_count": int(changed.size),
            "changed_fraction": float(
                np.count_nonzero(changed) / changed.size
            ),
            "changed_extent_deg": changed_extent,
            "poloidal_closure_max_abs_cm": float(
                np.max(
                    np.abs(
                        self.candidate_thickness_cm[:, 0]
                        - self.candidate_thickness_cm[:, -1]
                    )
                )
            ),
        }


@dataclass(frozen=True)
class RadialBuildConstraintResult:
    """Result of a pointwise radial-build clearance constraint."""

    original: dict[str, np.ndarray]
    corrected: dict[str, np.ndarray]
    reductions: dict[str, np.ndarray]
    available_space_cm: np.ndarray
    requested_clearance_cm: float
    required_reduction_cm: np.ndarray
    residual_excess_cm: np.ndarray
    changed_mask: np.ndarray

    @property
    def feasible(self) -> bool:
        """Return whether every sample satisfies the declared constraint."""

        return bool(np.all(self.residual_excess_cm <= 1.0e-12))

    def summary(self) -> dict:
        """Return compact JSON-serializable evidence."""

        return {
            "requested_clearance_cm": self.requested_clearance_cm,
            "sample_shape": list(self.available_space_cm.shape),
            "changed_sample_count": int(np.count_nonzero(self.changed_mask)),
            "sample_count": int(self.changed_mask.size),
            "maximum_required_reduction_cm": float(
                np.max(self.required_reduction_cm)
            ),
            "maximum_residual_excess_cm": float(
                np.max(self.residual_excess_cm)
            ),
            "feasible": self.feasible,
            "layer_volume_proxy_change_cm": {
                name: float(np.sum(self.corrected[name] - values))
                for name, values in self.original.items()
            },
        }


def _copy_thicknesses(
    thicknesses: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
) -> dict[str, np.ndarray]:
    copied = {
        name: np.array(values, dtype=float, copy=True)
        for name, values in thicknesses.items()
    }
    if not copied:
        raise ValueError("at least one radial-build layer is required")
    shapes = {values.shape for values in copied.values()}
    if len(shapes) != 1:
        raise ValueError("all radial-build thickness matrices must match")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError(
            "radial-build thickness matrices must be two-dimensional"
        )
    for name, values in copied.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"layer {name!r} contains non-finite thickness")
        if np.any(values < 0.0):
            raise ValueError(f"layer {name!r} contains negative thickness")
    return copied


def constrain_radial_build(
    thicknesses: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    available_space_cm: Sequence[Sequence[float]] | np.ndarray,
    *,
    clearance_cm: float,
    reduction_order: Sequence[str],
    minimum_thickness_cm: Mapping[str, float] | None = None,
) -> RadialBuildConstraintResult:
    """Reduce selected layers only where a clearance constraint is violated.

    ``available_space_cm`` is the independently measured distance from the
    radial-build reference surface to the nearest magnet outer surface along
    the declared construction ray.  The sum of all corrected layer thicknesses
    plus ``clearance_cm`` must not exceed that distance.

    Reductions are applied in ``reduction_order`` and stop at each layer's
    declared minimum.  Layers absent from the order are protected exactly.
    Inputs are copied and never mutated.  An infeasible request raises instead
    of silently clipping a protected layer.
    """

    original = _copy_thicknesses(thicknesses)
    shape = next(iter(original.values())).shape
    available = np.array(available_space_cm, dtype=float, copy=True)
    if available.shape != shape:
        raise ValueError("available-space matrix must match layer matrices")
    if (
        np.any(np.isnan(available))
        or np.any(np.isneginf(available))
        or np.any(available <= 0.0)
    ):
        raise ValueError(
            "available space must be strictly positive or positive infinity"
        )
    if not np.isfinite(clearance_cm) or clearance_cm < 0.0:
        raise ValueError("clearance_cm must be finite and nonnegative")
    if len(set(reduction_order)) != len(reduction_order):
        raise ValueError("reduction_order must not contain duplicates")

    minima = dict(minimum_thickness_cm or {})
    unknown_layers = (set(reduction_order) | set(minima)) - set(original)
    if unknown_layers:
        raise ValueError(
            f"unknown radial-build layers: {sorted(unknown_layers)}"
        )
    for name in reduction_order:
        minima.setdefault(name, 0.0)
    for name, minimum in minima.items():
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError(
                f"minimum for {name!r} must be finite and nonnegative"
            )
        if np.any(original[name] < minimum):
            raise ValueError(
                f"original {name!r} thickness is below its minimum"
            )

    total = np.sum(np.stack(list(original.values())), axis=0)
    required = np.maximum(total + clearance_cm - available, 0.0)
    residual = required.copy()
    corrected = {name: values.copy() for name, values in original.items()}
    reductions = {name: np.zeros(shape, dtype=float) for name in original}

    for name in reduction_order:
        capacity = corrected[name] - minima[name]
        amount = np.minimum(capacity, residual)
        corrected[name] -= amount
        reductions[name] += amount
        residual -= amount

    if np.any(residual > 1.0e-12):
        worst = float(np.max(residual))
        raise ValueError(
            "clearance constraint is infeasible without reducing a protected "
            f"layer or crossing a minimum; maximum residual is {worst:.12g} cm"
        )

    changed = np.zeros(shape, dtype=bool)
    for amount in reductions.values():
        changed |= amount > 0.0

    return RadialBuildConstraintResult(
        original=original,
        corrected=corrected,
        reductions=reductions,
        available_space_cm=available,
        requested_clearance_cm=float(clearance_cm),
        required_reduction_cm=required,
        residual_excess_cm=residual,
        changed_mask=changed,
    )


def sampled_interpolated_layer_thickness(
    reference: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    candidate: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    *,
    layer: str,
    toroidal_control_angles_deg: Sequence[float],
    poloidal_control_angles_deg: Sequence[float],
    toroidal_step_deg: float = 0.25,
    poloidal_step_deg: float = 0.5,
) -> InterpolatedThicknessComparison:
    """Sample the cumulative-offset PCHIP interpolation used by ParaStell."""
    from scipy.interpolate import RegularGridInterpolator

    reference_layers = _copy_thicknesses(reference)
    candidate_layers = _copy_thicknesses(candidate)
    if tuple(reference_layers) != tuple(candidate_layers):
        raise ValueError("reference and candidate layer order must match")
    if layer not in reference_layers:
        raise ValueError(f"unknown layer {layer!r}")
    phi_control = np.asarray(toroidal_control_angles_deg, dtype=float)
    theta_control = np.asarray(poloidal_control_angles_deg, dtype=float)
    expected = next(iter(reference_layers.values())).shape
    if expected != (len(phi_control), len(theta_control)):
        raise ValueError("control-angle counts must match layer matrices")
    for name, step in (
        ("toroidal_step_deg", toroidal_step_deg),
        ("poloidal_step_deg", poloidal_step_deg),
    ):
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    phi = np.linspace(
        phi_control[0],
        phi_control[-1],
        int(np.ceil((phi_control[-1] - phi_control[0]) / toroidal_step_deg))
        + 1,
    )
    theta = np.linspace(
        theta_control[0],
        theta_control[-1],
        int(
            np.ceil((theta_control[-1] - theta_control[0]) / poloidal_step_deg)
        )
        + 1,
    )
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")
    points = np.column_stack((phi_grid.ravel(), theta_grid.ravel()))
    target_index = tuple(reference_layers).index(layer)

    def thickness(layers: Mapping[str, np.ndarray]) -> np.ndarray:
        values = list(layers.values())
        inner = (
            np.sum(np.stack(values[:target_index]), axis=0)
            if target_index
            else np.zeros(expected)
        )
        outer = inner + values[target_index]
        inner_interpolator = RegularGridInterpolator(
            (phi_control, theta_control), inner, method="pchip"
        )
        outer_interpolator = RegularGridInterpolator(
            (phi_control, theta_control), outer, method="pchip"
        )
        return (
            outer_interpolator(points) - inner_interpolator(points)
        ).reshape(phi_grid.shape)

    reference_field = thickness(reference_layers)
    candidate_field = thickness(candidate_layers)
    return InterpolatedThicknessComparison(
        toroidal_angles_deg=phi,
        poloidal_angles_deg=theta,
        reference_thickness_cm=reference_field,
        candidate_thickness_cm=candidate_field,
        reduction_cm=reference_field - candidate_field,
    )


def eliminate_interpolated_layer_increase(
    reference: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    candidate: Mapping[str, Sequence[Sequence[float]] | np.ndarray],
    *,
    layer: str,
    toroidal_control_angles_deg: Sequence[float],
    poloidal_control_angles_deg: Sequence[float],
    minimum_thickness_cm: float,
    toroidal_step_deg: float = 0.25,
    poloidal_step_deg: float = 0.5,
    tolerance_cm: float = 1.0e-8,
    maximum_iterations: int = 25,
) -> tuple[dict[str, np.ndarray], dict]:
    """Conservatively remove dense PCHIP increases from one candidate layer.

    ParaStell interpolates cumulative offsets, so reducing every control value
    of one layer does not by itself prove that the reconstructed thickness is
    nowhere larger between controls.  This routine samples that exact rule,
    projects each dense increase to the surrounding controls, and lowers only
    the selected layer until the increase disappears or the declared minimum
    would be crossed.
    """

    from .cad_radial_clearance import project_dense_deficit_to_control_grid

    if not np.isfinite(minimum_thickness_cm) or minimum_thickness_cm < 0.0:
        raise ValueError("minimum_thickness_cm must be finite and nonnegative")
    if not np.isfinite(tolerance_cm) or tolerance_cm <= 0.0:
        raise ValueError("tolerance_cm must be finite and positive")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    reference_layers = _copy_thicknesses(reference)
    corrected = _copy_thicknesses(candidate)
    if tuple(reference_layers) != tuple(corrected):
        raise ValueError("reference and candidate layer order must match")
    if layer not in corrected:
        raise ValueError(f"unknown layer {layer!r}")
    if np.any(corrected[layer] > reference_layers[layer] + tolerance_cm):
        raise ValueError("candidate control values already increase the layer")

    iterations = []
    for iteration in range(maximum_iterations + 1):
        comparison = sampled_interpolated_layer_thickness(
            reference_layers,
            corrected,
            layer=layer,
            toroidal_control_angles_deg=toroidal_control_angles_deg,
            poloidal_control_angles_deg=poloidal_control_angles_deg,
            toroidal_step_deg=toroidal_step_deg,
            poloidal_step_deg=poloidal_step_deg,
        )
        increase = np.maximum(-comparison.reduction_cm, 0.0)
        maximum = float(np.max(increase))
        iterations.append(
            {
                "iteration": iteration,
                "maximum_dense_increase_cm": maximum,
                "increased_sample_count": int(
                    np.count_nonzero(increase > tolerance_cm)
                ),
            }
        )
        if maximum <= tolerance_cm:
            return corrected, {
                "status": "CONVERGED",
                "tolerance_cm": tolerance_cm,
                "iteration_count": iteration,
                "iterations": iterations,
                "final_summary": comparison.summary(
                    changed_tolerance_cm=tolerance_cm
                ),
            }
        if iteration == maximum_iterations:
            break
        projected = project_dense_deficit_to_control_grid(
            increase,
            comparison.toroidal_angles_deg,
            comparison.poloidal_angles_deg,
            np.asarray(toroidal_control_angles_deg, dtype=float),
            np.asarray(poloidal_control_angles_deg, dtype=float),
        )
        projected[projected <= tolerance_cm] = 0.0
        proposed = corrected[layer] - projected
        if np.any(proposed < minimum_thickness_cm - tolerance_cm):
            raise ValueError(
                "removing interpolated layer increases would cross the "
                "declared minimum thickness"
            )
        corrected[layer] = np.maximum(proposed, minimum_thickness_cm)
        closure = np.minimum(corrected[layer][:, 0], corrected[layer][:, -1])
        corrected[layer][:, 0] = closure
        corrected[layer][:, -1] = closure

    raise ValueError(
        "interpolated layer increase refinement did not converge within "
        f"{maximum_iterations} iterations"
    )
