"""Engineering port configuration and geometry result records.

A port is defined by an oriented axis and a clear two-dimensional aperture.
Endpoint references bound that aperture physically; optional lining expands
outward from the clear opening. The legacy ``layer_span`` form is accepted as
a deprecated shorthand and converted to :class:`PortExtent` during parsing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


class PortGeometryNotImplementedError(NotImplementedError):
    """Compatibility exception retained for downstream imports."""


def _validate_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _validate_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence")
    return value


def _validate_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    if not value.strip():
        raise ValueError(f"{path} cannot be empty")
    return value


def _validate_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if value < 1:
        raise ValueError(f"{path} must be positive")
    return value


def _validate_finite_scalar(value: Any, path: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _to_vector3(value: Any, path: str) -> np.ndarray:
    vector = np.asarray(list(_validate_sequence(value, path)), dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{path} must be a three-element vector")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{path} values must be finite")
    return vector


def _normalize_vector(vector: np.ndarray, path: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"{path} cannot be a zero vector")
    return vector / norm


def _ensure_no_unexpected_keys(
    mapping: Mapping[str, Any], path: str, allowed: set[str]
) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{path} contains unknown keys: {names}")


def _resolve_layer_indices(
    layer_names: Sequence[str], start: str, count: int, direction: str
) -> list[str]:
    if start not in layer_names:
        names = ", ".join(layer_names)
        raise ValueError(
            f"layer_span.start ({start!r}) is not in radial build layers {names}"
        )
    start_index = layer_names.index(start)
    step = 1 if direction == "outward" else -1
    stop_index = start_index + step * (count - 1)
    if stop_index < 0 or stop_index >= len(layer_names):
        raise ValueError(
            "layer_span extends beyond available radial layers; clipping is not allowed"
        )
    indices = (
        range(start_index, stop_index + 1)
        if step == 1
        else range(start_index, stop_index - 1, -1)
    )
    return [layer_names[index] for index in indices]


@dataclass(frozen=True)
class PortSurfaceAnchor:
    """A user-facing angular location on a continuous in-vessel surface."""

    reference: str
    toroidal_angle: float
    poloidal_angle: float
    layer: str | None = None

    def __post_init__(self) -> None:
        reference = _validate_string(
            self.reference, "placement.anchor.reference"
        ).lower()
        allowed = {
            "plasma_surface",
            "wall_surface",
            "layer_inner",
            "layer_outer",
        }
        if reference not in allowed:
            raise ValueError(
                "placement.anchor.reference must be one of "
                f"{sorted(allowed)}"
            )
        if reference.startswith("layer_"):
            if self.layer is None:
                raise ValueError(
                    f"{reference} anchor requires placement.anchor.layer"
                )
            layer = _validate_string(self.layer, "placement.anchor.layer")
        elif self.layer is not None:
            raise ValueError(
                f"{reference} anchor must not define placement.anchor.layer"
            )
        else:
            layer = None
        object.__setattr__(self, "reference", reference)
        object.__setattr__(
            self,
            "toroidal_angle",
            _validate_finite_scalar(
                self.toroidal_angle, "placement.anchor.toroidal_angle"
            ),
        )
        object.__setattr__(
            self,
            "poloidal_angle",
            _validate_finite_scalar(
                self.poloidal_angle, "placement.anchor.poloidal_angle"
            ),
        )
        object.__setattr__(self, "layer", layer)


@dataclass(frozen=True)
class PortSurfaceAxis:
    """Outward-normal axis with optional signed surface-tangent tilts."""

    mode: str = "outward_normal"
    poloidal_tilt: float = 0.0
    toroidal_tilt: float = 0.0

    def __post_init__(self) -> None:
        mode = _validate_string(self.mode, "placement.axis.mode").lower()
        if mode != "outward_normal":
            raise ValueError("placement.axis.mode must be 'outward_normal'")
        poloidal_tilt = _validate_finite_scalar(
            self.poloidal_tilt, "placement.axis.poloidal_tilt"
        )
        toroidal_tilt = _validate_finite_scalar(
            self.toroidal_tilt, "placement.axis.toroidal_tilt"
        )
        if abs(poloidal_tilt) >= 90.0 or abs(toroidal_tilt) >= 90.0:
            raise ValueError(
                "surface-axis tilts must have magnitude below 90 degrees"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "poloidal_tilt", poloidal_tilt)
        object.__setattr__(self, "toroidal_tilt", toroidal_tilt)


@dataclass(frozen=True)
class PortPlacement:
    """Declarative placement and resolved right-handed local frame.

    Positive distance along ``axis`` is from the plasma/inner side toward the
    blanket exterior. ``reference_direction`` controls aperture rotation.
    """

    anchor: tuple[float, float, float] | None = None
    axis: tuple[float, float, float] | None = None
    reference_direction: tuple[float, float, float] | None = None
    mode: str = "cartesian"
    max_search_length: float | None = None
    surface_anchor: PortSurfaceAnchor | None = None
    surface_axis: PortSurfaceAxis | None = None
    roll: float = 0.0
    is_resolved: bool = False
    local_axis: tuple[float, float, float] | None = None
    local_reference: tuple[float, float, float] | None = None
    local_normal: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        mode = _validate_string(self.mode, "placement.mode").lower()
        if mode not in {"cartesian", "surface"}:
            raise ValueError("placement.mode must be 'cartesian' or 'surface'")
        roll = _validate_finite_scalar(self.roll, "placement.roll")
        if mode == "surface" and not self.is_resolved:
            if self.surface_anchor is None or self.surface_axis is None:
                raise ValueError(
                    "surface placement requires surface anchor and axis specifications"
                )
            if any(
                value is not None
                for value in (self.anchor, self.axis, self.reference_direction)
            ):
                raise ValueError(
                    "unresolved surface placement must not contain Cartesian vectors"
                )
            max_search_length = (
                1000.0
                if self.max_search_length is None
                else _validate_finite_scalar(
                    self.max_search_length, "placement.max_search_length"
                )
            )
            if max_search_length <= 0.0:
                raise ValueError(
                    "placement.max_search_length must be positive"
                )
            object.__setattr__(self, "mode", mode)
            object.__setattr__(self, "roll", roll)
            object.__setattr__(self, "max_search_length", max_search_length)
            return
        if (
            self.anchor is None
            or self.axis is None
            or self.reference_direction is None
        ):
            raise ValueError(
                "resolved placement requires anchor, axis, and reference"
            )
        anchor = _to_vector3(self.anchor, "placement.anchor")
        axis = _normalize_vector(
            _to_vector3(self.axis, "placement.axis"), "placement.axis"
        )
        reference = _normalize_vector(
            _to_vector3(
                self.reference_direction, "placement.reference_direction"
            ),
            "placement.reference_direction",
        )
        local_reference = reference - np.dot(reference, axis) * axis
        if np.linalg.norm(local_reference) < 1e-12:
            raise ValueError(
                "placement.reference_direction must not be parallel to placement.axis"
            )
        local_reference = _normalize_vector(local_reference, "frame reference")
        local_normal = _normalize_vector(
            np.cross(axis, local_reference), "frame normal"
        )
        max_search_length = (
            max(1000.0, 2.0 * float(np.linalg.norm(anchor)))
            if self.max_search_length is None
            else _validate_finite_scalar(
                self.max_search_length, "placement.max_search_length"
            )
        )
        if max_search_length <= 0.0:
            raise ValueError("placement.max_search_length must be positive")
        object.__setattr__(self, "anchor", tuple(anchor))
        object.__setattr__(self, "axis", tuple(axis))
        object.__setattr__(self, "reference_direction", tuple(reference))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "roll", roll)
        object.__setattr__(self, "is_resolved", True)
        object.__setattr__(self, "max_search_length", float(max_search_length))
        object.__setattr__(self, "local_axis", tuple(axis))
        object.__setattr__(self, "local_reference", tuple(local_reference))
        object.__setattr__(self, "local_normal", tuple(local_normal))

    def resolve_surface_frame(
        self,
        anchor,
        axis,
        local_reference,
    ) -> "PortPlacement":
        """Return a resolved copy without exposing point-cloud indices."""
        if self.mode != "surface":
            return self
        return replace(
            self,
            anchor=tuple(np.asarray(anchor, dtype=float)),
            axis=tuple(np.asarray(axis, dtype=float)),
            reference_direction=tuple(
                np.asarray(local_reference, dtype=float)
            ),
            is_resolved=True,
        )


@dataclass(frozen=True)
class PortCrossSection:
    """Clear internal aperture dimensions."""

    shape: str
    radius: float | None = None
    width: float | None = None
    height: float | None = None
    dimensions_are: str = "clear_aperture"

    def __post_init__(self) -> None:
        shape = _validate_string(self.shape, "cross_section.shape").lower()
        dimensions_are = _validate_string(
            self.dimensions_are, "cross_section.dimensions_are"
        ).lower()
        if dimensions_are != "clear_aperture":
            raise ValueError(
                "cross_section.dimensions_are must be 'clear_aperture'"
            )
        if shape not in {"circle", "rectangle"}:
            raise ValueError(
                "cross_section.shape must be one of {'circle', 'rectangle'}"
            )
        if shape == "circle":
            if self.width is not None or self.height is not None:
                raise ValueError(
                    "circle cross_section must define only radius"
                )
            if self.radius is None:
                raise ValueError("circle cross_section requires a radius")
            radius = _validate_finite_scalar(
                self.radius, "cross_section.radius"
            )
            if radius <= 0.0:
                raise ValueError("cross_section.radius must be positive")
            object.__setattr__(self, "radius", radius)
            object.__setattr__(self, "width", None)
            object.__setattr__(self, "height", None)
        else:
            if self.radius is not None:
                raise ValueError(
                    "rectangle cross_section must not define radius"
                )
            if self.width is None or self.height is None:
                raise ValueError(
                    "rectangle cross_section must define both width and height"
                )
            width = _validate_finite_scalar(self.width, "cross_section.width")
            height = _validate_finite_scalar(
                self.height, "cross_section.height"
            )
            if width <= 0.0 or height <= 0.0:
                raise ValueError(
                    "cross_section width and height must be positive"
                )
            object.__setattr__(self, "radius", None)
            object.__setattr__(self, "width", width)
            object.__setattr__(self, "height", height)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dimensions_are", dimensions_are)


@dataclass(frozen=True)
class PortEndpoint:
    """Physical endpoint with an optional signed outward axial offset."""

    reference: str
    layer: str | None = None
    fraction: float | None = None
    axial_offset: float = 0.0

    def __post_init__(self) -> None:
        reference = _validate_string(
            self.reference, "endpoint.reference"
        ).lower()
        if reference not in {"plasma_surface", "wall_surface", "layer"}:
            raise ValueError(
                "endpoint.reference must be one of "
                "{'plasma_surface', 'wall_surface', 'layer'}"
            )
        if reference == "layer":
            if self.layer is None:
                raise ValueError("layer endpoint requires endpoint.layer")
            layer = _validate_string(self.layer, "endpoint.layer")
            if self.fraction is None:
                raise ValueError("layer endpoint requires endpoint.fraction")
            fraction = _validate_finite_scalar(
                self.fraction, "endpoint.fraction"
            )
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("endpoint.fraction must be between 0 and 1")
        else:
            if self.layer is not None or self.fraction is not None:
                raise ValueError(
                    f"{reference} endpoint must not define layer or fraction"
                )
            layer = None
            fraction = None
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(
            self,
            "axial_offset",
            _validate_finite_scalar(
                self.axial_offset, "endpoint.axial_offset"
            ),
        )


@dataclass(frozen=True)
class PortExtent:
    start: PortEndpoint
    end: PortEndpoint
    outer_extension: float = 0.0

    def __post_init__(self) -> None:
        extension = _validate_finite_scalar(
            self.outer_extension, "extent.outer_extension"
        )
        if extension < 0.0:
            raise ValueError("extent.outer_extension must be nonnegative")
        object.__setattr__(self, "outer_extension", extension)


@dataclass(frozen=True)
class PortLiner:
    enabled: bool = False
    thickness: float = 0.0
    mat_tag: str = "SS316L"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("liner.enabled must be a boolean")
        thickness = _validate_finite_scalar(self.thickness, "liner.thickness")
        if thickness < 0.0:
            raise ValueError("liner.thickness must be nonnegative")
        mat_tag = _validate_string(self.mat_tag, "liner.mat_tag")
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "mat_tag", mat_tag)
        object.__setattr__(self, "enabled", self.enabled and thickness > 0.0)


@dataclass(frozen=True)
class PortFill:
    mat_tag: str = "Vacuum"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mat_tag", _validate_string(self.mat_tag, "fill.mat_tag")
        )


@dataclass(frozen=True)
class PortRepetition:
    mode: str = "single"

    def __post_init__(self) -> None:
        mode = _validate_string(self.mode, "repetition.mode").lower()
        if mode not in {"single", "per_period"}:
            raise ValueError(
                "repetition.mode currently supports only 'single' and 'per_period'"
            )
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class PortCollisionPolicy:
    magnet_policy: str = "error"
    clearance_policy: str = "warn"
    minimum_magnet_clearance: float = 0.0

    def __post_init__(self) -> None:
        policies = {"error", "warn", "report", "ignore"}
        magnet_policy = _validate_string(
            self.magnet_policy, "collision.magnet_policy"
        ).lower()
        clearance_policy = _validate_string(
            self.clearance_policy, "collision.clearance_policy"
        ).lower()
        if magnet_policy not in policies or clearance_policy not in policies:
            raise ValueError(
                "collision policies must be one of "
                "{'error', 'warn', 'report', 'ignore'}"
            )
        clearance = _validate_finite_scalar(
            self.minimum_magnet_clearance,
            "collision.minimum_magnet_clearance",
        )
        if clearance < 0.0:
            raise ValueError(
                "collision.minimum_magnet_clearance must be nonnegative"
            )
        object.__setattr__(self, "magnet_policy", magnet_policy)
        object.__setattr__(self, "clearance_policy", clearance_policy)
        object.__setattr__(self, "minimum_magnet_clearance", clearance)


@dataclass(frozen=True)
class PortLayerSpan:
    """Deprecated contiguous-layer shorthand."""

    start: str
    count: int
    direction: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "start", _validate_string(self.start, "start")
        )
        object.__setattr__(
            self,
            "count",
            _validate_positive_int(self.count, "layer_span.count"),
        )
        direction = _validate_string(
            self.direction, "layer_span.direction"
        ).lower()
        if direction not in {"inward", "outward"}:
            raise ValueError(
                "layer_span.direction must be one of {'inward', 'outward'}"
            )
        object.__setattr__(self, "direction", direction)

    def resolve(self, layer_names: Sequence[str]) -> "PortResolution":
        layers = _resolve_layer_indices(
            list(layer_names), self.start, self.count, self.direction
        )
        return PortResolution(
            layers=tuple(layers),
            start=self.start,
            count=self.count,
            direction=self.direction,
        )


@dataclass(frozen=True)
class PortResolution:
    layers: tuple[str, ...]
    start: str
    count: int
    direction: str


@dataclass(frozen=True)
class PortGeometryResult:
    name: str
    resolved_start: float
    resolved_end: float
    outer_extension: float
    ordered_intersected_layers: tuple[str, ...]
    original_blanket_volume: float
    remaining_blanket_volume: float
    void_volume_inside_blanket: float
    liner_volume_inside_blanket: float
    void_volume_outside_blanket: float
    liner_volume_outside_blanket: float
    total_cut_volume: float
    closure_error: float
    maximum_liner_overlap_with_plasma: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortCollisionRecord:
    port_name: str
    coil_id: int | str
    magnet_region_kind: str
    actual_overlap_volume: float
    clearance_envelope_overlap_volume: float
    required_clearance: float
    estimated_minimum_distance: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortSpec:
    name: str
    placement: PortPlacement
    cross_section: PortCrossSection
    extent: PortExtent
    liner: PortLiner
    fill: PortFill
    repetition: PortRepetition
    collision: PortCollisionPolicy
    expected_layers: tuple[str, ...] | None = None
    layer_span: PortLayerSpan | None = None
    resolution: PortResolution | None = None


def _parse_endpoint(
    value: Any, path: str, layer_names: Sequence[str]
) -> PortEndpoint:
    data = _validate_mapping(value, path)
    _ensure_no_unexpected_keys(
        data, path, {"reference", "layer", "fraction", "axial_offset"}
    )
    endpoint = PortEndpoint(
        reference=data["reference"],
        layer=data.get("layer"),
        fraction=data.get("fraction"),
        axial_offset=data.get("axial_offset", 0.0),
    )
    if endpoint.reference == "layer" and endpoint.layer not in layer_names:
        names = ", ".join(layer_names)
        raise ValueError(
            f"endpoint.layer ({endpoint.layer!r}) is not in radial build layers {names}"
        )
    return endpoint


def parse_port_spec(
    name: str, value: Mapping[str, Any], user_layer_names: Sequence[str]
) -> PortSpec:
    data = _validate_mapping(value, f"port[{name!r}]")
    _ensure_no_unexpected_keys(
        data,
        f"port[{name!r}]",
        {
            "name",
            "placement",
            "cross_section",
            "extent",
            "layer_span",
            "liner",
            "fill",
            "repetition",
            "collision",
            "expected_layers",
        },
    )
    if "extent" in data and "layer_span" in data:
        raise ValueError("a port cannot define both extent and layer_span")
    if "extent" not in data and "layer_span" not in data:
        raise ValueError("a port requires extent or deprecated layer_span")

    port_name = _validate_string(
        data.get("name", name), f"port[{name!r}].name"
    )
    placement_data = _validate_mapping(
        data.get("placement", {}), f"port[{name!r}].placement"
    )
    _ensure_no_unexpected_keys(
        placement_data,
        f"port[{name!r}].placement",
        {
            "mode",
            "anchor",
            "axis",
            "reference_direction",
            "max_search_length",
            "roll",
        },
    )
    placement_mode = _validate_string(
        placement_data.get("mode", "cartesian"),
        f"port[{name!r}].placement.mode",
    ).lower()
    if placement_mode == "surface":
        if "reference_direction" in placement_data:
            raise ValueError(
                "surface placement uses placement.roll, not reference_direction"
            )
        anchor_data = _validate_mapping(
            placement_data.get("anchor"),
            f"port[{name!r}].placement.anchor",
        )
        _ensure_no_unexpected_keys(
            anchor_data,
            f"port[{name!r}].placement.anchor",
            {"reference", "toroidal_angle", "poloidal_angle", "layer"},
        )
        axis_data = _validate_mapping(
            placement_data.get("axis"),
            f"port[{name!r}].placement.axis",
        )
        _ensure_no_unexpected_keys(
            axis_data,
            f"port[{name!r}].placement.axis",
            {"mode", "poloidal_tilt", "toroidal_tilt"},
        )
        surface_anchor = PortSurfaceAnchor(
            reference=anchor_data["reference"],
            toroidal_angle=anchor_data["toroidal_angle"],
            poloidal_angle=anchor_data["poloidal_angle"],
            layer=anchor_data.get("layer"),
        )
        if (
            surface_anchor.layer is not None
            and surface_anchor.layer not in user_layer_names
        ):
            names = ", ".join(user_layer_names)
            raise ValueError(
                f"placement.anchor.layer ({surface_anchor.layer!r}) is not "
                f"in radial build layers {names}"
            )
        placement = PortPlacement(
            mode="surface",
            surface_anchor=surface_anchor,
            surface_axis=PortSurfaceAxis(
                mode=axis_data.get("mode", "outward_normal"),
                poloidal_tilt=axis_data.get("poloidal_tilt", 0.0),
                toroidal_tilt=axis_data.get("toroidal_tilt", 0.0),
            ),
            roll=placement_data.get("roll", 0.0),
            max_search_length=placement_data.get("max_search_length"),
        )
    else:
        placement = PortPlacement(
            mode=placement_mode,
            anchor=placement_data["anchor"],
            axis=placement_data["axis"],
            reference_direction=placement_data["reference_direction"],
            roll=placement_data.get("roll", 0.0),
            max_search_length=placement_data.get("max_search_length"),
        )

    if "cross_section" not in data:
        raise ValueError(f"port[{name!r}].cross_section is required")
    cross_data = _validate_mapping(
        data["cross_section"], f"port[{name!r}].cross_section"
    )
    _ensure_no_unexpected_keys(
        cross_data,
        f"port[{name!r}].cross_section",
        {"shape", "radius", "width", "height", "dimensions_are"},
    )
    cross_section = PortCrossSection(
        shape=cross_data["shape"],
        radius=cross_data.get("radius"),
        width=cross_data.get("width"),
        height=cross_data.get("height"),
        dimensions_are=cross_data.get("dimensions_are", "clear_aperture"),
    )

    layer_span = None
    resolution = None
    if "layer_span" in data:
        span_data = _validate_mapping(
            data["layer_span"], f"port[{name!r}].layer_span"
        )
        _ensure_no_unexpected_keys(
            span_data,
            f"port[{name!r}].layer_span",
            {"start", "count", "direction"},
        )
        layer_span = PortLayerSpan(
            start=span_data["start"],
            count=span_data["count"],
            direction=span_data["direction"],
        )
        resolution = layer_span.resolve(user_layer_names)
        extent = PortExtent(
            start=PortEndpoint(
                reference="layer", layer=resolution.layers[0], fraction=0.0
            ),
            end=PortEndpoint(
                reference="layer", layer=resolution.layers[-1], fraction=1.0
            ),
        )
        warnings.warn(
            "port.layer_span is deprecated; use explicit port.extent endpoints",
            DeprecationWarning,
            stacklevel=3,
        )
    else:
        extent_data = _validate_mapping(
            data["extent"], f"port[{name!r}].extent"
        )
        _ensure_no_unexpected_keys(
            extent_data,
            f"port[{name!r}].extent",
            {"start", "end", "outer_extension"},
        )
        extent = PortExtent(
            start=_parse_endpoint(
                extent_data["start"],
                f"port[{name!r}].extent.start",
                user_layer_names,
            ),
            end=_parse_endpoint(
                extent_data["end"],
                f"port[{name!r}].extent.end",
                user_layer_names,
            ),
            outer_extension=extent_data.get("outer_extension", 0.0),
        )

    liner_data = _validate_mapping(
        data.get("liner", {}), f"port[{name!r}].liner"
    )
    _ensure_no_unexpected_keys(
        liner_data,
        f"port[{name!r}].liner",
        {"enabled", "thickness", "mat_tag"},
    )
    liner = PortLiner(
        enabled=liner_data.get("enabled", False),
        thickness=liner_data.get("thickness", 0.0),
        mat_tag=liner_data.get("mat_tag", "SS316L"),
    )

    fill_data = _validate_mapping(data.get("fill", {}), f"port[{name!r}].fill")
    _ensure_no_unexpected_keys(fill_data, f"port[{name!r}].fill", {"mat_tag"})
    fill = PortFill(mat_tag=fill_data.get("mat_tag", "Vacuum"))

    repetition_data = _validate_mapping(
        data.get("repetition", {}), f"port[{name!r}].repetition"
    )
    _ensure_no_unexpected_keys(
        repetition_data, f"port[{name!r}].repetition", {"mode"}
    )
    repetition = PortRepetition(mode=repetition_data.get("mode", "single"))

    collision_data = _validate_mapping(
        data.get("collision", {}), f"port[{name!r}].collision"
    )
    _ensure_no_unexpected_keys(
        collision_data,
        f"port[{name!r}].collision",
        {"magnet_policy", "clearance_policy", "minimum_magnet_clearance"},
    )
    collision = PortCollisionPolicy(
        magnet_policy=collision_data.get("magnet_policy", "error"),
        clearance_policy=collision_data.get("clearance_policy", "warn"),
        minimum_magnet_clearance=collision_data.get(
            "minimum_magnet_clearance", 0.0
        ),
    )

    expected_layers = None
    if "expected_layers" in data:
        expected_layers = tuple(
            _validate_string(item, "expected_layers item")
            for item in _validate_sequence(
                data["expected_layers"], "expected_layers"
            )
        )
        unknown = set(expected_layers) - set(user_layer_names)
        if unknown:
            raise ValueError(
                f"expected_layers contains unknown layers: {sorted(unknown)}"
            )
        if len(set(expected_layers)) != len(expected_layers):
            raise ValueError("expected_layers must not contain duplicates")

    return PortSpec(
        name=port_name,
        placement=placement,
        cross_section=cross_section,
        extent=extent,
        liner=liner,
        fill=fill,
        repetition=repetition,
        collision=collision,
        expected_layers=expected_layers,
        layer_span=layer_span,
        resolution=resolution,
    )


def parse_ports(
    value: Any, user_layer_names: Sequence[str]
) -> tuple[PortSpec, ...]:
    """Parse and validate ``invessel_build["ports"]``."""
    if value is None:
        return ()
    ports = _validate_sequence(value, "ports")
    parsed: list[PortSpec] = []
    names: set[str] = set()
    for index, port_data in enumerate(ports):
        if not isinstance(port_data, Mapping):
            raise TypeError(f"ports[{index}] must be a mapping")
        port = parse_port_spec(f"ports[{index}]", port_data, user_layer_names)
        if port.name in names:
            raise ValueError(f"duplicate port name {port.name!r}")
        names.add(port.name)
        parsed.append(port)
    return tuple(parsed)
