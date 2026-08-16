"""Port-aware extensions for in-vessel model construction.

This module focuses on parsing and validating the prompt-1 port input contract.
Geometry generation for ports is deferred to a future prompt and is therefore
not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class PortGeometryNotImplementedError(NotImplementedError):
    """Raised when port geometry requests reach the generation stage."""


def _validate_mapping(
    value: Any,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        e = TypeError(f"{path} must be a mapping")
        raise e
    return value


def _validate_sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence):
        e = TypeError(f"{path} must be a sequence")
        raise e
    return value


def _validate_string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        e = TypeError(f"{path} must be a string")
        raise e
    if not value.strip():
        e = ValueError(f"{path} cannot be empty")
        raise e
    return value


def _validate_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int):
        e = TypeError(f"{path} must be an integer")
        raise e
    if value < 1:
        e = ValueError(f"{path} must be positive")
        raise e
    return value


def _validate_finite_scalar(value: Any, path: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        e = ValueError(f"{path} must be finite")
        raise e
    return number


def _to_vector3(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, Sequence):
        e = TypeError(f"{path} must be a three-element sequence")
        raise e
    vector = np.asarray(list(value), dtype=float)
    if vector.shape != (3,):
        e = ValueError(f"{path} must be a three-element vector")
        raise e
    if np.any(~np.isfinite(vector)):
        e = ValueError(f"{path} values must be finite")
        raise e
    return vector


def _normalize_vector(vector: np.ndarray, path: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        e = ValueError(f"{path} cannot be a zero vector")
        raise e
    return vector / norm


def _ensure_no_unexpected_keys(
    mapping: Mapping[str, Any],
    path: str,
    allowed: set[str],
) -> None:
    unknown = set(mapping.keys()) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        e = ValueError(f"{path} contains unknown keys: {names}")
        raise e


def _resolve_layer_indices(
    layer_names: Sequence[str],
    start: str,
    count: int,
    direction: str,
) -> list[str]:
    if start not in layer_names:
        names = ", ".join(layer_names)
        e = ValueError(
            f"layer_span.start ({start!r}) is not in radial build layers {names}"
        )
        raise e

    start_index = layer_names.index(start)
    step = 1 if direction == "outward" else -1
    stop_index = start_index + step * (count - 1)

    if stop_index < 0 or stop_index >= len(layer_names):
        e = ValueError(
            "layer_span extends beyond available radial layers; clipping is not allowed"
        )
        raise e

    if step == 1:
        index_range = range(start_index, stop_index + 1)
    else:
        index_range = range(start_index, stop_index - 1, -1)

    return [layer_names[index] for index in index_range]


@dataclass(frozen=True)
class PortPlacement:
    """Placement metadata for a port anchor and search orientation."""

    anchor: tuple[float, float, float]
    axis: tuple[float, float, float]
    reference_direction: tuple[float, float, float]
    mode: str = "cartesian"
    max_search_length: float | None = None
    local_axis: tuple[float, float, float] | None = None
    local_reference: tuple[float, float, float] | None = None
    local_normal: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        mode = _validate_string(self.mode, "placement.mode").lower()
        if mode != "cartesian":
            e = ValueError("placement.mode must be 'cartesian'")
            raise e
        anchor = _to_vector3(self.anchor, "placement.anchor")
        axis = _to_vector3(self.axis, "placement.axis")
        reference = _to_vector3(
            self.reference_direction,
            "placement.reference_direction",
        )

        local_axis = _normalize_vector(axis, "placement.axis")
        reference = _normalize_vector(
            reference, "placement.reference_direction"
        )

        if np.allclose(reference, local_axis) or np.allclose(
            reference, -local_axis
        ):
            e = ValueError(
                "placement.reference_direction must not be parallel to placement.axis"
            )
            raise e

        local_reference = np.cross(local_axis, reference)
        local_reference = _normalize_vector(local_reference, "frame reference")

        local_normal = np.cross(local_axis, local_reference)
        local_normal = _normalize_vector(local_normal, "frame normal")

        max_search_length = float(
            max(1000.0, 2.0 * float(np.linalg.norm(anchor)))
            if self.max_search_length is None
            else _validate_finite_scalar(
                self.max_search_length,
                "placement.max_search_length",
            )
        )

        object.__setattr__(self, "anchor", tuple(anchor))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "axis", tuple(local_axis))
        object.__setattr__(
            self,
            "reference_direction",
            tuple(reference),
        )
        object.__setattr__(self, "max_search_length", max_search_length)
        object.__setattr__(self, "local_axis", tuple(local_axis))
        object.__setattr__(self, "local_reference", tuple(local_reference))
        object.__setattr__(self, "local_normal", tuple(local_normal))


@dataclass(frozen=True)
class PortCrossSection:
    """Cross-section definition for a port cutout."""

    shape: str
    radius: float | None = None
    width: float | None = None
    height: float | None = None

    def __post_init__(self) -> None:
        shape = _validate_string(self.shape, "cross_section.shape").lower()
        if shape not in {"circle", "rectangle"}:
            e = ValueError(
                "cross_section.shape must be one of {'circle', 'rectangle'}"
            )
            raise e

        if shape == "circle":
            if self.width is not None or self.height is not None:
                e = ValueError("circle cross_section must define only radius")
                raise e
            if self.radius is None:
                e = ValueError("circle cross_section requires a radius")
                raise e
            radius = _validate_finite_scalar(
                self.radius, "cross_section.radius"
            )
            if radius <= 0.0:
                e = ValueError("cross_section.radius must be positive")
                raise e
            object.__setattr__(self, "radius", radius)
            object.__setattr__(self, "width", None)
            object.__setattr__(self, "height", None)
        else:
            if self.radius is not None:
                e = ValueError(
                    "rectangle cross_section must not define radius"
                )
                raise e
            if self.width is None or self.height is None:
                e = ValueError(
                    "rectangle cross_section must define both width and height"
                )
                raise e
            width = _validate_finite_scalar(self.width, "cross_section.width")
            height = _validate_finite_scalar(
                self.height, "cross_section.height"
            )
            if width <= 0.0:
                e = ValueError("cross_section.width must be positive")
                raise e
            if height <= 0.0:
                e = ValueError("cross_section.height must be positive")
                raise e
            object.__setattr__(self, "shape", shape)
            object.__setattr__(self, "width", width)
            object.__setattr__(self, "height", height)
            object.__setattr__(self, "radius", None)

        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class PortLayerSpan:
    """Named radial layer span that will be traversed for a port."""

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
        direction = _validate_string(self.direction, "layer_span.direction")
        direction = direction.lower()
        if direction not in {"inward", "outward"}:
            e = ValueError(
                "layer_span.direction must be one of {'inward', 'outward'}"
            )
            raise e
        object.__setattr__(self, "direction", direction)

    def resolve(self, layer_names: Sequence[str]) -> "PortResolution":
        names = list(layer_names)
        layers = _resolve_layer_indices(
            names, self.start, self.count, self.direction
        )
        return PortResolution(
            layers=tuple(layers),
            start=self.start,
            count=self.count,
            direction=self.direction,
        )


@dataclass(frozen=True)
class PortFill:
    """Material assignment for port volumes."""

    mat_tag: str = "Vacuum"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mat_tag", _validate_string(self.mat_tag, "mat_tag")
        )


@dataclass(frozen=True)
class PortRepetition:
    """Repetition metadata for multiple ports."""

    mode: str = "single"

    def __post_init__(self) -> None:
        mode = _validate_string(self.mode, "repetition.mode").lower()
        if mode not in {"single", "per_period"}:
            e = ValueError(
                "repetition.mode currently supports only 'single' and "
                "'per_period'"
            )
            raise e
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class PortResolution:
    """Resolved layer span for a parsed :class:`PortLayerSpan`."""

    layers: tuple[str, ...]
    start: str
    count: int
    direction: str

    def __post_init__(self) -> None:
        if not self.layers:
            e = ValueError("port resolution must include at least one layer")
            raise e
        object.__setattr__(
            self, "start", _validate_string(self.start, "start")
        )
        object.__setattr__(
            self,
            "count",
            _validate_positive_int(self.count, "port resolution count"),
        )
        direction = _validate_string(self.direction, "direction").lower()
        if direction not in {"inward", "outward"}:
            e = ValueError(
                "port resolution direction must be one of {'inward', 'outward'}"
            )
            raise e
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "layers", tuple(self.layers))


@dataclass(frozen=True)
class PortSpec:
    """Top-level port contract for one single port."""

    name: str
    placement: PortPlacement
    cross_section: PortCrossSection
    layer_span: PortLayerSpan
    fill: PortFill
    repetition: PortRepetition
    resolution: PortResolution


def parse_port_spec(
    name: str,
    value: Mapping[str, Any],
    user_layer_names: Sequence[str],
) -> PortSpec:
    port_data = _validate_mapping(value, f"port[{name!r}]")

    _ensure_no_unexpected_keys(
        port_data,
        f"port[{name!r}]",
        {
            "name",
            "placement",
            "cross_section",
            "layer_span",
            "fill",
            "repetition",
        },
    )

    if "name" in port_data:
        port_name = _validate_string(port_data["name"], f"port[{name!r}].name")
    else:
        port_name = _validate_string(name, f"port[{name!r}].name")

    placement_data = _validate_mapping(
        port_data.get("placement", {}),
        f"port[{name!r}].placement",
    )
    _ensure_no_unexpected_keys(
        placement_data,
        f"port[{name!r}].placement",
        {"mode", "anchor", "axis", "reference_direction", "max_search_length"},
    )
    placement = PortPlacement(
        mode=placement_data.get("mode", "cartesian"),
        anchor=placement_data["anchor"],
        axis=placement_data["axis"],
        reference_direction=placement_data["reference_direction"],
        max_search_length=placement_data.get(
            "max_search_length",
            None,
        ),
    )

    if "cross_section" not in port_data:
        e = ValueError(f"port[{name!r}].cross_section is required")
        raise e

    cross_section_data = _validate_mapping(
        port_data["cross_section"],
        f"port[{name!r}].cross_section",
    )
    _ensure_no_unexpected_keys(
        cross_section_data,
        f"port[{name!r}].cross_section",
        {"shape", "radius", "width", "height"},
    )
    cross_section = PortCrossSection(
        shape=cross_section_data["shape"],
        radius=cross_section_data.get("radius"),
        width=cross_section_data.get("width"),
        height=cross_section_data.get("height"),
    )

    layer_span_data = _validate_mapping(
        port_data["layer_span"],
        f"port[{name!r}].layer_span",
    )
    _ensure_no_unexpected_keys(
        layer_span_data,
        f"port[{name!r}].layer_span",
        {"start", "count", "direction"},
    )
    layer_span = PortLayerSpan(
        start=layer_span_data["start"],
        count=layer_span_data["count"],
        direction=layer_span_data["direction"],
    )

    fill_data = _validate_mapping(
        port_data.get("fill", {}),
        f"port[{name!r}].fill",
    )
    _ensure_no_unexpected_keys(
        fill_data,
        f"port[{name!r}].fill",
        {"mat_tag"},
    )
    fill = PortFill(mat_tag=fill_data.get("mat_tag", "Vacuum"))

    repetition_data = _validate_mapping(
        port_data.get("repetition", {}),
        f"port[{name!r}].repetition",
    )
    _ensure_no_unexpected_keys(
        repetition_data,
        f"port[{name!r}].repetition",
        {"mode"},
    )
    repetition = PortRepetition(mode=repetition_data.get("mode", "single"))

    layer_resolution = layer_span.resolve(user_layer_names)

    return PortSpec(
        name=port_name,
        placement=placement,
        cross_section=cross_section,
        layer_span=layer_span,
        fill=fill,
        repetition=repetition,
        resolution=layer_resolution,
    )


def parse_ports(
    value: Any,
    user_layer_names: Sequence[str],
) -> tuple[PortSpec, ...]:
    """Parse and validate ``invessel_build["ports"]`` from input data."""
    if value is None:
        return ()

    ports = _validate_sequence(value, "ports")
    if len(ports) == 0:
        return ()

    parsed_ports: list[PortSpec] = []
    names: set[str] = set()
    for idx, port_data in enumerate(ports):
        if not isinstance(port_data, Mapping):
            e = TypeError(f"ports[{idx}] must be a mapping")
            raise e
        port_name = f"ports[{idx}]"
        port = parse_port_spec(port_name, port_data, user_layer_names)
        if port.name in names:
            e = ValueError(f"duplicate port name {port.name!r}")
            raise e
        names.add(port.name)
        parsed_ports.append(port)

    return tuple(parsed_ports)
