"""Fail-closed OpenMC compatibility policy for the 0.16 release series."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any


OPENMC_MINIMUM = (0, 16, 0)
OPENMC_NEXT_UNQUALIFIED = (0, 17, 0)
_VERSION = re.compile(
    r"^\s*(\d+)\.(\d+)\.(\d+)"
    r"(?:(?:\.dev|-dev|rc|a|b)\d+)?(?:\+[0-9A-Za-z.-]+)?\s*$"
)


def parse_openmc_version(value: Any) -> tuple[int, int, int]:
    """Return the semantic release triplet from an OpenMC version string."""
    match = _VERSION.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid OpenMC version: {value!r}")
    return tuple(int(item) for item in match.groups())


def openmc_version_supported(value: Any) -> bool:
    """Return whether *value* is in the qualified OpenMC 0.16 series."""
    try:
        version = parse_openmc_version(value)
    except ValueError:
        return False
    rendered = str(value).strip()
    if version == OPENMC_MINIMUM and re.search(
        r"(?:\.dev|-dev|rc|a|b)\d+", rendered
    ):
        return False
    return OPENMC_MINIMUM <= version < OPENMC_NEXT_UNQUALIFIED


def statepoint_version_supported(value: Sequence[Any]) -> bool:
    """Return whether an HDF5 OpenMC version vector is supported."""
    if len(value) < 3:
        return False
    try:
        version = tuple(int(item) for item in value[:3])
    except (TypeError, ValueError):
        return False
    return OPENMC_MINIMUM <= version < OPENMC_NEXT_UNQUALIFIED


def require_supported_openmc_version(
    value: Any, *, context: str = "OpenMC runtime"
) -> str:
    """Validate and return a supported OpenMC version string."""
    version = str(value)
    if not openmc_version_supported(version):
        raise RuntimeError(
            f"{context}: OpenMC 0.16.0 required as the minimum; "
            f"supported range is >=0.16.0,<0.17.0; found {version}"
        )
    return version
