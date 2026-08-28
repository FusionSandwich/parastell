"""Frozen public-example inputs used only to derive recovery candidates."""

from __future__ import annotations

import numpy as np


TOROIDAL_ANGLES_DEG = np.linspace(0.0, 90.0, 9)
POLOIDAL_ANGLES_DEG = np.linspace(0.0, 360.0, 9)
WALL_S = 1.08
MAGNET_WIDTH_CM = 40.0
MAGNET_THICKNESS_CM = 50.0
TOROIDAL_EXTENT_DEG = 90.0
MAGNET_SAMPLE_MOD = 6
SOURCE_CFS_SAMPLES = 11
SOURCE_POLOIDAL_SAMPLES = 61
SOURCE_TOROIDAL_SAMPLES = 61


def public_reference_radial_build() -> dict[str, dict]:
    """Return a new copy of the exact public R1 radial-build dictionary."""

    unit = np.ones((9, 9), dtype=float)
    breeder = np.array(
        [
            [75, 75, 75, 25, 25, 25, 75, 75, 75],
            [75, 75, 75, 25, 25, 75, 75, 75, 75],
            [75, 75, 25, 25, 75, 75, 75, 75, 75],
            [65, 25, 25, 65, 75, 75, 75, 75, 65],
            [45, 45, 75, 75, 75, 75, 75, 45, 45],
            [65, 75, 75, 75, 75, 65, 25, 25, 65],
            [75, 75, 75, 75, 75, 25, 25, 75, 75],
            [75, 75, 75, 75, 25, 25, 75, 75, 75],
            [75, 75, 75, 25, 25, 25, 75, 75, 75],
        ],
        dtype=float,
    )
    return {
        "first_wall": {"thickness_matrix": unit * 5.0},
        "breeder": {"thickness_matrix": breeder},
        "back_wall": {"thickness_matrix": unit * 5.0},
        "shield": {"thickness_matrix": unit * 50.0},
        "vacuum_vessel": {
            "thickness_matrix": unit * 10.0,
            "mat_tag": "vac_vessel",
        },
    }


def thickness_matrices(radial_build: dict[str, dict]) -> dict[str, np.ndarray]:
    """Extract copied thickness matrices while preserving layer order."""

    return {
        name: np.array(layer["thickness_matrix"], dtype=float, copy=True)
        for name, layer in radial_build.items()
    }


def replace_thickness_matrices(
    radial_build: dict[str, dict],
    matrices: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Return copied build data with only thickness matrices replaced."""

    if tuple(radial_build) != tuple(matrices):
        raise ValueError("replacement layer order and identity must match")
    replaced = {}
    for name, layer in radial_build.items():
        copied = dict(layer)
        copied["thickness_matrix"] = np.array(
            matrices[name], dtype=float, copy=True
        )
        replaced[name] = copied
    return replaced
