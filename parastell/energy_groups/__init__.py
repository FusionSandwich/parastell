"""Versioned neutron and photon energy-structure registry."""

from .registry import compare_structures
from .registry import EnergyGroupStructure
from .registry import get_structure
from .registry import list_structures
from .registry import load_custom_structure
from .registry import validate_edges

__all__ = [
    "EnergyGroupStructure",
    "compare_structures",
    "get_structure",
    "list_structures",
    "load_custom_structure",
    "validate_edges",
]
