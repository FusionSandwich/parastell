"""ParaStell package with dependency-safe optional radiation features."""

from __future__ import annotations

from importlib import import_module


__version__ = "0.1.0"
__all__ = (
    "Stellarator",
    "MagnetRadiationFieldProducer",
    "MagnetRadiationWorkflow",
)


def __getattr__(name):
    """Load CAD, DAGMC, and workflow layers only when explicitly requested."""
    if name == "Stellarator":
        return import_module(".parastell", __name__).Stellarator
    if name == "MagnetRadiationFieldProducer":
        module = import_module(".magnet_radiation_field", __name__)
        return module.MagnetRadiationFieldProducer
    if name == "MagnetRadiationWorkflow":
        module = import_module(".magnet_field_workflow", __name__)
        return module.MagnetRadiationWorkflow
    raise AttributeError(name)
