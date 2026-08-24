"""Public ParaStell API with lazy optional transport dependencies."""

from importlib import import_module

from .hts_multilayer import HTSLayer
from .hts_multilayer import MaterialResponseLibrary
from .hts_multilayer import MultilayerStack
from .hts_multilayer import ReplaySummary
from .hts_multilayer import replay_phase_space
from .hts_multilayer import verification_rebco_stack


_LAZY_EXPORTS = {
    "CoordinateFrame": (".magnet_spectral_handoff", "CoordinateFrame"),
    "MagnetRegion": (".magnet_spectral_handoff", "MagnetRegion"),
    "MagnetCouplingPlane": (
        ".magnet_spectral_handoff",
        "MagnetCouplingPlane",
    ),
    "MagnetSpectralHandoff": (
        ".magnet_spectral_handoff",
        "MagnetSpectralHandoff",
    ),
    "MeshSpec": (".magnet_spectral_handoff", "MeshSpec"),
    "available_energy_group_structures": (
        ".magnet_spectral_handoff",
        "available_energy_group_structures",
    ),
    "load_energy_group_edges": (
        ".magnet_spectral_handoff",
        "load_energy_group_edges",
    ),
    "software_validation_energy_bounds": (
        ".magnet_spectral_handoff",
        "software_validation_energy_bounds",
    ),
    "CorrelatedBoundaryBank": (
        ".magnet_boundary_envelope",
        "CorrelatedBoundaryBank",
    ),
    "EnvelopeSurface": (".magnet_boundary_envelope", "EnvelopeSurface"),
    "MagnetBoundaryEnvelope": (
        ".magnet_boundary_envelope",
        "MagnetBoundaryEnvelope",
    ),
    "authoritative_energy_edges": (
        ".magnet_boundary_envelope",
        "authoritative_energy_edges",
    ),
    "build_correlated_bank": (
        ".magnet_boundary_envelope",
        "build_correlated_bank",
    ),
    "condition_on_independent_current": (
        ".magnet_boundary_envelope",
        "condition_on_independent_current",
    ),
    "derive_tally_conditioned_bank": (
        ".magnet_boundary_envelope",
        "derive_tally_conditioned_bank",
    ),
    "classify_crossing_bank": (
        ".magnet_boundary_envelope",
        "classify_crossing_bank",
    ),
    "canonical_dagmc_fingerprint": (
        ".dagmc_envelope",
        "canonical_dagmc_fingerprint",
    ),
    "write_radiation_field_bundle": (
        ".magnet_radiation_field_bundle",
        "write_radiation_field_bundle",
    ),
    "read_radiation_field_bundle": (
        ".magnet_radiation_field_bundle",
        "read_radiation_field_bundle",
    ),
    "export_volume_scalar_flux": (
        ".magnet_volume_flux",
        "export_volume_scalar_flux",
    ),
    "conservative_projection": (
        ".magnet_boundary_envelope",
        "conservative_projection",
    ),
    "production_mu_edges": (
        ".magnet_boundary_envelope",
        "production_mu_edges",
    ),
    "production_phi_edges": (
        ".magnet_boundary_envelope",
        "production_phi_edges",
    ),
    "read_handoff": (".magnet_boundary_envelope", "read_handoff"),
    "source_mesh_provenance": (
        ".magnet_boundary_envelope",
        "source_mesh_provenance",
    ),
    "write_handoff": (".magnet_boundary_envelope", "write_handoff"),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "HTSLayer",
    "MaterialResponseLibrary",
    "MultilayerStack",
    "ReplaySummary",
    "replay_phase_space",
    "verification_rebco_stack",
    *_LAZY_EXPORTS,
]
