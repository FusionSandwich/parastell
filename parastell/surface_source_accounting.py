"""Compatibility guard for retired assertion-only surface accounting."""

from __future__ import annotations

from collections.abc import Mapping


BANK_CLASSIFICATIONS = {
    "COMPLETE_CROSSING_BANK",
    "SAMPLED_CROSSING_BANK",
    "TRUNCATED_INVALID_BANK",
}


def validate_surface_source_accounting(evidence: Mapping) -> dict:
    """Reject caller-authored accounting receipts.

    Production completeness must be derived by
    :func:`parastell.openmc16_export.audit_openmc16_surface_run`, which parses
    and hash-binds the H5M topology, OpenMC model, statepoint tally results,
    native source-bank HDF5 files, and terminal log. Retaining this name as a
    hard failure prevents older callers from silently regaining the former
    assertion-only trust boundary.
    """

    del evidence
    raise RuntimeError(
        "caller-asserted surface-source accounting is disabled; use "
        "parastell.openmc16_export.audit_openmc16_surface_run so H5M, "
        "model.xml, statepoint, native source HDF5, and terminal log are "
        "parsed and hash-bound"
    )
