"""Fail-closed gate for a ParaStell/OpenMC production deck."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_GATES = (
    "geometry_full_assembly",
    "winding_pack_envelopes",
    "outer_casing_envelopes",
    "boundary_localization",
    "surface_bank_completeness",
    "short_run_field_analysis",
    "source_mesh_convergence",
    "all_magnet_coarse_field",
    "selected_magnet_local_field",
    "parallel_scaling",
)


def evaluate_production_deck_gate(
    gates: Mapping[str, Any],
    *,
    required_gates: Sequence[str] = REQUIRED_GATES,
) -> dict[str, Any]:
    """Return readiness only when every explicitly required gate passes."""

    if not isinstance(gates, Mapping):
        raise TypeError("gates must be a mapping")
    missing = [name for name in required_gates if name not in gates]
    nonpassing = []
    for name in required_gates:
        if name not in gates:
            continue
        value = gates[name]
        if value is not True and value != "PASS":
            nonpassing.append(name)
    ready = not missing and not nonpassing
    return {
        "status": "PRODUCTION_DECK_READY" if ready else "BLOCKED_GATES_FAILED",
        "ready": ready,
        "required_gates": list(required_gates),
        "missing_gates": missing,
        "nonpassing_gates": nonpassing,
        "submission_authorized": False,
        "submission_policy": (
            "A separate explicit user authorization is required after all "
            "gates pass."
        ),
    }
