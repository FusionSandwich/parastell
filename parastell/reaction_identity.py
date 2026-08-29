"""Canonical isotope/MT identities shared by tally and export adapters."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


MT_LABELS = {
    2: "elastic",
    4: "inelastic",
    16: "n,2n",
    102: "n,gamma",
    103: "n,p",
    107: "n,alpha",
}
_NUCLIDE = re.compile(r"^[A-Z][a-z]?[0-9]+(?:_m[0-9]+)?$")
MATERIAL_MT_SCHEMA = "parastell.material_derived_nuclide_mt_requests/v1.0.0"


def canonical_nuclide(value: Any) -> str:
    name = str(value).strip().replace("-", "")
    if not _NUCLIDE.fullmatch(name):
        raise ValueError(f"invalid OpenMC nuclide identity: {value!r}")
    return name


def canonical_mt(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("MT number cannot be boolean")
    try:
        mt = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid MT number: {value!r}") from exc
    if str(value).strip() != str(mt) and not isinstance(value, int):
        raise ValueError(f"invalid MT number: {value!r}")
    if mt <= 0 or mt > 999:
        raise ValueError(f"MT number is outside the supported range: {mt}")
    return mt


def mt_label(mt: Any) -> str:
    number = canonical_mt(mt)
    return MT_LABELS.get(number, f"MT={number}")


def canonicalize_nuclide_mt_requests(
    requests: Mapping[str, Sequence[int | str]],
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for nuclide, values in sorted(requests.items()):
        name = canonical_nuclide(nuclide)
        mts = [canonical_mt(value) for value in values]
        if not mts:
            raise ValueError(f"nuclide {name} has no requested MT reactions")
        if len(mts) != len(set(mts)):
            raise ValueError(f"nuclide {name} repeats an MT reaction")
        result[name] = sorted(mts)
    return result


def derive_material_nuclide_mt_requests(
    materials: Sequence[Mapping[str, Any]],
    *,
    default_mts: Sequence[int | str],
    nuclide_overrides: Mapping[str, Sequence[int | str]] | None = None,
) -> dict[str, Any]:
    """Derive isotope-specific MT requests from bound material records.

    The reaction policy remains explicit: every positive-fraction isotope gets
    ``default_mts`` unless an exact nuclide override is supplied.  Natural
    elements and zero-fraction placeholders are never promoted to requests.
    """
    if (
        not isinstance(materials, Sequence)
        or isinstance(materials, (str, bytes))
        or not materials
    ):
        raise ValueError("materials must be a nonempty sequence")
    default = [canonical_mt(value) for value in default_mts]
    if not default or len(default) != len(set(default)):
        raise ValueError("default MT policy must be nonempty and unique")
    overrides = canonicalize_nuclide_mt_requests(nuclide_overrides or {})
    material_bindings: dict[str, dict[str, Any]] = {}
    observed: set[str] = set()
    for row in materials:
        if not isinstance(row, Mapping):
            raise ValueError("material record must be a mapping")
        material_id = str(row.get("material_id", "")).strip()
        digest = str(row.get("composition_sha256", "")).lower()
        isotopes = row.get("isotopes")
        if (
            not material_id
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(isotopes, Mapping)
            or not isotopes
        ):
            raise ValueError(
                "material isotope identity/provenance is incomplete"
            )
        positive: list[str] = []
        total_fraction = 0.0
        for raw_nuclide, raw_fraction in isotopes.items():
            nuclide = canonical_nuclide(raw_nuclide)
            fraction = float(raw_fraction)
            if not math.isfinite(fraction) or fraction < 0.0:
                raise ValueError("material isotope fraction is invalid")
            total_fraction += fraction
            if fraction > 0.0:
                positive.append(nuclide)
                observed.add(nuclide)
        if not positive:
            raise ValueError("material has no positive-fraction isotopes")
        if not math.isclose(
            total_fraction, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-8
        ):
            raise ValueError("material isotope fractions must sum to one")
        if len(positive) != len(set(positive)):
            raise ValueError("material repeats a canonical isotope identity")
        binding = {
            "material_id": material_id,
            "composition_sha256": digest,
            "positive_fraction_nuclides": sorted(set(positive)),
        }
        previous = material_bindings.get(material_id)
        if previous is not None and previous != binding:
            raise ValueError(
                "material ID has conflicting isotope compositions"
            )
        material_bindings[material_id] = binding
    unknown_overrides = sorted(set(overrides) - observed)
    if unknown_overrides:
        raise ValueError(
            "nuclide MT overrides are absent from materials: "
            + ", ".join(unknown_overrides)
        )
    requests = {
        nuclide: list(overrides.get(nuclide, sorted(default)))
        for nuclide in sorted(observed)
    }
    result = {
        "schema": MATERIAL_MT_SCHEMA,
        "status": "MATERIAL_ISOTOPE_MT_REQUESTS_DERIVED",
        "policy": {
            "default_mts": sorted(default),
            "nuclide_overrides": overrides,
            "zero_fraction_isotopes_requested": False,
            "natural_elements_allowed": False,
        },
        "materials": [
            material_bindings[key] for key in sorted(material_bindings)
        ],
        "nuclide_mt_requests": requests,
        "missing_semantics": "MISSING_IS_NOT_ZERO",
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["derivation_sha256"] = hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    return result


def reaction_matrix_identity(
    *, nuclide: Any, mt: Any, nuclear_data_sha256: str
) -> dict[str, Any]:
    digest = str(nuclear_data_sha256).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("reaction identity needs a nuclear-data SHA-256")
    number = canonical_mt(mt)
    return {
        "nuclide": canonical_nuclide(nuclide),
        "mt": number,
        "reaction_label": mt_label(number),
        "nuclear_data_sha256": digest,
        "missing_semantics": "MISSING_IS_NOT_ZERO",
    }
