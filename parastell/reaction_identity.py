"""Canonical isotope/MT identities shared by tally and export adapters."""

from __future__ import annotations

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
