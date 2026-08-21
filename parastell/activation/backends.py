"""Activation backend discovery and explicit selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil
from typing import Any, Mapping


class ActivationBackend(str, Enum):
    OPENMC_R2S = "openmc-r2s"
    ALARA = "alara"
    FISPACT_II = "fispact-ii"


@dataclass(frozen=True)
class BackendCapability:
    backend: ActivationBackend
    available: bool
    complete: bool
    executable: str | None
    version: str | None
    data_paths: tuple[str, ...]
    missing: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "available": self.available,
            "complete": self.complete,
            "executable": self.executable,
            "version": self.version,
            "data_paths": list(self.data_paths),
            "missing": list(self.missing),
        }


def _external_capability(
    backend: ActivationBackend,
    command: str,
    executable: str | Path | None,
    data_paths: tuple[str | Path, ...],
) -> BackendCapability:
    located = str(executable) if executable else shutil.which(command)
    missing: list[str] = []
    if located is None or not Path(located).is_file():
        missing.append("executable")
        located = None
    resolved_data: list[str] = []
    for data_path in data_paths:
        resolved = Path(data_path).expanduser().resolve()
        resolved_data.append(str(resolved))
        if not resolved.exists():
            missing.append(f"data:{resolved}")
    return BackendCapability(
        backend=backend,
        available=located is not None,
        complete=not missing,
        executable=located,
        version=None,
        data_paths=tuple(resolved_data),
        missing=tuple(missing),
    )


def detect_activation_backends(
    *,
    openmc_report: Mapping[str, Any] | None = None,
    alara_executable: str | Path | None = None,
    alara_data: tuple[str | Path, ...] = (),
    fispact_executable: str | Path | None = None,
    fispact_data: tuple[str | Path, ...] = (),
) -> dict[ActivationBackend, BackendCapability]:
    """Inspect backend availability without running or installing software."""

    report = dict(openmc_report or {})
    required = ("R2SManager", "DecaySpectrum", "ParentNuclideFilter")
    missing = [name for name in required if not report.get(name, False)]
    if not report.get("passes", False):
        missing.append("OpenMC 0.16 capability gate")
    openmc_capability = BackendCapability(
        backend=ActivationBackend.OPENMC_R2S,
        available=bool(report),
        complete=bool(report) and not missing,
        executable=report.get("executable"),
        version=report.get("version"),
        data_paths=tuple(report.get("data_paths", ())),
        missing=tuple(dict.fromkeys(missing)),
    )
    return {
        ActivationBackend.OPENMC_R2S: openmc_capability,
        ActivationBackend.ALARA: _external_capability(
            ActivationBackend.ALARA,
            "alara",
            alara_executable,
            alara_data,
        ),
        ActivationBackend.FISPACT_II: _external_capability(
            ActivationBackend.FISPACT_II,
            "fispact",
            fispact_executable,
            fispact_data,
        ),
    }


def select_activation_backend(
    capabilities: Mapping[ActivationBackend, BackendCapability],
) -> dict[str, Any]:
    """Apply OpenMC, ALARA, FISPACT priority without silent fallback."""

    order = (
        ActivationBackend.OPENMC_R2S,
        ActivationBackend.ALARA,
        ActivationBackend.FISPACT_II,
    )
    selected = next(
        (item for item in order if capabilities[item].complete), None
    )
    if selected is None:
        details = "; ".join(
            f"{item.value}: {', '.join(capabilities[item].missing) or 'unknown'}"
            for item in order
        )
        raise RuntimeError(f"no complete activation backend: {details}")
    fispact = capabilities[ActivationBackend.FISPACT_II]
    return {
        "selected": selected.value,
        "selection_order": [item.value for item in order],
        "fispact_independent_benchmark": (
            selected != ActivationBackend.FISPACT_II and fispact.complete
        ),
        "capabilities": {
            item.value: capabilities[item].as_dict() for item in order
        },
    }
