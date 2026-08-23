"""Configuration gate for the port-free reactor-to-magnet workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ProductionConfigurationAudit:
    config_path: str | None
    config_sha256: str | None
    geometry_features: Mapping[str, bool]
    port_free: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_port_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return (
        normalized in {"port", "ports"}
        or normalized.startswith("port_")
        or normalized.endswith("_ports")
    )


def _find_prohibited_port_keys(
    value: Any, path: tuple[str, ...] = ()
) -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_path = (*path, str(key))
            if (
                _is_port_key(key)
                and next_path != ("geometry_features", "ports")
            ):
                found.append(".".join(next_path))
            found.extend(_find_prohibited_port_keys(item, next_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(
                _find_prohibited_port_keys(item, (*path, str(index)))
            )
    return found


def validate_no_port_configuration(
    configuration: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> ProductionConfigurationAudit:
    """Reject any production configuration that can enable port geometry."""
    if not isinstance(configuration, Mapping):
        raise ValueError("production configuration must be a mapping")
    features = configuration.get("geometry_features")
    if not isinstance(features, Mapping) or features.get("ports") is not False:
        raise ValueError(
            "production magnet handoff requires geometry_features.ports: false"
        )
    prohibited = _find_prohibited_port_keys(configuration)
    if prohibited:
        raise ValueError(
            "port-related configuration is prohibited in this workflow: "
            + ", ".join(sorted(prohibited))
        )
    path = Path(config_path).resolve() if config_path is not None else None
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path else None
    return ProductionConfigurationAudit(
        config_path=str(path) if path is not None else None,
        config_sha256=digest,
        geometry_features={
            str(key): bool(value) for key, value in features.items()
        },
        port_free=True,
    )


def load_and_validate_no_port_configuration(
    path: str | Path,
) -> tuple[dict[str, Any], ProductionConfigurationAudit]:
    config_path = Path(path).resolve()
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = validate_no_port_configuration(
        configuration, config_path=config_path
    )
    return dict(configuration), audit
