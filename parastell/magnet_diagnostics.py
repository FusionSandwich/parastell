"""Deterministic, hash-bound diagnostics for magnet-radiation products."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .material_manifest import deterministic_component_colors


SCHEMA = "parastell.magnet_radiation_figure_manifest/v1.0.0"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_figure_manifest(
    path: str | Path,
    *,
    input_paths: Sequence[str | Path],
    figures: Sequence[Mapping[str, Any]],
    component_colors: Mapping[str, str],
    rendering_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = []
    for value in input_paths:
        source = Path(value).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        inputs.append(
            {
                "path": str(source),
                "sha256": _sha256(source),
                "size_bytes": source.stat().st_size,
            }
        )
    outputs = []
    for value in figures:
        figure = Path(value["path"]).resolve()
        if not figure.is_file():
            raise FileNotFoundError(figure)
        outputs.append(
            {
                **dict(value),
                "path": str(figure),
                "sha256": _sha256(figure),
                "size_bytes": figure.stat().st_size,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "figures": outputs,
        "component_colors": dict(component_colors),
        "rendering_parameters": dict(rendering_parameters),
        "random_color_assignment": False,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_figure_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported figure-manifest schema")
    for key in ("inputs", "figures"):
        for item in manifest[key]:
            source = Path(item["path"])
            if not source.is_file() or _sha256(source) != item["sha256"]:
                raise ValueError(f"figure manifest hash mismatch: {source}")
    expected = deterministic_component_colors(
        manifest["component_colors"], manifest["component_colors"]
    )
    if expected != manifest["component_colors"]:
        raise ValueError("figure manifest contains invalid component colors")
    return manifest


def render_response_comparison(
    output_path: str | Path,
    *,
    response_ids: Sequence[str],
    unbiased: Sequence[float],
    weight_window: Sequence[float],
    component_names: Sequence[str] = (),
    explicit_colors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Render one optional WW comparison without importing plotting at base import."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render diagnostics"
        ) from exc
    labels = tuple(str(value) for value in response_ids)
    first = np.asarray(unbiased, dtype=float)
    second = np.asarray(weight_window, dtype=float)
    if first.shape != (len(labels),) or second.shape != first.shape:
        raise ValueError("response comparison arrays must align")
    colors = deterministic_component_colors(component_names, explicit_colors)
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(max(7.0, len(labels) * 0.45), 4.5))
    if labels:
        axis.bar(positions - 0.2, first, width=0.4, label="Unbiased")
        axis.bar(positions + 0.2, second, width=0.4, label="Weight window")
        axis.set_xticks(positions, labels, rotation=45, ha="right")
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "No comparable weight-window responses\nUnbiased fallback retained",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_ylabel("Response (configured units)")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return {
        "path": str(output.resolve()),
        "kind": "unbiased_weight_window_response_comparison",
        "component_colors": colors,
    }


def optional_radial_build_adapter(*args, **kwargs):
    """Invoke the user-owned plotting helper only when explicitly requested."""
    try:
        from radial_build_tools import RadialBuildPlot
    except ImportError as exc:
        raise RuntimeError(
            "radial_build_tools is optional and must be supplied by the user"
        ) from exc
    return RadialBuildPlot(*args, **kwargs)
