"""Emit a deterministic, read-only Bateman geometry-stack receipt."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys


MODULE_NAMES = (
    "cadquery",
    "cad_to_dagmc",
    "gmsh",
    "pymoab",
    "h5py",
    "OCP",
    "numpy",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Import the fixed geometry stack and print its exact identity as JSON."""
    executable = Path(sys.executable).resolve()
    modules = {}
    for name in MODULE_NAMES:
        module = importlib.import_module(name)
        module_path_text = getattr(module, "__file__", None)
        module_path = (
            Path(module_path_text).resolve() if module_path_text else None
        )
        modules[name] = {
            "version": str(getattr(module, "__version__", "UNKNOWN")),
            "path": str(module_path) if module_path else None,
            "file_sha256": (
                sha256_file(module_path)
                if module_path is not None and module_path.is_file()
                else None
            ),
        }
    print(
        json.dumps(
            {
                "schema": "wistell_d.bateman_geometry_runtime/v1.0.0",
                "status": "PASS",
                "scientific_geometry_started": False,
                "python": {
                    "executable": str(executable),
                    "sha256": sha256_file(executable),
                    "version": sys.version,
                    "platform": platform.platform(),
                },
                "modules": modules,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
