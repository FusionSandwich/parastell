import builtins
import importlib
import sys

import pytest

OPTIONAL_ROOTS = {"gmsh", "pydagmc", "pymoab"}


def test_magnet_source_cad_import_does_not_require_mesh_dependencies(
    monkeypatch,
):
    original_import = builtins.__import__

    def without_mesh_dependencies(name, *args, **kwargs):
        if name.split(".", 1)[0] in OPTIONAL_ROOTS:
            raise ImportError(f"blocked optional dependency {name}")
        return original_import(name, *args, **kwargs)

    for name in tuple(sys.modules):
        if (
            name == "parastell.magnet_coils"
            or name == "parastell.utils"
            or name == "parastell.cubit_utils"
        ):
            sys.modules.pop(name)
    monkeypatch.setattr(builtins, "__import__", without_mesh_dependencies)
    try:
        module = importlib.import_module("parastell.magnet_coils")
        assert module.gmsh is None
        assert module.core is None
        assert hasattr(module.MagnetSet, "export_selected_source_cad")
    finally:
        for name in (
            "parastell.magnet_coils",
            "parastell.utils",
            "parastell.cubit_utils",
        ):
            sys.modules.pop(name, None)
        parent = sys.modules.get("parastell")
        if parent is not None and hasattr(parent, "magnet_coils"):
            delattr(parent, "magnet_coils")


def test_missing_optional_mesh_dependency_fails_at_mesh_call(monkeypatch):
    magnet_coils = importlib.import_module("parastell.magnet_coils")

    monkeypatch.setattr(magnet_coils, "gmsh", None)
    instance = object.__new__(magnet_coils.MagnetSet)
    instance._logger = type("Logger", (), {"info": lambda *args: None})()
    with pytest.raises(ImportError, match="Gmsh"):
        instance.mesh_magnets_gmsh()
