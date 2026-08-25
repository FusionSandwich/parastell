import ast
import json
from pathlib import Path
import subprocess
import tomllib

import pytest

from parastell.magnet_field_workflow import MagnetRadiationWorkflow


FORBIDDEN_PACKAGE_PATHS = {
    "parastell/activation",
    "parastell/comsol_fusion_magnets.py",
    "parastell/evaluated_multigroup.py",
    "parastell/group_condensation.py",
    "parastell/hts_multilayer.py",
    "parastell/magnet_spectral_handoff.py",
    "parastell/multigroup_sn.py",
    "parastell/schemas/magnet_spectral_handoff.schema.json",
}
FORBIDDEN_MODULES = {
    "parastell.activation",
    "parastell.comsol_fusion_magnets",
    "parastell.evaluated_multigroup",
    "parastell.group_condensation",
    "parastell.hts_multilayer",
    "parastell.magnet_spectral_handoff",
    "parastell.multigroup_sn",
}
FORBIDDEN_DEPENDENCIES = {
    "alara",
    "fispact",
    "htscsimple",
    "spectra-pka",
}


def _root():
    return Path(__file__).resolve().parents[1]


def _module_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and path.parent.name == "parastell":
                module = f"parastell.{module}".rstrip(".")
            result.add(module)
    return result


def test_clean_producer_has_no_historical_port_or_downstream_solver_paths():
    root = _root()
    existing = {
        path.relative_to(root).as_posix()
        for path in (root / "parastell").rglob("*")
        if path.is_file()
    }
    assert FORBIDDEN_PACKAGE_PATHS.isdisjoint(existing)
    suspicious = {
        path
        for path in existing
        if any(
            part == "ports"
            or part.startswith("native_port")
            or part.startswith("port_visual")
            for part in Path(path).parts
        )
    }
    assert suspicious == set()


def test_python_import_graph_has_no_excluded_consumer_modules():
    imports = set()
    for path in (_root() / "parastell").rglob("*.py"):
        imports.update(_module_imports(path))
    bad = {
        name
        for name in imports
        if any(
            name == root or name.startswith(f"{root}.")
            for root in FORBIDDEN_MODULES
        )
    }
    assert bad == set()


def test_package_dependencies_and_cli_remain_producer_only():
    root = _root()
    project = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    dependency_rows = list(project.get("dependencies", []))
    for rows in project.get("optional-dependencies", {}).values():
        dependency_rows.extend(rows)
    names = {
        row.split("[", 1)[0]
        .split("=", 1)[0]
        .split(">", 1)[0]
        .split("<", 1)[0]
        .split("~", 1)[0]
        .strip()
        .lower()
        for row in dependency_rows
    }
    assert FORBIDDEN_DEPENDENCIES.isdisjoint(names)
    assert project["scripts"] == {"parastell": "parastell.cli:main"}
    cli = (root / "parastell" / "magnet_field_cli.py").read_text(
        encoding="utf-8"
    )
    for command in ("activation", "depletion", "spectra-pka", "hts-replay"):
        assert command not in cli.lower()


def test_nested_port_configuration_is_rejected(tmp_path):
    config = tmp_path / "nested-port.json"
    config.write_text(
        json.dumps(
            {
                "geometry_features": {"ports": False},
                "output_directory": str(tmp_path / "artifacts"),
                "stages": {"build_geometry": {"port_ducts": []}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="port-related configuration"):
        MagnetRadiationWorkflow.from_json(config)


def test_clean_mainline_is_an_ancestor_and_known_port_tips_are_not():
    root = _root()
    if not (root / ".git").exists():
        pytest.skip("Git ancestry is unavailable in an installed package")

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

    clean_mainline = "de7d2978ff314b060ca2e6b10745a034e8b2a3c4"
    assert (
        git("merge-base", "--is-ancestor", clean_mainline, "HEAD").returncode
        == 0
    )
    known_port_refs = (
        "refs/heads/agent/native-port-full-assembly-transport-20260822",
        "refs/heads/agent/port-visual-h5m-moab-20260820",
        "refs/remotes/origin/ports/full-reactor-validation",
    )
    for ref in known_port_refs:
        if git("show-ref", "--verify", "--quiet", ref).returncode == 0:
            assert (
                git("merge-base", "--is-ancestor", ref, "HEAD").returncode != 0
            )
