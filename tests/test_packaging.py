from pathlib import Path
import tomllib

from setuptools.discovery import PackageFinder


def test_package_discovery_excludes_dirty_build_tree(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads(
        (repository / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["setuptools"]["packages"]["find"]
    (tmp_path / "parastell").mkdir()
    (tmp_path / "parastell" / "__init__.py").touch()
    (tmp_path / "build" / "lib" / "parastell").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "parastell" / "__init__.py").touch()

    packages = PackageFinder.find(
        where=str(tmp_path),
        include=configuration["include"],
        exclude=configuration["exclude"],
    )

    assert packages == ["parastell"]
    assert not any(package.startswith("build") for package in packages)
