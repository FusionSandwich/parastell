"""Write a create-only CadQuery-to-Gmsh import mapping diagnostic."""

from __future__ import annotations

import argparse

from parastell.gmsh_import_diagnostic import diagnose_source_gmsh_import


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    diagnose_source_gmsh_import(
        source_dir=arguments.source_dir,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
