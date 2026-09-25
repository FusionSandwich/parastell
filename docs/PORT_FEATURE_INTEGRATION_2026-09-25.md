# Port feature integration receipt

Date: 2026-09-25. Branch: `JS/port-feature-integration-20260925`.
Base: verified `origin/main` at `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`.
Source: `ports/actual-plc-repair-20260824` at
`b9d672764a804ea97fa311a45ce820a610dcb5e8`.

The port source is the linear 19-commit sequence from `3a0ce81` through
`b9d6727` after `e223061`. Both `origin/main` and `e223061` are shallow
boundaries in this clone; `git merge-base` finds no provable common ancestor.
The source base differs from main only in `README.md` and
`parastell/__init__.py` among the port series' changed paths. The other 46
source paths were transplanted directly. The two overlapping files received
only their port-specific additions. The source base's COMSOL and magnet
spectral handoff files were excluded. There were no patch conflicts.

The integration includes port schema and CAD booleans, surface aperture and
native DAGMC/PLC geometry, MOAB artifacts, magnet clearance/component ledger,
visualization, examples, documentation, and focused tests. The port guide's
stale PLC failure statement was updated to point to the later repair record.
`parastell/native_port_geometry.py` received formatting only for the local
Black check.

## Local verification

The existing WSL `RadiantHTS-ParaStell-20260830` runtime at
`/opt/openmc-v0.16.0-venv/bin/python` supplied all test dependencies. No
dependency was installed, built, or downloaded. Tests ran with
`PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`.

* Repository root: `python -m pytest -q -p no:cacheprovider tests/test_ports.py tests/test_native_port_geometry.py tests/test_plc_diagnostic.py` — **105 passed**.
* Repository root: `python -m pytest -q -p no:cacheprovider tests/test_paraview_export.py tests/test_port_visualization.py tests/test_magnet_coils.py::test_filament_magnet_iterator_preserves_coil_region_metadata tests/test_magnet_coils.py::test_imported_step_magnet_iterator_exposes_in_memory_solids tests/test_parastell.py::test_shape_distance_rejects_out_of_bounds_occ_solution` — **13 passed** across the visualization and shape-distance tests; the three magnet cases failed fixture setup because their files are relative to `tests/`.
* `tests/` working directory with `PYTHONPATH=..`: `python -m pytest -q -p no:cacheprovider test_magnet_coils.py::test_filament_magnet_iterator_preserves_coil_region_metadata test_magnet_coils.py::test_imported_step_magnet_iterator_exposes_in_memory_solids test_invessel_build.py::test_ivb_basics test_invessel_build.py::test_ivb_cadquery_construction test_invessel_build.py::test_ivb_pydagmc_construction` — **9 passed**.
* `python -m black --check` on all changed Python files and `git diff --check` — passed.

An earlier combined test command also stopped during collection because
`test_invessel_build.py` expects a `tests/` working directory. The corrected
run above passed. None of these path failures required code changes.

## Scope and remaining gates

Native conformal export accepts exactly one surface-anchored port. Disconnected
multi-solid intersections, multiple/far-side centerline intervals, and the
legacy structured MOAB mesh for ported components remain unsupported and fail
explicitly. Cartesian CAD port behavior and the unported mesh path are retained.

The six-case repaired radial PLC matrix in
`ACTUAL_PLC_REPAIR_2026-08-24.md` is source-branch evidence, not a rerun on this
integration branch. Before full-reactor or transport acceptance, replay that
matrix here, then validate actual magnet/port overlap and clearance, combined
DAGMC senses and global graveyard, OpenMC geometry with the real cross-section
library, and fixed-source transport. No such expensive run or remote job was
launched for this integration.
