# Read-only workflow reference ledger

## Isolation and integrity

Two local clones were inspected without checkout, branch creation, edits, or
execution:

```text
D:\Beyound DPA Repos\BlanketNeutronics
D:\Beyound DPA Repos\blanket
```

For all five required branch tips, both clones and the live GitHub refs have the
same commit. Their tree IDs also match:

| Branch | Commit | Tree |
|---|---|---|
| `t1e_main` | `bc4ab3d0f27369d4eda908a3fc187a10b6c7fedb` | `1ad603087c2d491ee8165e846fbbbaa2f06bff30` |
| `kisslinger-input` | `e2e0c6e6571ad824db4fc537e12069d747b99c60` | `c3a4ca0d1cdbf337ced74fee96b1b98a42aa3fff` |
| `pydagmc_ivb_ww_meshing` | `c18dcc4bbbd7b3c343183c5072f93dd0111428ef` | `4b1ef0fa17a5505e4684adbfb5a0214855e505c1` |
| `fix_get_ww_mesh` | `a415de18be6e0ece0a3edc5b614bec6a1f578972` | `1a99ca934b0360b3ba991b44b3e6fbfa3b371a68` |
| `edgars_special_branch` | `69598e7b0575731d7d658897530a72eee3d69e49` | `2542dd89101a5a947cb30984c9e54be58718150e` |

The reference repositories do not supply a clear root license in the inspected
trees. No source text is copied. This ledger records conceptual provenance for
independent implementation against public ParaStell, MOAB, Gmsh, and OpenMC
APIs.

## Workflow trace

| Feature | Reference | Inputs and outputs | Public API/pattern | Known assumption or defect | Decision |
|---|---|---|---|---|---|
| Early NWL build and transport | `t1e_main@bc4ab3d`, `neutronics_wistelld_DCLL_1/2_build_NWL_geom.py`, module-level call; `3_run_NWL.py`, module-level call | VMEC path, source mesh, strengths, 90-degree sector; writes first-wall H5M and a surface source | Separate geometry and transport scripts | Fixed filenames, fixed one-million-particle invocation, no hash or restart manifest | Retain stage separation, not script text or fixed values |
| Early auxiliary meshing | `t1e_main@bc4ab3d`, `neutronics_wistelld_DCLL_1/makeUmesh.py:makeUmesh` | STEP/Cubit input to Exodus mesh | Cubit single-layer tetrahedral meshing | Cubit-only and paper-era procedural state | Do not inherit; prefer Gmsh/MOAB |
| Manual restartable sequence | `kisslinger-input@e2e0c6e`, `utilities/run_parastellarizer_example.py`, module-level sequence | YAML configuration; source/NWL/radial build/thickness/geometry/WW/OpenMC products | Explicit sequence of independently callable methods | Restartability relies on manually commenting stages; no state/hash validation | Reimplement as manifest-driven stages |
| Configuration-driven orchestration | `kisslinger-input@e2e0c6e`, `utilities/parastellarizer.py:Parastellarizer.__init__` | Plasma, filaments/custom STEP, grids, build layers, meshing, WW, tallies | One configuration object carries all stage parameters | Global mutable object, implicit working directory, mixed geometry/transport/plot concerns | Split into small workflow and neutral config models |
| ParaStell D–T source mesh construction | `kisslinger-input@e2e0c6e`, `Parastellarizer.build_source_mesh` | `source_mesh_size`, VMEC, plasma condition functions; writes H5M and strengths NPY | `Stellarator.construct_source_mesh` and `export_source_mesh` | Later OpenMC initializer discards the energy distribution and uses fixed 14.1 MeV | Retain ParaStell spatial/temperature-dependent source and provenance, reject fixed-energy replacement |
| NWL geometry and source generation | `kisslinger-input@e2e0c6e`, `build_nwl_geometry`, `nwl_transport`, `extract_nwl` | VMEC/Kisslinger surface, wall `s`, source mesh, counts; writes H5M, surface source, NWL arrays | ParaStell CAD/PyDAGMC and NWL helpers | Alternate Cubit/PyDAGMC identities and fixed filenames are implicit | Retain optional diagnostic stages with explicit identities and hashes |
| Radial distance and thickness matrices | `kisslinger-input@e2e0c6e`, `measure_radial_build_distance`, `smooth_radial_build_distance`, `get_thickness_matrices`, `interpolate_thickness_matrices` | Filaments/custom STEP, toroidal response table, limits; writes NPY matrices | ParaStell surfaces plus response-table interpolation | Cubit-dependent measurement and private response assumptions; optimization is outside producer scope | Retain only optional geometric diagnostics, not paper response selection |
| Filament/custom-STEP magnets | `kisslinger-input@e2e0c6e`, `add_magnet_geometry` | Filament file or `custom_magnet_geometry`, dimensions, casing thickness, tags | `construct_magnets_from_filaments` or `add_magnets_from_geometry` | Tally cell IDs are inferred from volume ordering and offsets | Retain both geometry paths; replace incidental ordering with stable tags/discovery |
| IVB and DAGMC construction | `kisslinger-input@e2e0c6e`, `build_ivb_geometry`, `build_dagmc_model` | Build matrices, CAD/PyDAGMC selector, meshing args; writes STEP/H5M/CUB5 | Existing ParaStell public geometry APIs | Hidden CWD paths and backend-specific IDs | Retain current ParaStell APIs and add canonical identity manifests |
| Component tally meshes | `kisslinger-input@e2e0c6e`, `build_unstructured_meshes`; `utilities/parastellarizer_utils.py:get_parastellarizer_tallies` | Per-layer Gmsh/MOAB/Cubit mesh settings and score names; writes H5M meshes/tallies | `UnstructuredMesh`, `MeshFilter`, `CellFilter` | Magnet loop indexes a string as a mapping in one revision; tally families and units are underspecified | Independently implement typed tally profiles and validate mesh mappings |
| Combined IVB/magnet WW mesh | `kisslinger-input@e2e0c6e`, `make_ww_mesh` | IVB component set, mesher/size args, optional magnets; writes combined MOAB H5M | Export component meshes, load both into one MOAB core, write combined mesh | Magnet temporary file is loaded/unlinked even when magnets are disabled; no positive-volume or coverage audit | Reimplement with explicit optional components, validation, and cleanup confined to a stage directory |
| Native/PyDAGMC WW mesh | `pydagmc_ivb_ww_meshing@c18dcc4`, `Parastellarizer.make_pydagmc_ww_mesh`; `edgars_special_branch@69598e7`, `parastellarizer_utils.py:PydagmcIvbWeightWindowMesh` | Ordered IVB surfaces and optional Gmsh magnet mesh; writes one MOAB H5M | Connect adjacent structured surfaces with tetrahedra; combine with magnet mesh | One-cell-thick assumption, temporary serialization, optional-magnet bug, no component/volume contract | Retain only the conceptual native-surface path; validate positive volumes and component coverage |
| Cubit WW mesh fallback | `fix_get_ww_mesh@a415de1`, `Parastellarizer.build_unstructured_ww_mesh`; `parastellarizer_utils.py:make_unstructured_mesh` | CUB5, ratio, angle; writes H5M | Cubit Exodus mesh followed by MOAB conversion | Cubit-only and legacy faceting; not acceptable as mandatory backend | Keep optional at most; Gmsh/MOAB is default |
| Basic OpenMC DAGMC model | `kisslinger-input@e2e0c6e`, `get_basic_openmc_model`, `initialize_source_mesh` | DAGMC H5M, materials XML, source mesh/strengths | `DAGMCUniverse`, periodic planes, `MeshSpatial`, fixed-source settings | Hard-coded surface/cell IDs, bounding sphere, implicit paths, fixed 14.1 MeV energy | Reimplement using collision-safe IDs and audited ParaStell D–T source |
| MAGIC WW generation | `kisslinger-input@e2e0c6e`, `get_openmc_ww_model`; also `fix_get_ww_mesh@a415de1`, `build_openmc_ww_model` | WW H5M, particles/batches, split cap, energy edges; writes OpenMC model/checkpoints/WW HDF5 | `UnstructuredMesh`, `WeightWindowGenerator(method="magic")`, on-the-fly updates and collision/surface checkpoints | Fixed four-edge grid, `100000` split cap, neutron-only assumption, no independent qualification or artifact binding | Independently implement OpenMC-0.16-adapted generator plus bounded grid/split study and contract |
| WW reuse in transport | `kisslinger-input@e2e0c6e`, `get_openmc_transport_model` | WW HDF5, transport settings and tallies | `hdf5_to_wws`, weight-window settings | No geometry/source/material/data compatibility contract; one branch uses obsolete `max_splits` spelling | Require a complete hash-bound artifact contract before reuse |
| Magnet cell/mesh tallies | `kisslinger-input@e2e0c6e`, `get_parastellarizer_tallies`; YAML `magnet_tallies` | Cell IDs, optional mesh, fast flux/heating/DPA/He/TBR | OpenMC cell and mesh filters | Hard-coded cell IDs and response aliases; surface current and boundary correlation absent | Discover all magnets and emit explicit estimator/unit/availability inventory |
| Contour and radial-build plots | `kisslinger-input@e2e0c6e`, `plot_prediction_matrices`, `plot_thickness_matrices`, `plot_radial_distances`, `plot_nwl`; `parastellarizer_utils.py:plotContour` | Matrices and angular axes; writes plots | Matplotlib contour workflow | Implicit filenames and missing artifact/hash manifest | Reimplement deterministic plots with axes, units, normalization, code/input/PNG hashes |
| Representative ARIES workflow | `edgars_special_branch@69598e7`, `neutronics_aries_shield_li02_parastell/run_parastellarizer.py` and `stellarator_inputs.yml` | Custom STEP magnets, ARIES build, low-stat WW/transport demo | Same staged calls on a concrete model | Demonstration statistics, hand-edited offsets, typo-like 1.5 GeV WW maximum, fixed cell IDs | Use only as a geometry/sequence reference, never as production defaults |

## Patterns retained

The clean implementation keeps this conceptual graph:

```text
validate inputs
→ build source
→ build and qualify the four-shape source-convergence ladder
→ build geometry
→ validate and inventory geometry
→ build tally/weight-window meshes
→ prepare unbiased model
→ run an independent multi-seed unbiased Gate-I campaign
→ qualify supported statistics and mark unsupported metrics unassessed
→ generate candidate weight windows
→ independently qualify candidates
→ prepare production model or disable WW
→ post-process neutral products
→ render hash-bound diagnostics
```

Each stage receives explicit paths, hashes inputs and outputs, records status,
and refuses stale or incompatible products. The global workflow does not own
downstream deterministic, HTS-layer, SPECTRA-PKA, activation, or response
optimization physics.

## Patterns explicitly rejected

- Constant 14.1 MeV replacement for ParaStell's spatial/temperature-dependent
  D–T source.
- Hard-coded magnet cell IDs or volume ordering.
- Fixed four-edge WW grid or unqualified `max_history_splits=100000`.
- `max_splits` legacy spelling.
- Cubit-only auxiliary meshing.
- Unconditional temporary magnet-mesh loads.
- Implicit current-working-directory paths.
- Mutable monolithic workflow objects without stage manifests.
- Treating a WW file as reusable without geometry/source/material/data hashes.
- Treating surface current as volume scalar flux.
- Treating signed closed-surface current or all crossing weights as inward-only
  current.
- Treating a scalar hotspot magnitude as 3-D hotspot-location convergence.
- Treating damage-energy as DPA or produced H/He atoms as appm.
- Treating paper-era response tables or plots as producer physics.

The inspected reference workflow contains neither a hash-bound four-shape
source-response convergence campaign nor an independent multi-seed Gate-I
qualification contract. Those clean-mainline additions were independently
implemented from the prompt's acceptance criteria and public OpenMC/ParaStell
interfaces; they are not attributed to the read-only reference repositories.

## Optional auxiliary repositories

`radial_build_tools` is clean at
`d195d5a9f777c3ac42a9033646937558e07748e0` and MIT licensed. Its
`RadialBuildPlot` accepts explicit colors, but `assign_colors` and
`generate_unique_color` otherwise use random XKCD colors. ParaStell will pass a
complete deterministic color map or use an independent adapter. Its
`ToroidalModel` remains a scoping visualization, not the 3-D reactor model.

`fusion-material-db` is clean at
`9d8f84ad7b3ac2c587edfbe8ec1ed74891484498` and MIT licensed. The checked-in
generated products include pure/mixed JSON, OpenMC atom/mass-fraction XML, and
MCNP atom/mass-fraction text. The producer will consume a pinned generated
artifact, not PyNE or a live repository import. The existing `HTSCsimple`
record is useful provenance but is not silently treated as manufacturer-
authoritative winding-pack composition.

## Source-copy audit

No source file, source fragment, or vendored module from either blanket clone
has been copied. Reuse is limited to the conceptual patterns and public-API
calls identified above. The two read-only repositories were clean before and
after this study.
