# Magnet-radiation producer validation

This document records the clean-main integration evidence generated on
2026-08-25.  Large transport, geometry, and bundle artifacts remain outside the
repository under
`D:\parastell-artifacts\mainline-integration-20260825`; they are not packaged or
committed.  The final pushed branch SHA is recorded in the external run ledger
and delivery report because a commit cannot embed its own object ID.

## Repository basis and scope

- Fork and upstream `main` were both
  `de7d2978ff314b060ca2e6b10745a034e8b2a3c4` at branch creation.
- The integration branch is
  `magnet-radiation-field-mainline-20260824` and has no merge from the historical
  producer branch.
- The historical producer branch and the BlanketNeutronics repositories were
  read-only design references.  No source was copied and no runtime dependency
  was added.
- Ports, COMSOL generation, activation execution, SPECTRA-PKA execution,
  production deterministic transport, and explicit heterogeneous conductor
  physics are absent from the ParaStell branch.

## Geometry and identity

The clean-main smoke geometry completed `build`, `validate-geometry`,
`inventory-magnets`, and `build-tally-meshes` with PASS stage manifests.

| Evidence | Result |
|---|---:|
| Raw DAGMC H5M SHA-256 | `7c710fe3dd261ce7f46e5d08b4f9d2924513994ccb6229468933d0b91b2cd7cb` |
| Canonical geometry fingerprint | `4f3588b5bdbc80977ed69cdb32188e2bd6ea4e2a414a8338801a0824fbf2c709` |
| DAGMC volumes / surfaces | 44 / 269 |
| Winding packs / casings | 18 / 18 |
| Native positive-volume CAD overlaps | 0 |
| Leaky volumes / unmatched edges | 0 / 0 |
| Magnet associations | 36 |
| Maximum DAGMC/source closed-boundary volume difference | 0.11475% |
| Maximum bounding-box difference | 0.0471 cm |
| Maximum source-refinement volume change | 0.18569% |
| Invalid source BReps | 0 |
| Maximum closed-boundary divergence residual | approximately `2.98e-15` |

The accepted winding-pack volume is the closed, audited faceted transport
boundary used by DAGMC/OpenMC.  OCC mass properties differed by as much as
11.69% for the ruled-shell source solids and are retained as diagnostics, not
silently promoted to the transport volume.  Faceted volume agreed with the
independent ruled-shell area-times-centreline-length and CadQuery tessellation
checks.

The coarse smoke run produced 18 magnet-aligned meshes with 20,030 bins at a
100 cm nominal resolution.  Production candidates of 5 cm, 2 cm, 1 cm, and
0.5 cm are declared but not statistically qualified by this smoke run.

## Source and transport

The primary `[3, 9, 9]` source contains 512 tetrahedra, has SHA-256
`424f961aaacab01daf80a7dad57303360262abf081dc298201aeff3a0e655b1c`,
and integrates to `2.4145539947872977e20` source particles/s.  The complete
source ladder also built `[5, 21, 17]`, `[7, 41, 31]`, and `[11, 81, 61]`; the
finest candidate has 230,400 tetrahedra and SHA-256
`dc8c597afabed83b83b648dac6fa13aa1a53428db44715009f064c8b27fca2e2`.

The convergence stage is operational but its scientific decision is
`INCOMPLETE_EVIDENCE`: compatible transport-response and cost reports do not
exist for all four candidates.  No default is falsely qualified; the configured
conservative fallback is `[11, 81, 61]`.

Model preparation validated the pinned fusion-material-db artifact and the
FENDL nuclear-data audit.  The transport runtime was OpenMC 0.16.0 at commit
`617d35a5063c57796b43428bc401e627d2011046`.  The model contains 92 tallies and
uses coupled neutron-photon transport.

The official unbiased smoke run used five batches of 1,000 histories:

- statepoint SHA-256
  `b5aaf639e01a3f7315c47a6301927d80a67446486f45cfb15eabf602fc09e7ef`;
- surface-source SHA-256
  `752c1bd44bed98378ea68a6dc6c55fb4e6b3539f5d97326641f927d3d766bf7e`,
  36 complete records (34 neutron and 2 photon);
- zero lost particles, navigation failures, or bank-capacity saturation;
- maximum direction-norm residual `2.22e-16`;
- nonzero neutron-induced photon production (`0.0038/source`) and nonzero
  transported photon flux.

Three independent 5,000-history seeds (`190734863`, `514229`, and `832040`)
also completed without lost particles, navigation failures, runaway histories,
or surface-bank truncation.  Six global responses meet the deliberately loose
smoke thresholds and two are under-resolved.  Per-magnet, patch-level,
direction-resolved incoming-current, and selected-magnet effective-record
targets are unassessed.  Gate I is therefore
`UNDER_RESOLVED_OR_EMPTY`/incomplete, not a production-statistics pass.  Tally
ESS is unavailable because statepoints expose batch moments rather than event
weights; boundary ESS is calculated only from actual record weights.

## Weight-window decision

The initial targeted WW build exposed two stale CAD-identity assumptions; the
producer now rejects the legacy OCC-mass identity route and uses the same closed
boundary signature as geometry qualification.  The corrected conformal mesh
attempt then terminated in Gmsh HXT with intersecting PLC facets.  This is
classified `REJECTED_INSTABILITY`.

The workflow consequently records `weight_windows_enabled: false` and
`production_transport: UNBIASED`.  Preparation, qualification, and production
stages completed with the explicit `SKIPPED_UNBIASED_FALLBACK` state.  No WW
artifact is enabled merely because generation was attempted.

## Producer products and neutral bundle

Postprocessing completed with physical source normalization and wrote:

| Product | SHA-256 / evidence |
|---|---|
| Scalar-flux fields | `442e305b77a5103e8f716806af1306ac617972b5fa53f8d9df802a5233f2c3f3` |
| Heating | `735a5bfd1803105d4a8530a0b069d6133d7241ccdc645c15e5135ff411f1a9a4` |
| Reaction/particle production | `7332a86b7fef712988ce1b1296152ade37730deabe3631ec8c027965515a715b` |
| Damage energy and gas production | `40b6162d474a382872c8eddabd76a05fb135896c3dffa561e2c1263e1855c807` |
| Boundary handoffs | 18 complete files, 36 total records |

The scalar product includes exact CCFE-709 (710 boundaries), UKAEA-1102, a
configured fine neutron grid, a separate photon grid, and magnet-aligned local
fields.  Units are `particles/cm2/s` for physical scalar flux.  Damage energy
(`266.912 eV/source` in this smoke) is not DPA, and hydrogen/helium production
in atoms/source is not appm.  Thirteen empty boundary banks are explicitly
marked as insufficient observations, not physical zero.

Deterministic diagnostic colours are fixed by configuration.  The figure
manifest SHA-256 is
`e9ac4d82fa4205982a474395a7f28c172cceebbe026a5f8d0aff3774b82fdca5`
and the smoke PNG SHA-256 is
`ccf096f417703c0f4598e92617ddcab4a7291a37cd0bbfa9ac1e9ec4fff87146`;
the figure labels the WW-disabled unbiased fallback and makes no unavailable
WW comparison.

The solver-neutral bundle contains 23 files.  Neutral validation, without
ParaStell/OpenMC/DAGMC imports, passed all product hashes, schemas, material and
nuclear-data provenance, all-18 magnet coverage, CCFE-709 structure, and
boundary completeness.  The initial external evidence has bundle tree SHA-256
`a654e528d2252b7d80387a33a3f3f8a981daf0c2102163b8abc5f6a51b27b6d7`,
receipt SHA-256
`92a56db38480174071543a82bd27db5dc7ce5efc72711b495b65911b86cf734e`,
and manifest SHA-256
`e5442f2b4ace0a43f36fe287191d012354be0c3947f433d99008744276dc7f43`.
These hashes are regenerated after the final commit is created so the final
bundle binds the delivered ParaStell SHA.

## Downstream DPA_workflow compatibility

The focused check used the live DPA worktree at
`f9eef940ca6dffcd2cecdafa4fd286cb59c32d0b` without changing its pre-existing
dirty planning files.  It read the bundle with JSON/HDF5 only and used the
public `parastell_damage` spectrum and SPECTRA-PKA adapter APIs.

- Bundle schema and every product hash loaded successfully.
- Canonical geometry fingerprint and physical source normalization matched the
  bundle, scalar-field manifest, and source manifest.
- A real 709-group CCFE spectrum for
  `example-stellarator-sector-00-90deg-coil-0014` loaded as an
  `IncidentSpectrum`; its integrated physical scalar flux was
  `8.532874259896243e12 particles/cm2/s` for the 5,000-history smoke.
- A boundary-current file was rejected as scalar flux.
- All 18 boundary files had byte-identical SHA-256 values, and all 54
  position/direction/weight array comparisons were exact.
- The installed SPECTRA-PKA executable and existing DPA adapter completed a
  real Zr control case using the exported CCFE spectrum and the official local
  example matrices.

The requested real YBCO execution is
`BLOCKED_MISSING_YBCO_RECOIL_MATRICES`.  A focused local and container search
found none of the 13 required natural-isotope files (`Y089s`, seven Ba files,
two Cu files, and three O files).  The adapter generated the YBa2Cu3O7 job from
the real exported spectrum, then correctly refused execution because it was
not isotope-matrix complete.  No Zr data were relabelled as YBCO and no matrices
were downloaded.  The machine-readable report is external at
`D:\parastell-artifacts\mainline-integration-20260825\dpa-compatibility\dpa_compatibility_report.json`.

## Software and package gates

- Clean-main baseline: 43 passed, 7 skipped in the original baseline image.
- Integrated supported suite: 204 passed, 8 skipped in 189.39 seconds.
- `black --check .`: PASS (91 files unchanged after formatting four files).
- `python -m compileall parastell examples tests`: PASS.
- `git diff --check`: PASS; only Git line-ending notices were emitted.
- Base wheel import and CLI help: PASS without loading OpenMC, PyMOAB,
  CadQuery, DAGMC, or geometer modules.
- No-isolation sdist/wheel build with already-installed setuptools: PASS.
- Wheel and sdist inspection: zero statepoints, surface-source banks, H5M,
  MCPL, nuclear-data files, SPECTRA-PKA matrices, `validation_output`, or build
  trees; schemas and energy-group data are present.

Two legacy tests resolve `files_for_tests` relative to the current directory.
The literal root invocation `python -m pytest -q tests` therefore fails during
collection, while the repository's supported `tests` working-directory
invocation completes the 204-test suite above.  This pre-existing path behavior
was not changed as part of the additive producer.

## Scientific limitations

The evidence validates the global producer contract and unbiased smoke path,
not high-statistics production resolution.  Fine local-mesh qualification,
complete source-response convergence, full Gate-I per-magnet/patch/incoming
statistics, and a stable qualified WW mesh remain incomplete.  The coil frame
is an engineering centreline frame rather than exact tape twist.  Charged
secondary transport, activation, SPECTRA-PKA response folding, deterministic
transport, Geant4/MCNP replay, explicit conductor layers, and Beyond-DPA
constitutive physics remain downstream.
