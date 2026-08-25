# Magnet-radiation producer

ParaStell can prepare, execute, and postprocess an OpenMC 0.16 coupled
neutron-photon calculation for geometrically resolved magnet casings and
compositionally homogenized winding packs. The feature is additive: ordinary
ParaStell CAD, DAGMC, and source workflows do not import OpenMC or PyDAGMC at
base-package import time.

The producer has two deliberately different radiation products:

- The closed-boundary product stores continuous correlated crossing records.
  Position, direction, energy, weight, particle, prompt flight time, DAGMC
  surface/facet identity, crossing sense, source-file record identity, surface
  coordinates, and a parallel-transport coil-centreline frame remain joined in
  each record. These records are authoritative for directional replay in a
  downstream heterogeneous model.
- The scalar-field product stores OpenMC track-length scalar flux divided by an
  explicit region volume. Whole-pack fields use the audited, closed faceted
  DAGMC transport-region volume. The corresponding CAD boundary is checked for
  validity, closure, orientation, convergence, and identity; the OCC mass
  property is retained only as a diagnostic because ruled-shell mass and the
  faceted transport boundary need not agree. Local fields use full rotated
  mesh-voxel volumes and may include
  winding pack, casing, nearby reactor structure, void, or graveyard within a
  voxel. They are therefore labelled `spatial_mixed`, not as pure winding-pack
  material.

A surface current is not scalar flux. The bundle writer and CCFE-709 validator
reject a surface-current payload presented as SPECTRA-PKA scalar flux.

## Stage workflow

The smoke configuration is `examples/magnet_radiation_smoke.json`; the
production override is `examples/magnet_radiation_production.json`. Artifacts
must be directed outside the Git checkout with `PARASTELL_ARTIFACT_ROOT`.
Representative commands are:

```text
parastell magnet-field validate-config CONFIG
parastell magnet-field validate-inputs CONFIG
parastell magnet-field build-source CONFIG
parastell magnet-field build-source-convergence-ladder CONFIG
parastell magnet-field qualify-source-convergence CONFIG
parastell magnet-field build CONFIG
parastell magnet-field validate-geometry CONFIG
parastell magnet-field inventory-magnets CONFIG
parastell magnet-field build-tally-meshes CONFIG
parastell magnet-field build-ww-mesh CONFIG
parastell magnet-field prepare-unbiased CONFIG
parastell magnet-field run-unbiased CONFIG
parastell magnet-field run-unbiased-campaign CONFIG
parastell magnet-field qualify-statistics CONFIG
parastell magnet-field prepare-ww-generation CONFIG
parastell magnet-field run-ww-generation CONFIG
parastell magnet-field run-ww-qualification CONFIG
parastell magnet-field qualify-ww-stage CONFIG
parastell magnet-field prepare-production CONFIG
parastell magnet-field postprocess CONFIG
parastell magnet-field render-diagnostics CONFIG
parastell magnet-field export-bundle CONFIG
parastell magnet-field validate --bundle BUNDLE_DIRECTORY
```

Every stage hashes its configuration, explicit inputs, upstream stage
manifests, and outputs. A matching PASS stage can be reused. Changed inputs are
reported as stale, and `--force` performs a genuine rerun. Transport is never
started by a preparation command.

`run-unbiased` checks the OpenMC version/commit, coupled photon mode, required
tallies, statepoint realizations, surface-bank capacity, direction norms,
lost-particle files, and nonzero neutron-induced photon-production evidence.
It writes a hash-bound transport report. `postprocess` exports:

- configured, CCFE-709, and UKAEA-1102 whole-volume neutron scalar flux;
- configured whole-volume photon scalar flux;
- role-labelled whole-volume casing and winding-pack scalar flux;
- magnet-aligned local neutron and photon scalar-flux fields;
- neutron and photon heating in per-source and physical units;
- damage/reaction and neutron/photon/electron/positron production products;
- distinct closed outer-magnet and winding-pack phase-space handoffs for every
  selected magnet;
- `parastell.activation_ready_metadata/v1.0.0`, with explicit
  DAGMC-volume-to-OpenMC-cell and material IDs, volumes, masses, densities,
  temperatures, composition hashes, exact mesh-bin volumes, physical source
  normalization, geometry fingerprint, and nuclear-data manifest.

`build_geometry` also writes
`parastell.magnet_geometry_interchange/v1.0.0`. It contains the hash-bound
filament source identity, native-global and sector transforms, full centreline
and arc-length samples, casing and winding-pack cross sections, continuous
right-handed engineering parallel-transport frames, component/surface IDs,
and available/unavailable STEP, STL, and H5M artifact evidence. The contract
always records `frame_kind = engineering_parallel_transport` and
`tape_twist_resolved = false`; it cannot be used to claim resolved conductor
twist.

The neutral bundle contains only versioned HDF5/JSON products and hashes. It
can be read and validated without importing ParaStell's geometry or OpenMC
runtime stacks.

Activation schedules, inventory evolution, cooling, decay-photon sources, and
shutdown transport remain downstream responsibilities. Mixed local meshes are
explicitly not post-transport depletable until material-intersection fractions
are supplied, and mesh R2S remains disabled until non-overlap is qualified.

## Normalization and statistics

OpenMC transport tallies are per source history. Physical scalar flux is

```text
track length per source / region volume * physical D-T source rate
```

and has units particles/cm2/s. Heating is exported as eV/source, W, and W/cm3.
Reaction and particle-production events are exported per source and per second.
Energy values are group-integrated, with exact ascending boundaries stored in
each field.

The statepoint exposes batch moments, not event weights. Scalar fields
therefore preserve OpenMC mean and one-standard-deviation uncertainty and
explicitly mark event-level ESS unavailable; they do not manufacture an ESS
from relative error. Resolution codes distinguish empty, under-resolved, and
qualified-batch-precision bins. A zero-scored bin is never automatically a
physical zero.

Boundary banks retain event weights, so their population reports include raw
record count, weighted count, sum of squared weights, and exact weighted ESS.
When a complete unsampled bank has zero selected records, its manifest reports
`EMPTY`, refuses a physical-zero claim, and gives a 95% Poisson upper bound on
expected crossings per source history. If a configured bank capacity is
reached, the bank is classified truncated and rejected.

Damage-energy is exported in `eV/source`, `eV/s`, and volume-normalized
`eV/cm3/s` where a region volume is available. It is not DPA. Hydrogen and
helium channels are produced atoms per source and atoms per second; they are
not appm. Missing nuclear-data support is recorded as unavailable rather than
silently converted to zero.

## Source and statistical qualification

The source ladder evaluates `[3,9,9]`, `[5,21,17]`, `[7,41,31]`, and
`[11,81,61]`. Source rate, six spatial moments, and two D-T energy moments are
bound to the candidate source manifests. A convergence decision additionally
requires real transport reports for whole-magnet neutron flux, photon flux,
direction-resolved inward current, heating, and a 3-D hotspot identity/location,
plus measured computational cost. Cost selects between physically converged
candidates; it is not itself a convergence metric. Missing or semantically
ambiguous response evidence yields `INCOMPLETE_EVIDENCE`.

The independent-seed production campaign records global neutron/photon flux,
heating, signed current, and bidirectional crossing-weight responses with
explicit thresholds. Signed current is not relabelled as inward current, and
all-crossing bank weights are not relabelled as incoming records. Per-magnet,
patch-level, direction-resolved incoming-current, and selected-magnet effective
record targets remain `UNASSESSED` until the corresponding evidence exists.
The smoke result is therefore `UNDER_RESOLVED_OR_EMPTY`, never a fabricated
Gate-I pass. Event-level ESS is reported only for weighted boundary records;
tally ESS remains unavailable when OpenMC exposes only batch moments.

## Materials and nuclear data

The material configuration pins the fusion-material-db repository and exact
generated-artifact hashes. Natural elements are expanded explicitly. The
homogenized winding-pack mixture is an engineering project assumption with
declared constituent volume fractions, density rule, temperature, and
uncertainty variants; it is not represented as manufacturer-authoritative.

The nuclear-data audit pins the `cross_sections.xml` hash, evaluation release,
photon data, nuclide availability, and requested versus available
temperatures. Any nearest-temperature fallback must be explicitly configured
with a tolerance and justification. Thermophysical material temperatures are
not silently changed by that interaction-data policy.

## Weight-window decision path

Weight windows are optional and fail closed:

```text
unbiased reference seeds
→ ParaStell-derived conformal WW mesh
→ bounded 4/8/16-group MAGIC candidates
→ independent qualification seeds
→ bias, stability, ESS, and FOM comparison
→ qualified reuse or explicit unbiased fallback
```

WW control energy groups are separate from reporting/tally groups. Contracts
bind the canonical geometry fingerprint, raw H5M, source definition and mesh,
physical source rate, materials, nuclear data, WW mesh, OpenMC build, seed,
histories, particle, energy grid, split limit, and selected magnets. Reuse is
rejected if any binding differs. The default policy requires three seeds,
Benjamini-Hochberg bias control, geometric-mean primary-response FOM ratio of
at least 2, improvement in at least 75% of primaries, no critical FOM below
0.8, and no lost particles, navigation failures, or pathological splitting.
Unqualified or non-beneficial windows remain disabled.

The WW workflow has explicit terminal states. A successful executable campaign
may enable windows only after independent-seed bias/FOM qualification. Mesh or
generator instability, insufficient pilot statistics, statistically detected
bias, or no material benefit produces a documented unbiased fallback; the
presence of a WW file alone never enables it.

## Scope and downstream use

The whole-pack frame is a continuous engineering centreline frame, not the
exact twist of every REBCO tape. Explicit Cu/Ag/REBCO/buffer/substrate/solder/
insulation layers belong in a downstream local model. SPECTRA-PKA, activation,
deterministic solvers, Geant4, MCNP, charged-secondary transport, and
Beyond-DPA constitutive physics are consumers, not producer responsibilities.
Electron and positron *production tallies* describe neutral-transport reaction
products; ParaStell does not transport those charged particles in this feature.
The exact CCFE-709 volume scalar neutron flux is the SPECTRA-PKA interoperability
product; activation consumers use the same scalar-flux/material/volume/source
provenance contract.

Ports, port apertures/liners, COMSOL generation, activation orchestration, and
production deterministic solvers are deliberately absent from this branch.
BlanketNeutronics and its mirror were studied read-only; no source was copied
and neither repository is a runtime dependency. `radial_build_tools` remains
optional, with deterministic colours required for reproducible figures.

## Known limitations and gate status

- The global producer geometry, all-18-magnet identity, watertightness, native
  CAD overlap, coupled tally inventory, and neutral schemas are independently
  validated by the smoke workflow.
- The four source meshes can be built reproducibly, but source-response
  convergence remains incomplete until all four receive compatible transport
  response and cost reports.
- The smoke local mesh is deliberately coarse. The 5 cm, 2 cm, 1 cm, and 5 mm
  candidates are declared, not claimed statistically qualified.
- Gate I remains incomplete where per-magnet, patch, direction-resolved
  incoming-current, and effective-record evidence is unavailable.
- The whole-pack centreline frame is an engineering frame, not exact tape
  twist. Charged-secondary transport, activation, response condensation,
  deterministic transport, Geant4, MCNP, ports, SPECTRA-PKA execution,
  optimization, atomistic/ML workflows, and explicit heterogeneous conductor
  physics remain downstream or out of scope.
