# OpenMC downstream radiation workflow status

Status date: 2026-08-28

## Scope decision

This lane qualifies workflow plumbing with bounded or synthetic fixtures. It
does not claim useful statistics, source convergence, production readiness, or
a heterogeneous global magnet. The global ParaStell magnet remains one
homogenized material region. The accepted direct 90-degree candidate uses a
continuous 30 cm radial magnet envelope and does not fabricate 18 swept global
coil cells or split casing and winding-pack solids. Its exact nine-volume H5M
passed native watertightness and overlap checks. The contained source selection
now passes at `outer_cfs_cap=0.9655`. OpenMC navigation remains gated on a
geometry-debug shared-interface diagnosis before the independent second seed
and full-response smoke.

ParaStell owns the global geometry, source, OpenMC tallies, material-volume
fields, and closed-boundary phase-space banks. `DPA_workflow` owns depletion,
SPECTRA-PKA folding, Beyond-DPA event/cascade logic, and the Geant4, MCNP,
OpenMC-replay, and deterministic local-model adapters.

## Implemented producer path

- OpenMC 0.16 tally configuration already requests separate neutron and photon
  volume scalar-flux spectra, neutron/photon heating, total heating, damage
  energy, H1/H2/H3 and He3/He4 production, selected reaction families, and
  neutron/photon/electron/positron production. An unavailable nuclear-data
  response remains `UNAVAILABLE_IN_CONFIGURED_NUCLEAR_DATA`; it is never
  silently exported as zero.
- `openmc16_volume_results.py` now reads a cell/particle/energy flux tally from
  an OpenMC 0.16.0 statepoint, preserves the energy-bin integral, and divides
  track length only by an independently audited cell volume.
- `transport_response_plan.py` is now the solver-neutral response inventory.
  It covers neutron and photon scalar flux, surface current and correlated
  phase space, neutron/photon/local-mesh heating, damage energy, H/He
  production, reaction families, nuclide-specific MT rates, and secondary
  particle production. Its proof level cannot advance from `DECLARED` to
  `WIRED` unless a complete surface-bank configuration and unique tally names
  are present.
- `selected_case_instrumentation.py` wires that inventory to arbitrary declared
  `magnet_id` to OpenMC-cell mappings. It does not assume WISTELL-D entity IDs,
  alter the H5M, or authorize a transport run.
- `openmc16_response_results.py` generically reads every filter, nuclide,
  score, first moment, and standard deviation for OpenMC 0.16 tally results.
  It converts cell-filtered results into portable domain estimators and marks
  missing tallies and unavailable covariance explicitly rather than exporting
  zeros.
- `material_identity.py` reads public fusion-material-db JSON and OpenMC
  materials XML without a PyNE dependency. Mass-fraction versus atom-fraction
  basis, density, temperature, isotope fractions, citations, and source hashes
  remain explicit.
- `downstream_response_export.py` and
  `scripts/export_downstream_radiation_inputs.py` generate deterministic JSON
  inputs for SPECTRA-PKA, activation, isotope/MT response matrices, and local
  OpenMC/MCNP/Geant4/OpenSn/RADIANT boundary replay. They prevent source-rate
  double application and prompt/delayed source-ID collisions.
- `reaction_identity.py` makes every isotope-specific response carry a
  canonical OpenMC nuclide, numeric MT, readable reaction label, and
  nuclear-data hash. Ambiguous reaction names and repeated MT requests are
  rejected rather than inferred.
- `activation_campaign.py` and
  `configs/wistell_d_activation_schedule.json` freeze the requested 1-day,
  1-week, 1-year, 5-year, and 10-year full-power checkpoints and independent
  cooling branches without copying an absolute source rate.
- `alara_activation.py` pins OpenMC 0.16.0 VITAMIN-J-175 edges, reverses the
  ascending OpenMC bins into descending ALARA order, requires explicit isotope
  constituents, renders one independent deck per irradiation checkpoint, and
  applies the accepted physical source rate exactly once.
- `radiation_consumer_handoff.py` validates a solver-neutral bundle containing
  geometry/source/statepoint/data provenance, material identities, volume
  estimators, the canonical correlated boundary bank, and explicit consumer
  routes.
- The SPECTRA-PKA projection emits the existing DPA required fields:
  `layer_id`, `material_id`, `energy_bin_eV`, `bin_integrated_flux`, OpenMC and
  nuclear-data identities, tally ID, and geometry hash.
- The surface bank remains authoritative for local replay. Deterministic
  space-angle-energy projections cannot replace it or recondition its weights.
- Activation cannot consume surface current. OpenMC flux is not labeled a PKA
  spectrum or a defect yield.

## Activation campaign requested from DPA_workflow

The executable schedule belongs in `DPA_workflow` under schema
`dpa_workflow.activation_campaign_schedule/v1.0.0`. ParaStell carries only its
ID and SHA-256 and binds full-power multiplier 1.0 to the exact accepted
producer rate and modeled-domain scope. This avoids hard-coding either of the
currently conflicting historical 90-degree source rates.

Use a Julian year of 365.25 days (31,557,600 seconds). Requested cumulative
equivalent-full-power irradiation checkpoints are:

- 1 day: 86,400 s
- 1 week: 604,800 s
- 1 year: 31,557,600 s
- 5 years: 157,788,000 s
- 10 years: 315,576,000 s

Each end-of-irradiation inventory must fork independently into zero-source
cooling branches at EOI, 1 s, 1 min, 1 h, 1 d, 1 week, 30 d, 1 year, 5 years,
and 10 years. EOI is the limiting post-shutdown inventory, not a positive
solver step. Cooling branches must not be serialized into one another or fed
back into the continuing irradiation path. Prompt sources are excluded during
cooling; delayed-photon spectra/rates remain a separate decay-source ledger.

## Consumer mapping

| Consumer | ParaStell product | Consumer-owned operation |
|---|---|---|
| Activation | Volume neutron scalar-flux spectrum, material/domain/volume, source rate and data hashes | MicroXS/depletion, branched irradiation/cooling campaign |
| SPECTRA-PKA | Bin-integrated neutron scalar flux and material/isotope provenance | Recoil-matrix selection and PKA folding |
| Beyond-DPA | Hash-bound SPECTRA-PKA output plus reaction/gas/activation provenance | BCA/MD/cascade/event ledgers and defect responses |
| OpenMC local replay | Canonical correlated boundary bank | File/compiled source and local geometry |
| Geant4 | Canonical correlated boundary bank | MCPL/primary-generator adapter and transport |
| MCNP | Canonical correlated boundary bank | MCPL/SSW/SDEF adapter and transport |
| Deterministic/OpenSn | Canonical bank plus patch areas/normals | Quadrature/group projection and deterministic transport |

OpenMC 0.16 does not expose genuine source-level event genealogy. Consumers
must record parent history as unavailable; they must not fabricate it from row
order or clusters.

## Bounded ALARA runtime result

A fresh ParaStell-owned one-zone synthetic fixture passed on `poly-bateman`
with one core, an 8 GiB hard memory cap, and swap disabled. It used the
qualified repaired ALARA executable and hash-bound FENDL/A-2.0 plus FENDL/D-2.0
175-group files. The result has empty stderr, 16 well-formed isotope labels,
zero pre-irradiation activity/heat, positive shutdown and one-day
activity/heat, and a nonempty delayed-photon source. The sealed receipt
SHA-256 is
`88ec3f7342183e1334a29a08f56d281bb7db0b6974abbd1e6578c5be6e4d5e38`.

This validates the workflow only. The input flux was synthetic and did not use
the accepted WISTELL-D geometry or an OpenMC statepoint. The
transport/activation cross-library combination is a smoke basis, not a
qualified production comparison. No artifact from `wistell-d-openmc` was used
to qualify it.

## Remaining bounded integration work

1. Complete the OpenMC navigation gate. Source containment and native direct-90
   DAGMC geometry now pass. The first exact geometry-debug seed stopped when
   OpenMC independently classified both sides of the imprinted chamber/first-
   wall interface at the crossing point. A matched regular two-history track
   crossed that interface and wrote both statepoints with no lost-particle or
   navigation errors. The staged official-regression-H5M reproducer must
   classify this debugger behavior before the gate can advance.
2. Run a deliberately small coupled neutron/photon OpenMC smoke calculation on
   that accepted H5M and export all configured volume responses. Statistical
   classification will remain `WORKFLOW_SMOKE_ONLY`.
3. Convert one accepted physical smoke statepoint with the generic extractor
   and import its generated JSON in the real consumer packages. Unit fixtures
   pass, but a geometry-bound OpenMC statepoint does not yet exist.
4. In an isolated `DPA_workflow` branch, bind its executable schedule to the
   now-implemented ParaStell schedule contract and update consumers to accept
   ParaStell boundary schema v2.2 without
   discarding facet, barycentric, topology, local-frame, or provenance fields.
5. Exercise one bounded fixture through SPECTRA-PKA input generation and each
   local-source adapter. These are software-path checks, not physics results.

No production run, production activation, port geometry, or heterogeneous
global magnet is authorized by this status.

## Verification

- Focused downstream/activation/ALARA/isotope-MT suite: 43 passed.
- Parametric source-CAD geometry suite: 33 passed.
- Broad host-supported suite: 446 passed, 5 skipped, 5 failed only because the
  host has incompatible optional `pydagmc`/`cad_to_dagmc` APIs; six additional
  legacy test modules cannot collect because host Python lacks PyMOAB, Gmsh,
  or OpenMC. No dependency environment was installed or modified.
- Changed Python files pass Black, compileall, and `git diff --check`; compact
  JSON documents parse successfully.
- The added response-plan, material-adapter, selected-case wiring, generic
  statepoint-reader, surface-bank, and downstream-export focused suite passes
  37 tests.
- Independent read-only QA initially found statepoint/version/history,
  estimator-volume, bank-completeness, and schedule-receipt trust gaps. After
  correction it returned `PASS` with no remaining blocker/high findings.
