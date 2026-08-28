# OpenMC downstream radiation workflow status

Status date: 2026-08-28

## Scope decision

This lane qualifies workflow plumbing with bounded or synthetic fixtures. It
does not claim useful statistics, source convergence, production readiness, or
a heterogeneous global magnet. The global ParaStell magnet remains a
homogenized material region.

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

## Remaining bounded integration work

1. Complete the accepted source-CAD/DAGMC 90-degree geometry gate. No physical
   WISTELL-D statepoint may be promoted while that gate is blocked.
2. Run a deliberately small coupled neutron/photon OpenMC smoke calculation on
   that accepted H5M and export all configured volume responses. Statistical
   classification will remain `WORKFLOW_SMOKE_ONLY`.
3. Add generic statepoint extraction for heating, damage, gas, reaction, and
   particle-production tallies; the tally definitions exist, but only the
   scalar-spectrum exporter is currently complete on this branch.
4. In an isolated `DPA_workflow` branch, implement the branched schedule schema
   and update consumers to accept ParaStell boundary schema v2.2 without
   discarding facet, barycentric, topology, local-frame, or provenance fields.
5. Exercise one bounded fixture through SPECTRA-PKA input generation and each
   local-source adapter. These are software-path checks, not physics results.

No production run, production activation, port geometry, or heterogeneous
global magnet is authorized by this status.

## Verification

- Focused downstream/activation suite: 42 passed.
- Broad host-supported suite: 378 passed, 5 skipped, 5 failed only because the
  host has incompatible optional `pydagmc`/`cad_to_dagmc` APIs; six additional
  legacy test modules cannot collect because host Python lacks PyMOAB, Gmsh,
  or OpenMC. No dependency environment was installed or modified.
- Changed Python files pass Black, compileall, and `git diff --check`; compact
  JSON documents parse successfully.
- Independent read-only QA initially found statepoint/version/history,
  estimator-volume, bank-completeness, and schedule-receipt trust gaps. After
  correction it returned `PASS` with no remaining blocker/high findings.
