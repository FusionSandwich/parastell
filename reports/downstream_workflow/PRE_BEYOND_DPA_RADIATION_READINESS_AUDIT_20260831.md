# Pre-Beyond-DPA radiation readiness audit — 2026-08-31

## Scope and controlling geometry

This audit covers the public `FusionSandwich/parastell` producer and its
radiation handoffs. It does not use `wistell-d-openmc`, ports, or the rejected
18-swept-coil global geometry. The accepted global candidate is the direct
90-degree WISTELL-D radial model with nine physical DAGMC volumes and one
continuous 30 cm homogenized-magnet layer.

Native geometry qualification is complete for H5M SHA-256
`d31ea04e4d2fda78870db1688d6cc9215079e20408393c5f5bd374d58f43eaf3`:
zero unmatched edges, zero unsealed surfaces/volumes, and zero native overlap
locations at precisions 1, 2, and 4. The accepted source is the independently
audited `outer_cfs_cap=0.9655` mesh with zero invalid containment samples and
0.4205935109 cm minimum chamber clearance. OpenMC geometry debug now remains
gated on a shared chamber/first-wall interface diagnostic.

## Implemented producer capabilities

| Capability | Status | Evidence/meaning |
| --- | --- | --- |
| Direct 90-degree parametric WISTELL-D radial geometry | PASS_NATIVE_GEOMETRY | One period is generated directly; it is not assembled from two 45-degree files. |
| Continuous global magnet identity | IMPLEMENTED | Physical cell 9 is the single homogenized-magnet layer. Engineering coil IDs remain metadata/local-model identities and are not fabricated as DAGMC cells. |
| Component neutron/photon scalar flux | IMPLEMENTED_NOT_EXECUTED_ON_FINAL_SOURCE | Cell-resolved spectra for first wall, breeder, back wall, both shields, vessel, and continuous magnet. |
| Component neutron/photon heating | IMPLEMENTED_NOT_EXECUTED_ON_FINAL_SOURCE | Separate particle heating plus existing magnet total/local-mesh heating. |
| Global reaction and breeder accounting | IMPLEMENTED_NOT_EXECUTED_ON_FINAL_SOURCE | Component absorption/(n,2n)/(n,3n) families and breeder H3-production (TBR numerator) per source history. |
| Magnet damage/gas/isotope-MT/secondary production | IMPLEMENTED | Damage energy, H/D/T/He, configured isotope/MT reactions, and outgoing particle-production tallies are wired by the response plan. Nuclear-data availability is explicit rather than converting missing scores to zero. |
| Complete correlated boundary phase space | MANIFEST_PASS_OPENMC_SMOKE_PENDING | The closed capture bank uses physical surfaces 22, 25, 26, and 27 of cell 9 and records both crossing directions. Energy, particle, position, direction, weight, surface/facet identity, normal sense, mu, and local coordinates are preserved. Surface 22 is separately identified as the gap-to-magnet incoming local-replay interface. The immutable manifest passed closure and topology checks; an exact OpenMC bank is still pending. |
| Periodic statepoints | IMPLEMENTED | Every run has an explicit final statepoint and may request regular intermediate batches. |
| Restart/recovery planning | IMPLEMENTED_AND_FOCUSED_TESTED | Long fixed-source work is split into create-only, independently seeded segments with regular statepoints and surface banks. Each scheduler invocation may run exactly one requested segment, and a remaining-walltime guard refuses to start work that cannot finish before the declared stop grace. Sealed complete segments can be reused only after their receipts and every bound statepoint/surface bank are rehashed and revalidated. Statepoints are not falsely claimed to resume unfinished histories; completed segment estimators are combined with exact history accounting. |
| Prompt/delayed radiation separation | IMPLEMENTED | Prompt photons are transported with neutrons; activation-derived delayed photon sources remain separate and hash-bound. |
| Activation schedule and ALARA | IMPLEMENTED_SYNTHETICALLY_VALIDATED | Independent full-power exposures at 1 day, 1 week, 1 year, 5 years, and 10 years, with independent cooling branches and create-only ALARA decks/results. Exact final spectra/material bindings still depend on the accepted global run. |
| Downstream handoffs | IMPLEMENTED_AS_CONTRACTS | Solver-neutral, provenance-bound exports exist for activation, SPECTRA-PKA/Beyond-DPA, OpenMC local replay, Geant4, MCNP, OpenSn, and RADIANT. Native execution of the final curved/local cases is downstream work, not a claim of this global producer. |
| Non-Cartesian poster campaign planning | IMPLEMENTED | Baseline, controlled breeder, shield, and selected-location changes derive independently from the baseline and record changed versus held-fixed fields. |

## Project-control and `parastell-magnet` comparison

The useful general capabilities found in `C:\HTS_transport\parastell-magnet`
were a bounded non-Cartesian campaign planner and content-addressed staged
recovery. The producer now has the corresponding geometry-neutral campaign and
segmented checkpoint/restart contracts. Private cases, experimental curved
tape CAD, and solver-specific smoke harnesses were not copied.

The project-control review confirms that the producer already contains the
activation schedule, ALARA conversion, delayed photons, isotope/MT identity,
surface-bank localization, magnet spectra/heating/damage/gas tallies, and
downstream provenance firewall. The new continuous-geometry binding removes
the prior critical mismatch between the accepted nine-volume geometry and the
old 26-volume tally path.

## Remaining gates before the poster run

1. Resolve the OpenMC 0.16 shared-interface geometry-debug diagnosis. Source
   containment now passes with `outer_cfs_cap=0.9655`, source SHA-256
   `ed4003589d2eaca445cbd0392b8b3d0465986a0adcdb220507607f7ad97861c5`,
   and selection receipt SHA-256
   `72309c1ba5ba6dca69b1c21ad9db00a7f1cab86345556c8f4e914fe0aed6d13c`.
   Exact model export also passes. Geometry debug stopped at the first shared
   chamber/first-wall crossing, while a matched two-history track transported
   that crossing and wrote both scheduled statepoints without lost-particle or
   navigation errors. The staged official OpenMC DAGMC reproducer must decide
   whether this is an OpenMC boundary-debug defect; the gate is not waived.
2. Pass two independent bounded OpenMC geometry-debug seeds with zero lost
   particles and zero DAGMC navigation errors.
3. Pass one bounded full-response smoke that produces every scheduled
   statepoint and a non-truncated surface bank.
4. Freeze the actual poster material bindings and the small staged case list.
5. Benchmark the exact model briefly on the authorized Alliance allocation and
   generate the eight-hour segmented run deck. Obtain explicit user
   authorization before submission.

## Deliberately not claimed

- No production-quality uncertainties or large-run statistics exist yet.
- Weight windows, MAGIC, and DoubleDown are not enabled. The first accepted
  run remains analog; variance reduction requires an independent multi-seed
  FOM comparison and an unbiased baseline.
- The provisional tape stack is not a user-qualified native curved magnet.
- SPECTRA-PKA recoil folding, covariance, arc-DPA, and Beyond-DPA event-source
  execution require the final spectra and accepted recoil matrices.
- Native Geant4, MCNP6.3, OpenSn, and RADIANT execution of the final selected
  heterogeneous magnet remains downstream of the accepted global bank.

Current decision: `BLOCKED_OPENMC_GEOMETRY_DEBUG_SHARED_INTERFACE_DIAGNOSIS`.
