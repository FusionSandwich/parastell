# Pre-Beyond-DPA radiation readiness audit — 2026-08-31

## Controlling geometry and source

The accepted global producer is the direct 0–90 degree WISTELL-D ParaStell
model with nine nested radial volumes and one continuous 30 cm homogenized
magnet layer. It is not assembled from two 45-degree models and does not use
the rejected 18-swept-coil geometry.

The exact H5M SHA-256 is
`d31ea04e4d2fda78870db1688d6cc9215079e20408393c5f5bd374d58f43eaf3`.
Native watertightness, overlap precisions 1/2/4, surface senses, material
ownership, and continuous-magnet topology pass. The accepted source mesh SHA
is `ed4003589d2eaca445cbd0392b8b3d0465986a0adcdb220507607f7ad97861c5`;
it has zero invalid samples and a 0.4205935109 cm minimum chamber clearance.

## Executed bounded evidence

| Capability | Status | Evidence |
|---|---|---|
| Physical direct-90 geometry | PASS | Nine volumes, 27 surfaces, zero native overlaps or leaks. |
| OpenMC 0.16 navigation | PASS | Two independent geometry-debug seeds; zero lost particles and navigation errors. |
| Surface phase space | PASS_BOUNDED | Closed surfaces 22/25/26/27; 930 records with position, direction, energy, time, weight, delayed group, surface ID, and particle identity. |
| Neutron/photon coverage | PASS_BOUNDED | 897 neutron and 33 photon records. |
| Bank completeness | PASS_BOUNDED | 930 records versus 100,000 per-file cap; one of eight allowed files written; no truncation warning. |
| Periodic statepoints | PASS | Batches 1, 2, 3, and 4 written in the 10,000-history smoke. |
| Reactor and magnet responses | PASS_WIRING_AND_EXECUTION | 25 tally objects cover reactor component flux/heating/reactions/T production and magnet spectra, current, heating, damage, gas, isotope/MT, and secondary production. |
| Checkpoint/recovery plan | IMPLEMENTED_FOCUSED_TESTED | Independent seeded segments, regular statepoints, complete banks, walltime guard, and hash-validated reuse. |
| Activation and downstream handoffs | IMPLEMENTED_AS_CONTRACTS | ALARA, OpenMC depletion, SPECTRA-PKA/Beyond-DPA, Geant4, MCNP, OpenSn, and RADIANT exports exist; accepted field extraction and native downstream runs remain. |

The shared-interface message from unpatched OpenMC geometry debug was proven to
be an OpenMC 0.16 DAGMC exact-facet false positive using OpenMC's own regression
geometry. A narrow local diagnostic patch passes that positive control, still
rejects a known nonadjacent physical overlap, and passes both exact WISTELL-D
seeds. The patch is never used for physics transport; the full-response smoke
used the stock OpenMC 0.16.0 binary.

## What remains before the poster-scale run

1. Complete strict facet/barycentric localization and exact-run bank/tally
   closure for the accepted 930-record bank.
2. Run the bounded source-mesh response-convergence ladder.
3. Run medium-statistics independent segments and inspect uncertainty/ESS.
4. Extract the accepted volume fields and surface bundle into activation,
   PKA, and local-magnet handoff formats.
5. Freeze the small, scientifically approved material/thickness case list.
6. Benchmark and prepare the restartable eight-hour Alliance deck.
7. Obtain explicit user authorization before any large run.

The 10,000-history result proves the workflow, not research statistics. Current
decision: `READY_FOR_SOURCE_MESH_AND_MEDIUM_STATISTICS`.

No production run is authorized.
