# OpenMC 0.16 direct-90 navigation diagnosis — 2026-08-31

The physical ParaStell geometry and source-domain gates now pass, but the
OpenMC geometry-debug gate remains fail-closed.

The immutable direct 90-degree H5M is
`d31ea04e4d2fda78870db1688d6cc9215079e20408393c5f5bd374d58f43eaf3`.
It has nine physical volumes and 27 surfaces. Native watertightness, overlap
checks at precisions 1/2/4, surface senses, volume closure, and material
ownership pass. The newly selected `outer_cfs_cap=0.9655` source mesh has SHA
`ed4003589d2eaca445cbd0392b8b3d0465986a0adcdb220507607f7ad97861c5`,
zero invalid source-domain samples, and a minimum chamber clearance of
0.4205935109 cm. Its 90-degree physical source rate is
`1.855454485735326e20 n/s`.

Exact OpenMC 0.16.0 geometry debug with seed 8310101 stopped in batch 1 with:

```text
Overlapping cells detected: 1, 2 on universe 33
```

Cells 1 and 2 are not the external CSG wrapper. They are the native chamber
and first-wall DAGMC volumes. They have one imprinted shared interface,
surface 1, with forward sense to chamber volume 1 and reverse sense to
first-wall volume 2. Their independent envelope audits close with positive
signed volume and opposite outward orientation on that exact shared surface.

A matched, non-debug diagnostic used the identical model XML and exact OpenMC
runtime with one particle per batch for two batches. It exited zero, wrote
statepoints at batches 1 and 2, and reported no lost particles or DAGMC
navigation errors. The first tracked history started in chamber cell 1 and
crossed into first-wall cell 2 at
`(916.82516755, 342.45603918, 77.43396207) cm`, then continued into breeder
cell 3 and completed normally. This identifies the fatal debug message with
the first legitimate shared-interface crossing rather than a wrapper-cell
overlap.

The OpenMC 0.16.0 implementation checks every cell's `contains()` result after
each move. Its DAGMC `contains()` implementation calls `point_in_volume`
independently for each volume and does not use the supplied `on_surface`
argument to disambiguate a particle known to be exactly on a shared DAGMC
surface. That is a strong external-debugger hypothesis, not yet an accepted
waiver or external-defect conclusion.

The official OpenMC 0.16 DAGMC regression H5M minimal reproducer is staged on
Bateman. It was not launched because the live shared core ledger reached its
64-core policy ceiling before the guarded one-core launch. The next launch
must remain one core, 1 GiB, swap disabled, and must not run until capacity is
available. If it reproduces the shared-boundary failure, the permitted next
step is a local diagnostic-only patch or equivalent boundary-aware proof; the
ParaStell H5M remains immutable.

Current status: `BLOCKED_OPENMC_GEOMETRY_DEBUG_SHARED_INTERFACE_DIAGNOSIS`.
No full-response smoke or production run is authorized.
