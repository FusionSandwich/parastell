# OpenMC 0.16 direct-90 navigation qualification — 2026-08-31

The accepted direct 90-degree ParaStell geometry, contained source, OpenMC
navigation, periodic statepoints, response tallies, and complete crossing-bank
workflow now pass bounded qualification.

The immutable H5M SHA-256 is
`d31ea04e4d2fda78870db1688d6cc9215079e20408393c5f5bd374d58f43eaf3`.
It has nine physical volumes and 27 surfaces. Native watertightness, overlap
checks at precisions 1/2/4, surface senses, and material ownership pass. The
selected `outer_cfs_cap=0.9655` source has zero invalid samples, 0.4205935109 cm
minimum chamber clearance, and a 90-degree physical source rate of
`1.855454485735326e20 n/s`.

The earlier unpatched geometry-debug failure at the chamber/first-wall shared
facet was reproduced with OpenMC's own DAGMC regression model. A narrow local
diagnostic patch was then qualified against both that positive control and a
known nonadjacent physical-overlap negative control. Independent WISTELL-D
debug seeds 8310101 and 8310102 pass with zero lost particles and zero DAGMC
navigation errors. The patch is diagnostic-only; the physical H5M is unchanged.

A 10,000-history full-response smoke then ran with the unpatched stock OpenMC
0.16.0 binary. It wrote statepoints at batches 1, 2, 3, and 4, instantiated 25
magnet/reactor tally objects, and produced a surface bank with 930 records:
897 neutron and 33 photon. The bank covers surfaces 22, 25, 26, and 27 and
retains position, direction, energy, time, weight, delayed group, surface ID,
and particle identity. Its 930 records are well below the explicit 100,000
record per-file cap; eight files were allowed and one was written. No lost
particle, navigation, truncation, or capacity warning occurred.

This is infrastructure evidence, not a statistical qualification. The current
decision is `READY_FOR_SOURCE_MESH_AND_MEDIUM_STATISTICS`. No production run is
authorized.
