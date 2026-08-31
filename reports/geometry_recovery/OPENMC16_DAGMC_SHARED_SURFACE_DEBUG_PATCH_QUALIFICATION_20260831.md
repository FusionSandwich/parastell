# OpenMC 0.16 DAGMC shared-surface debug qualification

The OpenMC geometry-debug failure was reproduced with OpenMC's own DAGMC
regression geometry, then resolved with a local diagnostic-only patch. The
patch does not change ParaStell geometry and is not used for physics transport.

OpenMC 0.16 leaves a particle exactly on a crossed DAGMC facet after `next_vol`
selects the destination volume. Its generic overlap checker then asks every
volume whether it contains that boundary point. The unpatched DAGMC
`contains()` path ignores the known crossed surface, allowing both legitimate
neighbors to answer true.

The qualified patch exempts only the one candidate volume proven by DAGMC
topology to be the neighbor across the exact crossed surface. It continues to
test every third/nonadjacent DAGMC volume and every CSG cell. Its SHA-256 is
`77c959527e589c7cc214a0db2942cf53e7d2b967f4c6b01613ba65ba951c50f5`.

The official shared-face regression model passes with the patched diagnostic
binary. A known physically overlapping ParaStell reference still fails with
`Overlapping cells detected: 26, 4 on universe 1`, proving that nonadjacent
overlaps remain visible. The exact accepted WISTELL-D model passes independent
seeds 8310101 and 8310102 with zero lost particles and zero DAGMC navigation
errors. The accepted H5M hash remains
`d31ea04e4d2fda78870db1688d6cc9215079e20408393c5f5bd374d58f43eaf3`.

Classification: `PASS_DIAGNOSTIC_PATCH_QUALIFIED`.

All response and statistics runs must use the unpatched stock OpenMC 0.16.0
binary. No upstream patch or pull request was created.
