# OpenMC 0.16 DAGMC shared-surface debug patch plan

This is a local, diagnostic-only patch against exact OpenMC commit
`617d35a5063c57796b43428bc401e627d2011046`. It is not an upstream change and
is not a production-transport modification.

The narrowed patch has SHA-256
`77c959527e589c7cc214a0db2942cf53e7d2b967f4c6b01613ba65ba951c50f5`.
An independent read-only source review accepted the topology-neighbor-only
design, subject to compilation and the positive/negative runtime controls
below. The earlier broad universe-level exemption was rejected and is not the
patch being built.

The failure has an upstream-independent reproducer: OpenMC's own 0.16 DAGMC
`tests/regression_tests/dagmc/legacy/dagmc.h5m` (SHA-256
`b4da1d4254a8476f31584f532e910aa46e5f5919c63343dd651f68be42384974`)
fails `openmc -g` at its first shared-surface crossing with:

```text
Overlapping cells detected: 1, 2 on universe 1
```

The exact reproducer model XML SHA-256 is
`aff51b2e7e66ab0e419212b1e22d4f5325df795a92f8ce4cf8cc4a17e9891744`;
the terminal log SHA-256 is
`ce5f31dde44b1b2f09051d1b99f650bf0aea81d1c241612cad8e4d95546a7dfa`.

OpenMC 0.16 selects the next DAGMC cell topologically with `next_vol`, leaves
the particle position exactly on the crossed facet, and immediately calls the
generic overlap checker. That checker asks every cell whether it contains the
boundary point. `DAGCell::contains` delegates to `point_in_volume` and ignores
the supplied `on_surface` argument, so both legitimate adjacent volumes can
answer true.

The patch exempts only the candidate DAGMC cell that `next_vol` proves is the
topological neighbor of the current cell across the exact crossed DAGMC
surface. It continues to check:

- every third or nonadjacent DAGMC cell at the same crossing;
- every CSG cell and wrapper;
- source sites, collisions, and positive flights;
- native DAGMC watertightness, surface-sense, and overlap checks;
- all stock OpenMC production transport.

Qualification sequence:

1. Build in a new detached source worktree with the existing local compiler,
   DAGMC, MOAB, HDF5, and CMake cache; no software acquisition.
2. Re-run the official OpenMC regression-H5M reproducer with the patched
   diagnostic binary and require it to pass.
3. Run a deliberately finite-overlap DAGMC negative control and require the
   patched diagnostic binary to reject it. This proves that the exemption is
   limited to the expected shared-face neighbor pair. The first control is the
   preserved public-reference R1 H5M (SHA-256
   `8741dd48fded42e8411816e56e3e5e10a29db26ddb785b4a389f8a38b09707a0`),
   whose native audit proves four nonadjacent vacuum-vessel/magnet overlaps;
   its corrected OpenMC model XML has SHA-256
   `6cf9b8bb19e2ad99e3634fc5a0e030b34918d8b2f852f86b2234bda4c4836b2b`.
   A small three-volume A/B/C fixture is retained as the stronger targeted
   control if the R1 run does not exercise a nonadjacent overlap deterministically.
4. Run the exact WISTELL-D seed 8310101 geometry debug. Only if it passes, run
   independent seed 8310102.
5. Run the bounded regular stock-binary response smoke and require zero lost
   particles/navigation errors plus all scheduled statepoints and a complete,
   non-truncated surface bank.

Any interior overlap, lost particle, navigation error, unexpected regression,
or input-hash drift remains a hard failure.

## Terminal result

All five qualification steps passed. The accepted diagnostic binary SHA-256
is `8a5c03d6cbad34af833ff95055b68100e24f104079598e6036c902a79fa28f64`.
The official shared-face control passes, the known nonadjacent-overlap control
is still rejected, both WISTELL-D seeds pass, and a 10,000-history response
smoke with the unpatched stock binary produced every scheduled statepoint and
a non-capacity-limited neutron/photon surface bank. See
`OPENMC16_DAGMC_SHARED_SURFACE_DEBUG_PATCH_QUALIFICATION_20260831.json`.
