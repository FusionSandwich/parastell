# Reference identity audit

## Decision

`SEPARATE_WISTELL_D_ASSETS_FOUND`

R1 is the exact current public ParaStell CAD-to-DAGMC example at fork and upstream SHA `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`. Its script, VMEC, coils, embedded radial build, source mesh, environment, H5M, and canonical fingerprint are frozen in `REFERENCE_GEOMETRY_MANIFEST.json`.

The public files do not contain machine-name evidence that identifies R1 as WISTELL-D. The name is therefore not inferred.

## Separate WISTELL-D evidence

Two clean local checkouts contain identical preserved WISTELL-D inputs at commit `bc4ab3d0f27369d4eda908a3fc187a10b6c7fedb`:

- `D:\Beyound DPA Repos\BlanketNeutronics`, branch `t1e_main`;
- `D:\Beyound DPA Repos\blanket`, branch `Edgar`.

The primary assets are `plasma_wistelld.nc`, `coils_wistelld.txt`, and `neutronics_wistelld_DCLL_1/radial_distances.csv`; their exact hashes are recorded in `WISTELL_D_REFERENCE_MANIFEST.json`.

The historical README describes Wistell D with a DCLL/FNSF blanket, explicitly says “no magnets,” and says magnets were intended as a later radial layer. The workflow uses a different VMEC, coil file, radial build, grid, and legacy Cubit path from R1. No preserved WISTELL-D `dagmc.h5m`, source-mesh H5M, or validation receipt was found in the requested local trees. Regeneration was not invented from incomplete historical metadata, and this separate R2 lane did not delay R1.

## R3 backend cross-check

The untouched public PyDAGMC example is not a matched-backend copy of R1: it uses a 40 cm rather than 50 cm shield, different rib settings, omits the R1 chamber and source mesh, and changes magnet faceting. It is therefore classified `NOT_RUN_NONCOMPARABLE_PUBLIC_INPUTS`; byte identity was never expected, and semantic parity was not falsely claimed.

