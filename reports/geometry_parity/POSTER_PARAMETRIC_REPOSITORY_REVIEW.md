# Poster parametric repository review

Date: 2026-09-01

## Scope and safety

The review was read-only in `BlanketNeutronics`, `radial_build_tools`,
`fusion-material-db`, `stellarator_optimization`, and
`C:/HTS_transport/parastell-magnet`. Only
`FusionSandwich/parastell` was changed. No private geometry/input file was
copied, no port was introduced, no OpenMC transport was launched, and
`wistell-d-openmc` was not used.

## Reusable study pattern

The historical studies used OpenMC with `radial_build_tools` toroidal screening
models, PyNE/OpenMC material mixing, NumPy parameter lists, per-case statepoint
files, and Slurm wrappers. The useful pattern is a baseline plus controlled
one-family variations. The older branches also contain large Cartesian
thickness loops; those were not copied because the current poster plan requires
a bounded staged campaign.

Reviewed identities include:

- BlanketNeutronics DCLL main reference: `bc4ab3d0f27369d4eda908a3fc187a10b6c7fedb`
- FLiBe-LIB: `ed3e8bad9c1f5f7910c452cfd4b3bc84e3088aee`
- HCLL: `080077656bd977cf9af9f92d09fe632b12691bdf`
- SCLV: `f7e14fb268aee6b1a3dcd2b1f10c4337942b59a6`
- HCPB torus: `d65ad05d1bba384da1de42730e7c0d144f3d2881`
- HCPB shielding: `5a57f81993bc9e0abea2a3b40a5e45dc2ad7d901`
- `radial_build_tools`: `d195d5a9f777c3ac42a9033646937558e07748e0`
- `fusion-material-db`: `9d8f84ad7b3ac2c587edfbe8ec1ed74891484498`

## Implemented disposition

The P00--P07 compiler is staged and non-Cartesian. P00 is the only parent.
Material assignment does not mutate the accepted 90-degree H5M.

- P00 DCLL/WC: resolved.
- P01 HCPB: resolved.
- P02 FLiBe-LIB: resolved using 60%-Li-6 FLiBe/Inconel recipes.
- P03 HCLL: conditionally resolved using the documented Pb-15.7Li90,
  EUROFER97, and helium recipe.
- P04 SCLV: fail-closed because an approved portable CaO row and exact 6.5%
  Li-6 primitive are not bound in the selected database.
- P05 natural-boron W2B5: resolved from the Type One 15.3 g/cm3 density and
  natural B/W isotope distributions in the bound database.
- P06 Type One HTS/LTS: resolved from the documented 60/20/20 W2B5/RAFS/He
  and 20/30/50 RAFS/borated-RAFS/water mixtures.
- P07 TiH2 then W2B5: fail-closed because the reviewed model contains an alias
  but no approved TiH2 density/isotope recipe or unambiguous mapping onto the
  accepted separated shield volumes.

## Real-database compiler smoke

Output root:
`D:/parastell-artifacts/poster-parametric-cases/20260901T000700`

Plan SHA-256:
`10637dae82c1c2ea776022ec988f3c0581c40ab2b388053d6abd193ce429235e`

Serialized plan file SHA-256:
`1b48b44e160c19cffde0e9da5f09054b9c9de44584e8801828e3b06be13fea33`

P05 OpenMC materials XML SHA-256:
`73101d7807657be329269b82c91ebd4142e66f7132ab0de3c8008679d1cb7a0a`

The smoke created deterministic bundles and OpenMC `materials.xml` files for
P00, P01, P02, P03, P05, and P06. It correctly created no runnable material
artifacts for P04 or P07. Geometry and source are intentionally `NOT_BOUND` in
this compiler-only smoke.

Verification: 79 focused geometry/material/OpenMC-control/surface/activation
tests passed; the modified Python files compiled; Black and `git diff --check`
passed. The repository-wide `compileall` traversal was stopped because local
filesystem latency made it non-informative, so the four changed Python files
were compiled directly instead.

## Remaining launch gates

Before even a bounded transport campaign, bind the accepted immutable 90-degree
H5M, accepted physical source, exact OpenMC 0.16 nuclear-data manifest, and the
qualified tally/surface-bank configuration. Run geometry/navigation smoke and
nuclear-data coverage for every case. Keep analog transport and weight windows
off until the accepted unbiased baseline exists. A large poster campaign still
requires separate explicit user authorization.
