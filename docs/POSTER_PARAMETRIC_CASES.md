# WISTELL-D poster parametric cases

The poster campaign uses a direct 90-degree ParaStell geometry and a staged,
non-Cartesian case plan. Every variation derives directly from `P00`; breeder
and shield changes are not multiplied together implicitly.

The reusable workflow is:

1. bind the accepted immutable 90-degree H5M and accepted physical source;
2. compile P00--P07 against a hash-bound `fusion-material-db` JSON;
3. run bounded geometry/navigation and tally smoke gates;
4. obtain separate user authorization before submitting the large campaign;
5. execute restartable segments with a statepoint at least every 60--120 minutes.

The material recipes were transcribed from read-only historical
BlanketNeutronics study branches and the Type One constants in
`stellarator_optimization`. No private geometry or result artifact is copied.

## Current case readiness

| Case | Concept | Preparation status |
| --- | --- | --- |
| P00 | LiPb/DCLL-like + WC baseline | ready for bounded smoke |
| P01 | HCPB | ready for bounded smoke |
| P02 | FLiBe-LIB, 60% Li-6 | ready for bounded smoke |
| P03 | HCLL | conditional, recipe resolved |
| P04 | SCLV | blocked: portable CaO and exact 6.5% Li-6 primitives incomplete |
| P05 | natural-boron W2B5 | ready for bounded smoke |
| P06 | exact documented Type One HTS/LTS mixture | ready for bounded smoke |
| P07 | TiH2 then W2B5 | blocked: approved TiH2 recipe/density and layer mapping missing |

“Ready for bounded smoke” does not mean nuclear-data coverage or the physical
geometry gate has passed. Those remain mandatory run-time gates. Weight windows
remain disabled until an unbiased analog baseline is accepted.

Use `scripts/build_poster_parametric_cases.py` to create one immutable material
bundle and OpenMC `materials.xml` per runnable case. The compiler never launches
OpenMC or writes a scheduler submission.

