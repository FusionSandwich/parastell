# Bateman refined-geometry preflight

## Authorization and routing

The user authorized bounded CPU-only work on multiple Polytechnique servers at no more than 25% of each server's physical cores. The geometry-only lane routes to `poly-bateman`; OpenMC transport is explicitly excluded from this host and skill. The persistent SSH broker was reused initially, then restarted once after its socket was confirmed closed. No repeated retry loop is permitted.

## Live host and resource identity

- Host/alias: `bateman` / `poly-bateman`; account `josma`; architecture `aarch64`.
- Processor topology: 256 physical cores, 256 scheduler-visible CPUs, one thread per core. The 25% cap is 64 cores.
- Guarded CPU ledger immediately before preparation: 0 reserved, 0 observed-unreserved, 64 available.
- Memory: 2,108,047,640 kB total; 2,079,259,124 kB available.
- Target filesystem: `/home/apollon`, 161,994,901,553,152 bytes total and 76,620,764,282,880 bytes available.
- No equivalent live CadQuery, cad-to-DAGMC, Gmsh, ParaStell, watertightness, or overlap process was observed for the account.

## Existing software and source

- Fixed interpreter: `/home/apollon/josma/data/transport/bin/python` resolving to Python 3.11.15.
- Import gate: Paramak 0.9.11, cad-to-DAGMC 0.11.5, CadQuery 2.8.0, Gmsh 4.15.2, PyMOAB 5.6.0, ParaStell 0.1.0, and parastell-damage 0.5.0 all imported successfully.
- Existing package cache: `/home/apollon/josma/.cache/pip`, 5.7 GiB. No acquisition or environment modification is needed.
- Existing clean source repository: `/home/apollon/josma/data/src/parastell`, `main` at `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`, matching the verified fork/upstream main authority.
- Frozen remote VMEC SHA-256: `1cebb8d46e60d77df4a6904662a9c9f943137a9fb59f7290e5309af15fa04797`.
- Frozen remote coil-file SHA-256: `de96b6009356c0d7b2abd9fd0507589651a7c04447011e0c4d3ed0d2f5756737`; the Git blob is the same `main` input, while checkout line-ending bytes differ from Windows and will be recorded explicitly.

## Planned additive acquisition

Local execution remains valid but is occupied by the exact coarse source-CAD Boolean audit and has 31.91 GiB RAM. Bateman's existing qualified stack and 2 TiB RAM allow the independently named refined candidate to proceed without contending for local memory.

The remote lane will create a fresh detached worktree from the exact main SHA and upload only four hash-bound files:

- `parastell/reference_geometry.py`: 23,840 bytes, SHA-256 `74436b8620b86cf117269a46c390919dacddb32b961b7f3e84387dc73a37d053`.
- `parastell/public_geometry_candidate.py`: 2,352 bytes, SHA-256 `3a14a615d04d6eb4fb8f95bb2a1b69904879108e627814eb5c98ca16917605a5`.
- `scripts/build_clearance_candidate.py`: 6,678 bytes, SHA-256 `c9026e9203c6076becd825d1ab3a02247434ac385c02f1144b0df6a97602d38a`.
- `clearance_measurement.json`: 174,773 bytes, SHA-256 `5b8d86f91c17b77151303f184e0ed1f28d4424ce39d8b13bb9cce87deb18a166`.

The checked-out source tree is approximately 22.6 MB. The planned remote source/config acquisition is bounded below 64 MB; geometry outputs are capped by a 2 GiB planning allowance. No package, compiler, or dependency download is planned.

## Fresh targets and staged execution

All were proven absent before creation:

- Worktree: `/home/apollon/josma/data/src/parastell-prompt7r-a1-refined-20260827T015636`.
- Construct-only output: `/home/apollon/josma/data/parastell_prompt7r_geometry/a1_construct_20260827T015636`.
- Refined full output: `/home/apollon/josma/data/parastell_prompt7r_geometry/a1_refined_20260827T015636`.
- Full-build log: `/home/apollon/josma/data/parastell_prompt7r_geometry/logs/a1_refined_20260827T015636.log`.

The staged plan is a guarded 4-core construct-only smoke, followed only on success by a separate guarded 32-core refined CAD-to-DAGMC export. Both commands set the fixed interpreter, `set -euo pipefail`, and BLAS/OpenMP caps. Outputs are create-only and commands remain in the foreground under the core-budget lease.

## Rollback and preservation

No existing remote file will be overwritten. Failed and successful attempt directories, logs, input hashes, and receipts are retained as evidence. If cleanup is later required, only the exact fresh detached worktree may be unregistered after its absolute path and SHA are reverified; the base repository, environments, caches, and attempt artifacts are never removed by this workflow.
