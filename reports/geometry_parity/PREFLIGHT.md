# Geometry Parity Preflight

Recorded: 2026-08-26T16:53:08-04:00

## Repository state

- User-owned fork: `FusionSandwich/parastell`.
- Read-only upstream: `svalinn/parastell`.
- Live fork `main`: `de7d2978ff314b060ca2e6b10745a034e8b2a3c4` (verified through authenticated GitHub metadata on 2026-08-26).
- Live upstream `main`: `de7d2978ff314b060ca2e6b10745a034e8b2a3c4` (verified through authenticated GitHub metadata and the public commit page on 2026-08-26).
- Failed evidence branch: `df53ed28edc8d8123b6e42a9e4e5c4e970e69dc8` (verified through authenticated GitHub metadata).
- Merge base of fork/upstream `main`: `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`.
- Merge base of `main` and the failed evidence branch: `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`.
- Detached R1 worktree: `D:\parastell-worktrees\main-reference-parastell`, clean at `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`.
- Feature worktree: `D:\parastell-worktrees\magnet-radiation-geometry-parity`, clean at creation on branch `magnet-radiation-geometry-parity-20260826` based directly on fork `main`.
- Failed feature worktree retained read-only: `D:\parastell-worktrees\magnet-surface-field-visual-production-gate`, clean at `df53ed28edc8d8123b6e42a9e4e5c4e970e69dc8`.
- The pre-existing primary checkout is dirty on `JS/openmc-0.16-compat`; its unrelated files were not touched.
- Existing stashes and all other worktrees are preserved.
- Ordinary `git fetch` transport was bounded and interrupted after hanging without output. Exact remote heads were therefore verified with the authenticated GitHub API instead of accepting cached refs by name.

## Machine and storage

- Host: Windows 11 Home, build 10.0.26200.
- CPU: 20 scheduler-visible logical processors locally.
- RAM: 34,260,418,560 bytes installed; approximately 5.2 GB physical memory was available during the initial snapshot.
- `C:`: 999,032,877,056 bytes total; 200,907,751,424 bytes free.
- `D:`: 2,000,397,791,232 bytes total; 476,091,953,152 bytes free.
- Significant active software included Codex, Docker Desktop/WSL, browsers, and ordinary desktop services. No pre-existing local transport or geometry build container remained active at the final Phase-0 snapshot.
- WSL distributions: Debian and docker-desktop running; OpenMC-Dev-D stopped.
- Active Codex team at the initial snapshot: root plus three bounded read-only reviewers. No reviewer writes to either task worktree or the artifact attempt.

## Existing capabilities

- Git 2.55.0.windows.3; Git LFS 3.7.1.
- Host Python 3.12.10; no active Conda environment; no host OpenMC, Gmsh, or MPI executable was selected for geometry work.
- Docker 28.4 and WSL 2.7 are installed.
- Existing image `parastell-openmc:0.16.0` (about 16.2 GB) contains Python 3.12.13, ParaStell 0.1.0, OpenMC 0.16.0, cadquery 2.7.0, cad_to_dagmc 0.11.5, Gmsh, PyMOAB 5.5.1, PyDAGMC 0.0.1, PyVista 0.48.4, VTK 9.3.1, and Matplotlib 3.9.1.
- Existing qualified nuclear data: `D:\parastell-artifacts\openmc-data\nndc_hdf5\cross_sections.xml` with its local HDF5 library. No nuclear-data download is required.
- Existing source/build caches and numerous user-owned geometry artifacts were inventoried read-only under `D:\parastell-artifacts`; none is an accepted R1 result merely because of its filename.
- The public example inputs are ordinary Git objects, not missing Git-LFS objects. The initial worktree population hang was recovered by creating no-checkout worktrees, restoring from `HEAD`, and rebuilding their indexes without network acquisition.

## Remote activity snapshot

- Polytechnique scheduler-visible total: 256 logical cores; cross-task Codex cap: 64; observed and leased use: 0; available under policy: 64. No remote work was launched.
- Alliance Trillium live broker: `squeue` returned no jobs for the user.
- Alliance Nibi broker state was stale/connection-refused. No new Duo login was initiated merely to enumerate it.
- No Alliance or Poly job is authorized or needed for Phase 1 while the local qualified environment remains sufficient.

## Acquisition gate

No dependency install, download, upgrade, or environment build is planned. Existing local software is sufficient. Planned acquisition bytes: **0**.

Task-created outputs will be written create-only below `D:\parastell-artifacts\geometry-parity-20260826`. The initial R1 attempt is `r1_public_main\20260826T165308-0400`. A conservative working allowance of 20 GB is reserved against 476 GB free on `D:`. Rollback is limited to removing the explicitly named new worktrees and task-created attempt directories after path verification; no shared environment or cache is modified.

### QA frontend addendum

The qualified container does not contain the optional `black` or `build` frontends. Before any acquisition, a disposable external target and limits were recorded: at most 25 MB transfer and 150 MB installed below `D:\parastell-artifacts\geometry-parity-20260826\qa_tools\black-build-20260826`, with rollback limited to that exact directory. The bounded PyPI attempt encountered DNS failure and was stopped; it installed no files.

An explicit check of the already-installed host Python 3.12.10 then located Black 25.12.0 and build 1.5.0. Those existing tools are used for the formatting and package gates. Consequently, actual acquisition bytes and installed bytes remain **0**, and neither the repository nor a dependency environment is modified.

## Scope locks

- R1 physical inputs are immutable.
- No ports, casing/winding split, transforms, radial-build changes, production transport, source convergence, variance reduction, activation, PR, upstream edit, or default-branch merge are authorized.
- Large STEP/H5M/VTK/XDMF/render caches remain outside Git.
