# OpenMC upstream-readiness acquisition preflight — 2026-09-25

This preflight was recorded before creating any new OpenMC checkout, build, or
dependency environment.

## Current local host

- Total RAM: 34,260,418,560 bytes.
- Free RAM at preflight: 3,403,771,904 bytes.
- `C:` free: 83,947,765,760 bytes of 998,965,768,192 bytes.
- `D:` free: 503,280,394,240 bytes of 2,000,397,791,232 bytes.
- Largest resident processes included WSL (`vmmemWSL`, 1.65 GB), ChatGPT,
  Codex, Chrome, OneDrive, and Windows Defender. No Docker daemon was running.
- Git: 2.55.0.windows.3.
- CMake: 4.4.0.
- Registered Python interpreters: CPython 3.12, 3.11, and 3.9. No Conda
  executable was present in `PATH`.
- Local pip cache: 1,731,292,872 bytes.
- Existing related trees inspected: the ParaStell feature worktree and the
  `bateman-openmc-dd-embree-20260831` and
  `alliance-parastell-openmc-dd-20260828` evidence trees. No local checkout of
  current `openmc-dev/openmc` was found. The private `wistell-d-openmc`
  repository is explicitly out of scope and was not inspected or modified.

The current free RAM is insufficient for a new local OpenMC+DAGMC compilation.
Local work is therefore limited to a source checkout, comparison, static
analysis, and Python-only tests. Native DAGMC compilation and transport will
use the already qualified Bateman toolchain only after separate live CPU and
memory admission checks.

## Planned acquisition

- Source: `https://github.com/openmc-dev/openmc.git`, exact live `develop`
  commit `1d75981dbf3fd78962e516c12b550d034f2e7daa`.
- Target: `D:\openmc-worktrees\fes-ii-upstream-readiness-20260925`.
- Method: partial clone without blobs followed by explicit fetches of the
  stable tag, FusionSandwich fork, Paul Romano comparison branch, and Patrick
  Shriwise comparison branches.
- Conservative local size allowance: 500 MB. No package installation, build,
  or new Python environment is authorized by this checkout step.
- Rollback: remove only the newly created target directory after verifying its
  resolved absolute path. Existing repositories, caches, and environments are
  not modified.

## Known remote identities before checkout

- Official `develop`: `1d75981dbf3fd78962e516c12b550d034f2e7daa`.
- Latest stable v0.16.0: `617d35a5063c57796b43428bc401e627d2011046`.
- FusionSandwich `develop`: `9a62e431d3101799e6179a6d0cf3b37440062e23`.
- Paul Romano `surface-source-fix`:
  `3c2e09f6b1861b627c59e8996fc513c4f495dced`.
- Patrick Shriwise `dagmc-surf-source`:
  `00477e15c0a2cfbd1680c3bd92f9d4c7458ec270`.
- Patrick Shriwise `dagmc_surf_norm`:
  `3a26fcb96ffd1d1fb6f568bf112ae1ed8f9e724c`.
- Patrick Shriwise `mu-surface`:
  `9c92971ff7a3f0582b26908492c1b876b69b3ede`.
- Patrick Shriwise `reset-dagmc-history-from-source`:
  `f69c1d9dae16245f186cef42067d6359f1f446ff`.

## Bateman build preflight

The qualified CPU-only route is `poly-bateman` (`aarch64`). A live compact
inventory immediately before staging reported:

- 256 physical cores; Codex policy ceiling 64 cores.
- 2,108,047,640 kB `MemTotal` and 2,078,089,928 kB `MemAvailable`.
- 88,153,068,142,592 bytes free on the user data filesystem and
  449,530,191,872 bytes free on `/tmp`.
- One unrelated guarded OpenMC job was active with four reserved cores, an
  8,589,934,592-byte `MemoryMax`, and `MemorySwapMax=0`.
- Shared budget at observation: 64-core ceiling, four reserved threads, four
  observed unreserved threads, 56 available threads.
- Existing compiler/runtime: GCC 13.3.0, CMake 3.28.3, Python 3.12.3, and the
  previously qualified DAGMC CMake package at
  `/home/apollon/josma/opt/openmc-0.16.0-dagmc-aarch64/lib/cmake/dagmc`.
- The existing OpenMC source checkout is clean and detached at v0.15.3. The
  qualified v0.16.0 binary is present but cannot establish compatibility with
  live `develop`.

The fresh remote attempt will receive only the 8,816,979-byte official source
archive (SHA-256
`7a8b2a3309cedb0b89de0cb245aea2d8784db1dc1365946f0fb77a31b5058229`)
and the 3,266-byte candidate patch (SHA-256
`14663dff13781f054a259930da0d8ed83cfbeea0399092f49eaac24657a777ed`).
No dependency download or installation is planned. The create-only target is
`/home/apollon/josma/data/codex-parastell/openmc-develop-fesii-compat-v1-20260925T135255Z`;
build scratch will be an attempt-specific `/tmp` directory. The bounded build
request is eight cores, 16 GiB hard memory limit, and swap disabled, followed
only by the official DAGMC shared-facet reproducer and targeted regression
test. Rollback is removal of only this fresh attempt root and its fresh scratch
directory; all qualified runtimes and source trees remain untouched.
