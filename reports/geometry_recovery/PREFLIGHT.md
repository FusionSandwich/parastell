# Prompt 7R preflight

## Independent vanilla-reference repository acquisition

- Observation time: 2026-08-27 America/New_York.
- System RAM: 31.91 GiB total, 2.88 GiB free while the bounded source-CAD audit was active in Docker.
- Local storage: `C:` 189.80 GiB free of 930.42 GiB; `D:` 469.92 GiB free of 1863.02 GiB.
- Significant active work: Docker Desktop and its backend were running the sole local source-CAD candidate audit; small Python controller processes were also present. No second CAD/DAGMC computation will share that local resource lane.
- Existing executables: Git 2.55.0.windows.3, Python 3.12.10, Docker client/server 28.4.0. Conda is not on `PATH`.
- Existing environments and caches: the qualified `parastell-openmc:0.16.0` Docker image is already local; the pip cache is at `C:\Users\joshu\AppData\Local\pip\cache` and is 1623.7 MB. No package acquisition is needed.
- Existing source checkouts: the user-owned working repository and its registered worktrees include a clean detached `main` reference at `D:\parastell-worktrees\main-reference-parastell`. Both fetched `origin/main` and `upstream/main` resolve to `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`.
- Why another checkout is needed: the existing reference is an isolated worktree but shares its object database and worktree registry with the development repository. The user explicitly required a separate repository. The new shallow repository supplies stronger provenance and guards against accidental feature-branch state or shared-worktree mutation.
- Planned acquisition: copy the already-local, fetched `main` commit into an independent shallow Git repository; no network download and no dependency installation. The checked-out tree contains 22,622,015 blob bytes. The conservative planned upper bound, including Git metadata, is 55,729,790 bytes.
- Target: `D:\parastell-reference-repos\parastell-vanilla-main-de7d297` (confirmed absent before creation).
- Rollback: if creation fails, quarantine only that exact newly created target after resolving and verifying its absolute path; never touch the source repository, existing worktree, reference artifacts, or dependency caches.
- Reference build outputs remain outside Git under a unique, no-overwrite directory in `D:\parastell-artifacts\geometry-recovery-20260827`.

The independent reference repository is read-only evidence. It will not receive feature commits, instrumentation, H5M mutation, ports, casing/winding splits, or post-export physical edits.
