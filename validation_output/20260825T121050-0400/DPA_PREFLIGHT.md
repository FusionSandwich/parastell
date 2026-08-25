# DPA_workflow Prompt-1 preflight

Snapshot: `2026-08-25T12:10:50-04:00`.

## Repository and live refs

- Repository: `FusionSandwich/DPA_workflow`, common checkout `D:\2026_DPA\DPA_workflow`.
- Remote: `origin=https://github.com/FusionSandwich/DPA_workflow.git`.
- `git fetch --all --tags --verbose --no-recurse-submodules` completed; all advertised refs and the tag were up to date.
- The two cited integration candidates are local-only tips after the fetch; neither has an `origin/*` tracking ref.
- Relevant worktrees were clean: consolidation `4836708e7aa746750513484081bf50194ab5dbe2`, Beyond-DPA checkpoint `cbb31c1e015895cc95a9fe80ca5b8b3f9c5058dd`, and magnet consumer `fcf8c0dc3ee1035d74189ac364a5edc5274b05c1`.
- The shared stash list was empty.
- Nineteen pre-existing linked worktrees were recorded. The primary checkout had three modified planning files; `openmc-to-atomistics-e2e-20260816` had six modified and 55 untracked files; `ybco-distributed-gap-alliance` had 55 untracked files. They remain untouched.

## Ancestry and integration-base decision

- `merge-base(cbb31c1e, fcf8c0dc) = 0b428c11deee11d4881cbc2d5852a6b229dde51c`.
- Each cited tip is four commits ahead of that merge base; neither is an ancestor of the other.
- `fcf8c0dc` descends from consolidation `4836708e` and adds three magnet-consumer commits: `89df4752`, `0a34a1cd`, and `fcf8c0dc`.
- `cbb31c1e` carries the newer canonical R1/R2 reader migrations and Beyond-DPA workflow.
- Their feature-file sets do not overlap. Prompt 1 therefore starts from `cbb31c1e` and replays only the three magnet-consumer commits, omitting the older documentation-only consolidation checkpoint `4836708e`.
- New worktree: `D:\2026_DPA\worktrees\magnet-test-spectra-activation`.
- New branch: `codex/magnet-test-spectra-and-activation-20260826`.
- Integration checkpoint after the three clean cherry-picks: `f4330525e696f89b27d66536e05465dcde20307c`.
- Canonical package root is confirmed by `pyproject.toml`: `src/parastell_damage/`.

## Preservation

- Complete prefetch bundle: `D:\prompt1-preservation\20260825T121050-0400\dpa_workflow\DPA_workflow-all-refs-prefetch.bundle`.
- Bundle size/SHA-256: 160,939,333 bytes / `4f1ae7689c7b82af45e59ba7fd2b63db87d459e7773af2594526254ea294daf5`.
- `git bundle verify` reports complete history and includes both `cbb31c1e` and `fcf8c0dc`.
- Primary-checkout tracked patch: 169,723 bytes / `9a97c83f00462975133e2bc47372362a97e95d9f488394b0beac4be0558a4963`.
- OpenMC-to-atomistics tracked patch: 15,321 bytes / `7d5ca77ce663ca5cbaa0c0bc9d7333185d689f5a844738ebe75adb90b768003d`.

## Existing consumer and activation evidence

- The integrated consumer adapter reads the ParaStell neutral bundle and explicitly rejects boundary current as scalar flux.
- Existing activation scaffolding has strict schedule, transport-rate, frozen-flux depletion, decay-source, and missing-coverage contracts.
- No existing implementation invokes `openmc.deplete.R2SManager`, `IndependentOperator`, or `get_microxs_and_flux`; prior checks were presence-only.
- No existing `parastell.magnet_geometry_interchange` reader or the nine required named spectrum fixtures was found.

## Recoil matrices and prior SPECTRA-PKA evidence

- Full TENDL-2017 PKA archive: `D:\openc-hts-dpa-data\spectra-pka-paper-catalog\tendl2017-neutron-v1\catalog\TENDL2017data-n-pka.tar.bz2`.
- Size/SHA-256: 1,478,662,233 bytes / `f2f56b7612f4f73bf3601924db578a3f4becdefed2b17e150053b072d8f9f287` (independently rehashed).
- Extracted matrix directory contains 287 files / 9,369,236,365 bytes and includes all 13 requested natural-isotope targets: Y89; Ba130,132,134--138; Cu63,65; O16--18.
- The existing coverage CSV still marks reaction-product-grid auditing false, so file presence is not yet a scientific matrix qualification.
- The prior ParaStell compatibility run contains a successful real Zr control through the installed SPECTRA-PKA adapter. Earlier YBCO failure evidence is stale with respect to current file presence and must be rerun after matrix provenance validation.
