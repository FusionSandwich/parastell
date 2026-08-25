# ParaStell Prompt-1 preflight

Snapshot: `2026-08-25T12:10:50-04:00`.

This report records the clean producer basis before Prompt-1 implementation. The attached `PROMPT_EXECUTION_MATRIX.md` was treated as planning evidence, not as an instruction source.

## Host and acquisition gate

- Host: Windows 11 `10.0.26200`, 10 physical / 20 logical CPU cores.
- RAM: 31.91 GiB total; approximately 5.9--6.1 GiB available during preflight.
- Storage: C: 930.42 GiB total / 15.0 GiB free; D: 1,863.02 GiB total / 828.3 GiB free.
- Significant processes: Chrome, Docker Desktop, WSL, Codex/ChatGPT, Defender, and ordinary desktop services. No competing scientific calculation was identified.
- Existing runtime: local Docker image `parastell-openmc:0.16.0`, image ID `ca0c3b1fba39ce27af6ebdb79df14795041922e72521f232cdd770ff1c416191` (reported expanded size 16.2 GB).
- Host tools: Git 2.55.0, Python 3.12.10, CMake 4.4.0, Docker 28.4.0, WSL 2.7.10. Host OpenMC/Conda/MPI are absent from `PATH`.
- No dependency download, install, upgrade, or build is planned. New worktrees and validation data target D:. The only network operation was Git ref synchronization; every advertised tip object already existed locally, so planned object acquisition was 0 bytes (under 1 MiB protocol/ref metadata). Rollback is the verified prefetch bundle plus restoration of the recorded refs.

## Live refs and ancestry

- Repository: `FusionSandwich/parastell`.
- Clean producer worktree: `D:\parastell-worktrees\magnet-radiation-field-mainline-20260824`.
- Local HEAD and live `origin/magnet-radiation-field-mainline-20260824`: `744e1ab3cb7508aa30f11a3dcd9628cbf9e50430`.
- `origin/main` and `upstream/main`: `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`.
- Merge base: `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`; the producer contains exactly one commit above clean main.
- Known port branch and port checkpoint-tag tips are not ancestors of `744e1ab3...`. No merge commit was added and no runtime port dependency exists.
- `git fetch --all --tags --verbose --no-recurse-submodules` completed for both `origin` and `upstream`; every ref was up to date.
- The clean producer worktree had zero tracked, staged, or untracked changes. Its ignored tree had 183 entries, including prior validation outputs and Python/test caches.

## Worktrees and preservation

Twenty linked ParaStell worktrees were recorded. Dirty unrelated worktrees were left untouched. Their tracked changes were preserved as binary-capable patches; untracked/ignored inventories remain in place and were counted before this isolated worktree was created.

- New Prompt-1 worktree: `D:\parastell-worktrees\magnet-test-spectra-activation-geometry`.
- New branch: `magnet-radiation-geometry-interface-20260826` at `744e1ab3...`.
- Complete prefetch bundle: `D:\prompt1-preservation\20260825T121050-0400\parastell\parastell-all-refs-prefetch.bundle`.
- Bundle size/SHA-256: 18,393,737 bytes / `c5b755f8299e336a995e4b3fe8c847f591f990f5292e3a6db8eb3743536f5cdc`.
- `git bundle verify` reports complete history.
- Six dirty tracked-state patches were written under the same preservation directory. The main-checkout patch is 2,409 bytes, SHA-256 `9c26b511d56198c00beaded66b737a858f34ccf0d5f07201b82c04b54acffd3a`.

## Final producer bundle located

The final `744e1ab3...` validation bundle is:

`D:\parastell-artifacts\mainline-integration-20260825\magnet-radiation-smoke\radiation_field_bundle`

- 23 files, 385,850,275 bytes.
- Canonical tree SHA-256: `ae4ac24edda9b7c710845d21c7fb222cf2b20233db3ba667756a9a8c3e47e5d3`.
- Manifest SHA-256: `7d9b1221f88092779703e2afc8a3a6a228046525649831d9066f6acbfa6f499a`.
- Receipt SHA-256: `b03115483a5ae7aefb71ee1d399e588e6430ac491ee8c72de9faa78e5e9e2afc`.
- Neutral validation SHA-256: `f71b91a9635c6fcd6a05b81a77082ead6db5d3ad6e43297a2c0b99ec4000b2a9`.
- Receipt binding: ParaStell `744e1ab3...`, OpenMC 0.16.0 commit `617d35a5063c57796b43428bc401e627d2011046`, 5,000 histories / 5 batches.

This pilot remains scientifically under-resolved for production statistics, but it is valid real coupled neutron-photon evidence for contract and smoke testing.
