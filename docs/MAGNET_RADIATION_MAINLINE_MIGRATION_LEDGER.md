# Magnet-radiation mainline migration ledger

## Scope and ancestry

This ledger compares live ParaStell `main` at
`de7d2978ff314b060ca2e6b10745a034e8b2a3c4` with the validated scientific
reference at `b62293735a748021ce2d84b5c0f742eb049b50a2`.

The reference is an 18-commit orphan-style history: Git reports no merge base
between it and live `main`. Its first commit materializes a complete repository
tree and combines ParaStell, COMSOL, HTS replay, and temporary validation
content. Therefore:

- the reference branch is never merged or rebased;
- no commit is cherry-picked;
- approved producer files are reviewed at their final tree, then selectively
  reimplemented or transplanted with dependencies refactored;
- excluded scientific consumers and historical workflows remain outside the
  mainline feature.

Categories are the categories required by the integration prompt:

- **A** — required producer implementation
- **B** — required producer tests, documentation, or schema
- **C** — useful verification-only code or evidence
- **D** — downstream scientific consumer
- **E** — activation-only
- **F** — COMSOL-only
- **G** — port-related
- **H** — temporary CI, materialization, environment, or history
- **I** — unrelated

No changed path is category G. This is positive evidence that port code need not
enter the clean branch.

## Commit classification

| Commit | Subject | Category | Mainline disposition |
|---|---|---|---|
| `e223061` | Validate and package COMSOL-ready fusion magnet models | H/F/D/A | Orphan root materialization. Never cherry-pick; inspect only final producer tree. |
| `24a61e3` | Finish magnet boundary source workflow | A/B/D | Retain boundary-contract concepts; exclude explicit HTS replay and monolithic spectral handoff. |
| `2f68fcf` | Add OpenMC 0.16 magnet handoff foundation | A/B/D | Reuse producer foundations; exclude the production deterministic solver. |
| `cb7b624` | Add authoritative neutron and photon energy-group registry | A/B | Retain validated registry and required neutral structures. |
| `e82d18a` | Add response-preserving neutron and photon group condensation | D | Exclude production response-condensation algorithm. |
| `78970df` | Keep base CLI independent of optional transport packages | A/B | Retain lazy optional-dependency design. |
| `bd30dc0` | Add audited activation backend workflow | E | Exclude completely. |
| `81e9316` | Add port-free magnet handoff production gates | A/B | Retain no-port configuration and producer gates. |
| `b0acfc2` | Add adaptive multi-magnet boundary export | A/B | Retain all/selected-magnet export concepts. |
| `33d3e34` | Add combined magnet transport CLI | A/B/D | Reimplement only producer CLI stages; remove downstream imports. |
| `aecb735` | Validate native combined magnet geometry | A/B/C | Retain native/CAD parity tests and geometry identity concepts. |
| `3b5ff2d` | Replay v2 magnet boundary sources through HTS layers | C/D | Exclude replay; retain only producer-contract evidence where independently tested. |
| `43aec9b` | Validate overlap-free magnet phase-space workflow | A/B/C | Retain closed-envelope and integrity concepts. |
| `2b47258` | Add production magnet radiation field bundle | A/B/D | Retain neutral bundle and volume scalar-flux producer; exclude HTS coupling. |
| `5a5c13a` | Complete production magnet radiation handoff validation | A/B/C/D | Retain heating/closure producer code; deterministic replay stays verification-only or excluded. |
| `6d18222` | Document production activation handoff | E | Exclude activation orchestration; document activation only as downstream use. |
| `dfa8b30` | Qualify HTS reaction and evaluated replay exports | A/B/D | Retain reaction-production exporter; exclude evaluated replay. |
| `b622937` | Close deterministic energy balance gate | C/D | Exclude production deterministic solver and its gate. |

## Changed-file classification

Every path in `git diff --name-status main reference` is classified below.

| Path | Category | Disposition |
|---|---|---|
| `.github/workflows/black.yml` | H | Exclude historical workflow edits. |
| `.github/workflows/build.yml` | H | Exclude historical workflow edits. |
| `.github/workflows/comsol-fusion-magnet-models.yml` | F/H | Exclude. |
| `.github/workflows/full-branch-ci-validation.yml` | H | Exclude temporary branch workflow. |
| `.github/workflows/magnet-handoff-validation.yml` | H | Exclude workstation/materialization workflow. |
| `README.md` | B/D/E/F | Do not transplant mixed edits; add focused producer documentation instead. |
| `docs/activation_workflow.md` | E | Exclude. |
| `docs/comsol_fusion_magnet_models.md` | F | Exclude. |
| `docs/energy_groups.md` | B | Reuse/refine for the neutral producer. |
| `docs/group_condensation.md` | D | Exclude. |
| `docs/magnet_boundary_envelope.md` | B | Reuse/refine. |
| `docs/magnet_boundary_production_validation.md` | B/C/E/D | Keep as evidence reference; write a clean producer validation guide. |
| `docs/magnet_radiation_field_bundle.md` | B | Reuse/refine. |
| `docs/magnet_spectral_handoff.md` | D | Exclude monolithic plane/consumer workflow. |
| `docs/parastell_geometry_handoff.md` | B | Reuse producer geometry concepts only. |
| `environment-openmc016.yml` | H | Exclude environment materialization; use optional dependencies and pinned runtime evidence. |
| `environment.yml` | H | Exclude historical environment changes. |
| `examples/comsol_fusion_magnets_example.py` | F | Exclude. |
| `examples/config.yaml` | I/H | Preserve clean-main example; add separate magnet-field examples. |
| `examples/magnet_spectral_handoff.yaml` | D/C | Exclude old plane/consumer configuration; add new producer configuration. |
| `parastell/__init__.py` | A | Selectively add lazy public producer exports; do not copy eager consumer imports. |
| `parastell/__main__.py` | A/B | Selectively add an additive command namespace while preserving legacy CLI. |
| `parastell/activation/__init__.py` | E | Exclude. |
| `parastell/activation/alara.py` | E | Exclude. |
| `parastell/activation/backends.py` | E | Exclude. |
| `parastell/activation/chain_audit.py` | E | Exclude. |
| `parastell/activation/cli.py` | E | Exclude. |
| `parastell/activation/fispact.py` | E | Exclude. |
| `parastell/activation/model.py` | E | Exclude. |
| `parastell/activation/openmc_r2s.py` | E | Exclude. |
| `parastell/activation/spectrum_export.py` | E | Exclude. |
| `parastell/cli.py` | A/E | Reimplement a producer-only lazy dispatcher; exclude activation commands. |
| `parastell/combined_openmc16_model.py` | A | Selectively transplant/refactor producer geometry and model preparation. |
| `parastell/comsol_fusion_magnets.py` | F | Exclude. |
| `parastell/dagmc_envelope.py` | A | Selectively transplant/refine discovery, fingerprints, and envelopes. |
| `parastell/dagmc_graveyard.py` | A | Retain optional closed-DAGMC assembly helper. |
| `parastell/dt_source.py` | A | Retain audited ParaStell temperature-dependent D–T mesh source. |
| `parastell/energy_groups/__init__.py` | A | Retain. |
| `parastell/energy_groups/cli.py` | A/B | Retain or fold into producer CLI. |
| `parastell/energy_groups/data/ccfe-162.json` | C | Optional audit structure; not required for production handoff. |
| `parastell/energy_groups/data/ccfe-709.json` | A | Retain exact SPECTRA-PKA interoperability structure. |
| `parastell/energy_groups/data/continuous-photon.json` | A | Retain particle-specific continuous-axis declaration. |
| `parastell/energy_groups/data/photon-master-v1.json` | A | Retain independent photon structure. |
| `parastell/energy_groups/data/regression-182.json` | C | Verification-only structure. |
| `parastell/energy_groups/data/smoke-42.json` | C | Test-only structure. |
| `parastell/energy_groups/data/smoke-7.json` | C | Test-only structure. |
| `parastell/energy_groups/data/ukaea-1102.json` | A | Retain audit/export structure. |
| `parastell/energy_groups/registry.py` | A | Retain/refine checksum-validated neutral registry. |
| `parastell/evaluated_multigroup.py` | D | Exclude production evaluated replay. |
| `parastell/group_condensation.py` | D | Exclude production response condensation. |
| `parastell/hts_multilayer.py` | D | Exclude explicit heterogeneous-layer consumer. |
| `parastell/independent_closure.py` | A | Retain independent-seed producer comparison. |
| `parastell/magnet_boundary_envelope.py` | A | Retain/refine closed correlated phase-space contract. |
| `parastell/magnet_coils.py` | A | Review only stable coil/magnet provenance additions. |
| `parastell/magnet_energy_architecture.py` | D/C | Exclude condensation/solver energy architecture; registry remains authoritative. |
| `parastell/magnet_handoff_cli.py` | A/D | Do not transplant monolith; independently implement producer-only staged CLI. |
| `parastell/magnet_handoff_validation.py` | C/D | Exclude executable HTS/deterministic replay; retain isolated producer-contract test ideas. |
| `parastell/magnet_heating.py` | A | Retain/refine particle-resolved neutral heating export. |
| `parastell/magnet_radiation_field_bundle.py` | A | Retain/refine neutral hash-bound bundle. |
| `parastell/magnet_reaction_production.py` | A | Retain/refine reaction and particle-production export. |
| `parastell/magnet_spectral_handoff.py` | D | Exclude monolithic coupling-plane and consumer-oriented API. |
| `parastell/magnet_volume_flux.py` | A | Retain/refine true volume scalar-flux export. |
| `parastell/multigroup_sn.py` | D | Exclude production deterministic solver. |
| `parastell/nwl_utils.py` | I | Feature-unrelated formatting only; may be independently formatted to satisfy the global gate. |
| `parastell/openmc16.py` | A | Retain/refine capability audit and tally construction. |
| `parastell/openmc16_cli.py` | A/B | Fold capability reporting into the producer CLI. |
| `parastell/openmc16_export.py` | A | Retain/refine raw OpenMC surface-source export and integrity closure. |
| `parastell/parastell.py` | A | Add only a small public producer hook if it fits legacy style. |
| `parastell/production_handoff.py` | A | Retain no-port configuration validation. |
| `parastell/source_mesh.py` | A | Retain only source provenance and source-rate additions. |
| `pyproject.toml` | A/B/H/E/F | Selectively add package data and optional extras; do not make radiation dependencies mandatory. |
| `schemas/magnet_spectral_handoff.schema.json` | D | Exclude old plane schema; add producer boundary/bundle/reaction schemas. |
| `tests/test_activation_backend_selection.py` | E | Exclude. |
| `tests/test_activation_chain_audit.py` | E | Exclude. |
| `tests/test_activation_spectrum_export.py` | E | Exclude. |
| `tests/test_alara_bridge.py` | E | Exclude. |
| `tests/test_comsol_fusion_magnets.py` | F | Exclude. |
| `tests/test_dagmc_envelope.py` | B | Retain/refine. |
| `tests/test_dagmc_graveyard.py` | B | Retain/refine. |
| `tests/test_energy_groups.py` | B | Retain/refine around required production structures. |
| `tests/test_evaluated_multigroup.py` | D | Exclude. |
| `tests/test_fispact_bridge.py` | E | Exclude. |
| `tests/test_group_condensation.py` | D | Exclude. |
| `tests/test_hts_multilayer.py` | D/C | Exclude except no source copying; producer contract gets independent tests. |
| `tests/test_hts_tally_columns.py` | D/C | Exclude. |
| `tests/test_independent_closure.py` | B | Retain/refine. |
| `tests/test_magnet_boundary_envelope.py` | B | Retain/refine. |
| `tests/test_magnet_handoff_cli.py` | B/D | Rewrite around producer-only CLI. |
| `tests/test_magnet_heating.py` | B | Retain/refine. |
| `tests/test_magnet_phase_space_production.py` | B/C | Retain producer geometry/transport cases; remove HTS replay coupling. |
| `tests/test_magnet_radiation_field_bundle.py` | B | Retain/refine. |
| `tests/test_magnet_reaction_production.py` | B | Retain/refine. |
| `tests/test_magnet_spectral_handoff.py` | D | Exclude. |
| `tests/test_magnet_volume_flux.py` | B | Retain/refine and add CCFE-709 projection checks. |
| `tests/test_multigroup_sn.py` | D | Exclude. |
| `tests/test_openmc16_foundations.py` | B | Retain/refine with optional dependency stubs. |
| `tests/test_openmc_r2s_activation.py` | E | Exclude. |
| `tests/test_packaging.py` | B | Retain/refine for optional imports and prohibited payloads. |

## Producer migration allowlist

The initial reviewed implementation allowlist is:

```text
parastell/combined_openmc16_model.py
parastell/dagmc_envelope.py
parastell/dagmc_graveyard.py
parastell/dt_source.py
parastell/energy_groups/
parastell/independent_closure.py
parastell/magnet_boundary_envelope.py
parastell/magnet_heating.py
parastell/magnet_radiation_field_bundle.py
parastell/magnet_reaction_production.py
parastell/magnet_volume_flux.py
parastell/openmc16.py
parastell/openmc16_export.py
parastell/production_handoff.py
```

Small reviewed changes are allowed in `parastell.py`, `source_mesh.py`,
`magnet_coils.py`, `__init__.py`, `__main__.py`, and `pyproject.toml`. The
historical CLI is not on the allowlist because it imports excluded HTS and
spectral-consumer modules; the clean branch receives a smaller producer-only
CLI and restartable workflow.

## Independent clean-mainline additions

The following requirements were not available as complete, trustworthy
producer contracts on the historical reference branch and were implemented
independently on clean main:

- global winding-pack/casing association using convergence-qualified closed CAD
  boundaries, closed-triangle DAGMC audits, and deterministic global assignment;
- canonical geometry fingerprinting, graveyard closure, watertightness, and a
  separate native-CAD positive-volume overlap gate;
- parallel-transport centreline linkage for local bins and boundary facets;
- restartable stage manifests with explicit input/output/upstream hashes;
- a mandatory four-shape source-mesh ladder with response-and-cost qualification
  that fails closed on missing or semantically invalid evidence;
- an executable three-seed unbiased campaign and Gate-I validator that
  reconstructs results from hash-bound seed reports and keeps unsupported
  per-magnet/patch/inward-current metrics unassessed;
- executable MAGIC WW preparation/campaign/qualification contracts with an
  explicit unbiased fallback;
- explicit damage-energy and gas-production products that cannot be relabelled
  as DPA or appm;
- hardened neutral bundle validation with exact 18-boundary coverage.

These additions are recorded separately so the migration ledger does not imply
that new validation semantics were copied from the historical feature branch.

## Hard exclusion audit

The clean branch must contain none of the following additions from the
reference tree:

```text
parastell/activation/
parastell/comsol_fusion_magnets.py
parastell/evaluated_multigroup.py
parastell/group_condensation.py
parastell/hts_multilayer.py
parastell/magnet_spectral_handoff.py
parastell/multigroup_sn.py
schemas/magnet_spectral_handoff.schema.json
```

No `*port*`, `native_port*`, or port-visualization path exists in the reference
diff. A scope test will enforce the exclusion against live `main`.
