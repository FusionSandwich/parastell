# Poster parametric OpenMC 0.16 gate

## Scope

This gate prepares bounded proof-of-concept runs for the staged material cases
P00, P01, P02, P03, P05, and P06. P04 remains blocked on material data and P07
is deliberately excluded rather than inventing an unvalidated TiH2 material.
No production run is authorized by this work.

## Immutable geometry and source binding

- Direct 90-degree continuous-radial WISTELL-D H5M SHA-256:
  `d31ea04e4d2fda78870db1688d6cc9215079e20408393c5f5bd374d58f43eaf3`
- Native geometry qualification receipt SHA-256:
  `f4c1799432ddb610968a11abb92b61b13cbd2bc78bf28c239bda9ad5a3ab8d38`
- Native result: 9 volumes, 27 surfaces, zero unmatched edges, zero unsealed
  surfaces or volumes, and zero overlaps at precisions 1, 2, and 4.
- Selected outer-CFS source mesh SHA-256:
  `ed4003589d2eaca445cbd0392b8b3d0465986a0adcdb220507607f7ad97861c5`
- Source-selection receipt SHA-256:
  `72309c1ba5ba6dca69b1c21ad9db00a7f1cab86345556c8f4e914fe0aed6d13c`
- Source-domain result: zero invalid source points, minimum clearance
  0.42059351086489494 cm, and omitted edge-source fraction
  1.7322591520857659e-06.
- Physical source rate for the modeled 90-degree period:
  1.855454485735326e20 neutrons/s.

## Existing instrumentation proof

The exact geometry/source pair already completed a 10,000-history OpenMC 0.16
full-response smoke. Its receipt SHA-256 is
`ce7b039fec54d7b4d3a09f3f9ef76d6770ad266d57bf9f1cc153852d374d16c5`.
The smoke produced four explicit statepoints, a surface bank on surfaces 22, 25,
26, and 27, neutron and photon phase-space currents, volume and surface flux,
heating, damage energy, gas production, particle production, reaction-family
tallies, and explicit Cu63 MT 2/16/102/103/107 tallies. It is an infrastructure
proof only; its statistics are not qualified.

## Runtime binding

- OpenMC: 0.16.0, commit
  `617d35a5063c57796b43428bc401e627d2011046`.
- Qualified CPython 3.11 executable SHA-256:
  `63770468d7041b46aa7fc01ad9a17b4e616dbbb7d613f5470e1cdc5359c83a86`.
- Nuclear-data catalog under evaluation SHA-256:
  `05870d46be9051a20035e4ec07514ba9be3343570ea48542d472b2d712f8d945`.
- Case plan SHA-256:
  `1b48b44e160c19cffde0e9da5f09054b9c9de44584e8801828e3b06be13fea33`.
- Bateman attempt root:
  `/home/apollon/josma/data/codex-parastell/poster-parametric-p00-p06-smoke-v1-20260901T002000Z`.

## Execution status

The first-stage case preparation passed for P00, P01, P02, P03, P05, and P06.
Its receipt SHA-256 is
`a1c9d256ed437359db6483a0ec0f5af6d06246d6fcc991233ae06c2adc6b96a4`.
Every case bound the exact geometry and source above, an explicit two-batch
statepoint policy, and complete neutron/photoatomic library coverage. The
case-specific nuclear-data-manifest SHA-256 values are:

- P00: `676357476976f1872c1f2cad02c6259d29da0a929b28c2527857b5dcc444e458`
- P01: `0891e9bbeb0f808044f88552350ff7d84dcbbc2e07571aa48a5d3c92c84d1858`
- P02: `51198d9df544ff03195c3d1f7f1d5cac10ac2c5071d721308dd8fb1b8477f1ff`
- P03: `676357476976f1872c1f2cad02c6259d29da0a929b28c2527857b5dcc444e458`
- P05: `676357476976f1872c1f2cad02c6259d29da0a929b28c2527857b5dcc444e458`
- P06: `676357476976f1872c1f2cad02c6259d29da0a929b28c2527857b5dcc444e458`

These first-stage models are validation artifacts, not transport inputs. The
transport runner requires the repository's two-runtime chain:

1. PyMOAB geometry inspection writes a create-only geometry transport seal.
2. The OpenMC 0.16-only runtime consumes that seal and writes the accepted
   sealed-model receipt and model XML.

An initial smoke attempt intentionally failed closed before transport because
it was given the first-stage receipt. Its log SHA-256 is
`bbd1d4223b036d0baf926d1487787b44900c2667b75c9b22a11ce2564d3d96ef`;
zero statepoints were created and zero histories ran. It was retired once the
deterministic contract mismatch was established.

The corrected create-only PyMOAB geometry seals passed for every case. Their
SHA-256 values are:

- P00: `b841e82a47f9117ff8e9d0f39b9831ee1d022d41ae9e87d479f18e3c3600ef0a`
- P01: `851661e7f76e2bc818bc0020da18ef36c20a529d81e7c49537cd712747a9ad8f`
- P02: `f1cde416c52e26f7283b354a430bf019651408a429a1008cee26102df8d8e73e`
- P03: `2d37c67110ccf24ccc63249377bc300bb4511543f9a60ad69260b4cafc01a234`
- P05: `8061baa43dc0053e2a37ff668d11a12a9bba0a59efcd665199c4d7acce7f0c40`
- P06: `1f2e201e8136d7c885d79a59ecdd1954df7cab4a01ed24df2c01a7f369b885bb`

The source-strength NPY SHA-256 is identical in all six seals:
`b08851ccbd1df05a05df1a5a9342100586bd92a43dd048abf066fc03eceb2e32`.
The seal stage completed under a 16 GiB/no-swap cgroup with 129.3 MiB peak
memory. Its log SHA-256 is
`efe425b8b6ac39bed5e316ab4db6761c30f7ab9dc00f54bd687042260720cb12`.

The OpenMC-0.16-only sealed model export also passed for every case. Receipt
SHA-256 values are:

- P00: `0af730ef1c8a25afa5960c09f6b8ab28140bfc56c7b70d230ab65cec40d9ed61`
- P01: `252c7ec62a0bbc1a33048578206531a3cb55964621bf6f1c758ee7b86ee7c3e1`
- P02: `84c121da8ef60392424b18d28d26c69d92bd89a1cab09573089937dcd7dc0251`
- P03: `20a1f3a6eb6134c73b789bca46ce7a3e766c1052f9d4f0c4e1b573cb6f5b723c`
- P05: `c77cd0ba504da36c4e0f9ad7d6196333cd1abe5991403027a0549f4911094b68`
- P06: `606a7311f55ff32e2bac52de670f19b3ae3a26f21b52acdea237d6ad65662eee`

Every receipt has status `MODEL_EXPORTED_TRANSPORT_PENDING`, OpenMC version
0.16.0, the accepted H5M hash and physical source rate, and explicit
statepoints at batches 1 and 2. The export log SHA-256 is
`0b83f55bcb3ec799f0c0e7f6fc53437835d308280023032aa3170faa7ef05f67`.

## Bounded transport result

All six independent OpenMC 0.16 smokes passed. Each case ran 400 source
histories as 2 batches of 200 with a unique seed, wrote `statepoint.1.h5` and
`statepoint.2.h5`, wrote a native `surface_source.h5`, exited zero, did not time
out, and preserved every bound input hash. The logs contain no lost-particle,
DAGMC-navigation, fatal, segmentation, or traceback diagnostics. The receipts
remain explicit that these runs do not qualify statistics.

Full-response receipt SHA-256 values are:

- P00: `7cef46468ac20c4b5d6d2b8bf55273407bd27482f54056b6e9242491169b7925`
- P01: `5e6202fe86631fd03c52ed46f2f471b704bbd9b74c5e112d9f26c10f10e2e8e8`
- P02: `c00eee44bf4f4740c52eea6b97424a8dab6fd7f4106a3876598c996821247607`
- P03: `880edab391dbfd91265d65dcd707c809ffd204f6e88b016bcb53c98ff16a0721`
- P05: `1c83cd5e9a1c9b8596677145e1e2d7c0c1caba0d48c28fcc04d415c653575f41`
- P06: `f022d1e02bce6eaf6afbd87bc13d1fa5b8b38c86510b6ae246a06560cb4bc1fc`

The six cases ran concurrently at one thread each under one 32 GiB/no-swap
cgroup. Runtime was 5 minutes 35.386 seconds, CPU time was 1 minute 42.359
seconds, peak memory was 2.8 GiB, and peak swap was zero. The terminal job log
SHA-256 is
`e590ea9404636b2722f988b04d0550b3519d56dafdaf7bf19f67314f768b56f9`.
The shared CPU ledger returned to zero after completion.

Each executed model contains the complete activation-ready magnet response
profile plus reactor-component flux, heating, reaction, and breeder tritium
production tallies. The magnet profile includes neutron/photon surface current,
directional current, surface flux, volume flux, neutron reaction families,
explicit Cu63 MT 2/16/102/103/107 rates, neutron/photon/total heating,
damage-energy, H1/H2/H3/He3/He4 production, and neutron/photon/electron/positron
production. Surface banks cover native surfaces 22, 25, 26, and 27 in both
directions with the raw OpenMC phase-space weight contract. The tiny banks are
preserved for later phase-space postprocessing; no claim of complete statistical
sampling is made from 400 histories.

## Decision

`BOUNDED_POSTER_PARAMETRIC_INFRASTRUCTURE_PASS`

The six proof-of-concept cases are transport-ready for a separately authorized
poster campaign. P04 remains blocked on complete material data. P07 is excluded
from this proof of concept rather than assigning an unvalidated TiH2 material.
No large run, production-statistics campaign, or production authorization was
performed here.

## Software validation

- Focused blanket/material, poster-case, OpenMC-model, geometry-seal, and
  sealed-runtime tests: 41 passed.
- Broad dependency-free repository run: 723 passed and 5 skipped.
- Five additional tests reached locally installed but incompatible optional
  CAD/DAGMC APIs (`pydagmc.Model` and `cad_to_dagmc.init_gmsh` were absent).
  Six legacy modules could not be collected in the same Windows Python because
  PyMOAB, Gmsh, or OpenMC was absent. These are local optional-runtime limits,
  not transport results; PyMOAB and OpenMC 0.16 were exercised successfully in
  the qualified Bateman runtimes above.
- Black, full Python compilation, package import, and `git diff --check`: pass.
- The two local CLI help probes are unavailable because this Windows Python
  lacks optional `cad_to_dagmc`; the globally installed `parastell` launcher
  also resolves to a separate checkout. Neither launcher was used for the
  qualified Bateman workflow and neither separate checkout was modified.
